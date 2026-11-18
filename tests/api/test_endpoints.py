"""Every API endpoint, against fixture data in a temp home. Jobs run inline (see conftest) with fake
backends for anything that would load a model."""

from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fliptuned.api.app import openapi_document
from fliptuned.config.app_home import AppPaths
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.jobs.state import JobType
from fliptuned.services import chat, evaluation, export, training
from fliptuned.services.errors import FlipTunedError
from tests.api.conftest import run_job
from tests.helpers.pipeline import FIXTURES
from tests.helpers.runs import FakeBackend, make_run

REPO = Path(__file__).resolve().parents[2]


def sse(text: str) -> list[dict]:
    """Parse a text/event-stream body into [{event, data, id}]."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields: dict = {}
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            fields[key] = value
        if "data" in fields:
            events.append(
                {
                    "event": fields.get("event"),
                    "data": json.loads(fields["data"]),
                    "id": fields.get("id"),
                }
            )
    return events


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBackend:
    backend = FakeBackend()
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    monkeypatch.setattr(evaluation, "load_model", backend.load)
    monkeypatch.setattr(chat, "load_model", backend.load)
    monkeypatch.setattr(export, "load_model", backend.load)
    monkeypatch.setattr(export, "get_backend", lambda name: backend)
    monkeypatch.setattr(export, "model_files", lambda entry, backend, allow_download: model_dir)
    return backend


@pytest.mark.parametrize(
    "path",
    [
        "/api/doctor",
        "/api/status",
        "/api/settings",
        "/api/fs/browse",
        "/api/sources",
        "/api/models",
        "/api/configs/schema/train",
        "/api/configs/schema/ingest",
        "/api/presets",
        "/api/training-choices",
        "/api/audit",
        "/api/datasets",
        "/api/datasets/fixtures/audit",
        "/api/configs",
        "/api/runs",
        "/api/jobs",
        "/api/exports",
        "/api/openapi.json",
    ],
)
def test_read_endpoints(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200, response.text


def test_read_endpoint_shapes(client: TestClient) -> None:
    assert client.get("/api/doctor").json()["machine"]["os_name"]
    assert client.get("/api/settings").json()["allowed_roots"] == [str(FIXTURES)]
    schema = client.get("/api/configs/schema/train").json()
    assert "model" in schema["properties"] and schema["properties"]["preset"]["description"]
    assert client.get("/api/configs/schema/nope").status_code == 404
    assert set(client.get("/api/presets").json()) >= {"hf_peft", "mlx"}
    choices = client.get("/api/training-choices", params={"model": "tiny-qwen2"}).json()
    assert {(o["backend"], o["device"]) for o in choices["options"]} == {
        ("mlx", "apple_gpu"),
        ("hf_peft", "cuda"),
        ("hf_peft", "cpu"),
    }
    assert choices["auto"]["backend"] in ("hf_peft", "mlx") and choices["presets"]["hf_peft"]
    assert client.get("/api/training-choices", params={"model": "nope"}).json()["auto"] is None
    assert [d["name"] for d in client.get("/api/datasets").json()] == ["fixtures"]
    models = client.get("/api/models", params={"include_test_only": True}).json()["models"]
    assert "tiny-qwen2" in {m["id"] for m in models}


def test_data_endpoints(client: TestClient, home: Path) -> None:
    scan = client.post("/api/scan", json={"path": str(FIXTURES / "scan_folder"), "name": "scanned"})
    assert scan.status_code == 200 and scan.json()["supported_files"] >= 1

    config = {
        "name": "api-source",
        "sources": [
            {
                "path": str(FIXTURES / "qa_basic.csv"),
                "columns": {"question": "question", "answer": "answer"},
            }
        ],
    }
    job = client.post("/api/ingest", json={"config": config})
    assert job.status_code == 200, job.text
    assert job.json()["type"] == "ingest" and run_job(home, job.json()["id"]) == "succeeded"
    assert "api-source" in {s["name"] for s in client.get("/api/sources").json()}

    records = client.post(
        "/api/records/search", json={"limit": 5, "filter": {"status": ["pending_review"]}}
    )
    assert records.status_code == 200, records.text
    items = records.json()["items"]
    if items:
        ref = {"source_id": items[0]["source_id"], "record_id": items[0]["record_id"]}
        decided = client.post(
            "/api/records/decisions", json={"records": [ref], "status": "approved"}
        )
        assert decided.status_code == 200, decided.text

    exported = client.post("/api/review/export", json={})
    assert exported.status_code == 200 and exported.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(exported.content.decode("utf-8-sig"))))
    assert rows and "decision" in rows[0]
    imported = client.post("/api/review/import", content=exported.content)
    assert imported.status_code == 200 and imported.json()["blank"] == len(rows)
    assert not list((home / "uploads").glob(".incoming-*"))  # the temp copy is gone

    dataset_config = json.loads(
        json.dumps({"kind": "dataset", "name": "api-ds", "model_catalog_id": "tiny-qwen2"})
    )
    built = client.post(
        "/api/datasets", json={"config": {k: v for k, v in dataset_config.items() if k != "kind"}}
    )
    assert built.status_code == 200, built.text
    assert built.json()["type"] == "build_dataset"
    assert client.post("/api/datasets/fixtures/validate").json()["ok"] is True
    assert client.post("/api/datasets", json={"config": {"name": 3}}).status_code == 422


def test_model_download_job_is_queued_not_run(client: TestClient) -> None:
    job = client.post("/api/models/tiny-qwen2/download", json={"backend": "hf_peft"})
    assert job.status_code == 200 and job.json()["type"] == "model_download"
    cancelled = client.post(f"/api/jobs/{job.json()['id']}/cancel", json={"timeout": 0})
    assert cancelled.json()["status"] == "cancelled"
    assert (
        client.post("/api/models/tiny-qwen2/download", json={"backend": "mlx"}).status_code == 404
    )


def test_model_download_is_refused_without_disk_space(client: TestClient, monkeypatch) -> None:
    from fliptuned.services import models as models_service

    monkeypatch.setattr(models_service, "is_cached", lambda source: False)
    monkeypatch.setattr(models_service.detect, "free_disk_gb", lambda path: (path, 0.0))
    response = client.post("/api/models/tiny-qwen2/download", json={"backend": "hf_peft"})
    assert response.status_code == 400
    assert "disk space" in response.json()["error"]["message"]


def test_config_and_training_endpoints(client: TestClient, home: Path, monkeypatch) -> None:
    config = {"name": "api-cfg", "model": {"catalog_id": "tiny-qwen2"}, "preset": "quick"}
    saved = client.post("/api/configs", json={"config": config})
    assert saved.status_code == 200 and saved.json()["version"] == 1
    stored = client.get("/api/configs/api-cfg").json()["config"]
    assert stored["preset"] == "quick" and "training" not in stored  # the preset stays in charge
    dup = client.post("/api/configs/api-cfg/duplicate", json={"new_name": "api-cfg-2"})
    assert dup.status_code == 200 and dup.json()["name"] == "api-cfg-2"
    assert client.post("/api/configs", json={"config": {"name": "x"}}).status_code == 422

    preflight = client.post("/api/preflight", json={"saved_name": "api-cfg"})
    assert preflight.status_code == 200 and preflight.json()["checks"]

    seen = {}

    def capture(request, *, detach):
        seen["request"], seen["detach"] = request, detach
        raise FlipTunedError("captured")

    monkeypatch.setattr(training, "start_training", capture)
    started = client.post("/api/runs", json={"saved_name": "api-cfg"})
    assert started.status_code == 400 and started.json()["error"]["message"] == "captured"
    written = Path(seen["request"].config_path)
    assert written.is_relative_to(home / "configs" / "generated") and seen["detach"] is True
    assert "api-cfg" in written.read_text(encoding="utf-8")
    both = client.post("/api/runs", json={"saved_name": "api-cfg", "config": config})
    assert both.status_code == 422

    run_id = make_run(home)
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "succeeded"
    assert run_id in {r["id"] for r in client.get("/api/runs").json()}
    assert client.post(f"/api/runs/{run_id}/resume", json={}).status_code == 400
    assert client.get("/api/runs/run_nope").status_code == 404

    state = client.get(f"/api/runs/{run_id}/monitor")
    assert state.status_code == 200 and state.json()["memory"]["pool"] == "RAM"
    assert client.get("/api/runs/run_nope/monitor").status_code == 404


def test_resume_refused_while_another_job_holds_the_model(client: TestClient, home: Path) -> None:
    run_id = make_run(home, status="cancelled")
    (AppPaths(home).runs / run_id / "checkpoints" / "checkpoint-1").mkdir(parents=True)
    holder = LocalJobRunner(AppPaths(home)).submit(JobType.EVALUATE, {}, detach=False)
    state = client.get(f"/api/runs/{run_id}/monitor").json()
    assert state["lock_holder"]["job_id"] == holder and not state["can_resume"]
    refused = client.post(f"/api/runs/{run_id}/resume", json={})
    assert refused.status_code == 409
    assert refused.json()["error"]["message"].startswith(
        f"Another job is using the model: {holder}"
    )


def test_evaluation_endpoints(client: TestClient, home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    start = client.post(f"/api/runs/{run_id}/evaluations", json={"limit": 4})
    assert start.status_code == 200, start.text
    eval_id, job_id = start.json()["evaluation_id"], start.json()["job"]["id"]
    assert client.get(f"/api/evaluations/{eval_id}/pairs").status_code == 400  # not generated yet
    assert run_job(home, job_id) == "succeeded"
    assert client.get(f"/api/evaluations/{eval_id}").json()["status"] == "ready"
    assert [e["id"] for e in client.get(f"/api/runs/{run_id}/evaluations").json()] == [eval_id]

    pair = client.get(f"/api/evaluations/{eval_id}/pairs").json()["pair"]
    assert set(pair) == set(evaluation.ReviewPair.model_fields)  # A/B only, no model identity
    judged = client.post(
        f"/api/evaluations/{eval_id}/judgements",
        json={"item_id": pair["item_id"], "judgement": "A"},
    )
    assert judged.status_code == 200 and judged.json()["judged"] == 1
    bad = client.post(
        f"/api/evaluations/{eval_id}/judgements",
        json={"item_id": pair["item_id"], "judgement": "C"},
    )
    assert bad.status_code == 422
    assert (
        client.get(f"/api/evaluations/{eval_id}/pairs", params={"position": 2}).json()["pair"][
            "position"
        ]
        == 2
    )

    sheet = Path(client.get(f"/api/evaluations/{eval_id}").json()["eval_dir"]) / "review_sheet.csv"
    with open(sheet, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    for row in rows:
        row["judgement"] = "tie"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    uploaded = client.post(
        f"/api/evaluations/{eval_id}/review-sheet", content=buffer.getvalue().encode("utf-8-sig")
    )
    assert uploaded.status_code == 200 and uploaded.json()["recorded"] == {"tie": 4}

    score = client.post(f"/api/evaluations/{eval_id}/score")
    assert score.status_code == 200 and score.json()["verdict"] == "inconclusive"
    assert score.json()["ties"] == 4


def test_export_endpoints(client: TestClient, home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    plain = client.post(f"/api/runs/{run_id}/exports", json={})
    assert plain.status_code == 200 and plain.json()["result"]["kind"] == "adapter"
    merged = client.post(f"/api/runs/{run_id}/exports", json={"merge": True})
    assert merged.status_code == 200 and merged.json()["job"]["type"] == "merge_export"
    assert run_job(home, merged.json()["job"]["id"]) == "succeeded"
    kinds = {
        (e["kind"], e["status"])
        for e in client.get("/api/exports", params={"run_id": run_id}).json()
    }
    assert kinds == {("adapter", "succeeded"), ("merged", "succeeded")}


def test_chat_endpoints(client: TestClient, home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    started = client.post("/api/chat/sessions", json={"run_id": run_id})
    assert started.status_code == 200, started.text
    job_id = started.json()["job"]["id"]
    worker = threading.Thread(target=run_job, args=(home, job_id), daemon=True)
    worker.start()

    turns = [{"role": "user", "content": "hello"}]
    reply = client.post(f"/api/chat/sessions/{job_id}/messages", json={"turns": turns})
    assert reply.status_code == 200 and reply.headers["content-type"].startswith(
        "text/event-stream"
    )
    events = sse(reply.text)
    assert "".join(e["data"]["text"] for e in events if e["event"] == "delta") == "tuned reply"
    assert events[-1]["event"] == "done"

    system = [{"role": "system", "content": "override"}, *turns]
    refused = sse(client.post(f"/api/chat/sessions/{job_id}/messages", json={"turns": system}).text)
    assert refused[-1]["event"] == "error" and "override" not in json.dumps(refused)

    busy = client.post("/api/chat/sessions", json={"run_id": run_id})
    assert busy.status_code == 409 and job_id in busy.json()["error"]["message"]

    assert client.delete(f"/api/chat/sessions/{job_id}").json() == {"status": "closed"}
    worker.join(30)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "succeeded"


def test_job_events_stream(client: TestClient, home: Path) -> None:
    job = client.post("/api/jobs/selftest", json={"seconds": 0.3, "steps": 3}).json()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"
    assert run_job(home, job["id"]) == "succeeded"

    stream = client.get(f"/api/jobs/{job['id']}/events")
    assert stream.status_code == 200 and stream.headers["content-type"].startswith(
        "text/event-stream"
    )
    events = sse(stream.text)
    assert [e["data"]["done"] for e in events if e["event"] == "progress"] == [1, 2, 3]
    assert events[-1]["event"] == "status" and events[-1]["data"]["status"] == "succeeded"
    assert events[-1]["id"] is not None
    # Resuming from the last ID returns nothing new; from an earlier one, only what came after.
    assert (
        sse(
            client.get(
                f"/api/jobs/{job['id']}/events", headers={"Last-Event-ID": events[-1]["id"]}
            ).text
        )
        == []
    )
    assert client.get("/api/jobs/job_nope/events").status_code == 404


def test_openapi_is_exported_for_the_web_client() -> None:
    """Contract check: web/openapi.json matches the app (regenerate with the npm gen:api script)."""
    committed = json.loads((REPO / "web" / "openapi.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(openapi_document())), (
        "web/openapi.json is stale: run `npm --prefix web run gen:api`"
    )


def test_web_tests_use_the_current_train_schema() -> None:
    """The UI tests render web/src/test/train-schema.json; it must match the config models."""
    from fliptuned.services.configs import json_schema

    committed = REPO / "web" / "src" / "test" / "train-schema.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == json_schema("train"), (
        "web/src/test/train-schema.json is stale: run `npm --prefix web run gen:api`"
    )


def test_typescript_client_is_up_to_date(tmp_path: Path) -> None:
    """web/src/api/schema.d.ts was generated from the committed web/openapi.json."""
    import shutil
    import subprocess

    cli = REPO / "web" / "node_modules" / "openapi-typescript" / "bin" / "cli.js"
    node = shutil.which("node")
    if node is None or not cli.exists():
        pytest.skip("Node or web/node_modules missing (run `npm --prefix web ci`)")
    out = tmp_path / "schema.d.ts"
    subprocess.run(
        [node, str(cli), str(REPO / "web" / "openapi.json"), "-o", str(out)],
        check=True,
        capture_output=True,
    )
    committed = (REPO / "web" / "src" / "api" / "schema.d.ts").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == committed, (
        "web/src/api/schema.d.ts is stale: run `npm --prefix web run gen:api`"
    )


def test_history_endpoints(client: TestClient, home: Path) -> None:
    a, b = make_run(home), make_run(home)
    assert {r["id"] for r in client.get("/api/history").json()} == {a, b}
    assert client.get("/api/history", params={"status": "failed"}).json() == []
    compared = client.get("/api/compare-runs", params={"a": a, "b": b})
    assert compared.status_code == 200 and compared.json()["same_dataset"] is True
    listing = client.get(f"/api/runs/{a}/artifacts").json()
    assert "metadata.json" in {x["path"] for x in listing["artifacts"]}
    download = client.get(f"/api/runs/{a}/artifacts/file", params={"path": "metadata.json"})
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    escape = client.get(f"/api/runs/{a}/artifacts/file", params={"path": "../../fliptuned.db"})
    assert escape.status_code in (404, 422)

    assert client.post(f"/api/runs/{a}/archive", json={"archived": True}).json()["archived"]
    assert [r["id"] for r in client.get("/api/history").json()] == [b]
    assert len(client.get("/api/history", params={"include_archived": True}).json()) == 2
    refused = client.post(f"/api/runs/{b}/delete", json={"confirm": "nope"})
    assert (
        refused.status_code == 422 and "nothing was deleted" in refused.json()["error"]["message"]
    )
    deleted = client.post(f"/api/runs/{b}/delete", json={"confirm": b})
    assert deleted.status_code == 200 and client.get(f"/api/runs/{b}").status_code == 404


def test_port_setting(client: TestClient) -> None:
    assert client.get("/api/settings").json()["port"] == 8765
    assert client.put("/api/settings", json={"port": 9001}).json()["port"] == 9001
    assert client.put("/api/settings", json={"port": 80}).status_code == 422
