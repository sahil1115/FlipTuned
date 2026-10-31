"""Export (SPEC §16): adapter export with metadata, merge as a job with the 5-prompt before/after
check, registry records. A fake backend stands in for the model; nothing is downloaded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.app_home import AppPaths
from fliptuned.errors import FlipTunedError
from fliptuned.models.catalog import get_entry
from fliptuned.services import export
from fliptuned.services import jobs as jobs_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples, use_fake_tokenizer
from tests.helpers.runs import FakeBackend, make_run

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    return home


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBackend:
    backend = FakeBackend()
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    monkeypatch.setattr(export, "load_model", backend.load)
    monkeypatch.setattr(export, "get_backend", lambda name: backend)
    monkeypatch.setattr(export, "model_files", lambda entry, backend, allow_download: model_dir)
    return backend


def _merge(home: Path, run_id: str, **kw) -> export.ExportResult:
    start = export.export_run(
        export.ExportRequest(home=str(home), run_id=run_id, merge=True, **kw), detach=False
    )
    assert start.job is not None and start.job.holds_model_lock
    info = jobs_service.run_inline(str(home), start.job.id)
    assert info.status == "succeeded", jobs_service.job_error(str(home), start.job.id)
    return jobs_service.job_result(str(home), start.job.id, export.ExportResult)


def test_adapter_export(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    start = export.export_run(export.ExportRequest(home=str(home), run_id=run_id), detach=False)
    assert start.job is None and start.result is not None
    result = start.result
    folder = Path(result.path)
    assert folder.parent == AppPaths(home).runs / run_id / "exports"
    assert sorted(p.name for p in folder.iterdir()) == [
        "adapter",
        "export.json",
        "hf_training_args.json",
        "run_metadata.json",
    ]
    assert (folder / "adapter" / "adapter_model.safetensors").read_bytes() == b"fake"
    manifest = json.loads((folder / "export.json").read_text(encoding="utf-8"))
    entry = get_entry("tiny-qwen2")
    assert manifest["base_model"]["repo"] == entry.sources["hf_peft"].repo
    assert manifest["base_model"]["revision"] == entry.sources["hf_peft"].revision
    plan = json.loads((AppPaths(home).runs / run_id / "run_plan.json").read_text("utf-8"))
    assert manifest["system_prompt"] == plan["system_prompt"]
    assert manifest["kind"] == "adapter" and manifest["merged"] is None
    assert manifest["backend"] == "hf_peft" and "PeftModel" in manifest["how_to_use"]
    assert fake.loads == []  # no model was loaded
    exports = export.list_exports(str(home), run_id)
    assert [(e.id, e.kind, e.status) for e in exports] == [
        (result.export_id, "adapter", "succeeded")
    ]


def test_merge_export_with_before_after_check(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    result = _merge(home, run_id)
    folder = Path(result.path)
    assert result.kind == "merged" and (folder / "merged" / "model.safetensors").exists()
    assert result.check is not None
    assert (result.check.prompts, result.check.identical) == (5, 5)
    assert result.notes == []
    check = json.loads((folder / "merge_check.json").read_text(encoding="utf-8"))
    assert [i["prompt_id"] for i in check["items"]] == [f"reg_0{n}" for n in range(1, 6)]
    manifest = json.loads((folder / "export.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "merged" and manifest["merged"]["check"] == {
        "prompts": 5,
        "identical": 5,
    }
    assert {"merged/", "merge_check.json"} <= set(manifest["files"])
    # before = base + adapter, after = the merged folder; each released after use
    assert fake.loads == [Path(make_adapter_path(home, run_id))]
    assert fake.unloaded == 2
    assert [(e.kind, e.status) for e in export.list_exports(str(home), run_id)] == [
        ("merged", "succeeded")
    ]


def make_adapter_path(home: Path, run_id: str) -> Path:
    return AppPaths(home).runs / run_id / "adapter"


def test_merge_check_reports_differences(home: Path, fake: FakeBackend) -> None:
    fake.merged_differs = True
    result = _merge(home, make_run(home))
    assert result.check is not None and result.check.identical == 0
    assert result.check.items[0].first_difference == 0
    assert "5 of 5 check answers differ" in result.notes[-1]


def test_export_out_dir_and_refusals(home: Path, fake: FakeBackend, tmp_path: Path) -> None:
    run_id = make_run(home)
    config = tmp_path / "export.yaml"
    out = tmp_path / "exports-here"
    config.write_text(f"kind: export\nout_dir: {out.as_posix()}\n", encoding="utf-8")
    start = export.export_run(
        export.ExportRequest(home=str(home), run_id=run_id, config_path=str(config)), detach=False
    )
    assert Path(start.path).parent == out and Path(start.path).name.startswith(run_id)

    unfinished = make_run(home, status="interrupted")
    with pytest.raises(FlipTunedError, match="finished training"):
        export.export_run(export.ExportRequest(home=str(home), run_id=unfinished), detach=False)


def test_cli_export(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    plain = runner.invoke(app, ["export", "--run", run_id, "--home", str(home)])
    assert plain.exit_code == 0, plain.output
    assert "(adapter)" in plain.output and "export.json" in plain.output
    merged = runner.invoke(app, ["export", "--run", run_id, "--merge", "--home", str(home)])
    assert merged.exit_code == 0, merged.output
    assert "5 of 5 answers identical" in merged.output
    as_json = runner.invoke(app, ["export", "--run", run_id, "--json", "--home", str(home)])
    assert json.loads(as_json.output)["kind"] == "adapter"
