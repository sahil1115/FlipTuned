"""Run history: filters, compare, artifacts, archive and confirmed hard delete (Phase 13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fliptuned.config.app_home import AppPaths
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.jobs.state import JobType
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services import history
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.services.jobs import CancelJobRequest, cancel_job
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples, use_fake_tokenizer
from tests.helpers.runs import make_run


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    return home


def _meta(home: Path, run_id: str, **fields) -> Path:
    run_dir = AppPaths(home).runs / run_id
    path = run_dir / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def _scored(home: Path, run_id: str, verdict: str) -> str:
    eval_id = f"eval_{run_id}"
    with connect(AppPaths(home).db) as conn:
        repositories.insert_evaluation(
            conn, evaluation_id=eval_id, run_id=run_id, eval_dir="x", config={}
        )
        metrics = {"verdict": verdict, "wins": 3, "losses": 1, "ties": 2, "sign_test_p": 0.6}
        repositories.update_evaluation(
            conn, eval_id, status="ready", verdict=verdict, metrics=metrics
        )
    return eval_id


def test_list_filters_and_latest_verdict(home: Path) -> None:
    a = make_run(home)
    b = make_run(home, status="cancelled")
    _meta(home, a, config_name="quick", config_version=2, result={"best_eval_loss": 1.5})
    _scored(home, a, "inconclusive")
    runs = history.list_history(history.RunFilter(home=str(home)))
    assert {r.id for r in runs} == {a, b}
    first = next(r for r in runs if r.id == a)
    assert first.verdict.verdict == "inconclusive" and first.verdict.wins == 3
    assert first.config_name == "quick" and first.best_eval_loss == 1.5 and first.evaluations == 1
    only = history.list_history(history.RunFilter(home=str(home), status="cancelled"))
    assert [r.id for r in only] == [b]
    assert [
        r.id for r in history.list_history(history.RunFilter(home=str(home), text="QUICK"))
    ] == [a]


def test_compare_shows_config_differences_and_figures(home: Path) -> None:
    a, b = make_run(home), make_run(home)
    base = {"name": "x", "training": {"learning_rate": 1e-4, "max_epochs": 3}, "lora": {"rank": 8}}
    _meta(
        home,
        a,
        effective_config=base,
        result={"best_eval_loss": 1.2, "best_step": 4},
        attempts=[
            {"started_at": "2026-10-30T10:00:00+00:00", "finished_at": "2026-10-30T10:01:30+00:00"}
        ],
    )
    other = {
        "name": "y",
        "training": {"learning_rate": 2e-4, "max_epochs": 3},
        "lora": {"rank": 16},
    }
    _meta(home, b, effective_config=other, result={"best_eval_loss": 1.1})
    result = history.compare_runs(str(home), a, b)
    assert [(d.key, d.a, d.b) for d in result.config_differences] == [
        ("lora.rank", 8, 16),
        ("training.learning_rate", 1e-4, 2e-4),
    ]
    assert result.a.best_step == 4 and result.a.train_seconds == 90.0 and result.a.attempts == 1
    assert result.same_dataset and result.note is None
    with pytest.raises(ConfigError):
        history.compare_runs(str(home), a, a)


def test_artifacts_list_and_safe_download(home: Path) -> None:
    run_id = make_run(home)
    run_dir = AppPaths(home).runs / run_id
    (run_dir / "evaluation" / "eval_1").mkdir(parents=True)
    (run_dir / "evaluation" / "eval_1" / "report.md").write_text("# Report\n", encoding="utf-8")
    (run_dir / "checkpoints" / "checkpoint-2").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-2" / "state.bin").write_bytes(b"x" * 10)
    listing = history.list_artifacts(str(home), run_id)
    by_path = {a.path: a for a in listing.artifacts}
    assert by_path["evaluation/eval_1/report.md"].downloadable
    assert by_path["checkpoints"].kind == "folder" and by_path["checkpoints"].files == 1
    assert by_path["adapter"].kind == "folder" and not by_path["adapter"].downloadable
    assert (
        history.artifact_file(str(home), run_id, "evaluation/eval_1/report.md").name == "report.md"
    )
    for bad in ("../../fliptuned.db", str(run_dir / "metadata.json"), "nope.json"):
        with pytest.raises((ConfigError, NotAvailableError)):
            history.artifact_file(str(home), run_id, bad)
    with pytest.raises(ConfigError, match="Only text artifacts"):
        history.artifact_file(str(home), run_id, "adapter/adapter_model.safetensors")


def test_archive_hides_and_restores(home: Path) -> None:
    run_id = make_run(home)
    _scored(home, run_id, "rejected")
    history.set_archived(str(home), run_id, True)
    assert history.list_history(history.RunFilter(home=str(home))) == []
    archived = history.list_history(history.RunFilter(home=str(home), include_archived=True))
    assert archived[0].archived
    history.set_archived(str(home), run_id, False)
    assert history.list_history(history.RunFilter(home=str(home)))[0].verdict.verdict == "rejected"


def test_hard_delete_needs_confirmation_and_no_active_job(home: Path) -> None:
    run_id = make_run(home)
    _scored(home, run_id, "accepted")
    runner = LocalJobRunner(AppPaths(home))
    job = runner.submit(JobType.EVALUATE, {}, detach=False, run_id=run_id)  # active, this process
    run_dir = AppPaths(home).runs / run_id

    with pytest.raises(ConfigError, match="doesn't match"):
        history.delete_run(history.DeleteRequest(home=str(home), run_id=run_id, confirm="run_x"))
    with pytest.raises(FlipTunedError, match="while its evaluate job"):
        history.delete_run(history.DeleteRequest(home=str(home), run_id=run_id, confirm=run_id))
    with pytest.raises(FlipTunedError, match="can't be archived"):
        history.set_archived(str(home), run_id, True)
    assert run_dir.exists()

    cancel_job(CancelJobRequest(home=str(home), job_id=job, timeout=0))
    result = history.delete_run(
        history.DeleteRequest(home=str(home), run_id=run_id, confirm=run_id)
    )
    assert str(run_dir) in result.removed_folders and not run_dir.exists()
    assert not (AppPaths(home).jobs / job).exists()
    with connect(AppPaths(home).db) as conn:
        assert repositories.get_run_any(conn, run_id) is None
        assert conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM jobs WHERE run_id = ?", (run_id,)).fetchone()[0] == 0
        )
    with pytest.raises(NotAvailableError):
        history.list_artifacts(str(home), run_id)
