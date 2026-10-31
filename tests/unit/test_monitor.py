"""Run monitor: the live state of a training run rebuilt from its attempts' events (Phase 12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fliptuned.config.app_home import AppPaths
from fliptuned.jobs.events import EventWriter
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.jobs.state import JobType
from fliptuned.jobs.worker import EVENTS_FILE, job_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services import monitor
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


def _attempt(home: Path, run_id: str, events: list[tuple[str, dict]], final: str | None) -> str:
    """A train job of the run (no worker) with these events, ended in ``final`` if given."""
    runner = LocalJobRunner(AppPaths(home))
    job_id = runner.submit(
        JobType.TRAIN, {"run_id": run_id}, detach=False, holds_model_lock=False, run_id=run_id
    )
    writer = EventWriter(job_dir(runner.paths, job_id) / EVENTS_FILE)
    writer.emit("status", status="running", pid=1)
    for kind, fields in events:
        writer.emit(kind, **fields)
    with connect(runner.paths.db) as conn, conn:
        repositories.transition_job(conn, job_id, "running", from_statuses=("queued",))
        if final:
            repositories.transition_job(conn, job_id, final, from_statuses=("running",))
            repositories.update_run(conn, run_id, status=final)
    if final:
        writer.emit("status", status=final, error=None)
    return job_id


def _train(step: int, loss=None, eval_loss=None, **extra) -> tuple[str, dict]:
    fields = {"step": step, "max_steps": 8, "epoch": step / 8, "loss": loss, "eval_loss": eval_loss}
    return "train", {"learning_rate": 1e-4, "eta_s": 10.0, "memory_gb": 0.5, **fields, **extra}


def _set_meta(home: Path, run_id: str, **fields) -> Path:
    run_dir = AppPaths(home).runs / run_id
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(fields)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def test_curve_across_a_cancel_and_a_resume(home: Path) -> None:
    run_id = make_run(home, status="queued")
    meta = {"preflight_warnings": ["Dataset size: small"], "result": None}
    run_dir = _set_meta(home, run_id, machine={"ram_gb": 16.0}, config_name="c", **meta)
    first = _attempt(
        home,
        run_id,
        [
            ("log", {"level": "warning", "message": "Keep-awake isn't available."}),
            _train(1, loss=2.0),
            _train(2, eval_loss=1.9),
            _train(2, loss=1.8),
            _train(3, loss=1.7),  # after the last checkpoint: replaced by the resume
            ("progress", {"done": 3, "total": 8, "message": None}),
        ],
        "cancelled",
    )
    (run_dir / "checkpoints" / "checkpoint-2").mkdir(parents=True)

    state = monitor.run_monitor(str(home), run_id)
    assert state.run.status == "cancelled" and state.can_resume and not state.can_cancel
    assert state.best == monitor.BestCheckpoint(step=2, eval_loss=1.9, final=False)
    assert state.warnings == ["Dataset size: small", "Keep-awake isn't available."]
    assert state.memory == monitor.MemoryInfo(pool="RAM", peak_gb=0.5, total_gb=16.0)
    assert state.step == 3 and state.max_steps == 8 and state.eta_s is None  # not running
    assert [line.message for line in state.log][-2:] == [
        "Step 3/8: train loss 1.7",
        f"Job {first} is cancelled.",
    ]

    second = _attempt(
        home,
        run_id,
        [_train(3, loss=1.6), _train(4, eval_loss=1.5), ("progress", {"done": 4, "total": 8})],
        None,
    )
    with connect(AppPaths(home).db) as conn, conn:
        repositories.update_run(conn, run_id, status="running")
    state = monitor.run_monitor(str(home), run_id)
    assert [a.id for a in state.attempts] == [first, second] and state.job.id == second
    assert [(p.step, p.job_id) for p in state.points] == [
        (1, first),
        (2, first),
        (2, first),
        (3, second),
        (4, second),
    ]
    assert state.best.step == 4 and state.eta_s == 10.0 and state.can_cancel
    assert not state.can_resume and state.resume_note is None  # running
    events_file = job_dir(AppPaths(home), second) / EVENTS_FILE
    assert state.offset == events_file.stat().st_size and state.last_seq == 5


def test_final_best_comes_from_the_result(home: Path) -> None:
    run_id = make_run(home, status="queued")
    result = {"best_step": 2, "best_eval_loss": 1.9, "notes": ["Stopped early."], "status": "x"}
    _set_meta(home, run_id, result=result, machine={})
    _attempt(home, run_id, [_train(2, eval_loss=1.9), _train(4, eval_loss=1.8)], "succeeded")
    state = monitor.run_monitor(str(home), run_id)
    assert state.best == monitor.BestCheckpoint(step=2, eval_loss=1.9, final=True)
    assert "Stopped early." in state.warnings and state.result["best_step"] == 2
    assert not state.can_resume and state.resume_note is None


def test_resume_needs_a_checkpoint_and_the_model_lock(home: Path) -> None:
    run_id = make_run(home, status="queued")
    _attempt(home, run_id, [_train(1, loss=2.0)], "interrupted")
    state = monitor.run_monitor(str(home), run_id)
    assert not state.can_resume and "no checkpoint" in state.resume_note

    (AppPaths(home).runs / run_id / "checkpoints" / "checkpoint-1").mkdir(parents=True)
    runner = LocalJobRunner(AppPaths(home))
    holder = runner.submit(JobType.EVALUATE, {}, detach=False)  # holds the lock (this process)
    state = monitor.run_monitor(str(home), run_id)
    assert not state.can_resume and state.lock_holder.job_id == holder
    assert state.lock_holder.message.startswith(f"Another job is using the model: {holder}")
    assert holder in state.resume_note

    cancel_job(CancelJobRequest(home=str(home), job_id=holder, timeout=0))
    state = monitor.run_monitor(str(home), run_id)
    assert state.can_resume and state.lock_holder is None


def test_memory_pool_follows_the_device() -> None:
    assert monitor.memory_pool("cuda") == "VRAM"
    assert monitor.memory_pool("apple_gpu") == "unified memory"
    assert monitor.memory_pool("cpu") == "RAM"


def test_unknown_run(home: Path) -> None:
    from fliptuned.errors import NotAvailableError

    with pytest.raises(NotAvailableError):
        monitor.run_monitor(str(home), "run_nope")


def test_server_fold_of_the_recorded_stream_matches_the_web_rules() -> None:
    """The web tests fold the same recording with ``applyEvent``; both must agree on it."""
    recorded = Path(__file__).resolve().parents[2] / "web" / "src" / "test" / "train-events.json"
    events = json.loads(recorded.read_text(encoding="utf-8"))
    fold = monitor._Fold()
    for event in events:
        fold.apply(event, "job_1")
    train = [e for e in events if e["type"] == "train"]
    evals = [e for e in train if e["eval_loss"] is not None]
    lowest = min(e["eval_loss"] for e in evals)
    assert len(fold.points) == len(train)
    assert fold.best() == monitor.BestCheckpoint(
        step=next(e["step"] for e in evals if e["eval_loss"] == lowest),
        eval_loss=lowest,
        final=False,
    )
    messages = [line.message for line in fold.log]
    assert messages[0] == "Job job_1 is queued." and messages[-1] == "Job job_1 is succeeded."
    assert sum("validation loss" in m for m in messages) == len(evals)
