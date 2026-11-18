"""Job runner, in process: states, events, results, errors, model lock, reconciliation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.app_home import AppPaths
from fliptuned.jobs import handlers, lock, worker
from fliptuned.jobs.events import EventWriter, read_events
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.jobs.state import JobStatus, JobType, allowed_from, can_transition
from fliptuned.platform import keepawake
from fliptuned.platform.process import ProcessHandle, handle_for_pid
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services import jobs as jobs_service
from fliptuned.services.datasets import BuildDatasetRequest
from tests.helpers.pipeline import SAMPLE_DATASET

runner_cli = CliRunner()


@pytest.fixture
def runner(tmp_path: Path) -> LocalJobRunner:
    return LocalJobRunner(AppPaths(tmp_path / "home"))


def _row(runner: LocalJobRunner, job_id: str):
    return runner.get_row(job_id)


def _events(runner: LocalJobRunner, job_id: str) -> list[dict]:
    return read_events(runner.paths.jobs / job_id / "events.jsonl")[0]


def _dead_handle() -> ProcessHandle:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    handle = handle_for_pid(proc.pid)
    proc.wait()
    assert handle is not None
    return handle


# --------------------------------------------------------------------------- states & events


def test_transition_table() -> None:
    assert can_transition(JobStatus.QUEUED, JobStatus.RUNNING)
    assert can_transition(JobStatus.RUNNING, JobStatus.INTERRUPTED)
    assert not can_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)
    assert not can_transition(JobStatus.QUEUED, JobStatus.INTERRUPTED)
    assert set(allowed_from(JobStatus.CANCELLED)) == {"queued", "running"}
    for terminal in ("succeeded", "failed", "cancelled", "interrupted"):
        assert not any(can_transition(JobStatus(terminal), s) for s in JobStatus)


def test_conditional_transition_refuses_wrong_state(runner: LocalJobRunner) -> None:
    job_id = runner.submit(JobType.SELFTEST, {}, detach=False)
    with connect(runner.paths.db) as conn, conn:
        assert not repositories.transition_job(
            conn, job_id, "succeeded", from_statuses=allowed_from(JobStatus.SUCCEEDED)
        )
        assert repositories.transition_job(
            conn, job_id, "running", from_statuses=allowed_from(JobStatus.RUNNING)
        )


def test_events_reader_skips_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.emit("status", status="queued")
    writer.emit("progress", done=1, total=2, message="café ✓")
    with open(path, "ab") as handle:
        handle.write(b'{"seq": 3, "type": "prog')
    events, offset = read_events(path)
    assert [e["seq"] for e in events] == [1, 2] and events[1]["message"] == "café ✓"
    assert read_events(path, offset) == ([], offset)
    with open(path, "ab") as handle:
        handle.write(b'ress"}\n')
    more, _ = read_events(path, offset)
    assert more == [{"seq": 3, "type": "progress"}]
    assert EventWriter(path).seq == 3  # a new writer continues the numbering


# --------------------------------------------------------------------------- running inline


def test_inline_selftest_succeeds_with_events_and_result(runner: LocalJobRunner) -> None:
    job_id = runner.submit(JobType.SELFTEST, {"seconds": 0.2, "steps": 3}, detach=False)
    assert _row(runner, job_id)["status"] == "queued"
    seen = []
    assert runner.run_inline(job_id, seen.append) is JobStatus.SUCCEEDED
    row = _row(runner, job_id)
    assert row["status"] == "succeeded" and row["progress"] == 1.0 and row["finished_at"]
    kinds = [(e["type"], e.get("status") or e.get("done")) for e in _events(runner, job_id)]
    assert kinds == [
        ("status", "queued"),
        ("status", "running"),
        ("progress", 1),
        ("progress", 2),
        ("progress", 3),
        ("status", "succeeded"),
    ]
    assert [e["done"] for e in seen] == [1, 2, 3]
    result = json.loads((runner.paths.jobs / job_id / "result.json").read_text("utf-8"))
    assert result == {"steps": 3, "seconds": 0.2}
    # Running a finished job again does nothing.
    assert runner.run_inline(job_id) is JobStatus.SUCCEEDED


def test_cancel_marker_before_start_cancels(runner: LocalJobRunner) -> None:
    job_id = runner.submit(JobType.SELFTEST, {"seconds": 5}, detach=False)
    (runner.paths.jobs / job_id / "cancel.requested").write_text("x", encoding="utf-8")
    assert runner.run_inline(job_id) is JobStatus.CANCELLED


def test_flip_tuned_error_is_recorded_with_exit_code(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    job = jobs_service.start_build_dataset(
        BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=home), detach=False
    )
    final = jobs_service.run_inline(home, job.id)
    assert final.status == "failed" and "ingested" in (final.error_summary or "")
    with pytest.raises(jobs_service.JobFailedError) as info:
        jobs_service.job_result(home, job.id, jobs_service.SelftestResult)
    assert int(info.value.exit_code) == 4 and info.value.hint
    assert jobs_service.job_error(home, job.id)["exit_code"] == 4


def test_unexpected_error_hides_its_message(
    runner: LocalJobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx):
        raise ValueError("SECRET record text")

    monkeypatch.setitem(handlers.HANDLERS, JobType.SELFTEST, boom)
    job_id = runner.submit(JobType.SELFTEST, {}, detach=False)
    assert runner.run_inline(job_id) is JobStatus.FAILED
    folder = runner.paths.jobs / job_id
    log = (folder / "log.txt").read_text(encoding="utf-8")
    assert "ValueError" in log and "SECRET" not in log
    assert "SECRET" not in (folder / "error.json").read_text(encoding="utf-8")
    assert "SECRET" not in (folder / "events.jsonl").read_text(encoding="utf-8")


def test_out_of_memory_gets_a_clear_message(
    runner: LocalJobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    for exc in (
        MemoryError("SECRET"),
        OutOfMemoryError("SECRET"),
        RuntimeError("CUDA out of memory"),
    ):

        def boom(ctx, exc=exc):
            raise exc

        monkeypatch.setitem(handlers.HANDLERS, JobType.SELFTEST, boom)
        job_id = runner.submit(JobType.SELFTEST, {}, detach=False)
        assert runner.run_inline(job_id) is JobStatus.FAILED
        error = json.loads((runner.paths.jobs / job_id / "error.json").read_text("utf-8"))
        assert error["message"] == "The job ran out of memory."
        assert "batch_size" in error["hint"] and "SECRET" not in json.dumps(error)


def test_keyboard_interrupt_cancels_and_reraises(
    runner: LocalJobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(ctx):
        raise KeyboardInterrupt

    monkeypatch.setitem(handlers.HANDLERS, JobType.SELFTEST, interrupted)
    job_id = runner.submit(JobType.SELFTEST, {}, detach=False, holds_model_lock=True)
    with pytest.raises(KeyboardInterrupt):
        runner.run_inline(job_id)
    assert _row(runner, job_id)["status"] == "cancelled"
    with connect(runner.paths.db) as conn:
        assert repositories.get_model_lock(conn) is None


def test_keep_awake_only_for_train_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = EventWriter(tmp_path / "e.jsonl")
    calls = []
    monkeypatch.setattr(
        keepawake,
        "select_method",
        lambda os_name: (
            calls.append(os_name)
            or keepawake.KeepAwakePlan(keepawake.KeepAwakeMethod.NONE, "no helper here")
        ),
    )
    with worker._keep_awake(JobType.SELFTEST, events):
        pass
    assert calls == []
    with worker._keep_awake(JobType.TRAIN, events):
        pass
    assert len(calls) == 1
    (warning,) = read_events(tmp_path / "e.jsonl")[0]
    assert warning["type"] == "log" and warning["message"] == "no helper here"


# --------------------------------------------------------------------------- model lock


def test_lock_contention_names_the_holder(runner: LocalJobRunner) -> None:
    holder = runner.submit(
        JobType.SELFTEST, {"seconds": 0.1, "steps": 1}, detach=False, holds_model_lock=True
    )
    with pytest.raises(lock.ModelLockedError) as info:
        runner.submit(JobType.TRAIN, {}, detach=False)
    assert holder in info.value.message and "jobs cancel" in (info.value.hint or "")
    runner.submit(JobType.SELFTEST, {}, detach=False)  # jobs without the lock still start
    runner.run_inline(holder)  # finishing releases the lock
    with connect(runner.paths.db) as conn:
        assert repositories.get_model_lock(conn) is None
    runner.submit(JobType.SELFTEST, {}, detach=False, holds_model_lock=True)


def test_stale_lock_and_pid_reuse_are_not_honoured(runner: LocalJobRunner) -> None:
    holder = runner.submit(JobType.SELFTEST, {}, detach=False, holds_model_lock=True)
    dead = _dead_handle()
    with connect(runner.paths.db) as conn, conn:
        repositories.update_model_lock_process(conn, holder, dead.pid, dead.create_time)
    with connect(runner.paths.db) as conn:
        assert lock.current_holder(conn) is None
    # Same PID as a live process but a different create time: a reused PID, not the holder.
    me = handle_for_pid(__import__("os").getpid())
    with connect(runner.paths.db) as conn, conn:
        repositories.update_model_lock_process(conn, holder, me.pid, me.create_time - 100)
    with connect(runner.paths.db) as conn:
        assert lock.current_holder(conn) is None
    taken = runner.submit(JobType.TRAIN, {}, detach=False)  # stale lock is replaced
    with connect(runner.paths.db) as conn:
        assert repositories.get_model_lock(conn)["job_id"] == taken


# --------------------------------------------------------------------------- reconciliation


def test_reconcile_marks_dead_jobs(runner: LocalJobRunner) -> None:
    running = runner.submit(JobType.SELFTEST, {}, detach=False, holds_model_lock=True)
    queued = runner.submit(JobType.SELFTEST, {}, detach=False)
    alive = runner.submit(JobType.SELFTEST, {}, detach=False)  # this process: still alive
    dead = _dead_handle()
    with connect(runner.paths.db) as conn, conn:
        repositories.transition_job(conn, running, "running", from_statuses=("queued",))
        for job_id in (running, queued):
            repositories.set_job_process(conn, job_id, dead.pid, dead.create_time)
        repositories.update_model_lock_process(conn, running, dead.pid, dead.create_time)
    assert sorted(runner.reconcile()) == sorted([running, queued])
    assert _row(runner, running)["status"] == "interrupted"
    assert _row(runner, queued)["status"] == "failed"
    assert _row(runner, alive)["status"] == "queued"
    assert _events(runner, running)[-1] == {
        **_events(runner, running)[-1],
        "type": "status",
        "status": "interrupted",
    }
    with connect(runner.paths.db) as conn:
        assert repositories.get_model_lock(conn) is None
    assert runner.reconcile() == []


def test_cancel_finished_or_queued_job(runner: LocalJobRunner) -> None:
    done = runner.submit(JobType.SELFTEST, {"seconds": 0.1, "steps": 1}, detach=False)
    runner.run_inline(done)
    assert runner.cancel(done).already_finished
    queued = runner.submit(JobType.SELFTEST, {}, detach=False)
    dead = _dead_handle()
    with connect(runner.paths.db) as conn, conn:
        repositories.set_job_process(conn, queued, dead.pid, dead.create_time)
    outcome = runner.cancel(queued, timeout=1)
    assert outcome.status == "cancelled" and not outcome.forced


# --------------------------------------------------------------------------- CLI


def test_cli_jobs_list_follow_and_selftest(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    ran = runner_cli.invoke(
        app, ["jobs", "selftest", "--seconds", "0.2", "--steps", "2", "--home", home]
    )
    assert ran.exit_code == 0, ran.output
    assert "Selftest passed: 2 steps" in ran.output and "1/2 step 1 of 2" in ran.output
    listed = runner_cli.invoke(app, ["jobs", "list", "--home", home, "--json"])
    data = json.loads(listed.output)
    (job,) = data["jobs"]
    assert job["type"] == "selftest" and job["status"] == "succeeded" and not job["detached"]
    followed = runner_cli.invoke(app, ["jobs", "follow", job["id"], "--home", home])
    assert followed.exit_code == 0 and "status   succeeded" in followed.output
    missing = runner_cli.invoke(app, ["jobs", "follow", "job_nope", "--home", home])
    assert missing.exit_code == 4 and "No job" in missing.output


def test_cli_failed_job_keeps_exit_code(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    built = runner_cli.invoke(
        app, ["build-dataset", "--config", str(SAMPLE_DATASET), "--home", home]
    )
    assert built.exit_code == 4 and "Nothing has been ingested" in built.output
    job_id = json.loads(runner_cli.invoke(app, ["jobs", "list", "--home", home, "--json"]).output)[
        "jobs"
    ][0]["id"]
    followed = runner_cli.invoke(app, ["jobs", "follow", job_id, "--home", home])
    assert followed.exit_code == 4 and "status   failed" in followed.output


def test_invalid_config_fails_before_any_job(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: dataset\n", encoding="utf-8")
    result = runner_cli.invoke(app, ["build-dataset", "--config", str(bad), "--home", home])
    assert result.exit_code == 3
    assert jobs_service.list_jobs(jobs_service.ListJobsRequest(home=home)).jobs == []
