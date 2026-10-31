"""Job runner with real detached worker processes (runs on all three OSes in CI)."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fliptuned.config.app_home import AppPaths
from fliptuned.jobs.lock import ModelLockedError
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.jobs.state import JobType
from fliptuned.platform.process import ProcessHandle, is_alive, kill_tree
from fliptuned.services import jobs as jobs_service


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def _wait_until(check: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError("condition not reached in time")


def _start(home: Path, **kw) -> jobs_service.JobInfo:
    request = jobs_service.SelftestRequest(home=str(home), **kw)
    return jobs_service.start_selftest(request, detach=True)


def _running_handle(home: Path, job_id: str) -> ProcessHandle:
    runner = LocalJobRunner(AppPaths(home))

    def started() -> bool:
        return runner.get_row(job_id)["status"] == "running"

    _wait_until(started)
    row = runner.get_row(job_id)
    return ProcessHandle(pid=row["pid"], create_time=row["pid_create_time"])


def test_detached_job_runs_to_success_and_follow_sees_it(home: Path) -> None:
    job = _start(home, seconds=1.0, steps=4)
    assert job.detached and job.status in ("queued", "running")
    events = list(jobs_service.follow_job(str(home), job.id, poll=0.1, timeout=60))
    statuses = [e["status"] for e in events if e["type"] == "status"]
    assert statuses == ["queued", "running", "succeeded"]
    assert [e["done"] for e in events if e["type"] == "progress"] == [1, 2, 3, 4]
    assert jobs_service.job_result(str(home), job.id, jobs_service.SelftestResult).steps == 4
    log = Path(jobs_service.get_job(str(home), job.id).log_path).read_text(encoding="utf-8")
    assert "Traceback" not in log


def test_cooperative_cancel(home: Path) -> None:
    job = _start(home, seconds=60, steps=60)
    handle = _running_handle(home, job.id)
    outcome = jobs_service.cancel_job(
        jobs_service.CancelJobRequest(home=str(home), job_id=job.id, timeout=20)
    )
    assert outcome.status == "cancelled" and not outcome.forced
    _wait_until(lambda: not is_alive(handle), timeout=10)


def test_stuck_worker_is_force_stopped(home: Path) -> None:
    job = _start(home, seconds=120, steps=120, ignore_cancel=True, hold_model_lock=True)
    handle = _running_handle(home, job.id)
    outcome = jobs_service.cancel_job(
        jobs_service.CancelJobRequest(home=str(home), job_id=job.id, timeout=1)
    )
    assert outcome.forced and outcome.status == "cancelled"
    assert not is_alive(handle)
    listed = jobs_service.list_jobs(jobs_service.ListJobsRequest(home=str(home)))
    assert listed.model_lock_holder is None
    final = list(jobs_service.follow_job(str(home), job.id, timeout=10))[-1]
    assert final["status"] == "cancelled" and final["forced"] is True


def test_worker_survives_its_parent_exiting(home: Path) -> None:
    script = (
        "import sys\n"
        "from fliptuned.services import jobs\n"
        "job = jobs.start_selftest(jobs.SelftestRequest(home=sys.argv[1], seconds=3, steps=3),"
        " detach=True)\n"
        "print(job.id)\n"
    )
    parent = subprocess.run(
        [sys.executable, "-c", script, str(home)], capture_output=True, text=True, timeout=60
    )
    assert parent.returncode == 0, parent.stderr
    job_id = parent.stdout.strip().splitlines()[-1]
    # The parent is gone; the worker keeps going and finishes on its own.
    events = list(jobs_service.follow_job(str(home), job_id, poll=0.1, timeout=60))
    assert events[-1]["status"] == "succeeded"


def test_killed_worker_is_reconciled_to_interrupted(home: Path) -> None:
    job = _start(home, seconds=120, steps=120, hold_model_lock=True)
    handle = _running_handle(home, job.id)
    kill_tree(handle)
    assert jobs_service.reconcile_jobs(str(home)) == [job.id]
    info = jobs_service.get_job(str(home), job.id)
    assert info.status == "interrupted" and "ended unexpectedly" in (info.error_summary or "")
    assert list(jobs_service.follow_job(str(home), job.id, timeout=5))[-1]["status"] == (
        "interrupted"
    )
    assert (
        jobs_service.list_jobs(jobs_service.ListJobsRequest(home=str(home))).model_lock_holder
        is None
    )


def test_lock_contention_with_a_detached_holder(home: Path) -> None:
    holder = _start(home, seconds=60, steps=60, hold_model_lock=True)
    _running_handle(home, holder.id)
    runner = LocalJobRunner(AppPaths(home))
    with pytest.raises(ModelLockedError, match=holder.id):
        runner.submit(JobType.TRAIN, {}, detach=False)
    jobs_service.cancel_job(jobs_service.CancelJobRequest(home=str(home), job_id=holder.id))
    runner.submit(JobType.TRAIN, {}, detach=False)  # free again
