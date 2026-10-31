"""Jobs: start, list, follow, cancel and reconcile (SPEC §13.3).

Start functions validate their request up front (a bad config fails before any job exists),
then submit a job. With ``detach`` the job runs in a background worker; otherwise the caller
runs it with :func:`run_inline` and reads the typed result with :func:`job_result`.
Every list/follow/start call first reconciles: active jobs whose process is gone become
``interrupted``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.config.loader import load_config
from fliptuned.config.models import DatasetConfig, IngestConfig
from fliptuned.errors import ExitCode, FlipTunedError, NotAvailableError
from fliptuned.jobs import lock
from fliptuned.jobs.events import read_events
from fliptuned.jobs.runner import CancelOutcome, LocalJobRunner, stored_envelope
from fliptuned.jobs.state import TERMINAL, JobStatus, JobType
from fliptuned.jobs.worker import ERROR_FILE, EVENTS_FILE, RESULT_FILE, job_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.datasets import BuildDatasetRequest
from fliptuned.services.ingest import IngestRequest

M = TypeVar("M", bound=BaseModel)

__all__ = ["CancelOutcome", "JobStatus", "JobType"]


class JobFailedError(FlipTunedError):
    """A job failed; carries the failing step's exit code so the CLI exits the same way."""

    def __init__(self, message: str, *, hint=None, details=None, exit_code: int = 1):
        super().__init__(message, hint=hint, details=details)
        self.exit_code = ExitCode(exit_code)


class JobInfo(BaseModel):
    id: str
    type: str
    status: str
    progress: float | None
    detached: bool
    holds_model_lock: bool
    pid: int | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    error_summary: str | None
    run_id: str | None
    events_path: str | None
    log_path: str | None


def _runner(home: str | None) -> LocalJobRunner:
    return LocalJobRunner(AppPaths(resolve_app_home(home).path))


def _info(row) -> JobInfo:
    envelope = stored_envelope(row)
    return JobInfo(
        id=row["id"],
        type=row["type"],
        status=row["status"],
        progress=row["progress"],
        detached=bool(envelope.get("detached")),
        holds_model_lock=bool(envelope.get("model_lock")),
        pid=row["pid"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_summary=row["error_summary"],
        run_id=row["run_id"],
        events_path=row["events_path"],
        log_path=row["log_path"],
    )


def get_job(home: str | None, job_id: str) -> JobInfo:
    runner = _runner(home)
    return _info(runner.get_row(job_id))


# --------------------------------------------------------------------------- start


class SelftestRequest(BaseModel):
    home: str | None = None
    seconds: float = Field(default=10.0, gt=0, le=3600, description="How long the job runs.")
    steps: int = Field(default=10, ge=1, le=10_000, description="Progress events to write.")
    hold_model_lock: bool = Field(
        default=False, description="Take the global model lock, like a training job would."
    )
    ignore_cancel: bool = Field(
        default=False, description="Simulate a stuck worker (checks force-stop)."
    )


class SelftestResult(BaseModel):
    steps: int
    seconds: float


def _submit(home: str | None, job_type: JobType, params: dict, detach: bool, **kw) -> JobInfo:
    runner = _runner(home)
    runner.reconcile()
    job_id = runner.submit(job_type, params, detach=detach, **kw)
    return _info(runner.get_row(job_id))


def start_ingest(request: IngestRequest, *, detach: bool) -> JobInfo:
    load_config(Path(request.config_path).expanduser().resolve(), IngestConfig)
    params = request.model_dump(exclude={"home"})
    params["config_path"] = str(Path(request.config_path).expanduser().resolve())
    return _submit(request.home, JobType.INGEST, params, detach)


def start_build_dataset(request: BuildDatasetRequest, *, detach: bool) -> JobInfo:
    load_config(Path(request.config_path).expanduser().resolve(), DatasetConfig)
    params = request.model_dump(exclude={"home"})
    params["config_path"] = str(Path(request.config_path).expanduser().resolve())
    return _submit(request.home, JobType.BUILD_DATASET, params, detach)


def start_selftest(request: SelftestRequest, *, detach: bool) -> JobInfo:
    params = request.model_dump(exclude={"home", "hold_model_lock"})
    return _submit(
        request.home,
        JobType.SELFTEST,
        params,
        detach,
        holds_model_lock=request.hold_model_lock,
    )


def run_inline(
    home: str | None, job_id: str, progress: Callable[[dict], None] | None = None
) -> JobInfo:
    runner = _runner(home)
    runner.run_inline(job_id, progress)
    return _info(runner.get_row(job_id))


def job_result[M: BaseModel](home: str | None, job_id: str, model: type[M]) -> M:
    """The typed result of a succeeded job; raises :class:`JobFailedError` otherwise."""
    paths = AppPaths(resolve_app_home(home).path)
    info = get_job(home, job_id)
    folder = job_dir(paths, job_id)
    if info.status == JobStatus.SUCCEEDED.value:
        data = json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))
        return model.model_validate(data)
    if info.status == JobStatus.FAILED.value and (folder / ERROR_FILE).exists():
        error = json.loads((folder / ERROR_FILE).read_text(encoding="utf-8"))
        raise JobFailedError(
            error["message"],
            hint=error.get("hint"),
            details=error.get("details"),
            exit_code=error.get("exit_code", 1),
        )
    raise JobFailedError(
        f"Job {job_id} ended as {info.status}"
        + (f": {info.error_summary}" if info.error_summary else "."),
        hint=f"See `fliptuned jobs follow {job_id}`.",
    )


def job_error(home: str | None, job_id: str) -> dict | None:
    """``error.json`` of a failed job (message, hint, details, exit_code), if any."""
    path = job_dir(AppPaths(resolve_app_home(home).path), job_id) / ERROR_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- list / follow / cancel


class ListJobsRequest(BaseModel):
    home: str | None = None
    statuses: list[JobStatus] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=1000)


class ListJobsResult(BaseModel):
    jobs: list[JobInfo]
    model_lock_holder: str | None
    reconciled: list[str] = Field(description="Jobs just marked interrupted (process gone).")


def list_jobs(request: ListJobsRequest) -> ListJobsResult:
    runner = _runner(request.home)
    reconciled = runner.reconcile()
    with connect(runner.paths.db) as conn:
        rows = repositories.list_jobs(conn, [s.value for s in request.statuses], request.limit)
        holder = lock.current_holder(conn)
    return ListJobsResult(
        jobs=[_info(r) for r in rows], model_lock_holder=holder, reconciled=reconciled
    )


def follow_job(
    home: str | None,
    job_id: str,
    *,
    poll: float = 0.2,
    timeout: float | None = None,
    offset: int = 0,
) -> Iterator[dict]:
    """Yield the job's events (from the start) until it reaches a final status.

    Reads the same ``events.jsonl`` the API streams. If the worker process disappears, the
    job is reconciled to ``interrupted`` and that final event is yielded too.
    """
    runner = _runner(home)
    runner.get_row(job_id)  # raises for an unknown job
    path = job_dir(runner.paths, job_id) / EVENTS_FILE
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        events, offset = read_events(path, offset)
        for event in events:
            yield event
            if event.get("type") == "status" and event.get("status") in TERMINAL:
                return
        job = runner.get_row(job_id)
        if JobStatus(job["status"]) in TERMINAL:
            # The final event may have been written between our read and this check.
            events, offset = read_events(path, offset)
            yield from events
            return
        if not events:
            runner.reconcile()
        if deadline is not None and time.monotonic() > deadline:
            raise NotAvailableError(
                f"Job {job_id} is still {job['status']}.",
                hint=f"Follow it again later: `fliptuned jobs follow {job_id}`.",
            )
        time.sleep(poll)


class JobOutcome(BaseModel):
    job: JobInfo
    result: dict | None = Field(description="The job's result, once it succeeded.")
    error: dict | None = Field(description="message / hint / details, if it failed.")


def job_outcome(home: str | None, job_id: str) -> JobOutcome:
    """A job's status with its result or error (``result.json`` / ``error.json``)."""
    paths = AppPaths(resolve_app_home(home).path)
    info = get_job(home, job_id)
    folder = job_dir(paths, job_id)
    result = None
    if info.status == JobStatus.SUCCEEDED.value and (folder / RESULT_FILE).exists():
        result = json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))
    return JobOutcome(job=info, result=result, error=job_error(home, job_id))


class JobEvents(BaseModel):
    events: list[dict]
    offset: int = Field(description="Byte offset to read from next (the SSE event ID).")
    finished: bool = Field(description="The job has reached a final status.")


def read_job_events(home: str | None, job_id: str, offset: int = 0) -> JobEvents:
    """Events written since ``offset``: one poll of the same ``events.jsonl`` that
    ``follow_job`` reads. Reconciles when nothing new arrived (a vanished worker ends the job)."""
    runner = _runner(home)
    runner.get_row(job_id)  # raises for an unknown job
    events, new_offset = read_events(job_dir(runner.paths, job_id) / EVENTS_FILE, offset)
    if not events:
        runner.reconcile()
    finished = JobStatus(runner.get_row(job_id)["status"]) in TERMINAL
    if finished:  # the final event may land between the read and the status check
        more, new_offset = read_events(job_dir(runner.paths, job_id) / EVENTS_FILE, new_offset)
        events += more
    return JobEvents(events=events, offset=new_offset, finished=finished)


class CancelJobRequest(BaseModel):
    home: str | None = None
    job_id: str
    timeout: float = Field(
        default=10.0, ge=0, le=600, description="Seconds to wait before force-stopping."
    )


def cancel_job(request: CancelJobRequest) -> CancelOutcome:
    return _runner(request.home).cancel(request.job_id, request.timeout)


def reconcile_jobs(home: str | None) -> list[str]:
    return _runner(home).reconcile()


def job_folder(home: str | None, job_id: str) -> Path:
    paths = AppPaths(resolve_app_home(home).path)
    folder = job_dir(paths, job_id)
    if not folder.exists():
        raise NotAvailableError(f"No job {job_id}.")
    return folder
