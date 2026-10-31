"""``JobRunner`` interface and its local implementation (SPEC §13.3, §13.5 seam).

A job is a registry row plus a folder ``<app_home>/jobs/<job_id>/`` holding ``events.jsonl``,
``log.txt``, ``cancel.requested``, ``result.json`` and ``error.json``. A job runs either in the
calling process (:meth:`run_inline`, the CLI's default "wait") or in a detached worker
(``detach=True``) that survives the caller, the browser and server restarts.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fliptuned.config.app_home import AppPaths
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.jobs import lock
from fliptuned.jobs.events import EventWriter
from fliptuned.jobs.state import (
    ACTIVE,
    MODEL_LOCK_TYPES,
    TERMINAL,
    JobStatus,
    JobType,
    allowed_from,
)
from fliptuned.jobs.worker import CANCEL_FILE, EVENTS_FILE, LOG_FILE, execute, job_dir
from fliptuned.platform import process
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

POLL_S = 0.1


class CancelOutcome(BaseModel):
    job_id: str
    status: str
    forced: bool = False
    already_finished: bool = False


def new_job_id() -> str:
    return f"job_{datetime.now(UTC):%Y%m%d%H%M%S}_{secrets.token_hex(3)}"


def stored_envelope(job) -> dict[str, Any]:
    return json.loads(job["params_json"] or "{}")


class JobRunner(ABC):
    @abstractmethod
    def submit(
        self,
        job_type: JobType,
        params: dict[str, Any],
        *,
        detach: bool,
        holds_model_lock: bool | None = None,
        run_id: str | None = None,
    ) -> str: ...

    @abstractmethod
    def run_inline(
        self, job_id: str, progress_callback: Callable[[dict], None] | None = None
    ) -> JobStatus: ...

    @abstractmethod
    def cancel(self, job_id: str, timeout: float = 10.0) -> CancelOutcome: ...

    @abstractmethod
    def reconcile(self) -> list[str]: ...


class LocalJobRunner(JobRunner):
    def __init__(self, paths: AppPaths):
        self.paths = paths
        paths.ensure()
        migrate(paths.db)

    # ------------------------------------------------------------------ submit / run

    def submit(
        self,
        job_type: JobType,
        params: dict[str, Any],
        *,
        detach: bool,
        holds_model_lock: bool | None = None,
        run_id: str | None = None,
    ) -> str:
        needs_lock = job_type in MODEL_LOCK_TYPES if holds_model_lock is None else holds_model_lock
        job_id = new_job_id()
        folder = job_dir(self.paths, job_id)
        folder.mkdir(parents=True)
        me = process.handle_for_pid(os.getpid())
        assert me is not None
        envelope = {"params": params, "detached": detach, "model_lock": needs_lock}
        with connect(self.paths.db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if needs_lock:
                    lock.check_free(conn, job_id)
                repositories.insert_job(
                    conn,
                    job_id=job_id,
                    job_type=job_type.value,
                    params=envelope,
                    log_path=str(folder / LOG_FILE),
                    events_path=str(folder / EVENTS_FILE),
                    run_id=run_id,
                )
                repositories.set_job_process(conn, job_id, me.pid, me.create_time)
                if needs_lock:
                    repositories.insert_model_lock(conn, job_id, me.pid, me.create_time)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        EventWriter(folder / EVENTS_FILE).emit("status", status=JobStatus.QUEUED.value)
        if detach:
            self._spawn(job_id, folder)
        return job_id

    def _spawn(self, job_id: str, folder: Path) -> None:
        argv = [sys.executable, "-m", "fliptuned.jobs.worker", str(self.paths.home), job_id]
        try:
            handle = process.start_detached(argv, log_path=folder / LOG_FILE)
        except OSError as exc:
            self._finish(job_id, JobStatus.FAILED, f"Could not start the worker ({exc}).")
            raise FlipTunedError(
                "The background worker could not be started.",
                hint="Run the command without --detach, or check `fliptuned doctor`.",
                details=str(exc),
            ) from None
        with connect(self.paths.db) as conn, conn:
            job = repositories.get_job(conn, job_id)
            if job["status"] == JobStatus.QUEUED.value:
                repositories.set_job_process(conn, job_id, handle.pid, handle.create_time)
                repositories.update_model_lock_process(conn, job_id, handle.pid, handle.create_time)
        log.info("started worker %s for job %s", handle.pid, job_id)

    def run_inline(
        self, job_id: str, progress_callback: Callable[[dict], None] | None = None
    ) -> JobStatus:
        return execute(self.paths, job_id, progress_callback)

    # ------------------------------------------------------------------ cancel / reconcile

    def get_row(self, job_id: str):
        with connect(self.paths.db) as conn:
            job = repositories.get_job(conn, job_id)
        if job is None:
            raise NotAvailableError(
                f"No job {job_id}.", hint="List jobs with `fliptuned jobs list`."
            )
        return job

    @staticmethod
    def _handle(job) -> process.ProcessHandle | None:
        if job["pid"] is None or job["pid_create_time"] is None:
            return None
        return process.ProcessHandle(pid=job["pid"], create_time=job["pid_create_time"])

    def _finish(
        self, job_id: str, status: JobStatus, summary: str | None, **event_fields: Any
    ) -> bool:
        with connect(self.paths.db) as conn:
            with conn:
                moved = repositories.transition_job(
                    conn,
                    job_id,
                    status,
                    from_statuses=allowed_from(status),
                    error_summary=summary,
                    finished=True,
                )
            lock.release(conn, job_id)
            run_id = repositories.get_job(conn, job_id)["run_id"]
            if moved and run_id:  # a training run follows its job when the runner ends it
                repositories.update_run(
                    conn, run_id, status=status.value, from_statuses=("queued", "running")
                )
        if moved:
            EventWriter(job_dir(self.paths, job_id) / EVENTS_FILE).emit(
                "status", status=status.value, error=None, **event_fields
            )
        return moved

    def cancel(self, job_id: str, timeout: float = 10.0) -> CancelOutcome:
        job = self.get_row(job_id)
        if JobStatus(job["status"]) in TERMINAL:
            return CancelOutcome(job_id=job_id, status=job["status"], already_finished=True)
        (job_dir(self.paths, job_id) / CANCEL_FILE).write_text("cancel\n", encoding="utf-8")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get_row(job_id)
            if JobStatus(job["status"]) in TERMINAL:
                return CancelOutcome(job_id=job_id, status=job["status"])
            handle = self._handle(job)
            if handle is None or not process.is_alive(handle):
                break
            time.sleep(POLL_S)

        handle = self._handle(job)
        forced = False
        if handle is not None and handle.pid == os.getpid():
            pass  # our own in-process job: the marker and the status stop it; never kill ourselves
        elif handle is not None and process.is_alive(handle):
            forced = True
            if stored_envelope(job).get("detached"):
                process.stop(handle, graceful_timeout=2.0)
            else:  # an in-process job: stop only that process, never its process group
                process.kill_tree(handle)
        self._finish(job_id, JobStatus.CANCELLED, None, forced=forced)
        return CancelOutcome(job_id=job_id, status=self.get_row(job_id)["status"], forced=forced)

    def reconcile(self) -> list[str]:
        """Active jobs whose process is gone become ``interrupted`` (or ``failed`` if they never
        started); a lock held by a dead process is released."""
        changed = []
        with connect(self.paths.db) as conn:
            jobs = repositories.list_jobs(conn, [s.value for s in ACTIVE], limit=10_000)
        for job in jobs:
            handle = self._handle(job)
            if handle is not None and process.is_alive(handle):
                continue
            if job["status"] == JobStatus.RUNNING.value:
                status, summary = JobStatus.INTERRUPTED, "The worker process ended unexpectedly."
            else:
                status, summary = JobStatus.FAILED, "The worker never started."
            if self._finish(job["id"], status, summary):
                changed.append(job["id"])
        with connect(self.paths.db) as conn:
            row = repositories.get_model_lock(conn)
            if row is not None and not lock.holder_alive(conn, row):
                with conn:
                    repositories.delete_model_lock(conn)
        return changed
