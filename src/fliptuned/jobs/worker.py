"""Job execution. The same :func:`execute` runs a job in this process (the CLI's default
wait mode) or in a detached worker started as ``python -m fliptuned.jobs.worker <home> <job_id>``.

The worker owns the job's progress: it moves the job to ``running``, writes events, honours the
``cancel.requested`` marker through :meth:`JobContext.check_cancel`, writes ``result.json`` or
``error.json``, sets the final status and releases the model lock. Logs and errors carry no
record content: unexpected exceptions are logged by type and stack frames, without the message.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fliptuned.config.app_home import AppPaths
from fliptuned.errors import ExitCode, FlipTunedError
from fliptuned.jobs import lock
from fliptuned.jobs.events import EventWriter
from fliptuned.jobs.state import KEEP_AWAKE_TYPES, JobStatus, JobType, allowed_from
from fliptuned.platform import keepawake
from fliptuned.platform.detect import current_os
from fliptuned.platform.process import handle_for_pid
from fliptuned.registry import repositories
from fliptuned.registry.db import connect

EVENTS_FILE = "events.jsonl"
LOG_FILE = "log.txt"
CANCEL_FILE = "cancel.requested"
RESULT_FILE = "result.json"
ERROR_FILE = "error.json"


class JobCancelled(Exception):
    """Raised inside a handler when the cancel marker appears."""


def job_dir(paths: AppPaths, job_id: str) -> Path:
    return paths.jobs / job_id


@dataclass
class JobContext:
    paths: AppPaths
    job_id: str
    job_type: JobType
    params: dict[str, Any]
    events: EventWriter
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    _last_db_progress: float = field(default=0.0, repr=False)

    @property
    def home(self) -> str:
        return str(self.paths.home)

    @property
    def cancel_marker(self) -> Path:
        return job_dir(self.paths, self.job_id) / CANCEL_FILE

    def cancel_requested(self) -> bool:
        return self.cancel_marker.exists()

    def check_cancel(self) -> None:
        if self.cancel_requested():
            raise JobCancelled

    def emit(self, event_type: str, **fields: Any) -> None:
        """Write an event (e.g. ``train`` metrics) and pass it to the live callback."""
        event = self.events.emit(event_type, **fields)
        if self.progress_callback:
            self.progress_callback(event)

    def progress(self, done: int, total: int, message: str | None = None) -> None:
        event = self.events.emit("progress", done=done, total=total, message=message)
        if self.progress_callback:
            self.progress_callback(event)
        now = time.monotonic()
        if total and (done == total or now - self._last_db_progress > 1.0):
            self._last_db_progress = now
            with connect(self.paths.db) as conn, conn:
                repositories.set_job_progress(conn, self.job_id, round(done / total, 4))


@contextmanager
def _keep_awake(job_type: JobType, events: EventWriter) -> Iterator[None]:
    if job_type not in KEEP_AWAKE_TYPES:
        yield
        return
    plan = keepawake.select_method(current_os())
    if plan.warning:
        events.emit("log", level="warning", message=plan.warning)
    if plan.method is keepawake.KeepAwakeMethod.WINDOWS_EXECUTION_STATE:
        with keepawake.keep_awake_in_this_process():
            yield
        return
    helper = keepawake.start_helper(plan, os.getpid())
    try:
        yield
    finally:
        if helper is not None:
            with contextlib.suppress(OSError):
                helper.terminate()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _log_unexpected(path: Path, exc: BaseException) -> None:
    """Exception type and stack frames (file, line, function) only: messages can contain
    record text."""
    frames = "".join(
        f'  File "{f.filename}", line {f.lineno}, in {f.name}\n'
        for f in traceback.extract_tb(exc.__traceback__)
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"Unexpected {type(exc).__name__} (message withheld)\n{frames}\n")


def execute(
    paths: AppPaths,
    job_id: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> JobStatus:
    """Run a queued job to completion in this process. Returns the final status."""
    from fliptuned.jobs.handlers import HANDLERS

    folder = job_dir(paths, job_id)
    events = EventWriter(folder / EVENTS_FILE)
    me = handle_for_pid(os.getpid())
    assert me is not None
    with connect(paths.db) as conn:
        job = repositories.get_job(conn, job_id)
        if job is None:
            raise FlipTunedError(f"No job {job_id}.")
        job_type = JobType(job["type"])
        params = json.loads(job["params_json"] or "{}").get("params", {})
        with conn:
            started = repositories.transition_job(
                conn,
                job_id,
                JobStatus.RUNNING,
                from_statuses=allowed_from(JobStatus.RUNNING),
                started=True,
            )
            if started:
                repositories.set_job_process(conn, job_id, me.pid, me.create_time)
                repositories.update_model_lock_process(conn, job_id, me.pid, me.create_time)
    if not started:  # cancelled before it started, or already handled
        with connect(paths.db) as conn:
            return JobStatus(repositories.get_job(conn, job_id)["status"])
    events.emit("status", status=JobStatus.RUNNING.value, pid=me.pid)

    ctx = JobContext(paths, job_id, job_type, params, events, progress_callback)
    final, error, result = JobStatus.SUCCEEDED, None, None
    interrupted = False
    try:
        with _keep_awake(job_type, events):
            ctx.check_cancel()
            result = HANDLERS[job_type](ctx)
    except JobCancelled:
        final = JobStatus.CANCELLED
    except KeyboardInterrupt:
        final, interrupted = JobStatus.CANCELLED, True
    except FlipTunedError as exc:
        final = JobStatus.FAILED
        error = {
            "message": exc.message,
            "hint": exc.hint,
            "details": exc.details,
            "exit_code": int(exc.exit_code),
        }
    except Exception as exc:
        final = JobStatus.FAILED
        _log_unexpected(folder / LOG_FILE, exc)
        error = {
            "message": f"The job failed with an unexpected {type(exc).__name__}.",
            "hint": f"See the job log: {folder / LOG_FILE}",
            "details": None,
            "exit_code": int(ExitCode.ERROR),
        }

    if result is not None:
        _write_json(folder / RESULT_FILE, result)
    if error is not None:
        _write_json(folder / ERROR_FILE, error)
    with connect(paths.db) as conn:
        with conn:
            moved = repositories.transition_job(
                conn,
                job_id,
                final,
                from_statuses=(JobStatus.RUNNING.value,),
                error_summary=error["message"] if error else None,
                finished=True,
                progress=1.0 if final is JobStatus.SUCCEEDED else None,
            )
        lock.release(conn, job_id)
        status = final if moved else JobStatus(repositories.get_job(conn, job_id)["status"])
    if moved:
        events.emit("status", status=final.value, error=error)
    if interrupted:
        raise KeyboardInterrupt
    return status


def main(argv: list[str] | None = None) -> int:
    from fliptuned.services.runtime import init_entry_point

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m fliptuned.jobs.worker <app_home> <job_id>", file=sys.stderr)
        return int(ExitCode.USAGE)
    init_entry_point()
    status = execute(AppPaths(Path(args[0])), args[1])
    return 0 if status is JobStatus.SUCCEEDED else 1


if __name__ == "__main__":
    # `python -m` runs this file as `__main__`, a second copy of the module whose classes
    # (e.g. JobCancelled) differ from the ones handlers import. Run the package module instead.
    from fliptuned.jobs import worker as _worker

    sys.exit(_worker.main())
