"""Global model lock (SPEC §13.3): one registry row plus a process check.

The lock is held by a job. It counts as held only while that job is queued or running *and* the
recorded process is alive with the recorded create time, so a crashed worker, or a new process
that reused its PID, never keeps the lock. Acquiring runs in a ``BEGIN IMMEDIATE`` transaction,
so two requests can't both get it.
"""

from __future__ import annotations

import sqlite3

from fliptuned.errors import ExitCode, FlipTunedError
from fliptuned.jobs.state import ACTIVE
from fliptuned.platform.process import ProcessHandle, is_alive
from fliptuned.registry import repositories


class ModelLockedError(FlipTunedError):
    exit_code = ExitCode.NOT_AVAILABLE


def holder_alive(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    job = repositories.get_job(conn, row["job_id"])
    if job is None or job["status"] not in ACTIVE:
        return False
    return is_alive(ProcessHandle(pid=row["pid"], create_time=row["pid_create_time"]))


def current_holder(conn: sqlite3.Connection) -> str | None:
    """Job ID holding the lock, or None (stale rows don't count)."""
    row = repositories.get_model_lock(conn)
    return row["job_id"] if row is not None and holder_alive(conn, row) else None


def check_free(conn: sqlite3.Connection, job_id: str) -> None:
    """Inside the caller's transaction: refuse if another live job holds the lock, else clear
    a stale row. Raises :class:`ModelLockedError` naming the holder."""
    row = repositories.get_model_lock(conn)
    if row is None or row["job_id"] == job_id:
        return
    if holder_alive(conn, row):
        holder = repositories.get_job(conn, row["job_id"])
        raise ModelLockedError(
            f"Another job is using the model: {row['job_id']} ({holder['type']}, "
            f"{holder['status']}).",
            hint=f"Wait for it to finish, or cancel it: `fliptuned jobs cancel {row['job_id']}`.",
        )
    repositories.delete_model_lock(conn)


def acquire(conn: sqlite3.Connection, job_id: str, handle: ProcessHandle) -> None:
    """Take the lock for ``job_id`` (standalone transaction)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        check_free(conn, job_id)
        if repositories.get_model_lock(conn) is None:
            repositories.insert_model_lock(conn, job_id, handle.pid, handle.create_time)
        else:
            repositories.update_model_lock_process(conn, job_id, handle.pid, handle.create_time)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def release(conn: sqlite3.Connection, job_id: str) -> None:
    with conn:
        repositories.delete_model_lock(conn, job_id)
