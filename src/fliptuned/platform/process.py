"""Detached worker processes: start, liveness (PID + create time), graceful and forced stop.

Graceful cancel is the job runner's ``cancel.requested`` marker, which works on every OS.
On POSIX, :func:`stop` also sends SIGTERM to the process group; Windows has no console
signal for detached processes, so it only waits before force-killing the tree.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import psutil

CREATE_TIME_TOLERANCE_S = 0.01

# Windows creation flags (defined here so the module imports on every OS).
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_ERROR_ACCESS_DENIED = 5


@dataclass(frozen=True)
class ProcessHandle:
    pid: int
    create_time: float


class StopOutcome(StrEnum):
    ALREADY_EXITED = "already_exited"
    GRACEFUL = "graceful"
    FORCED = "forced"


def start_detached(
    argv: Sequence[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcessHandle:
    """Start ``argv`` detached from this process; stdout and stderr go to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ if env is None else env)
    full_env.setdefault("PYTHONUTF8", "1")
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    with open(log_path, "ab") as log:
        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "cwd": str(cwd) if cwd else None,
            "env": full_env,
            "close_fds": True,
        }
        if sys.platform == "win32":
            proc = _popen_windows(list(argv), kwargs)
        else:
            proc = subprocess.Popen(list(argv), start_new_session=True, **kwargs)
    return ProcessHandle(pid=proc.pid, create_time=psutil.Process(proc.pid).create_time())


def _popen_windows(argv: list[str], kwargs: dict) -> subprocess.Popen:
    base = _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS
    try:
        # Break away from the launching terminal's job object so the worker survives it.
        return subprocess.Popen(argv, creationflags=base | _CREATE_BREAKAWAY_FROM_JOB, **kwargs)
    except OSError as exc:
        if getattr(exc, "winerror", None) != _ERROR_ACCESS_DENIED:
            raise
        return subprocess.Popen(argv, creationflags=base, **kwargs)


def handle_for_pid(pid: int) -> ProcessHandle | None:
    try:
        return ProcessHandle(pid=pid, create_time=psutil.Process(pid).create_time())
    except psutil.Error:
        return None


def is_alive(handle: ProcessHandle) -> bool:
    """True only if ``handle.pid`` is running *and* is the same process (guards PID reuse)."""
    _reap(handle.pid)
    try:
        proc = psutil.Process(handle.pid)
        if abs(proc.create_time() - handle.create_time) > CREATE_TIME_TOLERANCE_S:
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def wait_for_exit(handle: ProcessHandle, timeout: float, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(handle):
            return True
        time.sleep(poll)
    return not is_alive(handle)


def stop(
    handle: ProcessHandle, *, graceful_timeout: float = 10.0, kill_timeout: float = 5.0
) -> StopOutcome:
    """Wait for a graceful exit (after the caller wrote the cancel marker), then kill the tree."""
    if not is_alive(handle):
        return StopOutcome.ALREADY_EXITED
    if sys.platform != "win32":
        _signal_group(handle.pid)
    if wait_for_exit(handle, graceful_timeout):
        return StopOutcome.GRACEFUL
    kill_tree(handle, timeout=kill_timeout)
    return StopOutcome.FORCED


def kill_tree(handle: ProcessHandle, *, timeout: float = 5.0) -> None:
    """Force-stop the process and all of its descendants."""
    if not is_alive(handle):
        return
    try:
        root = psutil.Process(handle.pid)
        procs = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    psutil.wait_procs(procs, timeout=timeout)
    _reap(handle.pid)


def _signal_group(pid: int) -> None:
    import signal  # noqa: PLC0415

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def _reap(pid: int) -> None:
    """Collect an exited child so it doesn't linger as a zombie (POSIX only)."""
    if sys.platform == "win32":
        return
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)
