"""Keep the machine awake while a training or evaluation job runs.

- macOS: ``caffeinate -i -w <worker_pid>`` (exits by itself when the worker exits)
- Linux: ``systemd-inhibit ... tail --pid=<worker_pid> -f /dev/null`` if available, else a warning
- Windows: ``SetThreadExecutionState`` inside the worker process (:func:`keep_awake_in_this_process`)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwakeMethod(StrEnum):
    CAFFEINATE = "caffeinate"
    SYSTEMD_INHIBIT = "systemd-inhibit"
    WINDOWS_EXECUTION_STATE = "SetThreadExecutionState"
    NONE = "none"


@dataclass(frozen=True)
class KeepAwakePlan:
    method: KeepAwakeMethod
    warning: str | None = None


def select_method(os_name: str, which: Callable[[str], str | None] = shutil.which) -> KeepAwakePlan:
    if os_name == "windows":
        return KeepAwakePlan(KeepAwakeMethod.WINDOWS_EXECUTION_STATE)
    if os_name == "macos":
        if which("caffeinate"):
            return KeepAwakePlan(KeepAwakeMethod.CAFFEINATE)
        return KeepAwakePlan(
            KeepAwakeMethod.NONE, "caffeinate not found; the Mac may sleep during jobs"
        )
    if which("systemd-inhibit") and which("tail"):
        return KeepAwakePlan(KeepAwakeMethod.SYSTEMD_INHIBIT)
    return KeepAwakePlan(
        KeepAwakeMethod.NONE,
        "systemd-inhibit not found; disable sleep manually for long training jobs",
    )


def helper_command(method: KeepAwakeMethod, worker_pid: int) -> list[str] | None:
    """Command for an external keep-awake helper tied to ``worker_pid``'s lifetime."""
    if method is KeepAwakeMethod.CAFFEINATE:
        return ["caffeinate", "-i", "-w", str(worker_pid)]
    if method is KeepAwakeMethod.SYSTEMD_INHIBIT:
        return [
            "systemd-inhibit",
            "--what=idle:sleep",
            "--who=FlipTuned",
            "--why=Training or evaluation job running",
            "--mode=block",
            "tail",
            f"--pid={worker_pid}",
            "-f",
            "/dev/null",
        ]
    return None


def start_helper(plan: KeepAwakePlan, worker_pid: int) -> subprocess.Popen | None:
    """Start the external helper (macOS/Linux). Windows uses the in-process call instead."""
    argv = helper_command(plan.method, worker_pid)
    if argv is None:
        return None
    return subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


@contextmanager
def keep_awake_in_this_process() -> Iterator[bool]:
    """Windows: prevent system sleep while the block runs. Yields whether it was applied."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes  # noqa: PLC0415 - Windows only

    kernel32 = ctypes.windll.kernel32
    applied = bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    try:
        yield applied
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
