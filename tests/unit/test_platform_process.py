from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from fliptuned.platform.process import (
    ProcessHandle,
    StopOutcome,
    handle_for_pid,
    is_alive,
    kill_tree,
    start_detached,
    stop,
)

WORKER = Path(__file__).resolve().parents[1] / "helpers" / "dummy_worker.py"


def _wait_for(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path.name}")


def _start(tmp_path: Path, mode: str) -> ProcessHandle:
    return start_detached(
        [sys.executable, str(WORKER), str(tmp_path), mode],
        log_path=tmp_path / "log.txt",
        cwd=tmp_path,
    )


def test_detached_worker_is_alive_and_stops_gracefully_on_cancel_marker(tmp_path: Path) -> None:
    handle = _start(tmp_path, "cooperative")
    try:
        _wait_for(tmp_path / "worker.pid")
        assert is_alive(handle)
        (tmp_path / "cancel.requested").write_text("1", encoding="utf-8")
        assert stop(handle, graceful_timeout=30) is StopOutcome.GRACEFUL
        assert not is_alive(handle)
        log = (tmp_path / "log.txt").read_text(encoding="utf-8")
        assert "worker started ✓" in log
    finally:
        kill_tree(handle)


def test_stubborn_worker_and_grandchild_are_force_killed(tmp_path: Path) -> None:
    handle = _start(tmp_path, "stubborn")
    try:
        _wait_for(tmp_path / "worker.pid")
        _wait_for(tmp_path / "grandchild.pid")
        grandchild = int((tmp_path / "grandchild.pid").read_text(encoding="utf-8"))
        assert psutil.pid_exists(grandchild)
        (tmp_path / "cancel.requested").write_text("1", encoding="utf-8")
        assert stop(handle, graceful_timeout=1, kill_timeout=10) is StopOutcome.FORCED
        assert not is_alive(handle)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _running(grandchild):
            time.sleep(0.05)
        assert not _running(grandchild)
    finally:
        kill_tree(handle)


def _running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_stop_on_exited_process_reports_already_exited(tmp_path: Path) -> None:
    handle = _start(tmp_path, "cooperative")
    _wait_for(tmp_path / "worker.pid")
    (tmp_path / "cancel.requested").write_text("1", encoding="utf-8")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and is_alive(handle):
        time.sleep(0.05)
    assert stop(handle) is StopOutcome.ALREADY_EXITED


def test_pid_reuse_guard_rejects_mismatched_create_time() -> None:
    me = handle_for_pid(os.getpid())
    assert me is not None and is_alive(me)
    reused = ProcessHandle(pid=me.pid, create_time=me.create_time - 100.0)
    assert not is_alive(reused)


def test_missing_pid_is_not_alive() -> None:
    assert handle_for_pid(2**31 - 2) is None
    assert not is_alive(ProcessHandle(pid=2**31 - 2, create_time=0.0))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows creation flags")
def test_windows_detached_worker_has_no_console(tmp_path: Path) -> None:
    handle = _start(tmp_path, "cooperative")
    try:
        _wait_for(tmp_path / "worker.pid")
        assert is_alive(handle)
    finally:
        (tmp_path / "cancel.requested").write_text("1", encoding="utf-8")
        stop(handle, graceful_timeout=30)
