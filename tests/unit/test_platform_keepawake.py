from __future__ import annotations

import sys

from fliptuned.platform.keepawake import (
    KeepAwakeMethod,
    helper_command,
    keep_awake_in_this_process,
    select_method,
)


def _which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def test_selection_per_os() -> None:
    assert select_method("windows", _which(set())).method is KeepAwakeMethod.WINDOWS_EXECUTION_STATE
    assert select_method("macos", _which({"caffeinate"})).method is KeepAwakeMethod.CAFFEINATE
    linux = select_method("linux", _which({"systemd-inhibit", "tail"}))
    assert linux.method is KeepAwakeMethod.SYSTEMD_INHIBIT and linux.warning is None


def test_missing_tools_fall_back_with_warning() -> None:
    linux = select_method("linux", _which(set()))
    assert linux.method is KeepAwakeMethod.NONE and "systemd-inhibit" in (linux.warning or "")
    mac = select_method("macos", _which(set()))
    assert mac.method is KeepAwakeMethod.NONE and mac.warning


def test_helper_commands_are_tied_to_worker_pid() -> None:
    assert helper_command(KeepAwakeMethod.CAFFEINATE, 123) == ["caffeinate", "-i", "-w", "123"]
    inhibit = helper_command(KeepAwakeMethod.SYSTEMD_INHIBIT, 123)
    assert inhibit is not None and inhibit[0] == "systemd-inhibit" and "--pid=123" in inhibit
    assert helper_command(KeepAwakeMethod.WINDOWS_EXECUTION_STATE, 123) is None
    assert helper_command(KeepAwakeMethod.NONE, 123) is None


def test_in_process_keep_awake() -> None:
    with keep_awake_in_this_process() as applied:
        assert applied is (sys.platform == "win32")
