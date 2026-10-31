"""Error types and exit codes shared by every layer.

Error format: a plain-language ``message``, then a ``hint`` saying what to do, then optional
``details`` for the expandable section. None of these may contain record content or secrets.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1  # unexpected failure
    USAGE = 2  # bad command-line usage (Typer/Click convention)
    CONFIG_INVALID = 3
    NOT_AVAILABLE = 4  # backend/device/model/file not available on this machine
    CHECK_FAILED = 5  # a check (doctor, preflight, validation) reported a failure
    INTERRUPTED = 130


class FlipTunedError(Exception):
    exit_code: ExitCode = ExitCode.ERROR

    def __init__(self, message: str, *, hint: str | None = None, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


class ConfigError(FlipTunedError):
    exit_code = ExitCode.CONFIG_INVALID


class NotAvailableError(FlipTunedError):
    exit_code = ExitCode.NOT_AVAILABLE


class PathNotAllowedError(FlipTunedError):
    exit_code = ExitCode.NOT_AVAILABLE


class RegistryError(FlipTunedError):
    exit_code = ExitCode.ERROR
