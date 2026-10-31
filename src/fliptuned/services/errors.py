"""Error types re-exported for the CLI and API adapters."""

from fliptuned.errors import (
    ConfigError,
    ExitCode,
    FlipTunedError,
    NotAvailableError,
    PathNotAllowedError,
    RegistryError,
)
from fliptuned.jobs.lock import ModelLockedError

__all__ = [
    "ConfigError",
    "ExitCode",
    "FlipTunedError",
    "ModelLockedError",
    "NotAvailableError",
    "PathNotAllowedError",
    "RegistryError",
]
