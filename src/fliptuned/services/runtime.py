"""Process-wide setup for entry points: console encoding, privacy env vars, logging."""

from __future__ import annotations

import logging
import os

from fliptuned import __version__
from fliptuned.platform.console import setup_utf8_console
from fliptuned.utils.logging import configure_logging

# SPEC §3.2: no telemetry, offline by default (downloads opt in per command later).
PRIVACY_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "DO_NOT_TRACK": "1",
}


def version() -> str:
    return __version__


def apply_privacy_env(env: dict[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    for key, value in PRIVACY_ENV.items():
        target.setdefault(key, value)
    # Telemetry is never allowed back on, even if the user's shell enabled it.
    target["HF_HUB_DISABLE_TELEMETRY"] = "1"


def init_entry_point(verbose: bool = False) -> None:
    setup_utf8_console()
    apply_privacy_env()
    configure_logging(logging.INFO if verbose else logging.WARNING)
