"""Process-wide setup for entry points: console encoding, privacy env vars, logging."""

from __future__ import annotations

import logging
import os

from fliptuned import __version__
from fliptuned.platform.console import setup_utf8_console
from fliptuned.utils.logging import configure_logging

# No telemetry, offline by default. Allowed downloads switch the hub library's own flag for
# one call (models/hub.py); they never depend on these variables.
FORCED_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    # Never attach a saved Hugging Face account token to requests: catalog models are public.
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
}
DEFAULT_ENV = {"TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"}
PRIVACY_ENV = {**FORCED_ENV, **DEFAULT_ENV}


def version() -> str:
    return __version__


def apply_privacy_env(env: dict[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    for key, value in DEFAULT_ENV.items():
        target.setdefault(key, value)
    # Forced, even if the user's shell says otherwise.
    for key, value in FORCED_ENV.items():
        target[key] = value


def init_entry_point(verbose: bool = False) -> None:
    setup_utf8_console()
    apply_privacy_env()
    configure_logging(logging.INFO if verbose else logging.WARNING)
