"""Logging conventions.

- Log events, counts, IDs, paths and timings, never record content (questions, answers,
  chat text) or secrets. Content debug logging needs the explicit ``debug_log_content`` opt-in.
- Library code uses ``get_logger(__name__)`` and never configures handlers.
- Only entry points (CLI, server, workers) call :func:`configure_logging`.
"""

from __future__ import annotations

import logging
from pathlib import Path

ROOT_LOGGER = "fliptuned"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(ROOT_LOGGER) else f"{ROOT_LOGGER}.{name}")


def configure_logging(level: int = logging.WARNING, log_file: Path | None = None) -> None:
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(min(level, logging.INFO) if log_file else level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    root.propagate = False
