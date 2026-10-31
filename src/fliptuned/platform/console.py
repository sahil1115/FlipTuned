"""Console encoding setup."""

from __future__ import annotations

import contextlib
import sys


def setup_utf8_console() -> None:
    """On Windows, switch stdout/stderr to UTF-8 so non-ASCII output never crashes."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")
