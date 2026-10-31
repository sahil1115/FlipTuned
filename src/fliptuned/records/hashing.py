"""Deterministic IDs and hashes.

Stored text is never modified. Line endings are normalized only inside these functions,
so CRLF and LF copies of the same content hash identically.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_SEP = "\x1f"


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_for_dedup(text: str) -> str:
    """Whitespace- and line-ending-insensitive form used to detect duplicates."""
    return _WS.sub(" ", normalize_line_endings(text)).strip()


def sha256_hex(*parts: str) -> str:
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


def content_hash(*texts: str | None) -> str:
    return sha256_hex(*(normalize_line_endings(t or "") for t in texts))


def dedup_key(question: str | None, answer: str | None) -> str:
    return sha256_hex(normalize_for_dedup(question or ""), normalize_for_dedup(answer or ""))


def record_id(prefix: str, source_id: str, source_file: str, locator: str) -> str:
    """Stable across runs and OSes: depends only on the source, relative path and location."""
    return f"{prefix}_{sha256_hex(source_id, source_file, locator)[:16]}"
