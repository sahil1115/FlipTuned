"""File discovery, format detection and decoding."""

from __future__ import annotations

import codecs
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fliptuned.config.models import SourceFormat

EXTENSIONS = {
    ".csv": SourceFormat.CSV,
    ".xlsx": SourceFormat.XLSX,
    ".md": SourceFormat.MARKDOWN,
    ".markdown": SourceFormat.MARKDOWN,
    ".txt": SourceFormat.TXT,
}
_CHUNK = 1 << 20


def detect_format(path: Path) -> SourceFormat | None:
    return EXTENSIONS.get(path.suffix.lower())


def is_skipped_name(name: str) -> bool:
    """Hidden files and Office lock files (``~$name.xlsx``) are never data."""
    return name.startswith((".", "~$"))


def walk_files(root: Path) -> Iterator[tuple[Path, bool]]:
    """Yield ``(path, skipped)`` for every file under ``root`` in a stable, sorted order.

    Symlinked directories are not followed. Hidden directories are skipped entirely.
    """
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            yield Path(current) / name, is_skipped_name(name)


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


@dataclass(frozen=True)
class DecodedText:
    text: str
    valid: bool  # False if the bytes were not valid in the declared encoding


def decode_bytes(data: bytes, encoding: str) -> DecodedText:
    codec = "utf-8-sig" if codecs.lookup(encoding).name == "utf-8" else encoding
    try:
        return DecodedText(data.decode(codec), True)
    except UnicodeDecodeError:
        return DecodedText(data.decode(codec, errors="replace"), False)


def first_invalid_utf8_offset(path: Path) -> int | None:
    """Byte offset of the first invalid UTF-8 sequence, read in chunks; ``None`` if valid."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    offset = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                return offset + exc.start
            offset += len(chunk)
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            return offset + exc.start
    return None
