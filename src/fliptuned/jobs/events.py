"""``events.jsonl``: the single progress source of a job (SPEC §13.3).

Every line is one JSON object with ``seq``, ``time`` and ``type`` (``status``, ``progress``,
``log`` or ``result``) plus type-specific fields. The file is opened and closed for every write
and read, so no handle stays open (Windows). Readers track a byte offset, which the API also uses
as the SSE event ID; a partly written last line is left for the next read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fliptuned.registry.db import utc_now


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = _count_lines(path)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        event = {"seq": self.seq, "time": utc_now(), "type": event_type, **fields}
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Complete events after byte ``offset``, and the offset to continue from."""
    if not path.exists():
        return [], offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    events = [json.loads(line) for line in data[:end].split(b"\n") if line.strip()]
    return events, offset + end + 1
