"""Quality flags (SPEC §9.1). Flags never change stored text; some quarantine the record."""

from __future__ import annotations

import re

from fliptuned.config.models import QualityThresholds, RecordStatus
from fliptuned.records.hashing import dedup_key
from fliptuned.records.schema import QUARANTINE_FLAGS, QualityFlag, Record, RecordType

_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)", re.MULTILINE)


def has_broken_fence(text: str) -> bool:
    """True if a code fence is opened and never closed."""
    open_marker: str | None = None
    for match in _FENCE.finditer(text):
        marker = match.group(1)
        if open_marker is None:
            open_marker = marker
        elif marker == open_marker:
            open_marker = None
    return open_marker is not None


def _looks_swapped(question: str, answer: str) -> bool:
    q, a = question.strip(), answer.strip()
    return bool(q and a) and a.endswith("?") and not q.endswith("?")


def record_flags(record: Record, limits: QualityThresholds) -> list[QualityFlag]:
    """Flags that depend on one record only."""
    if record.record_type is not RecordType.QA_PAIR:
        return []
    flags: list[QualityFlag] = []
    q = record.question or ""
    a = record.answer or ""
    if not q.strip():
        flags.append(QualityFlag.EMPTY_QUESTION)
    if not a.strip():
        flags.append(QualityFlag.EMPTY_ANSWER)
    if (q.strip() and len(q.strip()) < limits.min_question_chars) or (
        a.strip() and len(a.strip()) < limits.min_answer_chars
    ):
        flags.append(QualityFlag.TOO_SHORT)
    if len(q) > limits.max_question_chars or len(a) > limits.max_answer_chars:
        flags.append(QualityFlag.TOO_LONG)
    if has_broken_fence(q) or has_broken_fence(a):
        flags.append(QualityFlag.BROKEN_CODE_FENCE)
    if _looks_swapped(q, a):
        flags.append(QualityFlag.LIKELY_QA_SWAP)
    return flags


def apply_cross_record_flags(records: list[Record]) -> None:
    """Duplicate IDs (from the user's id column) and exact duplicates, in input order."""
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for record in records:
        if record.record_type is not RecordType.QA_PAIR:
            continue
        if record.external_id:
            if record.external_id in seen_ids:
                _add(record, QualityFlag.DUPLICATE_ID)
            seen_ids.add(record.external_id)
        key = dedup_key(record.question, record.answer)
        if key in seen_content:
            _add(record, QualityFlag.EXACT_DUPLICATE)
        seen_content.add(key)


def _add(record: Record, flag: QualityFlag) -> None:
    if flag not in record.quality_flags:
        record.quality_flags.append(flag)


def quarantine_if_needed(record: Record) -> str | None:
    """Quarantine records with blocking flags. Returns the reason, if any."""
    blocking = [f for f in record.quality_flags if f in QUARANTINE_FLAGS]
    if blocking and record.status is not RecordStatus.QUARANTINED:
        record.status = RecordStatus.QUARANTINED
        return blocking[0].value
    return None
