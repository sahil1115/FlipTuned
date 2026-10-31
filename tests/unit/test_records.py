from __future__ import annotations

from fliptuned.config.models import QualityThresholds, RecordStatus
from fliptuned.records import quality
from fliptuned.records.hashing import content_hash, dedup_key, normalize_for_dedup, record_id
from fliptuned.records.schema import (
    ExtractionMethod,
    QualityFlag,
    Record,
    RecordType,
    SourceLocator,
)


def _record(question: str, answer: str, external_id: str | None = None, row: int = 2) -> Record:
    return Record(
        id=record_id("qa", "src", "f.csv", f"row={row}"),
        record_type=RecordType.QA_PAIR,
        source_file="f.csv",
        source_locator=SourceLocator(row=row),
        question=question,
        answer=answer,
        external_id=external_id,
        status=RecordStatus.APPROVED,
        extraction_method=ExtractionMethod.CSV_COLUMN_MAPPING,
        content_hash=content_hash(question, answer),
    )


def test_crlf_and_lf_hash_identically_but_text_is_untouched() -> None:
    lf, crlf = "line one\nline two\n", "line one\r\nline two\r\n"
    assert content_hash("q", lf) == content_hash("q", crlf)
    assert content_hash("q", lf) != content_hash("q", lf + "x")
    record = _record("q?", crlf)
    assert record.answer == crlf


def test_dedup_normalization_ignores_whitespace_only() -> None:
    assert normalize_for_dedup("  a \t b\r\n c ") == "a b c"
    assert dedup_key("How  do I?", "x\r\ny") == dedup_key("How do I?", "x y")
    assert dedup_key("How do I?", "x") != dedup_key("How do you?", "x")


def test_record_ids_are_deterministic_and_location_specific() -> None:
    assert record_id("qa", "s", "a/b.csv", "row=2") == record_id("qa", "s", "a/b.csv", "row=2")
    assert record_id("qa", "s", "a/b.csv", "row=2") != record_id("qa", "s", "a/b.csv", "row=3")
    assert record_id("qa", "s", "a/b.csv", "row=2").startswith("qa_")


def test_broken_fence_detection() -> None:
    assert not quality.has_broken_fence("```python\nx = 1\n```\n")
    assert quality.has_broken_fence("```python\nx = 1\n")
    assert not quality.has_broken_fence("~~~\n```\n~~~\n")  # ``` inside ~~~ is text
    assert not quality.has_broken_fence("inline ``` mention mid-line")


def test_record_flags() -> None:
    limits = QualityThresholds()
    flags = quality.record_flags(_record("   ", ""), limits)
    assert {QualityFlag.EMPTY_QUESTION, QualityFlag.EMPTY_ANSWER} <= set(flags)
    assert QualityFlag.TOO_SHORT in quality.record_flags(_record("Ok?", "y"), limits)
    assert QualityFlag.TOO_LONG in quality.record_flags(_record("What is it?", "x" * 20001), limits)
    swap = quality.record_flags(
        _record("Run it before deploying.", "When should I run it?"), limits
    )
    assert QualityFlag.LIKELY_QA_SWAP in swap
    assert quality.record_flags(_record("How do I do it?", "Like this."), limits) == []


def test_cross_record_flags_mark_later_occurrences() -> None:
    a = _record("How do I name things?", "Clearly.", "Q1", 2)
    b = _record("How do I  name things?", "Clearly.", "Q2", 3)
    c = _record("Different question?", "Different.", "Q1", 4)
    quality.apply_cross_record_flags([a, b, c])
    assert a.quality_flags == []
    assert b.quality_flags == [QualityFlag.EXACT_DUPLICATE]
    assert c.quality_flags == [QualityFlag.DUPLICATE_ID]


def test_blocking_flags_quarantine() -> None:
    record = _record("Which linter?", " ")
    record.quality_flags = quality.record_flags(record, QualityThresholds())
    assert quality.quarantine_if_needed(record) == "empty_answer"
    assert record.status is RecordStatus.QUARANTINED
