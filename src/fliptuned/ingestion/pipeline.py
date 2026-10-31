"""Turn configured sources into normalized records plus a report (no registry, no output files)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned.config.models import (
    ColumnMapping,
    IngestConfig,
    PrivacyPolicy,
    RecordStatus,
    SourceFormat,
    SourceSpec,
)
from fliptuned.errors import ConfigError
from fliptuned.ingestion.files import decode_bytes, detect_format, relative_posix, walk_files
from fliptuned.ingestion.tabular import TableRead, read_csv, read_xlsx
from fliptuned.ingestion.textqa import parse_text_qa
from fliptuned.privacy.scanner import SecretScanner
from fliptuned.records import quality
from fliptuned.records.hashing import content_hash, record_id
from fliptuned.records.schema import (
    ExtractionMethod,
    GroupSource,
    QualityFlag,
    Record,
    RecordType,
    SourceLocator,
)

STATUS_WORDS = {
    "approved": RecordStatus.APPROVED,
    "approve": RecordStatus.APPROVED,
    "yes": RecordStatus.APPROVED,
    "y": RecordStatus.APPROVED,
    "true": RecordStatus.APPROVED,
    "1": RecordStatus.APPROVED,
    "rejected": RecordStatus.REJECTED,
    "reject": RecordStatus.REJECTED,
    "no": RecordStatus.REJECTED,
    "n": RecordStatus.REJECTED,
    "false": RecordStatus.REJECTED,
    "0": RecordStatus.REJECTED,
    "pending": RecordStatus.PENDING_REVIEW,
    "pending_review": RecordStatus.PENDING_REVIEW,
    "review": RecordStatus.PENDING_REVIEW,
}
_TAG_SPLIT = re.compile(r"[,;]")


class FileReport(BaseModel):
    source_file: str
    format: str | None
    records: int = 0
    rows_read: int = 0
    blank_rows_skipped: int = 0
    invalid_encoding: bool = False
    errors: list[str] = Field(default_factory=list)


class Exclusion(BaseModel):
    record_id: str
    source_file: str
    location: str
    reason: str


class FindingReport(BaseModel):
    record_id: str
    source_file: str
    location: str
    field: str
    type: str
    redacted: str


class IngestOutcome(BaseModel):
    root: str
    files: list[FileReport]
    unsupported_files: list[str]
    skipped_files: int
    exclusions: list[Exclusion]
    findings: list[FindingReport]
    privacy_policy: PrivacyPolicy
    stopped_by_privacy: bool = False


@dataclass
class IngestRun:
    records: list[Record] = field(default_factory=list)
    outcome: IngestOutcome | None = None


def resolve_sources(
    config: IngestConfig, config_dir: Path
) -> tuple[Path, list[tuple[SourceSpec, Path]]]:
    """Absolute paths for each source, and the data source root (their common parent)."""
    resolved: list[tuple[SourceSpec, Path]] = []
    for spec in config.sources:
        path = Path(spec.path).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        path = Path(os.path.normpath(path))
        if not path.exists():
            raise ConfigError(
                f"Source not found: {path}",
                hint="Fix the 'path' of this source in the ingest config.",
            )
        resolved.append((spec, path))
    bases = [str(p if p.is_dir() else p.parent) for _, p in resolved]
    return Path(os.path.commonpath(bases)), resolved


def ingest_sources(config: IngestConfig, config_dir: Path, source_id: str) -> IngestRun:
    root, resolved = resolve_sources(config, config_dir)
    scanner = SecretScanner(config.privacy.extra_patterns)
    run = IngestRun()
    files: list[FileReport] = []
    unsupported: list[str] = []
    skipped = 0

    for spec, path in resolved:
        candidates = [(path, False)] if path.is_file() else list(walk_files(path))
        for file_path, skip in candidates:
            rel = relative_posix(file_path, root)
            if skip:
                skipped += 1
                continue
            fmt = detect_format(file_path) if spec.format is SourceFormat.AUTO else spec.format
            if fmt is None:
                unsupported.append(rel)
                continue
            report = FileReport(source_file=rel, format=fmt.value)
            files.append(report)
            run.records.extend(_read_file(spec, fmt, file_path, rel, source_id, report))

    for record in run.records:
        record.quality_flags.extend(quality.record_flags(record, config.quality))
        for name in ("question", "answer", "text"):
            record.privacy_flags.extend(scanner.scan(getattr(record, name), name))
    quality.apply_cross_record_flags(run.records)

    exclusions: list[Exclusion] = []
    findings: list[FindingReport] = []
    policy = config.privacy.policy
    for record in run.records:
        location = _location(record.source_locator)
        reason = quality.quarantine_if_needed(record)
        if record.privacy_flags:
            findings.extend(
                FindingReport(
                    record_id=record.id,
                    source_file=record.source_file,
                    location=location,
                    field=f.field,
                    type=f.type,
                    redacted=f.redacted,
                )
                for f in record.privacy_flags
            )
            if policy is PrivacyPolicy.EXCLUDE and record.status is not RecordStatus.QUARANTINED:
                record.status = RecordStatus.QUARANTINED
                reason = reason or "privacy_finding"
        if reason:
            exclusions.append(
                Exclusion(
                    record_id=record.id,
                    source_file=record.source_file,
                    location=location,
                    reason=reason,
                )
            )

    run.outcome = IngestOutcome(
        root=str(root),
        files=files,
        unsupported_files=unsupported,
        skipped_files=skipped,
        exclusions=exclusions,
        findings=findings,
        privacy_policy=policy,
        stopped_by_privacy=policy is PrivacyPolicy.STOP and bool(findings),
    )
    return run


def _location(loc: SourceLocator) -> str:
    parts = []
    if loc.sheet:
        parts.append(f"sheet {loc.sheet}")
    if loc.row is not None:
        parts.append(f"row {loc.row}")
    if loc.line is not None:
        parts.append(f"line {loc.line}")
    return ", ".join(parts) or "file"


def _read_file(
    spec: SourceSpec, fmt: SourceFormat, path: Path, rel: str, source_id: str, report: FileReport
) -> list[Record]:
    if fmt in (SourceFormat.CSV, SourceFormat.XLSX):
        if spec.columns is None:
            report.errors.append(f"{fmt.value} files need a 'columns' mapping in the ingest config")
            return []
        table = (
            read_csv(path, spec.encoding)
            if fmt is SourceFormat.CSV
            else read_xlsx(path, spec.sheet)
        )
        return _table_records(spec, fmt, table, rel, source_id, report)
    return _text_records(spec, fmt, path, rel, source_id, report)


def _table_records(
    spec: SourceSpec,
    fmt: SourceFormat,
    table: TableRead,
    rel: str,
    source_id: str,
    report: FileReport,
) -> list[Record]:
    columns: ColumnMapping = spec.columns  # type: ignore[assignment]
    report.errors.extend(table.errors)
    report.blank_rows_skipped = table.blank_rows
    report.invalid_encoding = not table.valid_encoding
    for sheet, headers in table.headers.items():
        missing = [c for c in (columns.question, columns.answer) if c not in headers]
        optional_missing = [
            c
            for c in (
                columns.id,
                columns.tags,
                columns.category,
                columns.status,
                columns.group,
                columns.thread,
            )
            if c and c not in headers
        ]
        if missing or optional_missing:
            where = f"sheet '{sheet}': " if sheet else ""
            report.errors.append(
                f"{where}column(s) {', '.join(repr(c) for c in missing + optional_missing)} not found "
                f"(columns: {', '.join(repr(h) for h in headers if h)})"
            )
    bad_sheets = {
        sheet
        for sheet, headers in table.headers.items()
        if columns.question not in headers or columns.answer not in headers
    }
    method = (
        ExtractionMethod.CSV_COLUMN_MAPPING
        if fmt is SourceFormat.CSV
        else ExtractionMethod.XLSX_COLUMN_MAPPING
    )
    records = []
    for row in table.rows:
        report.rows_read += 1
        if row.sheet in bad_sheets:
            continue
        values = row.values
        question = values.get(columns.question, "")
        answer = values.get(columns.answer, "")
        flags: list[QualityFlag] = []
        status = spec.default_status
        if columns.status and values.get(columns.status, "").strip():
            word = values[columns.status].strip().lower()
            if word in STATUS_WORDS:
                status = STATUS_WORDS[word]
            else:
                flags.append(QualityFlag.UNKNOWN_STATUS)
                status = RecordStatus.PENDING_REVIEW
        if row.extra_fields:
            flags.append(QualityFlag.MALFORMED_ROW)
        if not table.valid_encoding:
            flags.append(QualityFlag.INVALID_UTF8)
        group_id, group_source = _group(spec, rel, values, columns)
        locator = SourceLocator(sheet=row.sheet, row=row.row)
        loc_key = f"sheet={row.sheet or ''};row={row.row}"
        records.append(
            Record(
                id=record_id("qa", source_id, rel, loc_key),
                record_type=RecordType.QA_PAIR,
                source_file=rel,
                source_locator=locator,
                question=question,
                answer=answer,
                external_id=(values.get(columns.id, "").strip() or None) if columns.id else None,
                tags=_tags(values.get(columns.tags, "")) if columns.tags else [],
                category=(values.get(columns.category, "").strip() or None)
                if columns.category
                else None,
                status=status,
                extraction_method=method,
                content_hash=content_hash(question, answer),
                group_id=group_id,
                group_source=group_source,
                quality_flags=flags,
            )
        )
    report.records = len(records)
    return records


def _group(
    spec: SourceSpec, rel: str, values: dict[str, str], columns: ColumnMapping
) -> tuple[str | None, GroupSource | None]:
    if columns.group and values.get(columns.group, "").strip():
        return values[columns.group].strip(), GroupSource.EXPLICIT_COLUMN
    if columns.thread and values.get(columns.thread, "").strip():
        return values[columns.thread].strip(), GroupSource.THREAD
    if spec.group_by_source_file:
        return f"file:{rel}", GroupSource.SOURCE_FILE
    return None, None


def _tags(raw: str) -> list[str]:
    return list(dict.fromkeys(t.strip() for t in _TAG_SPLIT.split(raw) if t.strip()))


def _text_records(
    spec: SourceSpec, fmt: SourceFormat, path: Path, rel: str, source_id: str, report: FileReport
) -> list[Record]:
    decoded = decode_bytes(path.read_bytes(), spec.encoding)
    report.invalid_encoding = not decoded.valid
    markdown = fmt is SourceFormat.MARKDOWN
    items = parse_text_qa(
        decoded.text,
        spec.delimiters.question_prefix,
        spec.delimiters.answer_prefix,
        markdown=markdown,
    )
    qa_method = (
        ExtractionMethod.MARKDOWN_DELIMITERS if markdown else ExtractionMethod.TEXT_DELIMITERS
    )
    block_method = ExtractionMethod.MARKDOWN_BLOCK if markdown else ExtractionMethod.TEXT_BLOCK
    group_id, group_source = (
        (f"file:{rel}", GroupSource.SOURCE_FILE) if spec.group_by_source_file else (None, None)
    )
    records = []
    for item in items:
        flags = [] if decoded.valid else [QualityFlag.INVALID_UTF8]
        locator = SourceLocator(line=item.line, heading=item.heading)
        loc_key = f"line={item.line}"
        if item.kind == "qa":
            records.append(
                Record(
                    id=record_id("qa", source_id, rel, loc_key),
                    record_type=RecordType.QA_PAIR,
                    source_file=rel,
                    source_locator=locator,
                    question=item.question,
                    answer=item.answer,
                    status=spec.default_status,
                    extraction_method=qa_method,
                    content_hash=content_hash(item.question, item.answer),
                    group_id=group_id,
                    group_source=group_source,
                    quality_flags=flags,
                )
            )
        else:
            records.append(
                Record(
                    id=record_id("blk", source_id, rel, loc_key),
                    record_type=RecordType.SOURCE_BLOCK,
                    source_file=rel,
                    source_locator=locator,
                    text=item.text,
                    status=spec.default_status,
                    extraction_method=block_method,
                    content_hash=content_hash(item.text),
                    group_id=group_id,
                    group_source=group_source,
                    quality_flags=flags,
                )
            )
    report.records = len(records)
    return records
