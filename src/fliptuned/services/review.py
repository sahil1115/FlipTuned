"""Review: filter records, approve/reject singly or in bulk, CSV export/import (SPEC §9.3).

Decisions live in the registry (``review_decisions``), never in the records files. Each decision
stores the content hash it was made on; if a re-ingest changes the record at that location, the
decision is stale and ignored. Quarantined records and source blocks can't be approved.

Results carry record text for the local UI, with privacy findings redacted. The CLI prints
counts only. The CSV export is written for the user's own spreadsheet; on import only the
``decision`` and ``reason`` columns are read.
"""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.config.models import RecordStatus
from fliptuned.dataset.grouping import question_key
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.ingestion.pipeline import STATUS_WORDS
from fliptuned.privacy.scanner import redact_text
from fliptuned.records.schema import Record, RecordType, SourceLocator
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.services.ingest import RECORDS_FILE, load_records, source_dir
from fliptuned.utils.spreadsheet import spreadsheet_cell as _cell

DECIDABLE = (RecordStatus.APPROVED, RecordStatus.REJECTED, RecordStatus.PENDING_REVIEW)
CSV_COLUMNS = [
    "source_id",
    "record_id",
    "content_hash",
    "current_status",
    "decision",
    "reason",
    "category",
    "tags",
    "quality_flags",
    "privacy_findings",
    "duplicate_cluster",
    "source_file",
    "location",
    "question",
    "answer",
]
_REQUIRED_IMPORT_COLUMNS = ("source_id", "record_id", "content_hash", "decision")
_SHORT_DECISIONS = {
    "a": RecordStatus.APPROVED,
    "r": RecordStatus.REJECTED,
    "p": RecordStatus.PENDING_REVIEW,
}
_MAX_PROBLEMS = 200


# --------------------------------------------------------------------------- loading


class ReviewedRecord(BaseModel):
    source_id: str
    record: Record = Field(description="With the effective status applied.")
    ingested_status: RecordStatus
    decision_status: RecordStatus | None = None
    decision_origin: str | None = None
    decided_at: str | None = None
    decision_stale: bool = False


def resolve_source_ids(conn: sqlite3.Connection, paths: AppPaths, wanted: list[str]) -> list[str]:
    """Ingested data source IDs matching IDs or names (case-insensitive); empty = all."""
    rows = [
        r
        for r in repositories.list_data_sources(conn)
        if (source_dir(paths, r["id"]) / RECORDS_FILE).exists()
    ]
    if not wanted:
        ids = sorted(r["id"] for r in rows)
    else:
        ids = []
        for item in wanted:
            key = item.strip().lower()
            match = [r["id"] for r in rows if key in (r["id"].lower(), r["name"].lower())]
            if not match:
                raise ConfigError(
                    f"No ingested data source is called '{item}'.",
                    hint="Known sources: " + (", ".join(r["name"] for r in rows) or "none") + ".",
                )
            ids.extend(m for m in match if m not in ids)
        ids.sort()
    if not ids:
        raise NotAvailableError(
            "Nothing has been ingested yet.", hint="Run `fliptuned ingest --config <file>` first."
        )
    return ids


def effective(record: Record, decision: sqlite3.Row | None) -> tuple[RecordStatus, bool]:
    """(status, decision_is_stale). Quarantine always wins over a decision."""
    if decision is None or record.status is RecordStatus.QUARANTINED:
        return record.status, False
    if decision["content_hash"] != record.content_hash:
        return record.status, True
    return RecordStatus(decision["status"]), False


def _moved_decisions(
    source_id: str, records: list[Record], decisions: dict[tuple[str, str], sqlite3.Row]
) -> dict[str, sqlite3.Row]:
    """Latest decision per content hash among this source's decisions that no longer match
    the record at their ID (the record is gone or now holds other content). Record IDs come
    from row positions, so these are decisions whose row moved in the file."""
    current = {r.id: r.content_hash for r in records}
    moved: dict[str, sqlite3.Row] = {}
    for (decided_source, record_id), decision in decisions.items():
        content = decision["content_hash"]
        if decided_source != source_id or content is None or current.get(record_id) == content:
            continue
        if content not in moved or decision["id"] > moved[content]["id"]:
            moved[content] = decision
    return moved


def load_reviewed(
    conn: sqlite3.Connection, paths: AppPaths, source_ids: list[str]
) -> list[ReviewedRecord]:
    decisions = repositories.latest_review_decisions(conn, source_ids)
    out: list[ReviewedRecord] = []
    for source_id in source_ids:
        records = load_records(source_dir(paths, source_id) / RECORDS_FILE)
        moved = _moved_decisions(source_id, records, decisions)
        for record in records:
            decision = decisions.get((source_id, record.id))
            if decision is None or decision["content_hash"] != record.content_hash:
                # The row may have moved in the file: reuse the decision made for this content.
                decision = moved.get(record.content_hash, decision)
            status, stale = effective(record, decision)
            out.append(
                ReviewedRecord(
                    source_id=source_id,
                    record=record.model_copy(update={"status": status}),
                    ingested_status=record.status,
                    decision_status=RecordStatus(decision["status"]) if decision else None,
                    decision_origin=decision["origin"] if decision else None,
                    decided_at=decision["created_at"] if decision else None,
                    decision_stale=stale,
                )
            )
    return out


def open_registry(home: str | None) -> AppPaths:
    paths = AppPaths(resolve_app_home(home).path)
    if not paths.db.exists():
        raise NotAvailableError(
            "Nothing has been ingested yet.", hint="Run `fliptuned ingest --config <file>` first."
        )
    migrate(paths.db)
    return paths


# --------------------------------------------------------------------------- listing


class ReviewFilter(BaseModel):
    sources: list[str] = Field(
        default_factory=list, description="Source IDs or names; empty = all."
    )
    statuses: list[RecordStatus] = Field(default_factory=list, description="Empty = any status.")
    flag: str | None = Field(
        default=None, description="A quality flag name, or 'privacy' for any privacy finding."
    )
    category: str | None = None
    duplicate_cluster: str | None = None
    only_duplicates: bool = Field(
        default=False, description="Only records whose question appears more than once."
    )
    include_source_blocks: bool = Field(
        default=False, description="Also list source blocks (they never train)."
    )


class ReviewItem(BaseModel):
    source_id: str
    record_id: str
    record_type: RecordType
    content_hash: str
    status: RecordStatus
    ingested_status: RecordStatus
    decision_status: RecordStatus | None
    decision_origin: str | None
    decided_at: str | None
    decision_stale: bool
    category: str | None
    tags: list[str]
    quality_flags: list[str]
    privacy_findings: list[str] = Field(description="Finding types; values are never included.")
    duplicate_cluster: str | None
    source_file: str
    location: str
    question: str | None = Field(description="Privacy findings redacted.")
    answer: str | None = Field(description="Privacy findings redacted.")
    text: str | None = Field(description="Source blocks only; privacy findings redacted.")


class ListRecordsRequest(BaseModel):
    home: str | None = None
    filter: ReviewFilter = Field(default_factory=ReviewFilter)
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ListRecordsResult(BaseModel):
    total: int
    offset: int
    items: list[ReviewItem]
    by_status: dict[str, int] = Field(description="Effective status counts of matching records.")


def location_label(loc: SourceLocator) -> str:
    parts = []
    if loc.sheet:
        parts.append(f"sheet {loc.sheet}")
    if loc.row is not None:
        parts.append(f"row {loc.row}")
    if loc.line is not None:
        parts.append(f"line {loc.line}")
    return ", ".join(parts) or "file"


def _duplicate_clusters(reviewed: list[ReviewedRecord]) -> dict[tuple[str, str], str]:
    keys = {
        (r.source_id, r.record.id): question_key(r.record)
        for r in reviewed
        if r.record.record_type is RecordType.QA_PAIR
    }
    sizes = Counter(keys.values())
    return {ref: f"dup_{k[:12]}" for ref, k in keys.items() if sizes[k] > 1}


def _matches(item: ReviewItem, f: ReviewFilter) -> bool:
    if item.record_type is not RecordType.QA_PAIR and not f.include_source_blocks:
        return False
    if f.statuses and item.status not in f.statuses:
        return False
    if f.flag == "privacy":
        if not item.privacy_findings:
            return False
    elif f.flag and f.flag not in item.quality_flags:
        return False
    if f.category is not None and (item.category or "") != f.category:
        return False
    if f.duplicate_cluster and item.duplicate_cluster != f.duplicate_cluster:
        return False
    return not (f.only_duplicates and item.duplicate_cluster is None)


def _to_item(r: ReviewedRecord, clusters: dict[tuple[str, str], str]) -> ReviewItem:
    rec = r.record
    return ReviewItem(
        source_id=r.source_id,
        record_id=rec.id,
        record_type=rec.record_type,
        content_hash=rec.content_hash,
        status=rec.status,
        ingested_status=r.ingested_status,
        decision_status=r.decision_status,
        decision_origin=r.decision_origin,
        decided_at=r.decided_at,
        decision_stale=r.decision_stale,
        category=rec.category,
        tags=rec.tags,
        quality_flags=[f.value for f in rec.quality_flags],
        privacy_findings=[f.type for f in rec.privacy_flags],
        duplicate_cluster=clusters.get((r.source_id, rec.id)),
        source_file=rec.source_file,
        location=location_label(rec.source_locator),
        question=redact_text(rec.question, rec.privacy_flags, "question"),
        answer=redact_text(rec.answer, rec.privacy_flags, "answer"),
        text=redact_text(rec.text, rec.privacy_flags, "text"),
    )


def _filtered(paths: AppPaths, f: ReviewFilter) -> list[ReviewItem]:
    with connect(paths.db) as conn:
        ids = resolve_source_ids(conn, paths, f.sources)
        reviewed = load_reviewed(conn, paths, ids)
    clusters = _duplicate_clusters(reviewed)
    items = (_to_item(r, clusters) for r in reviewed)
    return [i for i in items if _matches(i, f)]


def list_records(request: ListRecordsRequest) -> ListRecordsResult:
    paths = open_registry(request.home)
    items = _filtered(paths, request.filter)
    return ListRecordsResult(
        total=len(items),
        offset=request.offset,
        items=items[request.offset : request.offset + request.limit],
        by_status=dict(sorted(Counter(i.status.value for i in items).items())),
    )


# --------------------------------------------------------------------------- deciding


class RecordRef(BaseModel):
    source_id: str
    record_id: str


class SkippedRecord(RecordRef):
    reason: str


class DecideRequest(BaseModel):
    home: str | None = None
    records: list[RecordRef] = Field(min_length=1)
    status: Literal["approved", "rejected", "pending_review"]
    reason: str | None = Field(default=None, max_length=500, description="Optional note.")
    origin: Literal["ui", "cli", "bulk"] = "ui"


class DecideResult(BaseModel):
    applied: int
    unchanged: int
    skipped: dict[str, int]
    skipped_records: list[SkippedRecord]


def _check_decidable(r: ReviewedRecord | None) -> str | None:
    if r is None:
        return "unknown_record"
    if r.record.record_type is not RecordType.QA_PAIR:
        return "not_a_qa_pair"
    if r.record.status is RecordStatus.QUARANTINED:
        return "quarantined"
    return None


def decide(request: DecideRequest) -> DecideResult:
    paths = open_registry(request.home)
    status = RecordStatus(request.status)
    with connect(paths.db) as conn:
        wanted = sorted({ref.source_id for ref in request.records})
        known = {r["id"] for r in repositories.list_data_sources(conn)}
        ids = [s for s in wanted if s in known and (source_dir(paths, s) / RECORDS_FILE).exists()]
        by_ref = {(r.source_id, r.record.id): r for r in load_reviewed(conn, paths, ids)}
        rows, unchanged, skipped = [], 0, []
        for ref in dict.fromkeys((r.source_id, r.record_id) for r in request.records):
            reviewed = by_ref.get(ref)
            problem = _check_decidable(reviewed)
            if problem:
                skipped.append(SkippedRecord(source_id=ref[0], record_id=ref[1], reason=problem))
                continue
            assert reviewed is not None
            if reviewed.record.status is status:
                unchanged += 1
                continue
            rows.append(
                {
                    "source_id": ref[0],
                    "record_id": ref[1],
                    "content_hash": reviewed.record.content_hash,
                    "status": status.value,
                    "reason": request.reason,
                    "origin": request.origin,
                }
            )
        repositories.insert_review_decisions(conn, rows)
    return DecideResult(
        applied=len(rows),
        unchanged=unchanged,
        skipped=dict(sorted(Counter(s.reason for s in skipped).items())),
        skipped_records=skipped,
    )


# --------------------------------------------------------------------------- CSV export/import


class ExportRequest(BaseModel):
    home: str | None = None
    out: str = Field(description="CSV file to write.")
    overwrite: bool = False
    filter: ReviewFilter = Field(default_factory=ReviewFilter)


class ExportResult(BaseModel):
    path: str
    rows: int
    by_status: dict[str, int]


def _write_csv(handle, items: list[ReviewItem]) -> None:
    writer = csv.writer(handle)
    writer.writerow(CSV_COLUMNS)
    for i in items:
        writer.writerow(
            [
                i.source_id,
                i.record_id,
                i.content_hash,
                i.status.value,
                "",
                "",
                _cell(i.category),
                _cell(", ".join(i.tags)),
                " ".join(i.quality_flags),
                " ".join(i.privacy_findings),
                i.duplicate_cluster or "",
                i.source_file,
                i.location,
                _cell(i.question if i.record_type is RecordType.QA_PAIR else i.text),
                _cell(i.answer),
            ]
        )


def _summary(path: str, items: list[ReviewItem]) -> ExportResult:
    return ExportResult(
        path=path,
        rows=len(items),
        by_status=dict(sorted(Counter(i.status.value for i in items).items())),
    )


def export_csv(request: ExportRequest) -> ExportResult:
    out = Path(request.out).expanduser().resolve()
    if out.exists() and not request.overwrite:
        raise FlipTunedError(f"{out.name} already exists.", hint="Re-run with --overwrite.")
    paths = open_registry(request.home)
    items = _filtered(paths, request.filter)
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so spreadsheet apps detect UTF-8; newline="" so embedded line breaks survive.
    with open(out, "w", encoding="utf-8-sig", newline="") as handle:
        _write_csv(handle, items)
    return _summary(str(out), items)


class CsvDownloadRequest(BaseModel):
    home: str | None = None
    filter: ReviewFilter = Field(default_factory=ReviewFilter)


def export_csv_bytes(request: CsvDownloadRequest) -> tuple[bytes, ExportResult]:
    """The review CSV as a download (UTF-8 with BOM), for the UI."""
    import io

    items = _filtered(open_registry(request.home), request.filter)
    buffer = io.StringIO(newline="")
    _write_csv(buffer, items)
    return buffer.getvalue().encode("utf-8-sig"), _summary("review.csv", items)


class ImportRequest(BaseModel):
    home: str | None = None
    path: str = Field(
        description="CSV written by review export, with the decision column filled in."
    )


class ImportProblem(BaseModel):
    row: int = Field(description="Spreadsheet row (header = 1).")
    reason: str


class ImportResult(BaseModel):
    rows: int
    applied: dict[str, int]
    unchanged: int
    blank: int = Field(description="Rows with an empty decision.")
    skipped: dict[str, int]
    problems: list[ImportProblem] = Field(description=f"First {_MAX_PROBLEMS} skipped rows.")


def _parse_decision(word: str) -> RecordStatus | None:
    word = word.strip().lower()
    status = STATUS_WORDS.get(word) or _SHORT_DECISIONS.get(word)
    return status if status in DECIDABLE else None


def _read_csv_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in _REQUIRED_IMPORT_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(
                    f"{path.name} is missing column(s): {', '.join(missing)}.",
                    hint="Import the CSV written by `fliptuned review export`, with 'decision' filled in.",
                )
            rows = list(enumerate(reader, start=2))  # spreadsheet row numbers
    except FileNotFoundError:
        raise ConfigError(f"File not found: {path}", hint="Check the path.") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path.name} is not UTF-8.", hint="Save it as UTF-8 CSV.") from None
    except csv.Error as exc:
        raise ConfigError(f"{path.name} is not a valid CSV file.", details=str(exc)) from None
    return rows


def import_csv(request: ImportRequest) -> ImportResult:
    path = Path(request.path).expanduser().resolve()
    rows = _read_csv_rows(path)
    paths = open_registry(request.home)
    with connect(paths.db) as conn:
        known = {r["id"] for r in repositories.list_data_sources(conn)}
        wanted = sorted({row.get("source_id", "") for _, row in rows} & known)
        ids = [s for s in wanted if (source_dir(paths, s) / RECORDS_FILE).exists()]
        by_ref = {(r.source_id, r.record.id): r for r in load_reviewed(conn, paths, ids)}

        decisions: dict[tuple[str, str], dict] = {}
        problems: list[ImportProblem] = []
        unchanged = blank = 0
        for line, row in rows:
            word = (row.get("decision") or "").strip()
            if not word:
                blank += 1
                continue
            status = _parse_decision(word)
            ref = ((row.get("source_id") or "").strip(), (row.get("record_id") or "").strip())
            reviewed = by_ref.get(ref)
            problem = "invalid_decision" if status is None else _check_decidable(reviewed)
            if (
                problem is None
                and reviewed is not None
                and (row.get("content_hash") or "").strip() != reviewed.record.content_hash
            ):
                problem = "stale_content"
            if problem:
                problems.append(ImportProblem(row=line, reason=problem))
                continue
            assert reviewed is not None and status is not None
            if reviewed.record.status is status:
                unchanged += 1
                decisions.pop(ref, None)  # a later row for the same record wins
                continue
            reason = (row.get("reason") or "").strip()[:500] or None
            decisions[ref] = {
                "source_id": ref[0],
                "record_id": ref[1],
                "content_hash": reviewed.record.content_hash,
                "status": status.value,
                "reason": reason,
                "origin": "csv",
            }
        repositories.insert_review_decisions(conn, list(decisions.values()))
    return ImportResult(
        rows=len(rows),
        applied=dict(sorted(Counter(d["status"] for d in decisions.values()).items())),
        unchanged=unchanged,
        blank=blank,
        skipped=dict(sorted(Counter(p.reason for p in problems).items())),
        problems=problems[:_MAX_PROBLEMS],
    )
