"""``data-audit``: statistics about normalized records. Never outputs record content (SPEC §6)."""

from __future__ import annotations

import json
import math
from collections import Counter

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.dataset.builder import BuildWarning, TokenStats
from fliptuned.dataset.grouping import assign_groups
from fliptuned.dataset.split import PARTITIONS
from fliptuned.errors import NotAvailableError
from fliptuned.records.schema import ExtractionMethod, Record, RecordType
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.services.datasets import locate_dataset, read_index, read_manifest
from fliptuned.services.ingest import RECORDS_FILE, REPORT_FILE, source_dir
from fliptuned.services.review import load_reviewed, open_registry

_FORMAT_OF = {
    ExtractionMethod.CSV_COLUMN_MAPPING: "csv",
    ExtractionMethod.XLSX_COLUMN_MAPPING: "xlsx",
    ExtractionMethod.MARKDOWN_DELIMITERS: "md",
    ExtractionMethod.MARKDOWN_BLOCK: "md",
    ExtractionMethod.TEXT_DELIMITERS: "txt",
    ExtractionMethod.TEXT_BLOCK: "txt",
}


class AuditRequest(BaseModel):
    home: str | None = None
    source: str | None = Field(default=None, description="Data source ID or name. Empty = all.")


class LengthStats(BaseModel):
    p50: int
    p90: int
    p99: int
    max: int


class GroupStats(BaseModel):
    grouped_records: int = Field(description="qa_pair records in a group of two or more.")
    distinct_groups: int
    largest_group_size: int
    largest_group_share: float = Field(
        description="Share of all qa_pair records in the largest group."
    )
    by_group_source: dict[str, int]


class AuditResult(BaseModel):
    sources: list[str] = Field(description="Data source IDs included.")
    records_total: int
    by_type: dict[str, int]
    by_format: dict[str, int]
    by_origin: dict[str, int]
    by_status: dict[str, int] = Field(description="Effective status (review decisions applied).")
    by_category: dict[str, int]
    groups: GroupStats = Field(
        description="Groups as a dataset build would assign them (SPEC §9.2)."
    )
    question_chars: LengthStats | None
    answer_chars: LengthStats | None
    question_words: LengthStats | None
    answer_words: LengthStats | None
    length_note: str
    quality_flags: dict[str, int]
    privacy_findings_by_type: dict[str, int]
    records_with_privacy_findings: int
    exclusions: dict[str, int]


def run_audit(request: AuditRequest) -> AuditResult:
    paths = AppPaths(resolve_app_home(request.home).path)
    if not paths.db.exists():
        raise NotAvailableError(
            "Nothing has been ingested yet.", hint="Run `fliptuned ingest --config <file>` first."
        )
    migrate(paths.db)
    with connect(paths.db) as conn:
        rows = repositories.list_data_sources(conn)
    wanted = request.source.strip().lower() if request.source else None
    ids = [
        r["id"] for r in rows if wanted is None or wanted in (r["id"].lower(), r["name"].lower())
    ]
    ids = [i for i in ids if (source_dir(paths, i) / RECORDS_FILE).exists()]
    if not ids:
        raise NotAvailableError(
            "No ingested data source matches." if wanted else "Nothing has been ingested yet.",
            hint="Run `fliptuned ingest --config <file>` first, or check the --source name.",
        )
    with connect(paths.db) as conn:
        items = [(r.source_id, r.record) for r in load_reviewed(conn, paths, ids)]
    reports = []
    for source_id in ids:
        report_path = source_dir(paths, source_id) / REPORT_FILE
        if report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    return audit_records(ids, items, reports)


def audit_records(
    ids: list[str], items: list[tuple[str, Record]], reports: list[dict]
) -> AuditResult:
    records = [r for _, r in items]
    qa_items = [(s, r) for s, r in items if r.record_type is RecordType.QA_PAIR]
    qa = [r for _, r in qa_items]
    assignments = assign_groups(qa_items)
    sizes = Counter(a.key for a in assignments)
    largest = max(sizes.values(), default=0)

    exclusions: Counter[str] = Counter()
    for report in reports:
        outcome = report.get("outcome", {})
        exclusions.update(e["reason"] for e in outcome.get("exclusions", []))
        counts = report.get("counts", {})
        for key in (
            "blank_rows_skipped",
            "unsupported_files",
            "skipped_files",
            "rows_not_ingested",
        ):
            if counts.get(key):
                exclusions[key] += counts[key]
        if counts.get("files_with_errors"):
            exclusions["files_with_errors"] += counts["files_with_errors"]

    return AuditResult(
        sources=ids,
        records_total=len(records),
        by_type=_count(r.record_type.value for r in records),
        by_format=_count(_FORMAT_OF[r.extraction_method] for r in records),
        by_origin=_count(r.origin for r in records),
        by_status=_count(r.status.value for r in records),
        by_category=_count(r.category or "(none)" for r in qa),
        groups=GroupStats(
            grouped_records=sum(n for n in sizes.values() if n > 1),
            distinct_groups=len(sizes),
            largest_group_size=largest,
            largest_group_share=round(largest / len(qa), 4) if qa else 0.0,
            by_group_source=_count(a.source.value for a in assignments),
        ),
        question_chars=_lengths([len(r.question or "") for r in qa]),
        answer_chars=_lengths([len(r.answer or "") for r in qa]),
        question_words=_lengths([len((r.question or "").split()) for r in qa]),
        answer_words=_lengths([len((r.answer or "").split()) for r in qa]),
        length_note=(
            "Character and word counts of qa_pair records. Token counts need the model's tokenizer: "
            "see `data-audit --processed <dataset_id>`."
        ),
        quality_flags=_count(f.value for r in records for f in r.quality_flags),
        privacy_findings_by_type=_count(f.type for r in records for f in r.privacy_flags),
        records_with_privacy_findings=sum(1 for r in records if r.privacy_flags),
        exclusions=dict(sorted(exclusions.items())),
    )


def _count(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _lengths(values: list[int]) -> LengthStats | None:
    if not values:
        return None
    ordered = sorted(values)

    def rank(p: float) -> int:  # nearest-rank percentile
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return LengthStats(p50=rank(0.5), p90=rank(0.9), p99=rank(0.99), max=ordered[-1])


# --------------------------------------------------------------------------- processed


class ProcessedAuditRequest(BaseModel):
    home: str | None = None
    dataset: str = Field(description="Dataset ID or name.")


class PartitionStats(BaseModel):
    examples: int
    groups: int
    largest_group_size: int
    by_category: dict[str, int]
    by_group_source: dict[str, int]
    tokens: TokenStats | None
    truncated: int


class ProcessedAuditResult(BaseModel):
    dataset_id: str
    name: str
    model: str = Field(description="Catalog ID whose tokenizer counted the tokens.")
    max_seq_length: int
    input_records: int
    included: int
    split_target: dict[str, float]
    split_actual: dict[str, float]
    partitions: dict[str, PartitionStats]
    largest_group_size: int
    largest_group_share: float = Field(description="Share of included examples.")
    excluded: dict[str, int]
    duplicates: int
    warnings: list[BuildWarning]


def run_processed_audit(request: ProcessedAuditRequest) -> ProcessedAuditResult:
    paths = open_registry(request.home)
    dataset_id, folder, _ = locate_dataset(paths, request.dataset)
    manifest = read_manifest(folder)
    index = read_index(folder)
    partitions = {}
    for p in PARTITIONS:
        rows = [r for r in index if r.partition == p]
        sizes = Counter(r.group_key for r in rows)
        partitions[p] = PartitionStats(
            examples=len(rows),
            groups=len(sizes),
            largest_group_size=max(sizes.values(), default=0),
            by_category=_count(r.category or "(none)" for r in rows),
            by_group_source=_count(r.group_source for r in rows),
            tokens=manifest.tokens.get(p),
            truncated=sum(1 for r in rows if r.truncated),
        )
    c = manifest.counts
    return ProcessedAuditResult(
        dataset_id=dataset_id,
        name=manifest.name,
        model=str(manifest.model.get("catalog_id", "")),
        max_seq_length=manifest.max_seq_length,
        input_records=c.input_records,
        included=c.included,
        split_target=manifest.split.target,
        split_actual=manifest.split.actual,
        partitions=partitions,
        largest_group_size=manifest.groups.largest_size,
        largest_group_share=manifest.groups.largest_share,
        excluded=c.excluded,
        duplicates=c.duplicates,
        warnings=manifest.warnings,
    )
