"""``ingest``: read configured sources into normalized records (synchronous; a job from Phase 5).

Writes ``<app_home>/sources/<source_id>/records.jsonl`` and ``ingest_report.json`` and records
the data source in the registry. Reports and logs carry counts, locations and redacted
findings only — never record text or secret values.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.config.loader import load_config
from fliptuned.config.models import IngestConfig
from fliptuned.errors import ExitCode, FlipTunedError
from fliptuned.ingestion.pipeline import IngestOutcome, ingest_sources
from fliptuned.records.schema import Record
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate, utc_now
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

RECORDS_FILE = "records.jsonl"
REPORT_FILE = "ingest_report.json"


class PrivacyStopError(FlipTunedError):
    exit_code = ExitCode.CHECK_FAILED


class IngestRequest(BaseModel):
    config_path: str = Field(description="Ingest config file.")
    home: str | None = None
    overwrite: bool = Field(default=False, description="Replace records from an earlier ingest.")


class IngestCounts(BaseModel):
    records_total: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    quality_flags: dict[str, int]
    privacy_findings: int
    excluded: int
    files_read: int
    files_with_errors: int
    unsupported_files: int
    skipped_files: int
    blank_rows_skipped: int
    rows_not_ingested: int = Field(
        description="Rows read from files whose required columns were missing."
    )


class IngestResult(BaseModel):
    data_source_id: str
    name: str
    records_path: str | None
    report_path: str
    counts: IngestCounts
    outcome: IngestOutcome


def source_dir(paths: AppPaths, source_id: str) -> Path:
    return paths.sources / source_id


def run_ingest(request: IngestRequest) -> IngestResult:
    config_path = Path(request.config_path).expanduser().resolve()
    config = load_config(config_path, IngestConfig)
    paths = AppPaths(resolve_app_home(request.home).path)
    source_id = repositories.data_source_id(config.name)
    out_dir = source_dir(paths, source_id)
    records_path = out_dir / RECORDS_FILE
    if records_path.exists() and not request.overwrite:
        raise FlipTunedError(
            f"Data source '{config.name}' was already ingested.",
            hint="Re-run with --overwrite to replace its records (review decisions are kept).",
        )

    run = ingest_sources(config, config_path.parent, source_id)
    outcome = run.outcome
    assert outcome is not None
    counts = _counts(run.records, outcome)
    paths.ensure()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_FILE
    _write_json(
        report_path,
        {
            "data_source_id": source_id,
            "name": config.name,
            "created_at": utc_now(),
            "counts": counts.model_dump(mode="json"),
            "outcome": outcome.model_dump(mode="json"),
        },
    )

    if outcome.stopped_by_privacy:
        log.warning("ingest stopped by privacy policy: %d findings", len(outcome.findings))
        raise PrivacyStopError(
            f"Ingestion stopped: {len(outcome.findings)} likely secret(s) found and the privacy policy is 'stop'.",
            hint="Remove the secrets from the source files, or set privacy.policy to 'exclude'.",
            details="\n".join(
                f"- {f.source_file} ({f.location}, {f.field}): {f.type} {f.redacted}"
                for f in outcome.findings
            ),
        )

    _write_records(records_path, run.records, debug_content=config.debug_log_content)
    migrate(paths.db)
    with connect(paths.db) as conn:
        repositories.upsert_data_source(
            conn,
            source_id=source_id,
            name=config.name,
            root_path=outcome.root,
            status="ingested",
            stats=counts.model_dump(mode="json"),
            config=config.model_dump(mode="json"),
        )
    log.info("ingested data source %s: %d records", source_id, counts.records_total)
    return IngestResult(
        data_source_id=source_id,
        name=config.name,
        records_path=str(records_path),
        report_path=str(report_path),
        counts=counts,
        outcome=outcome,
    )


def _counts(records: list[Record], outcome: IngestOutcome) -> IngestCounts:
    flags = Counter(f.value for r in records for f in r.quality_flags)
    return IngestCounts(
        records_total=len(records),
        by_type=dict(sorted(Counter(r.record_type.value for r in records).items())),
        by_status=dict(sorted(Counter(r.status.value for r in records).items())),
        quality_flags=dict(sorted(flags.items())),
        privacy_findings=len(outcome.findings),
        excluded=len(outcome.exclusions),
        files_read=len(outcome.files),
        files_with_errors=sum(1 for f in outcome.files if f.errors),
        unsupported_files=len(outcome.unsupported_files),
        skipped_files=outcome.skipped_files,
        blank_rows_skipped=sum(f.blank_rows_skipped for f in outcome.files),
        rows_not_ingested=sum(
            f.rows_read - f.records for f in outcome.files if f.format in ("csv", "xlsx")
        ),
    )


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _write_records(path: Path, records: list[Record], *, debug_content: bool) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            if debug_content:  # explicit opt-in only (IngestConfig.debug_log_content)
                log.debug("record %s question=%r", record.id, record.question)
    os.replace(tmp, path)


def load_records(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as handle:
        return [Record.model_validate_json(line) for line in handle if line.strip()]
