"""``scan``: inspect a raw folder without parsing record content (SPEC §6).

Reads only file metadata, CSV/XLSX header rows, and bytes for the UTF-8 check. The result
includes file and column names for the local UI; the CLI renders counts only.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.config.models import SourceFormat
from fliptuned.errors import NotAvailableError
from fliptuned.ingestion.files import (
    detect_format,
    first_invalid_utf8_offset,
    relative_posix,
    walk_files,
)
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

# Normalized header name → mapping target. Order matters: first match wins per target.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "question": ("question", "q", "prompt", "query", "ask", "title", "questiontext"),
    "answer": ("answer", "a", "response", "reply", "solution", "completion", "answertext"),
    "id": ("id", "qaid", "recordid", "key", "uid"),
    "tags": ("tags", "tag", "labels", "keywords"),
    "category": ("category", "topic", "area", "section", "type"),
    "status": ("status", "approved", "state", "reviewstatus", "approval"),
    "group": ("group", "groupid", "cluster", "groupname"),
    "thread": (
        "thread",
        "threadid",
        "ticket",
        "ticketid",
        "conversation",
        "conversationid",
        "case",
    ),
}
_NON_ALNUM = re.compile(r"[^a-z0-9]")


class ScanRequest(BaseModel):
    path: str = Field(description="Folder to scan.")
    name: str | None = Field(
        default=None, description="Data source name. Defaults to the folder name."
    )
    home: str | None = None


class TypeStats(BaseModel):
    count: int = 0
    bytes: int = 0


class TabularFile(BaseModel):
    path: str
    sheet: str | None = None
    headers: list[str]
    suggested_mapping: dict[str, str | None]
    mapping_complete: bool = Field(description="Both question and answer columns were recognized.")


class EncodingProblem(BaseModel):
    path: str
    byte_offset: int


class ScanResult(BaseModel):
    data_source_id: str
    name: str
    root: str
    files_total: int
    supported_files: int
    unsupported_files: list[str]
    skipped_files: int = Field(description="Hidden files and Office lock files.")
    by_type: dict[str, TypeStats]
    tabular: list[TabularFile]
    text_files: list[str] = Field(
        default_factory=list, description="Markdown and TXT files (paths relative to root)."
    )
    unreadable_files: list[str]
    encoding_problems: list[EncodingProblem]


def suggest_mapping(headers: list[str]) -> dict[str, str | None]:
    normalized = {h: _NON_ALNUM.sub("", h.lower()) for h in headers if h}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for target, names in SYNONYMS.items():
        mapping[target] = None
        for name in names:
            match = next((h for h, n in normalized.items() if n == name and h not in used), None)
            if match:
                mapping[target] = match
                used.add(match)
                break
    return mapping


def scan_folder(request: ScanRequest) -> ScanResult:
    root = Path(request.path).expanduser().resolve()
    if not root.is_dir():
        raise NotAvailableError(f"Folder not found: {root}", hint="Check the folder path.")
    name = request.name or root.name
    by_type: dict[str, TypeStats] = {}
    unsupported: list[str] = []
    tabular: list[TabularFile] = []
    text_files: list[str] = []
    unreadable: list[str] = []
    problems: list[EncodingProblem] = []
    total = supported = skipped = 0

    for path, skip in walk_files(root):
        rel = relative_posix(path, root)
        if skip:
            skipped += 1
            continue
        total += 1
        ext = path.suffix.lower() or "(none)"
        stats = by_type.setdefault(ext, TypeStats())
        stats.count += 1
        stats.bytes += path.stat().st_size
        fmt = detect_format(path)
        if fmt is None:
            unsupported.append(rel)
            continue
        supported += 1
        try:
            if fmt is not SourceFormat.XLSX:
                offset = first_invalid_utf8_offset(path)
                if offset is not None:
                    problems.append(EncodingProblem(path=rel, byte_offset=offset))
            if fmt is SourceFormat.CSV:
                tabular.append(_csv_headers(path, rel))
            elif fmt is SourceFormat.XLSX:
                tabular.extend(_xlsx_headers(path, rel))
            else:
                text_files.append(rel)
        except Exception:  # noqa: BLE001 - one unreadable file must not stop the scan
            unreadable.append(rel)

    result = ScanResult(
        data_source_id=repositories.data_source_id(name),
        name=name,
        root=str(root),
        files_total=total,
        supported_files=supported,
        unsupported_files=unsupported,
        skipped_files=skipped,
        by_type=dict(sorted(by_type.items())),
        tabular=tabular,
        text_files=text_files,
        unreadable_files=unreadable,
        encoding_problems=problems,
    )
    paths = AppPaths(resolve_app_home(request.home).path)
    paths.ensure()
    migrate(paths.db)
    with connect(paths.db) as conn:
        repositories.upsert_data_source(
            conn,
            source_id=result.data_source_id,
            name=name,
            root_path=str(root),
            status="scanned",
            stats=scan_counts(result),
        )
    log.info("scanned data source %s: %d files", result.data_source_id, total)
    return result


def scan_counts(result: ScanResult) -> dict[str, int]:
    """Counts only — safe for CLI output and logs."""
    return {
        "files_total": result.files_total,
        "supported_files": result.supported_files,
        "unsupported_files": len(result.unsupported_files),
        "skipped_files": result.skipped_files,
        "tabular_files": len({t.path for t in result.tabular}),
        "tabular_with_complete_mapping": sum(1 for t in result.tabular if t.mapping_complete),
        "unreadable_files": len(result.unreadable_files),
        "encoding_problems": len(result.encoding_problems),
    }


def _csv_headers(path: Path, rel: str) -> TabularFile:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle), [])
    return _tabular(rel, None, [h.strip() for h in header])


def _xlsx_headers(path: Path, rel: str) -> list[TabularFile]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        files = []
        for sheet in workbook.sheetnames:
            first = next(workbook[sheet].iter_rows(values_only=True, max_row=1), None)
            headers = [str(v).strip() if v is not None else "" for v in (first or [])]
            files.append(_tabular(rel, sheet, headers))
        return files
    finally:
        workbook.close()


def _tabular(rel: str, sheet: str | None, headers: list[str]) -> TabularFile:
    mapping = suggest_mapping(headers)
    return TabularFile(
        path=rel,
        sheet=sheet,
        headers=headers,
        suggested_mapping=mapping,
        mapping_complete=bool(mapping["question"] and mapping["answer"]),
    )
