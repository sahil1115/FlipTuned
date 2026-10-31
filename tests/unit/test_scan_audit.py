from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.errors import NotAvailableError
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.audit import AuditRequest, run_audit
from fliptuned.services.ingest import IngestRequest, load_records, run_ingest
from fliptuned.services.scan import ScanRequest, scan_folder, suggest_mapping

REPO = Path(__file__).resolve().parents[2]
SCAN_DIR = REPO / "tests" / "fixtures" / "scan_folder"
SAMPLE = REPO / "configs" / "ingest.sample.yaml"


def test_scan_counts_mapping_and_encoding(tmp_path: Path) -> None:
    result = scan_folder(ScanRequest(path=str(SCAN_DIR), home=str(tmp_path / "home")))
    assert (result.files_total, result.supported_files, result.skipped_files) == (4, 3, 1)
    assert result.unsupported_files == ["manual.pdf"]
    assert set(result.by_type) == {".csv", ".md", ".pdf", ".txt"}
    (faq,) = result.tabular
    assert faq.path == "faq.csv" and faq.mapping_complete
    assert faq.suggested_mapping["question"] == "Prompt"
    assert faq.suggested_mapping["answer"] == "Response"
    assert faq.suggested_mapping["thread"] == "Ticket ID"
    assert [p.path for p in result.encoding_problems] == ["latin1.txt"]
    assert result.encoding_problems[0].byte_offset == 3


def test_scan_registers_data_source(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = scan_folder(ScanRequest(path=str(SCAN_DIR), name="Team FAQ", home=str(home)))
    with connect(home / "fliptuned.db") as conn:
        (row,) = repositories.list_data_sources(conn)
    assert row["id"] == result.data_source_id == repositories.data_source_id("team faq")
    assert row["status"] == "scanned" and json.loads(row["stats_json"])["files_total"] == 4


def test_scan_xlsx_headers(tmp_path: Path) -> None:
    result = scan_folder(
        ScanRequest(path=str(REPO / "tests" / "fixtures"), home=str(tmp_path / "h"))
    )
    sheets = {(t.path, t.sheet): t for t in result.tabular}
    qa = sheets[("qa.xlsx", "QA")]
    assert qa.headers == ["question", "answer", "group"] and qa.mapping_complete
    assert not sheets[("qa.xlsx", "Notes")].mapping_complete


def test_suggest_mapping_uses_each_header_once() -> None:
    mapping = suggest_mapping(["Q", "A", "Type", "Ticket", "Tags"])
    assert mapping["question"] == "Q" and mapping["answer"] == "A"
    assert (
        mapping["category"] == "Type"
        and mapping["thread"] == "Ticket"
        and mapping["tags"] == "Tags"
    )
    assert mapping["id"] is None


def test_cli_scan_prints_counts_only(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["scan", str(SCAN_DIR), "--home", str(tmp_path / "home")])
    assert result.exit_code == 0, result.output
    for name in ("faq", "Prompt", "Response", "Ticket", "manual", "guide", "latin1", "nested"):
        assert name not in result.output
    assert "files total" in result.output


def test_cli_scan_missing_folder(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["scan", str(tmp_path / "nope"), "--home", str(tmp_path)])
    assert result.exit_code == 4


def _windows(text: str, size: int = 24) -> set[str]:
    text = " ".join(text.split())
    return (
        {text[i : i + size] for i in range(0, max(1, len(text) - size + 1), 4)}
        if len(text) >= size
        else {text}
    )


def test_audit_outputs_contain_no_record_text(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ingest = run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(home)))
    records = load_records(Path(ingest.records_path))
    runner = CliRunner()
    outputs = [
        runner.invoke(app, ["data-audit", "--home", str(home)]).output,
        runner.invoke(app, ["data-audit", "--home", str(home), "--json"]).output,
        runner.invoke(app, ["scan", str(REPO / "tests" / "fixtures"), "--home", str(home)]).output,
    ]
    haystack = " ".join(" ".join(outputs).split())
    snippets = set()
    for r in records:
        for text in (r.question, r.answer, r.text):
            if text and len(text.strip()) >= 12:
                snippets |= _windows(text)
    leaked = [s for s in snippets if s in haystack]
    assert leaked == []


def test_audit_counts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(home)))
    audit = run_audit(AuditRequest(home=str(home)))
    assert audit.records_total == 36 and audit.by_type == {"qa_pair": 34, "source_block": 2}
    assert audit.by_format == {"csv": 23, "md": 5, "txt": 3, "xlsx": 5}
    assert audit.privacy_findings_by_type["aws_access_key"] == 1
    assert audit.exclusions["privacy_finding"] == 5 and audit.exclusions["blank_rows_skipped"] == 1
    assert audit.groups.largest_group_size == 2
    assert audit.question_chars is not None and audit.question_chars.p50 <= audit.question_chars.max
    assert run_audit(AuditRequest(home=str(home), source="fixtures")).records_total == 36


def test_audit_without_ingest_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(NotAvailableError, match="Nothing has been ingested"):
        run_audit(AuditRequest(home=str(tmp_path / "empty")))
    home = tmp_path / "home"
    run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(home)))
    with pytest.raises(NotAvailableError, match="No ingested data source matches"):
        run_audit(AuditRequest(home=str(home), source="other"))
