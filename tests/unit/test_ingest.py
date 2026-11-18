from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.models import RecordStatus
from fliptuned.errors import FlipTunedError
from fliptuned.records.schema import QualityFlag, Record, RecordType
from fliptuned.services.ingest import IngestRequest, PrivacyStopError, load_records, run_ingest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
SAMPLE = REPO / "configs" / "ingest.sample.yaml"
FAKE_SECRETS = [
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_FAKEfakeFAKEfakeFAKEfakeFAKEfake1234",
    "devpass42",
    "hunter2hunter2",
    "AAAAfakekeymaterialAAAA",
]


@pytest.fixture
def ingested(tmp_path: Path):
    result = run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(tmp_path / "home")))
    return result, load_records(Path(result.records_path))


def _by_file(records: list[Record], name: str) -> list[Record]:
    return [r for r in records if r.source_file == name]


def _raw_rows(name: str, newline_bytes: bool = True) -> list[list[str]]:
    text = (FIXTURES / name).read_bytes().decode("utf-8")
    return list(csv.reader(io.StringIO(text, newline="")))


def test_no_rows_are_dropped_silently(ingested) -> None:
    result, records = ingested
    data_rows = _raw_rows("qa_basic.csv")[1:]
    basic = _by_file(records, "qa_basic.csv")
    report = next(f for f in result.outcome.files if f.source_file == "qa_basic.csv")
    assert len(basic) + report.blank_rows_skipped == len(data_rows)
    for name in ("qa_crlf.csv", "secrets.csv"):
        assert len(_by_file(records, name)) == len(_raw_rows(name)) - 1
    malformed = next(f for f in result.outcome.files if f.source_file == "malformed.csv")
    assert malformed.errors and "row 4" in malformed.errors[0]
    assert len(_by_file(records, "malformed.csv")) == 2


def test_text_is_preserved_byte_for_byte(ingested) -> None:
    _, records = ingested
    crlf_rows = _raw_rows("qa_crlf.csv")[1:]
    crlf = _by_file(records, "qa_crlf.csv")
    assert [r.answer for r in crlf] == [row[1] for row in crlf_rows]
    assert "\r\n```python\r\n" in crlf[0].answer
    basic_rows = _raw_rows("qa_basic.csv")
    q1 = next(r for r in records if r.external_id == "Q-001")
    assert q1.answer == basic_rows[1][2] and "```python\nfrom pydantic" in q1.answer
    unicode = next(r for r in records if r.external_id == "Q-009")
    assert "café-safe" in unicode.question and "café ✓" in unicode.answer


def test_csv_mapping_status_tags_and_flags(ingested) -> None:
    _, records = ingested
    by_ext = {
        r.external_id: r
        for r in records
        if r.source_file == "qa_basic.csv" and r.external_id != "Q-004"
    }
    q4 = [r for r in records if r.external_id == "Q-004"]
    assert by_ext["Q-001"].tags == ["python", "validation"] and by_ext["Q-001"].category == "coding"
    assert by_ext["Q-009"].tags == ["python", "unicode"]
    assert by_ext["Q-001"].status is RecordStatus.APPROVED
    assert by_ext["Q-003"].status is RecordStatus.PENDING_REVIEW
    assert by_ext["Q-009"].status is RecordStatus.REJECTED
    assert q4[0].quality_flags == [] and QualityFlag.DUPLICATE_ID in q4[1].quality_flags
    assert QualityFlag.EXACT_DUPLICATE in by_ext["Q-006"].quality_flags
    assert by_ext["Q-007"].status is RecordStatus.QUARANTINED
    assert QualityFlag.LIKELY_QA_SWAP in by_ext["Q-008"].quality_flags
    assert QualityFlag.BROKEN_CODE_FENCE in by_ext["Q-010"].quality_flags
    assert QualityFlag.TOO_SHORT in by_ext["Q-011"].quality_flags
    assert by_ext["Q-001"].source_locator.row == 2 and by_ext["Q-001"].source_file == "qa_basic.csv"


def test_xlsx_sheet_rows_and_groups(ingested) -> None:
    _, records = ingested
    xlsx = _by_file(records, "qa.xlsx")
    assert [r.source_locator.row for r in xlsx] == [2, 3, 4, 5, 6]
    assert {r.source_locator.sheet for r in xlsx} == {"QA"}
    assert [r.group_id for r in xlsx] == [
        "onboarding",
        "onboarding",
        "dependencies",
        "dependencies",
        None,
    ]
    assert xlsx[0].group_source == "explicit_column" and xlsx[4].group_source is None


def test_markdown_and_txt_extraction(ingested) -> None:
    _, records = ingested
    md = _by_file(records, "notes.md")
    md_qa = [r for r in md if r.record_type is RecordType.QA_PAIR]
    assert len(md_qa) == 4
    assert "```python\nfrom dataclasses import dataclass" in md_qa[0].answer
    assert "Q: this is not a new question" in md_qa[1].answer
    assert "| frozen | immutable |" in md_qa[1].answer
    assert QualityFlag.BROKEN_CODE_FENCE in md_qa[3].quality_flags
    blocks = [r for r in md if r.record_type is RecordType.SOURCE_BLOCK]
    assert len(blocks) == 1 and blocks[0].question is None and blocks[0].text
    txt = _by_file(records, "notes.txt")
    assert [r.record_type.value for r in txt] == ["source_block", "qa_pair", "qa_pair"]
    assert (
        txt[2].answer
        == "The on-call engineer for the service.\nThey confirm in the deploy channel first."
    )


def test_malformed_and_bad_encoding_are_reported(ingested, tmp_path: Path) -> None:
    _, records = ingested
    ragged = _by_file(records, "malformed.csv")[1]
    assert (
        QualityFlag.MALFORMED_ROW in ragged.quality_flags
        and ragged.status is RecordStatus.QUARANTINED
    )
    (bad,) = _by_file(records, "bad_encoding.csv")
    assert QualityFlag.INVALID_UTF8 in bad.quality_flags and bad.status is RecordStatus.QUARANTINED

    config = tmp_path / "cp1252.yaml"
    config.write_text(
        f"kind: ingest\nname: cp\nsources:\n  - path: {json.dumps(str(FIXTURES / 'bad_encoding.csv'))}\n"
        "    format: csv\n    encoding: cp1252\n    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    result = run_ingest(IngestRequest(config_path=str(config), home=str(tmp_path / "h2")))
    (fixed,) = load_records(Path(result.records_path))
    assert fixed.question == "Where is the café?" and fixed.quality_flags == []


def test_missing_required_column_is_reported_not_dropped(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        f"kind: ingest\nname: x\nsources:\n  - path: {json.dumps(str(FIXTURES / 'qa_crlf.csv'))}\n"
        "    format: csv\n    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    result = run_ingest(IngestRequest(config_path=str(config), home=str(tmp_path / "h")))
    assert result.counts.records_total == 0 and result.counts.rows_not_ingested == 3
    assert "'question'" in result.outcome.files[0].errors[0]
    assert "'Question'" in result.outcome.files[0].errors[0]


def test_broken_xlsx_is_reported_not_fatal(tmp_path: Path) -> None:
    bad = tmp_path / "data" / "broken.xlsx"
    bad.parent.mkdir()
    bad.write_bytes(b"this is not a workbook")
    config = tmp_path / "x.yaml"
    config.write_text(
        f"kind: ingest\nname: broken\nsources:\n  - path: {json.dumps(str(bad))}\n"
        "    format: xlsx\n    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    result = run_ingest(IngestRequest(config_path=str(config), home=str(tmp_path / "h")))
    assert result.counts.records_total == 0
    assert "not a valid .xlsx file" in result.outcome.files[0].errors[0]


def test_duplicate_column_names_are_reported(tmp_path: Path) -> None:
    data = tmp_path / "data" / "dup.csv"
    data.parent.mkdir()
    data.write_text("question,answer,answer\nQ one?,first,second\n", encoding="utf-8")
    config = tmp_path / "d.yaml"
    config.write_text(
        f"kind: ingest\nname: dup\nsources:\n  - path: {json.dumps(str(data))}\n"
        "    format: csv\n    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    result = run_ingest(IngestRequest(config_path=str(config), home=str(tmp_path / "h")))
    assert result.counts.records_total == 0
    assert "'answer' appear more than once" in result.outcome.files[0].errors[0]


def test_ingest_is_deterministic(tmp_path: Path) -> None:
    a = run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(tmp_path / "a")))
    b = run_ingest(IngestRequest(config_path=str(SAMPLE), home=str(tmp_path / "b")))
    assert Path(a.records_path).read_bytes() == Path(b.records_path).read_bytes()


def test_overwrite_is_required_to_replace(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    run_ingest(IngestRequest(config_path=str(SAMPLE), home=home))
    with pytest.raises(FlipTunedError, match="already ingested"):
        run_ingest(IngestRequest(config_path=str(SAMPLE), home=home))
    run_ingest(IngestRequest(config_path=str(SAMPLE), home=home, overwrite=True))


def _secrets_config(tmp_path: Path, policy: str) -> Path:
    config = tmp_path / f"secrets-{policy}.yaml"
    config.write_text(
        f"kind: ingest\nname: s-{policy}\nprivacy: {{policy: {policy}}}\nsources:\n"
        f"  - path: {json.dumps(str(FIXTURES / 'secrets.csv'))}\n"
        "    format: csv\n    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    return config


def test_privacy_policies(tmp_path: Path) -> None:
    excl = run_ingest(
        IngestRequest(
            config_path=str(_secrets_config(tmp_path, "exclude")), home=str(tmp_path / "h")
        )
    )
    records = load_records(Path(excl.records_path))
    flagged = [r for r in records if r.privacy_flags]
    assert len(flagged) == 5 and all(r.status is RecordStatus.QUARANTINED for r in flagged)
    placeholder = next(r for r in records if "placeholder" in (r.question or ""))
    assert placeholder.privacy_flags == [] and placeholder.status is RecordStatus.PENDING_REVIEW

    warn = run_ingest(
        IngestRequest(config_path=str(_secrets_config(tmp_path, "warn")), home=str(tmp_path / "h"))
    )
    warned = load_records(Path(warn.records_path))
    assert sum(1 for r in warned if r.privacy_flags) == 5
    assert all(r.status is RecordStatus.PENDING_REVIEW for r in warned)

    with pytest.raises(PrivacyStopError) as stop:
        run_ingest(
            IngestRequest(
                config_path=str(_secrets_config(tmp_path, "stop")), home=str(tmp_path / "h3")
            )
        )
    assert "5 likely secret" in stop.value.message
    stop_dirs = list((tmp_path / "h3" / "sources").iterdir())
    assert [p.name for p in stop_dirs[0].iterdir()] == ["ingest_report.json"]  # no records written


def test_secrets_never_appear_in_reports_cli_or_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="fliptuned")
    home = tmp_path / "home"
    runner = CliRunner()
    outputs = [
        runner.invoke(app, ["ingest", "--config", str(SAMPLE), "--home", str(home)]).output,
        runner.invoke(
            app, ["ingest", "--config", str(SAMPLE), "--home", str(home), "--overwrite", "--json"]
        ).output,
        runner.invoke(app, ["data-audit", "--home", str(home)]).output,
        runner.invoke(app, ["data-audit", "--home", str(home), "--json"]).output,
        runner.invoke(
            app, ["ingest", "--config", str(_secrets_config(tmp_path, "stop")), "--home", str(home)]
        ).output,
    ]
    reports = [
        p.read_text(encoding="utf-8") for p in (home / "sources").rglob("ingest_report.json")
    ]
    assert "REDACTED" in outputs[0] and "REDACTED" in outputs[4]
    haystack = "\n".join([*outputs, *reports, caplog.text])
    for secret in FAKE_SECRETS:
        assert secret not in haystack


def test_cli_ingest_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()
    stop = runner.invoke(
        app,
        [
            "ingest",
            "--config",
            str(_secrets_config(tmp_path, "stop")),
            "--home",
            str(tmp_path / "h"),
        ],
    )
    assert stop.exit_code == 5 and "What to do:" in stop.output
    missing = runner.invoke(
        app, ["ingest", "--config", str(tmp_path / "nope.yaml"), "--home", str(tmp_path)]
    )
    assert missing.exit_code == 3
