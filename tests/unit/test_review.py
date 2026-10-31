"""Review service: filters, decisions, staleness, CSV round trip (SPEC §9.3)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.models import RecordStatus
from fliptuned.errors import ConfigError, FlipTunedError
from fliptuned.services.ingest import IngestRequest, run_ingest
from fliptuned.services.review import (
    DecideRequest,
    ExportRequest,
    ImportRequest,
    ListRecordsRequest,
    RecordRef,
    ReviewFilter,
    decide,
    export_csv,
    import_csv,
    list_records,
)
from tests.helpers.pipeline import ingest_samples

runner = CliRunner()
FAKE_SECRETS = ["AKIAIOSFODNN7EXAMPLE", "ghp_FAKEfakeFAKEfakeFAKEfakeFAKEfake1234", "devpass42"]


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    ingest_samples(home)
    return home


def _list(home: Path, **filters):
    return list_records(
        ListRecordsRequest(home=str(home), filter=ReviewFilter(**filters), limit=1000)
    )


def _ref(item) -> RecordRef:
    return RecordRef(source_id=item.source_id, record_id=item.record_id)


def test_filters(home: Path) -> None:
    everything = _list(home)
    assert everything.total == 181  # qa pairs only; 2 source blocks hidden by default
    assert _list(home, include_source_blocks=True).total == 183
    assert everything.by_status == {
        "approved": 140,
        "pending_review": 26,
        "quarantined": 8,
        "rejected": 7,
    }
    pending = _list(home, statuses=["pending_review"])
    assert pending.total == 26 and {i.status for i in pending.items} == {"pending_review"}
    assert _list(home, sources=["synthetic"]).total == 147
    swaps = _list(home, flag="likely_qa_swap")
    assert swaps.total == 1 and swaps.items[0].record_id
    dups = _list(home, only_duplicates=True)
    clusters = {i.duplicate_cluster for i in dups.items}
    assert dups.total >= 4 and None not in clusters
    one = next(iter(sorted(clusters)))
    assert all(i.duplicate_cluster == one for i in _list(home, duplicate_cluster=one).items)
    assert _list(home, category="testing").total == 12
    page = list_records(ListRecordsRequest(home=str(home), limit=10, offset=175))
    assert page.total == 181 and len(page.items) == 6


def test_privacy_findings_are_redacted(home: Path) -> None:
    flagged = _list(home, flag="privacy")
    assert flagged.total == 5
    text = " ".join(f"{i.question} {i.answer}" for i in flagged.items)
    assert "[REDACTED" in text
    assert not [s for s in FAKE_SECRETS if s in text]
    assert all(i.privacy_findings for i in flagged.items)


def test_decide_bulk_and_skips(home: Path) -> None:
    pending = _list(home, statuses=["pending_review"]).items[:3]
    quarantined = _list(home, statuses=["quarantined"]).items[0]
    block = next(i for i in _list(home, include_source_blocks=True).items if i.text)
    approved = _list(home, statuses=["approved"]).items[0]
    refs = [*map(_ref, pending), _ref(quarantined), _ref(block), _ref(approved)]
    refs.append(RecordRef(source_id="src_x", record_id="qa_missing"))
    result = decide(DecideRequest(home=str(home), records=refs, status="approved", origin="bulk"))
    assert result.applied == 3 and result.unchanged == 1
    assert result.skipped == {"not_a_qa_pair": 1, "quarantined": 1, "unknown_record": 1}
    after = {i.record_id: i for i in _list(home).items}
    for item in pending:
        got = after[item.record_id]
        assert got.status == "approved" and got.decision_origin == "bulk"
        assert got.ingested_status == "pending_review"

    reject = decide(DecideRequest(home=str(home), records=[_ref(pending[0])], status="rejected"))
    assert reject.applied == 1
    assert {i.record_id: i for i in _list(home).items}[pending[0].record_id].status == "rejected"


def test_decision_goes_stale_when_content_changes(tmp_path: Path) -> None:
    data = tmp_path / "data" / "faq.csv"
    data.parent.mkdir()
    config = tmp_path / "ingest.yaml"
    config.write_text(
        "kind: ingest\nname: faq\nsources:\n  - path: data/faq.csv\n    format: csv\n"
        "    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"

    def ingest(answer: str, overwrite: bool = False) -> None:
        data.write_text(
            f"question,answer\nHow do I deploy?,{answer}\nWho owns CI?,The platform team.\n",
            encoding="utf-8",
        )
        run_ingest(IngestRequest(config_path=str(config), home=str(home), overwrite=overwrite))

    ingest("Run the deploy pipeline.")
    items = _list(home).items
    decide(DecideRequest(home=str(home), records=[_ref(i) for i in items], status="approved"))
    assert {i.status for i in _list(home).items} == {RecordStatus.APPROVED}

    ingest("Ask the release manager.", overwrite=True)
    after = {i.location: i for i in _list(home).items}
    assert after["row 2"].status == "pending_review" and after["row 2"].decision_stale
    assert after["row 3"].status == "approved" and not after["row 3"].decision_stale


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_csv_round_trip(home: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "review.csv"
    exported = export_csv(
        ExportRequest(
            home=str(home), out=str(sheet), filter=ReviewFilter(statuses=["pending_review"])
        )
    )
    assert exported.rows == 26 and exported.by_status == {"pending_review": 26}
    rows = _read(sheet)
    assert not [s for s in FAKE_SECRETS if s in sheet.read_text(encoding="utf-8-sig")]

    rows[0]["decision"] = "approved"
    rows[1]["decision"] = "Reject"
    rows[1]["reason"] = "outdated"
    rows[2]["decision"] = "a"
    rows[2]["question"] = "edited text is ignored"
    rows[3]["decision"] = "maybe"
    rows[4]["decision"] = "approved"
    rows[4]["content_hash"] = "0" * 64
    rows[5]["decision"] = "approved"
    rows[5]["record_id"] = "qa_unknown"
    rows[6]["decision"] = "pending"  # already pending
    _write(sheet, rows)

    result = import_csv(ImportRequest(home=str(home), path=str(sheet)))
    assert result.rows == 26 and result.blank == 19 and result.unchanged == 1
    assert result.applied == {"approved": 2, "rejected": 1}
    assert result.skipped == {"invalid_decision": 1, "stale_content": 1, "unknown_record": 1}
    assert [p.row for p in result.problems] == [5, 6, 7]

    after = {i.record_id: i for i in _list(home).items}
    assert after[rows[0]["record_id"]].status == "approved"
    assert after[rows[1]["record_id"]].status == "rejected"
    assert after[rows[2]["record_id"]].question != "edited text is ignored"
    assert after[rows[2]["record_id"]].decision_origin == "csv"

    again = import_csv(ImportRequest(home=str(home), path=str(sheet)))
    assert again.applied == {} and again.unchanged == 4


def test_export_refuses_overwrite_and_escapes_formulas(tmp_path: Path) -> None:
    data = tmp_path / "data" / "f.csv"
    data.parent.mkdir()
    data.write_text("question,answer\n=1+1 what?,@SUM(A1)\n", encoding="utf-8")
    config = tmp_path / "ingest.yaml"
    config.write_text(
        "kind: ingest\nname: f\nsources:\n  - path: data/f.csv\n    format: csv\n"
        "    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    run_ingest(IngestRequest(config_path=str(config), home=str(home)))
    sheet = tmp_path / "out.csv"
    export_csv(ExportRequest(home=str(home), out=str(sheet)))
    (row,) = _read(sheet)
    assert row["question"] == "'=1+1 what?" and row["answer"] == "'@SUM(A1)"
    with pytest.raises(FlipTunedError, match="already exists"):
        export_csv(ExportRequest(home=str(home), out=str(sheet)))


def test_import_rejects_foreign_csv(home: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "other.csv"
    sheet.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing column"):
        import_csv(ImportRequest(home=str(home), path=str(sheet)))


def test_cli_review_prints_counts_only(home: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    out = runner.invoke(
        app,
        [
            "review",
            "export",
            "--out",
            str(sheet),
            "--home",
            str(home),
            "--status",
            "pending_review",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "26 record(s)" in out.output
    rows = _read(sheet)
    rows[0]["decision"] = "approved"
    _write(sheet, rows)
    imported = runner.invoke(app, ["review", "import", str(sheet), "--home", str(home)])
    assert imported.exit_code == 0, imported.output
    assert "approved" in imported.output
    for row in rows:
        for text in (row["question"], row["answer"]):
            if len(text) > 20:
                assert text not in out.output + imported.output
    again = runner.invoke(app, ["review", "export", "--out", str(sheet), "--home", str(home)])
    assert again.exit_code == 1 and "--overwrite" in again.output
