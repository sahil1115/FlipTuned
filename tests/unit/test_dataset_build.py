"""build-dataset, validate-dataset and data-audit --processed (fake tokenizer, no downloads)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.loader import load_config
from fliptuned.config.models import DatasetConfig, RecordStatus
from fliptuned.dataset.builder import plan_dataset
from fliptuned.dataset.manifest import PARTITION_FILES
from fliptuned.dataset.split import PARTITIONS
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.records.schema import RecordType
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.audit import ProcessedAuditRequest, run_processed_audit
from fliptuned.services.datasets import (
    BuildDatasetRequest,
    ValidateDatasetRequest,
    build_dataset,
    read_index,
    validate_dataset,
)
from fliptuned.services.ingest import load_records
from fliptuned.services.review import DecideRequest, RecordRef, decide
from tests.helpers.pipeline import (
    SAMPLE_DATASET,
    FakeCounter,
    dataset_config,
    ingest_samples,
    use_fake_tokenizer,
)

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    return home


def _build(home: Path, config: Path = SAMPLE_DATASET, **kw):
    return build_dataset(BuildDatasetRequest(config_path=str(config), home=str(home), **kw))


def _all_records(home: Path):
    out = []
    for folder in sorted((home / "sources").iterdir()):
        out += [(folder.name, r) for r in load_records(folder / "records.jsonl")]
    return out


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_counts_reconcile_and_unapproved_are_excluded(home: Path) -> None:
    result = _build(home)
    c = result.counts
    records = _all_records(home)
    assert c.input_records == len(records) == 183
    assert c.included + sum(c.excluded.values()) == c.input_records
    assert sum(c.partitions.values()) == c.included
    assert c.excluded == {
        "duplicate": 2,
        "not_qa_pair": 2,
        "over_max_seq_length": 1,
        "pending_review": 26,
        "quarantined": 8,
        "rejected": 7,
    }

    folder = Path(result.path)
    included = {(r.source_id, r.record_id) for r in read_index(folder)}
    excluded = json.loads((folder / "exclusions.json").read_text(encoding="utf-8"))
    assert included.isdisjoint((x["source_id"], x["record_id"]) for x in excluded)
    assert len(included) + len(excluded) == len(records)
    by_ref = {(s, r.id): r for s, r in records}
    for ref in included:
        rec = by_ref[ref]
        assert rec.record_type is RecordType.QA_PAIR and rec.status is RecordStatus.APPROVED


def test_no_group_spans_partitions_and_no_record_repeats(home: Path) -> None:
    folder = Path(_build(home).path)
    index = read_index(folder)
    parts: dict[str, set[str]] = {}
    for row in index:
        parts.setdefault(row.group_key, set()).add(row.partition)
    assert all(len(p) == 1 for p in parts.values())
    assert len({(r.source_id, r.record_id) for r in index}) == len(index)
    for partition, filename in PARTITION_FILES.items():
        assert len(_lines(folder / filename)) == sum(1 for r in index if r.partition == partition)


def test_duplicates_report_and_cluster_grouping(home: Path) -> None:
    folder = Path(_build(home).path)
    dups = json.loads((folder / "duplicates.json").read_text(encoding="utf-8"))
    records = {r.id: r for _, r in _all_records(home)}
    pairs = {
        (records[d["record_id"]].external_id, records[d["kept_record_id"]].external_id)
        for d in dups
    }
    assert ("S-9001", "S-0001") in pairs
    assert ("Q-006", "Q-002") in pairs
    index = {records[r.record_id].external_id: r for r in read_index(folder)}
    # S-9002 has no group of its own but repeats S-0003's question, so it joins group G-00.
    assert index["S-9002"].group_key == index["S-0003"].group_key
    assert index["S-9002"].group_source == "duplicate_cluster"


def test_canonical_lines_keep_text_byte_for_byte(home: Path) -> None:
    records = _all_records(home)
    crlf = [(s, r) for s, r in records if r.source_file == "qa_crlf.csv"]
    decide(
        DecideRequest(
            home=str(home),
            records=[RecordRef(source_id=s, record_id=r.id) for s, r in crlf],
            status="approved",
        )
    )
    folder = Path(_build(home).path)
    config = load_config(SAMPLE_DATASET, DatasetConfig)
    by_ref = {(s, r.id): r for s, r in records}
    seen_crlf = 0
    for row in read_index(folder):
        line = _lines(folder / PARTITION_FILES[row.partition])[row.line - 1]
        rec = by_ref[(row.source_id, row.record_id)]
        assert line == {
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": rec.question},
                {"role": "assistant", "content": rec.answer},
            ]
        }
        seen_crlf += "\r\n" in rec.answer
    assert seen_crlf == 1  # the CRLF answer survived unchanged


def test_build_is_byte_identical_across_homes_and_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    first = Path(_build(home).path)
    other_home = tmp_path / "elsewhere" / "second-home"
    ingest_samples(other_home)
    second = Path(_build(other_home).path)
    assert first.name == second.name
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    config = load_config(SAMPLE_DATASET, DatasetConfig)
    items = _all_records(home)
    baseline = plan_dataset(items, config, FakeCounter())
    rng = random.Random(0)
    for _ in range(3):
        shuffled = items[:]
        rng.shuffle(shuffled)
        plan = plan_dataset(shuffled, config, FakeCounter())
        for p in PARTITIONS:
            assert [e.record.id for e in plan.partitions[p]] == [
                e.record.id for e in baseline.partitions[p]
            ]
        assert plan.exclusions == baseline.exclusions


def test_golden_dataset_digest(home: Path) -> None:
    """Pinned index digest: the full fixture pipeline must give the same split on every OS.
    If the algorithm or fixtures change on purpose, update this value (and BUILDER_VERSION)."""
    folder = Path(_build(home).path)
    digest = hashlib.sha256((folder / "index.jsonl").read_bytes()).hexdigest()
    assert digest == GOLDEN_INDEX_DIGEST


# Pinned on Windows (2026-10-29); CI checks it on Linux and macOS.
GOLDEN_INDEX_DIGEST = "9294969e8d32c677289b225ec26a3aec02f050310de1765fee755bb0de9bde64"


def test_seq_length_quarantine_and_truncation(home: Path, tmp_path: Path) -> None:
    quarantined = _build(home)
    assert quarantined.counts.excluded["over_max_seq_length"] == 1
    assert any(w.code == "over_max_seq_length" for w in quarantined.warnings)

    truncating = _build(home, dataset_config(tmp_path, name="trunc", truncate_long_records=True))
    assert "over_max_seq_length" not in truncating.counts.excluded
    assert truncating.counts.truncated == 1
    assert any(w.code == "truncated" for w in truncating.warnings)
    assert sum(r.truncated for r in read_index(Path(truncating.path))) == 1


def test_size_warnings_and_manifest(home: Path) -> None:
    result = _build(home)
    codes = {w.code for w in result.warnings}
    assert {"small_train_set", "small_test_set"} <= codes
    manifest = json.loads((Path(result.path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["catalog_id"] == "tiny-qwen2"
    assert manifest["counts"] == result.counts.model_dump()
    assert "created_at" not in json.dumps(manifest)
    for name, info in manifest["files"].items():
        data = (Path(result.path) / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"]


def test_rebuild_needs_overwrite_and_is_registered(home: Path) -> None:
    first = _build(home)
    with pytest.raises(FlipTunedError, match="already built"):
        _build(home)
    again = _build(home, overwrite=True)
    assert again.dataset_id == first.dataset_id and again.rebuilt
    assert not [p for p in (home / "datasets").iterdir() if p.name.startswith(".")]
    with connect(home / "fliptuned.db") as conn:
        row = repositories.get_dataset(conn, first.dataset_id)
        assert row["status"] == "ready" and row["manifest_hash"] == again.manifest_hash


def test_new_decision_changes_the_dataset_id(home: Path) -> None:
    first = _build(home)
    pending = next(r for s, r in _all_records(home) if r.status is RecordStatus.PENDING_REVIEW)
    source = next(s for s, r in _all_records(home) if r.id == pending.id)
    decide(
        DecideRequest(
            home=str(home),
            records=[RecordRef(source_id=source, record_id=pending.id)],
            status="approved",
        )
    )
    assert _build(home).dataset_id != first.dataset_id


def test_unknown_source_and_model_are_config_errors(home: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No ingested data source"):
        _build(home, dataset_config(tmp_path, source_ids=["nope"]))
    with pytest.raises(ConfigError, match="not in the catalog"):
        _build(home, dataset_config(tmp_path, name="m.yaml", model_catalog_id="nope"))
    only = _build(home, dataset_config(tmp_path, name="s.yaml", source_ids=["synthetic"]))
    assert only.counts.input_records == 147


def test_split_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home, synthetic=False)  # only a handful of approved records
    result = runner.invoke(
        app, ["build-dataset", "--config", str(SAMPLE_DATASET), "--home", str(home)]
    )
    assert result.exit_code == 5, result.output
    assert "split" in result.output and "What to do" in result.output


def test_validate_passes_then_detects_tampering(home: Path) -> None:
    result = _build(home)
    report = validate_dataset(ValidateDatasetRequest(home=str(home), dataset=result.dataset_id))
    assert report.ok
    status = {c.name: c.status.value for c in report.checks}
    assert status["group_leakage"] == status["files"] == status["format"] == "pass"
    assert status["train_size"] == "warn"

    train = Path(result.path) / "train.jsonl"
    lines = train.read_text(encoding="utf-8").splitlines(keepends=True)
    train.write_text("".join(lines[1:]) + '{"messages": []}\n', encoding="utf-8")
    report = validate_dataset(ValidateDatasetRequest(home=str(home), dataset="fixtures"))
    failed = {c.name for c in report.checks if c.status.value == "fail"}
    assert not report.ok and {"files", "format"} <= failed
    cli = runner.invoke(app, ["validate-dataset", "fixtures", "--home", str(home)])
    assert cli.exit_code == 5 and "INVALID" in cli.output


def test_validate_unknown_dataset(home: Path) -> None:
    with pytest.raises(NotAvailableError, match="No dataset"):
        validate_dataset(ValidateDatasetRequest(home=str(home), dataset="nope"))


def test_processed_audit_counts_and_no_content(home: Path) -> None:
    result = _build(home)
    audit = run_processed_audit(ProcessedAuditRequest(home=str(home), dataset=result.dataset_id))
    assert {p: s.examples for p, s in audit.partitions.items()} == result.counts.partitions
    assert audit.excluded == result.counts.excluded
    assert audit.partitions["train"].tokens is not None

    outputs = " ".join(
        runner.invoke(
            app, ["data-audit", "--processed", "fixtures", "--home", str(home), *flag]
        ).output
        for flag in ([], ["--json"])
    )
    outputs += runner.invoke(
        app, ["build-dataset", "--config", str(SAMPLE_DATASET), "--home", str(home), "--overwrite"]
    ).output
    haystack = " ".join(outputs.split())
    leaked = []
    for _, r in _all_records(home):
        for text in (r.question, r.answer, r.text):
            words = (text or "").split()
            leaked += [
                " ".join(words[i : i + 6])
                for i in range(len(words) - 5)
                if " ".join(words[i : i + 6]) in haystack
            ]
    assert leaked == []


def test_cli_build_and_audit_round_trip(home: Path) -> None:
    built = runner.invoke(
        app, ["build-dataset", "--config", str(SAMPLE_DATASET), "--home", str(home)]
    )
    assert built.exit_code == 0, built.output
    assert "Partitions" in built.output and "Warnings" in built.output
    audit = runner.invoke(app, ["data-audit", "--processed", "fixtures", "--home", str(home)])
    assert audit.exit_code == 0 and "Tokens" in audit.output
    valid = runner.invoke(app, ["validate-dataset", "fixtures", "--home", str(home), "--json"])
    assert valid.exit_code == 0 and json.loads(valid.output)["ok"] is True
