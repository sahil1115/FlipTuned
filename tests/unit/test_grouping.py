"""group_id priority (SPEC §9.2)."""

from __future__ import annotations

import random
from pathlib import Path

from fliptuned.config.models import RecordStatus
from fliptuned.dataset.builder import Example, categories_without_test_items, conflicting_answers
from fliptuned.dataset.grouping import assign_groups
from fliptuned.records.hashing import content_hash
from fliptuned.records.schema import (
    ExtractionMethod,
    GroupSource,
    Record,
    RecordType,
    SourceLocator,
)
from fliptuned.services.ingest import IngestRequest, load_records, run_ingest


def _rec(rid: str, question: str, group: str | None = None, source: GroupSource | None = None):
    return Record(
        id=rid,
        record_type=RecordType.QA_PAIR,
        source_file="f.csv",
        source_locator=SourceLocator(row=int(rid.split("_")[1])),
        question=question,
        answer=f"answer {rid}",
        status=RecordStatus.APPROVED,
        extraction_method=ExtractionMethod.CSV_COLUMN_MAPPING,
        content_hash=content_hash(question, f"answer {rid}"),
        group_id=group,
        group_source=source,
    )


def test_priority_explicit_thread_file_cluster_singleton() -> None:
    items = [
        ("s", _rec("qa_1", "Q one?", "g1", GroupSource.EXPLICIT_COLUMN)),
        ("s", _rec("qa_2", "Q two?", "t1", GroupSource.THREAD)),
        ("s", _rec("qa_3", "Q three?", "file:f.csv", GroupSource.SOURCE_FILE)),
        ("s", _rec("qa_4", "Q four?")),
        ("s", _rec("qa_5", "q   FOUR?")),  # differs in case and spacing only: same cluster as qa_4
        ("s", _rec("qa_6", "Q  four?")),  # same normalized question as qa_4
        ("s", _rec("qa_7", "Q one?")),  # duplicate question of an explicit group member
        ("s", _rec("qa_8", "Q eight?")),
    ]
    got = assign_groups(items)
    assert got[0].key == "s/explicit_column/g1" and got[0].source is GroupSource.EXPLICIT_COLUMN
    assert got[1].key == "s/thread/t1" and got[1].source is GroupSource.THREAD
    assert got[2].key == "s/source_file/file:f.csv" and got[2].source is GroupSource.SOURCE_FILE
    assert got[3].source is GroupSource.DUPLICATE_CLUSTER and got[3].key == got[5].key
    assert got[4].source is GroupSource.DUPLICATE_CLUSTER and got[4].key == got[3].key
    assert got[6].key == got[0].key and got[6].source is GroupSource.DUPLICATE_CLUSTER
    assert got[7].source is GroupSource.SINGLETON and got[7].key == "single/s/qa_8"


def test_column_values_are_scoped_per_source() -> None:
    a = _rec("qa_1", "Alpha?", "1", GroupSource.EXPLICIT_COLUMN)
    b = _rec("qa_2", "Beta?", "1", GroupSource.EXPLICIT_COLUMN)
    got = assign_groups([("src_a", a), ("src_b", b), ("src_a", b)])
    assert got[0].key != got[1].key and got[0].key == got[2].key


def test_cluster_owner_does_not_depend_on_order() -> None:
    items = [
        ("s", _rec("qa_1", "Same?", "zeta", GroupSource.EXPLICIT_COLUMN)),
        ("s", _rec("qa_2", "Same?", "alpha", GroupSource.EXPLICIT_COLUMN)),
        ("s", _rec("qa_3", "Same?")),
    ]
    expected = {r.id: g for (_, r), g in zip(items, assign_groups(items), strict=True)}
    rng = random.Random(7)
    for _ in range(5):
        shuffled = items[:]
        rng.shuffle(shuffled)
        got = {r.id: g for (_, r), g in zip(shuffled, assign_groups(shuffled), strict=True)}
        assert got == expected
    assert expected["qa_3"].key == "s/explicit_column/alpha"


def test_csv_without_group_columns_is_not_one_group(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "faq.csv"
    csv_path.parent.mkdir()
    rows = ["question,answer"] + [f"Question number {i}?,Answer {i}." for i in range(20)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = tmp_path / "ingest.yaml"
    config.write_text(
        "kind: ingest\nname: plain\nsources:\n  - path: data/faq.csv\n    format: csv\n"
        "    columns: {question: question, answer: answer}\n",
        encoding="utf-8",
    )
    result = run_ingest(IngestRequest(config_path=str(config), home=str(tmp_path / "home")))
    records = load_records(Path(result.records_path))
    groups = assign_groups([(result.data_source_id, r) for r in records])
    assert len({g.key for g in groups}) == 20
    assert {g.source for g in groups} == {GroupSource.SINGLETON}


def test_conflicting_answers_are_counted() -> None:
    examples = [
        Example("s", _rec("qa_1", "Same?"), 10, False),
        Example("s", _rec("qa_2", "Same?"), 10, False),
        Example("s", _rec("qa_3", "Other?"), 10, False),
    ]
    assert conflicting_answers(examples) == (1, 2)
    assert conflicting_answers(examples[1:]) == (0, 0)


def test_categories_without_test_items() -> None:
    def example(number: int, category: str) -> Example:
        record = _rec(f"qa_{number}", f"Q{number}?").model_copy(update={"category": category})
        return Example("s", record, 10, False)

    partitions = {
        "train": [example(i, "coding") for i in range(1, 6)] + [example(6, "ops")],
        "valid": [],
        "test": [example(7, "ops")],
    }
    assert categories_without_test_items(partitions) == ["coding"]
    partitions["test"].append(example(8, "coding"))
    assert categories_without_test_items(partitions) == []
