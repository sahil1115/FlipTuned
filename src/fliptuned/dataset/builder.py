"""Plan a dataset build from records whose review status is already applied (pure, no I/O).

Steps: select approved ``qa_pair`` records → dedup → token-length check → group → split.
Every input record ends up either in exactly one partition or in ``exclusions`` with one reason,
so the counts always reconcile. Nothing depends on input order: records are sorted by a stable
key first.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from fliptuned.config.models import DatasetConfig, RecordStatus
from fliptuned.dataset.grouping import GroupAssignment, assign_groups, question_key
from fliptuned.dataset.split import PARTITIONS, SplitResult, split_groups
from fliptuned.models.tokenizer import TokenCounter
from fliptuned.records.hashing import dedup_key, sha256_hex
from fliptuned.records.schema import Record, RecordType

LARGE_GROUP_SHARE = 0.10  # of all approved records (SPEC §9.2)
DOMINANT_GROUP_SHARE = 1 / 3  # of the valid or test partition

# Exclusion reasons, in the order they are checked.
REASON_NOT_QA = "not_qa_pair"
REASON_QUARANTINED = "quarantined"
REASON_PENDING = "pending_review"
REASON_REJECTED = "rejected"
REASON_DUPLICATE = "duplicate"
REASON_TOO_LONG = "over_max_seq_length"


class BuildWarning(BaseModel):
    code: str
    message: str


class TokenStats(BaseModel):
    p50: int
    p90: int
    p99: int
    max: int


@dataclass
class Example:
    source_id: str
    record: Record
    tokens: int
    truncated: bool
    group: GroupAssignment | None = None
    partition: str | None = None


@dataclass
class Exclusion:
    source_id: str
    record_id: str
    reason: str


@dataclass
class Duplicate:
    source_id: str
    record_id: str
    kept_source_id: str
    kept_record_id: str


@dataclass
class DatasetPlan:
    input_records: int
    partitions: dict[str, list[Example]]
    exclusions: list[Exclusion]
    duplicates: list[Duplicate]
    split: SplitResult
    warnings: list[BuildWarning] = field(default_factory=list)

    def excluded_by_reason(self) -> dict[str, int]:
        return dict(sorted(Counter(e.reason for e in self.exclusions).items()))


def stable_key(item: tuple[str, Record]) -> tuple:
    source_id, r = item
    loc = r.source_locator
    return (source_id, r.source_file, loc.sheet or "", loc.row or 0, loc.line or 0, r.id)


def canonical_messages(system_prompt: str, record: Record) -> list[dict[str, str]]:
    """SPEC §10.1. Text is passed through byte for byte; provenance never enters it."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record.question or ""},
        {"role": "assistant", "content": record.answer or ""},
    ]


def example_order_key(seed: int, source_id: str, record_id: str) -> str:
    return sha256_hex(str(seed), source_id, record_id)


def plan_dataset(
    items: Sequence[tuple[str, Record]], config: DatasetConfig, counter: TokenCounter
) -> DatasetPlan:
    ordered = sorted(items, key=stable_key)
    exclusions: list[Exclusion] = []
    candidates: list[tuple[str, Record]] = []
    for source_id, record in ordered:
        reason = _exclusion_reason(record)
        if reason:
            exclusions.append(Exclusion(source_id, record.id, reason))
        else:
            candidates.append((source_id, record))

    duplicates: list[Duplicate] = []
    kept_by_key: dict[str, tuple[str, Record]] = {}
    unique: list[tuple[str, Record]] = []
    for source_id, record in candidates:
        key = dedup_key(record.question, record.answer)
        if key in kept_by_key:
            kept_source, kept = kept_by_key[key]
            duplicates.append(Duplicate(source_id, record.id, kept_source, kept.id))
            exclusions.append(Exclusion(source_id, record.id, REASON_DUPLICATE))
        else:
            kept_by_key[key] = (source_id, record)
            unique.append((source_id, record))

    examples: list[Example] = []
    for source_id, record in unique:
        tokens = counter.count(canonical_messages(config.system_prompt, record))
        too_long = tokens > config.max_seq_length
        if too_long and not config.truncate_long_records:
            exclusions.append(Exclusion(source_id, record.id, REASON_TOO_LONG))
            continue
        examples.append(Example(source_id, record, tokens, truncated=too_long))

    for example, group in zip(
        examples, assign_groups([(e.source_id, e.record) for e in examples]), strict=True
    ):
        example.group = group
    sizes = Counter(e.group.key for e in examples if e.group)
    split = split_groups(sizes, config.split.model_dump(), config.seed)

    partitions: dict[str, list[Example]] = {p: [] for p in PARTITIONS}
    for example in examples:
        assert example.group is not None
        example.partition = split.assignment[example.group.key]
        partitions[example.partition].append(example)
    for part in partitions.values():
        part.sort(key=lambda e: example_order_key(config.seed, e.source_id, e.record.id))

    plan = DatasetPlan(
        input_records=len(ordered),
        partitions=partitions,
        exclusions=exclusions,
        duplicates=duplicates,
        split=split,
    )
    plan.warnings = _warnings(plan, sizes, config)
    return plan


def _exclusion_reason(record: Record) -> str | None:
    if record.record_type is not RecordType.QA_PAIR:
        return REASON_NOT_QA
    if record.status is RecordStatus.QUARANTINED:
        return REASON_QUARANTINED
    if record.status is RecordStatus.PENDING_REVIEW:
        return REASON_PENDING
    if record.status is RecordStatus.REJECTED:
        return REASON_REJECTED
    return None


def cross_partition_duplicates(partitions: dict[str, list[Example]]) -> int:
    """Records whose normalized question also appears in another partition."""
    seen: dict[str, set[str]] = {}
    for name, examples in partitions.items():
        for e in examples:
            seen.setdefault(question_key(e.record), set()).add(name)
    return sum(
        1
        for examples in partitions.values()
        for e in examples
        if len(seen[question_key(e.record)]) > 1
    )


def _warnings(plan: DatasetPlan, sizes: Counter[str], config: DatasetConfig) -> list[BuildWarning]:
    warnings: list[BuildWarning] = []
    included = sum(sizes.values())
    large = [s for s in sizes.values() if s / included > LARGE_GROUP_SHARE]
    if large:
        warnings.append(
            BuildWarning(
                code="large_group",
                message=(
                    f"{len(large)} group(s) hold more than {LARGE_GROUP_SHARE:.0%} of the approved "
                    f"records (largest: {max(large)} records, {max(large) / included:.1%}). Large "
                    "groups make the split lumpy; check the group/thread columns."
                ),
            )
        )
    for name in ("valid", "test"):
        part_sizes = Counter(e.group.key for e in plan.partitions[name] if e.group)
        biggest = max(part_sizes.values(), default=0)
        n = len(plan.partitions[name])
        if n and biggest / n > DOMINANT_GROUP_SHARE:
            warnings.append(
                BuildWarning(
                    code=f"{name}_dominated_by_group",
                    message=(
                        f"One group is {biggest} of the {n} {name} examples ({biggest / n:.0%}), "
                        f"so {name} results mostly reflect one topic. A different seed or more "
                        "approved records spreads it out."
                    ),
                )
            )
    leaks = cross_partition_duplicates(plan.partitions)
    if leaks:
        warnings.append(
            BuildWarning(
                code="cross_partition_duplicates",
                message=(
                    f"{leaks} record(s) share an identical normalized question with a record in "
                    "another partition (their explicit groups differ). The test set may overstate "
                    "quality."
                ),
            )
        )
    n_train, n_test = len(plan.partitions["train"]), len(plan.partitions["test"])
    t = config.thresholds
    if n_train < t.min_train_examples_warn:
        warnings.append(
            BuildWarning(
                code="small_train_set",
                message=(
                    f"Only {n_train} training examples (fewer than {t.min_train_examples_warn}). "
                    "Expect style-level gains at most."
                ),
            )
        )
    if n_test < t.min_test_examples_conclusive:
        warnings.append(
            BuildWarning(
                code="small_test_set",
                message=(
                    f"Only {n_test} test examples (fewer than {t.min_test_examples_conclusive}). "
                    "The evaluation verdict will be 'inconclusive'."
                ),
            )
        )
    too_long = sum(1 for e in plan.exclusions if e.reason == REASON_TOO_LONG)
    if too_long:
        warnings.append(
            BuildWarning(
                code="over_max_seq_length",
                message=(
                    f"{too_long} record(s) are longer than {config.max_seq_length} tokens and were "
                    "set aside. Raise max_seq_length or set truncate_long_records to keep them."
                ),
            )
        )
    truncated = sum(1 for p in plan.partitions.values() for e in p if e.truncated)
    if truncated:
        warnings.append(
            BuildWarning(
                code="truncated",
                message=(
                    f"{truncated} record(s) are longer than {config.max_seq_length} tokens and will "
                    "be truncated at training time. Truncated answers can teach bad habits."
                ),
            )
        )
    return warnings


def token_stats(values: list[int]) -> TokenStats | None:
    if not values:
        return None
    ordered = sorted(values)

    def rank(p: float) -> int:  # nearest-rank percentile
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return TokenStats(p50=rank(0.5), p90=rank(0.9), p99=rank(0.99), max=ordered[-1])
