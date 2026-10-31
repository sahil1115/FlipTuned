"""Grouped split: determinism, tolerance, infeasibility, fairness."""

from __future__ import annotations

import hashlib
import json

import pytest

from fliptuned.dataset.split import PARTITIONS, SplitInfeasibleError, split_groups

RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def _groups() -> dict[str, int]:
    sizes = {f"single/{i:03d}": 1 for i in range(120)}
    sizes |= {f"group/{i}": 2 + i % 5 for i in range(25)}
    sizes |= {"big/a": 12, "big/b": 10}
    return sizes


def _digest(assignment: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(assignment.items())).encode()).hexdigest()


def test_deterministic_and_independent_of_insertion_order() -> None:
    sizes = _groups()
    first = split_groups(sizes, RATIOS, seed=42)
    reversed_sizes = dict(reversed(list(sizes.items())))
    assert split_groups(reversed_sizes, RATIOS, seed=42).assignment == first.assignment
    assert split_groups(dict(sorted(sizes.items())), RATIOS, 42).assignment == first.assignment


# Pinned on Windows (2026-10-29); CI checks the same values on Linux and macOS.
GOLDEN_COUNTS = {"train": 194, "valid": 24, "test": 24}
GOLDEN_DIGEST = "ae030dd54cc4269a37d7fdd66b1cdac2dfc5b3f52115cea487bfe14f920b475a"


def test_golden_assignment() -> None:
    """The same input must split identically on every OS and Python build.
    If the algorithm changes on purpose, bump BUILDER_VERSION and update the pinned values."""
    result = split_groups(_groups(), RATIOS, seed=42)
    assert result.counts == GOLDEN_COUNTS
    assert _digest(result.assignment) == GOLDEN_DIGEST


def test_shares_within_tolerance_and_seed_matters() -> None:
    sizes = _groups()
    total = sum(sizes.values())
    a = split_groups(sizes, RATIOS, seed=1)
    b = split_groups(sizes, RATIOS, seed=2)
    assert a.assignment != b.assignment
    for result in (a, b):
        assert sum(result.counts.values()) == total
        for p in PARTITIONS:
            assert abs(result.counts[p] / total - RATIOS[p]) <= 0.05
            assert result.counts[p] == sum(sizes[k] for k, v in result.assignment.items() if v == p)


def test_large_groups_can_land_in_valid_and_test() -> None:
    """Unlike largest-first placement, big groups are not always sent to train."""
    placements = {split_groups(_groups(), RATIOS, seed=s).assignment["big/a"] for s in range(60)}
    assert placements == set(PARTITIONS)


def test_infeasible_split_fails_with_details() -> None:
    sizes = {"huge": 90} | {f"s{i}": 1 for i in range(10)}
    with pytest.raises(SplitInfeasibleError) as info:
        split_groups(sizes, RATIOS, seed=42)
    assert "±5 points" in info.value.message
    assert "largest groups (records): 90" in (info.value.details or "")


@pytest.mark.parametrize("sizes", [{"a": 1, "b": 1}, {"a": 5, "b": 5}, {}])
def test_too_few_records_or_groups(sizes: dict[str, int]) -> None:
    with pytest.raises(SplitInfeasibleError, match="Not enough approved records"):
        split_groups(sizes, RATIOS, seed=42)


def test_small_but_feasible_dataset() -> None:
    sizes = {f"s{i}": 1 for i in range(12)}
    result = split_groups(sizes, RATIOS, seed=3)
    assert result.counts == {"train": 10, "valid": 1, "test": 1}
