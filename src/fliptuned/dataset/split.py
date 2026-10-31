"""Grouped, deterministic train/valid/test split (SPEC §9.2, §10.2).

Whole groups are assigned, so no group spans partitions. Groups are shuffled by a seeded sha256
of the group key (never Python's ``hash`` or input order) and laid end to end; a group goes to the
partition whose cumulative range holds its midpoint. Every group, large or small, thus has the
same chance of landing in each partition, so valid and test are not left with only singletons.
Single-group moves and two-group swaps then pull the shares towards the targets until nothing
improves. The search is greedy, not exhaustive: a failure means no split within ``tolerance``
was found.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fliptuned.errors import ExitCode, FlipTunedError
from fliptuned.records.hashing import sha256_hex

PARTITIONS = ("train", "valid", "test")
DEFAULT_TOLERANCE = 0.05
_MAX_ROUNDS = 10_000


class SplitInfeasibleError(FlipTunedError):
    exit_code = ExitCode.CHECK_FAILED


@dataclass
class SplitResult:
    assignment: dict[str, str]  # group key -> partition
    counts: dict[str, int]
    shares: dict[str, float]
    group_counts: dict[str, int] = field(default_factory=dict)


def group_order_key(seed: int, group_key: str) -> str:
    return sha256_hex(str(seed), group_key)


def _objective(
    counts: Mapping[str, int], total: int, ratios: Mapping[str, float]
) -> tuple[int, float, float]:
    deviations = [abs(counts[p] / total - ratios[p]) for p in PARTITIONS]
    empty = sum(1 for p in PARTITIONS if counts[p] == 0)
    return empty, round(max(deviations), 12), round(sum(d * d for d in deviations), 12)


def split_groups(
    sizes: Mapping[str, int],
    ratios: Mapping[str, float],
    seed: int,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SplitResult:
    total = sum(sizes.values())
    if total < len(PARTITIONS) or len(sizes) < len(PARTITIONS):
        raise SplitInfeasibleError(
            f"Not enough approved records to split: {total} record(s) in {len(sizes)} group(s).",
            hint="Approve more records (fliptuned review export / review import), then build again.",
        )
    order = sorted(sizes, key=lambda k: group_order_key(seed, k))
    bounds, running = [], 0.0
    for p in PARTITIONS:
        running += ratios[p]
        bounds.append(running * total)
    counts = dict.fromkeys(PARTITIONS, 0)
    assignment: dict[str, str] = {}
    position = 0
    for key in order:
        midpoint = position + sizes[key] / 2
        index = next((i for i, b in enumerate(bounds) if midpoint < b), len(PARTITIONS) - 1)
        assignment[key] = PARTITIONS[index]
        counts[PARTITIONS[index]] += sizes[key]
        position += sizes[key]

    _improve(order, sizes, assignment, counts, total, ratios)

    shares = {p: counts[p] / total for p in PARTITIONS}
    result = SplitResult(
        assignment=assignment,
        counts=counts,
        shares=shares,
        group_counts={p: sum(1 for v in assignment.values() if v == p) for p in PARTITIONS},
    )
    empty, worst, _ = _objective(counts, total, ratios)
    if empty or worst > tolerance + 1e-9:
        largest = sorted(sizes.values(), reverse=True)[:5]
        raise SplitInfeasibleError(
            f"Could not split {total} records in {len(sizes)} groups within ±{tolerance * 100:.0f} "
            "points of the target ratios.",
            hint=(
                "Approve more records, split very large groups (check the group/thread columns and "
                "group_by_source_file), or change the split ratios."
            ),
            details="\n".join(
                [
                    "target: " + ", ".join(f"{p} {ratios[p]:.0%}" for p in PARTITIONS),
                    "best found: "
                    + ", ".join(f"{p} {counts[p]} ({shares[p]:.1%})" for p in PARTITIONS),
                    f"largest groups (records): {', '.join(str(s) for s in largest)}",
                ]
            ),
        )
    return result


def _improve(
    order: list[str],
    sizes: Mapping[str, int],
    assignment: dict[str, str],
    counts: dict[str, int],
    total: int,
    ratios: Mapping[str, float],
) -> None:
    """Local search over moves and swaps. Groups of equal size in one partition are
    interchangeable, so candidates are enumerated per distinct (partition, size)."""
    for _ in range(_MAX_ROUNDS):
        members: dict[str, dict[int, str]] = {p: {} for p in PARTITIONS}
        for key in order:  # the last key in seeded order represents each (partition, size)
            members[assignment[key]][sizes[key]] = key
        current = _objective(counts, total, ratios)
        best: tuple[tuple[int, float, float], list[tuple[str, str]]] | None = None

        for src in PARTITIONS:
            for size, key in sorted(members[src].items()):
                for dst in PARTITIONS:
                    if dst == src:
                        continue
                    trial = dict(counts)
                    trial[src] -= size
                    trial[dst] += size
                    score = _objective(trial, total, ratios)
                    if score < current and (best is None or score < best[0]):
                        best = (score, [(key, dst)])
        for i, pa in enumerate(PARTITIONS):
            for pb in PARTITIONS[i + 1 :]:
                for size_a, key_a in sorted(members[pa].items()):
                    for size_b, key_b in sorted(members[pb].items()):
                        if size_a == size_b:
                            continue
                        trial = dict(counts)
                        trial[pa] += size_b - size_a
                        trial[pb] += size_a - size_b
                        score = _objective(trial, total, ratios)
                        if score < current and (best is None or score < best[0]):
                            best = (score, [(key_a, pb), (key_b, pa)])
        if best is None:
            return
        for key, dst in best[1]:
            counts[assignment[key]] -= sizes[key]
            counts[dst] += sizes[key]
            assignment[key] = dst
