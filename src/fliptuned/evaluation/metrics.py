"""Evaluation metrics (SPEC §12.2) computed from ``results.jsonl`` rows and judgements.

A results row is ``{set, item_id, position, category, question, reference, base, tuned}``, where
``base`` and ``tuned`` each hold ``{output, finish_reason, prompt_tokens, completion_tokens,
seconds, checks: [{name, passed, detail}], passed}``. Metrics hold counts and IDs, never text.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from fliptuned.evaluation.blinding import Outcome
from fliptuned.evaluation.verdict import CategoryResult

SIDES = ("base", "tuned")
UNCATEGORIZED = "(none)"


class CheckCount(BaseModel):
    applicable: int
    passed: int


class SideRates(BaseModel):
    items: int
    passed_items: int = Field(description="Items where every applicable check passed.")
    pass_rate: float = Field(description="Percent of items where every applicable check passed.")
    checks: dict[str, CheckCount]


class CheckSummary(BaseModel):
    items: int
    base: SideRates
    tuned: SideRates
    drop_pts: float = Field(description="Base pass rate minus fine-tuned pass rate, in points.")


class ItemRegression(BaseModel):
    set: Literal["test", "regression"]
    item_id: str
    failed_checks: list[str] = Field(
        description="Checks the fine-tuned answer failed and the base answer passed."
    )


class GenerationStats(BaseModel):
    items: int
    finish_reasons: dict[str, int]
    completion_tokens: int
    seconds: float


class PriorExposure(BaseModel):
    earlier_evaluations: list[str]
    shared_items: int
    note: str | None


class EvaluationMetrics(BaseModel):
    evaluation_id: str
    run_id: str
    dataset_id: str
    test_items: int
    judged: int
    review_coverage: float
    wins: int = Field(description="Pairs the fine-tuned model won.")
    losses: int = Field(description="Pairs the base model won.")
    ties: int
    judgements_by_origin: dict[str, int]
    sign_test_p: float
    categories: list[CategoryResult]
    deterministic: CheckSummary
    regression_set: CheckSummary | None
    regressions: list[ItemRegression]
    generation: dict[str, GenerationStats]
    prior_exposure: PriorExposure
    limited_to: int | None = Field(description="Only the first N test items were evaluated.")


def _rate(passed: int, total: int) -> float:
    return 100.0 * passed / total if total else 0.0


def side_rates(rows: list[dict[str, Any]], side: str) -> SideRates:
    checks: dict[str, CheckCount] = {}
    for row in rows:
        for check in row[side]["checks"]:
            count = checks.setdefault(check["name"], CheckCount(applicable=0, passed=0))
            count.applicable += 1
            count.passed += bool(check["passed"])
    passed_items = sum(bool(r[side]["passed"]) for r in rows)
    return SideRates(
        items=len(rows),
        passed_items=passed_items,
        pass_rate=_rate(passed_items, len(rows)),
        checks=dict(sorted(checks.items())),
    )


def check_summary(rows: list[dict[str, Any]]) -> CheckSummary:
    base, tuned = side_rates(rows, "base"), side_rates(rows, "tuned")
    return CheckSummary(
        items=len(rows), base=base, tuned=tuned, drop_pts=base.pass_rate - tuned.pass_rate
    )


def regressions(rows: list[dict[str, Any]]) -> list[ItemRegression]:
    found = []
    for row in rows:
        base_ok = {c["name"] for c in row["base"]["checks"] if c["passed"]}
        failed = [c["name"] for c in row["tuned"]["checks"] if not c["passed"]]
        lost = [name for name in failed if name in base_ok]
        if lost:
            found.append(ItemRegression(set=row["set"], item_id=row["item_id"], failed_checks=lost))
    return found


def generation_stats(rows: list[dict[str, Any]], side: str) -> GenerationStats:
    return GenerationStats(
        items=len(rows),
        finish_reasons=dict(sorted(Counter(r[side]["finish_reason"] for r in rows).items())),
        completion_tokens=sum(r[side]["completion_tokens"] for r in rows),
        seconds=round(sum(r[side]["seconds"] for r in rows), 2),
    )


def category_results(
    categories: dict[str, str | None], outcomes: dict[str, Outcome]
) -> list[CategoryResult]:
    """``categories``: item → category; ``outcomes``: judged item → winner."""
    by_category: dict[str, Counter] = {}
    for item_id, category in categories.items():
        counter = by_category.setdefault(category or UNCATEGORIZED, Counter())
        counter["items"] += 1
        if item_id in outcomes:
            counter["judged"] += 1
            counter[outcomes[item_id]] += 1
    return [
        CategoryResult(
            category=name,
            items=c["items"],
            judged=c["judged"],
            wins=c["tuned"],
            losses=c["base"],
            ties=c["tie"],
        )
        for name, c in sorted(by_category.items())
    ]
