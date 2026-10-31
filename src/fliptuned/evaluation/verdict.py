"""Verdict logic (SPEC §12.3), applied in order:

1. **inconclusive** if test items < ``min_test_items``, review coverage < ``min_review_coverage``,
   or non-tied pairs < ``min_non_tied_pairs``;
2. **rejected** if the fine-tuned model doesn't have more wins than losses, the sign-test p-value
   is ≥ alpha, any qualifying category's (losses − wins) / items exceeds
   ``max_category_net_loss_share``, or the deterministic or regression-set pass rate drops by more
   than its limit;
3. **accepted** otherwise.

Every criterion is evaluated and reported, also those that didn't decide the verdict. A category
qualifies with at least ``CATEGORY_MIN_ITEMS`` reviewed items, and its share is taken over those
reviewed items (equal to all of its items at full review coverage).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from fliptuned.config.models import AcceptanceCriteria
from fliptuned.evaluation.stats import sign_test

CATEGORY_MIN_ITEMS = 5
_EPS = 1e-9

VerdictName = Literal["accepted", "rejected", "inconclusive"]


class CategoryResult(BaseModel):
    category: str
    items: int = Field(description="Test items in this category.")
    judged: int
    wins: int = Field(description="Pairs the fine-tuned model won.")
    losses: int
    ties: int

    @property
    def net_loss_share(self) -> float | None:
        return (self.losses - self.wins) / self.judged if self.judged else None


class VerdictInputs(BaseModel):
    test_items: int
    judged: int
    wins: int = Field(description="Pairs the fine-tuned model won.")
    losses: int = Field(description="Pairs the base model won.")
    ties: int
    categories: list[CategoryResult]
    deterministic_pass_base: float = Field(description="Percent of test items passing all checks.")
    deterministic_pass_tuned: float
    regression_pass_base: float | None = Field(
        default=None, description="Percent of regression-set items passing; None if not run."
    )
    regression_pass_tuned: float | None = None


class Criterion(BaseModel):
    name: str
    stage: Literal["inconclusive", "rejected"] = Field(
        description="The verdict this criterion gives when it isn't met."
    )
    description: str
    value: float | None
    threshold: float | None
    met: bool
    applicable: bool = True

    def _display(self, number: float | None) -> str:
        if number is None:
            return "not run"
        if self.name == "review_coverage" or self.name.startswith("category_net_loss"):
            return f"{number:.0%}"
        if self.name == "sign_test":
            return f"p = {number:.4g}"
        if self.name.endswith("_drop"):
            return f"{number:+.1f} points"
        if self.name == "more_wins_than_losses":
            return f"{number:+g} (wins − losses)"
        return f"{number:g}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_value(self) -> str:
        return self._display(self.value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_threshold(self) -> str:
        return "—" if self.threshold is None else self._display(self.threshold)


class Verdict(BaseModel):
    verdict: VerdictName
    reasons: list[str] = Field(description="The unmet criteria that decided the verdict.")
    review_coverage: float
    non_tied_pairs: int
    sign_test_p: float
    criteria: list[Criterion]
    inputs: VerdictInputs
    acceptance: AcceptanceCriteria


def decide(inputs: VerdictInputs, acceptance: AcceptanceCriteria) -> Verdict:
    a = acceptance
    coverage = inputs.judged / inputs.test_items if inputs.test_items else 0.0
    non_tied = inputs.wins + inputs.losses
    p_value = sign_test(inputs.wins, inputs.losses)
    criteria = [
        Criterion(
            name="test_items",
            stage="inconclusive",
            description=f"At least {a.min_test_items} test items",
            value=inputs.test_items,
            threshold=a.min_test_items,
            met=inputs.test_items >= a.min_test_items,
        ),
        Criterion(
            name="review_coverage",
            stage="inconclusive",
            description=f"At least {a.min_review_coverage:.0%} of test pairs reviewed",
            value=coverage,
            threshold=a.min_review_coverage,
            met=coverage + _EPS >= a.min_review_coverage,
        ),
        Criterion(
            name="non_tied_pairs",
            stage="inconclusive",
            description=f"At least {a.min_non_tied_pairs} reviews picked a winner",
            value=non_tied,
            threshold=a.min_non_tied_pairs,
            met=non_tied >= a.min_non_tied_pairs,
        ),
        Criterion(
            name="more_wins_than_losses",
            stage="rejected",
            description="The fine-tuned model wins more pairs than it loses",
            value=inputs.wins - inputs.losses,
            threshold=0,
            met=inputs.wins > inputs.losses,
        ),
        Criterion(
            name="sign_test",
            stage="rejected",
            description=f"Sign-test p-value below {a.sign_test_alpha:g}",
            value=p_value,
            threshold=a.sign_test_alpha,
            met=p_value < a.sign_test_alpha,
        ),
    ]
    for cat in inputs.categories:
        share = cat.net_loss_share
        if cat.judged < CATEGORY_MIN_ITEMS or share is None:
            continue
        criteria.append(
            Criterion(
                name=f"category_net_loss:{cat.category}",
                stage="rejected",
                description=(
                    f"Category '{cat.category}': (losses − wins) / reviewed items at most "
                    f"{a.max_category_net_loss_share:.0%}"
                ),
                value=share,
                threshold=a.max_category_net_loss_share,
                met=share <= a.max_category_net_loss_share + _EPS,
            )
        )
    drop = inputs.deterministic_pass_base - inputs.deterministic_pass_tuned
    criteria.append(
        Criterion(
            name="deterministic_drop",
            stage="rejected",
            description=(
                f"Automatic checks pass rate drops by at most {a.max_deterministic_drop_pts:g} "
                "points"
            ),
            value=drop,
            threshold=a.max_deterministic_drop_pts,
            met=drop <= a.max_deterministic_drop_pts + _EPS,
        )
    )
    ran_regression = (
        inputs.regression_pass_base is not None and inputs.regression_pass_tuned is not None
    )
    regression_drop = (
        inputs.regression_pass_base - inputs.regression_pass_tuned  # type: ignore[operator]
        if ran_regression
        else None
    )
    criteria.append(
        Criterion(
            name="regression_drop",
            stage="rejected",
            description=(
                "General coding prompts pass rate drops by at most "
                f"{a.max_regression_set_drop_pts:g} points"
            ),
            value=regression_drop,
            threshold=a.max_regression_set_drop_pts,
            met=regression_drop is None or regression_drop <= a.max_regression_set_drop_pts + _EPS,
            applicable=ran_regression,
        )
    )

    verdict: VerdictName = "accepted"
    for stage in ("inconclusive", "rejected"):
        unmet = [c for c in criteria if c.stage == stage and c.applicable and not c.met]
        if unmet:
            verdict = stage  # type: ignore[assignment]
            reasons = [f"Not met: {c.description}." for c in unmet]
            break
    else:
        reasons = []
    return Verdict(
        verdict=verdict,
        reasons=reasons,
        review_coverage=coverage,
        non_tied_pairs=non_tied,
        sign_test_p=p_value,
        criteria=criteria,
        inputs=inputs,
        acceptance=acceptance,
    )
