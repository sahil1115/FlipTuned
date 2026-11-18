"""Statistics for the verdict (SPEC §12.3)."""

from __future__ import annotations

import math
from fractions import Fraction


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test on non-tied pairs (H0: wins and losses equally likely).

    p = min(1, 2 · P(X ≤ min(wins, losses))) with X ~ Binomial(wins + losses, 0.5), computed with
    exact integers. No pairs gives p = 1.
    """
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**n)))


_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def win_rate_interval(wins: int, losses: int) -> tuple[float, float, float] | None:
    """(win rate, low, high): the fine-tuned model's share of non-tied pairs with its 95%
    Wilson score interval. None when there are no non-tied pairs."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    n = wins + losses
    if n == 0:
        return None
    p = wins / n
    z2 = _Z_95 * _Z_95
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = _Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)
