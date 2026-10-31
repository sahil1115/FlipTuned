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
