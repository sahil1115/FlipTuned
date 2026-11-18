"""Word overlap between an answer and the approved reference answer (token F1).

A wording-similarity signal only: it says nothing about correctness and never enters the
verdict. Pure Python; holds no text.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TOKEN = re.compile(r"\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def token_f1(answer: str, reference: str) -> float:
    """Harmonic mean of token precision and recall (multiset overlap). 1.0 = same tokens."""
    a, r = _tokens(answer), _tokens(reference)
    if not a or not r:
        return 1.0 if a == r else 0.0
    overlap = sum((Counter(a) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def mean_overlap(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """Mean token F1 of the base and fine-tuned answers over rows that have a reference."""
    scored = [r for r in rows if isinstance(r.get("reference"), str) and r["reference"].strip()]
    if not scored:
        return None
    return {
        "items": len(scored),
        "base": sum(token_f1(r["base"]["output"], r["reference"]) for r in scored) / len(scored),
        "tuned": sum(token_f1(r["tuned"]["output"], r["reference"]) for r in scored) / len(scored),
    }
