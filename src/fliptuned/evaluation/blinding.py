"""Blind paired review (SPEC §12.2): which model's answer is shown as A.

The mapping is drawn once per item when the evaluation is generated and stored in the registry
(``eval_items.a_model``). Review payloads and the review sheet carry only "A" and "B"; the
mapping is read back only when scoring.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Model = Literal["base", "tuned"]
Judgement = Literal["A", "B", "tie"]
Outcome = Literal["tuned", "base", "tie"]

_WORDS: dict[str, Judgement] = {
    "a": "A",
    "1": "A",
    "b": "B",
    "2": "B",
    "tie": "tie",
    "t": "tie",
    "3": "tie",
    "=": "tie",
}


def draw_a_model(seed: int, evaluation_id: str, item_id: str) -> Model:
    """Seeded coin flip per item; different evaluations get different orders."""
    digest = hashlib.sha256(f"{seed}\0{evaluation_id}\0{item_id}".encode()).digest()
    return "tuned" if digest[0] & 1 else "base"


def parse_judgement(word: str) -> Judgement | None:
    """``A``/``B``/``tie``, also ``1``/``2``/``3`` (the review keyboard shortcuts)."""
    return _WORDS.get(word.strip().lower())


def unblind(judgement: Judgement, a_model: Model) -> Outcome:
    """Which model won this pair."""
    if judgement == "tie":
        return "tie"
    if judgement == "A":
        return a_model
    return "base" if a_model == "tuned" else "tuned"
