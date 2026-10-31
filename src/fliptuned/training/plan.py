"""Backend-level training plan: presets, step/iteration counts, best checkpoint, early stop.

Pure functions only, so every rule is unit-tested without torch or mlx.

Units (decisions.md C6, C7): ``eval_every_steps`` counts optimizer updates on every backend.
Transformers counts updates natively. mlx-lm counts micro-batches ("iterations"): one
iteration is one batch of ``batch_size`` examples, the optimizer updates every
``grad_accumulation_steps`` iterations, and an epoch is ``floor(N_train / batch_size)``
iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from fliptuned.config.models import Backend, TrainConfig, TrainingOptions
from fliptuned.errors import ConfigError

# Presets per backend (SPEC §11.5). Only the fields listed change; anything the user wrote under
# ``training:`` overrides the preset. "standard" is the TrainingOptions defaults.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    Backend.HF_PEFT.value: {
        "quick": {"max_epochs": 1, "patience": 2, "eval_every_steps": 10},
        "standard": {},
        "thorough": {"max_epochs": 5, "patience": 5, "learning_rate": 5.0e-5},
    },
    Backend.MLX.value: {
        "quick": {"max_epochs": 1, "patience": 2, "eval_every_steps": 10, "mlx": {"num_layers": 4}},
        "standard": {},
        "thorough": {
            "max_epochs": 5,
            "patience": 5,
            "learning_rate": 5.0e-5,
            "mlx": {"num_layers": 16},
        },
    },
}


def resolve_training(config: TrainConfig, backend: Backend) -> TrainingOptions:
    """Effective training options: defaults ← backend preset ← fields the user set."""
    preset = PRESETS[backend.value][config.preset]
    merged: dict[str, Any] = TrainingOptions().model_dump()
    for key, value in preset.items():
        if isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    explicit = config.training
    for key in explicit.model_fields_set:
        value = getattr(explicit, key)
        if hasattr(value, "model_fields_set"):  # nested options: only their explicit fields
            merged[key] = {**merged[key], **value.model_dump(include=value.model_fields_set)}
        else:
            merged[key] = value
    return TrainingOptions.model_validate(merged)


# --------------------------------------------------------------------------- counts


@dataclass(frozen=True)
class StepPlan:
    backend: str
    train_examples: int
    updates_per_epoch: int
    total_updates: int
    eval_every_updates: int
    # mlx only (micro-batch iterations); None for hf_peft
    iterations: int | None = None
    steps_per_eval: int | None = None
    note: str = ""


def plan_steps(backend: Backend, n_train: int, training: TrainingOptions) -> StepPlan:
    b, accum, epochs = training.batch_size, training.grad_accumulation_steps, training.max_epochs
    if n_train < b:
        raise ConfigError(
            f"The training set has {n_train} examples, fewer than batch_size {b}.",
            hint="Lower training.batch_size or approve more records.",
        )
    if backend is Backend.MLX:
        iters_per_epoch = n_train // b
        iterations = max(1, math.floor(epochs * iters_per_epoch))
        return StepPlan(
            backend=backend.value,
            train_examples=n_train,
            updates_per_epoch=max(1, iters_per_epoch // accum),
            total_updates=max(1, iterations // accum),
            eval_every_updates=training.eval_every_steps,
            iterations=iterations,
            steps_per_eval=training.eval_every_steps * accum,
            note=(
                f"iters = floor({epochs:g} epochs × floor({n_train} / {b})) = {iterations}; "
                f"validation and checkpoints every {training.eval_every_steps * accum} iterations"
            ),
        )
    batches = math.ceil(n_train / b)
    updates_per_epoch = max(1, math.ceil(batches / accum))
    return StepPlan(
        backend=backend.value,
        train_examples=n_train,
        updates_per_epoch=updates_per_epoch,
        total_updates=max(1, math.ceil(epochs * updates_per_epoch)),
        eval_every_updates=training.eval_every_steps,
        note=f"about {updates_per_epoch} optimizer updates per epoch (ceiling {epochs:g} epochs)",
    )


# --------------------------------------------------------------------------- best / early stop


@dataclass
class BestTracker:
    """Keeps the best validation loss and decides when to stop early.

    Stop after ``patience`` evaluations in a row without an improvement larger than
    ``min_delta`` (the same rule as Transformers' EarlyStoppingCallback with threshold 0).
    """

    patience: int
    min_delta: float = 0.0
    best_loss: float | None = None
    best_step: int | None = None
    bad_evals: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def update(self, step: int, loss: float) -> tuple[bool, bool]:
        """Record an evaluation. Returns ``(is_new_best, should_stop)``."""
        self.history.append((step, loss))
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_step, self.bad_evals = loss, step, 0
            return True, False
        self.bad_evals += 1
        return False, self.bad_evals >= self.patience


def select_best(evals: list[tuple[int, float]]) -> tuple[int, float] | None:
    """Best (step, loss) from a validation log; ties go to the earlier step."""
    if not evals:
        return None
    return min(evals, key=lambda e: (e[1], e[0]))
