"""The ``TrainingBackend`` interface (SPEC §11.1).

Evaluation, chat, export, the API and the CLI never call mlx-lm, Transformers or PEFT directly;
they go through this interface. Capabilities, loading, unloading, generation and training are
implemented, and merging for export. Preflight lives in the preflight service
and data preparation inside each backend's ``train`` (decisions.md P6-1).
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fliptuned.config.models import Backend
from fliptuned.diagnostics.backend_select import (
    BackendOption,
    BackendSelection,
    EnvironmentProbe,
    list_options,
)
from fliptuned.models.catalog import CatalogEntry

Messages = list[dict[str, str]]


class BackendCapabilities(BaseModel):
    backend: Backend
    available: bool
    devices: list[str] = Field(description="Devices this backend can use on this machine.")
    precisions: list[str]
    qlora: bool
    reasons: list[str] = Field(description="Why each device is or isn't available.")


class GenerationConfig(BaseModel):
    """Always greedy, so results are repeatable."""

    max_new_tokens: int = Field(default=512, ge=1, le=8192)


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    seconds: float


@dataclass
class LoadedModel:
    """A model (plus optional adapter) ready to generate. Holds backend-specific objects."""

    backend: Backend
    entry: CatalogEntry
    selection: BackendSelection
    model_dir: Path
    adapter_path: Path | None
    model: Any
    tokenizer: Any
    template_checksum: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def template_kwargs(self) -> dict[str, Any]:
        return dict(self.entry.chat_template_kwargs)


class TrainingBackend(ABC):
    name: Backend

    def capabilities(self, probe: EnvironmentProbe) -> BackendCapabilities:
        options = [o for o in list_options(probe) if o.backend is self.name]
        return self._capabilities(probe, options)

    @abstractmethod
    def _capabilities(
        self, probe: EnvironmentProbe, options: list[BackendOption]
    ) -> BackendCapabilities: ...

    @abstractmethod
    def load_for_inference(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path | None = None,
    ) -> LoadedModel: ...

    def unload(self, loaded: LoadedModel) -> None:
        """Release a loaded model, so the next one fits (evaluation loads base, then fine-tuned)."""
        loaded.model = None
        loaded.tokenizer = None
        loaded.extra.clear()
        gc.collect()

    @abstractmethod
    def generate(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> GenerationResult:
        """Greedy generation of one reply to ``messages`` (which end with a user turn)."""

    @abstractmethod
    def stream(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> Iterator[str]:
        """Like :meth:`generate`, yielding text pieces as they are produced."""

    @abstractmethod
    def artifact_format(self) -> str: ...

    @abstractmethod
    def versions(self) -> dict[str, str | None]:
        """Package versions recorded in run and evaluation metadata."""

    @abstractmethod
    def train(self, spec: TrainSpec, hooks: TrainHooks) -> TrainResult:
        """Train a LoRA adapter on ``spec.dataset_dir`` into ``spec.run_dir``.

        Writes the backend config it generated, checkpoints for resume, and the best adapter
        to ``spec.run_dir / "adapter"`` (with :data:`ADAPTER_MARKER`). Honours
        ``hooks.cancel_requested()`` by saving a resumable checkpoint and returning a result
        with ``status="cancelled"``.
        """

    @abstractmethod
    def has_checkpoint(self, run_dir: Path) -> bool:
        """True if ``run_dir`` holds a checkpoint this backend can resume from."""

    @abstractmethod
    def merge(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path,
        out_dir: Path,
    ) -> MergeResult:
        """Write a standalone model (base + adapter merged, with tokenizer) to ``out_dir``."""


class MergeResult(BaseModel):
    path: str
    precision: str = Field(description="Precision of the merged weights (how to load them).")
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- training contract

ADAPTER_DIR = "adapter"
ADAPTER_MARKER = "fliptuned_adapter.json"


@dataclass
class TrainSpec:
    """Everything a backend needs for one training attempt (fresh or resumed)."""

    config: Any  # fliptuned.config.models.TrainConfig, with effective training options
    training: Any  # fliptuned.config.models.TrainingOptions (preset applied)
    entry: CatalogEntry
    selection: BackendSelection
    model_dir: Path
    dataset_dir: Path
    run_dir: Path
    system_prompt: str
    resume: bool = False


class TrainHooks:
    """Callbacks from a training loop to the job: events and cancel. Subclassed by the job."""

    def emit(self, event_type: str, **fields: Any) -> None:  # pragma: no cover - interface
        pass

    def cancel_requested(self) -> bool:  # pragma: no cover - interface
        return False


class TrainResult(BaseModel):
    status: Literal["completed", "cancelled"]
    adapter_path: str | None = Field(description="Best adapter (completed runs only).")
    best_checkpoint: str | None
    best_step: int | None = Field(description="Optimizer update of the best validation loss.")
    best_eval_loss: float | None
    steps_done: int = Field(description="Optimizer updates completed so far.")
    total_steps: int
    stopped_early: bool
    peak_memory_gb: float | None
    loss_curve: list[dict[str, Any]] = Field(
        description="Train and validation loss points: {step, loss?, eval_loss?, learning_rate?}."
    )
    backend_config: str = Field(description="Path of the generated backend config file.")
    resume_note: str | None = Field(
        default=None, description="What a resume restored and what it didn't."
    )
    notes: list[str] = Field(default_factory=list)


def adapter_backend(adapter_path: Path) -> str | None:
    """Backend recorded in an adapter folder's marker, if any."""
    import json

    marker = adapter_path / ADAPTER_MARKER
    if not marker.exists():
        return None
    return json.loads(marker.read_text(encoding="utf-8")).get("backend")


def write_adapter_marker(adapter_dir: Path, backend: TrainingBackend, spec: TrainSpec) -> None:
    """Record which backend and base model an adapter belongs to (SPEC §5.1)."""
    import json

    source = spec.entry.sources[spec.selection.backend.value]
    marker = {
        "backend": spec.selection.backend.value,
        "artifact_format": backend.artifact_format(),
        "device": spec.selection.device,
        "precision": spec.selection.precision,
        "model": spec.entry.id,
        "repo": source.repo,
        "revision": source.revision,
    }
    (adapter_dir / ADAPTER_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
