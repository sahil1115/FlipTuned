"""Backend registry: the only way other modules obtain a ``TrainingBackend``.

The MLX module is imported only on macOS arm64; elsewhere asking for it fails with the reason.
"""

from __future__ import annotations

import platform
from pathlib import Path

from fliptuned.config.models import Backend
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.errors import NotAvailableError
from fliptuned.models.catalog import CatalogEntry
from fliptuned.models.hub import model_files
from fliptuned.platform.detect import current_os, normalize_arch
from fliptuned.training.backends.base import LoadedModel, TrainingBackend, adapter_backend


def mlx_supported_here() -> bool:
    return current_os() == "macos" and normalize_arch(platform.machine()) == "arm64"


def get_backend(backend: Backend) -> TrainingBackend:
    if backend is Backend.HF_PEFT:
        from fliptuned.training.backends.hf_peft_backend import HfPeftBackend

        return HfPeftBackend()
    if backend is Backend.MLX:
        if not mlx_supported_here():
            raise NotAvailableError(
                "The MLX backend needs a Mac with Apple Silicon.",
                hint="Use backend 'auto' or 'hf_peft' on this machine.",
            )
        from fliptuned.training.backends.mlx_backend import MlxBackend

        return MlxBackend()
    raise ValueError(f"resolve 'auto' with select_backend() first (got {backend!r})")


def load_model(
    entry: CatalogEntry,
    selection: BackendSelection,
    *,
    allow_download: bool = False,
    adapter_path: Path | None = None,
) -> tuple[TrainingBackend, LoadedModel]:
    """Resolve the backend's model files (offline unless allowed) and load them."""
    if adapter_path is not None:
        trained_with = adapter_backend(adapter_path)
        if trained_with is not None and trained_with != selection.backend.value:
            raise NotAvailableError(
                f"This adapter was trained with the {trained_with} backend and can't be loaded "
                f"with {selection.backend.value}.",
                hint=f"Use backend '{trained_with}' for this run (adapters are backend-specific).",
            )
    backend = get_backend(selection.backend)
    folder = model_files(entry, selection.backend.value, allow_download=allow_download)
    return backend, backend.load_for_inference(entry, selection, folder, adapter_path)
