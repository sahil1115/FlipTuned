"""Per-machine model compatibility (SPEC §5.3, §7). Pure functions of a probe; all estimates.

Memory pools: CUDA uses free VRAM; MLX uses unified memory minus OS headroom; CPU uses system RAM
minus OS headroom. A model "fits" when its estimated peak training memory is at most 80% of the
pool, is "tight" up to 100%, and "won't fit" above that.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from fliptuned.config.models import Backend, Device, Precision
from fliptuned.diagnostics.backend_select import (
    BackendSelection,
    EnvironmentProbe,
    list_options,
    select_backend,
)
from fliptuned.models.catalog import CatalogEntry

OS_HEADROOM_GB = 4.0
FITS_SHARE = 0.8

Label = Literal["fits", "tight", "won't fit", "unknown"]


class Compatibility(BaseModel):
    label: Label
    estimate_gb: float | None
    pool_gb: float | None
    pool: str  # e.g. "free VRAM", "unified memory - 4 GB", "RAM - 4 GB"
    is_estimate: bool = True


def memory_key(selection: BackendSelection) -> str:
    if selection.backend is Backend.MLX:
        return "mlx"
    return f"hf_peft:{selection.device}:{selection.precision}"


def memory_pool(probe: EnvironmentProbe, selection: BackendSelection) -> tuple[float | None, str]:
    m = probe.machine
    if selection.backend is Backend.MLX:
        total = m.unified_memory_gb or m.ram_gb
        return max(total - OS_HEADROOM_GB, 0.0), f"unified memory - {OS_HEADROOM_GB:g} GB"
    if selection.device == "cuda":
        gpu = m.nvidia_gpus[0] if m.nvidia_gpus else None
        if gpu is None:
            return None, "free VRAM"
        free = gpu.memory_free_gb if gpu.memory_free_gb is not None else gpu.memory_total_gb
        return free, "free VRAM"
    return max(m.ram_gb - OS_HEADROOM_GB, 0.0), f"RAM - {OS_HEADROOM_GB:g} GB"


def compatibility(
    entry: CatalogEntry, selection: BackendSelection, probe: EnvironmentProbe
) -> Compatibility:
    estimate = entry.est_train_memory_gb.get(memory_key(selection))
    pool, pool_name = memory_pool(probe, selection)
    if estimate is None or pool is None:
        label: Label = "unknown"
    elif estimate <= FITS_SHARE * pool:
        label = "fits"
    elif estimate <= pool:
        label = "tight"
    else:
        label = "won't fit"
    return Compatibility(label=label, estimate_gb=estimate, pool_gb=pool, pool=pool_name)


def backends_here(entry: CatalogEntry, probe: EnvironmentProbe) -> list[str]:
    """Backends that are available on this machine and have a source for this model."""
    available = {o.backend.value for o in list_options(probe) if o.available}
    return [b for b in ("mlx", "hf_peft") if b in entry.sources and b in available]


def select_for_entry(
    probe: EnvironmentProbe,
    entry: CatalogEntry,
    backend: Backend = Backend.AUTO,
    device: Device = Device.AUTO,
    precision: Precision = Precision.AUTO,
) -> BackendSelection:
    """SPEC §5.2 selection, except that 'auto' skips a backend the model has no version for
    (e.g. the CPU-only tiny test model on a Mac)."""
    selection = select_backend(probe, backend, device, precision)
    if backend is Backend.AUTO and selection.backend.value not in entry.sources:
        fallback = select_backend(probe, Backend.HF_PEFT, device, precision)
        return fallback.model_copy(
            update={
                "reason": f"{fallback.reason}; '{entry.id}' has no {selection.backend.value} version"
            }
        )
    return selection
