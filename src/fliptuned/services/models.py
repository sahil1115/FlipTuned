"""``models list``: the curated catalog enriched for this machine (SPEC §7).

Never touches the network: cached state is read from the local Hugging Face cache, and memory
figures are catalog estimates compared against this machine's memory pool.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from fliptuned.config.models import TrainingOptions
from fliptuned.diagnostics.backend_select import BackendSelection, EnvironmentProbe
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.models.catalog import load_catalog
from fliptuned.models.compat import Compatibility, backends_here, compatibility, select_for_entry
from fliptuned.models.hub import is_cached
from fliptuned.platform import detect
from fliptuned.services.doctor import default_probe

ESTIMATE_NOTE = (
    "Memory figures are catalog estimates of peak training memory at default settings, "
    "not measurements of this machine."
)
DOWNLOAD_MARGIN_GB = 2.0  # spare space kept free on the model-cache drive


class ModelListRequest(BaseModel):
    include_test_only: bool = Field(
        default=False, description="Also list tiny test-only models (developer use)."
    )


class ModelSummary(BaseModel):
    id: str
    display_name: str
    params_b: float
    license: str
    model_card_url: str
    test_only: bool
    trainable: bool = Field(description="Template prefix test passed (SPEC §11.4).")
    prefix_test_verified: str
    backends_here: list[str] = Field(description="Backends on this machine with a version of it.")
    selection: BackendSelection | None = Field(description="Backend 'auto' would use for it.")
    compatibility: Compatibility | None
    cached: bool = Field(description="Model files for the selected backend are downloaded.")
    cached_by_backend: dict[str, bool]
    download_gb: float | None = Field(description="Download size for the selected backend.")
    notes: list[str]


class ModelListResult(BaseModel):
    models: list[ModelSummary]
    estimate_note: str = ESTIMATE_NOTE


def list_models(
    request: ModelListRequest, probe_fn: Callable[[], EnvironmentProbe] | None = None
) -> ModelListResult:
    probe = (probe_fn or default_probe)()
    cpu_limit = TrainingOptions().hf_peft.cpu_max_params_b
    out = []
    for entry in load_catalog().models:
        if entry.test_only and not request.include_test_only:
            continue
        notes: list[str] = []
        try:
            selection = select_for_entry(probe, entry)
        except NotAvailableError as exc:
            selection = None
            notes.append(exc.message)
        compat = compatibility(entry, selection, probe) if selection else None
        source = entry.sources.get(selection.backend.value) if selection else None
        if not entry.trainable:
            notes.append("Its chat template failed the prefix test, so it can't be trained.")
        if selection and selection.device == "cpu" and entry.params_b > cpu_limit:
            notes.append(
                f"Larger than the CPU training limit ({cpu_limit:g}B parameters); "
                "preflight will block CPU training."
            )
        cached_by = {name: is_cached(src) for name, src in entry.sources.items()}
        out.append(
            ModelSummary(
                id=entry.id,
                display_name=entry.display_name,
                params_b=entry.params_b,
                license=entry.license,
                model_card_url=entry.model_card_url,
                test_only=entry.test_only,
                trainable=entry.trainable,
                prefix_test_verified=entry.prefix_test.verified,
                backends_here=backends_here(entry, probe),
                selection=selection,
                compatibility=compat,
                cached=bool(selection and cached_by.get(selection.backend.value)),
                cached_by_backend=cached_by,
                download_gb=source.download_gb if source else None,
                notes=notes,
            )
        )
    return ModelListResult(models=out)


# --------------------------------------------------------------------------- download


class ModelDownloadRequest(BaseModel):
    home: str | None = None
    model: str = Field(description="Catalog model ID.")
    backend: str | None = Field(
        default=None, description="'hf_peft' or 'mlx'. Empty = the backend auto would use here."
    )


def start_model_download(request: ModelDownloadRequest, *, detach: bool):
    """Download a model's files (the only network access besides installs; SPEC §3.2) as a job."""
    from fliptuned.config.models import Backend
    from fliptuned.jobs.state import JobType
    from fliptuned.models.catalog import get_entry
    from fliptuned.services.jobs import _submit

    entry = get_entry(request.model)
    if request.backend is None:
        backend = select_for_entry(
            default_probe(), entry, Backend.AUTO, "auto", "auto"
        ).backend.value
    else:
        backend = request.backend
    if backend not in entry.sources:
        raise NotAvailableError(
            f"Model '{entry.id}' has no {backend} version.", hint="Pick another backend."
        )
    source = entry.sources[backend]
    if not is_cached(source):
        _, free = detect.free_disk_gb(detect.hf_cache_dir())
        needed = source.download_gb + DOWNLOAD_MARGIN_GB
        if free is not None and free < needed:
            raise FlipTunedError(
                f"Not enough free disk space to download '{entry.id}': about {needed:g} GB "
                f"needed, {free:.1f} GB free on the model-cache drive.",
                hint="Free up space, or move the model cache to a bigger drive with the "
                "HF_HOME environment variable (see docs/platforms.md).",
            )
    params = {"model": entry.id, "backend": backend}
    return _submit(request.home, JobType.MODEL_DOWNLOAD, params, detach)


def execute_model_download(ctx) -> dict:
    """Handler of the ``model_download`` job."""
    from fliptuned.models.catalog import get_entry
    from fliptuned.models.hub import model_files

    entry = get_entry(ctx.params["model"])
    backend = ctx.params["backend"]
    ctx.progress(0, 1, f"downloading {entry.id} ({backend})")
    folder = model_files(entry, backend, allow_download=True)
    ctx.progress(1, 1, "downloaded")
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return {"model": entry.id, "backend": backend, "bytes": size}
