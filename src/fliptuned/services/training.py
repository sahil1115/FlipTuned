"""Training runs (SPEC §11, §13.3): start, resume, and the train job itself.

``start_training`` runs preflight (refusing on any failed check), saves the config as a named
version, creates the run (``<app_home>/runs/<run_id>/``) and submits a ``train`` job, which holds
the model lock. The job calls the selected backend through ``TrainingBackend.train`` and records
everything SPEC §11.6 asks for in ``metadata.json``: effective and generated configs, seeds,
machine, packages, model revision and template checksum, dataset fingerprint, loss curve, best
checkpoint, peak memory and every attempt (resume history).
"""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.config.loader import load_config
from fliptuned.config.models import TrainConfig, TrainingOptions
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.diagnostics.checks import CheckStatus
from fliptuned.errors import ExitCode, FlipTunedError, NotAvailableError
from fliptuned.jobs.state import JobType
from fliptuned.jobs.worker import JobCancelled, JobContext
from fliptuned.models.catalog import CatalogEntry, get_entry
from fliptuned.models.hub import model_files
from fliptuned.models.tokenizer import load_tokenizer
from fliptuned.platform.detect import detect_machine
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate, utc_now
from fliptuned.services.configs import save_config
from fliptuned.services.datasets import read_manifest
from fliptuned.services.jobs import JobInfo, _submit
from fliptuned.services.preflight import PreflightResult, run_preflight
from fliptuned.services.runs import METADATA_FILE, PLAN_FILE
from fliptuned.training.backends import get_backend
from fliptuned.training.backends.base import TrainHooks, TrainResult, TrainSpec
from fliptuned.training.templates import template_checksum

RESUMABLE = ("interrupted", "failed", "cancelled")


class PreflightFailedError(FlipTunedError):
    exit_code = ExitCode.CHECK_FAILED


class TrainRequest(BaseModel):
    config_path: str = Field(description="Training config file (YAML).")
    home: str | None = None
    allow_download: bool = Field(default=False, description="Download model files if missing.")


class TrainStart(BaseModel):
    run_id: str
    config_name: str
    config_version: int
    job: JobInfo
    preflight: PreflightResult


class TrainOutcome(BaseModel):
    run_id: str
    status: str
    result: TrainResult
    metadata_path: str


class RunInfo(BaseModel):
    id: str
    status: str
    model_id: str
    backend: str
    device: str
    precision: str
    dataset_id: str | None
    run_dir: str
    created_at: str
    updated_at: str
    best_eval_loss: float | None = None
    adapter_path: str | None = None


def new_run_id() -> str:
    return f"run_{datetime.now(UTC):%Y%m%d%H%M%S}_{secrets.token_hex(3)}"


def _paths(home: str | None) -> AppPaths:
    return AppPaths(resolve_app_home(home).path)


def preflight(request: TrainRequest) -> PreflightResult:
    config = load_config(Path(request.config_path).expanduser().resolve(), TrainConfig)
    return run_preflight(config, home=request.home, allow_download=request.allow_download)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def start_training(request: TrainRequest, *, detach: bool) -> TrainStart:
    config = load_config(Path(request.config_path).expanduser().resolve(), TrainConfig)
    result = run_preflight(config, home=request.home, allow_download=request.allow_download)
    if not result.ok or result.context is None:
        failed = [c for c in result.checks if c.status is CheckStatus.FAIL]
        raise PreflightFailedError(
            f"Preflight failed: {len(failed)} check(s) must be fixed before training.",
            hint="Fix the checks below, or run `fliptuned train --dry-run` to see them all.",
            details="\n".join(f"- {c.name}: {c.message} Fix: {c.fix}" for c in failed),
        )
    ctx = result.context
    entry = get_entry(ctx.entry_id)
    saved = save_config(request.home, config)
    paths = _paths(request.home)
    run_id = new_run_id()
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True)
    backend = get_backend(ctx.selection.backend)
    manifest = read_manifest(Path(ctx.dataset_dir))
    try:
        checksum = template_checksum(load_tokenizer(entry), entry.chat_template_kwargs)
    except NotAvailableError:
        checksum = None  # tokenizer arrives with the model download; recorded then
    source = entry.sources[ctx.selection.backend.value]
    machine = detect_machine()
    _write_json(
        run_dir / PLAN_FILE,
        {
            "config": config.model_dump(mode="json"),
            "training": ctx.training.model_dump(mode="json"),
            "selection": ctx.selection.model_dump(mode="json"),
            "dataset_id": ctx.dataset_id,
            "dataset_dir": ctx.dataset_dir,
            "system_prompt": ctx.system_prompt,
            "plan": ctx.plan,
        },
    )
    _write_json(
        run_dir / METADATA_FILE,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "config_name": saved.name,
            "config_version": saved.version,
            "preset": config.preset,
            "effective_config": {
                **config.model_dump(mode="json"),
                "training": ctx.training.model_dump(mode="json"),
            },
            "lora": {
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
                "effective_scale": config.lora.alpha / config.lora.rank,
                "note": "PEFT uses lora_alpha / r; MLX receives scale = alpha / rank (C3).",
            },
            "seeds": {"training": ctx.training.seed, "dataset_split": manifest.seed},
            "backend": ctx.selection.backend.value,
            "device": ctx.selection.device,
            "precision": ctx.selection.precision,
            "selection_reason": ctx.selection.reason,
            "artifact_format": backend.artifact_format(),
            "packages": backend.versions(),
            "machine": {
                "os": machine.os_version,
                "arch": machine.arch,
                "python": machine.python_version,
                "cpu": machine.cpu_name,
                "ram_gb": machine.ram_gb,
                "gpus": [g.model_dump() for g in machine.nvidia_gpus],
                "apple_chip": machine.apple_chip,
                "unified_memory_gb": machine.unified_memory_gb,
            },
            "model": {
                "catalog_id": entry.id,
                "repo": source.repo,
                "revision": source.revision,
                "license": entry.license,
                "chat_template_kwargs": entry.chat_template_kwargs,
                "template_checksum": checksum,
                "checked_at": utc_now(),
            },
            "dataset": {
                "id": ctx.dataset_id,
                "fingerprint": manifest.fingerprint,
                "manifest_sha256": _manifest_hash(paths, ctx.dataset_id),
                "partitions": manifest.counts.partitions,
            },
            "plan": ctx.plan,
            "preflight_warnings": [
                f"{c.name}: {c.message}" for c in result.checks if c.status is CheckStatus.WARN
            ],
            "attempts": [],
            "result": None,
            "reproducibility": (
                "Same seeds, data and settings on the same hardware and package versions give "
                "very close results; bit-exact reproducibility across hardware or versions is "
                "not guaranteed."
            ),
        },
    )
    with connect(paths.db) as conn:
        repositories.insert_run(
            conn,
            run_id=run_id,
            dataset_id=ctx.dataset_id,
            config_id=saved.id,
            model_id=entry.id,
            backend=ctx.selection.backend.value,
            device=ctx.selection.device,
            precision=ctx.selection.precision,
            artifact_format=backend.artifact_format(),
            run_dir=str(run_dir),
        )
    params = {"run_id": run_id, "allow_download": request.allow_download, "resume": False}
    try:
        job = _submit(request.home, JobType.TRAIN, params, detach, run_id=run_id)
    except BaseException:
        with connect(paths.db) as conn:
            repositories.delete_run(conn, run_id)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return TrainStart(
        run_id=run_id,
        config_name=saved.name,
        config_version=saved.version,
        job=job,
        preflight=result,
    )


def _manifest_hash(paths: AppPaths, dataset_id: str) -> str | None:
    with connect(paths.db) as conn:
        row = repositories.get_dataset(conn, dataset_id)
    return row["manifest_hash"] if row else None


class ResumeRequest(BaseModel):
    home: str | None = None
    run_id: str
    allow_download: bool = False


def resume_run(request: ResumeRequest, *, detach: bool) -> TrainStart:
    paths = _paths(request.home)
    from fliptuned.services.jobs import reconcile_jobs

    reconcile_jobs(request.home)
    with connect(paths.db) as conn:
        run = repositories.get_run(conn, request.run_id)
    if run is None:
        raise NotAvailableError(f"No run {request.run_id}.", hint="See `fliptuned jobs list`.")
    if run["status"] not in RESUMABLE:
        raise FlipTunedError(
            f"Run {request.run_id} is {run['status']}; only interrupted, failed or cancelled runs "
            "can be resumed.",
            hint="Start a new run with `fliptuned train --config <file>`.",
        )
    backend = get_backend(
        BackendSelection.model_validate(_plan(Path(run["run_dir"]))["selection"]).backend
    )
    if not backend.has_checkpoint(Path(run["run_dir"])):
        raise FlipTunedError(
            f"Run {request.run_id} has no checkpoint to resume from.",
            hint="Start a new run with `fliptuned train --config <file>`.",
        )
    params = {"run_id": request.run_id, "allow_download": request.allow_download, "resume": True}
    job = _submit(request.home, JobType.TRAIN, params, detach, run_id=request.run_id)
    with connect(paths.db) as conn:
        repositories.update_run(conn, request.run_id, status="queued", from_statuses=RESUMABLE)
    plan = _plan(Path(run["run_dir"]))
    meta = json.loads((Path(run["run_dir"]) / METADATA_FILE).read_text(encoding="utf-8"))
    return TrainStart(
        run_id=request.run_id,
        config_name=meta["config_name"],
        config_version=meta["config_version"],
        job=job,
        preflight=PreflightResult(
            ok=True,
            checks=[],
            dataset_id=plan["dataset_id"],
            model=plan["config"]["model"]["catalog_id"],
            selection=BackendSelection.model_validate(plan["selection"]),
            plan=plan["plan"],
            effective_training=plan["training"],
        ),
    )


def _plan(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / PLAN_FILE).read_text(encoding="utf-8"))


def list_runs(home: str | None, limit: int = 50) -> list[RunInfo]:
    paths = _paths(home)
    paths.ensure()
    migrate(paths.db)
    with connect(paths.db) as conn:
        ids = [r["id"] for r in repositories.list_runs(conn, limit)]
    return [get_run(home, run_id) for run_id in ids]


def get_run(home: str | None, run_id: str) -> RunInfo:
    paths = _paths(home)
    with connect(paths.db) as conn:
        row = repositories.get_run(conn, run_id)
    if row is None:
        raise NotAvailableError(f"No run {run_id}.")
    meta = json.loads(row["metadata_json"] or "{}")
    result = meta.get("result") or {}
    return RunInfo(
        id=row["id"],
        status=row["status"],
        model_id=row["model_id"],
        backend=row["backend"],
        device=row["device"],
        precision=row["precision"],
        dataset_id=row["dataset_id"],
        run_dir=row["run_dir"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        best_eval_loss=result.get("best_eval_loss"),
        adapter_path=result.get("adapter_path"),
    )


# --------------------------------------------------------------------------- the train job


class _JobHooks(TrainHooks):
    def __init__(self, ctx: JobContext):
        self.ctx = ctx

    def emit(self, event_type: str, **fields: Any) -> None:
        if event_type == "progress":
            self.ctx.progress(fields["done"], fields["total"], fields.get("message"))
        else:
            self.ctx.emit(event_type, **fields)

    def cancel_requested(self) -> bool:
        return self.ctx.cancel_requested()


def execute_training(ctx: JobContext) -> dict[str, Any]:
    """Handler of the ``train`` job."""
    paths = ctx.paths
    run_id = ctx.params["run_id"]
    resume = bool(ctx.params.get("resume"))
    with connect(paths.db) as conn:
        run = repositories.get_run(conn, run_id)
        repositories.update_run(conn, run_id, status="running", from_statuses=("queued",))
    run_dir = Path(run["run_dir"])
    plan = _plan(run_dir)
    config = TrainConfig.model_validate(plan["config"])
    training = TrainingOptions.model_validate(plan["training"])
    selection = BackendSelection.model_validate(plan["selection"])
    entry = get_entry(config.model.catalog_id)
    backend = get_backend(selection.backend)
    attempt: dict[str, Any] = {
        "job_id": ctx.job_id,
        "resume": resume,
        "started_at": utc_now(),
    }
    status, result = "failed", None
    try:
        model_dir = model_files(
            entry, selection.backend.value, allow_download=bool(ctx.params.get("allow_download"))
        )
        spec = TrainSpec(
            config=config,
            training=training,
            entry=entry,
            selection=selection,
            model_dir=model_dir,
            dataset_dir=Path(plan["dataset_dir"]),
            run_dir=run_dir,
            system_prompt=plan["system_prompt"],
            resume=resume,
        )
        result = backend.train(spec, _JobHooks(ctx))
        status = "succeeded" if result.status == "completed" else "cancelled"
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    finally:
        attempt.update(finished_at=utc_now(), status=status)
        if result is not None:
            attempt.update(steps_done=result.steps_done, resume_note=result.resume_note)
        _record(paths, run_dir, run_id, status, attempt, result, entry)
    if status == "cancelled":
        raise JobCancelled
    assert result is not None
    return TrainOutcome(
        run_id=run_id, status=status, result=result, metadata_path=str(run_dir / METADATA_FILE)
    ).model_dump(mode="json")


def _record(
    paths: AppPaths,
    run_dir: Path,
    run_id: str,
    status: str,
    attempt: dict[str, Any],
    result: TrainResult | None,
    entry: CatalogEntry,
) -> None:
    meta_path = run_dir / METADATA_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["attempts"].append(attempt)
    meta["status"] = status
    if result is not None:
        meta["result"] = result.model_dump(mode="json")
        meta["backend_config_file"] = Path(result.backend_config).name
    if meta["model"].get("template_checksum") is None:
        try:
            tok = load_tokenizer(entry)
            meta["model"]["template_checksum"] = template_checksum(tok, entry.chat_template_kwargs)
        except NotAvailableError:
            pass
    _write_json(meta_path, meta)
    summary = {k: meta[k] for k in ("status", "backend", "device", "precision", "result")}
    with connect(paths.db) as conn:
        repositories.update_run(conn, run_id, status=status, metadata=summary)
