"""Preflight (SPEC §6): every check before training, each ``pass`` / ``warn`` / ``fail`` with a fix.

The UI renders these checks as they are, and Start stays disabled while any check fails.
Memory and time figures are estimates and say so.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fliptuned.config.models import Backend, TrainConfig, TrainingOptions
from fliptuned.diagnostics.backend_select import BackendSelection, EnvironmentProbe
from fliptuned.diagnostics.checks import Check, CheckStatus
from fliptuned.errors import ConfigError, NotAvailableError
from fliptuned.models.catalog import CatalogEntry, get_entry
from fliptuned.models.compat import compatibility, select_for_entry
from fliptuned.models.hub import is_cached
from fliptuned.models.tokenizer import load_tokenizer
from fliptuned.platform.detect import free_disk_gb
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.datasets import (
    ValidateDatasetRequest,
    locate_dataset,
    read_manifest,
    validate_dataset,
)
from fliptuned.services.doctor import default_probe
from fliptuned.services.review import open_registry
from fliptuned.training.plan import StepPlan, plan_steps, resolve_training
from fliptuned.training.templates import prefix_test

CHECKPOINT_ALLOWANCE_GB = 1.0  # adapters and optimizer state for small LoRA runs
DISK_WARN_MARGIN_GB = 10.0
# Rough CPU cost, seconds per example per billion parameters at seq 512 (decisions.md §5:
# Qwen2.5-Coder-0.5B took 11–14 s per step on a 4-core laptop CPU).
CPU_SECONDS_PER_EXAMPLE_PER_B = 25.0
PREFIX_SAMPLES = 5


class PreflightContext(BaseModel):
    """What training needs once preflight passed (internal; not shown to users)."""

    model_config = {"arbitrary_types_allowed": True}

    dataset_id: str
    dataset_dir: str
    system_prompt: str
    entry_id: str
    selection: BackendSelection
    training: TrainingOptions
    plan: dict[str, Any]


class PreflightResult(BaseModel):
    ok: bool = Field(description="No check failed (warnings allowed).")
    checks: list[Check]
    dataset_id: str | None
    model: str
    selection: BackendSelection | None
    plan: dict[str, Any] | None = Field(description="Derived update / iteration counts.")
    effective_training: dict[str, Any] | None = Field(description="Preset applied.")
    context: PreflightContext | None = Field(default=None, exclude=True)


def _check(name: str, status: CheckStatus, message: str, fix: str | None = None) -> Check:
    return Check(name=name, status=status, message=message, fix=fix if status != "pass" else None)


def _hours(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{max(minutes, 1)} min"


def run_preflight(
    config: TrainConfig,
    *,
    home: str | None,
    allow_download: bool = False,
    probe_fn: Callable[[], EnvironmentProbe] | None = None,
) -> PreflightResult:
    checks: list[Check] = [_check("config", CheckStatus.PASS, f"Config '{config.name}' is valid.")]
    P, W, F = CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL
    paths = open_registry(home)

    # ---- dataset
    dataset_id = manifest = folder = None
    with connect(paths.db) as conn:
        wanted = config.dataset_id or next(
            (r["id"] for r in repositories.list_datasets(conn)), None
        )
    if wanted is None:
        checks.append(
            _check("dataset", F, "No dataset has been built.", "Run `fliptuned build-dataset`.")
        )
    else:
        try:
            dataset_id, folder, _ = locate_dataset(paths, wanted)
            manifest = read_manifest(folder)
        except NotAvailableError as exc:
            checks.append(_check("dataset", F, exc.message, exc.hint))
    if manifest is not None:
        report = validate_dataset(ValidateDatasetRequest(home=home, dataset=dataset_id))
        failed = [c.name for c in report.checks if c.status is F]
        checks.append(
            _check(
                "dataset",
                F if failed else P,
                f"Dataset {dataset_id} ({manifest.counts.partitions['train']} train, "
                f"{manifest.counts.partitions['valid']} valid, "
                f"{manifest.counts.partitions['test']} test)"
                + (f" failed validation: {', '.join(failed)}." if failed else " is valid."),
                f"Run `fliptuned validate-dataset {dataset_id}` and rebuild the dataset.",
            )
        )
        for c in report.checks:
            if c.name in ("train_size", "test_size"):
                checks.append(_check(f"dataset_{c.name}", c.status, c.message, c.fix))
        checks.append(
            _check(
                "system_prompt",
                P if config.system_prompt == manifest.system_prompt else F,
                "The system prompt matches the dataset's."
                if config.system_prompt == manifest.system_prompt
                else "The system prompt differs from the one the dataset was built with.",
                "Use the dataset's system prompt (chat uses the same text), or rebuild the "
                "dataset with this one.",
            )
        )

    # ---- model
    entry: CatalogEntry | None = None
    try:
        entry = get_entry(config.model.catalog_id)
    except ConfigError as exc:
        checks.append(_check("model", F, exc.message, exc.hint))
    if entry is not None:
        if not entry.trainable:
            checks.append(
                _check(
                    "model",
                    F,
                    f"'{entry.id}' failed the chat-template prefix test, so it can't be trained.",
                    "Pick another model from `fliptuned models list`.",
                )
            )
        elif entry.test_only:
            checks.append(
                _check(
                    "model",
                    W,
                    f"'{entry.id}' is a tiny test model; its results mean nothing.",
                    "Use it only to check that training works.",
                )
            )
        else:
            checks.append(_check("model", P, f"{entry.display_name} ({entry.license})."))
        if manifest is not None and manifest.model.get("catalog_id") != entry.id:
            checks.append(
                _check(
                    "tokenizer_match",
                    W,
                    f"The dataset measured lengths with '{manifest.model.get('catalog_id')}', "
                    f"not '{entry.id}'.",
                    "Rebuild the dataset with model_catalog_id set to this model.",
                )
            )

    # ---- backend, memory, CPU limit, files, disk
    selection: BackendSelection | None = None
    training: TrainingOptions | None = None
    probe = (probe_fn or default_probe)()
    if entry is not None:
        try:
            selection = select_for_entry(
                probe, entry, config.backend, config.device, config.precision
            )
            checks.append(
                _check(
                    "backend",
                    P,
                    f"{selection.backend.value} on {selection.device} ({selection.precision}): "
                    f"{selection.reason}.",
                )
            )
        except NotAvailableError as exc:
            checks.append(_check("backend", F, exc.message, exc.hint))
    if entry is not None and selection is not None:
        training = resolve_training(config, selection.backend)
        compat = compatibility(entry, selection, probe)
        estimate = (
            f"estimated {compat.estimate_gb:g} GB of {compat.pool_gb:.1f} GB ({compat.pool})"
            if compat.estimate_gb is not None and compat.pool_gb is not None
            else "no estimate for this backend and precision"
        )
        status = {"fits": P, "tight": W, "won't fit": F, "unknown": W}[compat.label]
        checks.append(
            _check(
                "memory",
                status,
                f"Memory: {compat.label} ({estimate}). This is an estimate.",
                "Pick a smaller model, lower training.max_seq_length, or use a GPU / 4-bit QLoRA.",
            )
        )
        if selection.device == "cpu" and entry.params_b > training.hf_peft.cpu_max_params_b:
            checks.append(
                _check(
                    "cpu_model_size",
                    F,
                    f"'{entry.id}' has {entry.params_b:g}B parameters, above the CPU training "
                    f"limit of {training.hf_peft.cpu_max_params_b:g}B.",
                    "Use a model of about 1B parameters or less on CPU, train on a GPU, or raise "
                    "training.hf_peft.cpu_max_params_b if you accept a very long run.",
                )
            )
        source = entry.sources[selection.backend.value]
        cached = is_cached(source)
        if cached:
            checks.append(_check("model_files", P, "Model files are downloaded."))
        elif allow_download:
            checks.append(
                _check(
                    "model_files",
                    W,
                    f"Model files (about {source.download_gb:g} GB) will be downloaded first.",
                    "Nothing to do; downloads are allowed for this run.",
                )
            )
        else:
            checks.append(
                _check(
                    "model_files",
                    F,
                    f"The model files (about {source.download_gb:g} GB) are not downloaded.",
                    "Allow the download: --allow-download on the command line, or the download option in the UI.",
                )
            )
        need = (0.0 if cached else source.download_gb) + CHECKPOINT_ALLOWANCE_GB
        _, free = free_disk_gb(paths.home)
        if free is None:
            checks.append(_check("disk", W, "Free disk space is unknown.", "Check it manually."))
        else:
            status = F if free < need else W if free < need + DISK_WARN_MARGIN_GB else P
            checks.append(
                _check(
                    "disk",
                    status,
                    f"{free:.1f} GB free; this run needs about {need:.1f} GB.",
                    "Free up disk space on the app home drive.",
                )
            )

    # ---- template prefix test on real samples
    if entry is not None and folder is not None:
        try:
            tokenizer = load_tokenizer(entry, allow_download=False)
            text = (Path(folder) / "train.jsonl").read_text(encoding="utf-8")
            samples = [json.loads(line)["messages"] for line in text.split("\n") if line]
            result = (
                prefix_test(tokenizer, entry.chat_template_kwargs, samples[:PREFIX_SAMPLES] or None)
                if samples
                else None
            )
            if result is not None:
                checks.append(
                    _check(
                        "template",
                        P if result.passed else F,
                        f"Chat-template prefix test on {result.samples} training examples "
                        + ("passed." if result.passed else "failed."),
                        "This model's template can't mask prompts reliably; pick another model.",
                    )
                )
        except NotAvailableError as exc:
            checks.append(_check("template", W if allow_download else F, exc.message, exc.hint))

    # ---- steps and CPU time
    plan: StepPlan | None = None
    if manifest is not None and selection is not None and training is not None:
        longest = max(
            (
                stats.max
                for name, stats in manifest.tokens.items()
                if name in ("train", "valid") and stats is not None
            ),
            default=0,
        )
        if longest <= training.max_seq_length:
            checks.append(
                _check(
                    "max_seq_length",
                    P,
                    f"Every training example fits in {training.max_seq_length} tokens "
                    f"(longest: {longest}).",
                )
            )
        else:
            opted_in = (
                manifest.truncate_long_records
                and training.max_seq_length >= manifest.max_seq_length
            )
            checks.append(
                _check(
                    "max_seq_length",
                    W if opted_in else F,
                    f"The longest training example has {longest} tokens, but "
                    f"training.max_seq_length is {training.max_seq_length}: longer examples "
                    "are cut off mid-answer.",
                    "Rebuild the dataset without truncate_long_records to set long records aside."
                    if opted_in
                    else f"Set training.max_seq_length to {manifest.max_seq_length} "
                    "(the dataset's value) or higher.",
                )
            )
        try:
            plan = plan_steps(selection.backend, manifest.counts.partitions["train"], training)
            checks.append(
                _check(
                    "steps",
                    P,
                    f"{plan.total_updates} optimizer updates at most "
                    f"({plan.updates_per_epoch} per epoch); {plan.note}.",
                )
            )
            if training.eval_every_steps > plan.total_updates:
                checks.append(
                    _check(
                        "validation",
                        W,
                        f"Validation runs every {training.eval_every_steps} updates, but the run "
                        f"has only {plan.total_updates}: no best checkpoint can be chosen.",
                        f"Set training.eval_every_steps to {max(1, plan.total_updates // 4)} or less.",
                    )
                )
            if selection.device == "cpu" and entry is not None:
                examples = (
                    plan.total_updates * training.grad_accumulation_steps * training.batch_size
                )
                seconds = examples * max(entry.params_b, 0.002) * CPU_SECONDS_PER_EXAMPLE_PER_B
                checks.append(
                    _check(
                        "cpu_time",
                        W,
                        f"Training on the CPU: roughly {_hours(seconds)} at most "
                        "(a rough estimate from a 4-core laptop; early stopping can end sooner).",
                        "Use a GPU for anything beyond small experiments.",
                    )
                )
        except ConfigError as exc:
            checks.append(_check("steps", F, exc.message, exc.hint))

    ok = not any(c.status is F for c in checks)
    context = None
    if ok and manifest is not None and entry is not None and selection is not None and plan:
        context = PreflightContext(
            dataset_id=dataset_id,
            dataset_dir=str(folder),
            system_prompt=manifest.system_prompt,
            entry_id=entry.id,
            selection=selection,
            training=training,
            plan=plan.__dict__,
        )
    return PreflightResult(
        ok=ok,
        checks=checks,
        dataset_id=dataset_id,
        model=config.model.catalog_id,
        selection=selection,
        plan=plan.__dict__ if plan else None,
        effective_training=training.model_dump(mode="json") if training else None,
        context=context,
    )


__all__ = ["Backend", "PreflightResult", "run_preflight"]
