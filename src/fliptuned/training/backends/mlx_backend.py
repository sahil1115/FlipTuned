"""``mlx`` backend: mlx-lm on Apple Silicon (SPEC §11.3).

The backend registry only hands this out on macOS arm64; mlx and mlx-lm are imported inside
methods. API used (mlx-lm 0.32.2, verified from source, decisions.md §3.1 and §15):
``mlx_lm.load(path, adapter_path=...)`` returns ``(model, TokenizerWrapper)``;
``mlx_lm.stream_generate(model, tokenizer, prompt_ids, max_tokens=...)`` yields responses with
``text``, ``prompt_tokens``, ``generation_tokens`` and ``finish_reason`` ("stop" / "length"),
decoding greedily when no sampler is passed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fliptuned.config.models import Backend
from fliptuned.diagnostics.backend_select import (
    BackendOption,
    BackendSelection,
    EnvironmentProbe,
)
from fliptuned.errors import NotAvailableError
from fliptuned.models.catalog import CatalogEntry
from fliptuned.training.backends.base import (
    ADAPTER_DIR,
    BackendCapabilities,
    GenerationConfig,
    GenerationResult,
    LoadedModel,
    MergeResult,
    Messages,
    TrainHooks,
    TrainingBackend,
    TrainResult,
    TrainSpec,
    write_adapter_marker,
)
from fliptuned.training.plan import BestTracker, StepPlan, plan_steps
from fliptuned.training.templates import prompt_ids, template_checksum


class MlxBackend(TrainingBackend):
    name = Backend.MLX

    def _capabilities(
        self, probe: EnvironmentProbe, options: list[BackendOption]
    ) -> BackendCapabilities:
        available = any(o.available for o in options)
        return BackendCapabilities(
            backend=self.name,
            available=available,
            devices=["apple_gpu"] if available else [],
            precisions=["model (4-bit → QLoRA)"] if available else [],
            qlora=available,
            reasons=[f"{o.device}: {o.reason}" for o in options],
        )

    def artifact_format(self) -> str:
        return "mlx_lora_safetensors"

    def versions(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for name in ("mlx", "mlx-lm", "transformers"):
            try:
                out[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                out[name] = None
        return out

    def load_for_inference(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path | None = None,
    ) -> LoadedModel:
        try:
            from mlx_lm import load
        except ImportError:
            raise NotAvailableError(
                "mlx-lm is not installed, so the MLX backend can't run.",
                hint="Run `fliptuned doctor --suggest-install` for the install command.",
            ) from None
        try:
            model, tokenizer = load(
                str(model_dir), adapter_path=str(adapter_path) if adapter_path else None
            )
        except (OSError, ValueError) as exc:
            raise NotAvailableError(
                f"Model '{entry.id}' could not be loaded with MLX.",
                hint="Allow downloads to fetch the model files again: "
                "--allow-download, or the download option in the UI.",
                details=f"{type(exc).__name__}: {exc}",
            ) from None
        return LoadedModel(
            backend=self.name,
            entry=entry,
            selection=selection,
            model_dir=model_dir,
            adapter_path=adapter_path,
            model=model,
            tokenizer=tokenizer,
            template_checksum=template_checksum(tokenizer, entry.chat_template_kwargs),
        )

    def merge(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path,
        out_dir: Path,
    ) -> MergeResult:
        """``mlx_lm fuse`` (mlx-lm 0.32.2 ``fuse.py``): loads the model with the adapter, fuses
        every LoRA layer and saves model, config and tokenizer. A 4-bit base stays 4-bit (no
        ``--dequantize``). Output goes to ``out_dir/../mlx_fuse.log``."""
        log_path = out_dir.parent / "mlx_fuse.log"
        argv = [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(model_dir),
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(out_dir),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            done = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
        if done.returncode != 0:
            raise NotAvailableError(
                "mlx-lm could not merge the adapter into the model.",
                hint=f"See the log: {log_path}",
            )
        return MergeResult(
            path=str(out_dir),
            precision=selection.precision,
            notes=["Merged with mlx_lm fuse; a 4-bit base stays 4-bit."],
        )

    def _responses(self, loaded: LoadedModel, messages: Messages, config: GenerationConfig):
        from mlx_lm import stream_generate

        ids = prompt_ids(loaded.tokenizer, messages, loaded.template_kwargs)
        return stream_generate(
            loaded.model, loaded.tokenizer, ids, max_tokens=config.max_new_tokens
        )

    def generate(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> GenerationResult:
        start = time.perf_counter()
        pieces: list[str] = []
        last = None
        for response in self._responses(loaded, messages, config):
            pieces.append(response.text)
            last = response
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=last.prompt_tokens if last else 0,
            completion_tokens=last.generation_tokens if last else 0,
            finish_reason="length" if last and last.finish_reason == "length" else "stop",
            seconds=round(time.perf_counter() - start, 3),
        )

    def stream(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> Iterator[str]:
        for response in self._responses(loaded, messages, config):
            if response.text:
                yield response.text

    def has_checkpoint(self, run_dir: Path) -> bool:
        train_dir = run_dir / TRAIN_DIR
        return (train_dir / LATEST_FILE).exists() and (train_dir / STATE_FILE).exists()

    def train(self, spec: TrainSpec, hooks: TrainHooks) -> TrainResult:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            raise NotAvailableError(
                "mlx-lm is not installed, so the MLX backend can't train.",
                hint="Run `fliptuned doctor --suggest-install` for the install command.",
            ) from None
        return _mlx_train(self, spec, hooks)


# --------------------------------------------------------------------------- training

TRAIN_DIR = "mlx_train"
CONFIG_FILE = "mlx_lora.yaml"
STATE_FILE = "fliptuned_state.json"
BEST_FILE = "best_adapters.safetensors"
LATEST_FILE = "adapters.safetensors"


def mlx_lora_config(
    spec: TrainSpec, plan: StepPlan, *, iters: int, resume_file: str | None
) -> dict[str, Any]:
    """The YAML config for mlx-lm's LoRA trainer (keys verified in decisions.md §3.1).

    ``report_to`` is never written (no remote trackers). LoRA ``scale`` = alpha / rank so the
    adapter strength matches PEFT (C3). Validation and checkpoints share one interval in
    iterations (C6).
    """
    t, lora = spec.training, spec.config.lora
    return {
        "model": str(spec.model_dir),
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "data": str(spec.dataset_dir),
        "seed": t.seed,
        "num_layers": t.mlx.num_layers,
        "batch_size": t.batch_size,
        "iters": iters,
        "val_batches": -1,
        "learning_rate": t.learning_rate,
        "steps_per_report": t.grad_accumulation_steps,
        "steps_per_eval": plan.steps_per_eval,
        "save_every": plan.steps_per_eval,
        "resume_adapter_file": resume_file,
        "adapter_path": str(spec.run_dir / TRAIN_DIR),
        "max_seq_length": t.max_seq_length,
        "grad_checkpoint": t.gradient_checkpointing,
        "grad_accumulation_steps": t.grad_accumulation_steps,
        "lora_parameters": {
            "rank": lora.rank,
            "dropout": lora.dropout,
            "scale": lora.alpha / lora.rank,
        },
        "mask_prompt": True,
        "test": False,
    }


class _Stop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _read_state(train_dir: Path) -> dict[str, Any]:
    path = train_dir / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_state(train_dir: Path, state: dict[str, Any]) -> None:
    (train_dir / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _mlx_train(backend: MlxBackend, spec: TrainSpec, hooks: TrainHooks) -> TrainResult:
    import mlx.core as mx
    import yaml
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import ChatDataset

    training = spec.training
    accum = training.grad_accumulation_steps
    template_kwargs = dict(spec.entry.chat_template_kwargs)
    train_dir = spec.run_dir / TRAIN_DIR
    train_dir.mkdir(parents=True, exist_ok=True)

    def rows(partition: str) -> list[dict[str, Any]]:
        text = (spec.dataset_dir / f"{partition}.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.split("\n") if line]

    train_rows, valid_rows = rows("train"), rows("valid")
    plan = plan_steps(Backend.MLX, len(train_rows), training)
    assert plan.iterations is not None and plan.steps_per_eval is not None

    state = _read_state(train_dir) if spec.resume else {}
    done = int(state.get("completed_iters", 0))
    resume_file = str(train_dir / LATEST_FILE) if spec.resume and done else None
    remaining = max(1, plan.iterations - done)
    config = mlx_lora_config(spec, plan, iters=remaining, resume_file=resume_file)
    config_path = spec.run_dir / CONFIG_FILE
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    # The saved YAML is the single source of settings, on top of mlx-lm's own defaults.
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**{**CONFIG_DEFAULTS, **saved})

    model, tokenizer = load(str(spec.model_dir))

    class TemplateChatDataset(ChatDataset):
        """The mlx-lm ChatDataset, but with the catalog chat-template kwargs (C8)."""

        def process(self, d):
            messages = d[self.chat_key]
            tokens = self.tokenizer.apply_chat_template(
                messages, return_dict=False, **template_kwargs
            )
            offset = len(
                self.tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, return_dict=False, **template_kwargs
                )
            )
            return (tokens, offset)

    train_set = TemplateChatDataset(train_rows, tokenizer, mask_prompt=True)
    valid_set = TemplateChatDataset(valid_rows, tokenizer, mask_prompt=True)
    tracker = BestTracker(patience=training.patience)
    for step, loss in state.get("history", []):
        tracker.update(step, loss)
    curve: list[dict[str, Any]] = list(state.get("curve", []))
    peak = {"gb": float(state.get("peak_memory_gb", 0.0))}

    def save(path: Path) -> None:
        mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))

    def persist(completed: int) -> None:
        _write_state(
            train_dir,
            {
                "completed_iters": completed,
                "history": tracker.history,
                "curve": curve,
                "peak_memory_gb": peak["gb"],
                "total_iters": plan.iterations,
            },
        )

    class Callback(TrainingCallback):
        def on_val_loss_report(self, info: dict) -> None:
            # Reported before iteration `it` updates the weights, so the current weights are
            # the ones evaluated; saving them now keeps exactly the scored adapter (C4).
            absolute = done + int(info["iteration"])
            loss = float(info["val_loss"])
            is_best, stop = tracker.update(absolute // accum, loss)
            if is_best:
                save(train_dir / BEST_FILE)
            curve.append({"step": absolute // accum, "eval_loss": loss})
            hooks.emit(
                "train",
                step=absolute // accum,
                max_steps=plan.total_updates,
                eval_loss=loss,
                loss=None,
                learning_rate=None,
                eta_s=None,
                memory_gb=peak["gb"] or None,
            )
            persist(absolute)
            if stop:
                save(train_dir / LATEST_FILE)
                raise _Stop("early_stop")

        def on_train_loss_report(self, info: dict) -> None:
            absolute = done + int(info["iteration"])
            peak["gb"] = max(peak["gb"], float(info.get("peak_memory", 0.0)))
            point = {
                "step": absolute // accum,
                "loss": float(info["train_loss"]),
                "learning_rate": float(info["learning_rate"]),
            }
            curve.append(point)
            its = float(info.get("iterations_per_second") or 0.0)
            hooks.emit(
                "train",
                max_steps=plan.total_updates,
                eval_loss=None,
                eta_s=round((plan.iterations - absolute) / its, 1) if its else None,
                memory_gb=round(peak["gb"], 3),
                **point,
            )
            hooks.emit("progress", done=absolute // accum, total=plan.total_updates, message=None)
            if absolute % plan.steps_per_eval == 0:
                persist(absolute)  # mlx-lm saves adapters.safetensors right after this report
            if hooks.cancel_requested():
                save(train_dir / LATEST_FILE)
                persist(absolute)
                raise _Stop("cancelled")

    stopped = None
    try:
        train_model(args, model, train_set, valid_set, training_callback=Callback())
        persist(plan.iterations)
    except _Stop as exc:
        stopped = exc.reason

    completed = int(_read_state(train_dir).get("completed_iters", 0))
    cancelled = stopped == "cancelled"
    adapter_dir = None
    if not cancelled:
        adapter_dir = spec.run_dir / ADAPTER_DIR
        adapter_dir.mkdir(parents=True, exist_ok=True)
        best = train_dir / BEST_FILE
        source = best if best.exists() else train_dir / LATEST_FILE
        shutil.copyfile(source, adapter_dir / LATEST_FILE)
        shutil.copyfile(train_dir / "adapter_config.json", adapter_dir / "adapter_config.json")
        write_adapter_marker(adapter_dir, backend, spec)
    return TrainResult(
        status="cancelled" if cancelled else "completed",
        adapter_path=str(adapter_dir) if adapter_dir else None,
        best_checkpoint=str(train_dir / BEST_FILE) if tracker.best_step is not None else None,
        best_step=tracker.best_step,
        best_eval_loss=tracker.best_loss,
        steps_done=completed // accum,
        total_steps=plan.total_updates,
        stopped_early=stopped == "early_stop",
        peak_memory_gb=round(peak["gb"], 3) if peak["gb"] else None,
        loss_curve=curve,
        backend_config=str(config_path),
        resume_note=(
            f"Warm restart from iteration {done}: adapter weights restored; optimizer state, "
            "learning-rate schedule position and data order restart (mlx-lm "
            "resume_adapter_file, decisions.md C9). Ran the remaining iterations only."
            if resume_file
            else None
        ),
        notes=[plan.note],
    )
