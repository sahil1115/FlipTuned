"""``hf_peft`` backend: Transformers + PEFT (+ TRL for training) on CUDA or CPU (SPEC §11.2).

torch, peft and trl are imported inside methods, so this module loads (and reports itself
unavailable) on machines where they aren't installed. Loading is always local-only: files come
from ``models.hub``, which handles the offline default and ``--allow-download``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from threading import Event, Thread
from typing import Any

import psutil
from pydantic import BaseModel

from fliptuned.config.models import Backend, Precision
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
    ContextTooLongError,
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
from fliptuned.training.templates import template_checksum

_PACKAGES = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes")


class MaskCheckResult(BaseModel):
    passed: bool
    prompt_tokens: int
    masked_prompt_tokens: int
    completion_tokens: int
    labelled_completion_tokens: int


def prompt_completion_row(messages: Messages, template_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Canonical messages → TRL conversational prompt–completion row (decisions.md §3.2, C8)."""
    row: dict[str, Any] = {"prompt": messages[:-1], "completion": messages[-1:]}
    if template_kwargs:
        row["chat_template_kwargs"] = dict(template_kwargs)
    return row


CHECKPOINTS_DIR = "checkpoints"
ARGS_FILE = "hf_training_args.json"


def sft_config_kwargs(
    training: Any, selection: BackendSelection, output_dir: Path, *, bf16_supported: bool = False
) -> dict[str, Any]:
    """``SFTConfig`` arguments (SPEC §11.2; verified semantics in decisions.md §3.2, C10).

    Precision flags are always explicit (TRL defaults bf16 to True). ``eval_steps`` and
    ``save_steps`` are equal and count optimizer updates; the best checkpoint by validation loss
    is loaded at the end; ``num_train_epochs`` is a ceiling (early stopping ends sooner).
    """
    precision = Precision(selection.precision)
    qlora_bf16 = precision is Precision.QLORA_4BIT and bf16_supported
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": training.max_epochs,
        "per_device_train_batch_size": training.batch_size,
        "per_device_eval_batch_size": training.batch_size,
        "gradient_accumulation_steps": training.grad_accumulation_steps,
        "learning_rate": training.learning_rate,
        "warmup_steps": 0.05,  # a float below 1 is a share of the total steps (Transformers 5)
        "max_length": training.max_seq_length,
        "completion_only_loss": True,
        "eval_strategy": "steps",
        "eval_steps": training.eval_every_steps,
        "save_strategy": "steps",
        "save_steps": training.eval_every_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "report_to": "none",
        "bf16": precision is Precision.BF16 or qlora_bf16,
        "fp16": precision is Precision.FP16
        or (precision is Precision.QLORA_4BIT and not qlora_bf16),
        "use_cpu": selection.device == "cpu",
        "gradient_checkpointing": training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
        if training.gradient_checkpointing
        else None,
        "seed": training.seed,
        "data_seed": training.seed,
        "disable_tqdm": True,
    }


def lora_config_kwargs(config: Any, training: Any) -> dict[str, Any]:
    modules = training.hf_peft.target_modules
    return {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,  # PEFT scale = alpha / r (decisions.md C3)
        "lora_dropout": config.lora.dropout,
        "target_modules": "all-linear" if modules == "auto" else list(modules),
        "task_type": "CAUSAL_LM",
    }


def latest_checkpoint(run_dir: Path) -> Path | None:
    folder = run_dir / CHECKPOINTS_DIR
    found = [
        (int(p.name.split("-")[1]), p)
        for p in folder.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[1].isdigit()
    ]
    return max(found)[1] if found else None


def _checkpoint_step(path: str | None) -> int | None:
    if not path:
        return None
    tail = Path(path).name.split("-")[-1]
    return int(tail) if tail.isdigit() else None


class HfPeftBackend(TrainingBackend):
    name = Backend.HF_PEFT

    def _capabilities(
        self, probe: EnvironmentProbe, options: list[BackendOption]
    ) -> BackendCapabilities:
        devices = [o.device for o in options if o.available]
        precisions = []
        if "cuda" in devices:
            precisions += ["bf16"] if probe.torch.bf16_supported is not False else []
            precisions += ["fp16"]
            if probe.bitsandbytes_version:
                precisions.insert(0, "qlora_4bit")
        if "cpu" in devices:
            precisions.append("fp32")
        return BackendCapabilities(
            backend=self.name,
            available=bool(devices),
            devices=devices,
            precisions=precisions,
            qlora="qlora_4bit" in precisions,
            reasons=[f"{o.device}: {o.reason}" for o in options],
        )

    def artifact_format(self) -> str:
        return "peft_lora_safetensors"

    def versions(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for name in _PACKAGES:
            try:
                out[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                out[name] = None
        return out

    # ------------------------------------------------------------------ loading

    def load_for_inference(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path | None = None,
    ) -> LoadedModel:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise NotAvailableError(
                "PyTorch is not installed, so the Transformers backend can't run.",
                hint="Run `fliptuned doctor --suggest-install` for the install command.",
            ) from None

        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()  # no console bars while loading (chat output stays clean)
        kwargs: dict[str, Any] = {"local_files_only": True}
        precision = Precision(selection.precision)
        if selection.device == "cpu":
            kwargs["dtype"] = torch.float32
        elif precision is Precision.QLORA_4BIT:
            from transformers import BitsAndBytesConfig

            compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = torch.bfloat16 if precision is Precision.BF16 else torch.float16
            kwargs["device_map"] = {"": 0}

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        except (OSError, ValueError) as exc:
            raise NotAvailableError(
                f"Model '{entry.id}' could not be loaded.",
                hint="Allow downloads to fetch the model files again: "
                "--allow-download, or the download option in the UI.",
                details=f"{type(exc).__name__}: {exc}",
            ) from None
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return LoadedModel(
            backend=self.name,
            entry=entry,
            selection=selection,
            model_dir=model_dir,
            adapter_path=adapter_path,
            model=model,
            tokenizer=tokenizer,
            template_checksum=template_checksum(tokenizer, entry.chat_template_kwargs),
            extra={"eos_ids": _eos_ids(model, tokenizer)},
        )

    def unload(self, loaded: LoadedModel) -> None:
        on_cuda = loaded.selection.device == "cuda"
        super().unload(loaded)
        if on_cuda:
            import torch

            torch.cuda.empty_cache()

    def merge(
        self,
        entry: CatalogEntry,
        selection: BackendSelection,
        model_dir: Path,
        adapter_path: Path,
        out_dir: Path,
    ) -> MergeResult:
        """PEFT ``merge_and_unload`` into the unquantized base, saved as safetensors.

        A QLoRA adapter was trained on the 4-bit base; it's merged into the original base in
        bf16/fp16, because merging into 4-bit layers adds rounding errors (PEFT 0.21 warns about
        it). The merged model can then differ slightly from base-4-bit + adapter; the export's
        before/after check shows by how much.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        precision = Precision(selection.precision)
        notes = []
        kwargs: dict[str, Any] = {"local_files_only": True}
        if selection.device == "cpu":
            dtype, merged_precision = torch.float32, Precision.FP32
        else:
            bf16 = precision is Precision.BF16 or (
                precision is Precision.QLORA_4BIT and torch.cuda.is_bf16_supported()
            )
            dtype = torch.bfloat16 if bf16 else torch.float16
            merged_precision = Precision.BF16 if bf16 else Precision.FP16
            kwargs["device_map"] = {"": 0}
        if precision is Precision.QLORA_4BIT:
            notes.append(
                f"Trained as QLoRA on the 4-bit base; merged into the original base in "
                f"{merged_precision.value}, so answers can differ slightly from the 4-bit base "
                "with the adapter."
            )
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **kwargs)
        merged = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        del merged, model
        if selection.device == "cuda":
            torch.cuda.empty_cache()
        return MergeResult(path=str(out_dir), precision=merged_precision.value, notes=notes)

    # ------------------------------------------------------------------ generation

    def _inputs(self, loaded: LoadedModel, messages: Messages, config: GenerationConfig):
        encoded = loaded.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **loaded.template_kwargs,
        )
        limit = getattr(getattr(loaded.model, "config", None), "max_position_embeddings", None)
        prompt_len = int(encoded["input_ids"].shape[1])
        if isinstance(limit, int) and prompt_len + config.max_new_tokens > limit:
            raise ContextTooLongError(prompt_len, config.max_new_tokens, limit)
        encoded = {k: v.to(loaded.model.device) for k, v in encoded.items()}
        settings = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,  # greedy; clear sampling defaults from generation_config
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": loaded.tokenizer.pad_token_id,
        }
        return encoded, settings

    def generate(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> GenerationResult:
        import torch

        start = time.perf_counter()
        encoded, settings = self._inputs(loaded, messages, config)
        prompt_len = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = loaded.model.generate(**encoded, **settings)
        new_ids = output[0, prompt_len:].tolist()
        stopped = bool(new_ids) and new_ids[-1] in loaded.extra["eos_ids"]
        return GenerationResult(
            text=loaded.tokenizer.decode(new_ids, skip_special_tokens=True),
            prompt_tokens=prompt_len,
            completion_tokens=len(new_ids),
            finish_reason="length"
            if not stopped and len(new_ids) >= config.max_new_tokens
            else "stop",
            seconds=round(time.perf_counter() - start, 3),
        )

    def stream(
        self, loaded: LoadedModel, messages: Messages, config: GenerationConfig
    ) -> Iterator[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        encoded, settings = self._inputs(loaded, messages, config)
        prompt_len = int(encoded["input_ids"].shape[1])
        loaded.extra["last_finish_reason"] = None
        streamer = TextIteratorStreamer(
            loaded.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        errors: list[Exception] = []
        stop = Event()

        class _StopOnEvent(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],), stop.is_set(), dtype=torch.bool, device=input_ids.device
                )

        def run() -> None:
            try:
                with torch.inference_mode():
                    output = loaded.model.generate(
                        **encoded,
                        **settings,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_StopOnEvent()]),
                    )
                new_ids = output[0, prompt_len:].tolist()
                stopped = bool(new_ids) and new_ids[-1] in loaded.extra.get("eos_ids", ())
                loaded.extra["last_finish_reason"] = (
                    "length" if not stopped and len(new_ids) >= config.max_new_tokens else "stop"
                )
            except Exception as exc:
                errors.append(exc)
                # Wake the reader: without the stop signal it would wait forever.
                streamer.text_queue.put(streamer.stop_signal)

        worker = Thread(target=run, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            stop.set()  # the reader left (done, error, or hung up): stop generating now
            worker.join()
        if errors:
            raise errors[0]

    # ------------------------------------------------------------------ training

    def has_checkpoint(self, run_dir: Path) -> bool:
        return latest_checkpoint(run_dir) is not None

    def train(self, spec: TrainSpec, hooks: TrainHooks) -> TrainResult:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                EarlyStoppingCallback,
                TrainerCallback,
            )
            from trl import SFTConfig, SFTTrainer
        except ImportError:
            raise NotAvailableError(
                "PyTorch, PEFT or TRL is not installed, so the Transformers backend can't train.",
                hint="Run `fliptuned doctor --suggest-install` for the install command.",
            ) from None

        from datasets.utils.logging import disable_progress_bar as quiet_datasets
        from transformers.utils.logging import disable_progress_bar as quiet_transformers

        quiet_datasets()  # progress lives in our events, not in console bars
        quiet_transformers()
        training, selection, run_dir = spec.training, spec.selection, spec.run_dir
        cuda = selection.device == "cuda"
        precision = Precision(selection.precision)
        bf16_ok = bool(cuda and torch.cuda.is_bf16_supported())
        args_kwargs = sft_config_kwargs(
            training, selection, run_dir / CHECKPOINTS_DIR, bf16_supported=bf16_ok
        )
        lora_kwargs = lora_config_kwargs(spec.config, training)
        args_path = run_dir / ARGS_FILE
        args_path.write_text(
            json.dumps({"sft_config": args_kwargs, "lora_config": lora_kwargs}, indent=2) + "\n",
            encoding="utf-8",
        )

        load_kwargs: dict[str, Any] = {"local_files_only": True}
        if not cuda:
            load_kwargs["dtype"] = torch.float32
        elif precision is Precision.QLORA_4BIT:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
            )
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["dtype"] = torch.bfloat16 if precision is Precision.BF16 else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(spec.model_dir, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.model_dir, **load_kwargs)
        if precision is Precision.QLORA_4BIT:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training.gradient_checkpointing
            )

        kwargs = dict(spec.entry.chat_template_kwargs)
        train_rows = [prompt_completion_row(m, kwargs) for m in _read_messages(spec, "train")]
        valid_rows = [prompt_completion_row(m, kwargs) for m in _read_messages(spec, "valid")]
        callback = _make_callback(TrainerCallback, hooks, cuda)
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(**args_kwargs),
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(valid_rows),
            processing_class=tokenizer,
            peft_config=LoraConfig(**lora_kwargs),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training.patience), callback],
        )
        # With tqdm off, Transformers prints every log dict to stdout; our events carry them.
        from transformers.trainer_callback import PrinterCallback

        trainer.remove_callback(PrinterCallback)
        resume_from = latest_checkpoint(run_dir) if spec.resume else None
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

        state = trainer.state
        curve = [
            {k: entry[k] for k in ("step", "loss", "eval_loss", "learning_rate") if k in entry}
            for entry in state.log_history
            if "loss" in entry or "eval_loss" in entry
        ]
        cancelled = callback.cancelled
        adapter_dir = None
        if not cancelled:
            adapter_dir = run_dir / ADAPTER_DIR
            trainer.save_model(str(adapter_dir))  # the best weights (load_best_model_at_end)
            write_adapter_marker(adapter_dir, self, spec)
        return TrainResult(
            status="cancelled" if cancelled else "completed",
            adapter_path=str(adapter_dir) if adapter_dir else None,
            best_checkpoint=state.best_model_checkpoint,
            best_step=_checkpoint_step(state.best_model_checkpoint),
            best_eval_loss=state.best_metric,
            steps_done=state.global_step,
            total_steps=state.max_steps,
            stopped_early=not cancelled and state.global_step < state.max_steps,
            peak_memory_gb=callback.peak_memory_gb(),
            loss_curve=curve,
            backend_config=str(args_path),
            resume_note=(
                f"Resumed from {resume_from.name}: optimizer, scheduler, RNG state and step "
                "restored (Transformers resume_from_checkpoint)."
                if resume_from
                else None
            ),
        )

    # ------------------------------------------------------------------ checks

    def mask_check(
        self, loaded: LoadedModel, messages: Messages, work_dir: Path
    ) -> MaskCheckResult:
        """SPEC §11.4 mask test: run TRL's real data collator on one formatted example and
        check that prompt tokens are labelled -100 and answer tokens are not."""
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        kwargs = loaded.template_kwargs
        tokenizer = loaded.tokenizer
        full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
        )
        dataset = Dataset.from_list([prompt_completion_row(messages, kwargs)] * 2)
        args = SFTConfig(
            output_dir=str(work_dir),
            report_to="none",
            completion_only_loss=True,
            max_length=None,
            bf16=False,
            fp16=False,
            use_cpu=loaded.selection.device == "cpu",
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
        )
        trainer = SFTTrainer(
            model=loaded.model, args=args, train_dataset=dataset, processing_class=tokenizer
        )
        batch = trainer.data_collator([trainer.train_dataset[0]])
        labels = batch["labels"][0].tolist()
        n_prompt, n_full = len(prompt), len(full)
        masked = sum(1 for x in labels[:n_prompt] if x == -100)
        labelled = sum(1 for x in labels[n_prompt:n_full] if x != -100)
        return MaskCheckResult(
            passed=masked == n_prompt and labelled == n_full - n_prompt and n_full > n_prompt,
            prompt_tokens=n_prompt,
            masked_prompt_tokens=masked,
            completion_tokens=n_full - n_prompt,
            labelled_completion_tokens=labelled,
        )


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for value in (configured, tokenizer.eos_token_id):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list | tuple):
            ids.update(v for v in value if isinstance(v, int))
    return ids


def _read_messages(spec: TrainSpec, partition: str) -> list[Messages]:
    text = (spec.dataset_dir / f"{partition}.jsonl").read_text(encoding="utf-8")
    return [json.loads(line)["messages"] for line in text.split("\n") if line]


def _make_callback(base: type, hooks: TrainHooks, cuda: bool):
    """A ``TrainerCallback`` that writes events, tracks memory and honours cancel."""

    class FlipTunedCallback(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.cancelled = False
            self.started = time.monotonic()
            self.start_step: int | None = None
            self.peak_rss = 0

        def peak_memory_gb(self) -> float | None:
            if cuda:
                import torch

                return round(torch.cuda.max_memory_allocated() / 2**30, 3)
            info = psutil.Process().memory_info()
            peak = max(self.peak_rss, getattr(info, "peak_wset", 0) or 0, info.rss)
            return round(peak / 2**30, 3)

        def _sample(self) -> None:
            self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)

        def _eta(self, state) -> float | None:
            if self.start_step is None or state.global_step <= self.start_step:
                return None
            per_step = (time.monotonic() - self.started) / (state.global_step - self.start_step)
            return round(per_step * (state.max_steps - state.global_step), 1)

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_step = state.global_step
            self.started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            self._sample()
            hooks.emit("progress", done=state.global_step, total=state.max_steps, message=None)
            if hooks.cancel_requested():
                self.cancelled = True
                control.should_save = True  # a resumable checkpoint at this step
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs and "eval_loss" not in logs:
                return
            hooks.emit(
                "train",
                step=state.global_step,
                max_steps=state.max_steps,
                epoch=round(state.epoch or 0.0, 4),
                loss=logs.get("loss"),
                eval_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
                eta_s=self._eta(state),
                memory_gb=self.peak_memory_gb(),
            )

    return FlipTunedCallback()
