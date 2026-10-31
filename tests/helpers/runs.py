"""A fake finished training run and a fake backend, for chat, export and evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from fliptuned.config.app_home import AppPaths
from fliptuned.config.models import Backend, TrainConfig
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.configs import save_config
from fliptuned.services.datasets import locate_dataset, read_manifest
from fliptuned.services.training import new_run_id
from fliptuned.training.backends.base import (
    GenerationConfig,
    GenerationResult,
    LoadedModel,
    MergeResult,
)


def make_run(home: Path, status: str = "succeeded") -> str:
    """A training run in the registry (dataset "fixtures"), with an adapter folder."""
    paths = AppPaths(home)
    config = TrainConfig.model_validate({"name": "fake", "model": {"catalog_id": "tiny-qwen2"}})
    saved = save_config(str(home), config)
    dataset_id, folder, _ = locate_dataset(paths, "fixtures")
    manifest = read_manifest(folder)
    run_id = new_run_id()
    run_dir = paths.runs / run_id
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")
    (run_dir / "hf_training_args.json").write_text("{}", encoding="utf-8")
    selection = {"backend": "hf_peft", "device": "cpu", "precision": "fp32", "reason": "test"}
    plan = {
        "config": config.model_dump(mode="json"),
        "selection": selection,
        "dataset_id": dataset_id,
        "dataset_dir": str(folder),
        "system_prompt": manifest.system_prompt,
    }
    (run_dir / "run_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    meta = {
        "result": {"adapter_path": str(adapter)},
        "backend_config_file": "hf_training_args.json",
        "model": {"template_checksum": "c" * 64},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with connect(paths.db) as conn:
        repositories.insert_run(
            conn,
            run_id=run_id,
            dataset_id=dataset_id,
            config_id=saved.id,
            model_id="tiny-qwen2",
            backend="hf_peft",
            device="cpu",
            precision="fp32",
            artifact_format="peft_lora_safetensors",
            run_dir=str(run_dir),
        )
        repositories.update_run(conn, run_id, status=status)
    return run_id


class FakeBackend:
    """Answers "tuned reply" with an adapter (or merged), "base reply" without one."""

    name = Backend.HF_PEFT

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []
        self.loads: list[Path | None] = []
        self.unloaded = 0
        self.merged_differs = False

    def _text(self, loaded: LoadedModel) -> str:
        if loaded.model_dir.name == "merged":
            return "merged reply" if self.merged_differs else "tuned reply"
        return "tuned reply" if loaded.adapter_path is not None else "base reply"

    def load(self, entry, selection, *, allow_download=False, adapter_path=None):
        """Stands in for ``training.backends.load_model``."""
        self.loads.append(adapter_path)
        return self, self.load_for_inference(entry, selection, Path("base"), adapter_path)

    def load_for_inference(self, entry, selection, model_dir, adapter_path=None) -> LoadedModel:
        return LoadedModel(
            backend=selection.backend,
            entry=entry,
            selection=selection,
            model_dir=Path(model_dir),
            adapter_path=adapter_path,
            model=None,
            tokenizer=None,
            template_checksum="c" * 64,
        )

    def generate(self, loaded, messages, config: GenerationConfig) -> GenerationResult:
        self.seen.append(messages)
        text = self._text(loaded)
        return GenerationResult(
            text=text, prompt_tokens=5, completion_tokens=2, finish_reason="stop", seconds=0.01
        )

    def stream(self, loaded, messages, config: GenerationConfig):
        self.seen.append(messages)
        first, second = self._text(loaded).split(" ")
        yield first
        yield " " + second

    def merge(self, entry, selection, model_dir, adapter_path, out_dir: Path) -> MergeResult:
        out_dir.mkdir(parents=True)
        (out_dir / "model.safetensors").write_bytes(b"merged")
        return MergeResult(path=str(out_dir), precision="fp32", notes=[])

    def unload(self, loaded) -> None:
        self.unloaded += 1

    def artifact_format(self) -> str:
        return "peft_lora_safetensors"

    def versions(self) -> dict:
        return {"torch": "fake"}
