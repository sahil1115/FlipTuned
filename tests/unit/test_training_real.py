"""Real training jobs: start → events → cancel → resume → finish → load best adapter → generate.

- ``tiny``: the tiny test model on CPU (runs in CI on all three OSes);
- ``mlx`` + ``slow``: the smoke model on Apple Silicon;
- ``cuda`` + ``slow``: the tiny model on an NVIDIA GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from fliptuned.config.models import Backend
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.jobs.events import read_events
from fliptuned.models.catalog import get_entry
from fliptuned.services import jobs as jobs_service
from fliptuned.services import training as training_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.training import backends as registry
from fliptuned.training.backends.base import GenerationConfig
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples


def _wait_for(check, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.5)
    raise AssertionError("condition not reached in time")


def _cancel_resume_finish(tmp_path: Path, model: str, extra: dict) -> dict:
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(
        BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home), allow_download=True)
    )
    config = {
        "kind": "train",
        "name": "e2e",
        "model": {"catalog_id": model},
        "preset": "quick",
        "training": {"eval_every_steps": 4, "gradient_checkpointing": False, "max_epochs": 1},
        **extra,
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    request = training_service.TrainRequest(
        config_path=str(config_path), home=str(home), allow_download=True
    )

    # 1. start in the background and watch its events
    start = training_service.start_training(request, detach=True)
    events_path = Path(start.job.events_path)

    def progressed() -> bool:
        events, _ = read_events(events_path)
        return any(e["type"] == "progress" and e["done"] >= 6 for e in events)

    _wait_for(progressed)

    # 2. cancel: the worker saves a checkpoint and stops
    outcome = jobs_service.cancel_job(
        jobs_service.CancelJobRequest(home=str(home), job_id=start.job.id, timeout=120)
    )
    assert outcome.status == "cancelled" and not outcome.forced
    run = training_service.get_run(str(home), start.run_id)
    assert run.status == "cancelled"

    # 3. resume (in this process) and finish
    resumed = training_service.resume_run(
        training_service.ResumeRequest(home=str(home), run_id=start.run_id, allow_download=True),
        detach=False,
    )
    final = jobs_service.run_inline(str(home), resumed.job.id)
    assert final.status == "succeeded", final.error_summary
    result = jobs_service.job_result(str(home), resumed.job.id, training_service.TrainOutcome)
    r = result.result
    assert r.status == "completed" and r.resume_note and r.adapter_path
    assert r.best_eval_loss is not None and r.peak_memory_gb and r.loss_curve
    assert training_service.get_run(str(home), start.run_id).status == "succeeded"

    meta = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert [a["resume"] for a in meta["attempts"]] == [False, True]
    assert [a["status"] for a in meta["attempts"]] == ["cancelled", "succeeded"]
    assert (
        meta["model"]["template_checksum"] and meta["result"]["best_eval_loss"] == r.best_eval_loss
    )
    first_events, _ = read_events(events_path)
    assert any(e["type"] == "train" and e.get("eval_loss") is not None for e in first_events)

    # 4. load the best adapter and generate with it
    selection = BackendSelection.model_validate(
        meta["selection"]
        if "selection" in meta
        else {
            "backend": meta["backend"],
            "device": meta["device"],
            "precision": meta["precision"],
            "reason": "test",
        }
    )
    backend, loaded = registry.load_model(
        get_entry(model), selection, adapter_path=Path(r.adapter_path)
    )
    reply = backend.generate(
        loaded,
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}],
        GenerationConfig(max_new_tokens=8),
    )
    assert reply.completion_tokens > 0
    return {"result": r, "meta": meta}


@pytest.mark.tiny
def test_tiny_cpu_training_job(tmp_path: Path) -> None:
    out = _cancel_resume_finish(tmp_path, "tiny-qwen2", {"backend": "hf_peft", "device": "cpu"})
    assert out["meta"]["backend"] == "hf_peft" and out["meta"]["device"] == "cpu"
    assert "optimizer, scheduler, RNG" in out["result"].resume_note
    print(f"\npeak memory: {out['result'].peak_memory_gb} GB")


@pytest.mark.mlx
@pytest.mark.slow
def test_mlx_training_job(tmp_path: Path) -> None:
    out = _cancel_resume_finish(tmp_path, "qwen2.5-coder-0.5b", {"backend": "mlx"})
    assert out["meta"]["backend"] == Backend.MLX.value
    assert "Warm restart" in out["result"].resume_note
    print(f"\npeak memory: {out['result'].peak_memory_gb} GB")


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_training_job(tmp_path: Path) -> None:
    out = _cancel_resume_finish(tmp_path, "tiny-qwen2", {"backend": "hf_peft", "device": "cuda"})
    assert out["meta"]["device"] == "cuda"
    print(f"\npeak memory: {out['result'].peak_memory_gb} GB")
