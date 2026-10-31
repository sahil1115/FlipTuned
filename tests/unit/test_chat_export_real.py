"""Chat and export with the tiny test model on CPU (marker ``tiny``): a quick training run, then
chat with its adapter (no data file opened) and a real merge with the before/after check."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fliptuned.config.app_home import AppPaths
from fliptuned.services import chat, export
from fliptuned.services import jobs as jobs_service
from fliptuned.services import training as training_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    tmp = tmp_path_factory.mktemp("tiny-run")
    home = tmp / "home"
    ingest_samples(home)
    build_dataset(
        BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home), allow_download=True)
    )
    config = tmp / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "train",
                "name": "chat-export",
                "model": {"catalog_id": "tiny-qwen2"},
                "backend": "hf_peft",
                "device": "cpu",
                "preset": "quick",
                "training": {"max_epochs": 1, "gradient_checkpointing": False},
            }
        ),
        encoding="utf-8",
    )
    start = training_service.start_training(
        training_service.TrainRequest(config_path=str(config), home=str(home), allow_download=True),
        detach=False,
    )
    assert jobs_service.run_inline(str(home), start.job.id).status == "succeeded"
    return home, start.run_id


@pytest.mark.tiny
def test_tiny_chat_with_trained_adapter(trained, file_access) -> None:
    home, run_id = trained
    paths = AppPaths(home)
    with file_access.recording():
        session = chat.start_chat(
            chat.ChatRequest(home=str(home), run_id=run_id, max_new_tokens=16), detach=False
        )
        session.wait_ready(timeout=300)
        reply = "".join(session.reply([{"role": "user", "content": "How do I read a file?"}]))
        session.close()
    assert reply  # the tiny random model writes something
    assert jobs_service.get_job(str(home), session.job_id).status == "succeeded"
    for folder in (paths.datasets, paths.sources, paths.uploads):
        assert [p for e, p in file_access.under(folder) if Path(p) != folder] == [], folder.name
    run_dir = paths.runs / run_id
    touched = {Path(p).relative_to(run_dir).parts[0] for _, p in file_access.under(run_dir)}
    assert touched <= {"run_plan.json", "metadata.json", "adapter"}, touched
    assert "adapter" in touched  # the trained adapter was really loaded


@pytest.mark.tiny
def test_tiny_merge_export(trained) -> None:
    home, run_id = trained
    start = export.export_run(
        export.ExportRequest(home=str(home), run_id=run_id, merge=True), detach=False
    )
    assert jobs_service.run_inline(str(home), start.job.id).status == "succeeded"
    result = jobs_service.job_result(str(home), start.job.id, export.ExportResult)
    merged = Path(result.merged_path)
    assert any(merged.glob("*.safetensors")) and (merged / "tokenizer_config.json").exists()
    assert result.merged_precision == "fp32" and result.check is not None
    assert result.check.prompts == 5
    print(f"\nmerge check: {result.check.identical} of 5 identical; notes: {result.notes}")
