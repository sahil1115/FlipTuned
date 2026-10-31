"""Real evaluation with the tiny test model on CPU (marker ``tiny``): train a quick run, evaluate
it against the base model, fill the review sheet with synthetic judgements and score it."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from fliptuned.services import evaluation
from fliptuned.services import jobs as jobs_service
from fliptuned.services import training as training_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples

SYNTHETIC = ("A", "B", "tie", "B", "A", "A", "B")  # a fixed, model-blind pattern


def _write(path: Path, data: dict) -> str:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def _evaluate(home: Path, run_id: str, config: str) -> evaluation.EvaluationOutcome:
    start = evaluation.start_evaluation(
        evaluation.EvaluateRequest(
            home=str(home), run_id=run_id, config_path=config, allow_download=True
        ),
        detach=False,
    )
    info = jobs_service.run_inline(str(home), start.job.id)
    assert info.status == "succeeded", jobs_service.job_error(str(home), start.job.id)
    return jobs_service.job_result(str(home), start.job.id, evaluation.EvaluationOutcome)


@pytest.mark.tiny
def test_tiny_cpu_evaluation_with_synthetic_judgements(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(
        BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home), allow_download=True)
    )
    train_config = _write(
        tmp_path / "train.yaml",
        {
            "kind": "train",
            "name": "eval-e2e",
            "model": {"catalog_id": "tiny-qwen2"},
            "backend": "hf_peft",
            "device": "cpu",
            "preset": "quick",
            "training": {"max_epochs": 1, "eval_every_steps": 10, "gradient_checkpointing": False},
        },
    )
    start = training_service.start_training(
        training_service.TrainRequest(
            config_path=train_config, home=str(home), allow_download=True
        ),
        detach=False,
    )
    assert jobs_service.run_inline(str(home), start.job.id).status == "succeeded"

    # The SPEC §12.3 defaults; the fixture test set (14 items) is too small for them.
    default_config = _write(
        tmp_path / "eval.yaml", {"kind": "evaluate", "generation": {"max_new_tokens": 48}}
    )
    outcome = _evaluate(home, start.run_id, default_config)
    assert outcome.test_items == 14 and outcome.regression_items == 20
    assert not outcome.baseline_reused and outcome.score.verdict == "inconclusive"
    folder = Path(outcome.eval_dir)
    for name in ("results.jsonl", "metrics.json", "review_sheet.csv", "report.md", "verdict.json"):
        assert (folder / name).exists(), name

    with open(outcome.review_sheet, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    for number, row in enumerate(rows):
        row["judgement"] = SYNTHETIC[number % len(SYNTHETIC)]
    with open(outcome.review_sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    imported = evaluation.import_review_sheet(
        evaluation.SheetImportRequest(
            home=str(home), path=outcome.review_sheet, run_id=start.run_id
        )
    )
    assert sum(imported.recorded.values()) == 14
    score = evaluation.score_evaluation(
        evaluation.ScoreRequest(home=str(home), evaluation_id=outcome.evaluation_id)
    )
    assert score.judged == 14 and score.review_coverage == 1.0
    assert score.verdict == "inconclusive"  # 14 < 50 test items
    verdict = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["inputs"]["judged"] == 14

    print(
        f"\nverdict (defaults): {score.verdict}; wins {score.wins}, losses {score.losses}, "
        f"ties {score.ties}, p = {score.sign_test_p:.4g}; checks base "
        f"{score.deterministic_pass_base:.1f}% / fine-tuned {score.deterministic_pass_tuned:.1f}%; "
        f"regression set base {score.regression_pass_base:.1f}% / fine-tuned "
        f"{score.regression_pass_tuned:.1f}%; reasons: {score.reasons}"
    )
    print(f"report: {score.report_path}")
