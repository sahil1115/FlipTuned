"""Evaluation (SPEC §12): baselines, evaluating a run, blind review, scoring and the verdict.

**Baselines** (``evaluate --baseline-only``) write ``<app_home>/baselines/<baseline_id>/``:
``results.jsonl`` (base-model answers to the test split) and ``metadata.json``. The baseline ID
derives from everything that shapes the answers, so evaluations reuse a matching baseline.

**Evaluating a run** is an ``evaluate`` job under the model lock. Through the run's backend, with
the run's system prompt and greedy decoding, it gets base-model answers (from a matching baseline,
or generated and saved as one) and fine-tuned answers (base model + the run's best adapter) for
the test split and the bundled general-regression set; runs the deterministic checks on every
answer; draws the hidden A/B mapping per test item (registry only); and writes
``<run_dir>/evaluation/<eval_id>/``: ``results.jsonl``, ``metadata.json``, ``review_sheet.csv``,
then ``metrics.json``, ``verdict.json`` and ``report.md`` from a first scoring.

**Review**: judgements arrive one pair at a time (:func:`review_queue` and :func:`judge`, used by
the UI) or from the filled-in review sheet (:func:`import_review_sheet`). Both store them in the
registry, and :func:`score_evaluation` is the one scoring service. Review payloads and the sheet
carry only "A" and "B"; which model wrote which is read back only when scoring.

CLI output, metrics, the verdict and the report hold counts, IDs and category labels only.
``results.jsonl`` and the review sheet hold the questions and answers, on this machine only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import secrets
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths
from fliptuned.config.loader import load_config
from fliptuned.config.models import Backend, Device, EvaluateConfig, Precision
from fliptuned.dataset.manifest import PARTITION_FILES, DatasetManifest, IndexRow
from fliptuned.diagnostics.backend_select import BackendSelection, EnvironmentProbe
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.evaluation.blinding import (
    Judgement,
    Outcome,
    draw_a_model,
    parse_judgement,
    unblind,
)
from fliptuned.evaluation.checks import (
    Expectations,
    all_passed,
    expectations_from_reference,
    run_checks,
)
from fliptuned.evaluation.metrics import (
    EvaluationMetrics,
    PriorExposure,
    category_results,
    check_summary,
    generation_stats,
    regressions,
)
from fliptuned.evaluation.regression import load_regression_set
from fliptuned.evaluation.report import render_report
from fliptuned.evaluation.verdict import CategoryResult, Criterion, VerdictInputs, decide
from fliptuned.inference.prompting import chat_messages
from fliptuned.jobs.state import JobType
from fliptuned.jobs.worker import JobCancelled, JobContext
from fliptuned.models.catalog import CatalogEntry, ModelSource, get_entry
from fliptuned.models.compat import select_for_entry
from fliptuned.platform.paths import remove_tree, replace_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, utc_now
from fliptuned.services.datasets import locate_dataset, read_index, read_manifest
from fliptuned.services.doctor import default_probe
from fliptuned.services.jobs import JobInfo, _submit, reconcile_jobs
from fliptuned.services.review import ImportProblem, open_registry
from fliptuned.services.runs import load_run
from fliptuned.training.backends import get_backend, load_model
from fliptuned.training.backends.base import (
    GenerationConfig,
    GenerationResult,
    LoadedModel,
    TrainingBackend,
)
from fliptuned.utils.logging import get_logger
from fliptuned.utils.spreadsheet import spreadsheet_cell

log = get_logger(__name__)

RESULTS_FILE = "results.jsonl"
METADATA_FILE = "metadata.json"
REVIEW_SHEET = "review_sheet.csv"
METRICS_FILE = "metrics.json"
VERDICT_FILE = "verdict.json"
REPORT_FILE = "report.md"
EVALUATION_DIR = "evaluation"
SHEET_COLUMNS = [
    "evaluation_id",
    "item",
    "item_id",
    "category",
    "question",
    "reference_answer",
    "answer_a",
    "answer_b",
    "judgement",
]
_SHEET_REQUIRED = ("evaluation_id", "item_id", "judgement")
_MAX_PROBLEMS = 200


def _sha(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Split on newlines only: JSON lines may contain raw U+2028/U+0085 (P4-14).
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _load_eval_config(config_path: str | None) -> EvaluateConfig:
    if not config_path:
        return EvaluateConfig()
    return load_config(Path(config_path).expanduser().resolve(), EvaluateConfig)


# --------------------------------------------------------------------------- test items + baselines

TestItem = tuple[IndexRow, list[dict[str, str]]]


def _test_items(folder: Path, dataset_id: str, limit: int | None) -> list[TestItem]:
    """The test split's ``(index row, messages)`` pairs, in file order."""
    text = (folder / PARTITION_FILES["test"]).read_text(encoding="utf-8")
    lines = [line for line in text.split("\n") if line]
    index = [r for r in read_index(folder) if r.partition == "test"]
    if len(lines) != len(index):
        raise FlipTunedError(
            f"Dataset {dataset_id} is inconsistent (test file and index differ).",
            hint=f"Run `fliptuned validate-dataset {dataset_id}`, then rebuild it.",
        )
    if not lines:
        raise NotAvailableError(f"Dataset {dataset_id} has no test items.")
    pairs = [(row, json.loads(line)["messages"]) for row, line in zip(index, lines, strict=True)]
    return pairs[:limit]


def _baseline_identity(
    manifest_hash: str,
    entry: CatalogEntry,
    source: ModelSource,
    selection: BackendSelection,
    generation: GenerationConfig,
    limit: int | None,
) -> tuple[str, dict[str, Any]]:
    key = {
        "dataset_manifest": manifest_hash,
        "model": entry.id,
        "repo": source.repo,
        "revision": source.revision,
        "backend": selection.backend.value,
        "device": selection.device,
        "precision": selection.precision,
        "generation": generation.model_dump(),
        "limit": limit,
    }
    return f"bl_{_sha(json.dumps(key, sort_keys=True))[:12]}", key


def _generate(
    backend: TrainingBackend,
    loaded: LoadedModel,
    system_prompt: str,
    questions: list[str],
    generation: GenerationConfig,
    before: Callable[[], None] | None = None,
    after: Callable[[int, int], None] | None = None,
) -> list[GenerationResult]:
    """Greedy answers, one per question, with the fixed system prompt and nothing else."""
    out = []
    for number, question in enumerate(questions, start=1):
        if before:
            before()
        messages = chat_messages(system_prompt, [{"role": "user", "content": question}])
        out.append(backend.generate(loaded, messages, generation))
        if after:
            after(number, len(questions))
    return out


def _baseline_rows(items: list[TestItem], results: list[GenerationResult]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row.record_id,
            "source_id": row.source_id,
            "category": row.category,
            "line": row.line,
            "output": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
            "seconds": result.seconds,
        }
        for (row, _), result in zip(items, results, strict=True)
    ]


def _store_baseline(
    paths: AppPaths,
    *,
    baseline_id: str,
    key: dict[str, Any],
    dataset_id: str,
    manifest: DatasetManifest,
    entry: CatalogEntry,
    selection: BackendSelection,
    backend: TrainingBackend,
    template_checksum: str,
    generation: GenerationConfig,
    rows: list[dict[str, Any]],
    limit: int | None,
    started: str,
    warnings: list[str],
) -> tuple[Path, dict[str, Any]]:
    source = entry.sources[selection.backend.value]
    metadata = {
        "baseline_id": baseline_id,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": key["dataset_manifest"],
        "split": "test",
        "items": len(rows),
        "limit": limit,
        "model": {
            "catalog_id": entry.id,
            "repo": source.repo,
            "revision": source.revision,
            "license": entry.license,
            "chat_template_kwargs": entry.chat_template_kwargs,
            "template_checksum": template_checksum,
        },
        "system_prompt_sha256": manifest.system_prompt_sha256,
        "generation": {**generation.model_dump(), "decoding": "greedy"},
        "backend": selection.backend.value,
        "device": selection.device,
        "precision": selection.precision,
        "selection_reason": selection.reason,
        "artifact_format": backend.artifact_format(),
        "packages": backend.versions(),
        "machine": {"os": platform.platform(), "python": platform.python_version()},
        "started_at": started,
        "finished_at": utc_now(),
        "warnings": warnings,
    }
    out_dir = paths.baselines / baseline_id
    tmp_dir = paths.baselines / f".building-{baseline_id}"
    if tmp_dir.exists():
        remove_tree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    _write_jsonl(tmp_dir / RESULTS_FILE, rows)
    _write_json(tmp_dir / METADATA_FILE, metadata)
    replace_dir(tmp_dir, out_dir)
    with connect(paths.db) as conn:
        repositories.upsert_baseline(
            conn,
            baseline_id=baseline_id,
            dataset_id=dataset_id,
            model_id=entry.id,
            backend=selection.backend.value,
            device=selection.device,
            precision=selection.precision,
            status="succeeded",
            path=str(out_dir),
            metadata=metadata,
        )
    return out_dir, metadata


def _manifest_hash(paths: AppPaths, dataset_id: str, folder: Path) -> str:
    with connect(paths.db) as conn:
        row = repositories.get_dataset(conn, dataset_id)
    registered = row["manifest_hash"] if row else None
    return registered or _sha((folder / "manifest.json").read_text(encoding="utf-8"))


class BaselineRequest(BaseModel):
    home: str | None = None
    dataset: str = Field(description="Dataset ID or name.")
    model: str | None = Field(
        default=None, description="Catalog model ID. Empty = the model the dataset was built for."
    )
    backend: Backend = Backend.AUTO
    device: Device = Device.AUTO
    precision: Precision = Precision.AUTO
    config_path: str | None = Field(default=None, description="Evaluate config (YAML).")
    limit: int | None = Field(default=None, ge=1, description="Only the first N test items.")
    allow_download: bool = Field(default=False, description="Download model files if missing.")
    overwrite: bool = False


class BaselineResult(BaseModel):
    baseline_id: str
    path: str
    dataset_id: str
    model: str
    backend: str
    device: str
    precision: str
    items: int
    finish_reasons: dict[str, int]
    completion_tokens: int
    seconds: float
    rebuilt: bool
    warnings: list[str]


def run_baseline(
    request: BaselineRequest,
    progress: Callable[[int, int], None] | None = None,
    probe_fn: Callable[[], EnvironmentProbe] | None = None,
) -> BaselineResult:
    paths = open_registry(request.home)
    dataset_id, folder, _ = locate_dataset(paths, request.dataset)
    manifest = read_manifest(folder)
    config = _load_eval_config(request.config_path)
    generation = GenerationConfig(max_new_tokens=config.generation.max_new_tokens)
    entry = get_entry(request.model or str(manifest.model.get("catalog_id")))
    selection = select_for_entry(
        (probe_fn or default_probe)(), entry, request.backend, request.device, request.precision
    )
    source = entry.sources[selection.backend.value]
    items = _test_items(folder, dataset_id, request.limit)

    warnings = []
    if entry.id != manifest.model.get("catalog_id"):
        warnings.append(
            f"The dataset's token lengths were measured with '{manifest.model.get('catalog_id')}', "
            f"not '{entry.id}'."
        )
    baseline_id, key = _baseline_identity(
        _manifest_hash(paths, dataset_id, folder),
        entry,
        source,
        selection,
        generation,
        request.limit,
    )
    with connect(paths.db) as conn:
        existing = repositories.get_baseline(conn, baseline_id)
    if (existing is not None or (paths.baselines / baseline_id).exists()) and not request.overwrite:
        raise FlipTunedError(
            f"A baseline with exactly these settings already exists ({baseline_id}).",
            hint="Use it as it is, or re-run with --overwrite to generate it again.",
        )

    started = utc_now()
    backend, loaded = load_model(entry, selection, allow_download=request.allow_download)
    results = _generate(
        backend,
        loaded,
        manifest.system_prompt,
        [messages[1]["content"] for _, messages in items],
        generation,
        after=progress,
    )
    rows = _baseline_rows(items, results)
    out_dir, _ = _store_baseline(
        paths,
        baseline_id=baseline_id,
        key=key,
        dataset_id=dataset_id,
        manifest=manifest,
        entry=entry,
        selection=selection,
        backend=backend,
        template_checksum=loaded.template_checksum,
        generation=generation,
        rows=rows,
        limit=request.limit,
        started=started,
        warnings=warnings,
    )
    log.info("baseline %s: %d items", baseline_id, len(rows))
    return BaselineResult(
        baseline_id=baseline_id,
        path=str(out_dir),
        dataset_id=dataset_id,
        model=entry.id,
        backend=selection.backend.value,
        device=selection.device,
        precision=selection.precision,
        items=len(rows),
        finish_reasons=dict(sorted(Counter(r["finish_reason"] for r in rows).items())),
        completion_tokens=sum(r["completion_tokens"] for r in rows),
        seconds=round(sum(r["seconds"] for r in rows), 2),
        rebuilt=existing is not None,
        warnings=warnings,
    )


def _reusable_baseline(
    paths: AppPaths, baseline_id: str, items: list[TestItem], system_prompt_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """A stored baseline's rows and metadata, if it matches these test items exactly."""
    with connect(paths.db) as conn:
        row = repositories.get_baseline(conn, baseline_id)
    if row is None or row["status"] != "succeeded":
        return None
    results = Path(row["path"]) / RESULTS_FILE
    if not results.exists():
        return None
    rows = _read_jsonl(results)
    meta = json.loads(row["metadata_json"] or "{}")
    if meta.get("system_prompt_sha256") != system_prompt_sha256:
        return None
    if [r["item_id"] for r in rows] != [index.record_id for index, _ in items]:
        return None
    return rows, meta


# --------------------------------------------------------------------------- start an evaluation


class EvaluateRequest(BaseModel):
    home: str | None = None
    run_id: str = Field(description="The training run to evaluate (it must have finished).")
    config_path: str | None = Field(default=None, description="Evaluate config (YAML).")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Only the first N test items (a quick look; a verdict needs the full set).",
    )
    allow_download: bool = Field(default=False, description="Download model files if missing.")


class EvaluationStart(BaseModel):
    evaluation_id: str
    run_id: str
    eval_dir: str
    job: JobInfo


def new_evaluation_id() -> str:
    return f"ev_{datetime.now(UTC):%Y%m%d%H%M%S}_{secrets.token_hex(3)}"


def _finished_run(
    paths: AppPaths, run_id: str
) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    """(run folder, run plan, run metadata, best adapter) of a succeeded training run."""
    run = load_run(paths, run_id)
    assert run.adapter is not None
    return run.run_dir, run.plan, run.meta, run.adapter


def start_evaluation(request: EvaluateRequest, *, detach: bool) -> EvaluationStart:
    config = _load_eval_config(request.config_path)
    paths = open_registry(request.home)
    run_dir, plan, _, _ = _finished_run(paths, request.run_id)
    get_backend(BackendSelection.model_validate(plan["selection"]).backend)  # fails off-platform
    evaluation_id = new_evaluation_id()
    eval_dir = run_dir / EVALUATION_DIR / evaluation_id
    eval_dir.mkdir(parents=True)
    with connect(paths.db) as conn:
        repositories.insert_evaluation(
            conn,
            evaluation_id=evaluation_id,
            run_id=request.run_id,
            eval_dir=str(eval_dir),
            config=config.model_dump(mode="json"),
        )
    params = {
        "evaluation_id": evaluation_id,
        "limit": request.limit,
        "allow_download": request.allow_download,
    }
    try:
        job = _submit(request.home, JobType.EVALUATE, params, detach, run_id=request.run_id)
    except BaseException:
        with connect(paths.db) as conn:
            repositories.delete_evaluation(conn, evaluation_id)
        shutil.rmtree(eval_dir, ignore_errors=True)
        raise
    with connect(paths.db) as conn:
        repositories.update_evaluation(conn, evaluation_id, job_id=job.id)
    return EvaluationStart(
        evaluation_id=evaluation_id, run_id=request.run_id, eval_dir=str(eval_dir), job=job
    )


# --------------------------------------------------------------------------- the evaluate job


class EvaluationOutcome(BaseModel):
    evaluation_id: str
    run_id: str
    eval_dir: str
    baseline_id: str
    baseline_reused: bool
    test_items: int
    regression_items: int
    review_sheet: str
    score: ScoreResult


def _side(generated: dict[str, Any], expect: Expectations) -> dict[str, Any]:
    checks = run_checks(generated["output"], generated["finish_reason"], expect)
    return {
        **generated,
        "checks": [c.model_dump() for c in checks],
        "passed": all_passed(checks),
    }


def _as_row(result: GenerationResult) -> dict[str, Any]:
    return {
        "output": result.text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "seconds": result.seconds,
    }


def _from_baseline(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("output", "prompt_tokens", "completion_tokens", "finish_reason", "seconds")
    return {k: row[k] for k in keys}


def _item_hash(question: str, reference: str) -> str:
    return _sha(f"{question}\0{reference}")


def _answers_ab(row: dict[str, Any], a_model: str) -> tuple[str, str]:
    b_model = "base" if a_model == "tuned" else "tuned"
    return row[a_model]["output"], row[b_model]["output"]


def _write_sheet(
    path: Path, evaluation_id: str, rows: list[dict[str, Any]], a_models: dict[str, str]
) -> None:
    # utf-8-sig so spreadsheet apps detect UTF-8; newline="" so embedded line breaks survive.
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SHEET_COLUMNS)
        for row in rows:
            answer_a, answer_b = _answers_ab(row, a_models[row["item_id"]])
            writer.writerow(
                [
                    evaluation_id,
                    row["position"],
                    row["item_id"],
                    spreadsheet_cell(row["category"]),
                    spreadsheet_cell(row["question"]),
                    spreadsheet_cell(row["reference"]),
                    spreadsheet_cell(answer_a),
                    spreadsheet_cell(answer_b),
                    "",
                ]
            )


def _generate_answers(ctx: JobContext, evaluation: Any) -> dict[str, Any]:
    """Generate, check and store both models' answers. Returns the evaluation metadata."""
    paths = ctx.paths
    evaluation_id = evaluation["id"]
    run_id = evaluation["run_id"]
    eval_dir = Path(evaluation["eval_dir"])
    config = EvaluateConfig.model_validate(json.loads(evaluation["config_json"]))
    limit = ctx.params.get("limit")
    allow_download = bool(ctx.params.get("allow_download"))

    _, plan, _, adapter = _finished_run(paths, run_id)
    selection = BackendSelection.model_validate(plan["selection"])
    entry = get_entry(plan["config"]["model"]["catalog_id"])
    source = entry.sources[selection.backend.value]
    dataset_id = plan["dataset_id"]
    folder = Path(plan["dataset_dir"])
    manifest = read_manifest(folder)
    system_prompt = plan["system_prompt"]
    if system_prompt != manifest.system_prompt:
        raise FlipTunedError(
            "The run's system prompt differs from its dataset's, so the comparison wouldn't be fair.",
            hint="Rebuild the dataset or retrain, so both use the same system prompt.",
        )
    items = _test_items(folder, dataset_id, limit)
    generation = GenerationConfig(max_new_tokens=config.generation.max_new_tokens)
    manifest_hash = _manifest_hash(paths, dataset_id, folder)
    baseline_id, key = _baseline_identity(
        manifest_hash, entry, source, selection, generation, limit
    )
    reusable = _reusable_baseline(paths, baseline_id, items, manifest.system_prompt_sha256)
    regression = load_regression_set().items if config.include_regression_set else []
    questions = [messages[1]["content"] for _, messages in items]
    prompts = [r.prompt for r in regression]

    total = (0 if reusable else len(items)) + 2 * len(regression) + len(items)
    done = 0

    def counter(label: str) -> Callable[[int, int], None]:
        def after(number: int, count: int) -> None:
            nonlocal done
            done += 1
            ctx.progress(done, total, f"{label} {number}/{count}")

        return after

    started = utc_now()
    base_regression: list[GenerationResult] = []
    if reusable is None or regression:
        backend, loaded = load_model(entry, selection, allow_download=allow_download)
        try:
            if reusable is None:
                results = _generate(
                    backend,
                    loaded,
                    system_prompt,
                    questions,
                    generation,
                    ctx.check_cancel,
                    counter("base model: test item"),
                )
                generated = _baseline_rows(items, results)
                _, baseline_meta = _store_baseline(
                    paths,
                    baseline_id=baseline_id,
                    key=key,
                    dataset_id=dataset_id,
                    manifest=manifest,
                    entry=entry,
                    selection=selection,
                    backend=backend,
                    template_checksum=loaded.template_checksum,
                    generation=generation,
                    rows=generated,
                    limit=limit,
                    started=started,
                    warnings=[],
                )
                reused, base_rows = False, generated
            else:
                reused, (base_rows, baseline_meta) = True, reusable
            base_regression = _generate(
                backend,
                loaded,
                system_prompt,
                prompts,
                generation,
                ctx.check_cancel,
                counter("base model: regression prompt"),
            )
        finally:
            backend.unload(loaded)
    else:
        reused, (base_rows, baseline_meta) = True, reusable

    backend, loaded = load_model(
        entry, selection, allow_download=allow_download, adapter_path=adapter
    )
    try:
        tuned_test = _generate(
            backend,
            loaded,
            system_prompt,
            questions,
            generation,
            ctx.check_cancel,
            counter("fine-tuned: test item"),
        )
        tuned_regression = _generate(
            backend,
            loaded,
            system_prompt,
            prompts,
            generation,
            ctx.check_cancel,
            counter("fine-tuned: regression prompt"),
        )
        tuned_checksum = loaded.template_checksum
    finally:
        backend.unload(loaded)
    if baseline_meta["model"]["template_checksum"] != tuned_checksum:
        raise FlipTunedError(
            "The base and fine-tuned answers used different chat templates.",
            hint="Re-run the evaluation; if it repeats, re-download the model with --allow-download.",
        )

    rows: list[dict[str, Any]] = []
    for position, ((index, messages), base, tuned) in enumerate(
        zip(items, base_rows, tuned_test, strict=True), start=1
    ):
        expect = expectations_from_reference(messages[2]["content"])
        rows.append(
            {
                "set": "test",
                "item_id": index.record_id,
                "position": position,
                "line": index.line,
                "category": index.category,
                "question": messages[1]["content"],
                "reference": messages[2]["content"],
                "base": _side(_from_baseline(base), expect),
                "tuned": _side(_as_row(tuned), expect),
            }
        )
    for position, (item, base, tuned) in enumerate(
        zip(regression, base_regression, tuned_regression, strict=True), start=1
    ):
        expect = item.expect.expectations()
        rows.append(
            {
                "set": "regression",
                "item_id": item.id,
                "position": position,
                "line": None,
                "category": item.category,
                "question": item.prompt,
                "reference": None,
                "base": _side(_as_row(base), expect),
                "tuned": _side(_as_row(tuned), expect),
            }
        )

    test_rows = [r for r in rows if r["set"] == "test"]
    eval_items = [
        {
            "item_id": r["item_id"],
            "position": r["position"],
            "category": r["category"],
            "content_hash": _item_hash(r["question"], r["reference"]),
            "a_model": draw_a_model(config.review_seed, evaluation_id, r["item_id"]),
        }
        for r in test_rows
    ]
    metadata = {
        "evaluation_id": evaluation_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": manifest_hash,
        "split": "test",
        "test_items": len(test_rows),
        "test_set_sha256": _sha("\n".join(i["content_hash"] for i in eval_items)),
        "limit": limit,
        "regression_items": len(regression),
        "regression_set_version": load_regression_set().version if regression else None,
        "config": config.model_dump(mode="json"),
        "model": {
            "catalog_id": entry.id,
            "repo": source.repo,
            "revision": source.revision,
            "license": entry.license,
            "chat_template_kwargs": entry.chat_template_kwargs,
            "template_checksum": tuned_checksum,
        },
        "adapter_path": str(adapter),
        "system_prompt_sha256": manifest.system_prompt_sha256,
        "generation": {**generation.model_dump(), "decoding": "greedy"},
        "backend": selection.backend.value,
        "device": selection.device,
        "precision": selection.precision,
        "artifact_format": backend.artifact_format(),
        "packages": backend.versions(),
        "machine": {"os": platform.platform(), "python": platform.python_version()},
        "baseline_id": baseline_id,
        "baseline_reused": reused,
        "started_at": started,
        "finished_at": utc_now(),
    }
    _write_jsonl(eval_dir / RESULTS_FILE, rows)
    _write_json(eval_dir / METADATA_FILE, metadata)
    _write_sheet(
        eval_dir / REVIEW_SHEET,
        evaluation_id,
        test_rows,
        {i["item_id"]: i["a_model"] for i in eval_items},
    )
    with connect(paths.db) as conn:
        repositories.replace_eval_items(conn, evaluation_id, eval_items)
    return metadata


def execute_evaluation(ctx: JobContext) -> dict[str, Any]:
    """Handler of the ``evaluate`` job."""
    paths = ctx.paths
    evaluation_id = ctx.params["evaluation_id"]
    with connect(paths.db) as conn:
        evaluation = repositories.get_evaluation(conn, evaluation_id)
        if evaluation is None:
            raise NotAvailableError(f"No evaluation {evaluation_id}.")
        repositories.update_evaluation(
            conn, evaluation_id, status="running", from_statuses=("queued",)
        )
    status = "failed"
    try:
        metadata = _generate_answers(ctx, evaluation)
        status = "ready"
    except (JobCancelled, KeyboardInterrupt):
        status = "cancelled"
        raise
    finally:
        with connect(paths.db) as conn:
            repositories.update_evaluation(conn, evaluation_id, status=status)
    score = score_evaluation(ScoreRequest(home=ctx.home, evaluation_id=evaluation_id))
    eval_dir = Path(evaluation["eval_dir"])
    return EvaluationOutcome(
        evaluation_id=evaluation_id,
        run_id=evaluation["run_id"],
        eval_dir=str(eval_dir),
        baseline_id=metadata["baseline_id"],
        baseline_reused=metadata["baseline_reused"],
        test_items=metadata["test_items"],
        regression_items=metadata["regression_items"],
        review_sheet=str(eval_dir / REVIEW_SHEET),
        score=score,
    ).model_dump(mode="json")


# --------------------------------------------------------------------------- evaluations


class EvaluationInfo(BaseModel):
    id: str
    run_id: str
    status: str = Field(
        description="queued, running, ready (answers generated; review and score), "
        "failed, cancelled or interrupted."
    )
    verdict: str | None = Field(description="Verdict of the latest scoring.")
    eval_dir: str
    job_id: str | None
    created_at: str
    updated_at: str
    total: int = Field(description="Test pairs to review.")
    judged: int


def _sync_status(paths: AppPaths, row: Any) -> Any:
    """An evaluation whose job ended without finishing takes the job's final status."""
    if row["status"] not in ("queued", "running") or not row["job_id"]:
        return row
    with connect(paths.db) as conn:
        job = repositories.get_job(conn, row["job_id"])
        if job is not None and job["status"] in ("failed", "cancelled", "interrupted"):
            repositories.update_evaluation(
                conn, row["id"], status=job["status"], from_statuses=("queued", "running")
            )
            row = repositories.get_evaluation(conn, row["id"])
    return row


def _info(paths: AppPaths, row: Any) -> EvaluationInfo:
    row = _sync_status(paths, row)
    with connect(paths.db) as conn:
        total = len(repositories.list_eval_items(conn, row["id"]))
        judged = len(repositories.list_judgements(conn, row["id"]))
    return EvaluationInfo(
        id=row["id"],
        run_id=row["run_id"],
        status=row["status"],
        verdict=row["verdict"],
        eval_dir=row["eval_dir"],
        job_id=row["job_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        total=total,
        judged=judged,
    )


def get_evaluation(home: str | None, evaluation_id: str) -> EvaluationInfo:
    paths = open_registry(home)
    reconcile_jobs(home)
    with connect(paths.db) as conn:
        row = repositories.get_evaluation(conn, evaluation_id)
    if row is None:
        raise NotAvailableError(f"No evaluation {evaluation_id}.")
    return _info(paths, row)


def list_evaluations(home: str | None, run_id: str | None = None) -> list[EvaluationInfo]:
    paths = open_registry(home)
    reconcile_jobs(home)
    with connect(paths.db) as conn:
        rows = repositories.list_evaluations(conn, run_id)
    return [_info(paths, row) for row in rows]


def _ready(paths: AppPaths, evaluation_id: str) -> Any:
    with connect(paths.db) as conn:
        row = repositories.get_evaluation(conn, evaluation_id)
    if row is None:
        raise NotAvailableError(f"No evaluation {evaluation_id}.")
    row = _sync_status(paths, row)
    if row["status"] != "ready":
        raise FlipTunedError(
            f"Evaluation {evaluation_id} is {row['status']}; review and scoring start once both "
            "models' answers are generated.",
            hint=f"See `fliptuned jobs follow {row['job_id']}`." if row["job_id"] else None,
        )
    return row


# --------------------------------------------------------------------------- blind review


class ReviewPair(BaseModel):
    """One blind pair. Only "A" and "B": which model wrote which stays on the server."""

    evaluation_id: str
    item_id: str
    position: int = Field(description="1-based place in the review order.")
    category: str | None
    question: str
    reference: str = Field(description="The approved answer from the dataset.")
    answer_a: str
    answer_b: str
    judgement: Judgement | None = Field(description="The current judgement, if reviewed already.")


class ReviewRequest(BaseModel):
    home: str | None = None
    evaluation_id: str
    position: int | None = Field(
        default=None, ge=1, description="Show this item. Empty = the next unreviewed one."
    )


class ReviewQueue(BaseModel):
    evaluation_id: str
    total: int
    judged: int
    pair: ReviewPair | None = Field(description="None when every pair has been reviewed.")


def review_queue(request: ReviewRequest) -> ReviewQueue:
    paths = open_registry(request.home)
    row = _ready(paths, request.evaluation_id)
    with connect(paths.db) as conn:
        items = repositories.list_eval_items(conn, row["id"])
        judgements = {
            j["item_id"]: j["judgement"] for j in repositories.list_judgements(conn, row["id"])
        }
    if request.position is not None:
        chosen = next((i for i in items if i["position"] == request.position), None)
        if chosen is None:
            raise NotAvailableError(
                f"Evaluation {row['id']} has no item {request.position} (it has {len(items)})."
            )
    else:
        chosen = next((i for i in items if i["item_id"] not in judgements), None)
    queue = ReviewQueue(
        evaluation_id=row["id"], total=len(items), judged=len(judgements), pair=None
    )
    if chosen is None:
        return queue
    results = {
        r["item_id"]: r
        for r in _read_jsonl(Path(row["eval_dir"]) / RESULTS_FILE)
        if r["set"] == "test"
    }
    result = results[chosen["item_id"]]
    answer_a, answer_b = _answers_ab(result, chosen["a_model"])
    queue.pair = ReviewPair(
        evaluation_id=row["id"],
        item_id=chosen["item_id"],
        position=chosen["position"],
        category=chosen["category"],
        question=result["question"],
        reference=result["reference"],
        answer_a=answer_a,
        answer_b=answer_b,
        judgement=judgements.get(chosen["item_id"]),
    )
    return queue


class JudgeRequest(BaseModel):
    home: str | None = None
    evaluation_id: str
    item_id: str
    judgement: Judgement = Field(description="A, B or tie.")


class JudgeResult(BaseModel):
    evaluation_id: str
    item_id: str
    total: int
    judged: int


def _record_judgements(
    paths: AppPaths,
    evaluation_id: str,
    judgements: list[tuple[str, Judgement]],
    origin: Literal["ui", "csv"],
) -> tuple[int, int]:
    """Store judgements (the judgement service); returns (total pairs, judged pairs)."""
    with connect(paths.db) as conn:
        known = {i["item_id"] for i in repositories.list_eval_items(conn, evaluation_id)}
        unknown = [item for item, _ in judgements if item not in known]
        if unknown:
            raise NotAvailableError(
                f"Evaluation {evaluation_id} has no item {unknown[0]}.",
                hint="Judge items from this evaluation's review queue or review sheet.",
            )
        repositories.upsert_judgements(
            conn, evaluation_id, [(item, j, origin) for item, j in judgements]
        )
        judged = len(repositories.list_judgements(conn, evaluation_id))
    return len(known), judged


def judge(request: JudgeRequest) -> JudgeResult:
    paths = open_registry(request.home)
    row = _ready(paths, request.evaluation_id)
    total, judged = _record_judgements(
        paths, row["id"], [(request.item_id, request.judgement)], origin="ui"
    )
    return JudgeResult(evaluation_id=row["id"], item_id=request.item_id, total=total, judged=judged)


class SheetImportRequest(BaseModel):
    home: str | None = None
    path: str = Field(description="review_sheet.csv with the judgement column filled in.")
    run_id: str | None = Field(
        default=None, description="If given, the sheet must belong to an evaluation of this run."
    )


class SheetImportResult(BaseModel):
    evaluation_id: str
    rows: int
    recorded: dict[str, int] = Field(description="Judgements stored, by A / B / tie.")
    blank: int = Field(description="Rows with an empty judgement (not reviewed yet).")
    skipped: dict[str, int]
    problems: list[ImportProblem] = Field(description=f"First {_MAX_PROBLEMS} skipped rows.")


def _read_sheet(path: Path) -> list[tuple[int, dict[str, str]]]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in _SHEET_REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(
                    f"{path.name} is missing column(s): {', '.join(missing)}.",
                    hint="Use the review_sheet.csv written by the evaluation.",
                )
            return list(enumerate(reader, start=2))  # spreadsheet row numbers
    except FileNotFoundError:
        raise ConfigError(f"File not found: {path}", hint="Check the path.") from None
    except UnicodeDecodeError:
        raise ConfigError(f"{path.name} is not UTF-8.", hint="Save it as UTF-8 CSV.") from None
    except csv.Error as exc:
        raise ConfigError(f"{path.name} is not a valid CSV file.", details=str(exc)) from None


def import_review_sheet(request: SheetImportRequest) -> SheetImportResult:
    """The CSV adapter of the judgement service: reads only evaluation_id, item_id, judgement."""
    path = Path(request.path).expanduser().resolve()
    rows = _read_sheet(path)
    evaluation_ids = {(r.get("evaluation_id") or "").strip() for _, r in rows} - {""}
    if len(evaluation_ids) != 1:
        raise ConfigError(
            f"{path.name} must hold the items of exactly one evaluation "
            f"(found {len(evaluation_ids)}).",
            hint="Use the review_sheet.csv written by the evaluation, one sheet per evaluation.",
        )
    paths = open_registry(request.home)
    evaluation = _ready(paths, evaluation_ids.pop())
    if request.run_id is not None and evaluation["run_id"] != request.run_id:
        raise ConfigError(
            f"This sheet belongs to evaluation {evaluation['id']} of run {evaluation['run_id']}, "
            f"not run {request.run_id}.",
            hint=f"Use --run {evaluation['run_id']}.",
        )
    with connect(paths.db) as conn:
        known = {i["item_id"] for i in repositories.list_eval_items(conn, evaluation["id"])}
    recorded: dict[str, Judgement] = {}
    problems: list[ImportProblem] = []
    blank = 0
    for line, row in rows:
        word = (row.get("judgement") or "").strip()
        if not word:
            blank += 1
            continue
        judgement = parse_judgement(word)
        item_id = (row.get("item_id") or "").strip()
        problem = None
        if judgement is None:
            problem = "invalid_judgement"
        elif item_id not in known:
            problem = "unknown_item"
        if problem:
            problems.append(ImportProblem(row=line, reason=problem))
            continue
        assert judgement is not None
        recorded[item_id] = judgement  # a later row for the same item wins
    if recorded:
        _record_judgements(paths, evaluation["id"], list(recorded.items()), origin="csv")
    return SheetImportResult(
        evaluation_id=evaluation["id"],
        rows=len(rows),
        recorded=dict(sorted(Counter(recorded.values()).items())),
        blank=blank,
        skipped=dict(sorted(Counter(p.reason for p in problems).items())),
        problems=problems[:_MAX_PROBLEMS],
    )


# --------------------------------------------------------------------------- scoring


class ScoreRequest(BaseModel):
    home: str | None = None
    evaluation_id: str


class ScoreResult(BaseModel):
    evaluation_id: str
    run_id: str
    verdict: Literal["accepted", "rejected", "inconclusive"]
    reasons: list[str]
    criteria: list[Criterion]
    test_items: int
    judged: int
    review_coverage: float
    wins: int = Field(description="Pairs the fine-tuned model won.")
    losses: int = Field(description="Pairs the base model won.")
    ties: int
    sign_test_p: float
    categories: list[CategoryResult]
    deterministic_pass_base: float
    deterministic_pass_tuned: float
    regression_pass_base: float | None
    regression_pass_tuned: float | None
    regressions: int = Field(description="Items where the fine-tuned answer lost a check.")
    prior_exposure_note: str | None
    limitations: list[str]
    eval_dir: str
    report_path: str


def _prior_exposure(paths: AppPaths, evaluation_id: str) -> PriorExposure:
    with connect(paths.db) as conn:
        shared = repositories.earlier_shared_items(conn, evaluation_id)
    earlier = sorted({r["evaluation_id"] for r in shared})
    hashes = {r["content_hash"] for r in shared}
    note = None
    if earlier:
        note = (
            f"{len(hashes)} of these test items were already in {len(earlier)} earlier "
            "evaluation(s). If anything (data, review decisions, settings) was changed after "
            "seeing those results, this evaluation is not independent evidence of improvement."
        )
    return PriorExposure(earlier_evaluations=earlier, shared_items=len(hashes), note=note)


def _limitations(
    metrics: EvaluationMetrics, config: EvaluateConfig, exposure: PriorExposure
) -> list[str]:
    n, minimum = metrics.test_items, config.acceptance.min_test_items
    lines = [
        f"The test set has {n} items"
        + (f", fewer than the {minimum} a verdict needs." if n < minimum else "."),
        "No generated code was run: Python and JSON code blocks were parsed only, and other "
        "languages get no syntax check.",
        "The automatic checks look at an answer's shape; correctness comes from the blind review.",
        "Fine-tuning shifts style and recurring patterns; it doesn't reliably add facts, and the "
        "model can't check its answers against sources.",
        "Falling training loss alone is not evidence that the model improved.",
    ]
    if metrics.limited_to is not None:
        lines.append(
            f"Only the first {metrics.limited_to} test items were evaluated (--limit); evaluate "
            "the full test set for a verdict you rely on."
        )
    if exposure.note:
        lines.append(exposure.note)
    return lines


def score_evaluation(request: ScoreRequest) -> ScoreResult:
    """The one scoring service: judgements (from the UI or a sheet) + checks → verdict and
    ``metrics.json``, ``verdict.json``, ``report.md``."""
    paths = open_registry(request.home)
    evaluation = _ready(paths, request.evaluation_id)
    evaluation_id = evaluation["id"]
    eval_dir = Path(evaluation["eval_dir"])
    config = EvaluateConfig.model_validate(json.loads(evaluation["config_json"]))
    meta = json.loads((eval_dir / METADATA_FILE).read_text(encoding="utf-8"))
    rows = _read_jsonl(eval_dir / RESULTS_FILE)
    test_rows = [r for r in rows if r["set"] == "test"]
    regression_rows = [r for r in rows if r["set"] == "regression"]
    with connect(paths.db) as conn:
        items = repositories.list_eval_items(conn, evaluation_id)
        judgements = repositories.list_judgements(conn, evaluation_id)

    a_models = {i["item_id"]: i["a_model"] for i in items}
    outcomes: dict[str, Outcome] = {
        j["item_id"]: unblind(j["judgement"], a_models[j["item_id"]]) for j in judgements
    }
    counts = Counter(outcomes.values())
    categories = category_results({i["item_id"]: i["category"] for i in items}, outcomes)
    deterministic = check_summary(test_rows)
    regression = check_summary(regression_rows) if regression_rows else None
    inputs = VerdictInputs(
        test_items=len(items),
        judged=len(outcomes),
        wins=counts["tuned"],
        losses=counts["base"],
        ties=counts["tie"],
        categories=categories,
        deterministic_pass_base=deterministic.base.pass_rate,
        deterministic_pass_tuned=deterministic.tuned.pass_rate,
        regression_pass_base=regression.base.pass_rate if regression else None,
        regression_pass_tuned=regression.tuned.pass_rate if regression else None,
    )
    verdict = decide(inputs, config.acceptance)
    exposure = _prior_exposure(paths, evaluation_id)
    metrics = EvaluationMetrics(
        evaluation_id=evaluation_id,
        run_id=evaluation["run_id"],
        dataset_id=meta["dataset_id"],
        test_items=inputs.test_items,
        judged=inputs.judged,
        review_coverage=verdict.review_coverage,
        wins=inputs.wins,
        losses=inputs.losses,
        ties=inputs.ties,
        judgements_by_origin=dict(sorted(Counter(j["origin"] for j in judgements).items())),
        sign_test_p=verdict.sign_test_p,
        categories=categories,
        deterministic=deterministic,
        regression_set=regression,
        regressions=regressions(rows),
        generation={side: generation_stats(rows, side) for side in ("base", "tuned")},
        prior_exposure=exposure,
        limited_to=meta.get("limit"),
    )
    limitations = _limitations(metrics, config, exposure)
    scored_at = utc_now()
    _write_json(
        eval_dir / METRICS_FILE, {"scored_at": scored_at, **metrics.model_dump(mode="json")}
    )
    _write_json(
        eval_dir / VERDICT_FILE,
        {
            "evaluation_id": evaluation_id,
            "run_id": evaluation["run_id"],
            "scored_at": scored_at,
            **verdict.model_dump(mode="json"),
            "prior_exposure": exposure.model_dump(mode="json"),
            "limitations": limitations,
        },
    )
    report_path = eval_dir / REPORT_FILE
    report_path.write_bytes(render_report(metrics, verdict, meta, limitations).encode("utf-8"))
    with connect(paths.db) as conn:
        repositories.update_evaluation(
            conn,
            evaluation_id,
            verdict=verdict.verdict,
            metrics={
                "verdict": verdict.verdict,
                "test_items": inputs.test_items,
                "judged": inputs.judged,
                "wins": inputs.wins,
                "losses": inputs.losses,
                "ties": inputs.ties,
                "sign_test_p": verdict.sign_test_p,
                "scored_at": scored_at,
            },
        )
    return ScoreResult(
        evaluation_id=evaluation_id,
        run_id=evaluation["run_id"],
        verdict=verdict.verdict,
        reasons=verdict.reasons,
        criteria=verdict.criteria,
        test_items=inputs.test_items,
        judged=inputs.judged,
        review_coverage=verdict.review_coverage,
        wins=inputs.wins,
        losses=inputs.losses,
        ties=inputs.ties,
        sign_test_p=verdict.sign_test_p,
        categories=categories,
        deterministic_pass_base=deterministic.base.pass_rate,
        deterministic_pass_tuned=deterministic.tuned.pass_rate,
        regression_pass_base=inputs.regression_pass_base,
        regression_pass_tuned=inputs.regression_pass_tuned,
        regressions=len(metrics.regressions),
        prior_exposure_note=exposure.note,
        limitations=limitations,
        eval_dir=str(eval_dir),
        report_path=str(report_path),
    )


EvaluationOutcome.model_rebuild()
