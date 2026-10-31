"""What each job type runs. Handlers call services and return a JSON-ready result.

``ingest`` and ``build_dataset`` write their outputs atomically (temp file or folder, then
rename), so a cancel that has to force-stop them leaves no half-written data.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fliptuned.jobs.state import JobType
from fliptuned.jobs.worker import JobContext


def _ingest(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.ingest import IngestRequest, run_ingest

    return run_ingest(IngestRequest(home=ctx.home, **ctx.params)).model_dump(mode="json")


def _build_dataset(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.datasets import BuildDatasetRequest, build_dataset

    return build_dataset(BuildDatasetRequest(home=ctx.home, **ctx.params)).model_dump(mode="json")


def _selftest(ctx: JobContext) -> dict[str, Any]:
    """Diagnostic job: works in steps, reports progress and honours cancel. ``ignore_cancel``
    simulates a stuck worker so force-stopping can be checked."""
    steps = int(ctx.params.get("steps", 10))
    seconds = float(ctx.params.get("seconds", 10.0))
    ignore_cancel = bool(ctx.params.get("ignore_cancel", False))
    for step in range(1, steps + 1):
        deadline = time.monotonic() + seconds / steps
        while time.monotonic() < deadline:
            if not ignore_cancel:
                ctx.check_cancel()
            time.sleep(0.05)
        ctx.progress(step, steps, f"step {step} of {steps}")
    return {"steps": steps, "seconds": seconds}


def _train(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.training import execute_training

    return execute_training(ctx)


def _evaluate(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.evaluation import execute_evaluation

    return execute_evaluation(ctx)


def _chat_session(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.chat import serve_chat

    return serve_chat(ctx)


def _merge_export(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.export import execute_merge

    return execute_merge(ctx)


def _model_download(ctx: JobContext) -> dict[str, Any]:
    from fliptuned.services.models import execute_model_download

    return execute_model_download(ctx)


HANDLERS: dict[JobType, Callable[[JobContext], dict[str, Any]]] = {
    JobType.TRAIN: _train,
    JobType.MODEL_DOWNLOAD: _model_download,
    JobType.EVALUATE: _evaluate,
    JobType.CHAT_SESSION: _chat_session,
    JobType.MERGE_EXPORT: _merge_export,
    JobType.INGEST: _ingest,
    JobType.BUILD_DATASET: _build_dataset,
    JobType.SELFTEST: _selftest,
}
