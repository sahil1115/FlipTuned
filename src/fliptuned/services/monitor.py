"""Run monitor (SPEC §14.2): a training run's live state, rebuilt from the registry and the
``events.jsonl`` of every attempt (the first job and each resume).

The snapshot holds everything the Run monitor screen shows: status, step and ETA, the loss curve
across attempts with the best checkpoint, memory for the run's device, warnings, a log of events
and whether Cancel or Resume is possible (and the model-lock holder if another job blocks it).
``offset`` and ``last_seq`` say where the latest attempt's events were read up to, so a client
continues with the job's SSE stream from there, applying the same rules as :class:`_Fold`.
Nothing here reads records or job logs: events carry metrics and content-free messages only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.errors import NotAvailableError
from fliptuned.jobs import lock
from fliptuned.jobs.events import read_events
from fliptuned.jobs.state import ACTIVE, JobType
from fliptuned.jobs.worker import EVENTS_FILE, job_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.jobs import JobInfo, _info, get_job, job_error, reconcile_jobs
from fliptuned.services.runs import METADATA_FILE, PLAN_FILE
from fliptuned.services.training import RESUMABLE, RunInfo, get_run
from fliptuned.training.backends import get_backend

LOG_LINES = 300
ACTIVE_STATUSES = {s.value for s in ACTIVE}


class LossPoint(BaseModel):
    step: int = Field(description="Optimizer update.")
    loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    job_id: str = Field(description="The attempt (job) that reported it.")


class BestCheckpoint(BaseModel):
    step: int
    eval_loss: float
    final: bool = Field(description="True once training chose it; otherwise the best so far.")


class LogLine(BaseModel):
    seq: int
    time: str
    job_id: str
    level: str = Field(description="info | warning | error")
    message: str


class LockHolder(BaseModel):
    job_id: str
    type: str
    status: str
    run_id: str | None
    message: str


class MemoryInfo(BaseModel):
    pool: str = Field(description="VRAM, unified memory or RAM, after the run's device.")
    peak_gb: float | None = Field(description="Peak use reported by the training job.")
    total_gb: float | None = Field(description="Size of that pool on the machine it ran on.")


class RunMonitor(BaseModel):
    run: RunInfo
    config_name: str | None
    config_version: int | None
    attempts: list[JobInfo] = Field(description="Training jobs of this run, oldest first.")
    job: JobInfo | None = Field(description="The latest attempt.")
    offset: int = Field(description="Byte offset into the latest attempt's events (SSE ID).")
    last_seq: int = Field(description="Last event seq read from the latest attempt.")
    step: int | None
    max_steps: int | None
    epoch: float | None
    eta_s: float | None = Field(description="Seconds left at the last report.")
    eta_at: str | None = Field(description="Time of that report.")
    points: list[LossPoint]
    best: BestCheckpoint | None
    memory: MemoryInfo
    warnings: list[str]
    log: list[LogLine] = Field(description=f"The last {LOG_LINES} event lines, oldest first.")
    error: dict | None = Field(description="The latest attempt's error (message / hint).")
    result: dict | None = Field(description="Training result summary, once an attempt ended.")
    can_cancel: bool
    can_resume: bool
    resume_note: str | None = Field(description="Why Resume isn't possible, if it isn't.")
    lock_holder: LockHolder | None = Field(
        description="Another job holding the model lock (blocks Resume and new runs)."
    )


def memory_pool(device: str) -> str:
    return {"cuda": "VRAM", "apple_gpu": "unified memory"}.get(device, "RAM")


def _total_gb(device: str, machine: dict[str, Any]) -> float | None:
    if device == "cuda":
        gpus = machine.get("gpus") or []
        return gpus[0].get("memory_total_gb") if gpus else None
    if device == "apple_gpu":
        return machine.get("unified_memory_gb")
    return machine.get("ram_gb")


def _fmt(value: float) -> str:
    return f"{value:.6g}"


class _Fold:
    """Applies events in order. The web UI's ``applyEvent`` mirrors these rules."""

    def __init__(self) -> None:
        self.points: list[LossPoint] = []
        self.log: list[LogLine] = []
        self.warnings: list[str] = []
        self.step: int | None = None
        self.max_steps: int | None = None
        self.epoch: float | None = None
        self.eta_s: float | None = None
        self.eta_at: str | None = None
        self.peak_gb: float | None = None

    def line(self, event: dict, job_id: str, level: str, message: str) -> None:
        self.log.append(
            LogLine(
                seq=event["seq"], time=event["time"], job_id=job_id, level=level, message=message
            )
        )

    def apply(self, event: dict, job_id: str) -> None:
        kind = event.get("type")
        if kind == "progress":
            self.step, self.max_steps = event.get("done"), event.get("total") or self.max_steps
        elif kind == "train":
            self._train(event, job_id)
        elif kind == "log":
            level = str(event.get("level") or "info")
            message = str(event.get("message") or "")
            if level == "warning":
                self.warnings.append(message)
            self.line(event, job_id, level, message)
        elif kind == "status":
            status = str(event.get("status"))
            error = event.get("error") or {}
            text = f"Job {job_id} is {status}."
            if error.get("message"):
                text += f" {error['message']}"
            if event.get("forced"):
                text += " (force-stopped)"
            self.line(event, job_id, "error" if status == "failed" else "info", text)

    def _train(self, event: dict, job_id: str) -> None:
        step = int(event["step"])
        if self.points and self.points[-1].job_id != job_id:
            # A resumed attempt restarts from its checkpoint: drop what came after it.
            self.points = [p for p in self.points if p.step < step]
        point = LossPoint(
            step=step,
            loss=event.get("loss"),
            eval_loss=event.get("eval_loss"),
            learning_rate=event.get("learning_rate"),
            job_id=job_id,
        )
        self.points.append(point)
        self.step, self.max_steps = step, event.get("max_steps") or self.max_steps
        self.epoch = event.get("epoch", self.epoch)
        if event.get("eta_s") is not None:
            self.eta_s, self.eta_at = event["eta_s"], event["time"]
        if event.get("memory_gb") is not None:
            self.peak_gb = max(self.peak_gb or 0.0, float(event["memory_gb"]))
        total = f"/{self.max_steps}" if self.max_steps else ""
        if point.eval_loss is not None:
            self.line(
                event,
                job_id,
                "info",
                f"Step {step}{total}: validation loss {_fmt(point.eval_loss)}",
            )
        if point.loss is not None:
            self.line(event, job_id, "info", f"Step {step}{total}: train loss {_fmt(point.loss)}")

    def best(self) -> BestCheckpoint | None:
        best = None
        for p in self.points:
            if p.eval_loss is not None and (best is None or p.eval_loss < best.eval_loss):
                best = p
        return (
            None
            if best is None
            else BestCheckpoint(step=best.step, eval_loss=best.eval_loss, final=False)
        )


def run_monitor(home: str | None, run_id: str) -> RunMonitor:
    paths = AppPaths(resolve_app_home(home).path)
    reconcile_jobs(home)  # a worker that died (or a restart) shows as interrupted
    run = get_run(home, run_id)
    run_dir = Path(run.run_dir)
    meta = _read_json(run_dir / METADATA_FILE)
    with connect(paths.db) as conn:
        attempts = [_info(r) for r in repositories.list_run_jobs(conn, run_id, JobType.TRAIN.value)]
        holder_id = lock.current_holder(conn)

    fold = _Fold()
    fold.warnings.extend(meta.get("preflight_warnings") or [])
    offset, last_seq = 0, 0
    for job in attempts:
        events, offset = read_events(job_dir(paths, job.id) / EVENTS_FILE)
        for event in events:
            fold.apply(event, job.id)
        last_seq = events[-1]["seq"] if events else 0
    latest = attempts[-1] if attempts else None

    result = meta.get("result")
    best = fold.best()
    if result and result.get("best_step") is not None and result.get("best_eval_loss") is not None:
        best = BestCheckpoint(
            step=result["best_step"], eval_loss=result["best_eval_loss"], final=True
        )
    for note in (result or {}).get("notes") or []:
        if note not in fold.warnings:
            fold.warnings.append(note)

    own = {job.id for job in attempts}
    lock_holder = None
    if holder_id and holder_id not in own:
        holder = get_job(home, holder_id)
        lock_holder = LockHolder(
            job_id=holder.id,
            type=holder.type,
            status=holder.status,
            run_id=holder.run_id,
            message=f"Another job is using the model: {holder.id} ({holder.type}, {holder.status}).",
        )
    can_resume, resume_note = _resumable(run, run_dir, lock_holder)

    return RunMonitor(
        run=run,
        config_name=meta.get("config_name"),
        config_version=meta.get("config_version"),
        attempts=attempts,
        job=latest,
        offset=offset,
        last_seq=last_seq,
        step=fold.step,
        max_steps=fold.max_steps,
        epoch=fold.epoch,
        eta_s=fold.eta_s if latest and latest.status == "running" else None,
        eta_at=fold.eta_at,
        points=fold.points,
        best=best,
        memory=MemoryInfo(
            pool=memory_pool(run.device),
            peak_gb=(result or {}).get("peak_memory_gb") or fold.peak_gb,
            total_gb=_total_gb(run.device, meta.get("machine") or {}),
        ),
        warnings=fold.warnings,
        log=fold.log[-LOG_LINES:],
        error=job_error(home, latest.id) if latest else None,
        result=_result_summary(result),
        can_cancel=bool(latest and latest.status in ACTIVE_STATUSES),
        can_resume=can_resume,
        resume_note=resume_note,
        lock_holder=lock_holder,
    )


def _resumable(run: RunInfo, run_dir: Path, holder: LockHolder | None) -> tuple[bool, str | None]:
    if run.status not in RESUMABLE:
        return False, None
    plan = _read_json(run_dir / PLAN_FILE)
    try:
        backend = get_backend(BackendSelection.model_validate(plan["selection"]).backend)
        has_checkpoint = backend.has_checkpoint(run_dir)
    except (KeyError, ValueError, NotAvailableError):
        has_checkpoint = False
    if not has_checkpoint:
        return False, "This run has no checkpoint to resume from; start a new run instead."
    if holder is not None:
        return False, f"{holder.message} Resume when it has finished."
    return True, None


def _result_summary(result: dict | None) -> dict | None:
    if not result:
        return None
    keys = (
        "status",
        "best_step",
        "best_eval_loss",
        "steps_done",
        "total_steps",
        "stopped_early",
        "peak_memory_gb",
        "adapter_path",
        "resume_note",
    )
    return {k: result.get(k) for k in keys}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
