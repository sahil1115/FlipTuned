"""Run history (SPEC §14.2 Runs screen): list with filters, compare two runs, artifacts, archive and
hard delete.

Archiving hides a run with its evaluations and exports (it can be restored). A hard delete needs
the run ID typed back as confirmation, refuses while any of the run's jobs is active, and removes
the run's rows (evaluations, judgements, exports, jobs) and its files: the run folder (adapter,
checkpoints, evaluations, exports inside it) and its job folders. Exports written to a folder
outside the app home are left in place and listed in the result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.jobs.state import ACTIVE
from fliptuned.jobs.worker import job_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.services.jobs import reconcile_jobs
from fliptuned.services.runs import METADATA_FILE

ACTIVE_STATUSES = {s.value for s in ACTIVE}
# Files the UI may download from a run folder (text; the adapter and checkpoints stay on disk).
DOWNLOADABLE = {".json", ".jsonl", ".md", ".csv", ".yaml", ".yml", ".txt"}
MAX_DOWNLOAD_BYTES = 200 * 2**20
# Folders listed as one entry (their files aren't useful one by one).
COLLAPSED = {"adapter", "checkpoints", "merged"}


def _paths(home: str | None) -> AppPaths:
    paths = AppPaths(resolve_app_home(home).path)
    paths.ensure()
    migrate(paths.db)
    return paths


def _meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / METADATA_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class RunVerdict(BaseModel):
    evaluation_id: str
    verdict: str
    test_items: int | None = None
    judged: int | None = None
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None
    sign_test_p: float | None = None
    scored_at: str | None = None


class RunSummary(BaseModel):
    id: str
    status: str
    model_id: str
    backend: str
    device: str
    precision: str
    dataset_id: str | None
    config_name: str | None
    config_version: int | None
    created_at: str
    archived: bool
    best_eval_loss: float | None
    verdict: RunVerdict | None = Field(description="The latest scored evaluation.")
    evaluations: int
    exports: int


class RunFilter(BaseModel):
    home: str | None = None
    status: str | None = None
    model_id: str | None = None
    text: str | None = Field(default=None, description="Part of the run ID or config name.")
    include_archived: bool = False
    limit: int = Field(default=100, ge=1, le=1000)


def _latest_verdict(conn, run_id: str) -> tuple[RunVerdict | None, int]:
    rows = conn.execute(
        "SELECT id, verdict, metrics_json FROM evaluations WHERE run_id = ? "
        "ORDER BY created_at DESC, id DESC",
        (run_id,),
    ).fetchall()
    for row in rows:
        if row["verdict"]:
            metrics = json.loads(row["metrics_json"] or "{}")
            fields = {k: metrics.get(k) for k in RunVerdict.model_fields if k in metrics}
            fields.pop("verdict", None)
            return RunVerdict(evaluation_id=row["id"], verdict=row["verdict"], **fields), len(rows)
    return None, len(rows)


def _summary(conn, row) -> RunSummary:
    meta = _meta(Path(row["run_dir"]))
    verdict, evaluations = _latest_verdict(conn, row["id"])
    exports = conn.execute("SELECT COUNT(*) FROM exports WHERE run_id = ?", (row["id"],)).fetchone()
    return RunSummary(
        id=row["id"],
        status=row["status"],
        model_id=row["model_id"],
        backend=row["backend"],
        device=row["device"],
        precision=row["precision"],
        dataset_id=row["dataset_id"],
        config_name=meta.get("config_name"),
        config_version=meta.get("config_version"),
        created_at=row["created_at"],
        archived=row["archived_at"] is not None,
        best_eval_loss=(meta.get("result") or {}).get("best_eval_loss"),
        verdict=verdict,
        evaluations=evaluations,
        exports=int(exports[0]),
    )


def list_history(request: RunFilter) -> list[RunSummary]:
    paths = _paths(request.home)
    reconcile_jobs(request.home)
    with connect(paths.db) as conn:
        rows = repositories.list_runs_any(conn, 10_000, request.include_archived)
        out = []
        for row in rows:
            if request.status and row["status"] != request.status:
                continue
            if request.model_id and row["model_id"] != request.model_id:
                continue
            summary = _summary(conn, row)
            text = (request.text or "").strip().lower()
            if (
                text
                and text not in summary.id.lower()
                and text not in (summary.config_name or "").lower()
            ):
                continue
            out.append(summary)
            if len(out) >= request.limit:
                break
    return out


def _row(paths: AppPaths, run_id: str):
    with connect(paths.db) as conn:
        row = repositories.get_run_any(conn, run_id)
    if row is None:
        raise NotAvailableError(f"No run {run_id}.")
    return row


# --------------------------------------------------------------------------- compare


class ConfigDifference(BaseModel):
    key: str = Field(
        description="Dotted path in the effective config, e.g. training.learning_rate."
    )
    a: Any
    b: Any


class RunFigures(BaseModel):
    run: RunSummary
    best_step: int | None
    steps_done: int | None
    total_steps: int | None
    stopped_early: bool | None
    peak_memory_gb: float | None
    attempts: int
    train_seconds: float | None = Field(description="Time spent in training attempts.")


class RunComparison(BaseModel):
    a: RunFigures
    b: RunFigures
    config_differences: list[ConfigDifference]
    same_dataset: bool
    note: str | None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _seconds(attempts: list[dict[str, Any]]) -> float | None:
    from datetime import datetime

    total, seen = 0.0, False
    for a in attempts:
        if a.get("started_at") and a.get("finished_at"):
            start = datetime.fromisoformat(a["started_at"])
            end = datetime.fromisoformat(a["finished_at"])
            total += (end - start).total_seconds()
            seen = True
    return round(total, 1) if seen else None


def _figures(conn, row) -> RunFigures:
    meta = _meta(Path(row["run_dir"]))
    result = meta.get("result") or {}
    attempts = meta.get("attempts") or []
    return RunFigures(
        run=_summary(conn, row),
        best_step=result.get("best_step"),
        steps_done=result.get("steps_done"),
        total_steps=result.get("total_steps"),
        stopped_early=result.get("stopped_early"),
        peak_memory_gb=result.get("peak_memory_gb"),
        attempts=len(attempts),
        train_seconds=_seconds(attempts),
    )


def compare_runs(home: str | None, run_a: str, run_b: str) -> RunComparison:
    paths = _paths(home)
    if run_a == run_b:
        raise ConfigError("Pick two different runs to compare.")
    row_a, row_b = _row(paths, run_a), _row(paths, run_b)
    flat_a = _flatten(_meta(Path(row_a["run_dir"])).get("effective_config") or {})
    flat_b = _flatten(_meta(Path(row_b["run_dir"])).get("effective_config") or {})
    differences = [
        ConfigDifference(key=k, a=flat_a.get(k), b=flat_b.get(k))
        for k in sorted(set(flat_a) | set(flat_b))
        if k != "name" and flat_a.get(k) != flat_b.get(k)
    ]
    with connect(paths.db) as conn:
        a, b = _figures(conn, row_a), _figures(conn, row_b)
    same = row_a["dataset_id"] == row_b["dataset_id"]
    note = (
        None
        if same
        else (
            "The runs used different datasets, so their losses and verdicts aren't directly comparable."
        )
    )
    return RunComparison(a=a, b=b, config_differences=differences, same_dataset=same, note=note)


# --------------------------------------------------------------------------- artifacts


class Artifact(BaseModel):
    path: str = Field(description="Relative to the run folder, with / separators.")
    kind: str = Field(description="file or folder")
    bytes: int
    files: int
    downloadable: bool


class RunArtifactsList(BaseModel):
    run_id: str
    run_dir: str
    artifacts: list[Artifact]
    external_exports: list[str] = Field(description="Exports written outside the run folder.")


def _size(folder: Path) -> tuple[int, int]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files)


def list_artifacts(home: str | None, run_id: str) -> RunArtifactsList:
    paths = _paths(home)
    row = _row(paths, run_id)
    run_dir = Path(row["run_dir"])
    artifacts: list[Artifact] = []

    def walk(folder: Path) -> None:
        for entry in sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name)):
            rel = entry.relative_to(run_dir).as_posix()
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in COLLAPSED or entry.parent.name == "exports":
                    size, count = _size(entry)
                    artifacts.append(
                        Artifact(
                            path=rel, kind="folder", bytes=size, files=count, downloadable=False
                        )
                    )
                else:
                    walk(entry)
            else:
                size = entry.stat().st_size
                artifacts.append(
                    Artifact(
                        path=rel,
                        kind="file",
                        bytes=size,
                        files=1,
                        downloadable=entry.suffix.lower() in DOWNLOADABLE
                        and size <= MAX_DOWNLOAD_BYTES,
                    )
                )

    if run_dir.is_dir():
        walk(run_dir)
    return RunArtifactsList(
        run_id=run_id,
        run_dir=str(run_dir),
        artifacts=artifacts,
        external_exports=_external_exports(paths, run_id, run_dir),
    )


def _external_exports(paths: AppPaths, run_id: str, run_dir: Path) -> list[str]:
    with connect(paths.db) as conn:
        rows = conn.execute("SELECT path FROM exports WHERE run_id = ?", (run_id,)).fetchall()
    out = []
    for r in rows:
        p = Path(r["path"])
        if not p.resolve().is_relative_to(run_dir.resolve()):
            out.append(str(p))
    return out


def artifact_file(home: str | None, run_id: str, rel: str) -> Path:
    """A downloadable file inside the run folder (no escapes, no symlinks)."""
    paths = _paths(home)
    run_dir = Path(_row(paths, run_id)["run_dir"]).resolve()
    candidate = run_dir / rel
    if ".." in Path(rel).parts or Path(rel).is_absolute() or candidate.is_symlink():
        raise ConfigError("That file isn't part of this run.")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_dir) or not resolved.is_file():
        raise NotAvailableError("That file isn't part of this run.")
    if resolved.suffix.lower() not in DOWNLOADABLE or resolved.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ConfigError(
            "Only text artifacts (JSON, Markdown, CSV, YAML, text) can be downloaded here.",
            hint=f"The file is on this computer: {resolved}",
        )
    return resolved


# --------------------------------------------------------------------------- archive / delete


class ArchiveResult(BaseModel):
    run_id: str
    archived: bool


def _refuse_if_active(paths: AppPaths, run_id: str, action: str) -> None:
    reconcile_jobs(str(paths.home))
    with connect(paths.db) as conn:
        active = [
            j for j in repositories.run_job_ids(conn, run_id) if j["status"] in ACTIVE_STATUSES
        ]
    if active:
        job = active[0]
        raise FlipTunedError(
            f"Run {run_id} can't be {action} while its {job['type']} job {job['id']} is {job['status']}.",
            hint="Wait for it to finish, or cancel it first.",
        )


def set_archived(home: str | None, run_id: str, archived: bool) -> ArchiveResult:
    paths = _paths(home)
    _row(paths, run_id)
    if archived:
        _refuse_if_active(paths, run_id, "archived")
    with connect(paths.db) as conn:
        repositories.set_run_archived(conn, run_id, archived)
    return ArchiveResult(run_id=run_id, archived=archived)


class DeleteRequest(BaseModel):
    home: str | None = None
    run_id: str
    confirm: str = Field(description="The run ID typed again, to confirm the permanent delete.")


class DeleteResult(BaseModel):
    run_id: str
    removed_folders: list[str]
    kept_exports: list[str] = Field(description="Exports outside the app home, left in place.")


def delete_run(request: DeleteRequest) -> DeleteResult:
    paths = _paths(request.home)
    if request.confirm != request.run_id:
        raise ConfigError(
            "The confirmation doesn't match the run ID; nothing was deleted.",
            hint="Type the run ID exactly to delete it permanently.",
        )
    row = _row(paths, request.run_id)
    _refuse_if_active(paths, request.run_id, "deleted")
    run_dir = Path(row["run_dir"])
    kept = _external_exports(paths, request.run_id, run_dir)
    with connect(paths.db) as conn:
        jobs = [j["id"] for j in repositories.run_job_ids(conn, request.run_id)]
    folders = [run_dir, *(job_dir(paths, j) for j in jobs)]
    with connect(paths.db) as conn:
        repositories.purge_run(conn, request.run_id)
    removed = []
    for folder in folders:
        if folder.exists() and folder.resolve().is_relative_to(paths.home.resolve()):
            shutil.rmtree(folder)
            removed.append(str(folder))
    return DeleteResult(run_id=request.run_id, removed_folders=removed, kept_exports=kept)
