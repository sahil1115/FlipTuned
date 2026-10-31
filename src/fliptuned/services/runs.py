"""A training run's files, for the services that use a finished run (evaluation, chat, export).

Reads only the run's own records: the registry row, ``run_plan.json`` (effective config, backend
selection, system prompt) and ``metadata.json`` (result with the best adapter). Never the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fliptuned.config.app_home import AppPaths
from fliptuned.diagnostics.backend_select import BackendSelection
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.registry import repositories
from fliptuned.registry.db import connect

METADATA_FILE = "metadata.json"
PLAN_FILE = "run_plan.json"


@dataclass
class RunArtifacts:
    run_id: str
    status: str
    run_dir: Path
    plan: dict[str, Any]
    meta: dict[str, Any]
    adapter: Path | None

    @property
    def selection(self) -> BackendSelection:
        return BackendSelection.model_validate(self.plan["selection"])

    @property
    def catalog_id(self) -> str:
        return self.plan["config"]["model"]["catalog_id"]

    @property
    def system_prompt(self) -> str:
        return self.plan["system_prompt"]


def load_run(paths: AppPaths, run_id: str, *, require_adapter: bool = True) -> RunArtifacts:
    """The run's records; with ``require_adapter`` the run must have finished training."""
    with connect(paths.db) as conn:
        row = repositories.get_run(conn, run_id)
    if row is None:
        raise NotAvailableError(f"No run {run_id}.", hint="See `fliptuned jobs list`.")
    run_dir = Path(row["run_dir"])
    plan = json.loads((run_dir / PLAN_FILE).read_text(encoding="utf-8"))
    meta = json.loads((run_dir / METADATA_FILE).read_text(encoding="utf-8"))
    adapter_path = (meta.get("result") or {}).get("adapter_path")
    adapter = Path(adapter_path) if adapter_path and Path(adapter_path).exists() else None
    if require_adapter:
        if row["status"] != "succeeded":
            raise FlipTunedError(
                f"Run {run_id} is {row['status']}; this needs a run that finished training.",
                hint=f"Finish it with `fliptuned jobs resume {run_id}`, or train a new run.",
            )
        if adapter is None:
            raise NotAvailableError(
                f"Run {run_id} has no adapter.", hint=f"Check the run folder: {run_dir}"
            )
    return RunArtifacts(
        run_id=run_id,
        status=row["status"],
        run_dir=run_dir,
        plan=plan,
        meta=meta,
        adapter=adapter,
    )
