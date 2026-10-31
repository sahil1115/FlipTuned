"""``status``: registry summary for the app home (creates and migrates it if needed)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate


class StatusRequest(BaseModel):
    home: str | None = Field(default=None, description="App home override (--home).")


class ActiveJob(BaseModel):
    id: str
    type: str
    status: str
    created_at: str


class StatusResult(BaseModel):
    app_home: str
    app_home_source: str
    db_path: str
    schema_version: int
    migrated_from: int | None
    backup_path: str | None
    counts: dict[str, int]
    active_jobs: list[ActiveJob]
    disk_use_mb: dict[str, float]


def get_status(request: StatusRequest) -> StatusResult:
    resolved = resolve_app_home(request.home)
    paths = AppPaths(resolved.path)
    paths.ensure()
    result = migrate(paths.db)
    LocalJobRunner(paths).reconcile()  # jobs whose process is gone become 'interrupted'
    with connect(paths.db) as conn:
        counts = {
            table: repositories.count_rows(conn, table) for table in repositories.COUNTED_TABLES
        }
        jobs = [ActiveJob(**dict(row)) for row in repositories.active_jobs(conn)]
    return StatusResult(
        app_home=str(paths.home),
        app_home_source=resolved.source.value,
        db_path=str(paths.db),
        schema_version=result.to_version,
        migrated_from=result.from_version if result.from_version != result.to_version else None,
        backup_path=str(result.backup_path) if result.backup_path else None,
        counts=counts,
        active_jobs=jobs,
        disk_use_mb={p.name: _size_mb(p) for p in paths.subdirs()},
    )


def _size_mb(directory: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            with contextlib.suppress(OSError):
                total += (Path(root) / name).stat().st_size
    return round(total / 1e6, 1)
