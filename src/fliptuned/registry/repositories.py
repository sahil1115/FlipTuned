"""Registry repositories. Only what the current phase uses lives here."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from fliptuned.registry.db import utc_now

WORKSPACE_ID = "local"
CREATED_BY = "local-user"

COUNTED_TABLES = ("data_sources", "datasets", "runs", "jobs", "evaluations", "exports")
ACTIVE_JOB_STATUSES = ("queued", "running")


def count_rows(conn: sqlite3.Connection, table: str, *, include_archived: bool = False) -> int:
    if table not in COUNTED_TABLES:
        raise ValueError(f"unknown table {table!r}")
    where = "workspace_id = ?"
    has_archive = table != "jobs"
    if has_archive and not include_archived:
        where += " AND archived_at IS NULL"
    return int(
        conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", (WORKSPACE_ID,)).fetchone()[0]
    )


def active_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
    return conn.execute(
        f"SELECT id, type, status, created_at FROM jobs WHERE workspace_id = ? AND status IN ({placeholders}) "
        "ORDER BY created_at",
        (WORKSPACE_ID, *ACTIVE_JOB_STATUSES),
    ).fetchall()


def get_setting(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute(
        "SELECT value_json FROM settings WHERE workspace_id = ? AND key = ?", (WORKSPACE_ID, key)
    ).fetchone()
    return default if row is None else json.loads(row["value_json"])


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO settings (workspace_id, key, value_json, created_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (workspace_id, key) DO UPDATE SET value_json = excluded.value_json, "
            "updated_at = excluded.updated_at",
            (WORKSPACE_ID, key, json.dumps(value), CREATED_BY, now, now),
        )


# --------------------------------------------------------------------------- data sources


def data_source_id(name: str) -> str:
    """Stable ID for a named data source (scan and ingest with the same name share it)."""
    digest = hashlib.sha256(f"{WORKSPACE_ID}\x1f{name.strip().lower()}".encode()).hexdigest()
    return f"src_{digest[:12]}"


def upsert_data_source(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    name: str,
    root_path: str,
    status: str,
    stats: dict[str, Any],
    config: dict[str, Any] | None = None,
    kind: str = "folder",
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO data_sources (id, workspace_id, created_by, name, kind, root_path, config_json, "
            "status, stats_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET name = excluded.name, root_path = excluded.root_path, "
            "config_json = COALESCE(excluded.config_json, data_sources.config_json), "
            "status = excluded.status, stats_json = excluded.stats_json, "
            "updated_at = excluded.updated_at, archived_at = NULL",
            (
                source_id,
                WORKSPACE_ID,
                CREATED_BY,
                name,
                kind,
                root_path,
                json.dumps(config) if config is not None else None,
                status,
                json.dumps(stats),
                now,
                now,
            ),
        )


def list_data_sources(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, name, kind, root_path, status, stats_json, created_at, updated_at FROM data_sources "
        "WHERE workspace_id = ? AND archived_at IS NULL ORDER BY name",
        (WORKSPACE_ID,),
    ).fetchall()


# --------------------------------------------------------------------------- review decisions


def insert_review_decisions(conn: sqlite3.Connection, decisions: list[dict[str, Any]]) -> None:
    """Append decisions in one transaction. Each dict: source_id, record_id, content_hash,
    status, reason, origin. History is kept; the latest row per record wins."""
    now = utc_now()
    with conn:
        conn.executemany(
            "INSERT INTO review_decisions (workspace_id, created_by, source_id, record_id, "
            "content_hash, status, reason, origin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    WORKSPACE_ID,
                    CREATED_BY,
                    d["source_id"],
                    d["record_id"],
                    d["content_hash"],
                    d["status"],
                    d.get("reason"),
                    d["origin"],
                    now,
                )
                for d in decisions
            ],
        )


def latest_review_decisions(
    conn: sqlite3.Connection, source_ids: list[str]
) -> dict[tuple[str, str], sqlite3.Row]:
    """Latest decision per (source_id, record_id) for the given sources."""
    if not source_ids:
        return {}
    placeholders = ",".join("?" for _ in source_ids)
    rows = conn.execute(
        "SELECT d.source_id, d.record_id, d.content_hash, d.status, d.reason, d.origin, d.created_at "
        "FROM review_decisions d JOIN (SELECT MAX(id) AS id FROM review_decisions "
        f"WHERE workspace_id = ? AND source_id IN ({placeholders}) GROUP BY source_id, record_id) "
        "latest ON latest.id = d.id",
        (WORKSPACE_ID, *source_ids),
    ).fetchall()
    return {(r["source_id"], r["record_id"]): r for r in rows}


# --------------------------------------------------------------------------- datasets


def upsert_dataset(
    conn: sqlite3.Connection,
    *,
    dataset_id: str,
    name: str,
    status: str,
    path: str,
    config: dict[str, Any],
    manifest_hash: str,
    fingerprint: str,
    stats: dict[str, Any],
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO datasets (id, workspace_id, created_by, name, status, path, config_json, "
            "manifest_hash, fingerprint, stats_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET name = excluded.name, status = excluded.status, "
            "path = excluded.path, config_json = excluded.config_json, "
            "manifest_hash = excluded.manifest_hash, fingerprint = excluded.fingerprint, "
            "stats_json = excluded.stats_json, updated_at = excluded.updated_at, archived_at = NULL",
            (
                dataset_id,
                WORKSPACE_ID,
                CREATED_BY,
                name,
                status,
                path,
                json.dumps(config),
                manifest_hash,
                fingerprint,
                json.dumps(stats),
                now,
                now,
            ),
        )


_DATASET_COLUMNS = (
    "id, name, status, path, manifest_hash, fingerprint, stats_json, created_at, updated_at"
)


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {_DATASET_COLUMNS} FROM datasets WHERE workspace_id = ? AND id = ? "
        "AND archived_at IS NULL",
        (WORKSPACE_ID, dataset_id),
    ).fetchone()


def find_dataset(conn: sqlite3.Connection, id_or_name: str) -> sqlite3.Row | None:
    """By ID, else the most recently updated dataset with that name."""
    return (
        get_dataset(conn, id_or_name)
        or conn.execute(
            f"SELECT {_DATASET_COLUMNS} FROM datasets WHERE workspace_id = ? AND name = ? "
            "AND archived_at IS NULL ORDER BY updated_at DESC, id LIMIT 1",
            (WORKSPACE_ID, id_or_name),
        ).fetchone()
    )


def list_datasets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT {_DATASET_COLUMNS} FROM datasets WHERE workspace_id = ? AND archived_at IS NULL "
        "ORDER BY updated_at DESC, id",
        (WORKSPACE_ID,),
    ).fetchall()


# --------------------------------------------------------------------------- baselines


def upsert_baseline(
    conn: sqlite3.Connection,
    *,
    baseline_id: str,
    dataset_id: str,
    model_id: str,
    backend: str,
    device: str,
    precision: str,
    status: str,
    path: str,
    metadata: dict[str, Any],
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO baselines (id, workspace_id, created_by, dataset_id, model_id, backend, "
            "device, precision, status, path, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET status = excluded.status, path = excluded.path, "
            "metadata_json = excluded.metadata_json, updated_at = excluded.updated_at, "
            "archived_at = NULL",
            (
                baseline_id,
                WORKSPACE_ID,
                CREATED_BY,
                dataset_id,
                model_id,
                backend,
                device,
                precision,
                status,
                path,
                json.dumps(metadata),
                now,
                now,
            ),
        )


def get_baseline(conn: sqlite3.Connection, baseline_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, dataset_id, model_id, backend, device, precision, status, path, "
        "metadata_json, created_at, updated_at FROM baselines "
        "WHERE workspace_id = ? AND id = ? AND archived_at IS NULL",
        (WORKSPACE_ID, baseline_id),
    ).fetchone()


# --------------------------------------------------------------------------- jobs
# These functions don't commit: the job runner groups them into transactions.

JOB_COLUMNS = (
    "id, type, status, pid, pid_create_time, params_json, progress, log_path, events_path, "
    "error_summary, run_id, created_at, started_at, finished_at"
)


def insert_job(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    job_type: str,
    params: dict[str, Any],
    log_path: str,
    events_path: str,
    run_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO jobs (id, workspace_id, created_by, type, status, params_json, progress, "
        "log_path, events_path, run_id, created_at) VALUES (?, ?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)",
        (
            job_id,
            WORKSPACE_ID,
            CREATED_BY,
            job_type,
            json.dumps(params),
            log_path,
            events_path,
            run_id,
            utc_now(),
        ),
    )


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {JOB_COLUMNS} FROM jobs WHERE workspace_id = ? AND id = ?", (WORKSPACE_ID, job_id)
    ).fetchone()


def list_jobs(
    conn: sqlite3.Connection, statuses: list[str] | None = None, limit: int = 50
) -> list[sqlite3.Row]:
    where, args = "workspace_id = ?", [WORKSPACE_ID]
    if statuses:
        where += f" AND status IN ({','.join('?' for _ in statuses)})"
        args += statuses
    return conn.execute(
        f"SELECT {JOB_COLUMNS} FROM jobs WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ?",
        (*args, limit),
    ).fetchall()


def list_run_jobs(conn: sqlite3.Connection, run_id: str, job_type: str) -> list[sqlite3.Row]:
    """A run's jobs of one type, oldest first (a training run's attempts)."""
    return conn.execute(
        f"SELECT {JOB_COLUMNS} FROM jobs WHERE workspace_id = ? AND run_id = ? AND type = ? "
        "ORDER BY rowid",  # insertion order: created_at has whole seconds only
        (WORKSPACE_ID, run_id, job_type),
    ).fetchall()


def set_job_process(conn: sqlite3.Connection, job_id: str, pid: int, create_time: float) -> None:
    conn.execute(
        "UPDATE jobs SET pid = ?, pid_create_time = ? WHERE id = ?", (pid, create_time, job_id)
    )


def transition_job(
    conn: sqlite3.Connection,
    job_id: str,
    to_status: str,
    *,
    from_statuses: tuple[str, ...],
    error_summary: str | None = None,
    started: bool = False,
    finished: bool = False,
    progress: float | None = None,
) -> bool:
    """Move a job to ``to_status`` only if it is currently in ``from_statuses`` (no races)."""
    sets, args = ["status = ?"], [to_status]
    if error_summary is not None:
        sets.append("error_summary = ?")
        args.append(error_summary)
    if started:
        sets.append("started_at = ?")
        args.append(utc_now())
    if finished:
        sets.append("finished_at = ?")
        args.append(utc_now())
    if progress is not None:
        sets.append("progress = ?")
        args.append(progress)
    placeholders = ",".join("?" for _ in from_statuses)
    cursor = conn.execute(
        f"UPDATE jobs SET {', '.join(sets)} WHERE id = ? AND status IN ({placeholders})",
        (*args, job_id, *from_statuses),
    )
    return cursor.rowcount == 1


def set_job_progress(conn: sqlite3.Connection, job_id: str, progress: float) -> None:
    conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (progress, job_id))


def get_model_lock(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT job_id, pid, pid_create_time, acquired_at FROM model_lock WHERE id = 1"
    ).fetchone()


def insert_model_lock(conn: sqlite3.Connection, job_id: str, pid: int, create_time: float) -> None:
    conn.execute(
        "INSERT INTO model_lock (id, workspace_id, created_by, job_id, pid, pid_create_time, "
        "acquired_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
        (WORKSPACE_ID, CREATED_BY, job_id, pid, create_time, utc_now()),
    )


def update_model_lock_process(
    conn: sqlite3.Connection, job_id: str, pid: int, create_time: float
) -> None:
    conn.execute(
        "UPDATE model_lock SET pid = ?, pid_create_time = ? WHERE id = 1 AND job_id = ?",
        (pid, create_time, job_id),
    )


def delete_model_lock(conn: sqlite3.Connection, job_id: str | None = None) -> None:
    if job_id is None:
        conn.execute("DELETE FROM model_lock WHERE id = 1")
    else:
        conn.execute("DELETE FROM model_lock WHERE id = 1 AND job_id = ?", (job_id,))


# --------------------------------------------------------------------------- training configs


def insert_training_config(
    conn: sqlite3.Connection,
    *,
    name: str,
    version: int,
    config: dict[str, Any],
    parent_id: int | None = None,
) -> int:
    with conn:
        cursor = conn.execute(
            "INSERT INTO training_configs (workspace_id, created_by, name, version, config_json, "
            "parent_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                WORKSPACE_ID,
                CREATED_BY,
                name,
                version,
                json.dumps(config, sort_keys=True),
                parent_id,
                utc_now(),
            ),
        )
    return int(cursor.lastrowid)


_CONFIG_COLUMNS = "id, name, version, config_json, parent_id, created_at"


def get_training_config(
    conn: sqlite3.Connection, name: str, version: int | None = None
) -> sqlite3.Row | None:
    if version is None:
        return conn.execute(
            f"SELECT {_CONFIG_COLUMNS} FROM training_configs WHERE workspace_id = ? AND name = ? "
            "AND archived_at IS NULL ORDER BY version DESC LIMIT 1",
            (WORKSPACE_ID, name),
        ).fetchone()
    return conn.execute(
        f"SELECT {_CONFIG_COLUMNS} FROM training_configs WHERE workspace_id = ? AND name = ? "
        "AND version = ? AND archived_at IS NULL",
        (WORKSPACE_ID, name, version),
    ).fetchone()


def list_training_configs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Latest version of every named config, with its version count."""
    return conn.execute(
        f"SELECT {', '.join('c.' + c.strip() for c in _CONFIG_COLUMNS.split(','))}, "
        "counts.versions FROM training_configs c JOIN (SELECT name, MAX(version) AS v, "
        "COUNT(*) AS versions FROM training_configs WHERE workspace_id = ? AND archived_at IS NULL "
        "GROUP BY name) counts ON counts.name = c.name AND counts.v = c.version "
        "WHERE c.workspace_id = ? ORDER BY c.name",
        (WORKSPACE_ID, WORKSPACE_ID),
    ).fetchall()


# --------------------------------------------------------------------------- runs

RUN_COLUMNS = (
    "id, dataset_id, config_id, model_id, backend, device, precision, artifact_format, status, "
    "run_dir, metadata_json, created_at, updated_at"
)


def insert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    dataset_id: str,
    config_id: int,
    model_id: str,
    backend: str,
    device: str,
    precision: str,
    artifact_format: str,
    run_dir: str,
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO runs (id, workspace_id, created_by, dataset_id, config_id, model_id, "
            "backend, device, precision, artifact_format, status, run_dir, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
            (
                run_id,
                WORKSPACE_ID,
                CREATED_BY,
                dataset_id,
                config_id,
                model_id,
                backend,
                device,
                precision,
                artifact_format,
                run_dir,
                now,
                now,
            ),
        )


def get_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {RUN_COLUMNS} FROM runs WHERE workspace_id = ? AND id = ? AND archived_at IS NULL",
        (WORKSPACE_ID, run_id),
    ).fetchone()


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT {RUN_COLUMNS} FROM runs WHERE workspace_id = ? AND archived_at IS NULL "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (WORKSPACE_ID, limit),
    ).fetchall()


def update_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str | None = None,
    from_statuses: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Set status and/or metadata; with ``from_statuses`` only from those statuses."""
    sets, args = ["updated_at = ?"], [utc_now()]
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if metadata is not None:
        sets.append("metadata_json = ?")
        args.append(json.dumps(metadata))
    where, where_args = "id = ?", [run_id]
    if from_statuses:
        where += f" AND status IN ({','.join('?' for _ in from_statuses)})"
        where_args += list(from_statuses)
    with conn:
        cursor = conn.execute(
            f"UPDATE runs SET {', '.join(sets)} WHERE {where}", (*args, *where_args)
        )
    return cursor.rowcount == 1


def delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def get_run_any(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    """A run, archived or not."""
    return conn.execute(
        f"SELECT {RUN_COLUMNS}, archived_at FROM runs WHERE workspace_id = ? AND id = ?",
        (WORKSPACE_ID, run_id),
    ).fetchone()


def list_runs_any(
    conn: sqlite3.Connection, limit: int, include_archived: bool
) -> list[sqlite3.Row]:
    where = "workspace_id = ?" + ("" if include_archived else " AND archived_at IS NULL")
    return conn.execute(
        f"SELECT {RUN_COLUMNS}, archived_at FROM runs WHERE {where} "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (WORKSPACE_ID, limit),
    ).fetchall()


def set_run_archived(conn: sqlite3.Connection, run_id: str, archived: bool) -> None:
    """Archive a run with its evaluations and exports, or bring them back."""
    value = utc_now() if archived else None
    with conn:
        for table, column in (("runs", "id"), ("evaluations", "run_id"), ("exports", "run_id")):
            conn.execute(f"UPDATE {table} SET archived_at = ? WHERE {column} = ?", (value, run_id))


def purge_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete a run's rows: judgements, eval items, evaluations, exports, jobs, then the run."""
    with conn:
        evaluations = "SELECT id FROM evaluations WHERE run_id = ?"
        conn.execute(
            f"DELETE FROM eval_judgements WHERE evaluation_id IN ({evaluations})", (run_id,)
        )
        conn.execute(f"DELETE FROM eval_items WHERE evaluation_id IN ({evaluations})", (run_id,))
        conn.execute("DELETE FROM evaluations WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM exports WHERE run_id = ?", (run_id,))
        conn.execute(
            "DELETE FROM model_lock WHERE job_id IN (SELECT id FROM jobs WHERE run_id = ?)",
            (run_id,),
        )
        conn.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def run_job_ids(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """Every job linked to a run (any type), with its status."""
    return conn.execute(
        "SELECT id, type, status FROM jobs WHERE workspace_id = ? AND run_id = ?",
        (WORKSPACE_ID, run_id),
    ).fetchall()


# --------------------------------------------------------------------------- evaluations

EVALUATION_COLUMNS = (
    "id, run_id, status, eval_dir, verdict, metrics_json, job_id, config_json, created_at, "
    "updated_at"
)


def insert_evaluation(
    conn: sqlite3.Connection,
    *,
    evaluation_id: str,
    run_id: str,
    eval_dir: str,
    config: dict[str, Any],
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO evaluations (id, workspace_id, created_by, run_id, status, eval_dir, "
            "config_json, created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
            (
                evaluation_id,
                WORKSPACE_ID,
                CREATED_BY,
                run_id,
                eval_dir,
                json.dumps(config),
                now,
                now,
            ),
        )


def get_evaluation(conn: sqlite3.Connection, evaluation_id: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {EVALUATION_COLUMNS} FROM evaluations "
        "WHERE workspace_id = ? AND id = ? AND archived_at IS NULL",
        (WORKSPACE_ID, evaluation_id),
    ).fetchone()


def list_evaluations(conn: sqlite3.Connection, run_id: str | None = None) -> list[sqlite3.Row]:
    where, args = "workspace_id = ? AND archived_at IS NULL", [WORKSPACE_ID]
    if run_id is not None:
        where += " AND run_id = ?"
        args.append(run_id)
    return conn.execute(
        f"SELECT {EVALUATION_COLUMNS} FROM evaluations WHERE {where} "
        "ORDER BY created_at DESC, rowid DESC",
        args,
    ).fetchall()


def update_evaluation(
    conn: sqlite3.Connection,
    evaluation_id: str,
    *,
    status: str | None = None,
    from_statuses: tuple[str, ...] | None = None,
    verdict: str | None = None,
    metrics: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> bool:
    sets, args = ["updated_at = ?"], [utc_now()]
    for column, value in (("status", status), ("verdict", verdict), ("job_id", job_id)):
        if value is not None:
            sets.append(f"{column} = ?")
            args.append(value)
    if metrics is not None:
        sets.append("metrics_json = ?")
        args.append(json.dumps(metrics))
    where, where_args = "id = ?", [evaluation_id]
    if from_statuses:
        where += f" AND status IN ({','.join('?' for _ in from_statuses)})"
        where_args += list(from_statuses)
    with conn:
        cursor = conn.execute(
            f"UPDATE evaluations SET {', '.join(sets)} WHERE {where}", (*args, *where_args)
        )
    return cursor.rowcount == 1


def delete_evaluation(conn: sqlite3.Connection, evaluation_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM eval_judgements WHERE evaluation_id = ?", (evaluation_id,))
        conn.execute("DELETE FROM eval_items WHERE evaluation_id = ?", (evaluation_id,))
        conn.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))


def replace_eval_items(
    conn: sqlite3.Connection, evaluation_id: str, items: list[dict[str, Any]]
) -> None:
    """``items``: ``{item_id, position, category, content_hash, a_model}``."""
    with conn:
        conn.execute("DELETE FROM eval_judgements WHERE evaluation_id = ?", (evaluation_id,))
        conn.execute("DELETE FROM eval_items WHERE evaluation_id = ?", (evaluation_id,))
        conn.executemany(
            "INSERT INTO eval_items (workspace_id, created_by, evaluation_id, item_id, position, "
            "category, content_hash, a_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    WORKSPACE_ID,
                    CREATED_BY,
                    evaluation_id,
                    i["item_id"],
                    i["position"],
                    i["category"],
                    i["content_hash"],
                    i["a_model"],
                )
                for i in items
            ],
        )


def list_eval_items(conn: sqlite3.Connection, evaluation_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT item_id, position, category, content_hash, a_model FROM eval_items "
        "WHERE evaluation_id = ? ORDER BY position",
        (evaluation_id,),
    ).fetchall()


def upsert_judgements(
    conn: sqlite3.Connection, evaluation_id: str, judgements: list[tuple[str, str, str]]
) -> None:
    """``judgements``: ``(item_id, judgement, origin)``; a later judgement replaces an earlier one."""
    now = utc_now()
    with conn:
        conn.executemany(
            "INSERT INTO eval_judgements (workspace_id, created_by, evaluation_id, item_id, "
            "judgement, origin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (evaluation_id, item_id) DO UPDATE SET judgement = excluded.judgement, "
            "origin = excluded.origin, created_at = excluded.created_at",
            [
                (WORKSPACE_ID, CREATED_BY, evaluation_id, item_id, judgement, origin, now)
                for item_id, judgement, origin in judgements
            ],
        )


def list_judgements(conn: sqlite3.Connection, evaluation_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT item_id, judgement, origin, created_at FROM eval_judgements "
        "WHERE evaluation_id = ? ORDER BY id",
        (evaluation_id,),
    ).fetchall()


def earlier_shared_items(conn: sqlite3.Connection, evaluation_id: str) -> list[sqlite3.Row]:
    """``(evaluation_id, content_hash)`` of items in earlier evaluations that this one shares."""
    return conn.execute(
        "SELECT o.evaluation_id, o.content_hash FROM eval_items o "
        "JOIN evaluations e ON e.id = o.evaluation_id "
        "WHERE o.content_hash IN (SELECT content_hash FROM eval_items WHERE evaluation_id = ?) "
        "AND e.rowid < (SELECT rowid FROM evaluations WHERE id = ?) "
        "AND e.status = 'ready' AND e.archived_at IS NULL AND e.workspace_id = ?",
        (evaluation_id, evaluation_id, WORKSPACE_ID),
    ).fetchall()


# --------------------------------------------------------------------------- exports

EXPORT_COLUMNS = "id, run_id, kind, status, path, metadata_json, created_at, updated_at"


def insert_export(
    conn: sqlite3.Connection,
    *,
    export_id: str,
    run_id: str,
    kind: str,
    status: str,
    path: str,
    metadata: dict[str, Any],
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            "INSERT INTO exports (id, workspace_id, created_by, run_id, kind, status, path, "
            "metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                export_id,
                WORKSPACE_ID,
                CREATED_BY,
                run_id,
                kind,
                status,
                path,
                json.dumps(metadata),
                now,
                now,
            ),
        )


def get_export(conn: sqlite3.Connection, export_id: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {EXPORT_COLUMNS} FROM exports "
        "WHERE workspace_id = ? AND id = ? AND archived_at IS NULL",
        (WORKSPACE_ID, export_id),
    ).fetchone()


def list_exports(conn: sqlite3.Connection, run_id: str | None = None) -> list[sqlite3.Row]:
    where, args = "workspace_id = ? AND archived_at IS NULL", [WORKSPACE_ID]
    if run_id is not None:
        where += " AND run_id = ?"
        args.append(run_id)
    return conn.execute(
        f"SELECT {EXPORT_COLUMNS} FROM exports WHERE {where} ORDER BY created_at DESC, rowid DESC",
        args,
    ).fetchall()


def update_export(
    conn: sqlite3.Connection,
    export_id: str,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    sets, args = ["status = ?", "updated_at = ?"], [status, utc_now()]
    if metadata is not None:
        sets.append("metadata_json = ?")
        args.append(json.dumps(metadata))
    with conn:
        conn.execute(f"UPDATE exports SET {', '.join(sets)} WHERE id = ?", (*args, export_id))
