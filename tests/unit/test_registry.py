from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from fliptuned.errors import RegistryError
from fliptuned.registry import db as registry_db
from fliptuned.registry import repositories
from fliptuned.registry.db import Migration, connect, latest_version, load_migrations, migrate

TABLES = {
    "settings",
    "data_sources",
    "datasets",
    "review_decisions",
    "training_configs",
    "runs",
    "jobs",
    "model_lock",
    "evaluations",
    "eval_judgements",
    "exports",
    "baselines",
    "eval_items",
}


def test_migrate_empty_database(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    result = migrate(db_path)
    assert (result.from_version, result.to_version, result.backup_path) == (
        0,
        latest_version(),
        None,
    )
    with connect(db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert names >= TABLES
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        for table in TABLES:
            columns = {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}
            assert "workspace_id" in columns and "created_by" in columns, table
            assert columns["workspace_id"]["dflt_value"] == "'local'"
            assert columns["created_by"]["dflt_value"] == "'local-user'"


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    again = migrate(db_path)
    assert again.from_version == again.to_version and again.backup_path is None


def test_existing_database_is_backed_up_before_migrating(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE legacy (x INTEGER)")
        conn.execute("INSERT INTO legacy VALUES (7)")
        conn.commit()
    result = migrate(db_path)
    assert result.backup_path is not None and result.backup_path.exists()
    with closing(sqlite3.connect(result.backup_path)) as backup:
        assert backup.execute("SELECT x FROM legacy").fetchone()[0] == 7


def test_migrates_from_previous_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    real = load_migrations()
    extra = Migration(len(real) + 1, "9999_test.sql", "ALTER TABLE runs ADD COLUMN note TEXT;")
    monkeypatch.setattr(registry_db, "load_migrations", lambda: [*real, extra])
    result = migrate(db_path)
    assert (result.from_version, result.to_version) == (len(real), len(real) + 1)
    assert result.backup_path is not None and result.backup_path.exists()
    with connect(db_path) as conn:
        assert "note" in {r["name"] for r in conn.execute("PRAGMA table_info(runs)")}


def test_failed_migration_leaves_database_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    real = load_migrations()
    bad = Migration(len(real) + 1, "9999_bad.sql", "CREATE TABLE ok_one (x); THIS IS NOT SQL;")
    monkeypatch.setattr(registry_db, "load_migrations", lambda: [*real, bad])
    with pytest.raises(RegistryError):
        migrate(db_path)
    with connect(db_path) as conn:
        assert registry_db.user_version(conn) == len(real)
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='ok_one'").fetchone() is None


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(RegistryError, match="newer FlipTuned"):
        migrate(db_path)


def test_settings_repository_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    with connect(db_path) as conn:
        assert repositories.get_setting(conn, "allowed_roots", []) == []
        repositories.set_setting(conn, "allowed_roots", ["C:\\data", "/home/me/qa"])
        repositories.set_setting(conn, "idle_timeout_min", 10)
        assert repositories.get_setting(conn, "allowed_roots") == ["C:\\data", "/home/me/qa"]
        assert repositories.get_setting(conn, "idle_timeout_min") == 10
        assert repositories.count_rows(conn, "runs") == 0
        assert repositories.active_jobs(conn) == []


def test_job_status_constraint_enforced(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    with connect(db_path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs (id, type, status, created_at) VALUES ('j1', 'train', 'exploded', 'now')"
        )


def test_v1_review_decisions_survive_migration_to_v2(tmp_path: Path) -> None:
    """A database at schema v1 (Phase 2) keeps its decisions; they get content_hash = NULL,
    which the review service treats as stale (re-review needed)."""
    db_path = tmp_path / "fliptuned.db"
    v1 = [m for m in load_migrations() if m.version == 1]
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(f"BEGIN;\n{v1[0].sql}\nPRAGMA user_version = 1;\nCOMMIT;")
        conn.execute(
            "INSERT INTO data_sources (id, name, kind, root_path, created_at, updated_at) "
            "VALUES ('src_a', 'a', 'folder', '/x', 'now', 'now')"
        )
        conn.execute(
            "INSERT INTO review_decisions (source_id, record_id, status, created_at) "
            "VALUES ('src_a', 'qa_1', 'approved', 'now')"
        )
        conn.commit()
    result = migrate(db_path)
    assert (result.from_version, result.to_version) == (1, latest_version())
    assert result.backup_path is not None
    with connect(db_path) as conn:
        latest = repositories.latest_review_decisions(conn, ["src_a"])
        assert latest[("src_a", "qa_1")]["status"] == "approved"
        assert latest[("src_a", "qa_1")]["content_hash"] is None


def test_latest_review_decision_wins(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    migrate(db_path)
    with connect(db_path) as conn:
        repositories.upsert_data_source(
            conn, source_id="src_a", name="a", root_path="/x", status="ingested", stats={}
        )
        base = {"source_id": "src_a", "content_hash": "h", "origin": "ui"}
        repositories.insert_review_decisions(
            conn,
            [
                {**base, "record_id": "qa_1", "status": "approved"},
                {**base, "record_id": "qa_2", "status": "approved"},
            ],
        )
        repositories.insert_review_decisions(
            conn, [{**base, "record_id": "qa_1", "status": "rejected", "origin": "csv"}]
        )
        latest = repositories.latest_review_decisions(conn, ["src_a"])
        assert {k[1]: v["status"] for k, v in latest.items()} == {
            "qa_1": "rejected",
            "qa_2": "approved",
        }
        assert repositories.latest_review_decisions(conn, []) == {}


def test_v3_jobs_and_lock_survive_the_job_type_rebuild(tmp_path: Path) -> None:
    db_path = tmp_path / "fliptuned.db"
    with closing(sqlite3.connect(db_path)) as conn:
        for m in load_migrations()[:3]:
            conn.executescript(f"BEGIN;\n{m.sql}\nPRAGMA user_version = {m.version};\nCOMMIT;")
        conn.execute(
            "INSERT INTO jobs (id, type, status, pid, params_json, created_at) "
            "VALUES ('j1', 'ingest', 'running', 42, '{}', 'now')"
        )
        conn.execute(
            "INSERT INTO model_lock (id, job_id, pid, pid_create_time, acquired_at) "
            "VALUES (1, 'j1', 42, 1.5, 'now')"
        )
        conn.commit()
    assert migrate(db_path).from_version == 3
    with connect(db_path) as conn:
        assert repositories.get_job(conn, "j1")["pid"] == 42
        assert repositories.get_model_lock(conn)["job_id"] == "j1"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.execute(
            "INSERT INTO jobs (id, type, status, created_at) VALUES ('j2', 'selftest', 'queued', 'n')"
        )
