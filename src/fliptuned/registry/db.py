"""SQLite connection and versioned plain-SQL migrations (``PRAGMA user_version``).

Before migrating an existing database, a consistent copy is written next to it with the
SQLite backup API (safe with WAL). Connections are always closed promptly (Windows).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from fliptuned.errors import RegistryError

_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


@dataclass(frozen=True)
class MigrationResult:
    from_version: int
    to_version: int
    backup_path: Path | None


def load_migrations() -> list[Migration]:
    migrations = []
    for entry in resources.files("fliptuned.registry").joinpath("migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            migrations.append(
                Migration(int(match.group(1)), entry.name, entry.read_text(encoding="utf-8"))
            )
    migrations.sort(key=lambda m: m.version)
    expected = list(range(1, len(migrations) + 1))
    if [m.version for m in migrations] != expected:
        raise RegistryError("Registry migrations are not numbered 0001..N without gaps.")
    return migrations


def latest_version() -> int:
    return len(load_migrations())


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the registry (already migrated) and close it on exit."""
    with closing(sqlite3.connect(db_path)) as conn:
        _configure(conn)
        yield conn


def user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(db_path: Path) -> MigrationResult:
    migrations = load_migrations()
    target = len(migrations)
    existed = db_path.exists() and db_path.stat().st_size > 0
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        _configure(conn)
        current = user_version(conn)
        if current > target:
            raise RegistryError(
                f"The registry was created by a newer FlipTuned (schema {current}, this version knows {target}).",
                hint="Upgrade FlipTuned, or restore a backup of fliptuned.db.",
            )
        conn.execute("PRAGMA journal_mode = WAL")
        if current == target:
            return MigrationResult(current, current, None)

        backup_path = _backup(conn, db_path, current) if existed else None
        for migration in migrations[current:]:
            try:
                conn.executescript(
                    f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.version};\nCOMMIT;"
                )
            except sqlite3.Error as exc:
                conn.rollback()
                raise RegistryError(
                    f"Registry migration {migration.name} failed.",
                    hint=f"The database was not changed by this step. A backup is at {backup_path}."
                    if backup_path
                    else None,
                    details=str(exc),
                ) from None
        return MigrationResult(current, target, backup_path)


def _backup(conn: sqlite3.Connection, db_path: Path, version: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = db_path.with_name(f"{db_path.name}.v{version}-{stamp}.bak")
    with closing(sqlite3.connect(backup_path)) as dest:
        conn.backup(dest)
    return backup_path
