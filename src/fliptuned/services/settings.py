"""User settings (SPEC §13.2, §13.4, §14.2): allowed folders, idle timeout, upload limit, port.

Stored in the registry's ``settings`` table. The app home is shown read-only. The UI may browse
and scan only inside the allowed folders, plus ``<app_home>/uploads`` (always allowed). A folder
inside the app home, or one that contains it, can't be added: the app home holds the registry and
run files, which the folder browser must not expose.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.errors import ConfigError
from fliptuned.platform.paths import display_path, unsafe_reason
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate

DEFAULT_PORT = 8765
DEFAULTS = {
    "allowed_roots": [],
    "idle_timeout_min": 10,
    "upload_max_mb": 50,
    "dev_mode": False,
    "port": DEFAULT_PORT,
}


class Settings(BaseModel):
    app_home: str = Field(description="Where FlipTuned keeps its data (read-only here).")
    allowed_roots: list[str] = Field(
        description="Folders the app may browse and read data from (uploads are always allowed)."
    )
    idle_timeout_min: float = Field(
        description="Minutes without a chat message before the model is released."
    )
    upload_max_mb: int = Field(description="Largest file that can be uploaded, in MB.")
    dev_mode: bool = Field(description="Show developer options, such as tiny test-only models.")
    port: int = Field(
        description="Port `fliptuned serve` tries first (the next free one if it's taken); "
        "used from the next launch. `--port` overrides it."
    )


class SettingsUpdate(BaseModel):
    home: str | None = None
    allowed_roots: list[str] | None = Field(default=None, max_length=50)
    idle_timeout_min: float | None = Field(default=None, gt=0, le=24 * 60)
    upload_max_mb: int | None = Field(default=None, ge=1, le=4096)
    dev_mode: bool | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)


def open_home(home: str | None) -> AppPaths:
    """The app home, created and migrated if needed."""
    paths = AppPaths(resolve_app_home(home).path)
    paths.ensure()
    migrate(paths.db)
    return paths


def get_settings(home: str | None) -> Settings:
    paths = open_home(home)
    with connect(paths.db) as conn:
        values = {k: repositories.get_setting(conn, k, v) for k, v in DEFAULTS.items()}
    return Settings(app_home=display_path(paths.home), **values)


def _check_root(raw: str, app_home: Path) -> str:
    reason = unsafe_reason(raw)
    if reason:
        raise ConfigError(f"That folder can't be allowed: {reason}.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ConfigError("Allowed folders must be absolute paths.")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ConfigError("That folder doesn't exist.", hint="Check the path.") from None
    if not resolved.is_dir():
        raise ConfigError("Allowed folders must be folders, not files.")
    home = app_home.resolve()
    if resolved == home or resolved.is_relative_to(home) or home.is_relative_to(resolved):
        raise ConfigError(
            "The app home (or a folder containing it) can't be an allowed folder.",
            hint="Choose the folder that holds your Q&A files.",
        )
    return display_path(resolved)


def update_settings(request: SettingsUpdate) -> Settings:
    paths = open_home(request.home)
    changes: dict[str, object] = {}
    if request.allowed_roots is not None:
        roots = [_check_root(r, paths.home) for r in request.allowed_roots]
        changes["allowed_roots"] = list(dict.fromkeys(roots))
    if request.idle_timeout_min is not None:
        changes["idle_timeout_min"] = request.idle_timeout_min
    if request.upload_max_mb is not None:
        changes["upload_max_mb"] = request.upload_max_mb
    if request.dev_mode is not None:
        changes["dev_mode"] = request.dev_mode
    if request.port is not None:
        changes["port"] = request.port
    with connect(paths.db) as conn:
        for key, value in changes.items():
            repositories.set_setting(conn, key, value)
    return get_settings(request.home)


def allowed_roots(home: str | None) -> list[Path]:
    """Configured folders plus the uploads folder."""
    settings = get_settings(home)
    return [Path(r) for r in settings.allowed_roots] + [open_home(home).uploads]
