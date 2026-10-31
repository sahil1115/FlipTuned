from __future__ import annotations

from pathlib import Path

from fliptuned.config.app_home import AppPaths, HomeSource, resolve_app_home


def test_cli_home_wins_over_env(tmp_path: Path) -> None:
    resolved = resolve_app_home(tmp_path / "cli", env={"FLIPTUNED_HOME": str(tmp_path / "env")})
    assert resolved == resolve_app_home(str(tmp_path / "cli"), env={})
    assert resolved.path == tmp_path / "cli" and resolved.source is HomeSource.CLI


def test_env_wins_over_default(tmp_path: Path) -> None:
    resolved = resolve_app_home(None, env={"FLIPTUNED_HOME": str(tmp_path / "env")})
    assert resolved.path == tmp_path / "env" and resolved.source is HomeSource.ENV


def test_default_is_dot_fliptuned_in_user_home() -> None:
    resolved = resolve_app_home(None, env={"FLIPTUNED_HOME": "  "})
    assert resolved.path == Path.home() / ".fliptuned" and resolved.source is HomeSource.DEFAULT


def test_relative_home_resolves_against_cwd(tmp_path: Path) -> None:
    resolved = resolve_app_home("./.dev-home", env={}, cwd=tmp_path)
    assert resolved.path == tmp_path / ".dev-home"
    assert resolved.path.is_absolute()


def test_resolving_never_creates_anything(tmp_path: Path) -> None:
    resolve_app_home(tmp_path / "x", env={})
    assert not (tmp_path / "x").exists()


def test_ensure_creates_layout(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "home")
    paths.ensure()
    for name in ("sources", "uploads", "datasets", "runs", "configs", "logs", "jobs"):
        assert (paths.home / name).is_dir()
    assert paths.db == paths.home / "fliptuned.db"
