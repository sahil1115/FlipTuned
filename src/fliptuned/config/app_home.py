"""App home resolution: ``--home`` → ``FLIPTUNED_HOME`` → ``~/.fliptuned``.

Resolving a path never creates or opens it. Only :meth:`AppPaths.ensure` creates directories.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

ENV_VAR = "FLIPTUNED_HOME"
DEFAULT_DIRNAME = ".fliptuned"
DEV_HOME_DIRNAME = ".dev-home"  # repo-local home for development and E2E (fixtures only)
DB_FILENAME = "fliptuned.db"


class HomeSource(StrEnum):
    CLI = "--home"
    ENV = ENV_VAR
    DEFAULT = "default"


@dataclass(frozen=True)
class ResolvedHome:
    path: Path
    source: HomeSource


def resolve_app_home(
    cli_home: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> ResolvedHome:
    env = os.environ if env is None else env
    if cli_home not in (None, ""):
        return ResolvedHome(_absolute(Path(cli_home), cwd), HomeSource.CLI)
    env_value = env.get(ENV_VAR, "").strip()
    if env_value:
        return ResolvedHome(_absolute(Path(env_value), cwd), HomeSource.ENV)
    return ResolvedHome(Path.home() / DEFAULT_DIRNAME, HomeSource.DEFAULT)


def _absolute(path: Path, cwd: Path | None) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (cwd or Path.cwd()) / path
    return Path(os.path.normpath(path))


@dataclass(frozen=True)
class AppPaths:
    home: Path

    @property
    def db(self) -> Path:
        return self.home / DB_FILENAME

    @property
    def sources(self) -> Path:
        return self.home / "sources"

    @property
    def uploads(self) -> Path:
        return self.home / "uploads"

    @property
    def datasets(self) -> Path:
        return self.home / "datasets"

    @property
    def baselines(self) -> Path:
        return self.home / "baselines"

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def configs(self) -> Path:
        return self.home / "configs"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def jobs(self) -> Path:
        return self.home / "jobs"

    def subdirs(self) -> list[Path]:
        return [
            self.sources,
            self.uploads,
            self.datasets,
            self.baselines,
            self.runs,
            self.configs,
            self.logs,
            self.jobs,
        ]

    def ensure(self) -> None:
        for directory in [self.home, *self.subdirs()]:
            directory.mkdir(parents=True, exist_ok=True)
