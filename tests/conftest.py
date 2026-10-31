"""Shared test setup.

Every test runs with FLIPTUNED_HOME pointing at a pytest temp dir, so nothing can touch
``~/.fliptuned``. The ``file_access`` fixture records filesystem access through Python audit
hooks (open, listdir, scandir, mkdir, sqlite3.connect) to prove what a call does or doesn't touch.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "fliptuned"

_AUDITED = {
    "open",
    "os.listdir",
    "os.scandir",
    "os.mkdir",
    "os.remove",
    "os.rename",
    "sqlite3.connect",
}


@dataclass
class AccessRecorder:
    active: bool = False
    events: list[tuple[str, str]] = field(default_factory=list)

    @contextmanager
    def recording(self) -> Iterator[AccessRecorder]:
        self.events.clear()
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def under(self, root: Path) -> list[tuple[str, str]]:
        root_str = os.path.normcase(os.path.abspath(root))
        hits = []
        for event, path in self.events:
            norm = os.path.normcase(os.path.abspath(path))
            if norm == root_str or norm.startswith(root_str + os.sep):
                hits.append((event, path))
        return hits


_RECORDER = AccessRecorder()


def _hook(event: str, args: tuple) -> None:
    if not _RECORDER.active or event not in _AUDITED or not args:
        return
    target = args[0]
    if isinstance(target, int | type(None)):
        return
    try:
        path = os.fsdecode(target)
    except TypeError:
        return
    if path:
        _RECORDER.events.append((event, path))


sys.addaudithook(_hook)


@cache
def _missing_hardware(marker: str) -> str | None:
    """Why this machine can't run ``mlx`` / ``cuda`` tests, or None if it can."""
    if marker == "mlx":
        if not (sys.platform == "darwin" and platform.machine() == "arm64"):
            return "MLX needs a Mac with Apple Silicon"
        if importlib.util.find_spec("mlx_lm") is None:
            return "mlx-lm isn't installed (uv sync --extra mlx)"
        return None
    if importlib.util.find_spec("torch") is None:
        return "PyTorch isn't installed"
    import torch

    return None if torch.cuda.is_available() else "no CUDA GPU usable by PyTorch"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hardware-marked tests are skipped, not failed, on machines without that hardware."""
    for marker in ("mlx", "cuda"):
        if item.get_closest_marker(marker) and (reason := _missing_hardware(marker)):
            pytest.skip(reason)


@pytest.fixture
def file_access() -> Iterator[AccessRecorder]:
    """Use ``with file_access.recording(): ...`` around the code under test."""
    try:
        yield _RECORDER
    finally:
        _RECORDER.active = False
        _RECORDER.events.clear()


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    home = tmp_path_factory.mktemp("fliptuned-home")
    monkeypatch.setenv("FLIPTUNED_HOME", str(home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    return home
