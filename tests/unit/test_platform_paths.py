from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from fliptuned.errors import PathNotAllowedError
from fliptuned.platform.paths import (
    display_path,
    replace_dir,
    resolve_within_roots,
    unsafe_reason,
)


@pytest.mark.parametrize(
    "raw",
    [
        r"\\server\share\data.csv",
        "//server/share/data.csv",
        r"\\?\C:\data\file.csv",
        r"\\.\PhysicalDrive0",
        r"C:\data\..\secret.csv",
        r"C:\data\nul",
        r"relative\con.txt",
        "C:\\data\x00.csv",
        "",
    ],
)
def test_windows_style_unsafe_paths_rejected(raw: str) -> None:
    assert unsafe_reason(raw, windows=True) is not None


@pytest.mark.parametrize(
    "raw", [r"C:\data\file.csv", r"D:\Q&A exports\2024\qa.xlsx", r"data\notes.md"]
)
def test_windows_style_safe_paths_accepted(raw: str) -> None:
    assert unsafe_reason(raw, windows=True) is None


@pytest.mark.parametrize("raw", ["/data/../etc/passwd", "//host/share", "/data/a\x00b"])
def test_posix_style_unsafe_paths_rejected(raw: str) -> None:
    assert unsafe_reason(raw, windows=False) is not None


def test_posix_style_safe_path_accepted() -> None:
    assert unsafe_reason("/home/me/data/qa.csv", windows=False) is None


def test_resolve_within_roots_accepts_inside_and_rejects_outside(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    inside = root / "sub" / "qa.csv"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")

    assert resolve_within_roots(inside, [root]) == inside.resolve()
    assert resolve_within_roots(root, [root]) == root.resolve()
    with pytest.raises(PathNotAllowedError):
        resolve_within_roots(outside, [root])
    with pytest.raises(PathNotAllowedError):
        resolve_within_roots(str(root / "sub" / ".." / ".." / "outside.csv"), [root])
    with pytest.raises(PathNotAllowedError):
        resolve_within_roots(root / "missing.csv", [root])


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "secret"
    target.mkdir()
    link = root / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here (Windows without Developer Mode)")
    with pytest.raises(PathNotAllowedError):
        resolve_within_roots(link, [root])


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only")
def test_junction_escape_rejected(tmp_path: Path) -> None:
    import _winapi

    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "secret"
    target.mkdir()
    junction = root / "junction"
    _winapi.CreateJunction(str(target), str(junction))
    with pytest.raises(PathNotAllowedError):
        resolve_within_roots(junction, [root])


def test_display_path_is_native(tmp_path: Path) -> None:
    shown = display_path(tmp_path / "a" / "b.csv")
    assert shown == str(tmp_path / "a" / "b.csv")
    assert (os.sep + "a" + os.sep) in shown


def test_replace_dir_replaces_and_cleans_up(tmp_path: Path) -> None:
    src, dst = tmp_path / "new", tmp_path / "final"
    src.mkdir()
    (src / "a.txt").write_text("new", encoding="utf-8")
    dst.mkdir()
    (dst / "old.txt").write_text("old", encoding="utf-8")
    replace_dir(src, dst)
    assert [p.name for p in dst.iterdir()] == ["a.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final"]


def test_replace_dir_retries_transient_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "new"
    src.mkdir()
    real_replace = os.replace
    failures = iter([PermissionError("locked"), PermissionError("locked")])

    def flaky(a, b):
        error = next(failures, None)
        if error:
            raise error
        real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr("fliptuned.platform.paths.time.sleep", lambda _: None)
    replace_dir(src, tmp_path / "final")
    assert (tmp_path / "final").is_dir() and not src.exists()
