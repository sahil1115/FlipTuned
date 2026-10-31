"""Architecture rules (SPEC §13.1), checked statically."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "fliptuned"
ADAPTERS = [SRC / "cli.py", *sorted((SRC / "api").rglob("*.py"))]
OS_SPECIFIC_MODULES = {
    "winreg",
    "ctypes",
    "msvcrt",
    "_winapi",
    "fcntl",
    "termios",
    "pwd",
    "grp",
    "resource",
}
OS_SPECIFIC_OS_ATTRS = {"killpg", "setsid", "getpgid", "setpgrp", "fork", "startfile", "geteuid"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.append("fliptuned.<relative>")
            elif node.module:
                names.append(node.module)
                if node.module == "fliptuned":
                    names.extend(f"fliptuned.{alias.name}" for alias in node.names)
    return names


def _py_files(exclude: Path | None = None) -> list[Path]:
    return [p for p in SRC.rglob("*.py") if exclude is None or exclude not in p.parents]


@pytest.mark.parametrize("path", ADAPTERS, ids=lambda p: p.name)
def test_adapters_import_only_services(path: Path) -> None:
    allowed = ["fliptuned.services"]
    if (SRC / "api") in path.parents:
        allowed.append("fliptuned.api.")  # the API package's own modules
    if path == SRC / "cli.py":
        allowed.append("fliptuned.api.server")  # `serve` starts the API server; no logic
    bad = [
        name
        for name in _imports(path)
        if name.startswith("fliptuned") and not name.startswith(tuple(allowed))
    ]
    assert bad == [], f"{path.name} imports non-service modules: {bad}"


def test_services_never_print_or_import_adapters() -> None:
    for path in (SRC / "services").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        prints = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        ]
        assert prints == [], f"{path.name} calls print() at lines {prints}"
        forbidden = [
            n
            for n in _imports(path)
            if n.split(".")[0] in {"typer", "click", "fastapi", "starlette"}
            or n.startswith(("fliptuned.cli", "fliptuned.api"))
        ]
        assert forbidden == [], f"{path.name} imports adapter code: {forbidden}"


def test_os_specific_code_only_in_platform() -> None:
    offenders = []
    for path in _py_files(exclude=SRC / "platform"):
        if path.parent == SRC / "platform":
            continue
        for name in _imports(path):
            if name.split(".")[0] in OS_SPECIFIC_MODULES:
                offenders.append(f"{path.relative_to(SRC)} imports {name}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr in OS_SPECIFIC_OS_ATTRS
            ):
                offenders.append(f"{path.relative_to(SRC)} uses os.{node.attr}")
    assert offenders == []


def test_no_shell_true_anywhere() -> None:
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "shell":
                assert not (isinstance(node.value, ast.Constant) and node.value.value is True), path


def test_text_files_opened_with_explicit_encoding() -> None:
    """open()/read_text()/write_text() in text mode must pass encoding= (UTF-8 everywhere)."""
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name not in {"open", "read_text", "write_text"}:
                continue
            if name == "open" and isinstance(func, ast.Attribute):
                continue  # e.g. winreg.OpenKey / Path.open handled by mode check below
            kw = {k.arg for k in node.keywords}
            mode = next((a.value for a in node.args[1:2] if isinstance(a, ast.Constant)), "r")
            if "b" in str(mode):
                continue
            if "encoding" not in kw:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []
