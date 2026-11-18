from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fliptuned import __version__
from fliptuned.cli import app

IMPLEMENTED = {
    "doctor",
    "status",
    "scan",
    "ingest",
    "data-audit",
    "review",
    "build-dataset",
    "validate-dataset",
    "models",
    "evaluate",
    "jobs",
    "train",
    "configs",
    "chat",
    "export",
    "serve",
    "sources",
}
SUBCOMMANDS = {
    "review": ["export", "import"],
    "models": ["list"],
    "jobs": ["list", "follow", "cancel", "resume", "selftest"],
    "configs": ["list", "show", "duplicate"],
    "sources": ["list", "delete"],
}
LEAF_COMMANDS = sorted([c] for c in IMPLEMENTED if c not in SUBCOMMANDS) + [
    [group, sub] for group, subs in SUBCOMMANDS.items() for sub in subs
]
runner = CliRunner()


def _listed_commands(help_text: str) -> set[str]:
    return set(re.findall(r"^[|│]\s+([a-z][a-z-]*)\s{2,}", help_text, flags=re.MULTILINE))


def test_root_help_lists_only_implemented_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert _listed_commands(result.output) == IMPLEMENTED


@pytest.mark.parametrize("command", LEAF_COMMANDS, ids=" ".join)
def test_each_command_has_help_with_home_option(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0
    assert "--home" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_status_creates_and_reports_temp_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = runner.invoke(app, ["status", "--home", str(home), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["app_home"] == str(home) and data["app_home_source"] == "--home"
    assert (
        data["schema_version"] >= 1 and data["migrated_from"] == 0 and data["backup_path"] is None
    )
    assert set(data["counts"].values()) == {0} and data["active_jobs"] == []
    assert (home / "fliptuned.db").exists()

    human = runner.invoke(app, ["status", "--home", str(home)])
    assert human.exit_code == 0 and "schema v" in human.output and "Migrated" not in human.output


def test_status_uses_env_home_when_no_flag(_isolated_home: Path) -> None:
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["app_home"] == str(_isolated_home) and data["app_home_source"] == "FLIPTUNED_HOME"


def test_errors_are_actionable_and_exit_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 3
    assert "Error: Config file not found" in result.output and "What to do:" in result.output


def test_unknown_command_is_usage_error() -> None:
    assert runner.invoke(app, ["train"]).exit_code == 2


def test_serve_without_the_web_extra_gives_an_install_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fliptuned.api.server", None)  # makes the import fail
    result = runner.invoke(app, ["serve", "--no-browser", "--home", str(tmp_path / "home")])
    assert result.exit_code == 4
    assert "web' extra" in result.output and "--extra web" in result.output


def test_sources_list_and_delete(tmp_path: Path) -> None:
    home = str(tmp_path / "home")
    config = Path(__file__).resolve().parents[2] / "configs" / "ingest-synthetic.sample.yaml"
    ingested = runner.invoke(app, ["ingest", "--config", str(config), "--home", home])
    assert ingested.exit_code == 0, ingested.output
    listed = json.loads(runner.invoke(app, ["sources", "list", "--json", "--home", home]).output)
    name = listed[0]["name"]
    wrong = runner.invoke(app, ["sources", "delete", name, "--confirm", "no", "--home", home])
    assert wrong.exit_code == 3 and "nothing was deleted" in wrong.output
    done = runner.invoke(app, ["sources", "delete", name, "--confirm", name, "--home", home])
    assert done.exit_code == 0 and "Deleted data source" in done.output
    after = runner.invoke(app, ["sources", "list", "--json", "--home", home])
    assert json.loads(after.output) == []
