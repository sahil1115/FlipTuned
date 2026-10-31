from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_private_folders_are_git_ignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".dev-home/", ".scratch/", "configs/local/", "src/fliptuned/web_dist/"):
        assert entry in ignored
