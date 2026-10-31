"""Catalog loading and tokenizer access (offline by default; downloads only when allowed)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fliptuned.errors import ConfigError, NotAvailableError
from fliptuned.models import hub as tok
from fliptuned.models import tokenizer as tokmod
from fliptuned.models.catalog import get_entry, load_catalog


def test_catalog_entries_are_pinned() -> None:
    catalog = load_catalog()
    ids = [e.id for e in catalog.models]
    assert len(ids) == len(set(ids)) and "tiny-qwen2" in ids
    for entry in catalog.models:
        assert entry.sources, entry.id
        for source in entry.sources.values():
            assert re.fullmatch(r"[0-9a-f]{40}", source.revision), entry.id
    assert get_entry("tiny-qwen2").test_only
    assert get_entry("qwen3-1.7b").chat_template_kwargs == {"enable_thinking": False}


def test_unknown_model_lists_choices() -> None:
    with pytest.raises(ConfigError) as info:
        get_entry("gpt-17")
    assert "tiny-qwen2" in (info.value.hint or "")


def test_missing_tokenizer_offline_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    with pytest.raises(NotAvailableError) as info:
        tok.tokenizer_files(get_entry("tiny-qwen2"), allow_download=False)
    assert "--allow-download" in (info.value.hint or "")


def test_offline_never_enables_network_and_download_restores_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub import constants

    calls = []

    def fake_snapshot(repo, *, revision, allow_patterns, local_files_only=False):
        calls.append((repo, revision, local_files_only, constants.HF_HUB_OFFLINE))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    entry = get_entry("tiny-qwen2")
    tok.tokenizer_files(entry, allow_download=False)
    tok.tokenizer_files(entry, allow_download=True)
    source = entry.tokenizer_source
    assert calls == [
        (source.repo, source.revision, True, True),  # cache only, still offline
        (source.repo, source.revision, False, False),  # network only inside the call
    ]
    assert constants.HF_HUB_OFFLINE is True


@pytest.mark.tiny
def test_real_tiny_tokenizer_counts_chat_tokens() -> None:
    counter = tokmod.load_token_counter(get_entry("tiny-qwen2"), allow_download=True)
    short = counter.count(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi?"},
            {"role": "assistant", "content": "Hello."},
        ]
    )
    longer = counter.count(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi?"},
            {"role": "assistant", "content": "Hello. " * 50},
        ]
    )
    assert 0 < short < longer
    assert (
        counter.identity["tokenizer_revision"]
        == get_entry("tiny-qwen2").sources["hf_peft"].revision
    )


def test_overlapping_downloads_leave_the_hub_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two threads allowing network at once (web server requests) must restore offline mode."""
    import threading

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    inside = threading.Event()
    seen: list[bool] = []

    def first() -> None:
        with tok._network_allowed():
            inside.set()
            seen.append(constants.HF_HUB_OFFLINE)
            threading.Event().wait(0.2)  # the second thread starts meanwhile

    def second() -> None:
        inside.wait()
        with tok._network_allowed():
            seen.append(constants.HF_HUB_OFFLINE)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [False, False] and constants.HF_HUB_OFFLINE is True


@pytest.mark.tiny
def test_tokenizer_loads_from_two_threads_in_a_fresh_process() -> None:
    """The first transformers import from two threads at once failed (Phase 13 bug)."""
    import subprocess
    import sys

    tokmod.load_tokenizer(get_entry("tiny-qwen2"), allow_download=True)  # cached for the child
    script = (
        "import threading\n"
        "from fliptuned.models.catalog import get_entry\n"
        "from fliptuned.models.tokenizer import load_tokenizer\n"
        "errors = []\n"
        "def load():\n"
        "    try:\n"
        "        load_tokenizer(get_entry('tiny-qwen2'))\n"
        "    except Exception as exc:\n"
        "        errors.append(repr(exc))\n"
        "ts = [threading.Thread(target=load) for _ in range(4)]\n"
        "[t.start() for t in ts]\n"
        "[t.join(120) for t in ts]\n"
        "assert not any(t.is_alive() for t in ts), 'a load hung'\n"
        "assert errors == [], errors\n"
    )
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1", "PYTHONUTF8": "1"}
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, env=env
    )
    assert done.returncode == 0, done.stderr[-2000:]
