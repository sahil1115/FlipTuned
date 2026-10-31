"""Shared pipeline helpers for tests: ingest the sample fixtures, fake tokenizer, dataset configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fliptuned.services.ingest import IngestRequest, IngestResult, run_ingest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
SAMPLE_INGEST = REPO / "configs" / "ingest.sample.yaml"
SYNTHETIC_INGEST = REPO / "configs" / "ingest-synthetic.sample.yaml"
SAMPLE_DATASET = REPO / "configs" / "dataset.sample.yaml"


class FakeCounter:
    """Deterministic stand-in for a tokenizer: words + 4 per message. Never downloads anything."""

    identity = {
        "catalog_id": "tiny-qwen2",
        "tokenizer_repo": "fake/tokenizer",
        "tokenizer_revision": "0" * 40,
        "chat_template_kwargs": {},
    }

    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m["content"].split()) + 4 for m in messages)


def use_fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fliptuned.services.datasets.load_token_counter",
        lambda entry, allow_download=False: FakeCounter(),
    )


def ingest_samples(home: Path, *, synthetic: bool = True) -> list[IngestResult]:
    configs = [SAMPLE_INGEST, SYNTHETIC_INGEST] if synthetic else [SAMPLE_INGEST]
    return [run_ingest(IngestRequest(config_path=str(c), home=str(home))) for c in configs]


def dataset_config(tmp_path: Path, name: str = "cfg.yaml", **overrides) -> Path:
    data = yaml.safe_load(SAMPLE_DATASET.read_text(encoding="utf-8"))
    data.update(overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
