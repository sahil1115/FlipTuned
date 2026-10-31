from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fliptuned.config.loader import load_any_config, load_config
from fliptuned.config.models import (
    CONFIG_MODELS,
    AcceptanceCriteria,
    Backend,
    Device,
    IngestConfig,
    Precision,
    SplitRatios,
    TrainConfig,
)
from fliptuned.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = sorted((REPO_ROOT / "configs").glob("*.sample.yaml"))


def test_every_config_kind_has_a_sample() -> None:
    kinds = {type(load_any_config(p)).__name__ for p in SAMPLES}
    assert kinds == {model.__name__ for model in CONFIG_MODELS.values()}


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_sample_configs_validate(path: Path) -> None:
    load_any_config(path)


def test_sample_configs_point_at_fixtures_only() -> None:
    ingest = load_config(REPO_ROOT / "configs" / "ingest.sample.yaml", IngestConfig)
    for source in ingest.sources:
        assert source.path.startswith("../tests/fixtures/")


def test_train_defaults_are_auto_and_match_spec() -> None:
    config = TrainConfig(name="t", model={"catalog_id": "tiny-qwen2"})
    assert (config.backend, config.device, config.precision) == (
        Backend.AUTO,
        Device.AUTO,
        Precision.AUTO,
    )
    assert config.training.max_epochs == 3 and config.training.patience == 3
    assert config.lora.rank == 8 and config.lora.alpha == 16
    assert config.training.hf_peft.cpu_max_params_b == 1.0


def test_acceptance_defaults_match_spec() -> None:
    a = AcceptanceCriteria()
    assert (a.min_test_items, a.min_review_coverage, a.min_non_tied_pairs) == (50, 1.0, 20)
    assert (a.sign_test_alpha, a.max_category_net_loss_share) == (0.05, 0.20)
    assert (a.max_deterministic_drop_pts, a.max_regression_set_drop_pts) == (2.0, 5.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "mlx", "device": "cuda"},
        {"backend": "mlx", "precision": "bf16"},
        {"device": "cpu", "precision": "qlora_4bit"},
        {"backend": "tpu"},
        {"lora": {"rank": 0}},
        {"training": {"learning_rate": 0}},
        {"unknown_field": 1},
    ],
)
def test_invalid_train_configs_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(name="t", model={"catalog_id": "x"}, **overrides)


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="add up to 1.0"):
        SplitRatios(train=0.7, valid=0.1, test=0.1)


def test_tabular_source_requires_column_mapping() -> None:
    with pytest.raises(ValidationError, match="columns"):
        IngestConfig(name="x", sources=[{"path": "a.csv", "format": "csv"}])


def test_loader_errors_list_fields_without_echoing_values(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "kind: train\nname: t\nmodel: {catalog_id: x}\nsystem_prompt: 42\nlora: {rank: sk-SECRETVALUE}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as err:
        load_any_config(path)
    assert "lora.rank" in (err.value.details or "")
    assert "SECRETVALUE" not in f"{err.value.message} {err.value.hint} {err.value.details}"


def test_loader_reports_yaml_and_kind_problems(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "a.yaml"
    bad_yaml.write_text("kind: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_any_config(bad_yaml)
    no_kind = tmp_path / "b.yaml"
    no_kind.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="kind"):
        load_any_config(no_kind)
    with pytest.raises(ConfigError, match="not found"):
        load_any_config(tmp_path / "missing.yaml")


def _properties(schema: dict):
    yield from schema.get("properties", {}).items()
    for sub in schema.get("$defs", {}).values():
        yield from _properties(sub)


@pytest.mark.parametrize("model", CONFIG_MODELS.values(), ids=lambda m: m.__name__)
def test_every_field_has_a_description_for_the_ui(model) -> None:
    missing = [
        name for name, prop in _properties(model.model_json_schema()) if not prop.get("description")
    ]
    assert missing == []
