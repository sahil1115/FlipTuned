"""Curated model catalog (SPEC §7), loaded from the bundled ``catalog.yaml``."""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fliptuned.errors import ConfigError


class ModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str
    revision: str = Field(description="Exact commit hash on the Hugging Face Hub.")
    download_gb: float = Field(description="Approximate download size of the model files.")


class PrefixTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: Literal["pass", "fail"]
    verified: str = Field(description="Date the result was last verified (YYYY-MM-DD).")


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    params_b: float
    license: str
    model_card_url: str
    test_only: bool = False
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra chat-template arguments, applied identically in training, inference and tests.",
    )
    sources: dict[str, ModelSource] = Field(description="Per backend: 'hf_peft' and/or 'mlx'.")
    est_train_memory_gb: dict[str, float] = Field(
        description="Estimated peak training memory, keyed '<backend>:<device>:<precision>' or 'mlx'."
    )
    prefix_test: PrefixTest

    @property
    def trainable(self) -> bool:
        """Only models whose template prefix test passed can be selected for training (SPEC §7)."""
        return self.prefix_test.result == "pass"

    @property
    def tokenizer_source(self) -> ModelSource:
        """The original repo's tokenizer (mlx-community conversions reuse it)."""
        return self.sources.get("hf_peft") or next(iter(self.sources.values()))


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    models: list[CatalogEntry]


@cache
def load_catalog() -> Catalog:
    text = resources.files("fliptuned.models").joinpath("catalog.yaml").read_text(encoding="utf-8")
    return Catalog.model_validate(yaml.safe_load(text))


def get_entry(catalog_id: str) -> CatalogEntry:
    catalog = load_catalog()
    for entry in catalog.models:
        if entry.id == catalog_id:
            return entry
    raise ConfigError(
        f"Model '{catalog_id}' is not in the catalog.",
        hint=f"Use one of: {', '.join(e.id for e in catalog.models)}.",
    )
