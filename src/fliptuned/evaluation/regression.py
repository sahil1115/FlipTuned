"""The bundled general-regression set (SPEC §12.1): ~20 synthetic coding prompts with checks."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from fliptuned.evaluation.checks import Expectations, normalize_language

REGRESSION_FILE = Path(__file__).with_name("regression_set.yaml")


class RegressionExpect(BaseModel):
    code: bool = False
    language: str | None = None
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)

    @field_validator("required", "forbidden")
    @classmethod
    def _compiles(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            re.compile(pattern)
        return patterns

    def expectations(self) -> Expectations:
        return Expectations(
            code=self.code,
            language=normalize_language(self.language) if self.language else None,
            required=list(self.required),
            forbidden=list(self.forbidden),
        )


class RegressionItem(BaseModel):
    id: str
    category: str
    prompt: str
    expect: RegressionExpect


class RegressionSet(BaseModel):
    version: int
    items: list[RegressionItem]


@cache
def load_regression_set() -> RegressionSet:
    data = yaml.safe_load(REGRESSION_FILE.read_text(encoding="utf-8"))
    return RegressionSet.model_validate(data)
