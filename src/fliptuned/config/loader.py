"""Load YAML config files into validated models, with readable errors."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from fliptuned.config.models import CONFIG_MODELS
from fliptuned.errors import ConfigError

M = TypeVar("M", bound=BaseModel)


def load_config[M: BaseModel](path: Path, model: type[M]) -> M:
    data = _read_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"Config file {path.name} is not valid.",
            hint="Fix the fields listed below and try again.",
            details=format_validation_error(exc),
        ) from None


def load_any_config(path: Path) -> BaseModel:
    """Load a config file whose type is given by its ``kind`` field."""
    data = _read_yaml(path)
    kind = data.get("kind")
    if kind not in CONFIG_MODELS:
        raise ConfigError(
            f"Config file {path.name} has no valid 'kind'.",
            hint=f"Add 'kind:' with one of: {', '.join(CONFIG_MODELS)}.",
        )
    return load_config(path, CONFIG_MODELS[kind])


def _read_yaml(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}", hint="Check the path.") from None
    except UnicodeDecodeError:
        raise ConfigError(
            f"Config file {path.name} is not UTF-8.", hint="Save it as UTF-8."
        ) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        raise ConfigError(
            f"Config file {path.name} is not valid YAML{where}.", hint="Fix the YAML syntax."
        ) from None
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path.name} must contain a mapping of settings.")
    return data


def format_validation_error(exc: ValidationError) -> str:
    # Field paths and messages only; never echo the offending input values.
    lines = []
    for err in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(part) for part in err["loc"]) or "(top level)"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)
