"""Training config registry: named, versioned configs; duplicate; presets (SPEC §13.2, §11.5).

Saving a config whose content equals the latest version of that name reuses that version;
any change creates version N+1. Old versions are kept, so every run points at the exact
config it used.

Only the fields the user set are stored. A stored default would count as an explicit override
when the config is loaded again, and the preset could no longer fill it in.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from fliptuned.config.loader import format_validation_error
from fliptuned.config.models import Precision, TrainConfig
from fliptuned.diagnostics.backend_select import BackendSelection, EnvironmentProbe
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.training.plan import PRESETS


class SavedConfig(BaseModel):
    id: int
    name: str
    version: int
    created: bool = Field(description="False when an identical latest version was reused.")


class ConfigSummary(BaseModel):
    name: str
    version: int
    versions: int
    model: str
    backend: str
    preset: str
    created_at: str


class ConfigDetail(BaseModel):
    id: int
    name: str
    version: int
    parent_id: int | None
    created_at: str
    config: dict[str, Any]
    yaml: str


def _registry(home: str | None):
    from fliptuned.config.app_home import AppPaths, resolve_app_home
    from fliptuned.registry.db import migrate

    paths = AppPaths(resolve_app_home(home).path)
    paths.ensure()
    migrate(paths.db)
    return paths


def stored_form(config: TrainConfig) -> dict[str, Any]:
    """What the registry keeps: the fields the user set (presets fill the rest at train time)."""
    return config.model_dump(mode="json", exclude_unset=True)


def save_config(home: str | None, config: TrainConfig) -> SavedConfig:
    paths = _registry(home)
    data = stored_form(config)
    with connect(paths.db) as conn:
        latest = repositories.get_training_config(conn, config.name)
        if latest is not None and json.loads(latest["config_json"]) == data:
            return SavedConfig(
                id=latest["id"], name=config.name, version=latest["version"], created=False
            )
        version = 1 if latest is None else latest["version"] + 1
        config_id = repositories.insert_training_config(
            conn,
            name=config.name,
            version=version,
            config=data,
            parent_id=latest["id"] if latest is not None else None,
        )
    return SavedConfig(id=config_id, name=config.name, version=version, created=True)


class ListConfigsRequest(BaseModel):
    home: str | None = None


def list_configs(request: ListConfigsRequest) -> list[ConfigSummary]:
    paths = _registry(request.home)
    with connect(paths.db) as conn:
        rows = repositories.list_training_configs(conn)
    out = []
    for row in rows:
        data = json.loads(row["config_json"])
        out.append(
            ConfigSummary(
                name=row["name"],
                version=row["version"],
                versions=row["versions"],
                model=data["model"]["catalog_id"],
                backend=data.get("backend", "auto"),
                preset=data.get("preset", "standard"),
                created_at=row["created_at"],
            )
        )
    return out


class ShowConfigRequest(BaseModel):
    home: str | None = None
    name: str
    version: int | None = None


def show_config(request: ShowConfigRequest) -> ConfigDetail:
    paths = _registry(request.home)
    with connect(paths.db) as conn:
        row = repositories.get_training_config(conn, request.name, request.version)
    if row is None:
        which = f" version {request.version}" if request.version else ""
        raise NotAvailableError(
            f"No training config '{request.name}'{which}.",
            hint="List them with `fliptuned configs list`.",
        )
    data = json.loads(row["config_json"])
    return ConfigDetail(
        id=row["id"],
        name=row["name"],
        version=row["version"],
        parent_id=row["parent_id"],
        created_at=row["created_at"],
        config=data,
        yaml=yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    )


class DuplicateConfigRequest(BaseModel):
    home: str | None = None
    name: str
    new_name: str = Field(min_length=1)


def duplicate_config(request: DuplicateConfigRequest) -> SavedConfig:
    source = show_config(ShowConfigRequest(home=request.home, name=request.name))
    paths = _registry(request.home)
    with connect(paths.db) as conn:
        if repositories.get_training_config(conn, request.new_name) is not None:
            raise FlipTunedError(
                f"A training config called '{request.new_name}' already exists.",
                hint="Pick another name.",
            )
    data = {**source.config, "name": request.new_name}
    try:
        config = TrainConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - stored configs were valid when saved
        raise ConfigError(
            "The stored config is no longer valid.", details=format_validation_error(exc)
        ) from None
    with connect(paths.db) as conn:
        config_id = repositories.insert_training_config(
            conn,
            name=config.name,
            version=1,
            config=stored_form(config),
            parent_id=source.id,
        )
    return SavedConfig(id=config_id, name=config.name, version=1, created=True)


def presets() -> dict[str, dict[str, dict[str, Any]]]:
    """Preset overrides per backend, for the UI and `configs show`."""
    return PRESETS


# --------------------------------------------------------------------------- config input (API)


class ConfigInput(BaseModel):
    """Where a config comes from: a file path, the config itself, or a saved training config."""

    config_path: str | None = Field(default=None, description="A config file (YAML).")
    config: dict[str, Any] | None = Field(default=None, description="The config itself.")
    saved_name: str | None = Field(default=None, description="A saved training config's name.")
    saved_version: int | None = Field(default=None, ge=1)


def materialize_config(home: str | None, kind: str, source: ConfigInput) -> str:
    """A config file path for ``source``, validated as ``kind``. Inline and saved configs are
    written to ``<app_home>/configs/generated/`` so every job keeps the exact file it ran with."""
    from datetime import UTC, datetime
    from secrets import token_hex

    from fliptuned.config.models import CONFIG_MODELS

    given = [v for v in (source.config_path, source.config, source.saved_name) if v is not None]
    if len(given) != 1:
        raise ConfigError("Give exactly one of config_path, config or saved_name.")
    if source.config_path is not None:
        return source.config_path
    if source.saved_name is not None:
        if kind != "train":
            raise ConfigError("Only training configs are saved by name.")
        data = show_config(
            ShowConfigRequest(home=home, name=source.saved_name, version=source.saved_version)
        ).config
    else:
        data = {"kind": kind, **(source.config or {})}
    model = CONFIG_MODELS[kind]
    try:
        model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"The {kind} config is not valid.", details=format_validation_error(exc)
        ) from None
    folder = _registry(home).configs / "generated"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{kind}-{datetime.now(UTC):%Y%m%d%H%M%S}-{token_hex(3)}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(path)


def json_schema(kind: str) -> dict[str, Any]:
    """JSON Schema of a config kind: field descriptions, defaults and limits for the UI."""
    from fliptuned.config.models import CONFIG_MODELS

    if kind not in CONFIG_MODELS:
        raise NotAvailableError(
            f"No config kind '{kind}'.", hint=f"Use one of {sorted(CONFIG_MODELS)}."
        )
    return CONFIG_MODELS[kind].model_json_schema()


class SaveConfigRequest(BaseModel):
    home: str | None = None
    config: dict[str, Any] = Field(description="A training config (kind 'train').")


def save_config_data(request: SaveConfigRequest) -> SavedConfig:
    """Validate and save a training config as the next version of its name."""
    try:
        config = TrainConfig.model_validate({"kind": "train", **request.config})
    except ValidationError as exc:
        raise ConfigError(
            "The training config is not valid.", details=format_validation_error(exc)
        ) from None
    return save_config(request.home, config)


# --------------------------------------------------------------------------- choices on this machine


class HardwareChoice(BaseModel):
    backend: str
    device: str = Field(description="'apple_gpu', 'cuda' or 'cpu'.")
    available: bool
    reason: str
    precisions: list[str] = Field(description="Precisions this option accepts here ('auto' first).")


class TrainingChoices(BaseModel):
    auto: BackendSelection | None = Field(description="What 'auto' picks here (for the model).")
    auto_error: str | None = Field(description="Why 'auto' finds nothing, if it doesn't.")
    options: list[HardwareChoice]
    presets: dict[str, dict[str, dict[str, Any]]]


def _precisions(probe: EnvironmentProbe, device: str) -> list[str]:
    if device == "cuda":
        out = ["auto"]
        if probe.bitsandbytes_version is not None and probe.machine.os_name != "macos":
            out.append(Precision.QLORA_4BIT.value)
        if probe.torch.bf16_supported is not False:
            out.append(Precision.BF16.value)
        return [*out, Precision.FP16.value]
    if device == "cpu":
        return ["auto", Precision.FP32.value]
    return ["auto"]  # MLX: the model files decide (4-bit models train as QLoRA)


def training_choices(
    model: str | None = None, probe_fn: Callable[[], EnvironmentProbe] | None = None
) -> TrainingChoices:
    """Backend / device / precision options for a training config on this machine (SPEC §5.2)."""
    from fliptuned.diagnostics.backend_select import list_options, select_backend
    from fliptuned.models.catalog import get_entry
    from fliptuned.models.compat import select_for_entry
    from fliptuned.services.doctor import default_probe

    probe = (probe_fn or default_probe)()
    auto: BackendSelection | None = None
    error = None
    try:
        auto = select_for_entry(probe, get_entry(model)) if model else select_backend(probe)
    except FlipTunedError as exc:
        error = exc.message
    options = [
        HardwareChoice(
            backend=o.backend.value,
            device=o.device,
            available=o.available,
            reason=o.reason,
            precisions=_precisions(probe, o.device) if o.available else [],
        )
        for o in list_options(probe)
    ]
    return TrainingChoices(auto=auto, auto_error=error, options=options, presets=PRESETS)
