"""Training configs, preflight, runs (train, resume) and exports."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from fliptuned.api.routes import home_of
from fliptuned.services import configs, export, monitor, training
from fliptuned.services.preflight import PreflightResult

router = APIRouter()


@router.get("/configs", response_model=list[configs.ConfigSummary])
def list_configs(request: Request):
    return configs.list_configs(configs.ListConfigsRequest(home=home_of(request)))


class SaveBody(BaseModel):
    config: dict[str, Any] = Field(
        description="A training config; saved as its name's next version."
    )


@router.post("/configs", response_model=configs.SavedConfig)
def save_config(payload: SaveBody, request: Request):
    return configs.save_config_data(
        configs.SaveConfigRequest(home=home_of(request), config=payload.config)
    )


@router.get("/configs/{name}", response_model=configs.ConfigDetail)
def show_config(name: str, request: Request, version: int | None = Query(default=None, ge=1)):
    return configs.show_config(
        configs.ShowConfigRequest(home=home_of(request), name=name, version=version)
    )


class DuplicateBody(BaseModel):
    new_name: str = Field(min_length=1)


@router.post("/configs/{name}/duplicate", response_model=configs.SavedConfig)
def duplicate_config(name: str, payload: DuplicateBody, request: Request):
    return configs.duplicate_config(
        configs.DuplicateConfigRequest(home=home_of(request), name=name, new_name=payload.new_name)
    )


class TrainBody(BaseModel):
    saved_name: str | None = Field(default=None, description="A saved training config.")
    saved_version: int | None = Field(default=None, ge=1)
    config: dict[str, Any] | None = Field(default=None, description="Or the config itself.")
    allow_download: bool = Field(default=False, description="Download model files if missing.")


def _train_request(payload: TrainBody, request: Request) -> training.TrainRequest:
    home = home_of(request)
    source = configs.ConfigInput(
        config=payload.config, saved_name=payload.saved_name, saved_version=payload.saved_version
    )
    path = configs.materialize_config(home, "train", source)
    return training.TrainRequest(config_path=path, home=home, allow_download=payload.allow_download)


@router.post("/preflight", response_model=PreflightResult)
def preflight(payload: TrainBody, request: Request):
    return training.preflight(_train_request(payload, request))


@router.post("/runs", response_model=training.TrainStart)
def start_run(payload: TrainBody, request: Request):
    return training.start_training(_train_request(payload, request), detach=True)


@router.get("/runs", response_model=list[training.RunInfo])
def list_runs(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    return training.list_runs(home_of(request), limit)


@router.get("/runs/{run_id}", response_model=training.RunInfo)
def get_run(run_id: str, request: Request):
    return training.get_run(home_of(request), run_id)


@router.get("/runs/{run_id}/monitor", response_model=monitor.RunMonitor)
def run_monitor(run_id: str, request: Request):
    """The run's live state; continue with ``/api/jobs/{job.id}/events`` from ``offset``."""
    return monitor.run_monitor(home_of(request), run_id)


class ResumeBody(BaseModel):
    allow_download: bool = False


@router.post("/runs/{run_id}/resume", response_model=training.TrainStart)
def resume(run_id: str, payload: ResumeBody, request: Request):
    resume_request = training.ResumeRequest(
        home=home_of(request), run_id=run_id, allow_download=payload.allow_download
    )
    return training.resume_run(resume_request, detach=True)


class ExportBody(BaseModel):
    merge: bool = Field(default=False, description="Also build a merged standalone model.")
    allow_download: bool = False


@router.post("/runs/{run_id}/exports", response_model=export.ExportStart)
def start_export(run_id: str, payload: ExportBody, request: Request):
    export_request = export.ExportRequest(
        home=home_of(request),
        run_id=run_id,
        merge=payload.merge,
        allow_download=payload.allow_download,
    )
    return export.export_run(export_request, detach=True)


@router.get("/exports", response_model=list[export.ExportInfo])
def list_exports(request: Request, run_id: str | None = None):
    return export.list_exports(home_of(request), run_id)
