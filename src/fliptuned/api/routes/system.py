"""Doctor, status, settings, folder browsing, data sources and uploads, models and downloads,
config schemas and presets."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from fliptuned.api.routes import body, home_of, read_limited, with_home
from fliptuned.services import configs, doctor, models, settings, sources, status
from fliptuned.services.jobs import JobInfo

router = APIRouter()


@router.get("/doctor", response_model=doctor.DoctorReport)
def get_doctor(request: Request):
    return doctor.run_doctor(doctor.DoctorRequest(home=home_of(request)))


@router.get("/status", response_model=status.StatusResult)
def get_status(request: Request):
    return status.get_status(status.StatusRequest(home=home_of(request)))


@router.get("/settings", response_model=settings.Settings)
def get_settings(request: Request):
    return settings.get_settings(home_of(request))


SettingsBody = body(settings.SettingsUpdate)


@router.put("/settings", response_model=settings.Settings)
def put_settings(payload: SettingsBody, request: Request):  # type: ignore[valid-type]
    return settings.update_settings(with_home(settings.SettingsUpdate, payload, request))


@router.get("/fs/browse", response_model=sources.BrowseResult)
def browse(request: Request, path: str | None = Query(default=None)):
    return sources.browse(sources.BrowseRequest(home=home_of(request), path=path))


@router.get("/sources", response_model=list[sources.SourceInfo])
def list_sources(request: Request):
    return sources.list_sources(home_of(request))


@router.post("/sources/{source}/uploads", response_model=sources.UploadResult)
async def upload(
    source: str,
    request: Request,
    filename: str = Query(description="Name of the uploaded file."),
    overwrite: bool = Query(default=False),
):
    """Upload one file as the raw request body (.csv, .xlsx, .md or .txt)."""
    data = await read_limited(request, sources.upload_limit_bytes(home_of(request)))
    upload_request = sources.UploadRequest(
        home=home_of(request), source=source, filename=filename, overwrite=overwrite
    )
    return sources.save_upload(upload_request, data)


@router.get("/models", response_model=models.ModelListResult)
def list_models(request: Request, include_test_only: bool = Query(default=False)):
    return models.list_models(models.ModelListRequest(include_test_only=include_test_only))


DownloadBody = body(models.ModelDownloadRequest, "model")


@router.post("/models/{model_id}/download", response_model=JobInfo)
def download_model(model_id: str, payload: DownloadBody, request: Request):  # type: ignore[valid-type]
    download = with_home(models.ModelDownloadRequest, payload, request, model=model_id)
    return models.start_model_download(download, detach=True)


@router.get("/configs/schema/{kind}")
def config_schema(kind: str) -> dict[str, Any]:
    return configs.json_schema(kind)


@router.get("/training-choices", response_model=configs.TrainingChoices)
def training_choices(model: str | None = Query(default=None, description="Catalog model ID.")):
    """Backend, device and precision options available on this machine, and what 'auto' picks."""
    return configs.training_choices(model)


@router.get("/presets")
def presets() -> dict[str, dict[str, dict[str, Any]]]:
    return configs.presets()
