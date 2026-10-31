"""Scan, ingest, data-audit, review and datasets."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from fliptuned.api.routes import body, home_of, read_limited, with_home
from fliptuned.services import audit, configs, datasets, ingest, review, sources
from fliptuned.services.jobs import JobInfo, start_build_dataset, start_ingest
from fliptuned.services.scan import ScanRequest, ScanResult

router = APIRouter()

ScanBody = body(ScanRequest)


@router.post("/scan", response_model=ScanResult)
def scan(payload: ScanBody, request: Request):  # type: ignore[valid-type]
    """Scan a folder inside the allowed folders (file and column names, counts, mapping)."""
    return sources.scan_allowed(with_home(ScanRequest, payload, request))


class IngestBody(BaseModel):
    config: dict[str, Any] = Field(description="An ingest config; source paths must be allowed.")
    overwrite: bool = False


@router.post("/ingest", response_model=JobInfo)
def start_ingest_job(payload: IngestBody, request: Request):
    home = home_of(request)
    sources.check_ingest_config(home, payload.config)
    path = configs.materialize_config(home, "ingest", configs.ConfigInput(config=payload.config))
    job_request = ingest.IngestRequest(config_path=path, home=home, overwrite=payload.overwrite)
    return start_ingest(job_request, detach=True)


@router.get("/audit", response_model=audit.AuditResult)
def get_audit(request: Request, source: str | None = None):
    return audit.run_audit(audit.AuditRequest(home=home_of(request), source=source))


@router.get("/datasets/{dataset}/audit", response_model=audit.ProcessedAuditResult)
def get_processed_audit(dataset: str, request: Request):
    return audit.run_processed_audit(
        audit.ProcessedAuditRequest(home=home_of(request), dataset=dataset)
    )


RecordsBody = body(review.ListRecordsRequest)


@router.post("/records/search", response_model=review.ListRecordsResult)
def search_records(payload: RecordsBody, request: Request):  # type: ignore[valid-type]
    return review.list_records(with_home(review.ListRecordsRequest, payload, request))


DecideBody = body(review.DecideRequest)


@router.post("/records/decisions", response_model=review.DecideResult)
def decide(payload: DecideBody, request: Request):  # type: ignore[valid-type]
    return review.decide(with_home(review.DecideRequest, payload, request))


CsvBody = body(review.CsvDownloadRequest)


@router.post("/review/export", response_class=Response)
def export_review(payload: CsvBody, request: Request):  # type: ignore[valid-type]
    """The review sheet as a CSV download."""
    data, _ = review.export_csv_bytes(with_home(review.CsvDownloadRequest, payload, request))
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="review.csv"'},
    )


@router.post("/review/import", response_model=review.ImportResult)
async def import_review(request: Request):
    """Upload the filled-in review CSV as the raw request body."""
    home = home_of(request)
    data = await read_limited(request, sources.upload_limit_bytes(home))
    with sources.temp_file(home, data) as path:
        return review.import_csv(review.ImportRequest(home=home, path=path))


class BuildBody(BaseModel):
    config: dict[str, Any] = Field(description="A dataset config.")
    allow_download: bool = Field(default=False, description="Download the tokenizer if missing.")
    overwrite: bool = False


@router.post("/datasets", response_model=JobInfo)
def build(payload: BuildBody, request: Request):
    home = home_of(request)
    path = configs.materialize_config(home, "dataset", configs.ConfigInput(config=payload.config))
    job_request = datasets.BuildDatasetRequest(
        config_path=path,
        home=home,
        allow_download=payload.allow_download,
        overwrite=payload.overwrite,
    )
    return start_build_dataset(job_request, detach=True)


@router.get("/datasets", response_model=list[datasets.DatasetInfo])
def list_datasets(request: Request):
    return datasets.list_datasets(home_of(request))


@router.post("/datasets/{dataset}/validate", response_model=datasets.ValidateDatasetResult)
def validate(dataset: str, request: Request):
    return datasets.validate_dataset(
        datasets.ValidateDatasetRequest(home=home_of(request), dataset=dataset)
    )
