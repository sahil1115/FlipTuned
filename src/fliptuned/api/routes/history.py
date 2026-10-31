"""Run history: list with filters, compare, artifacts, archive and hard delete."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fliptuned.api.routes import home_of
from fliptuned.services import history

router = APIRouter()


@router.get("/history", response_model=list[history.RunSummary])
def list_history(
    request: Request,
    status: str | None = None,
    model_id: str | None = None,
    text: str | None = None,
    include_archived: bool = False,
    limit: int = Query(default=100, ge=1, le=1000),
):
    return history.list_history(
        history.RunFilter(
            home=home_of(request),
            status=status,
            model_id=model_id,
            text=text,
            include_archived=include_archived,
            limit=limit,
        )
    )


@router.get("/compare-runs", response_model=history.RunComparison)
def compare(request: Request, a: str, b: str):
    return history.compare_runs(home_of(request), a, b)


@router.get("/runs/{run_id}/artifacts", response_model=history.RunArtifactsList)
def artifacts(run_id: str, request: Request):
    return history.list_artifacts(home_of(request), run_id)


@router.get(
    "/runs/{run_id}/artifacts/file",
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def artifact_file(run_id: str, request: Request, path: str = Query(min_length=1)):
    """Download a text artifact (JSON, Markdown, CSV, YAML, text) from the run folder."""
    file = history.artifact_file(home_of(request), run_id, path)
    return FileResponse(file, filename=file.name, media_type="application/octet-stream")


class ArchiveBody(BaseModel):
    archived: bool = Field(description="True to archive, false to restore.")


@router.post("/runs/{run_id}/archive", response_model=history.ArchiveResult)
def archive(run_id: str, payload: ArchiveBody, request: Request):
    return history.set_archived(home_of(request), run_id, payload.archived)


class DeleteBody(BaseModel):
    confirm: str = Field(description="The run ID typed again.")


@router.post("/runs/{run_id}/delete", response_model=history.DeleteResult)
def delete(run_id: str, payload: DeleteBody, request: Request):
    """Permanently delete the run, its evaluations, exports, jobs and files."""
    return history.delete_run(
        history.DeleteRequest(home=home_of(request), run_id=run_id, confirm=payload.confirm)
    )
