"""Jobs: list, get, cancel, the self-test, and live events over SSE.

``GET /api/jobs/{id}/events`` streams the job's ``events.jsonl`` (the same file ``jobs follow``
reads). The SSE ``id`` is a byte offset into that file (on the last event of each batch), so a
reconnecting client resumes with ``Last-Event-ID``. The stream ends after the job's final status; comments keep it alive meanwhile.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from pydantic import BaseModel, Field

from fliptuned.api.routes import home_of
from fliptuned.services import jobs

router = APIRouter()

POLL_S = 0.25
KEEPALIVE_S = 15.0


@router.get("/jobs", response_model=jobs.ListJobsResult)
def list_jobs(
    request: Request,
    status: Annotated[list[jobs.JobStatus] | None, Query()] = None,
    limit: int = Query(default=20, ge=1, le=1000),
):
    return jobs.list_jobs(
        jobs.ListJobsRequest(home=home_of(request), statuses=status or [], limit=limit)
    )


@router.get("/jobs/{job_id}", response_model=jobs.JobInfo)
def get_job(job_id: str, request: Request):
    return jobs.get_job(home_of(request), job_id)


@router.get("/jobs/{job_id}/outcome", response_model=jobs.JobOutcome)
def job_outcome(job_id: str, request: Request):
    """The job with its result (once succeeded) or its error (if failed)."""
    return jobs.job_outcome(home_of(request), job_id)


class CancelBody(BaseModel):
    timeout: float = Field(default=10.0, ge=0, le=600, description="Seconds before force-stopping.")


@router.post("/jobs/{job_id}/cancel", response_model=jobs.CancelOutcome)
def cancel_job(job_id: str, payload: CancelBody, request: Request):
    return jobs.cancel_job(
        jobs.CancelJobRequest(home=home_of(request), job_id=job_id, timeout=payload.timeout)
    )


class SelftestBody(BaseModel):
    seconds: float = Field(default=10.0, gt=0, le=3600)
    steps: int = Field(default=10, ge=1, le=10_000)


@router.post("/jobs/selftest", response_model=jobs.JobInfo)
def selftest(payload: SelftestBody, request: Request):
    """A harmless background job that checks jobs, progress and cancel work on this machine."""
    return jobs.start_selftest(
        jobs.SelftestRequest(home=home_of(request), seconds=payload.seconds, steps=payload.steps),
        detach=True,
    )


@router.get(
    "/jobs/{job_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def job_events(
    job_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    home = home_of(request)
    jobs.get_job(home, job_id)  # an unknown job is a 404, before any streaming
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    return StreamingResponse(_stream(home, job_id, start), media_type="text/event-stream")


def _stream(home: str | None, job_id: str, offset: int) -> Iterator[bytes]:
    last_sent = time.monotonic()
    while True:
        batch = jobs.read_job_events(home, job_id, offset)
        offset = batch.offset
        for number, event in enumerate(batch.events, start=1):
            # Only the batch's last event carries the resume offset: a client cut off mid-batch
            # gets the whole batch again rather than missing part of it.
            yield format_sse_event(
                data_str=json.dumps(event, ensure_ascii=False),
                event=str(event.get("type", "message")),
                id=str(offset) if number == len(batch.events) else None,
            )
            last_sent = time.monotonic()
        if batch.finished:
            return
        if time.monotonic() - last_sent > KEEPALIVE_S:
            yield format_sse_event(comment="keepalive")
            last_sent = time.monotonic()
        time.sleep(POLL_S)
