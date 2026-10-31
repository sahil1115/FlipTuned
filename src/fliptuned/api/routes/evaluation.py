"""Evaluations (start, review one pair at a time, judge, score, review-sheet upload) and chat.

The base-model-only baseline isn't offered here: it loads a model synchronously, and the web
server never loads models itself (SPEC §13.3). Evaluations generate baselines as part of the job.

Chat replies stream over SSE: ``delta`` events with text pieces, then ``done`` (or ``error``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from pydantic import BaseModel, Field

from fliptuned.api.routes import home_of, read_limited
from fliptuned.services import chat, configs, evaluation, sources
from fliptuned.services.errors import FlipTunedError

router = APIRouter()


class EvaluateBody(BaseModel):
    config: dict[str, Any] | None = Field(
        default=None, description="Evaluate config (acceptance criteria). Empty = defaults."
    )
    limit: int | None = Field(default=None, ge=1, description="Only the first N test items.")
    allow_download: bool = False


@router.post("/runs/{run_id}/evaluations", response_model=evaluation.EvaluationStart)
def start_evaluation(run_id: str, payload: EvaluateBody, request: Request):
    home = home_of(request)
    path = None
    if payload.config is not None:
        path = configs.materialize_config(
            home, "evaluate", configs.ConfigInput(config=payload.config)
        )
    start_request = evaluation.EvaluateRequest(
        home=home,
        run_id=run_id,
        config_path=path,
        limit=payload.limit,
        allow_download=payload.allow_download,
    )
    return evaluation.start_evaluation(start_request, detach=True)


@router.get("/runs/{run_id}/evaluations", response_model=list[evaluation.EvaluationInfo])
def list_evaluations(run_id: str, request: Request):
    return evaluation.list_evaluations(home_of(request), run_id)


@router.get("/evaluations/{evaluation_id}", response_model=evaluation.EvaluationInfo)
def get_evaluation(evaluation_id: str, request: Request):
    return evaluation.get_evaluation(home_of(request), evaluation_id)


@router.get("/evaluations/{evaluation_id}/pairs", response_model=evaluation.ReviewQueue)
def review_pair(
    evaluation_id: str, request: Request, position: int | None = Query(default=None, ge=1)
):
    """The next unreviewed blind pair (or the one at ``position``). Only A and B, never which
    model wrote which."""
    return evaluation.review_queue(
        evaluation.ReviewRequest(
            home=home_of(request), evaluation_id=evaluation_id, position=position
        )
    )


class JudgeBody(BaseModel):
    item_id: str
    judgement: str = Field(pattern="^(A|B|tie)$", description="A, B or tie.")


@router.post("/evaluations/{evaluation_id}/judgements", response_model=evaluation.JudgeResult)
def judge(evaluation_id: str, payload: JudgeBody, request: Request):
    return evaluation.judge(
        evaluation.JudgeRequest(
            home=home_of(request),
            evaluation_id=evaluation_id,
            item_id=payload.item_id,
            judgement=payload.judgement,
        )
    )


@router.post("/evaluations/{evaluation_id}/score", response_model=evaluation.ScoreResult)
def score(evaluation_id: str, request: Request):
    return evaluation.score_evaluation(
        evaluation.ScoreRequest(home=home_of(request), evaluation_id=evaluation_id)
    )


@router.post(
    "/evaluations/{evaluation_id}/review-sheet", response_model=evaluation.SheetImportResult
)
async def import_sheet(evaluation_id: str, request: Request):
    """Upload a filled-in review_sheet.csv (raw request body) for this evaluation."""
    home = home_of(request)
    data = await read_limited(request, sources.upload_limit_bytes(home))
    with sources.temp_file(home, data) as path:
        info = evaluation.get_evaluation(home, evaluation_id)
        return evaluation.import_review_sheet(
            evaluation.SheetImportRequest(home=home, path=path, run_id=info.run_id)
        )


# --------------------------------------------------------------------------- chat


class ChatBody(BaseModel):
    run_id: str
    base: bool = Field(default=False, description="The base model instead of the fine-tuned one.")
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    allow_download: bool = False


@router.post("/chat/sessions", response_model=chat.ChatInfo)
def start_chat(payload: ChatBody, request: Request):
    """Start a chat worker (it holds the model until closed or idle)."""
    session = chat.start_chat(
        chat.ChatRequest(home=home_of(request), **payload.model_dump()), detach=True
    )
    return session.info


class MessageBody(BaseModel):
    turns: list[dict[str, str]] = Field(
        description="The conversation: alternating user/assistant turns ending with a user turn."
    )


@router.post(
    "/chat/sessions/{job_id}/messages",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def send_message(job_id: str, payload: MessageBody, request: Request) -> StreamingResponse:
    session = chat.attach_session(home_of(request), job_id)
    return StreamingResponse(_reply(session, payload.turns), media_type="text/event-stream")


def _reply(session: chat.ChatSession, turns: list[dict[str, str]]) -> Iterator[bytes]:
    try:
        for piece in session.reply(turns):
            yield format_sse_event(
                data_str=json.dumps({"text": piece}, ensure_ascii=False), event="delta"
            )
        yield format_sse_event(data_str="{}", event="done")
    except (FlipTunedError, ValueError) as exc:
        message = (
            exc.message if isinstance(exc, FlipTunedError) else f"Invalid conversation: {exc}."
        )
        yield format_sse_event(data_str=json.dumps({"message": message}), event="error")


@router.delete("/chat/sessions/{job_id}", response_model=dict)
def close_chat(job_id: str, request: Request) -> dict[str, str]:
    chat.attach_session(home_of(request), job_id).close()
    return {"status": "closed"}
