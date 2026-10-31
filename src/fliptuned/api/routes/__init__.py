"""Route modules. Each route parses its input, calls a service, and returns the service's result.

The app home always comes from the server (``--home`` / ``FLIPTUNED_HOME``), never from a request:
:func:`body` derives request bodies from service request models without their ``home`` field.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from fastapi import Request
from pydantic import BaseModel, create_model


def home_of(request: Request) -> str | None:
    return request.app.state.home


@cache
def body(model: type[BaseModel], *drop: str) -> type[BaseModel]:
    """``model`` as a request body without ``home`` (and any ``drop`` fields)."""
    hidden = {"home", *drop}
    fields: dict[str, Any] = {
        name: (info.annotation, info)
        for name, info in model.model_fields.items()
        if name not in hidden
    }
    return create_model(f"{model.__name__}Body", **fields)


def with_home(target: type[BaseModel], payload: BaseModel | None, request: Request, **extra):
    """Build the service request ``target`` from a body, adding the server's app home."""
    data = payload.model_dump() if payload is not None else {}
    return target(home=home_of(request), **data, **extra)


async def read_limited(request: Request, limit: int) -> bytes:
    """The raw request body, refusing more than ``limit`` bytes while reading."""
    from fastapi import HTTPException

    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        raise HTTPException(status_code=413, detail="The file is larger than the upload limit.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="The file is larger than the upload limit.")
        chunks.append(chunk)
    return b"".join(chunks)
