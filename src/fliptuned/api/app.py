"""The FastAPI app (SPEC §13.1, §13.4): thin routes over ``fliptuned.services``.

Errors come back as ``{"error": {"message", "hint", "details", "code"}}`` without record content:
request-validation errors drop the submitted values, unexpected errors carry no message at all.
There are no interactive docs pages (they load assets from CDNs); the OpenAPI schema is served
behind the token at ``/api/openapi.json`` and exported for the frontend client.

The built frontend (``fliptuned/web_dist``, Phase 10+) is served from ``/``; until it exists a
minimal placeholder page is served instead.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from fliptuned.api.routes import data, evaluation, history, jobs, system, training
from fliptuned.api.security import HostAndHeadersMiddleware, get_principal
from fliptuned.services import runtime
from fliptuned.services.errors import (
    ConfigError,
    ExitCode,
    FlipTunedError,
    ModelLockedError,
    NotAvailableError,
    PathNotAllowedError,
)

STATIC = Path(__file__).with_name("static")
WEB_DIST = Path(__file__).resolve().parents[1] / "web_dist"


def _status_for(exc: FlipTunedError) -> int:
    if isinstance(exc, ModelLockedError):
        return 409
    if isinstance(exc, PathNotAllowedError):
        return 403
    if isinstance(exc, ConfigError):
        return 422
    if isinstance(exc, NotAvailableError):
        return 404
    if exc.exit_code == ExitCode.CHECK_FAILED:
        return 409
    return 400


def _error(status: int, message: str, *, hint=None, details=None, code=None) -> JSONResponse:
    body = {"message": message, "hint": hint, "details": details, "code": code}
    return JSONResponse({"error": body}, status_code=status)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(FlipTunedError)
    async def _flip(request: Request, exc: FlipTunedError) -> JSONResponse:
        return _error(
            _status_for(exc),
            exc.message,
            hint=exc.hint,
            details=exc.details,
            code=int(exc.exit_code),
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: never the submitted values (they can be content).
        lines = [
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
            for err in exc.errors()
        ]
        return _error(422, "The request is not valid.", details="\n".join(lines))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request refused."
        return _error(exc.status_code, message)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, "Something went wrong on the server.", hint="See the server console.")


def create_app(*, home: str | None, token: str, port: int, web_dist: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="FlipTuned",
        version=runtime.version(),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.home = home
    app.state.token = token
    app.add_middleware(HostAndHeadersMiddleware, port=port)
    _install_error_handlers(app)

    api = APIRouter(prefix="/api", dependencies=[Depends(get_principal)])
    for module in (system, data, training, history, jobs, evaluation):
        api.include_router(module.router)

    @api.get("/openapi.json", include_in_schema=False)
    def openapi_schema() -> dict:
        return app.openapi()

    @api.api_route(
        "/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False
    )
    def unknown(rest: str) -> JSONResponse:
        return _error(404, "No such API endpoint.")

    app.include_router(api)

    web_dist = WEB_DIST if web_dist is None else web_dist
    if (web_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        def placeholder() -> FileResponse:
            return FileResponse(STATIC / "placeholder.html", media_type="text/html")

        @app.get("/placeholder.js", include_in_schema=False)
        def placeholder_js() -> FileResponse:
            return FileResponse(STATIC / "placeholder.js", media_type="text/javascript")

    return app


def openapi_document() -> dict:
    """The API schema, for the generated TypeScript client (no server needed)."""
    return create_app(home=None, token="unused", port=0).openapi()
