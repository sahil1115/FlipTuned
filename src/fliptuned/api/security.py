"""Local server hardening (SPEC §13.4).

- Requests whose ``Host`` isn't ``127.0.0.1:<port>`` or ``localhost:<port>`` are refused (blocks
  DNS rebinding).
- Every ``/api`` route depends on :func:`get_principal`, the single auth dependency: the per-launch
  token must arrive in the ``X-FlipTuned-Token`` header, otherwise 401. There are no CORS headers.
- Every response carries a strict Content-Security-Policy and no-sniff / no-referrer headers.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

TOKEN_HEADER = "X-FlipTuned-Token"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@dataclass(frozen=True)
class Principal:
    """Who is calling. v1 has one local user (SPEC §13.5 seam)."""

    workspace_id: str = "local"
    user_id: str = "local-user"


def allowed_hosts(port: int) -> frozenset[str]:
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})


class HostAndHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, port: int):
        super().__init__(app)
        self.hosts = allowed_hosts(port)

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("host", "").lower() not in self.hosts:
            response = JSONResponse(
                {
                    "error": {
                        "message": "This server only answers requests for 127.0.0.1.",
                        "hint": "Open the link printed by `fliptuned serve`.",
                        "details": None,
                        "code": None,
                    }
                },
                status_code=400,
            )
        else:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response


def get_principal(request: Request) -> Principal:
    """The single auth dependency: a valid per-launch token, or 401."""
    sent = request.headers.get(TOKEN_HEADER, "")
    if not sent or not hmac.compare_digest(sent, request.app.state.token):
        raise HTTPException(status_code=401, detail="Missing or wrong access token.")
    return Principal()
