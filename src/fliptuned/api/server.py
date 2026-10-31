"""``fliptuned serve``: run the API and UI on 127.0.0.1 (SPEC §13.4).

There is no option to listen on another interface. The port is the requested one or, if taken, the
next free one; the socket is bound here and handed to Uvicorn, so the port can't be lost in between.
Jobs left running by a previous server are reconciled first. The browser opens
``http://127.0.0.1:<port>/#token=<token>``; the fragment never reaches the server or its logs.
Uvicorn's access log is off (URLs can carry search filters).
"""

from __future__ import annotations

import secrets
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass

import uvicorn

from fliptuned.api.app import create_app
from fliptuned.services.errors import NotAvailableError
from fliptuned.services.jobs import reconcile_jobs
from fliptuned.services.settings import get_settings, open_home

HOST = "127.0.0.1"
PORT_ATTEMPTS = 20


@dataclass
class ServerPlan:
    port: int
    token: str
    url: str
    reconciled: list[str]
    sock: socket.socket


def bind_port(preferred: int, attempts: int = PORT_ATTEMPTS) -> socket.socket:
    """A listening-ready socket on 127.0.0.1: ``preferred`` or the next free port."""
    for port in range(preferred, preferred + attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((HOST, port))
        except OSError:
            sock.close()
            continue
        return sock
    raise NotAvailableError(
        f"No free port between {preferred} and {preferred + attempts - 1}.",
        hint="Pass another --port.",
    )


def prepare(home: str | None, port: int | None) -> ServerPlan:
    """Bind ``port``, or the port from Settings when it's None (the next free one if taken)."""
    open_home(home)
    if port is None:
        port = get_settings(home).port
    reconciled = reconcile_jobs(home)
    sock = bind_port(port)
    bound = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    return ServerPlan(
        port=bound,
        token=token,
        url=f"http://{HOST}:{bound}/#token={token}",
        reconciled=reconciled,
        sock=sock,
    )


def run(plan: ServerPlan, home: str | None, *, open_browser: bool) -> None:
    app = create_app(home=home, token=plan.token, port=plan.port)
    config = uvicorn.Config(app, host=HOST, port=plan.port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    if open_browser:
        threading.Thread(target=_open_when_started, args=(server, plan.url), daemon=True).start()
    server.run(sockets=[plan.sock])


def _open_when_started(server: uvicorn.Server, url: str) -> None:
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.1)
    if server.started:
        webbrowser.open(url)
