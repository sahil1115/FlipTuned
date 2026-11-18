"""Chat with a trained run (SPEC §3.1, §13.3): a chat worker holds the model under the lock.

A chat session is a ``chat_session`` job. Its worker loads the run's base model (plus the best
adapter, unless ``base``), then answers requests on a socket bound to 127.0.0.1 only. Each request
is one JSON line carrying a per-session token; the reply streams back as JSON lines. The worker
stops, releasing the model and the lock, when the client closes the session, when it's cancelled,
or after ``idle_timeout_s`` without a request (default: the ``idle_timeout_min`` setting, 10).

The worker writes ``chat.json`` (port and token) into its job folder while it serves and deletes it
when it stops. Nothing of the conversation is written anywhere: not to events, logs or the registry.
The worker reads only the run's plan, metadata and adapter, never dataset or source files.

``start_chat(detach=False)`` runs the worker on a thread of this process (the CLI); with
``detach=True`` it's a background process (the web server). Both are used through :class:`ChatSession`.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import secrets
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fliptuned.config.app_home import AppPaths, resolve_app_home
from fliptuned.errors import FlipTunedError, NotAvailableError
from fliptuned.inference.chat import reply_stream, validate_turns
from fliptuned.jobs.state import TERMINAL, JobStatus, JobType
from fliptuned.jobs.worker import CANCEL_FILE, JobCancelled, JobContext, job_dir
from fliptuned.models.catalog import get_entry
from fliptuned.registry import repositories
from fliptuned.registry.db import connect, migrate
from fliptuned.services.jobs import (
    JobFailedError,
    JobInfo,
    _submit,
    get_job,
    job_error,
    run_inline,
)
from fliptuned.services.runs import load_run
from fliptuned.training.backends import get_backend, load_model
from fliptuned.training.backends.base import ContextTooLongError, GenerationConfig

SESSION_FILE = "chat.json"
DEFAULT_IDLE_MIN = 10
_ACCEPT_POLL_S = 0.25
_REQUEST_TIMEOUT_S = 30.0
_MAX_REQUEST_BYTES = 8 * 1024 * 1024


class ChatSessionError(FlipTunedError):
    """The chat session isn't available (ended, idle, failed)."""


class ChatRequest(BaseModel):
    home: str | None = None
    run_id: str = Field(description="The training run to chat with.")
    base: bool = Field(default=False, description="Chat with the base model, without the adapter.")
    idle_timeout_s: float | None = Field(
        default=None,
        gt=0,
        le=24 * 3600,
        description="Release the model after this long without a message. Empty = the setting.",
    )
    max_new_tokens: int = Field(default=512, ge=1, le=8192, description="Longest reply, in tokens.")
    allow_download: bool = Field(default=False, description="Download model files if missing.")


class ChatInfo(BaseModel):
    run_id: str
    model: str
    base: bool
    backend: str
    device: str
    precision: str
    idle_timeout_s: float
    job: JobInfo


def _paths(home: str | None) -> AppPaths:
    paths = AppPaths(resolve_app_home(home).path)
    if not paths.db.exists():
        raise NotAvailableError(
            "No runs yet.", hint="Train one with `fliptuned train --config <file>`."
        )
    migrate(paths.db)
    return paths


def _idle_setting(paths: AppPaths) -> float:
    with connect(paths.db) as conn:
        minutes = repositories.get_setting(conn, "idle_timeout_min", DEFAULT_IDLE_MIN)
    return float(minutes) * 60


# --------------------------------------------------------------------------- client side


class ChatSession:
    """Client handle of a chat worker. Use :meth:`reply` for each user message, then :meth:`close`."""

    def __init__(self, home: str | None, info: ChatInfo, thread: threading.Thread | None = None):
        self.home = home
        self.info = info
        self._paths = AppPaths(resolve_app_home(home).path)
        self._thread = thread
        self._endpoint: tuple[int, str] | None = None
        self.last_finish_reason: str | None = None

    @property
    def job_id(self) -> str:
        return self.info.job.id

    def _session_file(self) -> Path:
        return job_dir(self._paths, self.job_id) / SESSION_FILE

    def _ended(self) -> FlipTunedError:
        job = get_job(self.home, self.job_id)
        error = job_error(self.home, self.job_id)
        if error is not None:  # e.g. the model files are missing
            return JobFailedError(
                error["message"],
                hint=error.get("hint"),
                details=error.get("details"),
                exit_code=error.get("exit_code", 1),
            )
        return ChatSessionError(
            f"The chat session has ended ({job.status}).",
            hint="It stops after the idle timeout or when closed; start chat again.",
        )

    def wait_ready(self, timeout: float | None = None) -> None:
        """Wait until the worker has loaded the model (or raise why it stopped)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._endpoint is None:
            path = self._session_file()
            if path.exists():  # written atomically by the worker (temp file, then rename)
                data = json.loads(path.read_text(encoding="utf-8"))
                self._endpoint = (int(data["port"]), str(data["token"]))
                return
            if JobStatus(get_job(self.home, self.job_id).status) in TERMINAL:
                raise self._ended()
            if deadline is not None and time.monotonic() > deadline:
                raise ChatSessionError(
                    "The chat model is still loading.",
                    hint=f"Follow it with `fliptuned jobs follow {self.job_id}`.",
                )
            time.sleep(0.1)

    def _request(self, payload: dict[str, Any]) -> socket.socket:
        self.wait_ready()
        assert self._endpoint is not None
        port, token = self._endpoint
        try:
            conn = socket.create_connection(("127.0.0.1", port), timeout=_REQUEST_TIMEOUT_S)
        except OSError:
            self._endpoint = None
            raise self._ended() from None
        conn.sendall((json.dumps({"token": token, **payload}) + "\n").encode("utf-8"))
        return conn

    def reply(self, turns: list[dict[str, str]]) -> Iterator[str]:
        """Stream the model's reply to ``turns`` (alternating user/assistant, ending with user)."""
        validate_turns(turns)
        self.last_finish_reason = None
        conn = self._request({"op": "generate", "turns": turns})
        conn.settimeout(None)  # generation can take a while on CPU
        with conn, conn.makefile("r", encoding="utf-8", newline="\n") as lines:
            for line in lines:
                message = json.loads(line)
                if message["type"] == "delta":
                    yield message["text"]
                elif message["type"] == "done":
                    self.last_finish_reason = message.get("finish_reason")
                    return
                else:
                    raise ChatSessionError(
                        message.get("message", "The chat worker refused the request.")
                    )
        raise self._ended()

    def close(self, timeout: float = 30.0) -> None:
        """End the session: the worker releases the model and the lock."""
        if JobStatus(get_job(self.home, self.job_id).status) not in TERMINAL:
            try:
                if self._session_file().exists() or self._endpoint is not None:
                    with self._request({"op": "close"}) as conn:
                        conn.recv(64)
                else:
                    # Still loading: the worker stops at its next cancel check.
                    marker = job_dir(self._paths, self.job_id) / CANCEL_FILE
                    marker.write_text("1", encoding="utf-8")
            except (ChatSessionError, OSError):
                pass
        if self._thread is not None:
            self._thread.join(timeout)


def start_chat(request: ChatRequest, *, detach: bool) -> ChatSession:
    """Start a chat worker (refused while another job holds the model) and return its handle."""
    paths = _paths(request.home)
    run = load_run(paths, request.run_id, require_adapter=not request.base)
    selection = run.selection
    get_backend(selection.backend)  # fails clearly off-platform (e.g. MLX on Windows)
    idle = request.idle_timeout_s or _idle_setting(paths)
    params = {
        "run_id": request.run_id,
        "base": request.base,
        "idle_timeout_s": idle,
        "max_new_tokens": request.max_new_tokens,
        "allow_download": request.allow_download,
    }
    job = _submit(request.home, JobType.CHAT_SESSION, params, detach, run_id=request.run_id)
    info = ChatInfo(
        run_id=request.run_id,
        model=run.catalog_id,
        base=request.base,
        backend=selection.backend.value,
        device=selection.device,
        precision=selection.precision,
        idle_timeout_s=idle,
        job=job,
    )
    thread = None
    if not detach:
        thread = threading.Thread(
            target=run_inline, args=(request.home, job.id), name=f"chat-{job.id}", daemon=True
        )
        thread.start()
    return ChatSession(request.home, info, thread)


# --------------------------------------------------------------------------- worker side


def _send(conn: socket.socket, message: dict[str, Any]) -> None:
    conn.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_request(conn: socket.socket) -> dict[str, Any]:
    conn.settimeout(_REQUEST_TIMEOUT_S)
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > _MAX_REQUEST_BYTES:
            raise ValueError("request too large")
    return json.loads(data.decode("utf-8"))


def _write_session_file(folder: Path, port: int, token: str) -> Path:
    path = folder / SESSION_FILE
    tmp = folder / f".{SESSION_FILE}.tmp"
    tmp.write_text(json.dumps({"port": port, "token": token}), encoding="utf-8")
    tmp.replace(path)
    return path


def _reply_error(exc: Exception) -> str:
    """What to tell the client. Only the exception type: messages can contain chat text."""
    if isinstance(exc, ContextTooLongError):
        return "The conversation is too long for this model. Start a new chat."
    name = type(exc).__name__
    if isinstance(exc, MemoryError) or name == "OutOfMemoryError":
        return "The model ran out of memory while answering. Try a shorter message or start a new chat."
    return f"The model failed to answer ({name}). Try again, or load the model again."


def serve_chat(ctx: JobContext) -> dict[str, Any]:
    """Handler of the ``chat_session`` job."""
    params = ctx.params
    base = bool(params.get("base"))
    run = load_run(ctx.paths, params["run_id"], require_adapter=not base)
    entry = get_entry(run.catalog_id)
    generation = GenerationConfig(max_new_tokens=int(params.get("max_new_tokens", 512)))
    idle = float(params["idle_timeout_s"])
    backend, loaded = load_model(
        entry,
        run.selection,
        allow_download=bool(params.get("allow_download")),
        adapter_path=None if base else run.adapter,
    )
    token = secrets.token_urlsafe(32)
    requests = 0
    ended = "closed"
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(_ACCEPT_POLL_S)
    session_file = _write_session_file(
        job_dir(ctx.paths, ctx.job_id), server.getsockname()[1], token
    )
    ctx.emit("chat", state="ready", model=entry.id, adapter=not base)
    last = time.monotonic()
    try:
        while True:
            if ctx.cancel_requested():
                raise JobCancelled
            if time.monotonic() - last > idle:
                ended = "idle_timeout"
                break
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            with conn:
                try:
                    request = _read_request(conn)
                except (ValueError, OSError):
                    continue  # malformed or abandoned connection: ignore it
                if not hmac.compare_digest(str(request.get("token", "")), token):
                    with contextlib.suppress(OSError):
                        _send(conn, {"type": "error", "message": "Not authorised."})
                    continue
                last = time.monotonic()
                op = request.get("op")
                if op == "close":
                    with contextlib.suppress(OSError):
                        _send(conn, {"type": "done"})
                    break
                if op != "generate":
                    with contextlib.suppress(OSError):
                        _send(conn, {"type": "error", "message": "Unknown request."})
                    continue
                try:
                    turns = validate_turns(request.get("turns"))
                except ValueError as exc:
                    with contextlib.suppress(OSError):
                        _send(conn, {"type": "error", "message": f"Invalid conversation: {exc}."})
                    continue
                requests += 1
                try:
                    for piece in reply_stream(
                        backend, loaded, run.system_prompt, turns, generation
                    ):
                        _send(conn, {"type": "delta", "text": piece})
                    _send(
                        conn,
                        {"type": "done", "finish_reason": loaded.extra.get("last_finish_reason")},
                    )
                except OSError:
                    pass  # the client hung up mid-reply
                except Exception as exc:  # generation failed: tell the client, keep serving
                    ctx.emit(
                        "log",
                        level="warning",
                        message=f"A reply failed with {type(exc).__name__}.",
                    )
                    with contextlib.suppress(OSError):
                        _send(conn, {"type": "error", "message": _reply_error(exc)})
                last = time.monotonic()
    finally:
        server.close()
        session_file.unlink(missing_ok=True)
        backend.unload(loaded)
    ctx.emit("chat", state=ended, requests=requests)
    return {"run_id": run.run_id, "base": base, "requests": requests, "ended": ended}


def attach_session(home: str | None, job_id: str) -> ChatSession:
    """The client handle of a running chat worker (the web server keeps no session state)."""
    job = get_job(home, job_id)
    if job.type != JobType.CHAT_SESSION.value:
        raise NotAvailableError(f"Job {job_id} is not a chat session.")
    paths = _paths(home)
    with connect(paths.db) as conn:
        params = json.loads(repositories.get_job(conn, job_id)["params_json"] or "{}")["params"]
    run = load_run(paths, params["run_id"], require_adapter=False)
    selection = run.selection
    info = ChatInfo(
        run_id=run.run_id,
        model=run.catalog_id,
        base=bool(params.get("base")),
        backend=selection.backend.value,
        device=selection.device,
        precision=selection.precision,
        idle_timeout_s=float(params["idle_timeout_s"]),
        job=job,
    )
    return ChatSession(home, info)
