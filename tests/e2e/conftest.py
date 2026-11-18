"""Shared E2E fixtures: a real ``fliptuned serve`` on a fresh app home, and a token-authed API call.

The app home is a pytest temp dir (set FLIPTUNED_E2E_HOME to use another, e.g. ./.dev-home).
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WEB_DIST = REPO / "src" / "fliptuned" / "web_dist"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def e2e_home(tmp_path: Path) -> Path:
    return Path(os.environ.get("FLIPTUNED_E2E_HOME") or tmp_path / "home")


def launch(home: Path) -> dict:
    """Start ``fliptuned serve`` on ``home``; returns its link, origin, token and process."""
    env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fliptuned",
            "serve",
            "--home",
            str(home),
            "--no-browser",
            "--port",
            str(_free_port()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
    )
    url = None
    assert proc.stdout is not None
    for line in proc.stdout:
        match = re.search(r"(http://127\.0\.0\.1:\d+/#token=\S+)", line)
        if match:
            url = match.group(1)
            break
    if not url:
        proc.terminate()
        raise AssertionError("the server didn't print its link")
    origin = url.split("/#")[0]
    token = url.split("#token=")[1]
    # Keep reading the server's output: a full pipe would block the server on its next write.
    output: deque[str] = deque(maxlen=200)
    threading.Thread(target=lambda: output.extend(proc.stdout), daemon=True).start()
    _wait_until_listening(origin)  # the port is bound before the link prints, listened on after
    return {
        "url": url,
        "origin": origin,
        "token": token,
        "home": home,
        "proc": proc,
        "output": output,
    }


def _wait_until_listening(origin: str, timeout: float = 60.0) -> None:
    port = int(origin.rsplit(":", 1)[1])
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def stop(server: dict) -> None:
    server["proc"].terminate()
    server["proc"].wait(timeout=30)


@pytest.fixture
def server(e2e_home: Path) -> Iterator[dict]:
    if not (WEB_DIST / "index.html").exists():
        pytest.skip("The UI isn't built: run `npm --prefix web run build`.")
    running = launch(e2e_home)
    try:
        yield running
    finally:
        stop(running)


def api(server: dict, method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        server["origin"] + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-FlipTuned-Token": server["token"], "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


@pytest.fixture(autouse=True)
def _screenshots_stay_out_of_the_repo(request, tmp_path, monkeypatch):
    """Screens go to a temp folder. Set FLIPTUNED_E2E_SCREENSHOTS=1 to refresh the images in
    docs/screenshots/ on purpose (for the UI guide)."""
    if os.environ.get("FLIPTUNED_E2E_SCREENSHOTS") != "1" and hasattr(request.module, "SHOTS"):
        monkeypatch.setattr(request.module, "SHOTS", tmp_path / "screenshots")
