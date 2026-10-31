"""SPEC §13.4 / §18 security: Host allowlist, token, no CORS, CSP, error content, 127.0.0.1 only,
allowed-folder browsing (traversal, UNC, links), upload limits."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from fliptuned.api import server
from fliptuned.api.app import create_app
from fliptuned.cli import app as cli_app
from fliptuned.services import status as status_service
from tests.api.conftest import BASE, PORT, TOKEN
from tests.helpers.pipeline import FIXTURES

SECRET = "zebra-37 private answer text"


def test_token_is_required(anon: TestClient, client: TestClient) -> None:
    for path in ("/api/doctor", "/api/status", "/api/settings", "/api/no-such-thing"):
        response = anon.get(path)
        assert response.status_code == 401, path
        assert response.json()["error"]["message"] == "Missing or wrong access token."
    assert anon.get("/api/status", headers={"X-FlipTuned-Token": "wrong"}).status_code == 401
    assert anon.get("/api/status", headers={"X-FlipTuned-Token": ""}).status_code == 401
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/no-such-thing").status_code == 404


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8765",
        "127.0.0.1",
        "127.0.0.1:9999",
        "localhost",
        "0.0.0.0:8765",
    ],
)
def test_wrong_host_is_rejected(client: TestClient, host: str) -> None:
    response = client.get("/api/status", headers={"Host": host})
    assert response.status_code == 400
    assert "127.0.0.1" in response.json()["error"]["message"]
    assert client.get("/", headers={"Host": host}).status_code == 400


def test_localhost_host_is_accepted(client: TestClient) -> None:
    assert client.get("/api/status", headers={"Host": f"localhost:{PORT}"}).status_code == 200


def test_security_headers_and_no_cors(anon: TestClient, client: TestClient) -> None:
    for response in (
        client.get("/api/status"),
        anon.get("/api/status"),
        client.get("/"),
        client.get("/api/status", headers={"Host": "evil.example"}),
    ):
        assert response.headers["content-security-policy"].startswith("default-src 'self'")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert not any(h.lower().startswith("access-control-") for h in response.headers)
    preflight = client.options(
        "/api/status",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in preflight.headers
    cross = client.get("/api/status", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in cross.headers


def test_placeholder_page_needs_no_inline_script(home: Path, tmp_path: Path) -> None:
    app = create_app(home=str(home), token=TOKEN, port=PORT, web_dist=tmp_path / "not-built")
    anon = TestClient(app, base_url=BASE)
    page = anon.get("/")
    assert page.status_code == 200 and "FlipTuned" in page.text
    assert '<script src="/placeholder.js"></script>' in page.text
    assert page.text.count("<script") == 1  # no inline script (CSP default-src 'self')
    script = anon.get("/placeholder.js")
    assert script.status_code == 200 and "X-FlipTuned-Token" in script.text
    assert "http" not in script.text.replace("HTTP ", "")  # no external URLs or CDNs
    assert anon.get("/docs").status_code == 404 and anon.get("/openapi.json").status_code == 404


def test_errors_carry_no_submitted_content(client: TestClient, monkeypatch) -> None:
    invalid = client.post(
        "/api/records/decisions", json={"records": [{"source_id": SECRET}], "status": SECRET}
    )
    assert invalid.status_code == 422 and SECRET not in invalid.text
    assert "status" in invalid.json()["error"]["details"]

    def boom(request):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(status_service, "get_status", boom)
    app = create_app(home=None, token=TOKEN, port=PORT)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        crashed = c.get("/api/status", headers={"X-FlipTuned-Token": TOKEN})
    assert crashed.status_code == 500 and SECRET not in crashed.text
    assert crashed.json()["error"]["message"] == "Something went wrong on the server."


def test_server_binds_127_0_0_1_only(tmp_path: Path) -> None:
    assert server.HOST == "127.0.0.1"
    plan = server.prepare(str(tmp_path / "home"), 0)
    try:
        assert plan.sock.getsockname()[0] == "127.0.0.1"
        assert plan.url.startswith(f"http://127.0.0.1:{plan.port}/#token=")
        assert len(plan.token) >= 32
    finally:
        plan.sock.close()
    help_text = CliRunner().invoke(cli_app, ["serve", "--help"]).output
    assert "--host" not in help_text and "--port" in help_text and "--no-browser" in help_text


def test_serve_uses_the_port_from_settings(tmp_path: Path) -> None:
    from fliptuned.services.settings import SettingsUpdate, update_settings

    with socket.socket() as probe:  # a port that is free right now
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    home = str(tmp_path / "home")
    update_settings(SettingsUpdate(home=home, port=free))
    plan = server.prepare(home, None)
    try:
        assert plan.port == free
    finally:
        plan.sock.close()


def test_port_falls_back_to_the_next_free_one() -> None:
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    busy = taken.getsockname()[1]
    try:
        sock = server.bind_port(busy)
        assert sock.getsockname()[1] != busy
        sock.close()
    finally:
        taken.close()


# --------------------------------------------------------------------------- folders and uploads


def test_browse_stays_inside_allowed_folders(client: TestClient, tmp_path: Path) -> None:
    roots = client.get("/api/fs/browse").json()
    assert {e["path"] for e in roots["entries"]} >= {str(FIXTURES)}
    listing = client.get("/api/fs/browse", params={"path": str(FIXTURES)}).json()
    names = {e["name"]: e for e in listing["entries"]}
    assert names["qa_basic.csv"]["supported"] and names["qa_basic.csv"]["kind"] == "file"
    assert not names["make_fixtures.py"]["supported"]
    assert names["scan_folder"]["kind"] == "folder"
    assert listing["parent"] is None  # the root itself: no way up

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for path in (
        str(outside),
        str(FIXTURES / ".." / ".."),
        "\\\\server\\share\\data" if os.name == "nt" else "//server/share/data",
        str(FIXTURES / "nope"),
    ):
        response = client.get("/api/fs/browse", params={"path": path})
        assert response.status_code == 403, path
    assert client.post("/api/scan", json={"path": str(outside)}).status_code == 403


def test_links_out_of_allowed_folders_are_refused(
    client: TestClient, tmp_path: Path, home: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "keys.txt").write_text("x", encoding="utf-8")
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here (Windows without Developer Mode)")
    assert client.put("/api/settings", json={"allowed_roots": [str(root)]}).status_code == 200
    listing = client.get("/api/fs/browse", params={"path": str(root)}).json()
    assert [e["name"] for e in listing["entries"]] == []  # the link is hidden
    assert client.get("/api/fs/browse", params={"path": str(link)}).status_code == 403


def test_settings_refuse_the_app_home_and_ignore_a_client_home(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    refused = client.put("/api/settings", json={"allowed_roots": [str(home)]})
    assert refused.status_code == 422 and "app home" in refused.json()["error"]["message"]
    other = tmp_path / "other-home"
    response = client.put("/api/settings", json={"home": str(other), "idle_timeout_min": 5})
    assert response.status_code == 200 and response.json()["app_home"] == str(home)
    assert not other.exists()


def test_uploads_are_checked(client: TestClient, home: Path) -> None:
    client.put("/api/settings", json={"upload_max_mb": 1})
    ok = client.post(
        "/api/sources/my-upload/uploads",
        params={"filename": "qa.csv"},
        content=b"question,answer\n",
    )
    assert ok.status_code == 200
    saved = Path(ok.json()["file"])
    assert saved.parent.parent == home / "uploads" and saved.read_bytes() == b"question,answer\n"
    again = client.post(
        "/api/sources/my-upload/uploads", params={"filename": "qa.csv"}, content=b"x"
    )
    assert again.status_code == 400
    too_big = client.post(
        "/api/sources/my-upload/uploads",
        params={"filename": "big.csv"},
        content=b"x" * (1024 * 1024 + 1),
    )
    assert too_big.status_code == 413
    for name in ("run.exe", "../escape.csv", "sub\\dir.csv", "CON.csv"):
        bad = client.post("/api/sources/my-upload/uploads", params={"filename": name}, content=b"x")
        assert bad.status_code == 422, name
    # The upload folder is browsable (it's always an allowed folder)
    listing = client.get("/api/fs/browse", params={"path": ok.json()["folder"]}).json()
    assert [e["name"] for e in listing["entries"]] == ["qa.csv"]


def test_ingest_config_paths_must_be_allowed(client: TestClient, tmp_path: Path) -> None:
    for path in (str(tmp_path), "relative/folder"):
        config = {"name": "x", "sources": [{"path": path}]}
        assert client.post("/api/ingest", json={"config": config}).status_code == 403, path


def test_built_ui_is_served_when_present(home: Path, tmp_path: Path) -> None:
    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<script type="module" src="/assets/app.js"></script>', "utf-8"
    )
    (dist / "assets" / "app.js").write_text("console.log(1)", "utf-8")
    anon = TestClient(
        create_app(home=str(home), token=TOKEN, port=PORT, web_dist=dist), base_url=BASE
    )
    page = anon.get("/")
    assert page.status_code == 200 and "/assets/app.js" in page.text
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    assert anon.get("/assets/app.js").status_code == 200
    assert anon.get("/api/status").status_code == 401  # the API still needs the token
    assert anon.get("/assets/app.js", headers={"Host": "evil.example"}).status_code == 400
