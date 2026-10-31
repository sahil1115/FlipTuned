"""E2E (markers ``e2e`` + ``tiny``): start, watch, cancel and resume a training run in the real UI.

Preflight → Start training → the run monitor shows live loss points → reload mid-run, still live →
restart ``serve`` mid-run, the new server shows the same live run → a second model job is refused
with the model-lock message → Cancel → cancelled → Resume → succeeded with the best checkpoint.

The tiny model trains on the CPU; the ``mlx`` variant uses the smoke model on Apple Silicon. The app
home gets a fixture dataset first, through the CLI. Every browser request must stay on 127.0.0.1 and
the page must log no console errors. Screens are saved to docs/screenshots/phase-12/.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.e2e.conftest import REPO, api, launch, stop

SHOTS = REPO / "docs" / "screenshots" / "phase-12"

pytestmark = pytest.mark.e2e

VARIANTS = [
    pytest.param("tiny-qwen2", {}, id="hf_peft-cpu", marks=pytest.mark.tiny),
    pytest.param(
        "qwen2.5-coder-0.5b",
        {"backend": "mlx"},
        id="mlx",
        marks=[
            pytest.mark.mlx,
            pytest.mark.slow,
            pytest.mark.skipif(
                not (sys.platform == "darwin" and platform.machine() == "arm64"),
                reason="MLX needs a Mac with Apple Silicon",
            ),
        ],
    ),
]


def _cli(home: Path, *args: str) -> None:
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run(
        [sys.executable, "-m", "fliptuned", *args, "--home", str(home)],
        check=True,
        capture_output=True,
        env=env,
        cwd=REPO,
        timeout=600,
    )


@pytest.fixture
def e2e_home(tmp_path: Path) -> Path:
    """A fresh home holding a fixture dataset (overrides the plain home from conftest)."""
    home = tmp_path / "home"
    _cli(home, "ingest", "--config", "configs/ingest-synthetic.sample.yaml")
    _cli(home, "build-dataset", "--config", "configs/dataset.sample.yaml", "--allow-download")
    return home


STEP_MOVED = (
    "n => [...document.querySelectorAll('dd')]"
    r".some(d => /^\d+ \/ \d+$/.test(d.textContent) && parseInt(d.textContent) > n)"
)


def _steps(page) -> int:
    """The step the monitor shows (0 before the first report)."""
    text = page.get_by_text(re.compile(r"^\d+ / \d+$")).first.text_content() or "0 /"
    return int(text.split("/")[0])


@pytest.mark.parametrize(("model", "extra"), VARIANTS)
def test_run_start_watch_cancel_resume(server: dict, model: str, extra: dict) -> None:
    from playwright.sync_api import expect, sync_playwright

    config = {
        "kind": "train",
        "name": "e2e-run",
        "model": {"catalog_id": model},
        "preset": "quick",
        "training": {
            "max_epochs": 3,
            "patience": 50,  # no early stop: the run must still be going at Cancel
            "eval_every_steps": 4,
            "gradient_checkpointing": False,
        },
        **extra,
    }
    api(server, "POST", "/api/configs", {"config": config})
    SHOTS.mkdir(parents=True, exist_ok=True)
    servers = [server]

    requests: list[str] = []
    console_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        def open_page(url: str):
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.on("request", lambda r: requests.append(r.url))
            page.on(
                "console", lambda m: console_errors.append(m.text) if m.type == "error" else None
            )
            page.set_default_timeout(60_000)
            page.goto(url)
            return page

        page = open_page(server["url"])
        nav = page.get_by_role("navigation", name="Main")

        # Preflight → Start: the monitor opens for the new run
        nav.get_by_role("link", name="Preflight").click()
        page.get_by_label("Download model files if they're missing").check()
        expect(page.get_by_text(re.compile(r"^Ready to train"))).to_be_visible(timeout=300_000)
        page.get_by_role("button", name="Start training").click()
        expect(page.get_by_role("heading", name="Run monitor", exact=True)).to_be_visible()
        run_id = re.search(r"run=(run_\w+)", page.url).group(1)
        expect(page.get_by_text("Training in the background", exact=False)).to_be_visible(
            timeout=120_000
        )

        # Live loss points appear
        log = page.get_by_role("log", name="Event log")
        expect(log.get_by_text(re.compile(r"train loss"))).not_to_have_count(0, timeout=300_000)
        expect(page.get_by_role("img", name=re.compile(r"^Loss chart"))).to_be_visible()
        expect(page.get_by_text("● Live")).to_be_visible()
        page.screenshot(path=str(SHOTS / "monitor-running.png"), full_page=True)

        # Reload mid-run: same run, still live, still moving
        before = _steps(page)
        page.reload()
        expect(page.get_by_label("Run", exact=True)).to_have_value(run_id)
        expect(page.get_by_text("● Live")).to_be_visible()
        expect(page.get_by_text("Training in the background", exact=False)).to_be_visible()
        page.wait_for_function(STEP_MOVED, arg=before, timeout=120_000)

        # Restart `serve` mid-run: the worker keeps going; the new server shows it live
        page.close()
        stop(server)
        servers.append(launch(server["home"]))
        page = open_page(servers[-1]["url"])
        active = page.get_by_role("region", name="Active job")
        expect(active.get_by_text("train", exact=True)).to_be_visible()
        page.screenshot(path=str(SHOTS / "dashboard-active-job.png"), full_page=True)
        active.get_by_role("link", name="Open the run monitor").click()
        expect(page.get_by_label("Run", exact=True)).to_have_value(run_id)
        expect(page.get_by_text("● Live")).to_be_visible()
        before = _steps(page)
        page.wait_for_function(STEP_MOVED, arg=before, timeout=120_000)

        # A second model job: Preflight names the holder and keeps Start disabled; the API refuses
        nav = page.get_by_role("navigation", name="Main")
        nav.get_by_role("link", name="Preflight").click()
        expect(
            page.get_by_text(
                re.compile(r"^Another job is using the model: job_\w+ \(train, running\)")
            )
        ).to_be_visible()
        expect(page.get_by_role("button", name="Start training")).to_be_disabled(timeout=180_000)
        page.screenshot(path=str(SHOTS / "preflight-model-locked.png"), full_page=True)
        with pytest.raises(urllib.error.HTTPError) as refused:
            api(servers[-1], "POST", "/api/runs", {"saved_name": "e2e-run"})
        assert refused.value.code == 409
        assert (
            "Another job is using the model" in json.loads(refused.value.read())["error"]["message"]
        )

        # Cancel → cancelled, Resume enabled
        page.get_by_role("link", name="Watch it").click()
        expect(page.get_by_label("Run", exact=True)).to_have_value(run_id)
        page.get_by_role("button", name="Cancel").click()
        expect(page.get_by_text("You cancelled training.", exact=False)).to_be_visible(
            timeout=120_000
        )
        expect(page.get_by_role("button", name="Resume")).to_be_enabled()
        expect(page.get_by_role("button", name="Cancel")).to_be_disabled()
        page.screenshot(path=str(SHOTS / "monitor-cancelled.png"), full_page=True)

        # Resume → running again → succeeded with the best checkpoint marked
        page.get_by_role("button", name="Resume").click()
        expect(page.get_by_role("list", name="Attempts").get_by_text("Resume 1")).to_be_visible()
        expect(page.get_by_text("Training finished.", exact=False)).to_be_visible(timeout=600_000)
        result = page.get_by_role("region", name="Result")
        expect(result.get_by_text("Best step")).to_be_visible()
        expect(page.get_by_role("img", name=re.compile(r"Best: validation loss"))).to_be_visible()
        page.screenshot(path=str(SHOTS / "monitor-succeeded.png"), full_page=True)
        page.get_by_role("button", name="Dark theme").click()
        page.screenshot(path=str(SHOTS / "monitor-succeeded-dark.png"), full_page=True)
        browser.close()

    try:
        hosts = {urlparse(u).hostname for u in requests if not u.startswith(("data:", "blob:"))}
        assert hosts == {"127.0.0.1"}, f"requests left 127.0.0.1: {hosts}"
        assert console_errors == [], console_errors
        state = api(servers[-1], "GET", f"/api/runs/{run_id}/monitor")
        assert state["run"]["status"] == "succeeded"
        assert [a["status"] for a in state["attempts"]] == ["cancelled", "succeeded"]
        assert state["best"]["final"] and state["result"]["adapter_path"]
    finally:
        if len(servers) > 1:
            stop(servers[-1])
