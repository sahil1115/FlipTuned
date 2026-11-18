"""E2E (marker ``e2e``): models, training config and preflight in the real UI.

choose the tiny model (dev mode) → save a config → duplicate it → preflight shows real results →
an oversized config (a 2B model on the CPU) shows a failing check with guidance and Start disabled

The app home gets a fixture dataset first, through the CLI (ingest + build-dataset). Every browser
request must stay on 127.0.0.1 and the page must log no console errors. Screens are saved to
docs/screenshots/phase-11/.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.e2e.conftest import REPO, api

SHOTS = REPO / "docs" / "screenshots" / "phase-11"

pytestmark = pytest.mark.e2e


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


def test_training_config_and_preflight_in_the_ui(server: dict) -> None:
    from playwright.sync_api import expect, sync_playwright

    api(server, "PUT", "/api/settings", {"dev_mode": True})  # shows the tiny test model
    SHOTS.mkdir(parents=True, exist_ok=True)

    requests: list[str] = []
    console_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("request", lambda r: requests.append(r.url))
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.set_default_timeout(60_000)
        page.goto(server["url"])
        nav = page.get_by_role("navigation", name="Main")

        # Models: catalog cards with this machine's estimate
        nav.get_by_role("link", name="Models").click()
        tiny = page.get_by_role("article", name=re.compile("Tiny"))
        # The first catalog read in a fresh server takes a few seconds (it probes the machine).
        expect(tiny.get_by_text("test only", exact=True)).to_be_visible(timeout=60_000)
        expect(tiny.get_by_text(re.compile(r"needed of .* GB"))).to_be_visible()
        page.screenshot(path=str(SHOTS / "models.png"), full_page=True)

        # Training config: beginner view, then save
        nav.get_by_role("link", name="Training config").click()
        expect(page.get_by_text(re.compile(r"Auto → hf_peft on cpu"))).to_be_visible()
        page.get_by_label("Name", exact=True).fill("e2e-quick")
        page.get_by_label("Model").select_option("tiny-qwen2")
        page.get_by_label("Preset").select_option("quick")
        expect(page.get_by_label("Max epochs")).to_have_attribute("placeholder", "1")
        page.get_by_text("Advanced settings").click()
        expect(page.get_by_label("Learning rate")).to_have_attribute("placeholder", "0.0001")
        page.get_by_label("Learning rate").fill("5")
        expect(page.get_by_text("Use a value above 0 and below 1.")).to_be_visible()
        page.get_by_role("button", name="Reset Learning rate").click()
        page.get_by_label("Validate every N updates").fill("2")
        page.screenshot(path=str(SHOTS / "training-config.png"), full_page=True)
        page.get_by_role("button", name="Save", exact=True).click()
        expect(page.get_by_text("Saved e2e-quick v1.")).to_be_visible()

        # Duplicate it, then edit the copy into an oversized config (a new version)
        page.get_by_role("button", name="Duplicate e2e-quick").click()
        page.get_by_label("Name of the copy of e2e-quick").fill("e2e-big")
        page.get_by_role("button", name="Make the copy").click()
        expect(page.get_by_text("Duplicated e2e-quick as e2e-big v1.")).to_be_visible()
        page.get_by_role("button", name="Edit e2e-big").click()
        expect(page.get_by_role("region", name="Edit e2e-big (v1)")).to_be_visible()
        page.get_by_label("Model").select_option("qwen3-1.7b")
        page.get_by_role("button", name="Save and preflight").click()

        # Preflight of the oversized config: a failing check with its fix, Start disabled
        expect(page.get_by_role("heading", name="Preflight", exact=True)).to_be_visible()
        checks = page.get_by_role("list", name="Preflight checks")
        expect(checks.get_by_text(re.compile("above the CPU training limit"))).to_be_visible(
            timeout=120_000
        )
        expect(
            checks.get_by_text(re.compile(r"What to do: Use a model of about 1B"))
        ).to_be_visible()
        expect(page.get_by_role("button", name="Start training")).to_be_disabled()
        page.screenshot(path=str(SHOTS / "preflight-fail.png"), full_page=True)

        # Preflight of the tiny config: real results, nothing fails, Start enabled (not clicked)
        page.get_by_label("Training config").select_option("e2e-quick")
        expect(page.get_by_text(re.compile(r"^Ready to train"))).to_be_visible(timeout=120_000)
        expect(
            checks.get_by_text(re.compile(r"Chat-template prefix test on \d+ training"))
        ).to_be_visible()
        expect(page.get_by_role("button", name="Start training")).to_be_enabled()
        page.screenshot(path=str(SHOTS / "preflight-pass.png"), full_page=True)

        # Dashboard cards for the new screens
        nav.get_by_role("link", name="Dashboard").click()
        configs_card = page.get_by_role("region", name="Training configs")
        expect(configs_card.get_by_text("e2e-big v2")).to_be_visible()
        expect(
            page.get_by_role("region", name="Models").get_by_text("In the catalog")
        ).to_be_visible()
        page.get_by_role("button", name="Dark theme").click()
        page.screenshot(path=str(SHOTS / "dashboard-dark.png"), full_page=True)
        browser.close()

    hosts = {urlparse(u).hostname for u in requests if not u.startswith(("data:", "blob:"))}
    assert hosts == {"127.0.0.1"}, f"requests left 127.0.0.1: {hosts}"
    assert console_errors == [], console_errors
    saved = {c["name"]: c for c in api(server, "GET", "/api/configs")}
    assert saved["e2e-quick"]["version"] == 1 and saved["e2e-big"]["version"] == 2
    assert api(server, "GET", "/api/configs/e2e-quick")["config"] == {
        "kind": "train",
        "name": "e2e-quick",
        "model": {"catalog_id": "tiny-qwen2"},
        "preset": "quick",
        "training": {"eval_every_steps": 2},
    }
