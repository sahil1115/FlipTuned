"""E2E (marker ``e2e``): the data flow in the real UI against a real ``fliptuned serve``.

pick the fixture folder → scan → ingest → approve → build → statistics visible

The server (tests/e2e/conftest.py) runs on a fresh app home with a copy of the synthetic fixture
CSV as its only data. Every browser request must stay on 127.0.0.1, and the page must log no
console errors (a CSP violation would). Screens are saved to docs/screenshots/phase-10/.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.e2e.conftest import REPO, api

SHOTS = REPO / "docs" / "screenshots" / "phase-10"

pytestmark = pytest.mark.e2e


def test_data_flow_in_the_ui(server: dict, tmp_path: Path) -> None:
    from playwright.sync_api import expect, sync_playwright

    data = tmp_path / "qa-data"
    data.mkdir()
    shutil.copyfile(REPO / "tests" / "fixtures" / "qa_synthetic.csv", data / "qa_synthetic.csv")
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

        # Dashboard
        page.goto(server["url"])
        expect(page.get_by_role("heading", name="Dashboard")).to_be_visible()
        # the machine probe (torch import) can take a while on a cold start
        expect(page.get_by_text("Auto → hf_peft")).to_be_visible(timeout=60_000)
        nav = page.get_by_role("navigation", name="Main")
        page.screenshot(path=str(SHOTS / "dashboard.png"), full_page=True)

        # Data sources: allow the folder, browse to it, scan, ingest
        nav.get_by_role("link", name="Data sources").click()
        page.get_by_label("Add a folder").fill(str(data))
        page.get_by_role("button", name="Add folder").click()
        browse = page.get_by_role("region", name="Browse")
        browse.get_by_role("button", name=re.compile(re.escape(data.name))).click()
        page.get_by_role("button", name="Scan this folder").click()
        expect(
            page.get_by_role("region", name=re.compile("Columns of qa_synthetic.csv"))
        ).to_be_visible()
        expect(page.locator("#qa-synthetic-csv-question")).to_have_value("question")
        page.get_by_role("button", name="Ingest 1 file").click()
        expect(page.get_by_text("By status:")).to_be_visible(timeout=120_000)
        expect(page.get_by_text("pending_review: 9")).to_be_visible()
        page.screenshot(path=str(SHOTS / "data-sources.png"), full_page=True)

        # Review: open a record, then approve every pending one
        nav.get_by_role("link", name="Review").click()
        expect(page.get_by_role("region", name="Records (9)")).to_be_visible()
        page.get_by_role("list", name="Records").get_by_role("button").first.click()
        expect(page.get_by_role("heading", name="Answer")).to_be_visible()
        page.screenshot(path=str(SHOTS / "review.png"), full_page=True)
        page.get_by_label("Select all on this page").check()
        page.get_by_role("button", name="Approve 9").click()
        expect(page.get_by_text("No records match these filters.")).to_be_visible()

        # Dataset: build with the tiny test model, then read the statistics
        nav.get_by_role("link", name="Dataset").click()
        page.get_by_label("Name").fill("e2e")
        page.get_by_label("Model").select_option("tiny-qwen2")
        page.get_by_label(re.compile("Download the tokenizer")).check()
        page.get_by_role("button", name="Build dataset").click()
        expect(page.get_by_text(re.compile(r"Built e2e"))).to_be_visible(timeout=180_000)
        page.get_by_role("button", name="Show its statistics").click()
        stats = page.get_by_role("region", name="Statistics: e2e")
        expect(stats.get_by_role("table")).to_be_visible()
        expect(stats.get_by_role("rowheader", name="train")).to_be_visible()
        expect(stats.get_by_text("Records considered")).to_be_visible()
        page.screenshot(path=str(SHOTS / "dataset.png"), full_page=True)

        # Dark theme renders too
        page.get_by_role("button", name="Dark theme").click()
        page.screenshot(path=str(SHOTS / "dataset-dark.png"), full_page=True)
        browser.close()

    hosts = {urlparse(u).hostname for u in requests if not u.startswith(("data:", "blob:"))}
    assert hosts == {"127.0.0.1"}, f"requests left 127.0.0.1: {hosts}"
    assert console_errors == [], console_errors
    datasets = api(server, "GET", "/api/datasets")
    assert [d["name"] for d in datasets] == ["e2e"]
