"""E2E (markers ``e2e`` + ``tiny``): the whole workflow in the UI, never touching the CLI (SPEC §18).

select fixture folder → scan → ingest → approve → build → choose a model → save config →
preflight → train → live loss visible → evaluate with in-UI blind review → verdict → chat →
run listed in history (then archive and restore it) → settings.

The tiny model runs on the CPU. Every browser request must stay on 127.0.0.1 and the page must log
no console errors. Screens are saved to docs/screenshots/phase-13/.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests.e2e.conftest import REPO, api

SHOTS = REPO / "docs" / "screenshots" / "phase-13"

pytestmark = [pytest.mark.e2e, pytest.mark.tiny]


def test_full_workflow_in_the_ui(server: dict, tmp_path: Path) -> None:
    from playwright.sync_api import expect, sync_playwright

    data = tmp_path / "qa-data"
    data.mkdir()
    shutil.copyfile(REPO / "tests" / "fixtures" / "qa_synthetic.csv", data / "qa_synthetic.csv")
    SHOTS.mkdir(parents=True, exist_ok=True)

    requests: list[str] = []
    console_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("request", lambda r: requests.append(r.url))
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.set_default_timeout(60_000)
        page.goto(server["url"])
        nav = page.get_by_role("navigation", name="Main")

        # Settings: developer mode shows the tiny test model
        nav.get_by_role("link", name="Settings").click()
        page.get_by_label(re.compile("Developer mode")).check()
        page.get_by_role("button", name="Save", exact=True).click()
        expect(page.get_by_text("Saved.")).to_be_visible()

        # Data: allow the fixture folder, scan, ingest, approve, build
        nav.get_by_role("link", name="Data sources").click()
        page.get_by_label("Add a folder").fill(str(data))
        page.get_by_role("button", name="Add folder").click()
        page.get_by_role("region", name="Browse").get_by_role(
            "button", name=re.compile(re.escape(data.name))
        ).click()
        page.get_by_role("button", name="Scan this folder").click()
        page.get_by_role("button", name="Ingest 1 file").click()
        expect(page.get_by_text("pending_review: 9")).to_be_visible(timeout=120_000)
        nav.get_by_role("link", name="Review").click()
        page.get_by_label("Select all on this page").check()
        page.get_by_role("button", name="Approve 9").click()
        expect(page.get_by_text("No records match these filters.")).to_be_visible()
        nav.get_by_role("link", name="Dataset").click()
        page.get_by_label("Name").fill("e2e")
        page.get_by_label("Model").select_option("tiny-qwen2")
        page.get_by_label(re.compile("Download the tokenizer")).check()
        page.get_by_role("button", name="Build dataset").click()
        expect(page.get_by_text(re.compile(r"Built e2e"))).to_be_visible(timeout=180_000)

        # Model and config: the tiny model, quick preset, saved, then preflight
        nav.get_by_role("link", name="Training config").click()
        page.get_by_label("Name", exact=True).fill("e2e-full")
        page.get_by_label("Model").select_option("tiny-qwen2")
        page.get_by_label("Preset").select_option("quick")
        page.get_by_text("Advanced settings").click()
        page.get_by_label("Validate every N updates").fill("4")
        page.get_by_role("button", name="Save and preflight").click()
        page.get_by_label("Download model files if they're missing").check()
        expect(page.get_by_text(re.compile(r"^Ready to train"))).to_be_visible(timeout=300_000)

        # Train: the monitor shows live loss, then the run finishes
        page.get_by_role("button", name="Start training").click()
        expect(page.get_by_role("heading", name="Run monitor", exact=True)).to_be_visible()
        run_id = re.search(r"run=(run_\w+)", page.url).group(1)
        log = page.get_by_role("log", name="Event log")
        expect(log.get_by_text(re.compile(r"train loss"))).not_to_have_count(0, timeout=300_000)
        expect(page.get_by_role("img", name=re.compile(r"^Loss chart"))).to_be_visible()
        expect(page.get_by_text("Training finished.", exact=False)).to_be_visible(timeout=600_000)
        page.get_by_role("link", name="evaluate the run").click()

        # Evaluate: a quick evaluation (4 test items), reviewed blind with the keyboard
        expect(page.get_by_label("Training run")).to_have_value(run_id)
        page.get_by_label("Only the first N test items (optional)").fill("4")
        page.get_by_role("button", name="Start an evaluation").click()
        review = page.get_by_role("region", name="Blind review")
        expect(review).to_be_visible(timeout=900_000)
        for key in ("1", "2", "3", "1"):
            expect(review.get_by_role("region", name="Answer A")).to_be_visible()
            before = review.get_by_role("status").text_content()
            if key == "1" and before and before.startswith("0 of"):
                page.screenshot(path=str(SHOTS / "evaluation-review.png"), full_page=True)
            page.keyboard.press(key)
            expect(review.get_by_role("status")).not_to_have_text(before or "")
        expect(review.get_by_text("Every pair is reviewed.", exact=False)).to_be_visible()
        results = page.get_by_role("region", name="Results")
        expect(results.get_by_text("inconclusive").first).to_be_visible()
        expect(results.get_by_role("table", name="Verdict criteria")).to_be_visible()
        expect(results.get_by_text("4 (100%)")).to_be_visible()
        page.screenshot(path=str(SHOTS / "evaluation-results.png"), full_page=True)

        # Chat with the fine-tuned model
        nav.get_by_role("link", name="Chat").click()
        expect(page.get_by_role("note")).to_contain_text("no document retrieval")
        page.get_by_label("Training run").select_option(run_id)
        page.get_by_role("button", name="Load the model and chat").click()
        message = page.get_by_label("Message")
        expect(message).to_be_enabled(timeout=120_000)
        message.fill("How do I validate input in Python?")
        message.press("Enter")
        conversation = page.get_by_role("list", name="Conversation")
        expect(conversation.get_by_text("Model", exact=True)).to_be_visible()
        expect(page.get_by_role("button", name="Send")).to_be_visible(timeout=120_000)
        expect(conversation.get_by_text("Thinking…")).to_have_count(0, timeout=120_000)
        page.screenshot(path=str(SHOTS / "chat.png"), full_page=True)
        page.get_by_role("button", name="Close and release the model").click()
        expect(page.get_by_role("button", name="Load the model and chat")).to_be_visible()

        # History: the run is listed with its verdict; archive and restore it
        nav.get_by_role("link", name="Runs").click()
        runs = page.get_by_role("table", name="Runs")
        row = runs.get_by_role("row").filter(has_text=run_id)
        expect(row.get_by_text("inconclusive")).to_be_visible()
        row.get_by_role("link", name=run_id).click()
        expect(page.get_by_role("button", name="Download metadata.json")).to_be_visible()
        page.screenshot(path=str(SHOTS / "runs.png"), full_page=True)
        page.get_by_role("button", name="Archive", exact=True).click()
        expect(runs.get_by_role("row").filter(has_text=run_id)).to_have_count(0)
        page.get_by_label("Show archived runs").check()
        expect(runs.get_by_text("archived")).to_be_visible()
        page.get_by_role("button", name="Restore").click()
        page.get_by_label("Show archived runs").uncheck()
        expect(runs.get_by_role("row").filter(has_text=run_id)).to_have_count(1)

        # Settings and the dashboard
        nav.get_by_role("link", name="Settings").click()
        expect(page.get_by_role("region", name="Install suggestions")).to_be_visible(timeout=60_000)
        page.screenshot(path=str(SHOTS / "settings.png"), full_page=True)
        nav.get_by_role("link", name="Dashboard").click()
        expect(
            page.get_by_role("region", name="Training runs").get_by_text("verdict: inconclusive")
        ).to_be_visible()
        page.get_by_role("button", name="Dark theme").click()
        page.screenshot(path=str(SHOTS / "dashboard-dark.png"), full_page=True)
        browser.close()

    hosts = {urlparse(u).hostname for u in requests if not u.startswith(("data:", "blob:"))}
    assert hosts == {"127.0.0.1"}, f"requests left 127.0.0.1: {hosts}"
    assert console_errors == [], console_errors
    history = api(server, "GET", "/api/history")
    assert [r["id"] for r in history] == [run_id]
    assert history[0]["verdict"]["verdict"] == "inconclusive"
