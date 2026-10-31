"""API test fixtures: a fixture-data app home, a TestClient with the token, inline jobs.

Routes start jobs detached (a background worker process). Here ``LocalJobRunner.submit`` is
patched to queue them without a worker, and :func:`run_job` runs them in this process, so fake
backends stand in for models and nothing is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fliptuned.api.app import create_app
from fliptuned.jobs.runner import LocalJobRunner
from fliptuned.services import jobs as jobs_service
from fliptuned.services.datasets import BuildDatasetRequest, build_dataset
from fliptuned.services.settings import SettingsUpdate, update_settings
from tests.helpers.pipeline import FIXTURES, SAMPLE_DATASET, ingest_samples, use_fake_tokenizer

TOKEN = "test-token-123"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    update_settings(SettingsUpdate(home=str(home), allowed_roots=[str(FIXTURES)]))
    return home


@pytest.fixture
def inline_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    real_submit = LocalJobRunner.submit

    def submit(self, job_type, params, *, detach, **kw):
        return real_submit(self, job_type, params, detach=False, **kw)

    monkeypatch.setattr(LocalJobRunner, "submit", submit)


def run_job(home: Path, job_id: str) -> str:
    return jobs_service.run_inline(str(home), job_id).status


@pytest.fixture
def client(home: Path, inline_jobs) -> TestClient:
    app = create_app(home=str(home), token=TOKEN, port=PORT)
    with TestClient(app, base_url=BASE, headers={"X-FlipTuned-Token": TOKEN}) as c:
        yield c


@pytest.fixture
def anon(home: Path) -> TestClient:
    """A client without the token."""
    app = create_app(home=str(home), token=TOKEN, port=PORT)
    with TestClient(app, base_url=BASE) as c:
        yield c
