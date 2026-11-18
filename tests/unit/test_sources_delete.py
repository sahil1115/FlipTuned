from __future__ import annotations

from pathlib import Path

import pytest

from fliptuned.config.app_home import AppPaths
from fliptuned.errors import ConfigError, NotAvailableError
from fliptuned.services.review import (
    DecideRequest,
    ListRecordsRequest,
    RecordRef,
    ReviewFilter,
    decide,
    list_records,
)
from fliptuned.services.sources import DeleteSourceRequest, delete_source, list_sources
from tests.helpers.pipeline import ingest_samples


def _records(home: Path, **filters):
    request = ListRecordsRequest(home=str(home), filter=ReviewFilter(**filters), limit=1000)
    return list_records(request).items


def test_delete_source_removes_records_decisions_and_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ingest_samples(home)
    pending = _records(home, statuses=["pending_review"])[0]
    target = next(s for s in list_sources(str(home)) if s.id == pending.source_id)
    ref = RecordRef(source_id=pending.source_id, record_id=pending.record_id)
    assert decide(DecideRequest(home=str(home), records=[ref], status="rejected")).applied == 1

    with pytest.raises(ConfigError, match="nothing was deleted"):
        delete_source(DeleteSourceRequest(home=str(home), source=target.name, confirm="no"))
    assert (AppPaths(home).sources / target.id).exists()

    result = delete_source(
        DeleteSourceRequest(home=str(home), source=target.name, confirm=target.name)
    )
    assert result.source_id == target.id and result.decisions_removed == 1
    assert result.removed_folders and "Datasets already built" in result.note
    assert not (AppPaths(home).sources / target.id).exists()
    assert target.id not in {s.id for s in list_sources(str(home))}
    remaining = _records(home)
    assert remaining and all(i.source_id != target.id for i in remaining)

    with pytest.raises(NotAvailableError):
        delete_source(DeleteSourceRequest(home=str(home), source=target.name, confirm=target.name))
