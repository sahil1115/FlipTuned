"""``manifest.json`` and ``index.jsonl`` of a built dataset.

The manifest holds statistics and settings, never record text. It has no timestamps, so the same
inputs give a byte-identical manifest on every OS (the build time lives in the registry).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fliptuned.dataset.builder import BuildWarning, TokenStats

MANIFEST_VERSION = "1.0"
DATASET_FORMAT = "canonical_messages_v1"
PARTITION_FILES = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}
INDEX_FILE = "index.jsonl"
EXCLUSIONS_FILE = "exclusions.json"
DUPLICATES_FILE = "duplicates.json"
MANIFEST_FILE = "manifest.json"


class IndexRow(BaseModel):
    """One line per example, aligned with the partition files. IDs and metadata only."""

    partition: str
    line: int = Field(description="1-based line in the partition file.")
    source_id: str
    record_id: str
    category: str | None
    group_key: str
    group_source: str
    tokens: int
    truncated: bool


class FileInfo(BaseModel):
    sha256: str
    lines: int | None = None


class SplitInfo(BaseModel):
    target: dict[str, float]
    actual: dict[str, float]
    tolerance_pts: float
    groups: dict[str, int]


class GroupInfo(BaseModel):
    total: int
    by_source: dict[str, int] = Field(description="Records per group_source.")
    largest_size: int
    largest_share: float


class CountInfo(BaseModel):
    input_records: int
    included: int
    partitions: dict[str, int]
    excluded: dict[str, int]
    duplicates: int
    truncated: int


class DatasetManifest(BaseModel):
    manifest_version: str = MANIFEST_VERSION
    format: str = DATASET_FORMAT
    dataset_id: str
    name: str
    fingerprint: str
    builder_version: str
    model: dict[str, Any] = Field(description="Catalog ID and the tokenizer used to count tokens.")
    system_prompt: str
    system_prompt_sha256: str
    seed: int
    max_seq_length: int
    truncate_long_records: bool
    split: SplitInfo
    counts: CountInfo
    groups: GroupInfo
    tokens: dict[str, TokenStats | None]
    categories: dict[str, dict[str, int]] = Field(
        description="Examples per category per partition."
    )
    warnings: list[BuildWarning]
    sources: dict[str, str] = Field(description="Source ID → sha256 of its records file.")
    config: dict[str, Any]
    files: dict[str, FileInfo]
