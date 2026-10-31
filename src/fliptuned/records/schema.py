"""Normalized record schema v1.0 (SPEC §8.3)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from fliptuned.config.models import RecordStatus

SCHEMA_VERSION = "1.0"


class RecordType(StrEnum):
    QA_PAIR = "qa_pair"
    SOURCE_BLOCK = "source_block"
    CANDIDATE_QA = "candidate_qa"


class ExtractionMethod(StrEnum):
    CSV_COLUMN_MAPPING = "csv_column_mapping"
    XLSX_COLUMN_MAPPING = "xlsx_column_mapping"
    MARKDOWN_DELIMITERS = "markdown_delimiters"
    TEXT_DELIMITERS = "text_delimiters"
    MARKDOWN_BLOCK = "markdown_block"
    TEXT_BLOCK = "text_block"


class GroupSource(StrEnum):
    EXPLICIT_COLUMN = "explicit_column"
    THREAD = "thread"
    DUPLICATE_CLUSTER = "duplicate_cluster"
    SINGLETON = "singleton"
    SOURCE_FILE = "source_file"


class QualityFlag(StrEnum):
    EMPTY_QUESTION = "empty_question"
    EMPTY_ANSWER = "empty_answer"
    DUPLICATE_ID = "duplicate_id"
    EXACT_DUPLICATE = "exact_duplicate"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    BROKEN_CODE_FENCE = "broken_code_fence"
    LIKELY_QA_SWAP = "likely_qa_swap"
    INVALID_UTF8 = "invalid_utf8"
    MALFORMED_ROW = "malformed_row"
    UNKNOWN_STATUS = "unknown_status"


# Flags that make a record unusable for training until fixed.
QUARANTINE_FLAGS = frozenset(
    {
        QualityFlag.EMPTY_QUESTION,
        QualityFlag.EMPTY_ANSWER,
        QualityFlag.INVALID_UTF8,
        QualityFlag.MALFORMED_ROW,
    }
)


class SourceLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet: str | None = None
    row: int | None = Field(default=None, description="Spreadsheet/CSV row, 1-based, header = 1.")
    line: int | None = Field(default=None, description="Markdown/TXT line, 1-based.")
    page: int | None = None
    heading: str | None = None


class PrivacyFinding(BaseModel):
    """Where a likely secret is. Never holds the value itself."""

    model_config = ConfigDict(extra="forbid")

    type: str
    field: str  # "question" | "answer"
    start: int
    end: int

    @property
    def redacted(self) -> str:
        return f"[REDACTED {self.type}, {self.end - self.start} chars]"


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    record_type: RecordType
    source_file: str = Field(description="POSIX-style path relative to the data source root.")
    source_locator: SourceLocator
    question: str | None = None
    answer: str | None = None
    text: str | None = Field(default=None, description="Body of a source_block record.")
    external_id: str | None = Field(default=None, description="The ID from the user's id column.")
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    status: RecordStatus
    origin: str = "human_provided"
    extraction_method: ExtractionMethod
    content_hash: str
    group_id: str | None = None
    group_source: GroupSource | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    privacy_flags: list[PrivacyFinding] = Field(default_factory=list)
