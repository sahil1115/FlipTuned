"""Structured check results shared by validation and preflight (SPEC §6)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Check(BaseModel):
    name: str
    status: CheckStatus
    message: str
    fix: str | None = None
