"""CSV cells that spreadsheet apps open safely (review sheets come from third-party content)."""

from __future__ import annotations

FORMULA_START = ("=", "+", "@")


def spreadsheet_cell(value: str | None) -> str:
    """Text a spreadsheet won't run as a formula."""
    value = value or ""
    return "'" + value if value.startswith(FORMULA_START) else value
