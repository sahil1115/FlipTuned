"""CSV and XLSX row readers. Cell text is returned exactly as stored."""

from __future__ import annotations

import csv
import datetime as dt
import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from fliptuned.ingestion.files import decode_bytes


@dataclass
class TableRow:
    row: int  # 1-based, header row = 1
    values: dict[str, str]
    sheet: str | None = None
    extra_fields: bool = False  # more cells than headers


@dataclass
class TableRead:
    headers: dict[str | None, list[str]] = field(default_factory=dict)  # sheet -> headers
    rows: list[TableRow] = field(default_factory=list)
    blank_rows: int = 0
    valid_encoding: bool = True
    errors: list[str] = field(default_factory=list)


def _duplicate_names(headers: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in headers:
        if name and name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def _duplicates_error(duplicates: list[str]) -> str:
    names = ", ".join(repr(d) for d in duplicates)
    return f"column name(s) {names} appear more than once; rename them so each is unique"


def read_csv(path: Path, encoding: str) -> TableRead:
    decoded = decode_bytes(path.read_bytes(), encoding)
    result = TableRead(valid_encoding=decoded.valid)
    reader = csv.reader(io.StringIO(decoded.text, newline=""), strict=True)
    try:
        header = next(reader, None)
    except csv.Error as exc:
        result.errors.append(f"could not read the header row: {exc}")
        return result
    if header is None:
        result.errors.append("the file is empty")
        return result
    headers = [h.strip() for h in header]
    result.headers[None] = headers
    duplicates = _duplicate_names(headers)
    if duplicates:
        result.errors.append(_duplicates_error(duplicates))
        return result
    row_number = 1
    try:
        for cells in reader:
            row_number += 1
            _add_row(result, headers, cells, row_number, None)
    except csv.Error as exc:
        result.errors.append(f"CSV parsing stopped at row {row_number + 1}: {exc}")
    return result


def read_xlsx(path: Path, sheet: str | None) -> TableRead:
    from zipfile import BadZipFile

    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    result = TableRead()
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError) as exc:
        result.errors.append(
            f"could not open the workbook ({type(exc).__name__}); it is not a valid .xlsx file"
        )
        return result
    try:
        names = workbook.sheetnames
        if sheet is not None and sheet not in names:
            result.errors.append(f"sheet '{sheet}' not found (sheets: {', '.join(names)})")
            return result
        for name in [sheet] if sheet else names:
            rows = workbook[name].iter_rows(values_only=True)
            header = next(rows, None)
            if header is None:
                continue
            headers = [_cell_text(h).strip() for h in header]
            result.headers[name] = headers
            duplicates = _duplicate_names(headers)
            if duplicates:
                result.errors.append(f"sheet '{name}': {_duplicates_error(duplicates)}")
                continue
            for row_number, values in enumerate(rows, start=2):
                _add_row(result, headers, [_cell_text(v) for v in values], row_number, name)
    finally:
        workbook.close()
    return result


def _add_row(
    result: TableRead, headers: list[str], cells: list[str], row: int, sheet: str | None
) -> None:
    if not any(c.strip() for c in cells):
        result.blank_rows += 1
        return
    extra = len(cells) > len(headers) and any(c.strip() for c in cells[len(headers) :])
    padded = cells + [""] * (len(headers) - len(cells))
    values = {h: padded[i] for i, h in enumerate(headers) if h}
    result.rows.append(TableRow(row=row, values=values, sheet=sheet, extra_fields=extra))


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    return str(value)


def iter_headers(read: TableRead) -> Iterator[tuple[str | None, list[str]]]:
    yield from read.headers.items()
