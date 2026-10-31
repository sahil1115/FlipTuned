"""Data sources for the UI (SPEC §13.4, §14.2): folder browsing inside the allowed folders,
uploads, and temporary files for uploaded CSVs (review and evaluation sheets).

Browsing resolves every path (following symlinks and junctions) and requires it to stay inside an
allowed folder; device, UNC/network and ``..`` paths are refused before touching the filesystem.
Uploads accept only the v1 input types, up to the upload size limit, saved under
``<app_home>/uploads/<source_id>/``; existing files are replaced only with ``overwrite``.
"""

from __future__ import annotations

import contextlib
import secrets
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from fliptuned.config.loader import format_validation_error
from fliptuned.config.models import IngestConfig
from fliptuned.errors import ConfigError, FlipTunedError, PathNotAllowedError
from fliptuned.platform.paths import display_path, resolve_within_roots, unsafe_reason
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services.scan import ScanRequest, ScanResult, scan_folder
from fliptuned.services.settings import allowed_roots, get_settings, open_home

UPLOAD_TYPES = (".csv", ".xlsx", ".md", ".txt")


class BrowseRequest(BaseModel):
    home: str | None = None
    path: str | None = Field(
        default=None, description="Folder to list. Empty = the allowed folders."
    )


class BrowseEntry(BaseModel):
    name: str
    path: str
    kind: str = Field(description="'folder' or 'file'.")
    size: int | None = None
    supported: bool = Field(description="A file type FlipTuned can read.")


class BrowseResult(BaseModel):
    path: str | None
    parent: str | None = Field(description="Parent folder, if it's also inside an allowed folder.")
    entries: list[BrowseEntry]


def browse(request: BrowseRequest) -> BrowseResult:
    roots = allowed_roots(request.home)
    if not request.path:
        entries = [
            BrowseEntry(name=display_path(r), path=display_path(r), kind="folder", supported=False)
            for r in roots
            if r.is_dir()
        ]
        return BrowseResult(path=None, parent=None, entries=entries)
    folder = resolve_within_roots(request.path, roots)
    if not folder.is_dir():
        raise PathNotAllowedError("That's a file, not a folder.")
    entries = []
    for child in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            resolved = resolve_within_roots(child, roots)  # skips links that leave the roots
        except FlipTunedError:
            continue
        if resolved.is_dir():
            entries.append(
                BrowseEntry(
                    name=child.name, path=display_path(child), kind="folder", supported=False
                )
            )
        elif resolved.is_file():
            entries.append(
                BrowseEntry(
                    name=child.name,
                    path=display_path(child),
                    kind="file",
                    size=resolved.stat().st_size,
                    supported=child.suffix.lower() in UPLOAD_TYPES,
                )
            )
    parent = None
    with contextlib.suppress(FlipTunedError):
        if folder.parent != folder and folder not in [r.resolve() for r in roots]:
            parent = display_path(resolve_within_roots(folder.parent, roots))
    return BrowseResult(path=display_path(folder), parent=parent, entries=entries)


# --------------------------------------------------------------------------- uploads


class UploadRequest(BaseModel):
    home: str | None = None
    source: str = Field(min_length=1, max_length=100, description="Data source name.")
    filename: str = Field(min_length=1, max_length=255)
    overwrite: bool = False


class UploadResult(BaseModel):
    source_id: str
    folder: str = Field(description="Upload folder of this source (scan and ingest it).")
    file: str
    bytes: int


def upload_limit_bytes(home: str | None) -> int:
    return get_settings(home).upload_max_mb * 1024 * 1024


def _safe_name(filename: str) -> str:
    name = filename.strip()
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ConfigError("The file name can't contain folders.")
    reason = unsafe_reason(name)
    if reason:
        raise ConfigError(f"That file name isn't allowed: {reason}.")
    if Path(name).suffix.lower() not in UPLOAD_TYPES:
        raise ConfigError(
            f"Unsupported file type: {Path(name).suffix or '(none)'}.",
            hint=f"Upload {', '.join(UPLOAD_TYPES)} files.",
        )
    return name


def save_upload(request: UploadRequest, data: bytes) -> UploadResult:
    name = _safe_name(request.filename)
    limit = upload_limit_bytes(request.home)
    if len(data) > limit:
        raise ConfigError(
            f"The file is larger than the upload limit ({limit // (1024 * 1024)} MB).",
            hint="Raise the limit in Settings, or put the file in an allowed folder instead.",
        )
    paths = open_home(request.home)
    source_id = repositories.data_source_id(request.source)
    folder = paths.uploads / source_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists() and not request.overwrite:
        raise FlipTunedError(
            f"{name} was already uploaded.", hint="Upload it again with overwrite."
        )
    tmp = folder / f".{secrets.token_hex(4)}.part"
    tmp.write_bytes(data)
    tmp.replace(target)
    return UploadResult(
        source_id=source_id, folder=display_path(folder), file=display_path(target), bytes=len(data)
    )


class SourceInfo(BaseModel):
    id: str
    name: str
    kind: str
    root_path: str
    status: str
    updated_at: str


def list_sources(home: str | None) -> list[SourceInfo]:
    paths = open_home(home)
    with connect(paths.db) as conn:
        rows = repositories.list_data_sources(conn)
    return [
        SourceInfo(
            id=r["id"],
            name=r["name"],
            kind=r["kind"],
            root_path=r["root_path"],
            status=r["status"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@contextlib.contextmanager
def temp_file(home: str | None, data: bytes, suffix: str = ".csv") -> Iterator[str]:
    """An uploaded file (e.g. a filled-in review sheet) on disk for the length of one call."""
    if len(data) > upload_limit_bytes(home):
        raise ConfigError("The file is larger than the upload limit.")
    paths = open_home(home)
    path = paths.uploads / f".incoming-{secrets.token_hex(8)}{suffix}"
    path.write_bytes(data)
    try:
        yield str(path)
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- allowed-folder checks


def scan_allowed(request: ScanRequest) -> ScanResult:
    """``scan`` for the UI: only folders inside the allowed folders."""
    folder = resolve_within_roots(request.path, allowed_roots(request.home))
    return scan_folder(request.model_copy(update={"path": str(folder)}))


def check_ingest_config(home: str | None, data: dict) -> None:
    """An ingest config from the UI may only read absolute paths inside the allowed folders."""
    try:
        config = IngestConfig.model_validate({"kind": "ingest", **data})
    except ValidationError as exc:
        raise ConfigError(
            "The ingest config is not valid.", details=format_validation_error(exc)
        ) from None
    roots = allowed_roots(home)
    for spec in config.sources:
        if not Path(spec.path).is_absolute():
            raise PathNotAllowedError("Source paths must be absolute (pick them in the browser).")
        resolve_within_roots(spec.path, roots)
