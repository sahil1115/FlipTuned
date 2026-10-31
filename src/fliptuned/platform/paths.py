"""Path helpers: native display, allowed-root checks, unsafe-path rejection, folder replacement.

The string checks take an explicit ``windows`` flag so Windows-style inputs can be tested on any OS.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from fliptuned.errors import PathNotAllowedError
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

_DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "//?/", "//./")
_RESERVED_WINDOWS_NAMES = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)


def display_path(path: Path) -> str:
    """The path as the user's OS would show it."""
    return str(path)


def unsafe_reason(raw: str, *, windows: bool | None = None) -> str | None:
    """Why ``raw`` must be rejected before touching the filesystem, or ``None`` if it looks safe."""
    windows = sys.platform == "win32" if windows is None else windows
    if not raw or "\x00" in raw:
        return "empty path or NUL byte"
    if windows:
        if raw.startswith(_DEVICE_PREFIXES):
            return "device or extended-length paths are not allowed"
        pure = PureWindowsPath(raw)
        if raw.startswith(("\\\\", "//")) or pure.drive.startswith("\\\\"):
            return "network (UNC) paths are not allowed"
        parts = pure.parts
        names = parts[1:] if pure.anchor else parts
        if any(_RESERVED_WINDOWS_NAMES.match(part) for part in names):
            return "reserved Windows device name"
    else:
        if raw.startswith("//"):
            return "network-style paths are not allowed"
        parts = PurePosixPath(raw).parts
    if ".." in parts:
        return "parent-directory ('..') segments are not allowed"
    return None


def resolve_within_roots(raw: str | os.PathLike[str], roots: Iterable[Path]) -> Path:
    """Resolve ``raw`` (following symlinks and junctions) and require it to lie inside a root."""
    text = os.fspath(raw)
    reason = unsafe_reason(text)
    if reason:
        raise PathNotAllowedError(f"That path isn't allowed: {reason}.", hint=_ROOTS_HINT)
    try:
        resolved = Path(text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise PathNotAllowedError(
            "That path doesn't exist or can't be read.", hint=_ROOTS_HINT
        ) from None
    for root in roots:
        try:
            root_resolved = root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return resolved
    raise PathNotAllowedError("That path is outside the allowed folders.", hint=_ROOTS_HINT)


_ROOTS_HINT = "Choose a folder inside one of the allowed folders (Settings → Allowed folders)."


# --------------------------------------------------------------------------- folder replacement

_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def _retry(action: Callable[[], None]) -> None:
    """Retry transient PermissionErrors. On Windows, indexers, antivirus and editor file
    watchers briefly hold handles that block deleting or renaming folders."""
    for delay in _RETRY_DELAYS:
        try:
            action()
            return
        except PermissionError:
            time.sleep(delay)
    action()


def remove_tree(path: Path) -> None:
    _retry(lambda: shutil.rmtree(path))


def replace_dir(src: Path, dst: Path) -> None:
    """Move folder ``src`` to ``dst``, replacing ``dst``. The old folder is renamed aside first,
    so a failure never leaves ``dst`` half-deleted; leftovers are named ``.old-<name>-<id>``."""
    if not dst.exists():
        _retry(lambda: os.replace(src, dst))
        return
    aside = dst.with_name(f".old-{dst.name}-{uuid.uuid4().hex[:8]}")
    _retry(lambda: os.replace(dst, aside))
    _retry(lambda: os.replace(src, dst))
    try:
        remove_tree(aside)
    except OSError:
        log.warning("could not remove old folder %s; delete it later", aside.name)
