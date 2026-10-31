"""Hugging Face Hub file access: offline by default, downloads only when explicitly allowed.

Files are read from the local HF cache at the catalog's pinned revision. With ``allow_download``
the requested files are fetched first; network access is switched on for that one call and off
again afterwards, so ``HF_HUB_OFFLINE=1`` stays in force everywhere else (SPEC §3.2).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fliptuned.errors import NotAvailableError
from fliptuned.models.catalog import CatalogEntry, ModelSource
from fliptuned.utils.logging import get_logger

log = get_logger(__name__)

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "config.json",  # small; lets AutoTokenizer pick the tokenizer class
]
# Everything a model needs to load: weights, configs, tokenizer. Never Python files
# (no remote code is ever executed).
MODEL_FILES = [*TOKENIZER_FILES, "*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]


# Held while network access is on, so overlapping downloads in one process (the web server's
# threads) can't restore the offline flag in the wrong order and leave the hub online.
_NETWORK_LOCK = threading.Lock()


@contextmanager
def _network_allowed() -> Iterator[None]:
    from huggingface_hub import constants

    with _NETWORK_LOCK:
        previous = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = False
        try:
            yield
        finally:
            constants.HF_HUB_OFFLINE = previous


def snapshot(
    entry: CatalogEntry,
    source: ModelSource,
    patterns: list[str],
    *,
    allow_download: bool,
    what: str,
    size_hint: str,
) -> Path:
    """Local folder with ``patterns`` from ``source`` at its pinned revision."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        if allow_download:
            log.info("downloading %s for %s@%s", what, source.repo, source.revision)
            with _network_allowed():
                folder = huggingface_hub.snapshot_download(
                    source.repo, revision=source.revision, allow_patterns=patterns
                )
        else:
            folder = huggingface_hub.snapshot_download(
                source.repo,
                revision=source.revision,
                allow_patterns=patterns,
                local_files_only=True,
            )
    except LocalEntryNotFoundError:
        if not allow_download:
            raise NotAvailableError(
                f"The {what} for '{entry.id}' is not downloaded.",
                hint=f"Allow the download ({size_hint}): --allow-download, or the download option in the UI.",
            ) from None
        raise NotAvailableError(
            f"Could not download the {what} for '{entry.id}'.",
            hint="Check your internet connection and try again.",
        ) from None
    except OSError as exc:
        raise NotAvailableError(
            f"Could not download the {what} for '{entry.id}'.",
            hint="Check your internet connection and try again.",
            details=type(exc).__name__,
        ) from None
    return Path(folder)


def tokenizer_files(entry: CatalogEntry, *, allow_download: bool) -> Path:
    return snapshot(
        entry,
        entry.tokenizer_source,
        TOKENIZER_FILES,
        allow_download=allow_download,
        what="tokenizer",
        size_hint="tokenizer files only, a few MB",
    )


def model_source(entry: CatalogEntry, backend: str) -> ModelSource:
    source = entry.sources.get(backend)
    if source is None:
        raise NotAvailableError(
            f"Model '{entry.id}' has no {backend} version in the catalog.",
            hint="Pick another model, or another backend.",
        )
    return source


def model_files(entry: CatalogEntry, backend: str, *, allow_download: bool) -> Path:
    source = model_source(entry, backend)
    folder = snapshot(
        entry,
        source,
        MODEL_FILES,
        allow_download=allow_download,
        what="model",
        size_hint=f"about {source.download_gb:g} GB",
    )
    if not any(folder.glob("*.safetensors")):
        raise NotAvailableError(
            f"The model files for '{entry.id}' are incomplete (no weights found).",
            hint="Allow downloads to fetch them: --allow-download, or the download option in the UI.",
        )
    return folder


def is_cached(source: ModelSource) -> bool:
    """True if this revision's weights are in the local HF cache. Never touches the network."""
    from huggingface_hub import constants

    folder = (
        Path(constants.HF_HUB_CACHE)
        / f"models--{source.repo.replace('/', '--')}"
        / "snapshots"
        / source.revision
    )
    return folder.is_dir() and any(folder.glob("*.safetensors"))
