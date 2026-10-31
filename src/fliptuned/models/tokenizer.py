"""Tokenizer access for sequence-length checks (offline by default; see ``models.hub``)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from fliptuned.errors import NotAvailableError
from fliptuned.models.catalog import CatalogEntry
from fliptuned.models.hub import tokenizer_files


class TokenCounter(Protocol):
    """Counts the tokens of a full chat example, exactly as training will see it."""

    identity: dict[str, Any]

    def count(self, messages: list[dict[str, str]]) -> int: ...


@dataclass
class HfTokenCounter:
    tokenizer: Any
    template_kwargs: dict[str, Any]
    identity: dict[str, Any]

    def count(self, messages: list[dict[str, str]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, **self.template_kwargs
        )
        return len(encoded["input_ids"])


# Transformers' lazy module isn't safe to import from two threads at once: the web server runs
# requests in threads, and two first-time preflights failed with "cannot import name
# 'AutoTokenizer'" and hung. Tokenizer loads in one process run one at a time.
_LOAD_LOCK = threading.Lock()


def load_tokenizer(entry: CatalogEntry, *, allow_download: bool = False) -> Any:
    with _LOAD_LOCK:
        return _load_tokenizer(entry, allow_download)


def _load_tokenizer(entry: CatalogEntry, allow_download: bool) -> Any:
    from transformers import AutoTokenizer

    folder = tokenizer_files(entry, allow_download=allow_download)
    try:
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise NotAvailableError(
            f"The tokenizer files for '{entry.id}' could not be loaded.",
            hint="Allow downloads to fetch them again: --allow-download, or the download option in the UI.",
            details=type(exc).__name__,
        ) from None
    if not getattr(tokenizer, "chat_template", None):
        raise NotAvailableError(
            f"The tokenizer for '{entry.id}' has no chat template.",
            hint="Pick another model from the catalog.",
        )
    return tokenizer


def load_token_counter(entry: CatalogEntry, *, allow_download: bool = False) -> HfTokenCounter:
    source = entry.tokenizer_source
    return HfTokenCounter(
        tokenizer=load_tokenizer(entry, allow_download=allow_download),
        template_kwargs=dict(entry.chat_template_kwargs),
        identity={
            "catalog_id": entry.id,
            "tokenizer_repo": source.repo,
            "tokenizer_revision": source.revision,
            "chat_template_kwargs": dict(entry.chat_template_kwargs),
        },
    )
