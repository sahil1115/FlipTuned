"""Chat-template checks shared by every backend (SPEC §11.4).

The prefix test: rendering the conversation without the answer, with the generation prompt, must
give an exact token prefix of the full conversation. If it doesn't, the loss mask would cut the
answer in the wrong place (or train template tokens), so training is blocked for that model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

Messages = list[dict[str, str]]

# Synthetic samples covering plain text, code fences and multi-line answers.
SAMPLE_CONVERSATIONS: list[Messages] = [
    [
        {"role": "system", "content": "You are a careful technical assistant."},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A."},
    ],
    [
        {"role": "system", "content": "You are a careful technical assistant."},
        {"role": "user", "content": "How do I parse JSON in Python?"},
        {
            "role": "assistant",
            "content": "Use `json.loads`.\n\n```python\nimport json\ndata = json.loads(text)\n```\n",
        },
    ],
    [
        {"role": "system", "content": "Follow the team's conventions."},
        {"role": "user", "content": "Line one\nline two:\n- a list item"},
        {
            "role": "assistant",
            "content": "First line.\r\nSecond line with a CRLF.\n\n| a | b |\n|---|---|",
        },
    ],
]


class PrefixTestResult(BaseModel):
    passed: bool
    samples: int
    failures: list[int]  # sample indexes that failed; never content


def _ids(tokenizer: Any, messages: Messages, kwargs: dict[str, Any], *, prompt: bool) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=prompt,
            return_dict=False,
            **kwargs,
        )
    )


def prompt_ids(tokenizer: Any, messages: Messages, kwargs: dict[str, Any]) -> list[int]:
    """Token IDs of ``messages`` followed by the generation prompt (what inference feeds in)."""
    return _ids(tokenizer, messages, kwargs, prompt=True)


def prefix_test(
    tokenizer: Any,
    kwargs: dict[str, Any],
    samples: Sequence[Messages] = SAMPLE_CONVERSATIONS,
) -> PrefixTestResult:
    failures = []
    for index, messages in enumerate(samples):
        full = _ids(tokenizer, messages, kwargs, prompt=False)
        prompt = _ids(tokenizer, messages[:-1], kwargs, prompt=True)
        if len(prompt) >= len(full) or full[: len(prompt)] != prompt:
            failures.append(index)
    return PrefixTestResult(passed=not failures, samples=len(samples), failures=failures)


def template_checksum(tokenizer: Any, kwargs: dict[str, Any]) -> str:
    """sha256 of the chat template text plus the template kwargs (recorded in run metadata)."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, dict):  # some tokenizers keep several named templates
        template = json.dumps(template, sort_keys=True)
    payload = template + "\x1f" + json.dumps(kwargs, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
