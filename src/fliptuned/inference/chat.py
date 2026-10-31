"""The chat path (SPEC §3.1): fixed system prompt + the user's conversation → the model's reply.

Nothing else goes in. The conversation comes from the user only: alternating user and assistant
turns ending with a user turn. A "system" turn from the client is refused, so the fixed system
prompt (byte-identical to training) can't be replaced. Error messages never quote the text.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fliptuned.inference.prompting import Messages, chat_messages
from fliptuned.training.backends.base import GenerationConfig, LoadedModel, TrainingBackend

MAX_TURNS = 200
MAX_TURN_CHARS = 100_000


def validate_turns(turns: Any) -> Messages:
    """A clean copy of ``turns``; raises ValueError (without quoting content) if malformed."""
    if not isinstance(turns, list) or not turns:
        raise ValueError("a conversation needs at least one message")
    if len(turns) > MAX_TURNS:
        raise ValueError(f"a conversation can have at most {MAX_TURNS} messages")
    clean: Messages = []
    for number, turn in enumerate(turns):
        expected = "user" if number % 2 == 0 else "assistant"
        if not isinstance(turn, dict) or turn.get("role") != expected:
            raise ValueError(f"message {number + 1} must have role '{expected}'")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {number + 1} has no text")
        if len(content) > MAX_TURN_CHARS:
            raise ValueError(f"message {number + 1} is longer than {MAX_TURN_CHARS} characters")
        clean.append({"role": expected, "content": content})
    if clean[-1]["role"] != "user":
        raise ValueError("a conversation must end with a user message")
    return clean


def reply_stream(
    backend: TrainingBackend,
    loaded: LoadedModel,
    system_prompt: str,
    turns: Messages,
    config: GenerationConfig,
) -> Iterator[str]:
    """Stream the reply: SPEC §3.1 steps 2–4 (template, generate, return)."""
    yield from backend.stream(loaded, chat_messages(system_prompt, validate_turns(turns)), config)
