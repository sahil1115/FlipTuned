"""Inference prompts (SPEC §3.1 steps 1–2): the fixed system prompt plus the conversation.

Nothing else goes in: no retrieved text, no dataset records. The system prompt must be
byte-identical to the one the model was trained with (SPEC §10.1, §11.4 parity test).
"""

from __future__ import annotations

Messages = list[dict[str, str]]


def chat_messages(system_prompt: str, turns: Messages) -> Messages:
    """``turns`` alternate user/assistant and end with a user turn."""
    if not turns or turns[-1].get("role") != "user":
        raise ValueError("a conversation must end with a user message")
    return [{"role": "system", "content": system_prompt}, *turns]
