"""Markdown/TXT Q&A extraction with configurable delimiters.

Rules:
- a line starting with the question prefix (column 0, outside a code fence) starts a pair;
- a line starting with the answer prefix ends the question and starts the answer;
- the answer runs until the next question, a Markdown heading, or the end of the file;
- code fences (``` and ~~~) are tracked so delimiter lines inside them are plain text;
- everything outside pairs becomes a block (a ``source_block`` record, which never trains alone).

Text is sliced from the original lines (line endings included), so code, lists and tables
are preserved exactly. Only leading/trailing blank lines and the space after a prefix are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)")
_HEADING = re.compile(r"^#{1,6}[ \t]+(.*?)[ \t#]*$")


@dataclass
class TextItem:
    kind: str  # "qa" | "block"
    line: int
    heading: str | None
    question_lines: list[str] = field(default_factory=list)
    answer_lines: list[str] | None = None
    block_lines: list[str] = field(default_factory=list)

    @property
    def question(self) -> str:
        return _trim(self.question_lines)

    @property
    def answer(self) -> str:
        return _trim(self.answer_lines or [])

    @property
    def text(self) -> str:
        return _trim(self.block_lines)


def parse_text_qa(
    text: str, question_prefix: str, answer_prefix: str, *, markdown: bool
) -> list[TextItem]:
    items: list[TextItem] = []
    current: TextItem | None = None
    block: TextItem | None = None
    heading: str | None = None
    fence: str | None = None

    def close_block() -> None:
        nonlocal block
        if block is not None and any(
            line.strip() and not (markdown and _HEADING.match(line.rstrip("\r\n")))
            for line in block.block_lines
        ):
            items.append(block)  # blocks holding only headings are structure, not content
        block = None

    def close_current() -> None:
        nonlocal current
        if current is not None:
            items.append(current)
        current = None

    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        bare = line.rstrip("\r\n")
        fence_match = _FENCE.match(bare)
        if fence is None:
            heading_match = _HEADING.match(bare) if markdown else None
            if bare.startswith(question_prefix):
                close_current()
                close_block()
                current = TextItem("qa", number, heading, [_after_prefix(line, question_prefix)])
                continue
            if (
                current is not None
                and current.answer_lines is None
                and bare.startswith(answer_prefix)
            ):
                current.answer_lines = [_after_prefix(line, answer_prefix)]
                continue
            if heading_match:
                close_current()
                heading = heading_match.group(1) or None
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        if current is not None:
            (
                current.question_lines if current.answer_lines is None else current.answer_lines
            ).append(line)
        else:
            if block is None:
                block = TextItem("block", number, heading)
            block.block_lines.append(line)

    close_current()
    close_block()
    return items


def _after_prefix(line: str, prefix: str) -> str:
    rest = line[len(prefix) :]
    return rest[1:] if rest.startswith((" ", "\t")) else rest


def _trim(lines: list[str]) -> str:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    joined = "".join(lines[start:end])
    if joined.endswith("\r\n"):
        return joined[:-2]
    if joined.endswith(("\n", "\r")):
        return joined[:-1]
    return joined
