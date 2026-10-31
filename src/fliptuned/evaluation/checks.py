"""Deterministic answer checks (SPEC §12.2). Static only: code is parsed, never run.

Each check applies to an item only where it fits:

- always: ``non_empty``, ``not_truncated`` (the token budget didn't run out), ``no_repetition``;
- when the item expects code: ``has_code_block``, and ``code_language`` when a language is known;
- when the answer has code blocks: ``fences_closed``, and ``syntax`` for blocks in a language with
  a local parser (Python via ``ast.parse``, JSON via ``json.loads``);
- when the item lists them: ``required_patterns`` and ``forbidden_patterns`` (case-insensitive
  regexes, searched in the code blocks when the item expects code, otherwise in the whole answer).

Test-set items take their expectations from the approved reference answer; regression-set items
state them. Check details name the check and a location, never answer text.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

CHECKS = {
    "non_empty": "The answer isn't empty.",
    "not_truncated": "The answer ended before the token limit.",
    "no_repetition": "The answer doesn't loop on the same text.",
    "has_code_block": "The answer has a code block where one is expected.",
    "code_language": "A code block is tagged with the expected language.",
    "fences_closed": "Every code block is closed.",
    "syntax": "Python and JSON code blocks parse (parsed only, never run).",
    "required_patterns": "Text the item requires is present.",
    "forbidden_patterns": "Text the item forbids is absent.",
}

LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "py3": "python",
    "js": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "yml": "yaml",
    "c++": "cpp",
    "cs": "csharp",
    "c#": "csharp",
    "golang": "go",
    "rs": "rust",
    "ps1": "powershell",
    "pwsh": "powershell",
    "postgresql": "sql",
    "sqlite": "sql",
}

_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^`\s]*)")
_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_TAIL_LOOP = re.compile(r"(.{2,80}?)\1{5,}\s*$", re.DOTALL)
_TAIL_CHARS = 600


def normalize_language(tag: str) -> str:
    tag = tag.strip().lower()
    return LANGUAGE_ALIASES.get(tag, tag)


@dataclass(frozen=True)
class CodeBlock:
    language: str  # normalized; "" when untagged
    code: str
    closed: bool


def code_blocks(text: str) -> list[CodeBlock]:
    """Fenced code blocks (``` or ~~~), CommonMark-style: a closing fence uses the same character,
    is at least as long as the opening one and has no info string."""
    blocks: list[CodeBlock] = []
    fence: str | None = None
    language = ""
    body: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if fence is None:
            opened = _OPEN.match(line)
            if opened:
                fence, language, body = opened.group(1), normalize_language(opened.group(2)), []
            continue
        closed = _CLOSE.match(line)
        if closed and closed.group(1)[0] == fence[0] and len(closed.group(1)) >= len(fence):
            blocks.append(CodeBlock(language, "\n".join(body), True))
            fence = None
        else:
            body.append(line)
    if fence is not None:
        blocks.append(CodeBlock(language, "\n".join(body), False))
    return blocks


@dataclass
class Expectations:
    """What an answer to this item should contain."""

    code: bool = False
    language: str | None = None
    required: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


def expectations_from_reference(answer: str) -> Expectations:
    blocks = code_blocks(answer)
    language = next((b.language for b in blocks if b.language), None)
    return Expectations(code=bool(blocks), language=language)


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


def _repetitive(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 8:
        top, count = Counter(lines).most_common(1)[0]
        if count >= 6 and count / len(lines) > 0.5 and sum(c.isalnum() for c in top) >= 2:
            return True
    loop = _TAIL_LOOP.search(text[-_TAIL_CHARS:])
    return bool(loop and any(c.isalnum() for c in loop.group(1)))


def _parse_error(language: str, code: str) -> str | None:
    """None if the code parses; otherwise where it failed. Never executes anything."""
    if language == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(code)
        except SyntaxError as exc:
            return f"line {exc.lineno}"
        except (ValueError, RecursionError, MemoryError) as exc:
            return type(exc).__name__
        return None
    if language == "json":
        try:
            json.loads(code)
        except ValueError as exc:
            return f"line {getattr(exc, 'lineno', '?')}"
        return None
    raise KeyError(language)


PARSED_LANGUAGES = ("python", "json")


def _pattern_hits(patterns: list[str], text: str) -> list[int]:
    return [i for i, p in enumerate(patterns, start=1) if re.search(p, text, re.IGNORECASE)]


def run_checks(output: str, finish_reason: str, expect: Expectations) -> list[CheckResult]:
    results = [
        CheckResult(name="non_empty", passed=bool(output.strip())),
        CheckResult(name="not_truncated", passed=finish_reason != "length"),
        CheckResult(name="no_repetition", passed=not _repetitive(output)),
    ]
    blocks = code_blocks(output)
    if expect.code:
        results.append(CheckResult(name="has_code_block", passed=bool(blocks)))
        if expect.language:
            results.append(
                CheckResult(
                    name="code_language",
                    passed=any(b.language == expect.language for b in blocks),
                    detail=f"expected {expect.language}",
                )
            )
    if blocks:
        unclosed = sum(not b.closed for b in blocks)
        results.append(
            CheckResult(
                name="fences_closed",
                passed=unclosed == 0,
                detail=f"{unclosed} unclosed" if unclosed else None,
            )
        )
    parsed = [(i, b) for i, b in enumerate(blocks, start=1) if b.language in PARSED_LANGUAGES]
    if parsed:
        errors = [
            f"{b.language} block {i}: {err}"
            for i, b in parsed
            if (err := _parse_error(b.language, b.code)) is not None
        ]
        results.append(
            CheckResult(name="syntax", passed=not errors, detail="; ".join(errors) or None)
        )
    scope = "\n".join(b.code for b in blocks) if expect.code else output
    if expect.required:
        missing = [
            i
            for i in range(1, len(expect.required) + 1)
            if i not in _pattern_hits(expect.required, scope)
        ]
        results.append(
            CheckResult(
                name="required_patterns",
                passed=not missing,
                detail=f"missing pattern(s) {missing}" if missing else None,
            )
        )
    if expect.forbidden:
        found = _pattern_hits(expect.forbidden, scope)
        results.append(
            CheckResult(
                name="forbidden_patterns",
                passed=not found,
                detail=f"found pattern(s) {found}" if found else None,
            )
        )
    return results


def all_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)
