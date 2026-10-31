"""Local secret scanner (SPEC §3.3).

Findings carry a type and character offsets, never the matched value. Use
:meth:`PrivacyFinding.redacted` for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fliptuned.errors import ConfigError
from fliptuned.records.schema import PrivacyFinding


@dataclass(frozen=True)
class Pattern:
    type: str
    regex: re.Pattern[str]
    value_group: int = 0  # the group holding the secret value (for placeholder checks)
    skip_code_values: bool = False  # ignore values that are code (dotted names, calls)


BUILTIN_PATTERNS: tuple[Pattern, ...] = (
    Pattern("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    Pattern("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    Pattern("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    Pattern("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    Pattern("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    Pattern("api_secret_key", re.compile(r"\bsk-(?:[A-Za-z]+-)?[A-Za-z0-9_-]{20,}")),
    Pattern("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    Pattern(
        "connection_string",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqps?|mssql|sqlserver)"
            r"://[^\s:@/]+:([^\s@/]+)@"
        ),
        value_group=1,
    ),
    Pattern(
        "account_key", re.compile(r"(?i)\b(?:AccountKey|SharedAccessKey)=([A-Za-z0-9+/=]{20,})"), 1
    ),
    Pattern(
        "password_assignment",
        re.compile(
            # key names such as password, db_password, SECRET_KEY, apiKey, auth-token
            r"(?i)\b[\w-]*?(?:password|passwd|pwd|secret|api[_-]?key|token)[\w-]*"
            r"[\"']?\s*[:=]\s*[\"']?([^\s\"'<>,;()\[\]{}]{6,})"
        ),
        value_group=1,
        skip_code_values=True,
    ),
)

_DOTTED_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")

_PLACEHOLDER = re.compile(
    r"(?i)^(?:\*+|x{3,}|\.{3,}|\d+|<[^>]*>|\$\{[^}]*\}|\{\{[^}]*\}\}|%[A-Z_]+%|os\.environ\S*|env\S*|"
    r"your[_-]?\w*|example\w*|changeme|placeholder|redacted|dummy|none|null|true|false)$"
)


class SecretScanner:
    def __init__(self, extra_patterns: list[str] | None = None):
        patterns = list(BUILTIN_PATTERNS)
        for index, expression in enumerate(extra_patterns or []):
            try:
                patterns.append(Pattern(f"custom_{index + 1}", re.compile(expression)))
            except re.error as exc:
                raise ConfigError(
                    f"Custom privacy pattern #{index + 1} is not a valid regular expression.",
                    hint="Fix privacy.extra_patterns in the ingest config.",
                    details=str(exc),
                ) from None
        self.patterns = tuple(patterns)

    def scan(self, text: str | None, field: str) -> list[PrivacyFinding]:
        if not text:
            return []
        findings: list[PrivacyFinding] = []
        taken: list[tuple[int, int]] = []
        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                value = match.group(pattern.value_group) if pattern.value_group else match.group(0)
                if _PLACEHOLDER.match(value.strip("\"'")):
                    continue
                if pattern.skip_code_values and _is_code(text, match, pattern.value_group):
                    continue
                start, end = match.span()
                if any(start < t_end and t_start < end for t_start, t_end in taken):
                    continue  # one finding per location
                taken.append((start, end))
                findings.append(
                    PrivacyFinding(type=pattern.type, field=field, start=start, end=end)
                )
        findings.sort(key=lambda f: f.start)
        return findings


def _is_code(text: str, match: re.Match[str], group: int) -> bool:
    value = match.group(group)
    following = text[match.end(group) : match.end(group) + 1]
    return following in ("(", "[") or bool(_DOTTED_NAME.match(value))


def redact_text(text: str | None, findings: list[PrivacyFinding], field: str) -> str | None:
    """``text`` with every finding for ``field`` replaced by its redacted display form."""
    if text is None:
        return None
    spans = sorted((f for f in findings if f.field == field), key=lambda f: f.start, reverse=True)
    for finding in spans:  # from the end, so earlier offsets stay valid
        text = text[: finding.start] + finding.redacted + text[finding.end :]
    return text
