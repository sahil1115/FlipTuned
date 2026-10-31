"""Regenerate the synthetic fixtures: ``uv run python tests/fixtures/make_fixtures.py``.

Text fixtures are reproduced byte for byte. ``qa.xlsx`` is written only when missing, because
openpyxl stamps the save time into the file.

Everything here is invented. The "secrets" are fake values shaped like real ones so the
privacy scanner has something to find; none of them work anywhere.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from openpyxl import Workbook

HERE = Path(__file__).resolve().parent

BASIC_ROWS = [
    ["id", "question", "answer", "tags", "category", "status"],
    [
        "Q-001",
        "How do I validate input in Python?",
        "Use an explicit schema and return useful errors.\n\n```python\nfrom pydantic import BaseModel\n\nclass User(BaseModel):\n    name: str\n```\n",
        "python, validation",
        "coding",
        "approved",
    ],
    [
        "Q-002",
        "How should I name boolean variables?",
        "Prefix them with `is_`, `has_` or `can_` so the meaning reads naturally, e.g. `is_active`.",
        "style",
        "conventions",
        "approved",
    ],
    [
        "Q-003",
        "What is our rule for logging request bodies?",
        "Never log request bodies. Log the request ID, route and status code instead.",
        "logging",
        "conventions",
        "pending",
    ],
    [
        "Q-004",
        "How do I read a file safely in Python?",
        'Open it with a context manager and an explicit encoding:\n\n```python\nwith open(path, encoding="utf-8") as handle:\n    data = handle.read()\n```',
        "python",
        "coding",
        "approved",
    ],
    # duplicate external id
    [
        "Q-004",
        "How do I format dates for the API?",
        "Use ISO 8601 in UTC, for example `2024-05-01T12:00:00Z`.",
        "api",
        "conventions",
        "approved",
    ],
    # exact duplicate of Q-002 (whitespace differs)
    [
        "Q-006",
        "How should I name  boolean variables?",
        "Prefix them with `is_`, `has_` or `can_` so the meaning reads naturally, e.g. `is_active`.",
        "style",
        "conventions",
        "approved",
    ],
    # empty answer
    ["Q-007", "Which linter do we use for TypeScript?", "   ", "typescript", "tooling", "approved"],
    # likely question/answer swap
    [
        "Q-008",
        "Run the migrations before starting the server.",
        "When should I run database migrations?",
        "database",
        "operations",
        "approved",
    ],
    # blank row
    ["", "", "", "", "", ""],
    [
        "Q-009",
        "How do I write a café-safe Unicode string ✓?",
        'Python 3 strings are Unicode already: `s = "café ✓"`. Encode explicitly when writing bytes.',
        "python; unicode",
        "coding",
        "rejected",
    ],
    # broken code fence
    [
        "Q-010",
        "How do I list files recursively?",
        "Use pathlib:\n\n```python\nfor p in Path('.').rglob('*'):\n    print(p)\n",
        "python",
        "coding",
        "approved",
    ],
    ["Q-011", "Ok?", "y", "", "misc", "approved"],
]

CRLF_ROWS = [
    ["Question", "Answer"],
    [
        "How do I join paths portably?",
        'Use pathlib:\r\n\r\n```python\r\nfrom pathlib import Path\r\nconfig = Path.home() / ".app" / "config.yaml"\r\n```',
    ],
    ["What line endings do we commit?", "LF for text files; CRLF only for .cmd, .bat and .ps1."],
    [
        "How do I check the exit code in PowerShell?",
        "Read `$LASTEXITCODE` right after the command.",
    ],
]

XLSX_ROWS = [
    ["question", "answer", "group"],
    ["How do I start the dev server?", "Run `npm run dev` from the web folder.", "onboarding"],
    [
        "Where do I find the dev server port?",
        "It is printed on start; the default is 5173.",
        "onboarding",
    ],
    ["How do I add a dependency?", "Use `uv add <package>` and commit uv.lock.", "dependencies"],
    [
        "How do I upgrade one dependency?",
        "Use `uv lock --upgrade-package <package>`.",
        "dependencies",
    ],
    ["What is the review SLA?", "Reviews are answered within one working day.", ""],
]

MARKDOWN = """# Team engineering notes

These notes collect common questions from the team channel. They are synthetic.

## Python

Q: How do I make a dataclass immutable?
A: Pass `frozen=True`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: int
    y: int
```

Q: Can I put Q: lines inside code?
A: Yes, fences are kept intact:

```text
Q: this is not a new question
A: and this is not a new answer
```

| Option | Meaning |
|---|---|
| frozen | immutable |
| slots | less memory |

## Git

Q: How do I undo the last local commit but keep the changes?
A: Run `git reset --soft HEAD~1`.

- the changes stay staged
- nothing is pushed

Q: How do I see which files changed?
A: Use `git status --short`.
```bash
git status --short

## Release trains

Free-form text about release trains that is not a question and becomes a source block.
"""

TEXT = """General reminder: keep the runbook up to date.

Q: How long do we keep build logs?
A: Thirty days, then they are deleted automatically.

Q: Who approves production deploys?
A: The on-call engineer for the service.
They confirm in the deploy channel first.
"""

SECRETS_ROWS = [
    ["question", "answer"],
    [
        "How do I configure the AWS CLI?",
        "Set the keys in your profile, e.g. aws_access_key_id = AKIAIOSFODNN7EXAMPLE.",
    ],
    [
        "How do I call the internal API?",
        "Send the header Authorization: Bearer ghp_FAKEfakeFAKEfakeFAKEfakeFAKEfake1234 with each request.",
    ],
    ["Which database URL do I use locally?", "postgres://devuser:devpass42@localhost:5432/app"],
    [
        "What should the config look like?",
        'Put password = "hunter2hunter2" in settings.ini and restart.',
    ],
    [
        "What does a placeholder look like?",
        "Write password = <your-password> in the template; never a real one.",
    ],
    [
        "Where is the deploy key?",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAfakekeymaterialAAAA\n-----END OPENSSH PRIVATE KEY-----",
    ],
]


# A dataset large enough to split: 12 tasks x 6 languages x 2 phrasings, plus edge cases.
# Tasks 0, 3, 6, 9 use the group column (one group per task, 12 rows); tasks 1, 4, 7, 10 use
# the thread column (one thread per task and language, 2 rows); the rest have neither.
SYN_LANGS = [
    ("Python", "python"),
    ("TypeScript", "ts"),
    ("Go", "go"),
    ("Rust", "rust"),
    ("SQL", "sql"),
    ("Bash", "bash"),
]
SYN_TASKS = [
    (
        "read a config file",
        "load it once at startup, validate every field and fail fast with a clear message",
        "coding",
    ),
    (
        "log an error",
        "log the error type, a request ID and the operation name, never the payload",
        "conventions",
    ),
    (
        "write a unit test",
        "test one behaviour per test, name it after the behaviour and keep fixtures synthetic",
        "testing",
    ),
    (
        "handle a timeout",
        "set an explicit timeout on every network call and retry only idempotent requests",
        "operations",
    ),
    (
        "name a function",
        "use a verb phrase that says what it does, such as load_user or send_invoice",
        "conventions",
    ),
    (
        "parse a date",
        "parse ISO 8601 strings into timezone-aware values and store them in UTC",
        "coding",
    ),
    (
        "validate user input",
        "validate at the boundary with an explicit schema and return field-level errors",
        "coding",
    ),
    (
        "structure a module",
        "keep public functions first, private helpers below, and one responsibility per module",
        "conventions",
    ),
    (
        "paginate results",
        "use cursor-based pagination with a stable sort key and a maximum page size",
        "coding",
    ),
    (
        "store a secret",
        "read it from the environment or the secret manager at runtime, never from the repository",
        "operations",
    ),
    (
        "review a pull request",
        "check behaviour and tests first, then naming, then style, and leave actionable comments",
        "conventions",
    ),
    (
        "deprecate an API",
        "mark it deprecated, log a warning naming the replacement, and remove it after two releases",
        "operations",
    ),
]


def synthetic_rows() -> list[list[str]]:
    rows = [["id", "question", "answer", "category", "status", "group", "thread"]]
    n = 0
    for t, (task, how, category) in enumerate(SYN_TASKS):
        for lang, tag in SYN_LANGS:
            for phrasing in range(2):
                n += 1
                if phrasing == 0:
                    question = f"How do I {task} in {lang}?"
                    answer = f"In {lang}, {how}."
                else:
                    question = f"What is our {lang} convention to {task}?"
                    answer = f"Our {lang} convention: {how}. Link the decision record in the PR description."
                if n % 3 == 0:
                    answer += f"\n\n```{tag}\n// example: {task}\n```"
                status = "pending" if n % 17 == 5 else "rejected" if n % 23 == 7 else "approved"
                group = f"G-{t:02d}" if t % 3 == 0 else ""
                thread = f"T-{t:02d}-{tag}" if t % 3 == 1 else ""
                rows.append([f"S-{n:04d}", question, answer, category, status, group, thread])
    # Edge cases: an exact duplicate (whitespace differs), a same-question/different-answer pair
    # with no group of its own, and an answer far beyond 1024 tokens.
    rows.append(
        [
            "S-9001",
            "How do I  read a config file in Python?",
            rows[1][2],
            "coding",
            "approved",
            "",
            "",
        ]
    )
    rows.append(
        [
            "S-9002",
            rows[3][1],
            "Ask in the team channel first; the convention changed last year.",
            "conventions",
            "approved",
            "",
            "",
        ]
    )
    long_answer = " ".join(
        f"Step {i}: check the input, log the outcome and move on." for i in range(1, 301)
    )
    rows.append(
        [
            "S-9003",
            "Can you describe the full release checklist?",
            long_answer,
            "operations",
            "approved",
            "",
            "",
        ]
    )
    return rows


def _csv_bytes(rows: list[list[str]], line_terminator: str) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator=line_terminator).writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    (HERE / "qa_basic.csv").write_bytes(_csv_bytes(BASIC_ROWS, "\n"))
    (HERE / "qa_crlf.csv").write_bytes(_csv_bytes(CRLF_ROWS, "\r\n"))
    (HERE / "secrets.csv").write_bytes(_csv_bytes(SECRETS_ROWS, "\n"))
    (HERE / "notes.md").write_bytes(MARKDOWN.encode("utf-8"))
    (HERE / "notes.txt").write_bytes(TEXT.encode("utf-8"))

    # Malformed: a ragged row, then an unterminated quote.
    (HERE / "malformed.csv").write_bytes(
        b"question,answer\n"
        b"How do I rotate logs?,Use logrotate weekly.\n"
        b"Too many fields?,yes,unexpected,extra\n"
        b'Is this quote closed?,"no it is not\n'
    )
    # Not UTF-8: cp1252 bytes for "cafe" with e-acute.
    (HERE / "bad_encoding.csv").write_bytes(
        b"question,answer\nWhere is the caf\xe9?,On the second floor.\n"
    )

    (HERE / "qa_synthetic.csv").write_bytes(_csv_bytes(synthetic_rows(), "\n"))

    if not (HERE / "qa.xlsx").exists():
        wb = Workbook()
        sheet = wb.active
        sheet.title = "QA"
        for row in XLSX_ROWS:
            sheet.append(row)
        notes = wb.create_sheet("Notes")
        notes.append(["This sheet is ignored by the sample config."])
        wb.properties.creator = "FlipTuned fixtures"
        wb.save(HERE / "qa.xlsx")

    scan_dir = HERE / "scan_folder"
    (scan_dir / "nested").mkdir(parents=True, exist_ok=True)
    (scan_dir / "faq.csv").write_bytes(
        _csv_bytes([["Prompt", "Response", "Ticket ID"], ["a?", "b"]], "\n")
    )
    (scan_dir / "nested" / "guide.md").write_bytes(b"Q: x?\nA: y\n")
    (scan_dir / "manual.pdf").write_bytes(b"%PDF-1.4 fake\n")
    (scan_dir / "latin1.txt").write_bytes(b"caf\xe9\n")
    (scan_dir / "~$faq.xlsx").write_bytes(b"lock file")


if __name__ == "__main__":
    main()
