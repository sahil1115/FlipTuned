from __future__ import annotations

from fliptuned.ingestion.textqa import parse_text_qa

MD = (
    "# Title\n"
    "Intro paragraph.\n"
    "\n"
    "## Python\n"
    "Q: How do I freeze a dataclass?\n"
    "A: Pass `frozen=True`:\n"
    "\n"
    "```python\n"
    "@dataclass(frozen=True)\n"
    "class P:\n"
    "    x: int\n"
    "```\n"
    "\n"
    "Q: Fences?\n"
    "A: Kept:\n"
    "```text\n"
    "Q: not a question\n"
    "A: not an answer\n"
    "```\n"
    "## Next\n"
    "Q: Multi-line\n"
    "question?\n"
    "A: answer\n"
)


def _qa(items):
    return [i for i in items if i.kind == "qa"]


def test_markdown_pairs_fences_and_headings() -> None:
    items = parse_text_qa(MD, "Q:", "A:", markdown=True)
    qa = _qa(items)
    assert [q.question for q in qa] == [
        "How do I freeze a dataclass?",
        "Fences?",
        "Multi-line\nquestion?",
    ]
    assert (
        qa[0].answer
        == "Pass `frozen=True`:\n\n```python\n@dataclass(frozen=True)\nclass P:\n    x: int\n```"
    )
    assert "Q: not a question\nA: not an answer" in qa[1].answer
    assert qa[1].answer.endswith("```")  # the heading ended the answer
    assert [(q.line, q.heading) for q in qa] == [(5, "Python"), (14, "Python"), (21, "Next")]
    blocks = [i for i in items if i.kind == "block"]
    assert len(blocks) == 1 and blocks[0].text.startswith("# Title\nIntro paragraph.")


def test_heading_only_blocks_are_dropped() -> None:
    items = parse_text_qa("Q: a?\nA: b\n## Section\n\nQ: c?\nA: d\n", "Q:", "A:", markdown=True)
    assert [i.kind for i in items] == ["qa", "qa"]


def test_crlf_text_is_preserved_exactly() -> None:
    text = "Q: How?\r\nA: Like this:\r\n```\r\nx\r\n```\r\n"
    (item,) = parse_text_qa(text, "Q:", "A:", markdown=True)
    assert item.question == "How?"
    assert item.answer == "Like this:\r\n```\r\nx\r\n```"


def test_txt_has_no_headings_and_keeps_blocks() -> None:
    text = "Reminder first.\n\nQ: One?\nA: Yes.\n# not a heading in txt\n"
    items = parse_text_qa(text, "Q:", "A:", markdown=False)
    assert [i.kind for i in items] == ["block", "qa"]
    assert items[1].answer == "Yes.\n# not a heading in txt"


def test_question_without_answer_and_custom_delimiters() -> None:
    (item,) = parse_text_qa("Question: Lonely?\n", "Question:", "Answer:", markdown=False)
    assert item.question == "Lonely?" and item.answer == ""
    (pair,) = parse_text_qa("**Q** x?\n**A** y\n", "**Q**", "**A**", markdown=True)
    assert (pair.question, pair.answer) == ("x?", "y")
