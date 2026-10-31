"""Evaluation (SPEC §12): sign test, deterministic checks, verdict branches, blind review,
judgements (one at a time and from the review sheet), scoring and artifacts.

The service tests use a fake backend and a fake finished run, so nothing is downloaded.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from fliptuned.cli import app
from fliptuned.config.app_home import AppPaths
from fliptuned.config.models import AcceptanceCriteria, Backend
from fliptuned.errors import ConfigError, FlipTunedError, NotAvailableError
from fliptuned.evaluation.blinding import draw_a_model, parse_judgement, unblind
from fliptuned.evaluation.checks import (
    Expectations,
    code_blocks,
    expectations_from_reference,
    run_checks,
)
from fliptuned.evaluation.regression import load_regression_set
from fliptuned.evaluation.stats import sign_test
from fliptuned.evaluation.verdict import CategoryResult, VerdictInputs, decide
from fliptuned.jobs.worker import CANCEL_FILE, job_dir
from fliptuned.registry import repositories
from fliptuned.registry.db import connect
from fliptuned.services import evaluation
from fliptuned.services import jobs as jobs_service
from fliptuned.services.datasets import (
    BuildDatasetRequest,
    build_dataset,
)
from fliptuned.training.backends.base import GenerationConfig, GenerationResult, LoadedModel
from tests.helpers.pipeline import SAMPLE_DATASET, ingest_samples, use_fake_tokenizer
from tests.helpers.runs import make_run

SRC = Path(__file__).resolve().parents[2] / "src" / "fliptuned"
runner = CliRunner()

# --------------------------------------------------------------------------- sign test


@pytest.mark.parametrize(
    ("wins", "losses", "expected"),
    [
        (15, 5, 2 * 21700 / 2**20),  # = 0.0414: sum of C(20, 0..5) = 21700
        (14, 6, 2 * 60460 / 2**20),  # = 0.1153: sum of C(20, 0..6) = 60460
        (9, 1, 2 * 11 / 2**10),  # = 0.0215
        (10, 0, 2 / 2**10),
        (3, 0, 0.25),
        (5, 5, 1.0),
        (0, 0, 1.0),
        (1, 0, 1.0),
    ],
)
def test_sign_test_known_values(wins: int, losses: int, expected: float) -> None:
    assert sign_test(wins, losses) == pytest.approx(expected, rel=1e-12)
    assert sign_test(losses, wins) == sign_test(wins, losses)  # two-sided


def test_sign_test_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        sign_test(-1, 3)


# --------------------------------------------------------------------------- checks

GOOD_PY = "Use this:\n\n```python\ndef add(a, b):\n    return a + b\n```\n"


def _by_name(results) -> dict[str, bool]:
    return {r.name: r.passed for r in results}


def test_code_blocks_parse_fences_and_languages() -> None:
    text = "a\n```py\nx = 1\n```\n~~~~json\n{}\n~~~~\n````\nnested ``` inside\n````\n```sh\necho"
    blocks = code_blocks(text)
    assert [(b.language, b.closed) for b in blocks] == [
        ("python", True),
        ("json", True),
        ("", True),
        ("bash", False),
    ]
    assert blocks[2].code == "nested ``` inside"
    assert code_blocks("```python\r\nx = 1\r\n```\r\n")[0].code == "x = 1"


def test_checks_on_a_good_answer() -> None:
    expect = expectations_from_reference("```python\npass\n```")
    assert (expect.code, expect.language) == (True, "python")
    results = run_checks(GOOD_PY, "stop", expect)
    assert _by_name(results) == {
        "non_empty": True,
        "not_truncated": True,
        "no_repetition": True,
        "has_code_block": True,
        "code_language": True,
        "fences_closed": True,
        "syntax": True,
    }


def test_checks_find_problems_without_quoting_text() -> None:
    expect = Expectations(code=True, language="python")
    bad = "```python\ndef add(a, b)\n    return a + b\n```"
    results = run_checks(bad, "length", expect)
    named = _by_name(results)
    assert named["syntax"] is False and named["not_truncated"] is False
    syntax = next(r for r in results if r.name == "syntax")
    assert syntax.detail == "python block 1: line 1" and "return" not in syntax.detail
    assert _by_name(run_checks("", "stop", expect))["non_empty"] is False
    assert _by_name(run_checks("plain text", "stop", expect))["has_code_block"] is False
    js_only = _by_name(run_checks("```js\nlet x = 1\n```", "stop", expect))
    assert js_only["code_language"] is False and "syntax" not in js_only  # no JS parser
    unclosed = _by_name(run_checks("```python\nx = 1\n", "stop", expect))
    assert unclosed["fences_closed"] is False
    assert (
        _by_name(run_checks('```json\n{"a": 1,}\n```', "stop", Expectations()))["syntax"] is False
    )


def test_checks_never_run_code(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    code = f"```python\nopen({str(marker)!r}, 'w').write('x')\n```"
    assert _by_name(run_checks(code, "stop", Expectations(code=True)))["syntax"] is True
    assert not marker.exists()


@pytest.mark.parametrize(
    ("text", "repetitive"),
    [
        ("the answer " * 30, True),
        ("\n".join(["print('hello world')"] * 10), True),
        ("intro\n" + "x = compute(1)\n" * 7 + "done", True),
        ("| a | b |\n|---|---|---|---|---|---|---|\n| 1 | 2 |", False),
        ("}\n" * 12, False),
        (GOOD_PY, False),
        ("Line one.\nLine two.\nLine three.", False),
    ],
)
def test_repetition_check(text: str, repetitive: bool) -> None:
    passed = _by_name(run_checks(text, "stop", Expectations()))["no_repetition"]
    assert passed is not repetitive


def test_required_and_forbidden_patterns_search_code_when_code_expected() -> None:
    expect = Expectations(code=True, required=[r"json\.loads\("], forbidden=[r"\beval\("])
    answer = "Never call eval() on input.\n```python\nimport json\njson.loads(text)\n```"
    named = _by_name(run_checks(answer, "stop", expect))
    assert named["required_patterns"] is True and named["forbidden_patterns"] is True
    unsafe = "```python\nvalue = eval(text)\n```"
    named = _by_name(run_checks(unsafe, "stop", expect))
    assert named["required_patterns"] is False and named["forbidden_patterns"] is False


def test_regression_set_is_bundled_and_valid() -> None:
    reg = load_regression_set()
    assert 18 <= len(reg.items) <= 25
    assert len({i.id for i in reg.items}) == len(reg.items)
    for item in reg.items:
        assert item.prompt.strip()
        assert not item.expect.code or item.expect.language, item.id
    good = (
        "```python\ndef is_palindrome(text: str) -> bool:\n"
        "    s = [c.lower() for c in text if c.isalnum()]\n    return s == s[::-1]\n```"
    )
    first = reg.items[0]
    assert all(r.passed for r in run_checks(good, "stop", first.expect.expectations()))


def test_evaluation_code_has_no_execution_paths() -> None:
    """Deterministic checks parse code; nothing in evaluation runs it (SPEC §12.2, v1)."""
    files = [*sorted((SRC / "evaluation").glob("*.py")), SRC / "services" / "evaluation.py"]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile", "__import__"}, path
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert not any(n.split(".")[0] in {"subprocess", "runpy"} for n in names), path


# --------------------------------------------------------------------------- blinding helpers


def test_blinding_helpers() -> None:
    draws = {draw_a_model(1234, "ev_1", f"qa_{i}") for i in range(40)}
    assert draws == {"base", "tuned"}
    assert draw_a_model(1234, "ev_1", "qa_1") == draw_a_model(1234, "ev_1", "qa_1")
    assert [parse_judgement(w) for w in ("A", " b ", "TIE", "1", "2", "3", "x", "")] == [
        "A",
        "B",
        "tie",
        "A",
        "B",
        "tie",
        None,
        None,
    ]
    assert unblind("A", "tuned") == "tuned" and unblind("B", "tuned") == "base"
    assert unblind("A", "base") == "base" and unblind("B", "base") == "tuned"
    assert unblind("tie", "base") == "tie"


# --------------------------------------------------------------------------- verdict branches


def _inputs(**overrides) -> VerdictInputs:
    data = dict(
        test_items=60,
        judged=60,
        wins=40,
        losses=10,
        ties=10,
        categories=[
            CategoryResult(category="python", items=30, judged=30, wins=20, losses=5, ties=5),
            CategoryResult(category="sql", items=30, judged=30, wins=20, losses=5, ties=5),
        ],
        deterministic_pass_base=80.0,
        deterministic_pass_tuned=80.0,
        regression_pass_base=70.0,
        regression_pass_tuned=70.0,
    )
    data.update(overrides)
    return VerdictInputs(**data)


SMALL_LOSING_CATEGORY = CategoryResult(
    category="bash", items=10, judged=10, wins=1, losses=5, ties=4
)


@pytest.mark.parametrize(
    ("overrides", "verdict", "unmet"),
    [
        ({}, "accepted", []),
        ({"test_items": 40, "judged": 40}, "inconclusive", ["test_items"]),
        ({"judged": 59}, "inconclusive", ["review_coverage"]),
        ({"wins": 12, "losses": 5, "ties": 43}, "inconclusive", ["non_tied_pairs"]),
        # inconclusive is decided before rejected
        (
            {"test_items": 10, "judged": 10, "wins": 1, "losses": 9, "ties": 0},
            "inconclusive",
            ["test_items", "non_tied_pairs"],
        ),
        ({"wins": 25, "losses": 25}, "rejected", ["more_wins_than_losses", "sign_test"]),
        ({"wins": 28, "losses": 22}, "rejected", ["sign_test"]),
        ({"categories": [SMALL_LOSING_CATEGORY]}, "rejected", ["category_net_loss:bash"]),
        ({"deterministic_pass_tuned": 77.5}, "rejected", ["deterministic_drop"]),
        ({"deterministic_pass_tuned": 78.0}, "accepted", []),  # exactly the 2-point limit
        ({"regression_pass_tuned": 64.0}, "rejected", ["regression_drop"]),
        ({"regression_pass_base": None, "regression_pass_tuned": None}, "accepted", []),
    ],
)
def test_verdict_branches(overrides: dict, verdict: str, unmet: list[str]) -> None:
    result = decide(_inputs(**overrides), AcceptanceCriteria())
    assert result.verdict == verdict
    stage_unmet = [c.name for c in result.criteria if c.applicable and not c.met]
    if verdict == "inconclusive":
        stage_unmet = [
            n for n in stage_unmet if n in {"test_items", "review_coverage", "non_tied_pairs"}
        ]
    assert stage_unmet == unmet
    assert len(result.reasons) == len(unmet)


def test_verdict_reports_every_input() -> None:
    result = decide(
        _inputs(regression_pass_base=None, regression_pass_tuned=None), AcceptanceCriteria()
    )
    names = [c.name for c in result.criteria]
    assert names == [
        "test_items",
        "review_coverage",
        "non_tied_pairs",
        "more_wins_than_losses",
        "sign_test",
        "category_net_loss:python",
        "category_net_loss:sql",
        "deterministic_drop",
        "regression_drop",
    ]
    regression = result.criteria[-1]
    assert not regression.applicable and regression.display_value == "not run"
    assert result.sign_test_p == sign_test(40, 10)


def test_small_categories_do_not_count() -> None:
    tiny = CategoryResult(category="go", items=4, judged=4, wins=0, losses=4, ties=0)
    result = decide(_inputs(categories=[tiny]), AcceptanceCriteria())
    assert result.verdict == "accepted"
    assert not any(c.name.startswith("category_net_loss") for c in result.criteria)


def test_missing_review_coverage_is_inconclusive_even_when_winning() -> None:
    result = decide(_inputs(judged=30, wins=30, losses=0, ties=0), AcceptanceCriteria())
    assert result.verdict == "inconclusive"
    assert result.reasons == ["Not met: At least 100% of test pairs reviewed."]


# --------------------------------------------------------------------------- service (fake model)

ALPHA = "alpha answer\n```python\ndef f(:\n```"  # base: syntax error
OMEGA = "=omega answer\n```python\ndef f():\n    return 1\n```"  # fine-tuned; starts with "="


class FakeBackend:
    name = Backend.HF_PEFT

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.unloaded = 0
        self.on_call = None

    def generate(self, loaded, messages, config: GenerationConfig) -> GenerationResult:
        side = "tuned" if loaded.adapter_path is not None else "base"
        assert len(messages) == 2 and messages[0]["role"] == "system"
        self.calls.append((side, messages[1]["content"]))
        if self.on_call:
            self.on_call(len(self.calls))
        return GenerationResult(
            text=OMEGA if side == "tuned" else ALPHA,
            prompt_tokens=7,
            completion_tokens=5,
            finish_reason="stop",
            seconds=0.01,
        )

    def unload(self, loaded) -> None:
        self.unloaded += 1

    def artifact_format(self) -> str:
        return "peft_lora_safetensors"

    def versions(self) -> dict:
        return {"torch": "fake"}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()

    def load(entry, selection, *, allow_download=False, adapter_path=None):
        return backend, LoadedModel(
            backend=selection.backend,
            entry=entry,
            selection=selection,
            model_dir=Path("."),
            adapter_path=adapter_path,
            model=None,
            tokenizer=None,
            template_checksum="c" * 64,
        )

    monkeypatch.setattr(evaluation, "load_model", load)
    return backend


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    use_fake_tokenizer(monkeypatch)
    home = tmp_path / "home"
    ingest_samples(home)
    build_dataset(BuildDatasetRequest(config_path=str(SAMPLE_DATASET), home=str(home)))
    return home


def small_config(tmp_path: Path, **acceptance) -> str:
    """Acceptance criteria scaled to the 14-item fixture test set."""
    data = {"kind": "evaluate", "acceptance": {"min_test_items": 10, "min_non_tied_pairs": 10}}
    data["acceptance"].update(acceptance)
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def run_evaluation(home: Path, run_id: str, config: str | None = None, **kw):
    start = evaluation.start_evaluation(
        evaluation.EvaluateRequest(home=str(home), run_id=run_id, config_path=config, **kw),
        detach=False,
    )
    assert start.job.holds_model_lock and start.job.type == "evaluate"
    info = jobs_service.run_inline(str(home), start.job.id)
    assert info.status == "succeeded", jobs_service.job_error(str(home), start.job.id)
    return jobs_service.job_result(str(home), start.job.id, evaluation.EvaluationOutcome)


def test_evaluation_job_writes_artifacts(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    outcome = run_evaluation(home, run_id)
    folder = Path(outcome.eval_dir)
    assert folder.parent.name == "evaluation" and folder.parent.parent.name == run_id
    for name in ("results.jsonl", "metrics.json", "review_sheet.csv", "report.md", "verdict.json"):
        assert (folder / name).exists(), name
    assert outcome.test_items == 14 and outcome.regression_items == 20
    assert not outcome.baseline_reused and outcome.baseline_id.startswith("bl_")

    # base: 14 test + 20 regression; fine-tuned: the same; models released in between
    assert [side for side, _ in fake.calls].count("base") == 34
    assert [side for side, _ in fake.calls].count("tuned") == 34
    assert fake.unloaded == 2

    score = outcome.score
    assert score.verdict == "inconclusive" and score.judged == 0
    assert any("reviewed" in r for r in score.reasons)
    assert score.deterministic_pass_base == 0.0  # every base answer has a syntax error
    assert score.deterministic_pass_tuned >= score.deterministic_pass_base
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["generation"] == {"max_new_tokens": 512, "decoding": "greedy"}
    assert meta["backend"] == "hf_peft" and meta["model"]["template_checksum"] == "c" * 64
    verdict = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "inconclusive" and verdict["inputs"]["test_items"] == 14
    info = evaluation.get_evaluation(str(home), outcome.evaluation_id)
    assert (info.status, info.verdict, info.total, info.judged) == ("ready", "inconclusive", 14, 0)

    # A second evaluation reuses the saved baseline and notes the shared test items.
    again = run_evaluation(home, run_id)
    assert again.baseline_reused and again.baseline_id == outcome.baseline_id
    assert [side for side, _ in fake.calls].count("base") == 34 + 20  # regression set only
    assert (
        again.score.prior_exposure_note
        and "1 earlier evaluation" in again.score.prior_exposure_note
    )
    assert outcome.score.prior_exposure_note is None


def test_reports_hold_no_record_text(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    outcome = run_evaluation(home, run_id)
    folder = Path(outcome.eval_dir)
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text("utf-8").splitlines()]
    texts = [r["question"] for r in rows if r["set"] == "test"] + ["omega answer", "alpha answer"]
    for name in ("report.md", "metrics.json", "verdict.json"):
        content = (folder / name).read_text(encoding="utf-8")
        assert not [t for t in texts if t in content], name


def test_blinding_round_trip(
    home: Path, fake: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fixed ID makes the seeded A/B draws (and so this test) deterministic.
    monkeypatch.setattr(evaluation, "new_evaluation_id", lambda: "ev_20261029000000_abcdef")
    run_id = make_run(home)
    outcome = run_evaluation(home, run_id, small_config(tmp_path))
    eval_id = outcome.evaluation_id
    allowed = {
        "evaluation_id",
        "item_id",
        "position",
        "category",
        "question",
        "reference",
        "answer_a",
        "answer_b",
        "judgement",
    }
    assert set(evaluation.ReviewPair.model_fields) == allowed

    def review(tuned_wins: bool) -> None:
        seen = 0
        while True:
            queue = evaluation.review_queue(
                evaluation.ReviewRequest(home=str(home), evaluation_id=eval_id)
            )
            if queue.pair is None:
                break
            payload = json.loads(queue.model_dump_json())
            assert set(payload["pair"]) == allowed
            assert (
                "tuned" not in payload["pair"].values() and "base" not in payload["pair"].values()
            )
            pair = queue.pair
            a_is_tuned = "omega" in pair.answer_a
            assert a_is_tuned != ("omega" in pair.answer_b)
            pick = "A" if a_is_tuned == tuned_wins else "B"
            result = evaluation.judge(
                evaluation.JudgeRequest(
                    home=str(home), evaluation_id=eval_id, item_id=pair.item_id, judgement=pick
                )
            )
            seen += 1
            assert result.judged == seen
        assert seen == 14

    review(tuned_wins=True)
    # Both orders occur, so the mapping is really hidden per item.
    with connect(AppPaths(home).db) as conn:
        assert {i["a_model"] for i in repositories.list_eval_items(conn, eval_id)} == {
            "base",
            "tuned",
        }
    score = evaluation.score_evaluation(
        evaluation.ScoreRequest(home=str(home), evaluation_id=eval_id)
    )
    assert (score.wins, score.losses, score.ties) == (14, 0, 0)
    assert score.verdict == "accepted", score.reasons

    # Re-judge every pair the other way: the same items, now all base wins.
    with connect(AppPaths(home).db) as conn:
        conn.execute("DELETE FROM eval_judgements WHERE evaluation_id = ?", (eval_id,))
        conn.commit()
    review(tuned_wins=False)
    score = evaluation.score_evaluation(
        evaluation.ScoreRequest(home=str(home), evaluation_id=eval_id)
    )
    assert (score.wins, score.losses) == (0, 14)
    assert score.verdict == "rejected"

    # Revisiting an item shows its judgement; positions out of range are refused.
    first = evaluation.review_queue(
        evaluation.ReviewRequest(home=str(home), evaluation_id=eval_id, position=1)
    )
    assert first.pair is not None and first.pair.judgement in {"A", "B"}
    with pytest.raises(NotAvailableError):
        evaluation.review_queue(
            evaluation.ReviewRequest(home=str(home), evaluation_id=eval_id, position=99)
        )
    with pytest.raises(NotAvailableError, match="no item"):
        evaluation.judge(
            evaluation.JudgeRequest(
                home=str(home), evaluation_id=eval_id, item_id="qa_nope", judgement="A"
            )
        )


def _read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_sheet(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def test_review_sheet_import_and_scoring(home: Path, fake: FakeBackend, tmp_path: Path) -> None:
    run_id = make_run(home)
    outcome = run_evaluation(home, run_id, small_config(tmp_path))
    header, rows = _read_sheet(Path(outcome.review_sheet))
    assert header == evaluation.SHEET_COLUMNS
    assert not {"a_model", "model", "base", "tuned"} & set(header)
    assert len(rows) == 14 and all(r["judgement"] == "" for r in rows)
    # formula-like answers are escaped for spreadsheets
    assert any(
        r["answer_a"].startswith("'=omega") or r["answer_b"].startswith("'=omega") for r in rows
    )

    for row in rows:  # the fine-tuned answer wins every pair
        row["judgement"] = "1" if "omega" in row["answer_a"] else "b"
    rows[0]["judgement"] = "tie"
    rows[1]["judgement"] = "maybe"
    rows[2]["judgement"] = ""
    extra = dict(rows[3], item_id="qa_unknown", judgement="A")
    filled = tmp_path / "filled.csv"
    _write_sheet(filled, header, [*rows, extra])

    imported = evaluation.import_review_sheet(
        evaluation.SheetImportRequest(home=str(home), path=str(filled), run_id=run_id)
    )
    assert imported.rows == 15 and imported.blank == 1
    assert sum(imported.recorded.values()) == 12 and imported.recorded["tie"] == 1
    assert imported.skipped == {"invalid_judgement": 1, "unknown_item": 1}
    assert [p.row for p in imported.problems] == [3, 16]

    score = evaluation.score_evaluation(
        evaluation.ScoreRequest(home=str(home), evaluation_id=imported.evaluation_id)
    )
    assert (score.judged, score.wins, score.losses, score.ties) == (12, 11, 0, 1)
    assert score.verdict == "inconclusive"  # 12 of 14 reviewed
    metrics = json.loads((Path(score.eval_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["judgements_by_origin"] == {"csv": 12}

    with pytest.raises(ConfigError, match="not run"):
        evaluation.import_review_sheet(
            evaluation.SheetImportRequest(home=str(home), path=str(filled), run_id="run_other")
        )
    mixed = tmp_path / "mixed.csv"
    _write_sheet(mixed, header, [rows[0], dict(rows[1], evaluation_id="ev_other")])
    with pytest.raises(ConfigError, match="exactly one evaluation"):
        evaluation.import_review_sheet(
            evaluation.SheetImportRequest(home=str(home), path=str(mixed))
        )
    missing = tmp_path / "missing.csv"
    _write_sheet(missing, ["evaluation_id", "item_id"], [])
    with pytest.raises(ConfigError, match="missing column"):
        evaluation.import_review_sheet(
            evaluation.SheetImportRequest(home=str(home), path=str(missing))
        )


def test_only_finished_runs_can_be_evaluated(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home, status="cancelled")
    with pytest.raises(FlipTunedError, match="needs a run that finished training"):
        evaluation.start_evaluation(
            evaluation.EvaluateRequest(home=str(home), run_id=run_id), detach=False
        )
    with pytest.raises(NotAvailableError, match="No run"):
        evaluation.start_evaluation(
            evaluation.EvaluateRequest(home=str(home), run_id="run_missing"), detach=False
        )
    with connect(AppPaths(home).db) as conn:
        assert repositories.list_evaluations(conn) == []


def test_review_waits_for_generation_and_cancel_is_recorded(home: Path, fake: FakeBackend) -> None:
    run_id = make_run(home)
    start = evaluation.start_evaluation(
        evaluation.EvaluateRequest(home=str(home), run_id=run_id), detach=False
    )
    with pytest.raises(FlipTunedError, match="is queued"):
        evaluation.review_queue(
            evaluation.ReviewRequest(home=str(home), evaluation_id=start.evaluation_id)
        )
    cancel = job_dir(AppPaths(home), start.job.id) / CANCEL_FILE
    fake.on_call = lambda n: cancel.write_text("1", encoding="utf-8") if n == 3 else None
    info = jobs_service.run_inline(str(home), start.job.id)
    assert info.status == "cancelled"
    assert len(fake.calls) == 3
    assert evaluation.get_evaluation(str(home), start.evaluation_id).status == "cancelled"
    with pytest.raises(FlipTunedError, match="is cancelled"):
        evaluation.score_evaluation(
            evaluation.ScoreRequest(home=str(home), evaluation_id=start.evaluation_id)
        )


def test_regression_set_can_be_switched_off(home: Path, fake: FakeBackend, tmp_path: Path) -> None:
    run_id = make_run(home)
    config = tmp_path / "noreg.yaml"
    config.write_text("kind: evaluate\ninclude_regression_set: false\n", encoding="utf-8")
    outcome = run_evaluation(home, run_id, str(config), limit=3)
    assert outcome.regression_items == 0 and outcome.test_items == 3
    assert len(fake.calls) == 6
    regression = next(c for c in outcome.score.criteria if c.name == "regression_drop")
    assert not regression.applicable
    assert any("--limit" in line for line in outcome.score.limitations)


# --------------------------------------------------------------------------- CLI


def test_cli_evaluate_and_score(home: Path, fake: FakeBackend, tmp_path: Path) -> None:
    run_id = make_run(home)
    base = ["evaluate", "--home", str(home)]
    missing = runner.invoke(app, base)
    assert missing.exit_code == 2 and "--run is required" in missing.output
    mixed = runner.invoke(app, [*base, "--run", run_id, "--dataset", "fixtures"])
    assert mixed.exit_code == 2 and "only apply" in mixed.output

    started = runner.invoke(app, [*base, "--run", run_id, "--config", small_config(tmp_path)])
    assert started.exit_code == 0, started.output
    assert "Verdict        INCONCLUSIVE" in started.output and "omega" not in started.output
    sheet = next(Path(home).glob("runs/*/evaluation/*/review_sheet.csv"))
    header, rows = _read_sheet(sheet)
    for row in rows:
        row["judgement"] = "A" if "omega" in row["answer_a"] else "B"
    _write_sheet(sheet, header, rows)

    scored = runner.invoke(app, [*base, "--run", run_id, "--score", str(sheet)])
    assert scored.exit_code == 0, scored.output
    assert "recorded" in scored.output and "Verdict        ACCEPTED" in scored.output
    assert "omega" not in scored.output and "alpha" not in scored.output
    as_json = runner.invoke(app, [*base, "--run", run_id, "--score", str(sheet), "--json"])
    assert json.loads(as_json.output)["score"]["verdict"] == "accepted"
