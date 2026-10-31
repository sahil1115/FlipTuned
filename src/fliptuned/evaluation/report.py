"""``report.md`` (SPEC §12.3–12.4): every input to the verdict, per-category results, regressions
and limitations. Statistics, item IDs and category labels only; no questions or answers."""

from __future__ import annotations

from typing import Any

from fliptuned.evaluation.checks import CHECKS
from fliptuned.evaluation.metrics import CheckSummary, EvaluationMetrics
from fliptuned.evaluation.verdict import CATEGORY_MIN_ITEMS, Verdict

_SIDE_LABELS = {"base": "base", "tuned": "fine-tuned"}


def _checks_table(summary: CheckSummary) -> list[str]:
    lines = [
        "| Check | Base passed | Fine-tuned passed |",
        "|---|---|---|",
        f"| **All checks on an item** | {summary.base.passed_items}/{summary.items} "
        f"({summary.base.pass_rate:.1f}%) | {summary.tuned.passed_items}/{summary.items} "
        f"({summary.tuned.pass_rate:.1f}%) |",
    ]
    for name in sorted(set(summary.base.checks) | set(summary.tuned.checks)):
        cells = []
        for side in (summary.base, summary.tuned):
            count = side.checks.get(name)
            cells.append(f"{count.passed}/{count.applicable}" if count else "—")
        lines.append(f"| {name}: {CHECKS.get(name, '')} | {cells[0]} | {cells[1]} |")
    return lines


def render_report(
    metrics: EvaluationMetrics,
    verdict: Verdict,
    meta: dict[str, Any],
    limitations: list[str],
) -> str:
    m = metrics
    out = [
        f"# Evaluation {m.evaluation_id}",
        "",
        f"- Run: `{m.run_id}`; dataset: `{m.dataset_id}` (test split)",
        f"- Model: `{meta['model']['catalog_id']}` ({meta['model']['repo']} @ "
        f"{meta['model']['revision'][:12]}) on {meta['backend']}/{meta['device']} "
        f"({meta['precision']})",
        f"- Decoding: greedy, at most {meta['generation']['max_new_tokens']} new tokens; "
        "the same system prompt for both models",
        f"- Base answers: baseline `{meta['baseline_id']}`"
        + (" (reused)" if meta.get("baseline_reused") else " (generated now)"),
        "",
        f"## Verdict: **{verdict.verdict.upper()}**",
        "",
    ]
    out += [f"- {r}" for r in verdict.reasons] or ["- Every criterion is met."]
    out += [
        "",
        "## Inputs to the verdict",
        "",
        "| Criterion | Value | Limit | Gives | Result |",
        "|---|---|---|---|---|",
    ]
    for c in verdict.criteria:
        result = "not applicable" if not c.applicable else ("met" if c.met else "**not met**")
        out.append(
            f"| {c.description} | {c.display_value} | {c.display_threshold} | {c.stage} | "
            f"{result} |"
        )
    out += [
        "",
        "## Blind review",
        "",
        f"- Test items: {m.test_items}; reviewed: {m.judged} ({m.review_coverage:.0%})",
        f"- Fine-tuned wins: {m.wins}; base wins: {m.losses}; ties: {m.ties}",
        f"- Exact two-sided sign test on {m.wins + m.losses} non-tied pairs: p = {m.sign_test_p:.4g}",
        "- Judgements by origin: "
        + (", ".join(f"{k} {v}" for k, v in m.judgements_by_origin.items()) or "none yet"),
        "",
        "## Per category",
        "",
        f"Categories with at least {CATEGORY_MIN_ITEMS} reviewed items count toward the verdict.",
        "",
        "| Category | Items | Reviewed | Fine-tuned wins | Base wins | Ties | (losses − wins) / reviewed |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat in m.categories:
        share = cat.net_loss_share
        out.append(
            f"| {cat.category} | {cat.items} | {cat.judged} | {cat.wins} | {cat.losses} | "
            f"{cat.ties} | {'—' if share is None else f'{share:.0%}'} |"
        )
    out += ["", "## Automatic checks (test set)", "", *_checks_table(m.deterministic), ""]
    out += ["## General-regression set", ""]
    if m.regression_set is None:
        out += ["Not run (`include_regression_set: false`).", ""]
    else:
        out += [*_checks_table(m.regression_set), ""]
    out += ["## Regressions", ""]
    if m.regressions:
        out += [
            "Items where the fine-tuned answer failed a check the base answer passed:",
            "",
            "| Set | Item | Checks |",
            "|---|---|---|",
        ]
        out += [
            f"| {r.set} | `{r.item_id}` | {', '.join(r.failed_checks)} |" for r in m.regressions
        ]
    else:
        out.append("None: the fine-tuned answers pass every check the base answers pass.")
    out += ["", "## Limitations", ""]
    out += [f"- {line}" for line in limitations]
    out += [
        "",
        "## Generation",
        "",
        "| Model | Items | Finish reasons | Tokens generated | Seconds |",
        "|---|---|---|---|---|",
    ]
    for side, stats in m.generation.items():
        reasons = ", ".join(f"{k} {v}" for k, v in stats.finish_reasons.items())
        out.append(
            f"| {_SIDE_LABELS[side]} | {stats.items} | {reasons} | {stats.completion_tokens} | "
            f"{stats.seconds:g} |"
        )
    return "\n".join(out) + "\n"
