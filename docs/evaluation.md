# Evaluation

How FlipTuned decides whether fine-tuning helped (SPEC §12). Code: `src/fliptuned/evaluation/`
(checks, blinding, sign test, verdict, metrics, report) and `src/fliptuned/services/evaluation.py`.

## What runs

`fliptuned evaluate --run <run_id>` starts an `evaluate` job. It holds the model lock, so no other
model-loading job can run at the same time. It keeps the machine awake while it runs.

1. **Base model**, through the run's backend, device and precision, answers every test item. If a
   baseline with exactly the same dataset, model revision, backend, device, precision, generation
   settings and `--limit` already exists, its answers are reused. Otherwise they are generated and
   saved as a new baseline, so the next evaluation can reuse them.
2. The base model also answers the **general-regression set**: 20 bundled synthetic coding prompts
   (`src/fliptuned/evaluation/regression_set.yaml`) across Python, JavaScript, TypeScript, SQL,
   Bash, JSON, Go, Rust, Java and one no-code question.
3. The base model is released, and the **fine-tuned model** (the base model plus the run's best
   adapter) answers both sets.
4. Both models get the run's system prompt, byte for byte, and greedy decoding. Nothing else goes
   into a prompt.
5. Every answer gets the deterministic checks, and every test item gets a hidden A/B order.
6. A first scoring runs. With no reviews yet, the verdict is `inconclusive`.

Cancel works like any job (`fliptuned jobs cancel <job_id>`). The evaluation is then marked
`cancelled`, and any baseline it already finished is kept.

`--limit N` evaluates only the first N test items. Use it for a quick look only: the report says the
set was cut short, and a verdict you rely on needs the full test set.

## Deterministic checks (no code is run)

| Check | Applies when | Passes when |
|---|---|---|
| `non_empty` | always | the answer has non-whitespace text |
| `not_truncated` | always | generation stopped before the token limit |
| `no_repetition` | always | no line makes up most of an answer of 8+ lines, and the answer doesn't end by repeating a short phrase 6+ times |
| `has_code_block` | the reference answer (test set) or the item (regression set) has code | the answer has a fenced code block |
| `code_language` | as above, and the language is known | a code block carries that language tag (aliases such as `py` → `python` are normalised) |
| `fences_closed` | the answer has a code block | every block is closed |
| `syntax` | the answer has a Python or JSON block | every such block parses (`ast.parse`, `json.loads`); other languages aren't checked |
| `required_patterns` / `forbidden_patterns` | regression-set items that list them | case-insensitive regexes, searched in the code blocks when code is expected, otherwise in the whole answer |

An item **passes** when every check that applies to it passes. The pass rate is the share of
passing items. The verdict compares the base and fine-tuned pass rates, separately for the test set
and for the regression set.

## Blind review

For each test item, a seeded coin flip (`review_seed` + evaluation ID + item ID) decides which
model's answer is shown as **A**. The mapping is stored only in the registry (`eval_items.a_model`).
The review sheet and the review payloads (`review_queue`, used by the UI later) contain A and B,
the question and the approved reference answer, and nothing that identifies the models. The
mapping is read back only when scoring.

There are two ways to give judgements. Both use one judgement service and store judgements in the
registry, and a later judgement for an item replaces an earlier one:

- **Review sheet.** Fill the `judgement` column of `review_sheet.csv` (in the evaluation folder) with
  `A`, `B` or `tie`. `1`, `2` and `3` also work. Then run
  `fliptuned evaluate --run <run_id> --score <sheet>`. Only `evaluation_id`, `item_id` and
  `judgement` are read. Blank rows count as not reviewed yet. Invalid words and unknown items are
  reported by spreadsheet row. Cells that start with `=`, `+` or `@` get a leading `'`, so
  spreadsheet apps don't run them as formulas.
- **One pair at a time** (the UI). It always shows the next unreviewed pair, so you can
  stop and come back later.

`results.jsonl` in the evaluation folder is *not* blind: it lists each model's answers by name.
Review from the sheet or the UI, not from that file.

## Verdict (SPEC §12.3)

Applied in order, with the limits from the evaluate config (`configs/evaluate.sample.yaml` holds
the defaults):

1. **inconclusive** if there are fewer than `min_test_items` test items, review coverage is below
   `min_review_coverage`, or fewer than `min_non_tied_pairs` reviews picked a winner;
2. **rejected** if any of these holds:
   - the fine-tuned model doesn't win more pairs than it loses;
   - the exact two-sided sign-test p-value is ≥ `sign_test_alpha`;
   - in any category with at least 5 reviewed items, (losses − wins) / reviewed items is above
     `max_category_net_loss_share`;
   - the test-set pass rate drops by more than `max_deterministic_drop_pts`;
   - the regression-set pass rate drops by more than `max_regression_set_drop_pts`;
3. **accepted** otherwise.

The **sign test** treats each non-tied pair as a coin flip under the null hypothesis:
p = min(1, 2 · P(X ≤ min(wins, losses))) with X ~ Binomial(wins + losses, ½), computed with exact
integers. Ties are left out.

Every criterion is evaluated and reported, including those that didn't decide the verdict.
Falling training loss never counts as success.

## Re-using a test set

If any test items were already in an earlier evaluation, the report and the verdict say how many
items were shared, and with which evaluations. If anything (data, review decisions, settings) was
changed after looking at those earlier results, the new evaluation is not independent evidence. The
test set should never be used to repair data or tune settings.

## Artifacts

`<app_home>/runs/<run_id>/evaluation/<eval_id>/`:

| File | Contents |
|---|---|
| `results.jsonl` | per item: question, reference, both answers, token counts, finish reason, check results (holds content; stays local) |
| `metadata.json` | model repo, revision and template checksum, adapter, backend, device, precision, package versions, generation settings, baseline used, config |
| `review_sheet.csv` | the blind A/B sheet |
| `metrics.json` | counts, pass rates per check, per category results, regressions (item IDs), generation stats, prior exposure |
| `verdict.json` | verdict, reasons, every criterion with value and limit, inputs, limitations |
| `report.md` | all of the above, readable; statistics, item IDs and category labels only |

Scoring again (for example after more reviews) rewrites `metrics.json`, `verdict.json` and
`report.md`.

## Limits of this evaluation

- A test set of fewer than 50 items can't give a verdict with the default criteria (the fixtures
  have 14–15).
- Checks look at an answer's shape. Correctness comes from your blind review.
- Generated code is never run in v1 (sandboxed execution is planned for v2).
