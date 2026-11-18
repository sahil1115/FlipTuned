# Privacy

FlipTuned is local-first: your Q&A, the datasets, training, evaluation, chat and the web UI stay on
your machine. This page lists what that means in practice.

## What leaves the machine

Only these, and only when you ask for them:

- **Package installs** (`uv sync`, `npm --prefix web ci`).
- **Model and tokenizer downloads** from Hugging Face: with `--allow-download` or a download button
  in the UI. Everything else runs offline and fails with a clear message if files are missing: FlipTuned
  forces `HF_HUB_OFFLINE=1` (also `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE`) even if your
  shell sets them differently. A Hugging Face account token saved on this machine is never
  attached to downloads (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`), so gated models can't be fetched.

There is **no telemetry**: `HF_HUB_DISABLE_TELEMETRY=1` is set, Transformers/TRL get
`report_to="none"`, mlx-lm's `--report-to` is never passed, and there are no hosted trackers, crash
reporters or remote judges. The web UI bundles all its assets: no CDNs, web fonts or analytics, and
its Content-Security-Policy blocks anything else.

## Secrets in your data

During ingest every record is scanned for likely secrets: API keys, tokens (including Hugging Face, GitHub, GitLab, Slack and Stripe formats and
`Bearer` headers), passwords, private keys, connection strings, credentials inside URLs, and
any extra patterns you add (`privacy.extra_patterns`). The policy
(`privacy.policy`) decides what happens:

| Policy | Effect |
|---|---|
| `exclude` (default) | Records with findings are left out of datasets |
| `warn` | Records stay; the findings are reported |
| `stop` | The ingest stops |

Reports, the Review screen and `data-audit` show the file, location and type of a finding, never
the value.

## Logs and outputs

- Logs and job events hold counts, metrics, statuses and messages, never Q&A or chat text. Content
  debug logging for an ingest needs an explicit opt-in (`debug_log_content: true`).
- CLI commands print counts and statistics only.
- Chat messages are never written to disk.
- Your Q&A *is* stored, by design, in the app home: ingested records, datasets, evaluation answers
  and review sheets. Treat the app home like the data itself. Deleting a run in the UI removes its
  adapter, checkpoints, evaluations, exports and job logs.
- To remove a data source completely (ingested records, uploaded files and review decisions):
  `fliptuned sources delete <name> --confirm <name>`. Datasets already built from it keep their
  own copies; rebuild or delete those separately.

## The web server

It listens on 127.0.0.1 only, so other computers can't reach it, and every API call needs the token
from the link `serve` prints (new for every launch). Details: `docs/architecture.md`.

## Your responsibilities

You need permission to use the data you fine-tune on and must follow the base model's license. A
fine-tuned model can reproduce fragments of its training data, so treat the adapter and any merged
model as being as sensitive as the Q&A it was trained on.

## Development

Development and tests use only the synthetic fixtures in `tests/fixtures/`, `./.dev-home` and
temporary folders, never a real app home.
