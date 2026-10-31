# Architecture

How FlipTuned is put together.

## Layers

```text
Web UI (React, browser) ──HTTP──▶ API (FastAPI) ──┐
CLI (Typer) ───────────────────────────────────────┼──▶ services ──▶ pipeline modules + training backends
                                                   │        ├──▶ registry (SQLite)
                                                   │        └──▶ job runner ──▶ worker processes
```

- **`services/`** holds all business logic. Each service takes a typed (Pydantic) request and returns
  a typed result; it never prints and never sees CLI arguments or HTTP objects.
- **`cli.py`** and **`api/`** are thin adapters: parse input, call a service, render the result. A
  test (`tests/unit/test_layering.py`) fails if they import anything but services and schemas.
- **The API schema has one source.** The service models generate OpenAPI (`web/openapi.json`), and
  the UI's TypeScript client is generated from it (`npm --prefix web run gen:api`); a test fails when
  the two drift apart.
- **The UI's forms come from the config models.** Field descriptions, defaults and validation are
  read from their JSON Schema.

## Modules (`src/fliptuned/`)

| Module | Role |
|---|---|
| `services/` | Everything the CLI and the API can do |
| `api/` | FastAPI app, routes, security middleware (Host check, token, CSP), SSE |
| `registry/` | SQLite schema, migrations, repositories (data sources, datasets, runs, jobs, evaluations, exports) |
| `jobs/` | Job runner, worker entry point, model lock, cancel markers, reconciliation |
| `platform/` | All OS-specific code: detection, detached processes, liveness, stop, keep-awake, paths |
| `config/` | Config models (ingest, dataset, train, evaluate, export) and presets |
| `ingestion/`, `records/`, `privacy/` | Reading CSV/XLSX/MD/TXT into records; quality flags; the secret scanner |
| `dataset/` | Dedup, grouping, split, token-length check, manifest |
| `models/` | The model catalog, Hugging Face cache access (offline by default), tokenizers |
| `training/` | The `TrainingBackend` interface, the `hf_peft` and `mlx` backends, chat-template tests |
| `evaluation/` | Generation, automatic checks, blind A/B mapping, sign test, verdict, report |
| `inference/` | The chat path (model weights only) |
| `export/` | Adapter export and merge |
| `diagnostics/` | `doctor` and backend auto-selection |
| `web_dist/` | The built UI, served by `serve` (build output, not in git) |

## Backends

Training, evaluation, chat and export go through `TrainingBackend`
(`training/backends/base.py`): capabilities, preflight, data preparation, training, loading for
inference, generation and merge. `hf_peft` uses Transformers + PEFT + TRL in a worker process (CUDA
or CPU); `mlx` generates an `mlx_lm.lora` config and runs it as a subprocess (Apple Silicon). Nothing
else calls those libraries directly.

## Jobs

Ingest, dataset builds, model downloads, training, evaluation, chat sessions and merges run as jobs:

- A job is a row in the registry plus a folder `<app home>/jobs/<job_id>/` with `events.jsonl`
  (status, progress, metrics, warnings: never record text).
- Each job runs in a **detached worker process**, so it survives closing the terminal, the browser
  or `fliptuned serve`.
- Liveness is checked by PID **and** process create time, so a reused PID isn't mistaken for the job.
  A job whose process is gone becomes `interrupted` on the next look (`jobs list`, `serve` start).
- **Cancel** writes a `cancel.requested` marker; `hf_peft` checks it at every step and stops after
  saving a checkpoint. The `mlx` subprocess is terminated, then killed.
- A **model lock** lets one model-loading job (train, evaluate, chat, merge) run at a time; others
  are refused with the holder's job ID (HTTP 409 in the API).
- The UI follows a job over Server-Sent Events read from `events.jsonl` (byte offsets as event IDs,
  so a reconnect continues where it stopped). The run monitor rebuilds a run's state from every
  attempt's events, so a reload or a `serve` restart shows the same run.

## Storage (the app home)

`~/.fliptuned` by default, `--home` or `FLIPTUNED_HOME` to change it, `./.dev-home` for development:

| Path | Holds |
|---|---|
| `fliptuned.db` | The SQLite registry |
| `sources/` | Ingested records per data source |
| `uploads/` | Files uploaded in the UI |
| `datasets/` | Built datasets (train/valid/test, manifest) |
| `runs/<run_id>/` | Config, adapter, checkpoints, metadata, evaluations, exports |
| `baselines/` | Base-model answers reused across evaluations |
| `configs/` | Saved, versioned training configs |
| `jobs/<job_id>/` | Job events and logs |
| `logs/` | Application logs (no record text) |

## The web server

`fliptuned serve` binds 127.0.0.1 only (the requested port or the next free one), rejects any `Host`
header other than `127.0.0.1:<port>` / `localhost:<port>`, requires the per-launch token on every API
call (`X-FlipTuned-Token`, from the `#token=` link, kept in the tab's `sessionStorage`), sends no CORS
headers and sets `Content-Security-Policy: default-src 'self'`. Folder access is limited to the
allowed folders; traversal, symlink escapes and UNC paths are refused. Errors never include record
content or secrets.

## No retrieval

The chat path loads the base model and adapter, applies the tokenizer's chat template to the fixed
system prompt and the conversation, generates, and returns. Two tests guard it: a static import scan
of `inference/` and a runtime file-access hook proving a chat call opens no data or dataset file.
