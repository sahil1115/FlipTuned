# Local API

`fliptuned serve` runs a FastAPI app on **127.0.0.1** only. The routes are thin adapters over the
same services the CLI uses (`src/fliptuned/api/routes/`). The schema is in `web/openapi.json`, and
the TypeScript types generated from it are in `web/src/api/schema.d.ts`.

## Starting it

```text
uv run fliptuned serve --home ./.dev-home              # opens the browser
uv run fliptuned serve --home ./.dev-home --no-browser --port 8765
```

- **Port:** 8765, or the next free port if it's taken (up to 20 tries). There is no option to
  listen on another interface.
- **On startup:** jobs whose process is gone are marked `interrupted`.
- **Access token:** each launch makes a new random token. The browser opens
  `http://127.0.0.1:<port>/#token=<token>`. The page keeps the token in `sessionStorage` and sends
  it as the `X-FlipTuned-Token` header. The fragment never reaches the server.
- **No access log:** Uvicorn's access log is off.

## Security (SPEC §13.4)

| Rule | How |
|---|---|
| Loopback only | The socket is bound to `127.0.0.1` before Uvicorn starts |
| DNS rebinding | Requests whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>` get 400 |
| Auth | Every `/api` route depends on `get_principal()`: a missing or wrong token gets 401 (also for unknown paths) |
| Cross-origin | No CORS headers at all |
| Content | Every response has `Content-Security-Policy: default-src 'self'; frame-ancestors 'none'; base-uri 'none'`, `nosniff`, `no-referrer`, `no-store` |
| Errors | `{"error": {"message", "hint", "details", "code"}}`. Validation errors list field locations only, never submitted values. Unexpected errors have no message |
| Folders | Browsing, scanning and ingest sources must resolve (following links and junctions) inside an allowed folder (Settings) or `<app_home>/uploads`. `..`, UNC/network and device paths are refused. The app home can't be an allowed folder |
| Uploads | Raw request body, `.csv/.xlsx/.md/.txt` only, at most `upload_max_mb` (default 50), saved under `<app_home>/uploads/<source_id>/`, never overwritten without `overwrite=true` |
| App home | Always the server's (`--home` / `FLIPTUNED_HOME`); a `home` field in a request body is ignored |
| No docs pages | `/docs` and `/redoc` are off (they load CDN assets). The schema is at `/api/openapi.json`, behind the token |

## Routes

| Area | Routes |
|---|---|
| System | `GET /api/doctor`, `GET /api/status`, `GET/PUT /api/settings` |
| Files | `GET /api/fs/browse?path=`, `GET /api/sources`, `POST /api/sources/{source}/uploads?filename=` |
| Data | `POST /api/scan`, `POST /api/ingest` (job), `GET /api/audit`, `GET /api/datasets/{id}/audit` |
| Review | `POST /api/records/search`, `POST /api/records/decisions`, `POST /api/review/export` (CSV), `POST /api/review/import` (CSV body) |
| Datasets | `POST /api/datasets` (job), `GET /api/datasets`, `POST /api/datasets/{id}/validate` |
| Models | `GET /api/models`, `POST /api/models/{id}/download` (job) |
| Configs | `GET/POST /api/configs`, `GET /api/configs/{name}`, `POST /api/configs/{name}/duplicate`, `GET /api/configs/schema/{kind}`, `GET /api/presets` |
| Training | `POST /api/preflight`, `POST /api/runs` (job), `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/monitor`, `POST /api/runs/{id}/resume` |
| Jobs | `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/selftest`, `GET /api/jobs/{id}/events` (SSE) |
| Evaluation | `POST/GET /api/runs/{id}/evaluations`, `GET /api/evaluations/{id}`, `GET /api/evaluations/{id}/pairs?position=`, `POST /api/evaluations/{id}/judgements`, `POST /api/evaluations/{id}/score`, `POST /api/evaluations/{id}/review-sheet` (CSV body) |
| Chat | `POST /api/chat/sessions`, `POST /api/chat/sessions/{job_id}/messages` (SSE), `DELETE /api/chat/sessions/{job_id}` |
| Export | `POST /api/runs/{id}/exports`, `GET /api/exports` |
| History | `GET /api/history?status=&model_id=&text=&include_archived=`, `GET /api/compare-runs?a=&b=`, `GET /api/runs/{id}/artifacts`, `GET /api/runs/{id}/artifacts/file?path=` (text files only, inside the run folder), `POST /api/runs/{id}/archive` (`{"archived": true/false}`), `POST /api/runs/{id}/delete` (`{"confirm": "<run id>"}`) |

Configs from the UI are sent inline (or, for training, as a saved config name), never as a file
path on the server. The server writes them to `<app_home>/configs/generated/`, so every job keeps
the exact file it ran with.

Jobs started through the API run in detached worker processes. The web server never loads a
model: chat uses a chat worker, and the base-model-only baseline isn't offered (evaluations make
their own baselines).

## Server-sent events

- **Job events.** `GET /api/jobs/{id}/events` streams the job's `events.jsonl`, the same file
  `fliptuned jobs follow` reads. Each event's SSE `event` name is its type: `status`,
  `progress`, `train`, `log` or `chat`. The last event of each batch carries an `id`: the byte
  offset into `events.jsonl`. Reconnecting with `Last-Event-ID` resumes from there. The stream
  ends after the job's final status, with keep-alive comments every 15 s meanwhile.
- **Run monitor.** `GET /api/runs/{id}/monitor` returns a training run's state rebuilt from the
  registry and every attempt's `events.jsonl` (loss curve across resumes, best checkpoint, step and
  ETA, memory, warnings, a content-free log, Cancel/Resume availability and the model-lock holder),
  plus `offset` and `last_seq` for the latest attempt. The UI then streams
  `/api/jobs/{job.id}/events` with `Last-Event-ID: <offset>` and applies each event with the same
  rules (`services/monitor.py` and `web/src/lib/monitor.ts`); when the stream drops it reconnects
  from the last event ID and skips events it has already seen (by `seq`). A 401 or 404 is not
  retried.
- **Chat replies.** `POST /api/chat/sessions/{job_id}/messages` streams `delta` events
  (`{"text": ...}`), then `done`, or `error` (`{"message": ...}`, which never quotes the
  conversation).
- **Headers.** The token is a header, so the browser reads these streams with `fetch`, not
  `EventSource`.

## Contract check

`npm --prefix web run gen:api` exports `web/openapi.json` and the training config's JSON Schema
(`web/src/test/train-schema.json`, which the UI tests render), then regenerates
`web/src/api/schema.d.ts`. Checks catch a stale copy:

- `npm --prefix web run check:api` (run by CI) fails if either file changed.
- The API tests compare `web/openapi.json` and `train-schema.json` with the app, and regenerate
  the types when Node is available.
