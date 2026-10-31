-- FlipTuned registry schema v1. Timestamps are ISO-8601 UTC strings.
-- Every row carries workspace_id and created_by (SPEC §13.5 seams).

CREATE TABLE settings (
    workspace_id TEXT NOT NULL DEFAULT 'local',
    key          TEXT NOT NULL,
    value_json   TEXT NOT NULL,
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (workspace_id, key)
);

CREATE TABLE data_sources (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'local',
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('folder', 'upload')),
    root_path    TEXT NOT NULL,
    config_json  TEXT,
    status       TEXT NOT NULL DEFAULT 'registered',
    stats_json   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    archived_at  TEXT
);

CREATE TABLE datasets (
    id               TEXT PRIMARY KEY,
    workspace_id     TEXT NOT NULL DEFAULT 'local',
    created_by       TEXT NOT NULL DEFAULT 'local-user',
    name             TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'building',
    path             TEXT NOT NULL,
    config_json      TEXT NOT NULL,
    manifest_hash    TEXT,
    fingerprint      TEXT,
    stats_json       TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    archived_at      TEXT
);

CREATE TABLE review_decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL DEFAULT 'local',
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    source_id    TEXT NOT NULL REFERENCES data_sources(id),
    record_id    TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('approved', 'pending_review', 'rejected', 'quarantined')),
    reason       TEXT,
    origin       TEXT NOT NULL DEFAULT 'ui' CHECK (origin IN ('ui', 'csv', 'cli', 'bulk')),
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_review_decisions_record ON review_decisions (source_id, record_id, id);

CREATE TABLE training_configs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL DEFAULT 'local',
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    name         TEXT NOT NULL,
    version      INTEGER NOT NULL,
    config_json  TEXT NOT NULL,
    parent_id    INTEGER REFERENCES training_configs(id),
    created_at   TEXT NOT NULL,
    archived_at  TEXT,
    UNIQUE (workspace_id, name, version)
);

CREATE TABLE runs (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT 'local',
    created_by      TEXT NOT NULL DEFAULT 'local-user',
    dataset_id      TEXT REFERENCES datasets(id),
    config_id       INTEGER REFERENCES training_configs(id),
    model_id        TEXT NOT NULL,
    backend         TEXT NOT NULL,
    device          TEXT NOT NULL,
    precision       TEXT NOT NULL,
    artifact_format TEXT NOT NULL,
    status          TEXT NOT NULL,
    run_dir         TEXT NOT NULL,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    archived_at     TEXT
);

CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT 'local',
    created_by      TEXT NOT NULL DEFAULT 'local-user',
    type            TEXT NOT NULL CHECK (type IN (
                        'ingest', 'build_dataset', 'model_download', 'train',
                        'evaluate', 'merge_export', 'chat_session')),
    status          TEXT NOT NULL CHECK (status IN (
                        'queued', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted')),
    pid             INTEGER,
    pid_create_time REAL,
    params_json     TEXT,
    progress        REAL,
    log_path        TEXT,
    events_path     TEXT,
    error_summary   TEXT,
    run_id          TEXT REFERENCES runs(id),
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);
CREATE INDEX idx_jobs_status ON jobs (status);

-- Single-row global model lock (SPEC §13.3).
CREATE TABLE model_lock (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    workspace_id    TEXT NOT NULL DEFAULT 'local',
    created_by      TEXT NOT NULL DEFAULT 'local-user',
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    pid             INTEGER NOT NULL,
    pid_create_time REAL NOT NULL,
    acquired_at     TEXT NOT NULL
);

CREATE TABLE evaluations (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'local',
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    run_id       TEXT NOT NULL REFERENCES runs(id),
    status       TEXT NOT NULL,
    eval_dir     TEXT NOT NULL,
    verdict      TEXT CHECK (verdict IN ('accepted', 'rejected', 'inconclusive')),
    metrics_json TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    archived_at  TEXT
);

CREATE TABLE eval_judgements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id  TEXT NOT NULL DEFAULT 'local',
    created_by    TEXT NOT NULL DEFAULT 'local-user',
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    item_id       TEXT NOT NULL,
    judgement     TEXT NOT NULL CHECK (judgement IN ('A', 'B', 'tie')),
    origin        TEXT NOT NULL CHECK (origin IN ('ui', 'csv')),
    created_at    TEXT NOT NULL,
    UNIQUE (evaluation_id, item_id)
);

CREATE TABLE exports (
    id           TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'local',
    created_by   TEXT NOT NULL DEFAULT 'local-user',
    run_id       TEXT NOT NULL REFERENCES runs(id),
    kind         TEXT NOT NULL CHECK (kind IN ('adapter', 'merged')),
    status       TEXT NOT NULL,
    path         TEXT NOT NULL,
    metadata_json TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    archived_at  TEXT
);
