-- Adds the 'selftest' job type (a diagnostic job that checks background jobs work on this OS).
-- SQLite can't change a CHECK constraint in place, so the table is rebuilt (same columns, same
-- order). model_lock references jobs(id); its row (if any) is set aside during the rebuild so
-- dropping the old table can't violate the foreign key.

CREATE TEMP TABLE model_lock_backup AS SELECT * FROM model_lock;
DELETE FROM model_lock;

CREATE TABLE jobs_new (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT 'local',
    created_by      TEXT NOT NULL DEFAULT 'local-user',
    type            TEXT NOT NULL CHECK (type IN (
                        'ingest', 'build_dataset', 'model_download', 'train',
                        'evaluate', 'merge_export', 'chat_session', 'selftest')),
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
INSERT INTO jobs_new SELECT * FROM jobs;
DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
CREATE INDEX idx_jobs_status ON jobs (status);

INSERT INTO model_lock SELECT * FROM model_lock_backup;
DROP TABLE model_lock_backup;
