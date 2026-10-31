-- Base-model results on a dataset's test set (Phase 4). Evaluations of a run (Phase 7) reuse a
-- baseline whose dataset, model revision, backend, device, precision and generation settings match.
CREATE TABLE baselines (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'local',
    created_by    TEXT NOT NULL DEFAULT 'local-user',
    dataset_id    TEXT NOT NULL REFERENCES datasets(id),
    model_id      TEXT NOT NULL,
    backend       TEXT NOT NULL,
    device        TEXT NOT NULL,
    precision     TEXT NOT NULL,
    status        TEXT NOT NULL,
    path          TEXT NOT NULL,
    metadata_json TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    archived_at   TEXT
);
CREATE INDEX idx_baselines_dataset ON baselines (workspace_id, dataset_id);
