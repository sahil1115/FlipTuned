-- Evaluations of a run (Phase 7). One eval_items row per test item: its review position, its
-- category, a hash of its question and reference answer (to spot test items seen in earlier
-- evaluations) and the hidden A/B mapping (a_model = the model shown as answer A). The mapping
-- never leaves the server before scoring.
ALTER TABLE evaluations ADD COLUMN job_id TEXT;
ALTER TABLE evaluations ADD COLUMN config_json TEXT;

CREATE TABLE eval_items (
    workspace_id  TEXT NOT NULL DEFAULT 'local',
    created_by    TEXT NOT NULL DEFAULT 'local-user',
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
    item_id       TEXT NOT NULL,
    position      INTEGER NOT NULL,
    category      TEXT,
    content_hash  TEXT NOT NULL,
    a_model       TEXT NOT NULL CHECK (a_model IN ('base', 'tuned')),
    PRIMARY KEY (evaluation_id, item_id),
    UNIQUE (evaluation_id, position)
);
CREATE INDEX idx_eval_items_hash ON eval_items (content_hash);
CREATE INDEX idx_evaluations_run ON evaluations (workspace_id, run_id);
