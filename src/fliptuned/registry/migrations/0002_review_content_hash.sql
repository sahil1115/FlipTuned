-- Review decisions remember the content they were made on (re-ingest can move rows).
-- A decision whose content_hash no longer matches the record is ignored as stale.
ALTER TABLE review_decisions ADD COLUMN content_hash TEXT;

CREATE INDEX idx_datasets_name ON datasets (workspace_id, name, created_at);
