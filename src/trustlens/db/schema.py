from __future__ import annotations

SCHEMA_SQL = """
-- Core run table: one row per analysis execution
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  claim_text TEXT NOT NULL,
  query_text TEXT,
  status TEXT NOT NULL,        -- e.g. started, completed, failed
  params_json TEXT,            -- JSON-serialized run parameters (portable across DBs)
  error_text TEXT              -- populated on failure
);

-- Evidence items retrieved for a run (GDELT later)
CREATE TABLE IF NOT EXISTS evidence_items (
  evidence_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  retrieved_at TIMESTAMP NOT NULL,
  published_at TIMESTAMP,      -- if available from source
  url TEXT NOT NULL,
  domain TEXT NOT NULL,
  title TEXT,
  source TEXT NOT NULL,        -- e.g. "gdelt_doc2" later
  raw_json TEXT,               -- JSON-serialized raw payload (portable)
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_run_id ON evidence_items(run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_domain ON evidence_items(domain);

-- Feature store: key-value features per run
CREATE TABLE IF NOT EXISTS features (
  feature_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  feature_group TEXT NOT NULL, -- e.g. "prior", "corroboration", "time"
  feature_name TEXT NOT NULL,
  feature_value DOUBLE NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_features_run_id ON features(run_id);
CREATE INDEX IF NOT EXISTS idx_features_name ON features(feature_name);

-- Model registry: trained model versions + metadata
CREATE TABLE IF NOT EXISTS model_versions (
  model_version_id TEXT PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  name TEXT NOT NULL,              -- human-friendly identifier
  model_type TEXT NOT NULL,         -- e.g. "logreg", "xgb"
  calibration_method TEXT,          -- e.g. "platt", "isotonic"
  artifact_uri TEXT,               -- S3 URI later (or local path during dev)
  params_json TEXT,                -- JSON-serialized training params
  metrics_json TEXT                -- JSON-serialized evaluation metrics
);

CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at);
"""
