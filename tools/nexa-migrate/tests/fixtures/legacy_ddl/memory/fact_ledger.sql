-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/memory/fact_ledger.py
-- DDL block

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY, content TEXT, category TEXT NOT NULL,
  confidence REAL NOT NULL, state TEXT NOT NULL,
  extracted_at REAL NOT NULL, expires_at REAL,
  revision INTEGER NOT NULL, confirmed_at REAL, confirmed_by TEXT
);
CREATE TABLE IF NOT EXISTS fact_sources (
  fact_id TEXT NOT NULL REFERENCES facts(id), capture_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL, captured_at REAL NOT NULL,
  expires_at REAL NOT NULL, origin TEXT NOT NULL, dependencies TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY(fact_id, capture_id)
);
CREATE TABLE IF NOT EXISTS fact_dependencies (
  fact_id TEXT NOT NULL REFERENCES facts(id), source_fact_id TEXT NOT NULL REFERENCES facts(id),
  PRIMARY KEY(fact_id, source_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_dependency_source ON fact_dependencies(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_conversation ON fact_sources(conversation_id);
CREATE INDEX IF NOT EXISTS idx_fact_expiry ON facts(expires_at);
CREATE TABLE IF NOT EXISTS conversation_tombstones (
  conversation_id TEXT PRIMARY KEY, deleted_at REAL NOT NULL, deleted_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_jobs (
  fact_id TEXT PRIMARY KEY REFERENCES facts(id), revision INTEGER NOT NULL,
  operation TEXT NOT NULL, queued_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);

-- DDL block

CREATE TABLE fact_owners (fact_id TEXT NOT NULL PRIMARY KEY REFERENCES facts(id), person_id TEXT NOT NULL, method TEXT NOT NULL, revision INTEGER NOT NULL, assigned_at REAL NOT NULL, assigned_by TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '{}');
CREATE INDEX fact_owners_person ON fact_owners(person_id);
CREATE TABLE fact_owner_revisions (fact_id TEXT NOT NULL PRIMARY KEY REFERENCES facts(id), revision INTEGER NOT NULL, last_change TEXT NOT NULL, confirmed_by_assignment INTEGER NOT NULL, changed_at REAL NOT NULL);
CREATE INDEX fact_owner_revisions_change ON fact_owner_revisions(last_change, changed_at);
CREATE TABLE fact_owner_schema (version INTEGER PRIMARY KEY);
INSERT INTO fact_owner_schema VALUES (1);

