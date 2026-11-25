-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/routines/routine_store.py
-- DDL block

CREATE TABLE IF NOT EXISTS routines (
    routine_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
    created_by TEXT NOT NULL, created_at TEXT NOT NULL, definition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routine_runs (
    run_id TEXT PRIMARY KEY, routine_id TEXT NOT NULL, occurrence_key TEXT NOT NULL,
    actor TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
    snapshot_json TEXT NOT NULL, scheduled_for TEXT,
    UNIQUE (routine_id, occurrence_key)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_routine
    ON routine_runs(routine_id) WHERE status = 'running';
CREATE TABLE IF NOT EXISTS routine_run_actions (
    run_id TEXT NOT NULL, action_index INTEGER NOT NULL, entity_id TEXT NOT NULL,
    status TEXT NOT NULL, reason TEXT, PRIMARY KEY (run_id, action_index),
    FOREIGN KEY (run_id) REFERENCES routine_runs(run_id)
);

