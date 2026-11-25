-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/ambient/control.py
-- DDL block
CREATE TABLE IF NOT EXISTS ambient_runtime_control (
                id INTEGER PRIMARY KEY CHECK(id=1),paused INTEGER NOT NULL CHECK(paused IN (0,1)),
                revision INTEGER NOT NULL CHECK(revision>=0))
