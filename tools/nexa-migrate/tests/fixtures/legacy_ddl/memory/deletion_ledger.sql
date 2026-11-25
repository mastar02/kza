-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/memory/deletion_ledger.py
-- DDL block

CREATE TABLE deletion_ledger_v1 (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, subject_sha256 TEXT NOT NULL, at REAL NOT NULL, prev_sha256 TEXT NOT NULL, row_sha256 TEXT NOT NULL, UNIQUE(kind, subject_sha256));
CREATE TABLE deletion_ledger_salt_v1 (id INTEGER PRIMARY KEY CHECK (id = 1), salt TEXT NOT NULL)

