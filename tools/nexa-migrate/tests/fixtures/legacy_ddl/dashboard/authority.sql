-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/dashboard/authority.py
-- DDL block

CREATE TABLE IF NOT EXISTS auth_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS credentials (
 id TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, subject TEXT NOT NULL, role TEXT NOT NULL,
 user_id TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL,
 samples TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
 digest TEXT PRIMARY KEY, credential TEXT NOT NULL, legacy_digest TEXT,
 issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS retired_users (
 user_id TEXT PRIMARY KEY, retired_at REAL NOT NULL);

