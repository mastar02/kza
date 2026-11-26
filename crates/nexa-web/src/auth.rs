//! Authority DDL shared with the offline nexa-migrate schema reader.

pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS auth_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS credentials (
        id TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, subject TEXT NOT NULL,
        role TEXT NOT NULL, user_id TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL,
        revision INTEGER NOT NULL, samples TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        digest TEXT PRIMARY KEY, credential TEXT NOT NULL REFERENCES credentials(id) ON DELETE CASCADE,
        legacy_digest TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS nexa_sessions_credential ON sessions(credential);
    CREATE TABLE IF NOT EXISTS retired_users (user_id TEXT PRIMARY KEY, retired_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS nexa_accounts (
        id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE, permission_level TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS nexa_purge_pending (
        user_id TEXT PRIMARY KEY REFERENCES retired_users(user_id)
    );
    CREATE TABLE IF NOT EXISTS nexa_credential_attempts (
        request_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, intent TEXT NOT NULL, receipt TEXT NOT NULL
    );";
