-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/memory/voice_schema.py
-- DDL block

CREATE TABLE IF NOT EXISTS voice_fingerprints (
 id TEXT PRIMARY KEY, model_key TEXT NOT NULL, dimensions INTEGER NOT NULL,
 embedding BLOB NOT NULL, duration_s REAL NOT NULL, created_at REAL NOT NULL,
 expires_at REAL NOT NULL, sources_json TEXT NOT NULL,
 suggested_profile TEXT, suggested_revision INTEGER, similarity REAL
);
CREATE TABLE IF NOT EXISTS voice_fingerprint_sources (
 fingerprint_id TEXT NOT NULL REFERENCES voice_fingerprints(id) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL, PRIMARY KEY(fingerprint_id,conversation_id)
);
CREATE INDEX IF NOT EXISTS voice_fingerprint_expiry ON voice_fingerprints(expires_at);
CREATE INDEX IF NOT EXISTS voice_fingerprint_conversation ON voice_fingerprint_sources(conversation_id);
CREATE TABLE IF NOT EXISTS voice_capture_reviews (
 fingerprint_id TEXT PRIMARY KEY REFERENCES voice_fingerprints(id) ON DELETE CASCADE,
 audio_wav BLOB NOT NULL, transcript TEXT NOT NULL, room_id TEXT,
 duration_s REAL NOT NULL, clip_duration_s REAL NOT NULL,
 audio_metadata_json TEXT, transcript_metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS voice_profiles (
 id TEXT PRIMARY KEY, name TEXT, model_key TEXT NOT NULL, dimensions INTEGER NOT NULL,
 embedding BLOB, sources_json TEXT NOT NULL, confirmed_at REAL NOT NULL,
 confirmed_by TEXT NOT NULL, revision INTEGER NOT NULL, deleted_at REAL
);
CREATE TABLE IF NOT EXISTS voice_profile_sources (
 profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL, PRIMARY KEY(profile_id,conversation_id)
);
CREATE INDEX IF NOT EXISTS voice_profile_conversation ON voice_profile_sources(conversation_id);
CREATE TABLE IF NOT EXISTS voice_review_requests (
 actor TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
 profile_id TEXT NOT NULL, PRIMARY KEY(actor,request_id)
);
CREATE TABLE IF NOT EXISTS voice_groups (
 id TEXT PRIMARY KEY, revision INTEGER NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL,
 similarity REAL, created_at REAL NOT NULL, expires_at REAL NOT NULL, sources_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS voice_group_members (
 fingerprint_id TEXT PRIMARY KEY REFERENCES voice_fingerprints(id) ON DELETE CASCADE,
 group_id TEXT NOT NULL REFERENCES voice_groups(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS voice_group_members_group ON voice_group_members(group_id);
CREATE TABLE IF NOT EXISTS voice_group_requests (
 actor TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
 expires_at REAL NOT NULL, sources_json TEXT NOT NULL, PRIMARY KEY(actor,request_id)
);
CREATE TABLE IF NOT EXISTS voice_group_schema (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS voice_group_lineage (
 group_id TEXT PRIMARY KEY REFERENCES voice_groups(id) ON DELETE CASCADE,
 source_digest TEXT NOT NULL, member_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS voice_group_request_lineage (
 actor TEXT NOT NULL, request_id TEXT NOT NULL, source_digest TEXT NOT NULL,
 PRIMARY KEY(actor,request_id),
 FOREIGN KEY(actor,request_id) REFERENCES voice_group_requests(actor,request_id) ON DELETE CASCADE
);

-- DDL block
CREATE TABLE voice_profile_people (
                profile_id TEXT PRIMARY KEY REFERENCES voice_profiles(id) ON DELETE CASCADE,
                user_id TEXT, revision INTEGER NOT NULL CHECK(revision>=1),
                updated_at REAL NOT NULL, updated_by TEXT NOT NULL)
-- DDL block

CREATE TABLE voice_attribution_consents (person_id TEXT NOT NULL PRIMARY KEY, state TEXT NOT NULL, revision INTEGER NOT NULL, verified_profile_id TEXT REFERENCES voice_profiles(id), verified_profile_revision INTEGER, verified_link_revision INTEGER, changed_at REAL NOT NULL, changed_by TEXT NOT NULL);
CREATE TABLE voice_verification_attempts (person_id TEXT NOT NULL, attempted_at REAL NOT NULL, outcome TEXT NOT NULL, PRIMARY KEY(person_id, attempted_at));
CREATE TABLE voice_attribution_events (id TEXT NOT NULL PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES voice_profiles(id), kind TEXT NOT NULL, score_bucket INTEGER NOT NULL, at REAL NOT NULL);
CREATE INDEX voice_attribution_events_profile ON voice_attribution_events(profile_id, at);
CREATE TABLE voice_attribution_suspensions (profile_id TEXT NOT NULL PRIMARY KEY REFERENCES voice_profiles(id), revision INTEGER NOT NULL, suspended_at REAL NOT NULL, reason TEXT NOT NULL);
CREATE TABLE voice_attribution_control (id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, revision INTEGER NOT NULL, changed_at REAL NOT NULL, changed_by TEXT NOT NULL);
CREATE TABLE voice_attribution_schema (version INTEGER PRIMARY KEY);
INSERT INTO voice_attribution_schema VALUES (1);

