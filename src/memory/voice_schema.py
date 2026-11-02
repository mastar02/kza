"""Private voice retention tables owned by the shared privacy authority."""

import sqlite3

VOICE_SCHEMA = '''
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
 duration_s REAL NOT NULL, clip_duration_s REAL NOT NULL
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
'''


def initialize_voice_people_schema(db: sqlite3.Connection) -> None:
    """Atomically migrate older ledgers; reject incomplete versioned restores."""
    db.execute('BEGIN IMMEDIATE')
    try:
        present = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('voice_person_schema','voice_profile_people')")}
        if not present:
            db.execute('''CREATE TABLE voice_profile_people (
                profile_id TEXT PRIMARY KEY REFERENCES voice_profiles(id) ON DELETE CASCADE,
                user_id TEXT, revision INTEGER NOT NULL CHECK(revision>=1),
                updated_at REAL NOT NULL, updated_by TEXT NOT NULL)''')
            db.execute('CREATE INDEX voice_profile_people_user ON voice_profile_people(user_id)')
            db.execute('CREATE TABLE voice_person_schema (version INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO voice_person_schema(version) VALUES (1)')
        elif (present != {'voice_person_schema','voice_profile_people'}
                or [tuple(row) for row in db.execute('SELECT version FROM voice_person_schema LIMIT 2')] != [(1,)]):
            raise RuntimeError('voice_person_schema_unavailable')
        columns = [(row[1], row[2], row[3], row[5])
            for row in db.execute('PRAGMA table_info(voice_profile_people)')]
        marker = [(row[1], row[2], row[3], row[5])
            for row in db.execute('PRAGMA table_info(voice_person_schema)')]
        if (columns != [('profile_id','TEXT',0,1), ('user_id','TEXT',0,0),
                ('revision','INTEGER',1,0), ('updated_at','REAL',1,0), ('updated_by','TEXT',1,0)]
                or marker != [('version','INTEGER',0,1)]):
            raise RuntimeError('voice_person_schema_unavailable')
        db.commit()
    except BaseException:
        db.rollback()
        raise
