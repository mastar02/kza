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


_VOICE_REVIEW_LEGACY_COLUMNS = [
    ('fingerprint_id', 'TEXT', 0, 1),
    ('audio_wav', 'BLOB', 1, 0),
    ('transcript', 'TEXT', 1, 0),
    ('room_id', 'TEXT', 0, 0),
    ('duration_s', 'REAL', 1, 0),
    ('clip_duration_s', 'REAL', 1, 0),
]
_VOICE_REVIEW_COLUMNS = _VOICE_REVIEW_LEGACY_COLUMNS + [
    ('audio_metadata_json', 'TEXT', 0, 0),
    ('transcript_metadata_json', 'TEXT', 0, 0),
]
_VOICE_REVIEW_FOREIGN_KEYS = [
    (0, 0, 'voice_fingerprints', 'fingerprint_id', 'id', 'NO ACTION', 'CASCADE', 'NONE'),
]


def initialize_voice_review_schema(db: sqlite3.Connection) -> None:
    """Atomically add optional provenance, without reconstructing old captures.

    This runs before generic schema creation so an incomplete versioned restore
    cannot be repaired into apparently valid unknown metadata. An existing
    six-column table is the one supported pre-feature shape; all its rows keep
    SQL NULL in both new columns. No payload is read or rewritten.
    """
    db.execute('BEGIN IMMEDIATE')
    try:
        present = dict(db.execute(
            "SELECT name,type FROM sqlite_master WHERE name IN "
            "('voice_capture_review_schema','voice_capture_reviews')"))
        if any(kind != 'table' for kind in present.values()):
            raise RuntimeError('voice_review_schema_unavailable')
        if not present:
            db.execute('''CREATE TABLE voice_capture_reviews (
                fingerprint_id TEXT PRIMARY KEY REFERENCES voice_fingerprints(id) ON DELETE CASCADE,
                audio_wav BLOB NOT NULL, transcript TEXT NOT NULL, room_id TEXT,
                duration_s REAL NOT NULL, clip_duration_s REAL NOT NULL,
                audio_metadata_json TEXT, transcript_metadata_json TEXT)''')
        elif present == {'voice_capture_reviews': 'table'}:
            columns = [(row[1], row[2], row[3], row[5])
                for row in db.execute('PRAGMA table_info(voice_capture_reviews)')]
            foreign_keys = [tuple(row)
                for row in db.execute('PRAGMA foreign_key_list(voice_capture_reviews)')]
            if columns != _VOICE_REVIEW_LEGACY_COLUMNS or foreign_keys != _VOICE_REVIEW_FOREIGN_KEYS:
                raise RuntimeError('voice_review_schema_unavailable')
            db.execute('ALTER TABLE voice_capture_reviews ADD COLUMN audio_metadata_json TEXT')
            db.execute('ALTER TABLE voice_capture_reviews ADD COLUMN transcript_metadata_json TEXT')
        elif present != {'voice_capture_reviews': 'table', 'voice_capture_review_schema': 'table'}:
            raise RuntimeError('voice_review_schema_unavailable')

        if 'voice_capture_review_schema' not in present:
            db.execute('CREATE TABLE voice_capture_review_schema (version INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO voice_capture_review_schema(version) VALUES (1)')
        columns = [(row[1], row[2], row[3], row[5])
            for row in db.execute('PRAGMA table_info(voice_capture_reviews)')]
        marker = [(row[1], row[2], row[3], row[5])
            for row in db.execute('PRAGMA table_info(voice_capture_review_schema)')]
        foreign_keys = [tuple(row)
            for row in db.execute('PRAGMA foreign_key_list(voice_capture_reviews)')]
        if (columns != _VOICE_REVIEW_COLUMNS or marker != [('version', 'INTEGER', 0, 1)]
                or foreign_keys != _VOICE_REVIEW_FOREIGN_KEYS
                or list(db.execute('PRAGMA foreign_key_list(voice_capture_review_schema)'))):
            raise RuntimeError('voice_review_schema_unavailable')
        versions = list(db.execute('SELECT version FROM voice_capture_review_schema LIMIT 2'))
        if len(versions) != 1 or type(versions[0][0]) is not int or versions[0][0] != 1:
            raise RuntimeError('voice_review_schema_unavailable')
        db.commit()
    except BaseException:
        db.rollback()
        raise
