"""Private voice retention tables owned by the shared privacy authority."""

import math
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


# U3b-4a-0: voice-attribution authority (design u3b-4 v4). The single DDL source
# for fresh and migrated ledgers, the admission registration and every
# rehearsal. No CHECK constraints: values are validated against the closed
# enums in src.memory.voice_attribution at open (and by the later writers).
# No row is ever written at open, so a derivation or restore of an aged state
# leaves every protected row unchanged (review R1, G1).
VOICE_ATTRIBUTION_SCHEMA = """
CREATE TABLE voice_attribution_consents (person_id TEXT NOT NULL PRIMARY KEY, state TEXT NOT NULL, revision INTEGER NOT NULL, verified_profile_id TEXT REFERENCES voice_profiles(id), verified_profile_revision INTEGER, verified_link_revision INTEGER, changed_at REAL NOT NULL, changed_by TEXT NOT NULL);
CREATE TABLE voice_verification_attempts (person_id TEXT NOT NULL, attempted_at REAL NOT NULL, outcome TEXT NOT NULL, PRIMARY KEY(person_id, attempted_at));
CREATE TABLE voice_attribution_events (id TEXT NOT NULL PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES voice_profiles(id), kind TEXT NOT NULL, score_bucket INTEGER NOT NULL, at REAL NOT NULL);
CREATE INDEX voice_attribution_events_profile ON voice_attribution_events(profile_id, at);
CREATE TABLE voice_attribution_suspensions (profile_id TEXT NOT NULL PRIMARY KEY REFERENCES voice_profiles(id), revision INTEGER NOT NULL, suspended_at REAL NOT NULL, reason TEXT NOT NULL);
CREATE TABLE voice_attribution_control (id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, revision INTEGER NOT NULL, changed_at REAL NOT NULL, changed_by TEXT NOT NULL);
CREATE TABLE voice_attribution_schema (version INTEGER PRIMARY KEY);
INSERT INTO voice_attribution_schema VALUES (1);
"""
VOICE_ATTRIBUTION_TABLES = ("voice_attribution_consents", "voice_verification_attempts",
                            "voice_attribution_events", "voice_attribution_suspensions",
                            "voice_attribution_control", "voice_attribution_schema")


def _attribution_statements() -> list[str]:
    return [part.strip() for part in VOICE_ATTRIBUTION_SCHEMA.split(";") if part.strip()]


def _attribution_objects(db: sqlite3.Connection) -> list[tuple]:
    placeholders = ",".join("?" for _ in VOICE_ATTRIBUTION_TABLES)
    return sorted(tuple(row) for row in db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN (" + placeholders + ") "
        "OR name IN (" + placeholders + ")", VOICE_ATTRIBUTION_TABLES * 2))


def _attribution_pragmas(db: sqlite3.Connection) -> dict:
    shape = {}
    for table in VOICE_ATTRIBUTION_TABLES:
        shape[table] = (
            [tuple(row) for row in db.execute("PRAGMA table_info(" + table + ")")],
            sorted((row[1], row[2], row[3], row[4],
                    tuple(info[2] for info in db.execute("PRAGMA index_info(" + row[1] + ")")))
                   for row in db.execute("PRAGMA index_list(" + table + ")")),
            [tuple(row) for row in db.execute("PRAGMA foreign_key_list(" + table + ")")])
    return shape


def _attribution_reference() -> tuple[list[tuple], dict]:
    """The exact objects and PRAGMA shape the DDL produces, built in memory."""
    reference = sqlite3.connect(":memory:")
    try:
        for statement in _attribution_statements():
            reference.execute(statement)
        return _attribution_objects(reference), _attribution_pragmas(reference)
    finally:
        reference.close()


# Built once at import: initialisation must open no connection besides the ledger's own.
_ATTRIBUTION_REFERENCE = _attribution_reference()


def voice_attribution_schema_objects() -> list[dict]:
    """The sqlite_master rows (type, name, tbl_name, sql) that VOICE_ATTRIBUTION_SCHEMA creates.

    SQLite stores CREATE statements verbatim, so these rows are the byte-exact
    DDL. The ledger initializer and the payload mixed-cut rule compare against them.
    """
    objects, _shape = _ATTRIBUTION_REFERENCE
    return [{"type": kind, "name": name, "tbl_name": table, "sql": sql} for kind, name, table, sql in objects]


def _identity(value: object) -> bool:
    return (type(value) is str and 1 <= len(value) <= 128
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _revision(value: object) -> bool:
    return type(value) is int and value >= 1


def _scan_voice_attribution_rows(db: sqlite3.Connection) -> None:
    """Validate every voice-attribution row; an unknown value fails closed (1a-F4 read point)."""
    from src.memory.voice_attribution import (
        VoiceAttributionEventKind,
        VoiceConsentState,
        VoiceSuspensionReason,
        VoiceVerificationOutcome,
    )
    states = {item.value for item in VoiceConsentState}
    outcomes = {item.value for item in VoiceVerificationOutcome}
    kinds = {item.value for item in VoiceAttributionEventKind}
    reasons = {item.value for item in VoiceSuspensionReason}
    valid = True
    for person, state, revision, profile, profile_revision, link_revision, changed_at, changed_by in db.execute(
            "SELECT person_id,state,revision,verified_profile_id,verified_profile_revision,"
            "verified_link_revision,changed_at,changed_by FROM voice_attribution_consents"):
        verified = (profile, profile_revision, link_revision)
        valid &= (_identity(person) and state in states and _revision(revision) and _finite(changed_at)
                  and _identity(changed_by)
                  and (verified == (None, None, None)
                       or (_identity(profile) and _revision(profile_revision) and _revision(link_revision)))
                  and (state != VoiceConsentState.ACTIVE or verified != (None, None, None)))
    for person, attempted_at, outcome in db.execute(
            "SELECT person_id,attempted_at,outcome FROM voice_verification_attempts"):
        valid &= _identity(person) and _finite(attempted_at) and outcome in outcomes
    for identity, profile, kind, bucket, at in db.execute(
            "SELECT id,profile_id,kind,score_bucket,at FROM voice_attribution_events"):
        valid &= (_identity(identity) and _identity(profile) and kind in kinds
                  and type(bucket) is int and -20 <= bucket <= 20 and _finite(at))
    for profile, revision, suspended_at, reason in db.execute(
            "SELECT profile_id,revision,suspended_at,reason FROM voice_attribution_suspensions"):
        valid &= _identity(profile) and _revision(revision) and _finite(suspended_at) and reason in reasons
    control = [tuple(row) for row in db.execute(
        "SELECT id,enabled,revision,changed_at,changed_by FROM voice_attribution_control LIMIT 2")]
    valid &= len(control) <= 1 and all(
        identity == 1 and type(enabled) is int and enabled in (0, 1) and _revision(revision)
        and _finite(changed_at) and _identity(changed_by)
        for identity, enabled, revision, changed_at, changed_by in control)
    if not valid:
        raise RuntimeError("voice_attribution_value_unknown")


def initialize_voice_attribution_schema(db: sqlite3.Connection) -> None:
    """Create or verify the voice-attribution tables atomically; never prune.

    A ledger without any of these objects gets the exact
    VOICE_ATTRIBUTION_SCHEMA. Any partial or foreign shape (table, index,
    foreign key or marker missing, extra or different) raises
    ``voice_attribution_schema_unavailable`` with nothing written. Every row is
    then validated (``voice_attribution_value_unknown``). Nothing time-based
    runs here: aged attempts and events are pruned only by the in-run
    maintenance tick, so opening a closed state never changes a protected row.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        if not _attribution_objects(db):
            for statement in _attribution_statements():
                db.execute(statement)
        objects, shape = _ATTRIBUTION_REFERENCE
        versions = [tuple(row) for row in db.execute("SELECT version FROM voice_attribution_schema LIMIT 2")] \
            if any(row[1] == "voice_attribution_schema" for row in _attribution_objects(db)) else []
        if (_attribution_objects(db) != objects or _attribution_pragmas(db) != shape
                or len(versions) != 1 or type(versions[0][0]) is not int or versions[0][0] != 1):
            raise RuntimeError("voice_attribution_schema_unavailable")
        _scan_voice_attribution_rows(db)
        db.commit()
    except BaseException:
        db.rollback()
        raise
