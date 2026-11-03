"""Synthetic review migration and immutable metadata under the source lifetime."""

from contextlib import contextmanager
from dataclasses import replace
import sqlite3

import pytest

from src.evaluation.provenance import CaptureLane, CaptureProvenance, CaptureSource, FrameInterval
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.voice_schema import initialize_voice_review_schema
from src.users.voice_capture import VoiceCaptureReview, build_review_capture
from src.users.voice_capture_metadata import decode_audio_metadata, decode_transcript_metadata
from tests.integration.test_live_stream_receipts import flush
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import ACTOR, AUDIO, VECTOR, graph


LEGACY_COLUMNS = (
    'fingerprint_id', 'audio_wav', 'transcript', 'room_id', 'duration_s', 'clip_duration_s',
)
CURRENT_COLUMNS = LEGACY_COLUMNS + ('audio_metadata_json', 'transcript_metadata_json')
PUBLIC_REVIEW_KEYS = {
    'id', 'observed_at', 'expires_at', 'transcript', 'room_id', 'duration_s',
    'clip_duration_s', 'clip_truncated', 'transcript_scope', 'audio_mime', 'audio_base64',
}


@contextmanager
def connection(path=':memory:'):
    db = sqlite3.connect(path, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        yield db
    finally:
        db.close()


def columns(db):
    return tuple(row[1] for row in db.execute('PRAGMA table_info(voice_capture_reviews)'))


def feature_tables(db):
    return {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('voice_capture_reviews','voice_capture_review_schema')")}


def legacy(db, *, on_delete='CASCADE', audio_type='BLOB'):
    assert on_delete in {'CASCADE', 'RESTRICT'} and audio_type in {'BLOB', 'TEXT'}
    db.execute('CREATE TABLE voice_fingerprints (id TEXT PRIMARY KEY)')
    db.execute(f'''CREATE TABLE voice_capture_reviews (
        fingerprint_id TEXT PRIMARY KEY REFERENCES voice_fingerprints(id) ON DELETE {on_delete},
        audio_wav {audio_type} NOT NULL, transcript TEXT NOT NULL, room_id TEXT,
        duration_s REAL NOT NULL, clip_duration_s REAL NOT NULL)''')
    db.execute("INSERT INTO voice_fingerprints(id) VALUES ('synthetic-fingerprint')")
    db.execute('INSERT INTO voice_capture_reviews VALUES (?,?,?,?,?,?)',
        ('synthetic-fingerprint', b'synthetic-audio', 'synthetic-text', 'synthetic-room', 8., 6.))


def test_fresh_schema_has_nullable_metadata_and_reopens_without_repair():
    with connection() as db:
        initialize_voice_review_schema(db)
        assert columns(db) == CURRENT_COLUMNS
        assert feature_tables(db) == {'voice_capture_reviews', 'voice_capture_review_schema'}
        assert [tuple(row) for row in db.execute('SELECT version FROM voice_capture_review_schema')] == [(1,)]
        assert [(row[1], row[2], row[3], row[5]) for row in db.execute(
            'PRAGMA table_info(voice_capture_reviews)')][-2:] == [
                ('audio_metadata_json', 'TEXT', 0, 0), ('transcript_metadata_json', 'TEXT', 0, 0)]
        assert [tuple(row) for row in db.execute('PRAGMA foreign_key_list(voice_capture_reviews)')] == [
            (0, 0, 'voice_fingerprints', 'fingerprint_id', 'id', 'NO ACTION', 'CASCADE', 'NONE')]
        before = db.execute("SELECT sql FROM sqlite_master WHERE name='voice_capture_reviews'").fetchone()[0]
        initialize_voice_review_schema(db)
        assert db.execute("SELECT sql FROM sqlite_master WHERE name='voice_capture_reviews'").fetchone()[0] == before


def test_six_column_legacy_migration_preserves_bytes_and_leaves_null_metadata():
    with connection() as db:
        legacy(db)
        original = tuple(db.execute('SELECT * FROM voice_capture_reviews').fetchone())
        initialize_voice_review_schema(db)
        current = tuple(db.execute('SELECT * FROM voice_capture_reviews').fetchone())
        assert current == (*original, None, None)
        initialize_voice_review_schema(db)
        assert tuple(db.execute('SELECT * FROM voice_capture_reviews').fetchone()) == current


@pytest.mark.parametrize('damage', [
    'missing_marker', 'missing_review', 'empty_marker', 'unsupported_marker',
    'extra_marker_row', 'missing_column', 'wrong_marker_shape', 'marker_is_view',
])
def test_partial_or_versioned_restore_fails_without_generic_repair(damage):
    with connection() as db:
        initialize_voice_review_schema(db)
        if damage == 'missing_marker':
            db.execute('DROP TABLE voice_capture_review_schema')
        elif damage == 'missing_review':
            db.execute('DROP TABLE voice_capture_reviews')
        elif damage == 'empty_marker':
            db.execute('DELETE FROM voice_capture_review_schema')
        elif damage == 'unsupported_marker':
            db.execute('UPDATE voice_capture_review_schema SET version=2')
        elif damage == 'extra_marker_row':
            db.execute('INSERT INTO voice_capture_review_schema(version) VALUES (2)')
        elif damage == 'missing_column':
            db.execute('ALTER TABLE voice_capture_reviews DROP COLUMN audio_metadata_json')
        elif damage == 'marker_is_view':
            db.execute('DROP TABLE voice_capture_review_schema')
            db.execute('CREATE VIEW voice_capture_review_schema AS SELECT 1 AS version')
        else:
            db.execute('DROP TABLE voice_capture_review_schema')
            db.execute('CREATE TABLE voice_capture_review_schema (unexpected INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO voice_capture_review_schema VALUES (1)')
        before = [tuple(row) for row in db.execute(
            "SELECT name,sql FROM sqlite_master WHERE name LIKE 'voice_capture_review%' ORDER BY name")]
        for _ in range(2):
            with pytest.raises(RuntimeError, match='voice_review_schema_unavailable'):
                initialize_voice_review_schema(db)
            assert not db.in_transaction
            assert [tuple(row) for row in db.execute(
                "SELECT name,sql FROM sqlite_master WHERE name LIKE 'voice_capture_review%' ORDER BY name")] == before


@pytest.mark.parametrize('field,value', [('on_delete', 'RESTRICT'), ('audio_type', 'TEXT')])
@pytest.mark.parametrize('versioned', [False, True])
def test_wrong_shape_or_foreign_key_cannot_be_repaired(field, value, versioned):
    with connection() as db:
        legacy(db, **{field: value})
        if versioned:
            db.execute('ALTER TABLE voice_capture_reviews ADD COLUMN audio_metadata_json TEXT')
            db.execute('ALTER TABLE voice_capture_reviews ADD COLUMN transcript_metadata_json TEXT')
            db.execute('CREATE TABLE voice_capture_review_schema (version INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO voice_capture_review_schema VALUES (1)')
        before = tuple(db.execute('SELECT * FROM voice_capture_reviews').fetchone())
        with pytest.raises(RuntimeError, match='voice_review_schema_unavailable'):
            initialize_voice_review_schema(db)
        assert columns(db) == (CURRENT_COLUMNS if versioned else LEGACY_COLUMNS)
        assert tuple(db.execute('SELECT * FROM voice_capture_reviews').fetchone()) == before
        assert ('voice_capture_review_schema' in feature_tables(db)) is versioned


def test_partial_unmarked_columns_are_not_a_legacy_migration():
    with connection() as db:
        legacy(db)
        db.execute('ALTER TABLE voice_capture_reviews ADD COLUMN audio_metadata_json TEXT')
        with pytest.raises(RuntimeError, match='voice_review_schema_unavailable'):
            initialize_voice_review_schema(db)
        assert feature_tables(db) == {'voice_capture_reviews'}
        assert columns(db) == LEGACY_COLUMNS + ('audio_metadata_json',)


@pytest.mark.parametrize('fresh', [False, True])
def test_ddl_failure_rolls_back_own_migration_and_retry_can_finish(fresh):
    with connection() as db:
        if not fresh:
            legacy(db)
        alters = []

        def authorize(action, first, second, database, trigger):
            if fresh and action == sqlite3.SQLITE_CREATE_TABLE and first == 'voice_capture_review_schema':
                return sqlite3.SQLITE_DENY
            if not fresh and action == sqlite3.SQLITE_ALTER_TABLE and second == 'voice_capture_reviews':
                alters.append(True)
                if len(alters) == 2:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(authorize)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                initialize_voice_review_schema(db)
        finally:
            db.set_authorizer(None)
        assert not db.in_transaction
        assert feature_tables(db) == (set() if fresh else {'voice_capture_reviews'})
        if not fresh:
            assert columns(db) == LEGACY_COLUMNS
            assert db.execute('SELECT transcript FROM voice_capture_reviews').fetchone()[0] == 'synthetic-text'
        initialize_voice_review_schema(db)
        assert columns(db) == CURRENT_COLUMNS


def test_initializer_never_publishes_failed_migration_and_retry_validates(tmp_path, monkeypatch):
    import src.memory.fact_ledger as module

    ledger = FactLedger(str(tmp_path / 'ledger.db'))
    initialize = module.initialize_voice_review_schema

    def fail(db):
        def authorize(action, first, second, database, trigger):
            if action == sqlite3.SQLITE_CREATE_TABLE and first == 'voice_capture_review_schema':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize)
        try:
            initialize(db)
        finally:
            db.set_authorizer(None)

    monkeypatch.setattr(module, 'initialize_voice_review_schema', fail)
    for _ in range(2):
        with pytest.raises(sqlite3.DatabaseError):
            ledger.initialize()
        assert ledger._db is None and not ledger.authority_available
        with connection(str(tmp_path / 'ledger.db')) as db:
            assert not feature_tables(db)
    monkeypatch.setattr(module, 'initialize_voice_review_schema', initialize)
    try:
        ledger.initialize()
        assert ledger._db is not None and ledger.authority_available
        assert columns(ledger._db) == CURRENT_COLUMNS
    finally:
        ledger.close()


def capture_provenance(channel, frames):
    # UNKNOWN DSP is truthful metadata; storage grants no comparability.
    return CaptureProvenance(CaptureSource.LIVE_INPUT, 'synthetic-opening', CaptureLane.AMBIENT,
        4, 17, 2, channel, 16000, frames)


def known_review(text='Synthetic transcript'):
    return replace(build_review_capture(AUDIO, text, 'synthetic-room'),
        audio_provenance=capture_provenance(0, AUDIO.size), audio_span=FrameInterval(0, min(AUDIO.size, 96000)),
        transcript_provenance=capture_provenance(1, AUDIO.size), transcript_span=FrameInterval(0, AUDIO.size),
        transcript_truncated=False)


def raw_review(voices, identity):
    with voices.privacy.ledger._transaction() as db:
        return db.execute('SELECT * FROM voice_capture_reviews WHERE fingerprint_id=?', (identity,)).fetchone()


@pytest.mark.asyncio
async def test_metadata_round_trip_is_private_and_public_review_keys_stay_unchanged(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        source = memory_source(now=clock.now)
        review = known_review()
        identity = await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1., (source,), review)
        row = await voices.acall(raw_review, voices, identity)
        audio, audio_span = decode_audio_metadata(row['audio_metadata_json'])
        transcript, transcript_span, truncated = decode_transcript_metadata(row['transcript_metadata_json'])
        restored = VoiceCaptureReview(row['audio_wav'], row['transcript'], row['room_id'],
            row['duration_s'], row['clip_duration_s'], audio_provenance=audio, audio_span=audio_span,
            transcript_provenance=transcript, transcript_span=transcript_span, transcript_truncated=truncated)
        assert restored == review and audio.selected_channel == 0 and transcript.selected_channel == 1
        public = await voices.acall(voices.ledger.review_capture, identity, ACTOR)
        assert set(public) == PUBLIC_REVIEW_KEYS
        assert public['expires_at'] == source.expires_at
        assert 'synthetic-opening' not in str(public)


@pytest.mark.asyncio
@pytest.mark.parametrize('first_known', [False, True])
async def test_first_writer_known_or_unknown_metadata_cannot_be_replaced_or_renewed(tmp_path, first_known):
    async with graph(tmp_path) as (_, _, voices, clock):
        source = memory_source(now=clock.now, ttl=10)
        known = known_review()
        unknown = build_review_capture(AUDIO, 'Synthetic old transcript', 'synthetic-old-room')
        first, second = (known, unknown) if first_known else (unknown, known)
        identity = await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1., (source,), first)
        before = tuple(await voices.acall(raw_review, voices, identity))
        clock.now += 9
        assert await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1., (source,), second) == identity
        assert tuple(await voices.acall(raw_review, voices, identity)) == before
        if not first_known:
            assert before[-2:] == (None, None)
        clock.now += 1
        with pytest.raises(LookupError):
            await voices.acall(voices.ledger.review_capture, identity, ACTOR)


@pytest.mark.asyncio
@pytest.mark.parametrize('typed', [False, True])
async def test_typed_provenance_rejects_one_frame_duration_contradiction_but_legacy_keeps_tolerance(tmp_path, typed):
    async with graph(tmp_path) as (_, _, voices, clock):
        review = known_review() if typed else build_review_capture(AUDIO, 'Synthetic transcript', None)
        source = memory_source(now=clock.now)
        mismatched = (AUDIO.size - 1) / 16000
        if typed:
            with pytest.raises(ValueError, match='invalid_voice_review_capture'):
                await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, mismatched, (source,), review)
            with voices.privacy.ledger._transaction() as db:
                assert db.execute('SELECT COUNT(*) FROM voice_fingerprints').fetchone()[0] == 0
        else:
            identity = await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, mismatched, (source,), review)
            assert identity is not None
            assert (await voices.acall(raw_review, voices, identity))['audio_metadata_json'] is None


@pytest.mark.asyncio
async def test_put_revalidates_review_instead_of_trusting_a_previously_built_object(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        review = known_review()
        object.__setattr__(review, 'audio_span', FrameInterval(0, AUDIO.size + 1))
        with pytest.raises(ValueError):
            await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1.,
                (memory_source(now=clock.now),), review)
        with voices.privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM voice_fingerprints').fetchone()[0] == 0
            assert db.execute('SELECT COUNT(*) FROM voice_capture_reviews').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('field,raw', [
    ('audio_metadata_json', '{}'), ('audio_metadata_json', '{'),
    ('audio_metadata_json', '{}\x00'), ('audio_metadata_json', b'{}'),
    ('audio_metadata_json', 'x' * 4097), ('audio_metadata_json', 'invalid_utf8'),
    ('transcript_metadata_json', '{}'), ('transcript_metadata_json', '{'),
    ('transcript_metadata_json', b'{}'), ('transcript_metadata_json', 'x' * 4097),
    ('transcript_metadata_json', 'invalid_utf8'),
])
async def test_malformed_known_metadata_denies_instead_of_becoming_legacy_unknown(tmp_path, field, raw):
    async with graph(tmp_path) as (_, _, voices, clock):
        identity = await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1.,
            (memory_source(now=clock.now),), known_review())
        assert field in {'audio_metadata_json', 'transcript_metadata_json'}
        with voices.privacy.ledger._transaction() as db:
            if raw == 'invalid_utf8':
                db.execute(f'UPDATE voice_capture_reviews SET {field}=CAST(? AS TEXT) WHERE fingerprint_id=?',
                    (b'\xff', identity))
            else:
                db.execute(f'UPDATE voice_capture_reviews SET {field}=? WHERE fingerprint_id=?', (raw, identity))
        with pytest.raises(LookupError, match='voice_capture_unavailable'):
            await voices.acall(voices.ledger.review_capture, identity, ACTOR)


@pytest.mark.asyncio
async def test_source_erasure_cascades_metadata_with_existing_preview_and_text(tmp_path):
    async with graph(tmp_path) as (_, owner, voices, clock):
        source = memory_source(now=clock.now)
        identity = await voices.acall(voices.ledger.put, 'synthetic-model', VECTOR, 1., (source,), known_review())
        row = await voices.acall(raw_review, voices, identity)
        assert row['audio_metadata_json'] is not None and row['transcript_metadata_json'] is not None
        await owner.request(source.conversation_id, MemoryReviewer('synthetic-admin'))
        await flush(owner)
        assert await voices.acall(raw_review, voices, identity) is None
        with voices.privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM voice_capture_reviews').fetchone()[0] == 0
            assert db.execute('SELECT version FROM voice_capture_review_schema').fetchone()[0] == 1
