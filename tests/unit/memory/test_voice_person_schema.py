"""Synthetic ledger migrations distinguish older data from incomplete restores."""

import sqlite3

import pytest

from src.memory.fact_ledger import FactLedger
from src.memory.voice_schema import initialize_voice_people_schema


@pytest.mark.parametrize('older', [False, True])
def test_person_schema_migrates_once_without_replacing_older_data(tmp_path, older):
    path = tmp_path / 'ledger.db'
    if older:
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE preserved (value TEXT)')
            db.execute("INSERT INTO preserved VALUES ('synthetic')")
    ledger = FactLedger(str(path))
    try:
        ledger.initialize()
        assert list(map(tuple, ledger._db.execute('SELECT version FROM voice_person_schema'))) == [(1,)]
        ledger.close()
        ledger.initialize()
        assert list(map(tuple, ledger._db.execute('SELECT version FROM voice_person_schema'))) == [(1,)]
        if older:
            assert ledger._db.execute('SELECT value FROM preserved').fetchone()[0] == 'synthetic'
    finally:
        ledger.close()


@pytest.mark.parametrize('damage', ['table', 'marker', 'empty', 'version', 'columns', 'primary_key'])
def test_incomplete_restore_fails_before_schema_repair_and_rejects_every_retry(tmp_path, damage):
    path = tmp_path / 'ledger.db'
    ledger = FactLedger(str(path))
    ledger.initialize()
    ledger.close()
    with sqlite3.connect(path) as db:
        if damage == 'table':
            db.execute('DROP TABLE voice_profile_people')
        elif damage == 'marker':
            db.execute('DROP TABLE voice_person_schema')
        elif damage == 'empty':
            db.execute('DELETE FROM voice_person_schema')
        elif damage == 'version':
            db.execute('UPDATE voice_person_schema SET version=2')
        else:
            db.execute('DROP TABLE voice_profile_people')
            definition = 'user_id TEXT' if damage == 'columns' else (
                'profile_id TEXT, user_id TEXT, revision INTEGER NOT NULL, '
                'updated_at REAL NOT NULL, updated_by TEXT NOT NULL')
            db.execute('CREATE TABLE voice_profile_people (' + definition + ')')
        before = list(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY name'))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='voice_person_schema_unavailable'):
            ledger.initialize()
        assert ledger._db is None
        assert not ledger._voice_person_proof.snapshot.available
        with sqlite3.connect(path) as db:
            assert list(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY name')) == before


def test_failed_schema_migration_rolls_back_both_tables_and_can_retry():
    db = sqlite3.connect(':memory:', isolation_level=None)
    try:
        db.set_authorizer(lambda operation, *_: sqlite3.SQLITE_DENY
            if operation == sqlite3.SQLITE_CREATE_INDEX else sqlite3.SQLITE_OK)
        with pytest.raises(sqlite3.DatabaseError):
            initialize_voice_people_schema(db)
        db.set_authorizer(None)
        assert not list(db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'voice_%'"))
        assert not db.in_transaction
        initialize_voice_people_schema(db)
        assert list(db.execute('SELECT version FROM voice_person_schema')) == [(1,)]
    finally:
        db.close()


def test_closed_positive_proof_cannot_survive_a_failed_restore(tmp_path):
    from src.users.voice_ledger import VoiceLedger

    ledger = FactLedger(str(tmp_path / 'ledger.db'))
    people = VoiceLedger(ledger).people
    ledger.initialize()
    people.reconcile()
    assert people.user_links_cleared(frozenset({'synthetic'}))
    ledger.close()
    assert not people.user_links_cleared(frozenset({'synthetic'}))
    with sqlite3.connect(ledger.path) as db:
        db.execute('DROP TABLE voice_profile_people')
    with pytest.raises(RuntimeError, match='voice_person_schema_unavailable'):
        ledger.initialize()
    assert not people.user_links_cleared(frozenset({'synthetic'}))
    assert ledger._db is None


@pytest.mark.parametrize('stage', ['pragma', 'receipts', 'close'])
def test_failed_initializer_never_publishes_a_partial_connection(tmp_path, monkeypatch, stage):
    ledger = FactLedger(str(tmp_path / 'ledger.db'))
    original_connect = sqlite3.connect
    original_receipts = ledger._read_fact_receipts
    failing = [True]
    closed = []

    class Fault(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if stage == 'pragma' and failing[0] and sql.startswith('PRAGMA'):
                raise RuntimeError('synthetic_pragma_failure')
            return super().execute(sql, *args, **kwargs)

        def close(self):
            super().close()
            closed.append(True)
            if stage == 'close' and failing[0]:
                raise RuntimeError('synthetic_close_failure')

    def receipts(db):
        if stage != 'pragma' and failing[0]:
            raise RuntimeError('synthetic_receipts_failure')
        return original_receipts(db)

    monkeypatch.setattr(sqlite3, 'connect', lambda *args, **kwargs:
        original_connect(*args, **kwargs, factory=Fault))
    monkeypatch.setattr(ledger, '_read_fact_receipts', receipts)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='synthetic_.*_failure'):
            ledger.initialize()
        assert ledger._db is None and not ledger.authority_available
        assert not ledger._voice_person_proof.snapshot.available
    assert len(closed) == 2
    failing[0] = False
    try:
        ledger.initialize()
        assert ledger._db is not None and ledger.authority_available
        assert not ledger._voice_person_proof.snapshot.available
    finally:
        ledger.close()
