"""Bounded dual-participant observations use the original ledger transaction."""

import threading

import pytest

from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactLedger


PARTICIPANTS = (ErasureParticipant.EVENTS, ErasureParticipant.SUGGESTIONS)


@pytest.fixture
def ledger():
    value = FactLedger()
    value.initialize()
    try:
        yield value
    finally:
        value.close()


def test_dual_snapshot_cannot_mix_a_concurrent_original_registry_writer(ledger):
    journal = ledger.erasures
    first = journal.register_target(ErasureParticipant.EVENTS, 1, {"root": "old-events"})
    second = journal.register_target(ErasureParticipant.SUGGESTIONS, 1, {"root": "old-suggestions"})
    start, attempting, finished = threading.Event(), threading.Event(), threading.Event()
    added, errors = [], []

    def register():
        try:
            assert start.wait(3)
            attempting.set()
            added.append(journal.register_target(ErasureParticipant.SUGGESTIONS, 1, {"root": "new"}))
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    def row_factory(cursor, row):
        if len(row) == 12 and row[5] == b"events":
            start.set()
            assert attempting.wait(3)
            assert not finished.is_set()
        return row

    writer = threading.Thread(target=register)
    writer.start()
    original = ledger._db.row_factory
    try:
        ledger._db.row_factory = row_factory
        observed = journal.target_snapshot()
    finally:
        ledger._db.row_factory = original
        start.set()
        writer.join(3)
    assert not writer.is_alive() and not errors and len(added) == 1
    assert observed.status == "complete"
    assert {row.id for row in observed.entries} == {first, second}
    assert observed.ledger is ledger and observed.participants == PARTICIPANTS
    latest = journal.target_snapshot()
    assert {row.id for row in latest.entries} == {first, second, added[0]}
    assert latest.digest != observed.digest


@pytest.mark.parametrize("count", [256, 257], ids=["limit256", "overflow257"])
def test_snapshot_bounds_actual_sql_rows_before_materializing_descriptors(ledger, count):
    journal = ledger.erasures
    with ledger._transaction() as db:
        for number in range(count):
            participant = PARTICIPANTS[number % 2]
            journal._register_target(db, participant, 1, {"number": number})
    fetched = []

    def row_factory(cursor, row):
        if len(row) == 12:
            fetched.append(tuple(len(value) for value in (row[2], row[5], row[8], row[11])))
        return row

    original = ledger._db.row_factory
    try:
        ledger._db.row_factory = row_factory
        snapshot = journal.target_snapshot()
    finally:
        ledger._db.row_factory = original
    assert len(snapshot.entries) == 256
    assert len(fetched) == count
    assert snapshot.status == ("complete" if count == 256 else "overflow")
    assert all(a <= 65 and b <= 33 and c <= 33 and d <= 8193 for a, b, c, d in fetched)


@pytest.mark.parametrize("fault", ["oversized", "malformed", "unknown_version"])
def test_invalid_or_unsupported_targets_remain_visible_without_private_repr(ledger, fault):
    journal = ledger.erasures
    if fault == "unknown_version":
        identity = journal.register_target(ErasureParticipant.EVENTS, 2, {"root": "synthetic-private"})
    else:
        identity = "a" * 64
        descriptor = "synthetic-private" * 1024 if fault == "oversized" else "synthetic-private{"
        with ledger._transaction() as db:
            db.execute("INSERT INTO erasure_targets VALUES (?,?,?,?)", (identity, "events", 1, descriptor))
    snapshot = journal.target_snapshot()
    assert len(snapshot.entries) == 1 and "synthetic-private" not in repr(snapshot)
    [entry] = snapshot.entries
    if fault == "unknown_version":
        assert snapshot.status == "complete" and entry.status == "unsupported_version"
        assert entry.id == identity and entry.version == 2
        detached = entry.descriptor
        detached["root"] = "mutated"
        assert entry.descriptor["root"] == "synthetic-private"
    else:
        assert snapshot.status == "corrupt"
        assert entry.status == ("oversized" if fault == "oversized" else "corrupt")
        assert entry.id.startswith("opaque:") and entry.opaque_digest
        assert entry.descriptor is None


def test_snapshot_requires_both_participants_and_original_current_transaction(ledger):
    journal = ledger.erasures
    with pytest.raises(ValueError, match="invalid_target_snapshot_scope"):
        journal.target_snapshot((ErasureParticipant.EVENTS,))
    with pytest.raises(ValueError, match="target_snapshot_requires_original_transaction"):
        journal._target_snapshot_db(ledger._db, PARTICIPANTS)
    with ledger._transaction() as db:
        assert journal._target_snapshot_db(db, PARTICIPANTS).status == "complete"
        with pytest.raises(RuntimeError, match="target_snapshot_requires_single_transaction"):
            journal.target_snapshot()


def test_closed_snapshot_does_not_reopen_original_ledger(ledger):
    ledger.close()
    generation = ledger.lifecycle_generation
    snapshot = ledger.erasures.target_snapshot()
    assert snapshot.status == "unavailable" and not snapshot.entries
    assert ledger._db is None and ledger.lifecycle_generation == generation
