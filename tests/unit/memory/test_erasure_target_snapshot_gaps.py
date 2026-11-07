"""Original in-memory ledger failure and encoded descriptor boundary contracts.

These are real SQLite/schema/authorizer operations, without managed admission,
compiled graph claims, application mocks returning successful snapshots, or SDKs.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.memory.erasure_journal import ErasureParticipant, MAX_TARGET_DESCRIPTOR_BYTES
from src.memory.fact_ledger import FactLedger


@pytest.fixture
def ledger():
    value = FactLedger()
    value.initialize()
    try:
        yield value
    finally:
        value.close()


def test_missing_target_table_is_unavailable_without_recreating_schema(ledger, monkeypatch):
    original, generation = ledger._db, ledger.lifecycle_generation
    before = ledger.erasures.target_snapshot()
    assert before.status == "complete" and before.entries == ()
    with ledger._transaction() as database:
        database.execute("DROP TABLE erasure_targets")

    opened = []
    def forbidden_reopen(*args, **kwargs):
        opened.append(True)
        raise AssertionError("snapshot attempted to replace its original ledger")
    monkeypatch.setattr(sqlite3, "connect", forbidden_reopen)
    snapshot = ledger.erasures.target_snapshot()
    assert snapshot.status == "unavailable" and snapshot.entries == ()
    assert snapshot.digest != before.digest
    assert snapshot.ledger is ledger and snapshot.lifecycle_generation == generation
    assert ledger._db is original and not original.in_transaction and opened == []
    assert original.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='erasure_targets'"
    ).fetchall() == []


def test_target_snapshot_native_read_denial_is_unavailable_and_recoverable(ledger):
    journal, original = ledger.erasures, ledger._db
    event_id = journal.register_target(ErasureParticipant.EVENTS, 1, {"path": "synthetic-events"})
    suggestion_id = journal.register_target(ErasureParticipant.SUGGESTIONS, 1, {"path": "synthetic-suggestions"})
    before = journal.target_snapshot()
    assert before.status == "complete" and {entry.id for entry in before.entries} == {event_id, suggestion_id}
    denied = []

    def authorizer(action, table, column, database, origin):
        if action == sqlite3.SQLITE_READ and table == "erasure_targets":
            denied.append((table, column))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    original.set_authorizer(authorizer)
    try:
        snapshot = journal.target_snapshot()
    finally:
        original.set_authorizer(None)
    assert denied  # A real SQLite read reached the denial, not a mocked result.
    assert snapshot.status == "unavailable" and snapshot.entries == ()
    assert snapshot.ledger is ledger and snapshot.lifecycle_generation == before.lifecycle_generation
    assert ledger._db is original and not original.in_transaction
    recovered = journal.target_snapshot()
    assert recovered.status == "complete" and recovered.digest == before.digest
    assert {entry.id for entry in recovered.entries} == {event_id, suggestion_id}


@pytest.mark.parametrize("participant", [ErasureParticipant.EVENTS, ErasureParticipant.SUGGESTIONS])
def test_target_descriptor_exact_limit_and_one_byte_over_remain_fail_closed(ledger, participant):
    journal = ledger.erasures
    # The existing registration contract counts encoded JSON bytes. Include a
    # Unicode character so this cannot accidentally become a raw string limit.
    prefix = "synthetic-boundary-\u2603"
    overhead = len(json.dumps({"payload": prefix}, sort_keys=True, separators=(",", ":")).encode())
    descriptor = {"payload": prefix + "x" * (MAX_TARGET_DESCRIPTOR_BYTES - overhead)}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) == MAX_TARGET_DESCRIPTOR_BYTES
    assert len(descriptor["payload"]) < MAX_TARGET_DESCRIPTOR_BYTES
    identity = journal.register_target(participant, 1, descriptor)
    other = ErasureParticipant.SUGGESTIONS if participant is ErasureParticipant.EVENTS else ErasureParticipant.EVENTS
    other_id = journal.register_target(other, 1, {"path": "synthetic-other-participant"})
    before = journal.target_snapshot()
    entry = next(entry for entry in before.entries if entry.id == identity)
    assert before.status == "complete" and entry.status == "registered"
    assert entry.participant is participant and entry.version == 1 and entry.descriptor == descriptor
    assert ledger._db.execute(
        "SELECT length(CAST(descriptor AS BLOB)) FROM erasure_targets WHERE id=?", (identity,)
    ).fetchone()[0] == MAX_TARGET_DESCRIPTOR_BYTES

    oversized = {"payload": descriptor["payload"] + "x"}
    encoded_over = json.dumps(oversized, sort_keys=True, separators=(",", ":"))
    assert len(encoded_over.encode()) == MAX_TARGET_DESCRIPTOR_BYTES + 1
    with pytest.raises(ValueError, match="erasure_target_limit"):
        journal.register_target(participant, 1, oversized)
    after_rejected_registration = journal.target_snapshot()
    assert after_rejected_registration.status == "complete"
    assert after_rejected_registration.digest == before.digest
    assert {entry.id for entry in after_rejected_registration.entries} == {identity, other_id}

    # Model an oversized persisted row bypassing the supported registration
    # boundary. It must remain visible as opaque corruption, never disappear
    # into an apparently complete empty/partial registry or expose payloads.
    with ledger._transaction() as database:
        database.execute("UPDATE erasure_targets SET descriptor=? WHERE id=?", (encoded_over, identity))
    snapshot = journal.target_snapshot()
    assert snapshot.status == "corrupt" and len(snapshot.entries) == 2
    opaque = next(entry for entry in snapshot.entries if entry.participant is participant)
    assert opaque.status == "oversized" and opaque.id.startswith("opaque:") and opaque.opaque_digest
    assert opaque.version is None and opaque.descriptor is None
    assert "synthetic-boundary" not in repr(snapshot)
    assert next(entry for entry in snapshot.entries if entry.id == other_id).status == "registered"
