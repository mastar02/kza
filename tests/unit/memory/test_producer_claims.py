"""Producer claim identity, retries and leased recovery of the real journal.

Every finish(True) here is a JOURNAL PROTOCOL receipt.  No physical producer
driver is registered and these tests do not certify native cleanup or an
aggregate erasure result.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import sqlite3

import pytest

from src.memory.erasure_journal import (
    ErasureError, ErasureParticipant, ErasureResult, ErasureStep,
)
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source


pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer("synthetic-claim-reviewer")
DESCRIPTOR = {"kind": "runtime_producers", "protocol": 1}


async def claim_due(privacy, operation_id, target_id):
    journal = privacy.ledger.erasures
    due = await privacy.acall(journal.due)
    row, = [row for row in due if row["operation_id"] == operation_id and row["target_id"] == target_id]
    claim = await privacy.acall(journal.claim, row)
    assert type(claim) is ErasureStep
    return row, claim


async def local_step(privacy, operation_id):
    receipt = await privacy.acall(privacy.ledger.erasures.snapshot, operation_id)
    step, = [step for step in receipt["steps"] if step["participant"] == "producers"]
    # Other missing participants are deliberately not completed by this fixture.
    assert receipt["state"] != "complete"
    return step


async def protocol_complete(privacy, claim):
    """Exercise journal ACKs only; this is not a cleanup callback."""
    return await privacy.acall(privacy.ledger.erasures.finish, claim, ErasureResult(True))


@asynccontextmanager
async def claimed_journal(tmp_path):
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock)
    try:
        await privacy.acall(privacy.initialize)
        journal = privacy.ledger.erasures
        target = await privacy.acall(journal.register_target,
            ErasureParticipant.PRODUCERS, 1, dict(DESCRIPTOR))
        source = memory_source(now=clock.now)
        operation = await privacy.acall(journal.request, source.conversation_id, ACTOR)
        row, claim = await claim_due(privacy, operation["id"], target)
        assert await privacy.acall(journal.is_current_claim, claim)
        assert not privacy.sources_available((source,))
        yield SimpleNamespace(privacy=privacy, clock=clock, journal=journal,
            target=target, source=source, operation=operation, row=row, claim=claim)
    finally:
        await privacy.begin_close()


class ForeignStep(ErasureStep):
    pass


class ForeignDescriptor(dict):
    pass


def forged_claim(claim, case):
    if case == "foreign_dto":
        return ForeignStep(**vars(claim))
    changes = {
        "other_operation": {"operation_id": str(uuid4())},
        "other_conversation": {"conversation_id": str(uuid4())},
        "other_participant": {"participant": ErasureParticipant.CONTEXTS},
        "other_target": {"target_id": "different-registered-identity"},
        "other_version": {"version": claim.version + 1},
        "boolean_version": {"version": True},
        "float_version": {"version": 1.0},
        "descriptor_bool_one": {"descriptor": {"kind": "runtime_producers", "protocol": True}},
        "descriptor_extra": {"descriptor": {**DESCRIPTOR, "unowned": True}},
        "descriptor_non_json": {"descriptor": {**DESCRIPTOR, "unowned": object()}},
        "foreign_descriptor_type": {"descriptor": ForeignDescriptor(DESCRIPTOR)},
        "other_token": {"claim_token": str(uuid4())},
        "other_attempt": {"attempts": claim.attempts + 1},
        "boolean_attempt": {"attempts": True},
        "float_attempt": {"attempts": 1.0},
        "stale_cursor": {"cursor": "different:position"},
    }
    return replace(claim, **changes[case])


@pytest.mark.parametrize("case", [
    "other_operation", "other_conversation", "other_participant", "other_target",
    "other_version", "boolean_version", "float_version", "descriptor_bool_one",
    "descriptor_extra", "descriptor_non_json", "foreign_descriptor_type",
    "other_token", "other_attempt", "boolean_attempt", "float_attempt", "stale_cursor", "foreign_dto",
])
async def test_only_the_exact_current_registered_claim_is_recognized(tmp_path, case):
    async with claimed_journal(tmp_path) as g:
        before = await local_step(g.privacy, g.operation["id"])
        assert not await g.privacy.acall(g.journal.is_current_claim, forged_claim(g.claim, case))
        # Descriptor order is immaterial; JSON true and integer 1 are not.
        equivalent = replace(g.claim, descriptor={"protocol": 1, "kind": "runtime_producers"})
        assert await g.privacy.acall(g.journal.is_current_claim, equivalent)
        assert await local_step(g.privacy, g.operation["id"]) == before
        assert await g.privacy.acall(g.journal.is_current_claim, g.claim)


async def test_retry_replaces_token_cursor_and_attempt_without_accepting_the_old_ack(tmp_path):
    async with claimed_journal(tmp_path) as g:
        progress = ErasureResult(False, cursor="batch:one", pending_count=1)
        assert await g.privacy.acall(g.journal.finish, g.claim, progress)
        pending = await local_step(g.privacy, g.operation["id"])
        assert pending["state"] == "pending" and pending["pending_count"] == 1
        assert not await g.privacy.acall(g.journal.is_current_claim, g.claim)
        assert await g.privacy.acall(g.journal.claim, g.row) is None
        # Only the retry clock advances; no claim state or token is patched.
        g.clock.now += .02
        _, current = await claim_due(g.privacy, g.operation["id"], g.target)
        assert current.claim_token != g.claim.claim_token
        assert current.attempts == g.claim.attempts + 1 and current.cursor == "batch:one"
        assert not await protocol_complete(g.privacy, g.claim)
        assert not await g.privacy.acall(g.journal.finish, g.claim, None, ErasureError.FAILED)
        assert await g.privacy.acall(g.journal.is_current_claim, current)
        assert await protocol_complete(g.privacy, current)
        complete = await local_step(g.privacy, g.operation["id"])
        assert complete["state"] == "complete"
        assert not await protocol_complete(g.privacy, g.claim)
        assert await local_step(g.privacy, g.operation["id"]) == complete


async def test_repeated_initialization_on_the_same_lease_does_not_recover_or_mint_claims(tmp_path):
    async with claimed_journal(tmp_path) as g:
        running = await local_step(g.privacy, g.operation["id"])
        for _ in range(2):
            await g.privacy.acall(g.privacy.initialize)
            assert await g.privacy.acall(g.journal.is_current_claim, g.claim)
            assert await g.privacy.acall(g.journal.claim, g.row) is None
            assert await local_step(g.privacy, g.operation["id"]) == running
        assert await protocol_complete(g.privacy, g.claim)
        complete = await local_step(g.privacy, g.operation["id"])
        assert complete["state"] == "complete"
        for _ in range(2):
            await g.privacy.acall(g.privacy.initialize)
            assert not await g.privacy.acall(g.journal.is_current_claim, g.claim)
            assert await g.privacy.acall(g.journal.claim, g.row) is None
            assert await local_step(g.privacy, g.operation["id"]) == complete


@pytest.mark.parametrize("prior_state", ["running", "complete"])
async def test_new_leased_instance_reopens_producer_proof_and_rejects_old_claim(tmp_path, prior_state):
    async with claimed_journal(tmp_path) as g:
        if prior_state == "complete":
            assert await protocol_complete(g.privacy, g.claim)
        old_step = await local_step(g.privacy, g.operation["id"])
        assert old_step["state"] == prior_state
        assert g.privacy.resource_status()["writer_lease"]
        await g.privacy.begin_close()
        assert g.privacy.ledger._db is None and not g.privacy.resource_status()["writer_lease"]
        replacement = PrivacyServices(str(tmp_path / "privacy.db"), clock=g.clock)
        try:
            await replacement.acall(replacement.initialize)
            journal = replacement.ledger.erasures
            assert replacement.resource_status()["writer_lease"]
            reopened = await local_step(replacement, g.operation["id"])
            assert reopened["state"] == "pending"
            assert reopened["last_error"] == "interrupted_by_restart"
            assert reopened["attempts"] == g.claim.attempts
            assert not replacement.sources_available((g.source,))
            assert not await replacement.acall(journal.is_current_claim, g.claim)
            _, current = await claim_due(replacement, g.operation["id"], g.target)
            assert current.attempts == g.claim.attempts + 1 and current.cursor is None
            assert current.claim_token != g.claim.claim_token
            assert not await protocol_complete(replacement, g.claim)
            assert await replacement.acall(journal.is_current_claim, current)
            assert await protocol_complete(replacement, current)
            receipt = await replacement.acall(journal.snapshot, g.operation["id"])
            assert receipt["requested_at"] == g.operation["requested_at"]
            assert receipt["requested_by"] == g.operation["requested_by"]
        finally:
            await replacement.begin_close()


async def test_intent_without_a_committed_tombstone_cannot_authorize_a_claim(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=Clock())
    try:
        await privacy.acall(privacy.initialize)
        journal = privacy.ledger.erasures
        target = await privacy.acall(journal.register_target,
            ErasureParticipant.PRODUCERS, 1, dict(DESCRIPTOR))
        source = memory_source(now=privacy.ledger.clock())

        def intent_only():
            with privacy.ledger._transaction() as db:
                return journal.ensure_operation(db, source.conversation_id, ACTOR)

        # The real schema prevents this incomplete intent from existing at all.
        # Keep foreign keys enabled and exercise the transaction rollback.
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await privacy.acall(intent_only)
        assert privacy.sources_available((source,))
        assert await privacy.acall(journal.due) == []

        # A committed public request is the positive control for the same target.
        operation = await privacy.acall(journal.request, source.conversation_id, ACTOR)
        _, claim = await claim_due(privacy, operation["id"], target)
        assert not privacy.sources_available((source,))
        assert await privacy.acall(journal.is_current_claim, claim)
        # No finish(True): this case proves authority, not physical cleanup.
    finally:
        await privacy.begin_close()
