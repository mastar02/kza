"""A compatible data replay through two actual application generations."""
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
import time

import pytest

from src.memory.erasure_journal import ErasureError, ErasureParticipant, ErasureResult
from src.memory.prompt_context import SourcesInvalidated
from src.users.voice_ledger import VoiceLedger
from tests.mocks import application_restore as replay
from tests.mocks import integrated_restore as data_helpers
from tests.mocks.memory_graph import Clock
from tests.mocks.producer_erasure_graph import NativeGate as ProducerGate
from tests.mocks.restore_application_graph import application_graph
from tests.mocks.restore_preservation import (
    ReplayRejected, assert_manual_sentinels, assert_preserved, capture_preservation,
    seed_manual_sentinels,
)


pytestmark = pytest.mark.asyncio


async def test_public_graph_replay_keeps_current_authority_and_waits_for_native_returns(tmp_path, monkeypatch):
    clock = Clock(time.time())
    subject_gate, unrelated_gate = ProducerGate(), ProducerGate()
    async with application_graph(tmp_path, monkeypatch, clock,
        subject_gate=subject_gate, unrelated_gate=unrelated_gate) as first:
        subject, foreign = await replay.seed_all(first)
        claims = []
        claim = first.privacy.ledger.erasures.claim

        def observe_claim(row):
            result = claim(row)
            if result is not None and result.participant is ErasureParticipant.PRODUCERS:
                claims.append(result)
            return result

        monkeypatch.setattr(first.privacy.ledger.erasures, "claim", observe_claim)
        saved, schema = await replay.snapshot_payload(first, subject)
        sources = await replay.manual_voice_inputs(first, foreign)
        manual = await seed_manual_sentinels(first, sources=sources,
            voice_ids=tuple(foreign.voice_ids), suggestion_ids=tuple(foreign.suggestion_ids),
            root_lease=replay.root_lease(first), delete_profile=True)
        foreign_before = await replay.payload(first, foreign)
        original_u = foreign.source.to_dict()
        operation = await replay.erase_with_real_producer(first, subject, foreign,
            subject_gate, unrelated_gate)
        await replay.clear_browser(first, subject)
        await replay.settle(first, operation)
        selected = await data_helpers.steps(first, operation)
        assert any(step["participant"] == "producers" and step["version"] == 1
            and step["state"] == "complete" for step in selected)
        old_claim = next(item for item in reversed(claims) if item.operation_id == operation["id"])
        await replay.assert_subject_erased(first, subject)
        await replay.assert_foreign_unchanged(first, foreign, foreign_before)
        await assert_manual_sentinels(first, manual)
        assert first.home["service_calls"] == 0
        old_graph, old_privacy, old_owner = first.graph, first.privacy, first.owner
        old_inventory = first.graph.producer_erasure.inventory
        old_namespace = first.reader.namespace
    assert first.client.closed
    assert not first.privacy.resource_status()["writer_lease"]
    assert subject_gate.returned.is_set() and unrelated_gate.returned.is_set()

    # No live application/SDK reference is used by the offline replay.
    prepared = await asyncio.to_thread(replay.restore_payload, tmp_path, saved, schema)
    assert prepared.preservation is not None
    reader_checks = []

    async def before_voices(privacy):
        def selected_reviews():
            return {name: [row for row in data_helpers.rows(tmp_path / "privacy.db", name)
                if row.get("id", row.get("fingerprint_id")) in subject.voice_ids]
                for name in ("voice_fingerprints", "voice_capture_reviews")}
        before = await privacy.acall(selected_reviews)
        def reject_stale_claim():
            # Serialize this metadata observation with the actual journal
            # writer; unrelated cleanup is otherwise free to progress.
            with privacy.ledger._lock:
                selected = data_helpers.stored_steps(tmp_path, operation)
                producer, = [step for step in selected
                    if step["participant"] == "producers" and step["version"] == 1]
                assert not privacy.ledger.erasures.is_current_claim(old_claim)
                # Intentionally rejected protocol data, never a stand-in
                # successful callback for physical cleanup.
                assert not privacy.ledger.erasures.finish(old_claim, ErasureResult(True))
                assert data_helpers.stored_steps(tmp_path, operation) == selected
                return producer

        producer = await privacy.acall(reject_stale_claim)
        assert producer["state"] == "pending" and producer["claim_token"] is None
        # The real coordinator may already have observed that main has not
        # yet registered its new producer driver at this construction point.
        assert producer["last_error"] in {ErasureError.RESTARTED.value, ErasureError.UNAVAILABLE.value}
        assert before["voice_fingerprints"] and before["voice_capture_reviews"]
        assert all(row["audio_wav"] and row["transcript"] == "Synthetic reviewed clip"
            for row in before["voice_capture_reviews"])
        # Authority does not exist yet in main at this boundary. The nominal
        # ledger reader proves source filtering with the raw payload present.
        visible = await privacy.acall(VoiceLedger(privacy.ledger).snapshot)
        assert not set(subject.voice_ids) & {row["id"] for row in visible["fingerprints"]}
        assert await privacy.acall(selected_reviews) == before
        reader_checks.append(privacy)

    deleting, unlinking = data_helpers.NativeGate(), data_helpers.NativeGate()
    data_helpers.install_restore_gates(monkeypatch, subject, deleting, unlinking)
    async with AsyncExitStack() as stack:
        building = asyncio.create_task(stack.enter_async_context(application_graph(
            tmp_path, monkeypatch, clock, gates=(deleting, unlinking), before_voices=before_voices)))
        observing = None
        try:
            # Main may await semantic-index recovery before capture builders
            # exist. Observe those two real leaves in their actual order.
            await data_helpers.until(lambda: deleting.entered.is_set()
                or (building.done() and building.exception() is not None))
            if building.done() and building.exception() is not None:
                building.result()
            assert deleting.entered.is_set() and deleting.threads
            assert not deleting.returned.is_set()
            selected = await asyncio.to_thread(data_helpers.stored_steps, tmp_path, operation)
            assert any(step["participant"] == "semantic_index" and step["state"] != "complete"
                for step in selected)
            target = building
            if building.done():
                ready = building.result()
                eraser = ready.owner._drivers[ErasureParticipant.SEMANTIC_INDEX].callback.__self__
                pending = set(ready.memory.manager.long_term._operations) | set(eraser._work._tasks)
                target = next(task for task in pending if not task.done())
            entered = asyncio.Event()

            async def observe():
                entered.set()
                return await asyncio.shield(target)

            observing = asyncio.create_task(observe())
            await entered.wait()
            assert not observing.done()
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
            assert not deleting.returned.is_set()
            deleting.release.set()
            await data_helpers.until(lambda: unlinking.entered.is_set()
                or (building.done() and building.exception() is not None))
            if building.done() and building.exception() is not None:
                building.result()
            assert unlinking.entered.is_set() and unlinking.threads
            assert not unlinking.returned.is_set()
            assert Path(subject.archive_path).read_bytes() == saved.data["files"][str(subject.archive_path)]
            selected = await asyncio.to_thread(data_helpers.stored_steps, tmp_path, operation)
            assert any(step["participant"] == "captures" and step["state"] != "complete"
                for step in selected)
            unlinking.release.set()
            second = await building
            assert second.graph is not old_graph and second.privacy is not old_privacy
            assert second.owner is not old_owner and second.owner.privacy is second.privacy
            assert {part.value for part in second.owner._drivers} == replay.PARTICIPANTS
            assert second.graph.producer_erasure.inventory.verify().valid
            assert second.graph.producer_erasure.inventory is not old_inventory
            before_calls = list(second.models.leaf.calls)
            with pytest.raises(SourcesInvalidated, match="command_sources_invalidated"):
                await replay.launch(second.graph, subject.source, .125)
            assert second.models.leaf.calls == before_calls
            assert not second.processor.conversation_busy(subject.source.conversation_id)
            assert reader_checks == [second.privacy]
            assert second.reader.namespace == old_namespace
            # Only the original historical client ID is reused. Current
            # hello/reset/ACK is performed with Graph2's current real lease.
            await replay.clear_browser(second, subject)
            await replay.settle(second, operation)
            assert deleting.returned.is_set() and unlinking.returned.is_set()
            await replay.assert_subject_erased(second, subject)
            with pytest.raises(LookupError, match="voice_capture_unavailable"):
                await second.voices.acall(second.voices.ledger.review_capture,
                    subject.voice_ids[0], replay.voice_actor(second))
            await replay.assert_foreign_unchanged(second, foreign, foreign_before, rehydrated=True)
            await assert_manual_sentinels(second, manual)
            assert foreign.source.to_dict() == original_u
            assert second.home["service_calls"] == 0
        finally:
            # Release before either graph/owner ExitStack can wait on its leaf.
            deleting.release.set()
            unlinking.release.set()
            if observing is not None:
                await asyncio.gather(observing, return_exceptions=True)
            await asyncio.gather(building, return_exceptions=True)
    assert not second.privacy.resource_status()["writer_lease"]
    assert second.client.closed


async def test_snapshot_conflicting_with_current_manual_decisions_is_rejected_before_any_write(tmp_path, monkeypatch):
    clock = Clock(time.time())
    async with application_graph(tmp_path, monkeypatch, clock) as first:
        subject, foreign = await replay.seed_all(first)
        # This is a real automatic snapshot, before any profile or response
        # decision. Its old U rows must never overwrite later manual state.
        saved_u, schema = await replay.snapshot_payload(first, foreign)
        sources = await replay.manual_voice_inputs(first, foreign)
        manual = await seed_manual_sentinels(first, sources=sources,
            voice_ids=tuple(foreign.voice_ids), suggestion_ids=tuple(foreign.suggestion_ids),
            root_lease=replay.root_lease(first))
        foreign_before = await replay.payload(first, foreign)
        await assert_manual_sentinels(first, manual)
        operation = await first.owner.request(subject.source.conversation_id, replay.memory_actor(first))
        await replay.clear_browser(first, subject)
        await replay.settle(first, operation)
        await replay.assert_subject_erased(first, subject)
        old_privacy = first.privacy
    assert first.client.closed and not first.privacy.resource_status()["writer_lease"]

    current_index = await asyncio.to_thread(replay.read_index, tmp_path, saved_u, schema)
    before = await asyncio.to_thread(capture_preservation, tmp_path, index_state=current_index)
    with pytest.raises(ReplayRejected, match="restore_mixed_row_conflict"):
        await asyncio.to_thread(replay.restore_payload, tmp_path, saved_u, schema)
    after_index = await asyncio.to_thread(replay.read_index, tmp_path, saved_u, schema)
    after = await asyncio.to_thread(capture_preservation, tmp_path, index_state=after_index)
    assert_preserved(before, after)

    async with application_graph(tmp_path, monkeypatch, clock) as second:
        assert second.privacy is not old_privacy
        assert second.graph.producer_erasure.inventory.verify().valid
        assert {part.value for part in second.owner._drivers} == replay.PARTICIPANTS
        await replay.settle(second, operation)
        await replay.assert_subject_erased(second, subject)
        await replay.assert_foreign_unchanged(second, foreign, foreign_before, rehydrated=True)
        await assert_manual_sentinels(second, manual)
        assert second.home["service_calls"] == 0
    assert second.client.closed and not second.privacy.resource_status()["writer_lease"]
