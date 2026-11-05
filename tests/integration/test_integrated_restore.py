"""Integrated local erasure and compatible restores, without global acceptance.

Manual voice profiles, external SDK allocations, unknown producers and physical
browser memory remain outside these receipts and remain visible blockers.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

import numpy as np
import pytest

from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import producer_context
from src.users.voice_ledger import VoiceLedger, VoiceReviewer
from tests.mocks.integrated_restore import (
    ACTOR, LOCAL, NativeGate, assert_foreign_unchanged, assert_honest_blockers,
    assert_subject_erased, clear_browser, install_restore_gates, integrated_graph, payload,
    restore_payload, rows, seed_all, settle, snapshot_payload, stored_steps,
    until,
)
from tests.mocks.memory_graph import Clock


pytestmark = pytest.mark.asyncio


async def test_integrated_population_keeps_unresolved_inventory_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "synthetic-integrated-reader-token")
    async with integrated_graph(tmp_path) as g:
        subject, foreign = await seed_all(g)
        saved = await snapshot_payload(g, subject)
        foreign_before = await payload(g, foreign)
        assert set(g.owner._drivers) and len(g.owner._drivers) == 9
        assert saved.data["files"] and saved.data["index"]["ids"] == [subject.fact]
        operation = await g.owner.request(subject.source.conversation_id, ACTOR)
        await clear_browser(g, subject)
        await settle(g, operation)
        await assert_honest_blockers(g, operation)
        await assert_subject_erased(g, subject)
        await assert_foreign_unchanged(g, foreign, foreign_before)


async def test_restored_nine_families_wait_for_two_actual_native_operations(tmp_path, monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "synthetic-integrated-reader-token")
    clock = Clock()
    async with integrated_graph(tmp_path, clock) as g:
        subject, foreign = await seed_all(g)
        saved = await snapshot_payload(g, subject)
        foreign_before = await payload(g, foreign)
        operation = await g.owner.request(subject.source.conversation_id, ACTOR)
        await clear_browser(g, subject)
        await settle(g, operation)
        await assert_honest_blockers(g, operation)
        await assert_subject_erased(g, subject)
        await assert_foreign_unchanged(g, foreign, foreign_before)
    # All clients/owners and the writer lease have left their public scopes.
    await asyncio.to_thread(restore_payload, tmp_path, saved)
    reader_checked = []

    async def before_voices(privacy):
        def observed_rows():
            return {table: [row for row in rows(tmp_path / "privacy.db", table)
                if row.get("id", row.get("fingerprint_id")) in subject.voice_ids]
                for table in ("voice_fingerprints", "voice_capture_reviews")}
        before = await privacy.acall(observed_rows)
        assert before["voice_fingerprints"] and before["voice_capture_reviews"]
        ledger = VoiceLedger(privacy.ledger)
        visible = await privacy.acall(ledger.snapshot)
        assert not set(subject.voice_ids) & {row["id"] for row in visible["fingerprints"]}
        for identity in subject.voice_ids:
            with pytest.raises(LookupError, match="voice_capture_unavailable"):
                await privacy.acall(ledger.review_capture, identity, VoiceReviewer("synthetic-reviewer"))
        assert await privacy.acall(observed_rows) == before
        reader_checked.append(True)

    deleting, unlinking = NativeGate(), NativeGate()
    install_restore_gates(monkeypatch, subject, deleting, unlinking)
    async with AsyncExitStack() as stack:
        building = asyncio.create_task(stack.enter_async_context(integrated_graph(tmp_path, clock,
            gates=(deleting, unlinking), before_voices=before_voices)))
        try:
            await until(lambda: (deleting.entered.is_set() and unlinking.entered.is_set())
                or (building.done() and building.exception() is not None))
            if building.done():
                building.result()
            assert deleting.entered.is_set() and unlinking.entered.is_set()
            selected = await asyncio.to_thread(stored_steps, tmp_path, operation)
            assert {step["participant"] for step in selected} == LOCAL | {"producers"}
            for participant in ("captures", "semantic_index"):
                assert any(step["participant"] == participant and step["state"] != "complete"
                    for step in selected)
            assert not deleting.returned.is_set() and not unlinking.returned.is_set()
            # Observe an actually pending setup or native owner task. Setup
            # may already have yielded while maintenance owns the held delete.
            target = building
            if building.done():
                ready = building.result()
                eraser = ready.owner._drivers[ErasureParticipant.SEMANTIC_INDEX].callback.__self__
                pending = set(ready.memory.manager.long_term._operations) | set(eraser._work._tasks)
                target = next(task for task in pending if not task.done())
            started = asyncio.Event()
            async def observe():
                started.set()
                return await asyncio.shield(target)
            observing = asyncio.create_task(observe())
            await started.wait()
            assert not observing.done()
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
            assert not deleting.returned.is_set()
            unlinking.release.set()
            await until(unlinking.returned.is_set)
            selected = await asyncio.to_thread(stored_steps, tmp_path, operation)
            assert any(step["participant"] == "semantic_index" and step["state"] != "complete"
                for step in selected)
            assert not deleting.returned.is_set()
            deleting.release.set()
            g = await building
            assert reader_checked == [True]
            await clear_browser(g, subject)
            await settle(g, operation)
            assert deleting.returned.is_set() and unlinking.returned.is_set()
            await assert_honest_blockers(g, operation)
            await assert_subject_erased(g, subject)
            await assert_foreign_unchanged(g, foreign, foreign_before, rehydrated=True)
        finally:
            # Also covers assertions before construction has yielded its graph.
            deleting.release.set()
            unlinking.release.set()
            await asyncio.gather(building, return_exceptions=True)


async def test_actual_command_producer_retains_shared_writer_lease_after_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "synthetic-integrated-reader-token")
    native = NativeGate()
    async with integrated_graph(tmp_path, stt_gate=native) as g:
        subject, _ = await seed_all(g)
        with producer_context("synthetic-room", (subject.source,)):
            command = asyncio.create_task(g.processor.process_command(
                np.full(16000, .1, np.float32), use_parallel=False))
        closing = None
        try:
            await until(native.entered.is_set)
            assert g.processor.conversation_busy(subject.source.conversation_id)
            operation = await g.owner.request(subject.source.conversation_id, ACTOR)
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
            assert not native.returned.is_set()
            assert g.processor.conversation_busy(subject.source.conversation_id)
            selected = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, operation["id"])
            assert selected["state"] != "complete"
            closing = g.privacy.begin_close()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(.05):
                    await asyncio.shield(closing)
            assert not closing.done() and not native.returned.is_set()
            assert g.privacy.resource_status()["writer_lease"]
            competitor = PrivacyServices(str(tmp_path / "privacy.db"), clock=g.clock)
            try:
                with pytest.raises(RuntimeError, match="writer_already_active"):
                    await competitor.acall(competitor.initialize)
            finally:
                await competitor.aclose()
        finally:
            native.release.set()
            await asyncio.gather(command, return_exceptions=True)
            if closing is not None:
                await closing
        assert native.returned.is_set()
        assert not g.processor.snapshot_operations()
        assert g.privacy.ledger._db is None and not g.privacy.resource_status()["writer_lease"]
