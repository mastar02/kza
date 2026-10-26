"""Prompt provenance through actual memory, queue, HTTP, context and disk."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import sqlite3
import threading

import pytest

from src.memory.fact_ledger import FactDependency, FactLedger, MemoryReviewer, MemorySource
from src.memory.prompt_context import PromptContext
from src.memory.transient import TransientCache
from src.orchestrator.cancellation import CancellationReason
from src.orchestrator.context_persister import ContextPersister
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import Priority, RequestStatus
from tests.integration.test_transient_privacy import graph, until
from tests.mocks.llm_http import event
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio
REVIEWER = MemoryReviewer('synthetic-admin')


@asynccontextmanager
async def owner_graph(tmp_path, monkeypatch, *, parts=None, spoken=False):
    async with graph(tmp_path) as (manager, memory, clock, stack):
        handler, engine, backend, endpoint, _ = voice_graph(monkeypatch, parts=parts)
        stack.push_async_callback(handler.aclose)
        owner = MultiUserOrchestrator(None, None, None, llm=handler.llm, memory_manager=manager,
                                     response_handler=handler if spoken else None)
        stack.push_async_callback(owner.stop)
        owner.context_manager.get_or_create('household')
        owner._running = True
        worker = asyncio.create_task(owner._process_queue())
        owner._processor_task = worker
        yield owner, manager, memory, clock, endpoint, handler, engine, backend


def reference(fact):
    return FactDependency(fact.id, fact.revision, fact.expires_at)


async def stored(memory, clock, text='Un dato privado sintético.'):
    source = memory_source(now=clock.now)
    fact_id = await memory.astore_fact(text, 'fact', .9, sources=(source,))
    fact = await memory.acall(memory.ledger.get, fact_id)
    async with asyncio.timeout(2):
        while any(job['fact_id'] == fact_id for job in await memory.acall(memory.ledger.jobs)):
            await memory.areconcile()
            await asyncio.sleep(.001)
    return source, fact


async def test_semantic_context_outlives_raw_capture_without_renewing_fact_deadline(tmp_path):
    async with graph(tmp_path) as (manager, memory, clock, _):
        source, fact = await stored(memory, clock)
        clock.now += 3*86400
        selected = await manager.aprompt_context('dato privado')
        assert fact.content in selected.text
        assert selected.facts == (reference(fact),) and selected.sources == ()
        capture = memory_source(now=clock.now)
        bound = selected.bind_sources((capture,))
        assert [s.capture_id for s in bound] == [capture.capture_id]
        assert source.capture_id not in [s.capture_id for s in bound]
        assert memory.ledger.sources_available(bound)
        assert (await memory.acall(memory.ledger.get, fact.id)).expires_at == fact.expires_at
        await memory.acall(memory.ledger.delete_conversation, source.conversation_id, REVIEWER)
        assert not memory.ledger.sources_available(bound)


async def test_confirmed_preference_has_receipt_and_retraction_invalidates_its_response(tmp_path):
    async with graph(tmp_path) as (manager, memory, clock, _):
        source = memory_source(now=clock.now)
        proposal = await memory.acall(memory.propose, 'color: azul', (source,), category='preference')
        fact = await memory.acall(memory.confirm, proposal['id'], REVIEWER)
        clock.now += 40*86400
        selected = await manager.aprompt_context('color')
        assert 'color: azul' in selected.text
        assert selected.facts[0].expires_at is None
        bound = selected.bind_sources((memory_source(now=clock.now),))
        assert memory.ledger.sources_available(bound)
        await memory.acall(memory.ledger.delete_fact, fact['id'], REVIEWER)
        assert not memory.ledger.sources_available(bound)


async def test_fact_retraction_cascades_across_semantic_derivations_and_restart(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, first = await stored(memory, clock)
        clock.now += 28*86400
        second_sources = PromptContext('', facts=(reference(first),)).bind_sources((memory_source(now=clock.now),))
        second_id = await memory.astore_fact('Derivado intermedio.', 'fact', sources=second_sources)
        second = await memory.acall(memory.ledger.get, second_id)
        third_sources = PromptContext('', facts=(reference(second),)).bind_sources((memory_source(now=clock.now),))
        third_id = await memory.astore_fact('Derivado transitivo.', 'fact', sources=third_sources)
        third = await memory.acall(memory.ledger.get, third_id)
        assert second.expires_at == third.expires_at == first.expires_at
        memory._collection.delete_error = RuntimeError('synthetic index unavailable')
        assert await memory.acall(memory.delete_fact, first.id, REVIEWER)
        assert not memory.ledger.sources_available(second_sources)
        assert not memory.ledger.sources_available(third_sources)
        assert await memory.acall(memory.ledger.get, second_id) is None
        assert await memory.acall(memory.ledger.get, third_id) is None
        assert {j['fact_id'] for j in await memory.acall(memory.ledger.jobs)} == {first.id, second_id, third_id}
        reopened = FactLedger(str(tmp_path/'privacy.db'), clock=clock)
        await asyncio.to_thread(reopened.initialize)
        try:
            assert not reopened.sources_available(third_sources)
            assert await asyncio.to_thread(reopened.get, third_id) is None
        finally:
            await asyncio.to_thread(reopened.close)
        with pytest.raises(PermissionError):
            await memory.astore_fact('Resultado tardío.', 'fact', sources=third_sources)


async def test_automatic_derivation_expires_at_parent_deadline_without_waiting_for_cleanup(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, fact = await stored(memory, clock)
        clock.now += 29*86400
        capture = memory_source(now=clock.now)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((capture,))
        assert bound[0].expires_at == fact.expires_at < capture.expires_at
        child_id = await memory.astore_fact('Derivado cercano al vencimiento.', 'fact', sources=bound)
        child = await memory.acall(memory.ledger.get, child_id)
        cache = TransientCache(4, source_policy=memory.ledger)
        response_sources = PromptContext('', facts=(reference(child),)).bind_sources((capture,))
        cache.put('privado', {'text': 'Respuesta privada.'}, response_sources)
        clock.now = fact.expires_at
        assert not memory.ledger.sources_available(response_sources)
        assert len(cache) == 0
        assert await memory.acall(memory.ledger.get, child.id) is None


async def test_generated_context_cannot_be_promoted_as_a_direct_voice_proposal(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, fact = await stored(memory, clock)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        with pytest.raises(PermissionError, match='direct_capture'):
            await memory.acall(memory.propose, 'Afirmación inferida.', bound)
        direct = await memory.acall(memory.propose, 'Nueva afirmación explícita.', (memory_source(now=clock.now),))
        assert direct['state'] == 'proposed'


async def test_receipts_are_denied_during_the_durable_commit_publication_window(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, fact = await stored(memory, clock)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        committed, release = threading.Event(), threading.Event()
        real = memory.ledger._db
        class CommitGate:
            def __getattr__(self, name):
                return getattr(real, name)
            def commit(self):
                real.commit()
                state = real.execute('SELECT state FROM facts WHERE id=?', (fact.id,)).fetchone()[0]
                if state == 'deleted' and not committed.is_set():
                    committed.set()
                    assert release.wait(3)
        memory.ledger._db = CommitGate()
        deleting = asyncio.create_task(memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER))
        try:
            assert await asyncio.to_thread(committed.wait, 2)
            assert not await asyncio.wait_for(asyncio.to_thread(memory.ledger.sources_available, bound), .2)
        finally:
            release.set()
            await deleting
            memory.ledger._db = real
        assert not memory.ledger.sources_available(bound)


@pytest.mark.parametrize('change', ['delete_fact', 'delete_conversation', 'expire'])
async def test_recalled_fact_invalidation_cancels_real_queued_generation(tmp_path, monkeypatch, change):
    async with owner_graph(tmp_path, monkeypatch, parts=[event('Respuesta parcial'), asyncio.Event()]) as (owner, _, memory, clock, endpoint, _, _, _):
        source, fact = await stored(memory, clock)
        if change == 'expire':
            clock.now = fact.expires_at-60
        capture = memory_source(now=clock.now)
        receipt = owner.queue.enqueue('household', 'Consulta privada.', Priority.LOW,
                                      memory_sources=(capture,))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 2)
        assert fact.content in endpoint.requests[-1].content.decode()
        assert any(reference(fact) in s.fact_dependencies for s in receipt.memory_sources)
        if change == 'delete_fact':
            await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
        elif change == 'delete_conversation':
            await memory.acall(memory.ledger.delete_conversation, source.conversation_id, REVIEWER)
        else:
            clock.now = fact.expires_at
            assert capture.expires_at > clock.now
        await until(lambda: receipt.status is RequestStatus.CANCELLED and endpoint.body.closed)
        assert receipt.cancellation_token.reason is CancellationReason.PRIVACY
        assert receipt.result is receipt.error_response is None
        assert owner.context_manager.get('household').conversation_history == []
        await until(lambda: owner.queue._current is None)
        assert not owner._processor_task.done()


async def test_late_memory_snapshot_cannot_send_a_retracted_fact_to_http(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, manager, memory, clock, endpoint, _, _, _):
        _, fact = await stored(memory, clock)
        selected = await manager.aprompt_context('consulta')
        ready, release = asyncio.Event(), asyncio.Event()
        async def delayed(query):
            ready.set()
            await release.wait()
            return selected
        monkeypatch.setattr(manager, 'aprompt_context', delayed)
        receipt = owner.queue.enqueue('household', 'Consulta nueva.', Priority.LOW,
                                      memory_sources=(memory_source(now=clock.now),))
        await asyncio.wait_for(ready.wait(), 1)
        await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
        release.set()
        await until(lambda: receipt.status is RequestStatus.CANCELLED and owner.queue._current is None)
        assert endpoint.requests == [] and endpoint.client is None
        assert receipt.result is None and not owner._processor_task.done()


@pytest.mark.parametrize('change', ['clear', 'replace', 'revoke'])
async def test_included_history_and_context_epoch_guard_active_generation(tmp_path, monkeypatch, change):
    async with owner_graph(tmp_path, monkeypatch, parts=[event(''), asyncio.Event()]) as (owner, _, memory, clock, endpoint, _, _, _):
        old = memory_source(now=clock.now)
        ctx = owner.context_manager.get('household')
        ctx.add_turn('user', 'Historial privado.', sources=(old,))
        receipt = owner.queue.enqueue('household', 'Consulta reciente.', Priority.LOW,
                                      memory_sources=(memory_source(now=clock.now),))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 2)
        assert old in receipt.memory_sources
        if change == 'clear':
            ctx.clear_history()
        elif change == 'replace':
            owner.context_manager.remove_context('household')
            owner.context_manager.get_or_create('household')
        else:
            await memory.acall(memory.ledger.delete_conversation, old.conversation_id, REVIEWER)
        await until(lambda: receipt.status is RequestStatus.CANCELLED and endpoint.body.closed)
        assert receipt.cancellation_token.reason is CancellationReason.PRIVACY
        assert not owner._processor_task.done()


async def test_success_keeps_fact_receipts_in_context_snapshot_and_cache(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, _, memory, clock, _, _, _, _):
        _, fact = await stored(memory, clock)
        capture = memory_source(now=clock.now)
        result = asyncio.create_task(owner.dispatcher._slow_path('Consulta.', 'household', None, None,
            Priority.LOW, None, 2, memory_sources=(capture,)))
        delivered = await result
        assert delivered.success and delivered.memory_sources
        ctx = owner.context_manager.get('household')
        assert all(any(reference(fact) in s.fact_dependencies for s in t.sources) for t in ctx.conversation_history)
        ctx.compacted_summary = 'Resumen privado.'
        ctx.summary_sources = delivered.memory_sources
        cache = TransientCache(4, source_policy=memory.ledger)
        cache.put('consulta', {'text': delivered.response}, delivered.memory_sources)
        persister = ContextPersister(tmp_path/'contexts', source_policy=memory.ledger, clock=clock)
        await memory.acall(persister.save, ctx)
        assert await memory.acall(persister.load, 'household')
        await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
        assert ctx.get_history_for_prompt() == []
        assert ctx.compacted_summary is None and len(cache) == 0
        assert await memory.acall(persister.load, 'household') is None
        assert await memory.acall(persister.purge_expired) == 1


async def test_existing_ledger_schema_adds_dependency_storage_without_inventing_sources(tmp_path):
    path = str(tmp_path/'old-ledger.db')
    ledger = FactLedger(path)
    source = memory_source()
    original = await asyncio.to_thread(ledger.store, 'Hecho previo.', 'fact', .9, (source,))
    proposed = await asyncio.to_thread(ledger.store, 'Confirmación previa.', 'fact', .9, (source,), proposed=True)
    confirmed = await asyncio.to_thread(ledger.confirm, proposed.id, REVIEWER)
    await asyncio.to_thread(ledger.close)
    def old_shape():
        with sqlite3.connect(path) as db:
            db.execute('DROP TABLE fact_dependencies')
            db.execute('ALTER TABLE fact_sources DROP COLUMN dependencies')
    await asyncio.to_thread(old_shape)
    reopened = FactLedger(path)
    try:
        assert await asyncio.to_thread(reopened.get, original.id) is None
        assert await asyncio.to_thread(reopened.get, original.id, review=True) is None
        assert reopened.facts_available((reference(original),)) is False
        def preserved():
            row = reopened._db.execute('SELECT content, state, expires_at FROM facts WHERE id=?', (original.id,)).fetchone()
            source_row = reopened._db.execute('SELECT capture_id, dependencies FROM fact_sources WHERE fact_id=?', (original.id,)).fetchone()
            return tuple(row), tuple(source_row)
        row, source_row = await asyncio.to_thread(preserved)
        assert row == (original.content, 'quarantined', original.expires_at)
        assert source_row == (source.capture_id, '[]')
        assert next(job for job in await asyncio.to_thread(reopened.jobs) if job['fact_id'] == original.id)['operation'] == 'delete'
        assert await asyncio.to_thread(reopened.get, confirmed.id) == confirmed

    finally:
        await asyncio.to_thread(reopened.close)


async def test_serialization_roundtrip_and_missing_authority_fail_closed(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, fact = await stored(memory, clock)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        assert MemorySource.from_dict(bound[0].to_dict()) == bound[0]
        cache = TransientCache(4, clock=clock)
        assert not cache.put('private', {'response': 'private'}, bound)
        changed = replace(reference(fact), revision=fact.revision+1)
        assert not memory.ledger.facts_available((changed,))


async def test_retraction_during_discovery_never_posts_the_prompt(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, _, memory, clock, endpoint, _, _, _):
        _, fact = await stored(memory, clock)
        handle = endpoint.handle
        async def retracted(request):
            response = await handle(request)
            if request.method == 'GET':
                await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
            return response
        monkeypatch.setattr(endpoint, 'handle', retracted)
        receipt = owner.queue.enqueue('household', 'Consulta.', Priority.LOW,
                                      memory_sources=(memory_source(now=clock.now),))
        await until(lambda: receipt.status is RequestStatus.CANCELLED and owner.queue._current is None)
        assert [request.method for request in endpoint.requests] == ['GET']
        assert endpoint.client.is_closed() and not owner._processor_task.done()


def command_event(clock):
    import numpy as np
    from src.pipeline.command_event import CommandEvent
    source = memory_source(now=clock.now)
    return CommandEvent(np.zeros(16000, dtype=np.float32), 'origin', timestamp=clock.now,
        captured_at=clock.now, capture_id=source.capture_id, conversation_id=source.conversation_id,
        source_expires_at=source.expires_at)


def request_router(manager, handler, *, owner=None, fast=None, privacy=None):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from src.pipeline.request_router import RequestRouter
    processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
        text='Nexa contame una historia corta', user=None, emotion=None, timings={}, stt_confidence=None)))
    ha = SimpleNamespace(get_domotics_entities=AsyncMock(return_value=[]))
    audio = MagicMock()
    audio.detect_source_zone.return_value = 'zone_origin'
    return RequestRouter(processor, handler, audio, memory_manager=manager,
        orchestrator=owner, orchestrator_enabled=owner is not None, ha_client=ha,
        chroma_sync=SimpleNamespace(search_command=lambda *args, **kwargs: None),
        fast_router=fast, privacy=privacy, wake_acoustically_confirmed=True,
        routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_real_voice_routes_record_full_lineage_and_forget_derived_interactions(tmp_path, monkeypatch, orchestrated):
    async with owner_graph(tmp_path, monkeypatch, spoken=True) as (owner, manager, memory, clock, _, handler, _, _):
        _, fact = await stored(memory, clock)
        router = request_router(manager, handler, owner=owner if orchestrated else None)
        try:
            result = await router.process_command(command_event(clock))
            assert result['success'] and result['response'] == 'Una respuesta completa.'
            recent = manager.short_term.get_recent()
            assert len(recent) == 1
            assert any(reference(fact) in s.fact_dependencies for s in recent[0].sources)
            await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
            assert manager.short_term.get_recent() == []
            if orchestrated:
                assert owner.context_manager.get('household') is not None
                assert all(not ctx.get_history_for_prompt() for ctx in owner.context_manager._contexts.values())
        finally:
            await router.aclose()


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_real_voice_routes_cancel_retracted_prompt_without_error_audio(tmp_path, monkeypatch, orchestrated):
    async with owner_graph(tmp_path, monkeypatch, spoken=True, parts=[event(''), asyncio.Event()]) as (owner, manager, memory, clock, endpoint, handler, engine, _):
        _, fact = await stored(memory, clock)
        router = request_router(manager, handler, owner=owner if orchestrated else None)
        try:
            pending = asyncio.create_task(router.process_command(command_event(clock)))
            await asyncio.wait_for(endpoint.body.waiting.wait(), 2)
            await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
            result = await asyncio.wait_for(pending, 2)
            assert not result['success'] and result['intent'] == 'cancelled'
            assert result['text'] == result['response'] == ''
            assert manager.short_term.get_recent() == []
            await until(lambda: endpoint.body.closed and endpoint.client.is_closed())
            assert not engine.calls and not owner._processor_task.done()
        finally:
            await router.aclose()


async def test_legacy_fast_reply_rejects_late_native_result_after_history_fact_retraction(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (_, manager, memory, clock, _, handler, engine, _):
        _, fact = await stored(memory, clock)
        old_sources = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        await manager.arecord_interaction('Consulta previa.', 'Respuesta previa privada.', sources=old_sources)
        entered, release = threading.Event(), threading.Event()
        class NativeRouter:
            context = None
            def classify_and_respond(self, text, *, context, max_tokens):
                self.context = context
                entered.set()
                assert release.wait(3)
                return False, 'Respuesta privada tardía.'
        native = NativeRouter()
        router = request_router(manager, handler, fast=native)
        pending = asyncio.create_task(router.process_command(command_event(clock)))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert 'Respuesta previa privada.' in native.context
            await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
            result = await asyncio.wait_for(pending, 1)
            assert result['intent'] == 'cancelled' and result['response'] == ''
            assert router.resource_status()['queries']['operations'] == 1
            release.set()
            await until(lambda: router.resource_status()['queries']['operations'] == 0)
            assert manager.short_term.get_recent() == [] and not engine.calls
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await router.aclose()


@pytest.mark.parametrize('durable', [False, True])
async def test_commit_error_reloads_actual_authority_instead_of_restoring_an_assumption(tmp_path, durable):
    async with graph(tmp_path) as (_, memory, clock, _):
        _, fact = await stored(memory, clock)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        real = memory.ledger._db
        class FailedCommit:
            def __getattr__(self, name):
                return getattr(real, name)
            def commit(self):
                state = real.execute('SELECT state FROM facts WHERE id=?', (fact.id,)).fetchone()[0]
                if state == 'deleted':
                    if durable:
                        real.commit()
                    raise OSError('synthetic ambiguous commit')
                real.commit()
        memory.ledger._db = FailedCommit()
        try:
            with pytest.raises(OSError, match='ambiguous commit'):
                await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
        finally:
            memory.ledger._db = real
        assert memory.ledger.sources_available(bound) is not durable
        assert (await memory.acall(memory.ledger.get, fact.id) is None) is durable


async def test_fact_receipts_survive_event_serialization_and_revoke_before_physical_cleanup(tmp_path):
    from src.analytics.services import build_event_logger
    async with graph(tmp_path) as (_, memory, clock, stack):
        _, fact = await stored(memory, clock)
        bound = PromptContext('', facts=(reference(fact),)).bind_sources((memory_source(now=clock.now),))
        events = await build_event_logger({'analytics': {'events_db': str(tmp_path/'events.db')}},
            shutdown=stack, source_policy=memory.ledger, clock=clock, io_worker=memory.acall)
        await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto derivado.', sources=bound)
        rows = await events.acall(events.get_events)
        assert rows[0].sources == bound and rows[0].trigger_phrase == 'Texto derivado.'
        await memory.acall(memory.ledger.delete_fact, fact.id, REVIEWER)
        assert await events.acall(events.get_events) == []
        await events.acall(events.cleanup_old_events)
        assert await events.acall(events.get_events) == []


async def test_a_competing_writer_cannot_run_schema_migration_before_obtaining_its_lease(tmp_path):
    import fcntl
    import os
    from unittest.mock import MagicMock
    from src.memory.long_term import LongTermMemory
    path = str(tmp_path/'private'/'old-ledger.db')
    ledger = FactLedger(path)
    await asyncio.to_thread(ledger.initialize)
    await asyncio.to_thread(ledger.close)
    def old_shape():
        with sqlite3.connect(path) as db:
            db.execute('DROP TABLE fact_dependencies')
            db.execute('ALTER TABLE fact_sources DROP COLUMN dependencies')
    await asyncio.to_thread(old_shape)
    lease = os.open(path+'.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client = MagicMock()
    candidate = LongTermMemory(client, ledger_path=path)
    try:
        with pytest.raises(RuntimeError, match='writer_already_active'):
            await asyncio.to_thread(candidate.initialize)
        def columns():
            with sqlite3.connect(path) as db:
                return {row[1] for row in db.execute('PRAGMA table_info(fact_sources)')}
        assert 'dependencies' not in await asyncio.to_thread(columns)
        client.get_or_create_collection.assert_not_called()
    finally:
        os.close(lease)
        await candidate.aclose()


async def test_matching_captures_merge_receipts_without_dropping_older_dependencies(tmp_path):
    from src.memory.prompt_context import merge_sources
    async with graph(tmp_path) as (_, memory, clock, _):
        _, first = await stored(memory, clock)
        _, second = await stored(memory, clock, 'Otro dato independiente.')
        capture = memory_source(now=clock.now)
        first_sources = PromptContext('', facts=(reference(first),)).bind_sources((capture,))
        second_sources = PromptContext('', facts=(reference(second),)).bind_sources((capture,))
        merged = merge_sources(first_sources, second_sources)
        assert len(merged) == 1
        assert set(merged[0].fact_dependencies) == {reference(first), reference(second)}
        assert merged[0].expires_at == min(first_sources[0].expires_at, second_sources[0].expires_at)
        await memory.acall(memory.ledger.delete_fact, first.id, REVIEWER)
        assert not memory.ledger.sources_available(merged)


async def test_privacy_monitor_retains_delivery_cleanup_through_repeated_cancellation():
    from src.memory.prompt_context import run_private
    admitted = [True]
    entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def delivery():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
    task = asyncio.create_task(run_private(delivery(), lambda: admitted[0]))
    await asyncio.wait_for(entered.wait(), 1)
    admitted[0] = False
    await asyncio.wait_for(closing.wait(), 1)
    task.cancel()
    task.cancel()
    await asyncio.sleep(.02)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_prior_summaries_are_not_hydrated_as_independent_and_keep_their_original_expiry(tmp_path):
    import json
    async with graph(tmp_path) as (_, memory, clock, _):
        source = memory_source(now=clock.now, ttl=60)
        persister = ContextPersister(tmp_path/'contexts', source_policy=memory.ledger, clock=clock)
        previous = tmp_path/'contexts'/'policy_v2'
        await asyncio.to_thread(previous.mkdir, parents=True)
        path = previous/'household.json'
        payload = {'version': 2, 'user_id': 'household', 'compacted_summary': 'Resumen anterior sin dependencias verificables.',
                   'preserved_ids': [], 'summary_sources': [source.to_dict()]}
        await asyncio.to_thread(path.write_text, json.dumps(payload))
        assert await memory.acall(persister.load, 'household') is None
        assert await memory.acall(persister.purge_expired) == 0 and path.exists()
        clock.now += 60
        assert await memory.acall(persister.purge_expired) == 1 and not path.exists()
