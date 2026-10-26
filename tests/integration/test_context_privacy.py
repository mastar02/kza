"""Actual compaction, private files and catalogue revocations under late I/O."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import threading

import pytest

from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.services import build_memory_services
from src.orchestrator.compactor import Compactor
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.context_persister import ContextPersister, PERSISTED_VERSION
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


class Reasoner:
    _resolved_model = 'synthetic-test-model'

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def complete(self, *args, **kwargs):
        self.entered.set()
        await self.release.wait()
        return json.dumps({'summary': 'Resumen privado sintético.'})


@asynccontextmanager
async def graph(tmp_path):
    clock, reasoner = Clock(), Reasoner()
    async with AsyncExitStack() as stack:
        services = await build_memory_services({'memory': {
            'privacy_db_path': str(tmp_path/'privacy.db'),
            'preferences_path': str(tmp_path/'preferences.json')}}, shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        policy = services.manager.long_term.ledger
        persister = ContextPersister(tmp_path/'contexts', source_policy=policy, clock=clock)
        manager = ContextManager(persister=persister, source_policy=policy, clock=clock,
            io_worker=services.manager.long_term.acall,
            compactor=Compactor(reasoner), compaction_threshold=100, keep_recent_turns=0)
        stack.push_async_callback(manager.aclose)
        yield manager, persister, services.manager.long_term, clock, reasoner


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


async def add_summary(manager, clock, *, source=None):
    ctx = await manager.aget_or_create('test_user', 'Prueba')
    source = source or memory_source(now=clock.now)
    manager.add_turn('test_user', 'user', 'Texto privado sintético.', entities=['light.synthetic'], sources=(source,))
    await manager._compact_background('test_user')
    assert ctx.compacted_summary == 'Resumen privado sintético.'
    return ctx, source


async def test_raw_context_expires_at_source_deadline_in_every_projection(tmp_path):
    async with graph(tmp_path) as (manager, _, _, clock, _):
        source = memory_source(now=clock.now, ttl=60)
        ctx = await manager.aget_or_create('test_user')
        manager.add_turn('test_user', 'user', 'Texto privado.', entities=['light.synthetic'], sources=(source,))
        clock.now += 59
        assert 'Texto privado.' in manager.build_prompt('test_user', 'nuevo')
        assert ctx.get_recent_entities() == ['light.synthetic']
        clock.now += 1
        assert 'Texto privado.' not in manager.build_prompt('test_user', 'nuevo')
        assert all('Texto privado.' not in row['content'] for row in manager.build_chat_messages('test_user', 'nuevo'))
        assert ctx.get_history_for_prompt() == [] and ctx.get_recent_entities() == []
        assert ctx.to_dict()['conversation_history'] == []


async def test_recompaction_snapshot_and_restart_never_renew_summary_deadline(tmp_path):
    async with graph(tmp_path) as (manager, persister, _, clock, _):
        source = memory_source(now=clock.now, ttl=60)
        ctx, _ = await add_summary(manager, clock, source=source)
        clock.now += 30
        fresh = memory_source(now=clock.now)
        manager.add_turn('test_user', 'user', 'Otra frase.', sources=(fresh,))
        await manager._compact_background('test_user')
        assert len(ctx.summary_sources) == 2
        await manager._offload(persister.save, ctx)
        path = persister.base_path/'test_user.json'
        assert path.stat().st_mode & 0o777 == 0o600
        manager.remove_context('test_user')
        clock.now += 29
        ctx = await manager.aget_or_create('test_user')
        assert 'Resumen privado' in manager.build_prompt('test_user', 'nuevo')
        assert min(s.expires_at for s in ctx.summary_sources) == source.expires_at
        clock.now += 1
        assert 'Resumen privado' not in manager.build_prompt('test_user', 'nuevo')
        assert ctx.preserved_ids == []
        assert await manager._offload(persister.load, 'test_user') is None
        assert await manager._offload(persister.purge_expired) == 1 and not path.exists()


async def test_legacy_context_and_unsourced_policy_summary_are_not_hydrated(tmp_path):
    async with graph(tmp_path) as (manager, persister, _, clock, _):
        legacy = tmp_path/'contexts'/'test_user.json'
        legacy.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({'version': 1, 'user_id': 'test_user', 'compacted_summary': 'Texto legacy privado.'})
        legacy.write_text(original)
        assert (await manager.aget_or_create('test_user')).compacted_summary is None
        persister.base_path.mkdir(parents=True, exist_ok=True)
        path = persister.base_path/'test_user.json'
        path.write_text(json.dumps({'version': PERSISTED_VERSION, 'user_id': 'test_user',
            'last_seen': clock.now, 'compacted_summary': 'Texto sin fuentes.', 'preserved_ids': []}))
        assert await manager._offload(persister.load, 'test_user') is None
        await manager._offload(persister.purge_expired)
        assert legacy.read_text() == original


@pytest.mark.parametrize('action', ['expire', 'clear', 'replace_context', 'revoke'])
async def test_late_compaction_cannot_restore_expired_cleared_or_revoked_context(tmp_path, action):
    async with graph(tmp_path) as (manager, _, memory, clock, reasoner):
        source = memory_source(now=clock.now, ttl=10)
        await manager.aget_or_create('test_user')
        manager.add_turn('test_user', 'user', 'Texto que no debe volver.', sources=(source,))
        reasoner.release.clear()
        task = asyncio.create_task(manager._compact_background('test_user'))
        try:
            await asyncio.wait_for(reasoner.entered.wait(), 1)
            if action == 'expire':
                clock.now += 10
            elif action == 'clear':
                manager.clear_user_history('test_user')
            elif action == 'replace_context':
                manager.remove_context('test_user')
                await manager.aget_or_create('test_user')
            else:
                await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            reasoner.release.set()
            await task
            ctx = manager.get('test_user')
            assert ctx.compacted_summary is None and ctx.conversation_history == []
            assert not ctx.compaction_inflight and ctx.preserved_ids == []
            assert 'Texto que no debe volver.' not in manager.build_prompt('test_user', 'nuevo')
        finally:
            reasoner.release.set()
            await task


async def test_snapshot_queued_before_tombstone_is_rejected_after_native_wait(tmp_path, monkeypatch):
    async with graph(tmp_path) as (manager, persister, memory, clock, _):
        _, source = await add_summary(manager, clock)
        entered, release = threading.Event(), threading.Event()
        save = persister.save_payload

        def delayed(payload):
            entered.set()
            assert release.wait(3)
            save(payload)

        monkeypatch.setattr(persister, 'save_payload', delayed)
        task = asyncio.create_task(manager._snapshot_and_remove('test_user'))
        try:
            await wait_for(entered.is_set)
            await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            release.set()
            await task
            assert not (persister.base_path/'test_user.json').exists()
            assert manager.get('test_user').compacted_summary is None
        finally:
            release.set()
            await task


async def test_file_commit_finishes_before_tombstone_and_is_then_unreadable(tmp_path, monkeypatch):
    import os
    async with graph(tmp_path) as (manager, persister, memory, clock, _):
        ctx, source = await add_summary(manager, clock)
        entered, release = threading.Event(), threading.Event()
        replace = os.replace

        def blocked_replace(*args):
            entered.set()
            assert release.wait(3)
            return replace(*args)

        monkeypatch.setattr(os, 'replace', blocked_replace)
        saving = asyncio.create_task(manager._offload(persister.save, ctx))
        deleting = None
        try:
            await wait_for(entered.is_set)
            deleting = asyncio.create_task(memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin')))
            await asyncio.sleep(.02)
            assert not deleting.done()
            release.set()
            await saving
            await deleting
            assert await manager._offload(persister.load, 'test_user') is None
            assert await manager._offload(persister.purge_expired) == 1
        finally:
            release.set()
            await asyncio.gather(saving, *([deleting] if deleting else []), return_exceptions=True)


async def test_cancelled_hydration_stays_owned_without_holding_context_lock(tmp_path, monkeypatch):
    async with graph(tmp_path) as (manager, persister, memory, _, _):
        existing = await manager.aget_or_create('other_user')
        entered, release = threading.Event(), threading.Event()
        load = persister.load

        def delayed(user_id):
            entered.set()
            assert release.wait(3)
            return load(user_id)

        monkeypatch.setattr(persister, 'load', delayed)
        loading = asyncio.create_task(manager.aget_or_create('test_user'))
        try:
            await wait_for(entered.is_set)
            loading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loading
            assert memory.resource_status()['operations'] >= 1
            assert await asyncio.wait_for(asyncio.to_thread(manager.get, 'other_user'), .2) is existing
        finally:
            release.set()
            await wait_for(lambda: memory.resource_status()['operations'] == 0)


async def test_revocation_survives_fresh_policy_and_maintenance_purges_snapshot(tmp_path):
    async with graph(tmp_path) as (manager, persister, memory, clock, _):
        ctx, source = await add_summary(manager, clock)
        await manager._offload(persister.save, ctx)
        await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
        reopened = FactLedger(str(tmp_path/'privacy.db'), clock=clock)
        try:
            await memory.acall(reopened.initialize)
            recovered = ContextPersister(tmp_path/'contexts', source_policy=reopened, clock=clock)
            assert await memory.acall(recovered.load, 'test_user') is None
            await manager._cleanup_inactive_async()
            assert not (persister.base_path/'test_user.json').exists()
            assert manager.get('test_user').compacted_summary is None
        finally:
            await memory.acall(reopened.close)


async def test_revocation_removes_short_term_prompt_and_starts_a_new_capture_scope(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        from uuid import uuid4
        services = await build_memory_services({'memory': {'privacy_db_path': str(tmp_path/'privacy.db'),
            'preferences_path': str(tmp_path/'preferences.json')}}, shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        manager, memory = services.manager, services.manager.long_term
        source = manager.source_for_capture(str(uuid4()), clock.now, 'test_room')
        await manager.arecord_interaction('Texto privado anterior.', 'Respuesta.', sources=(source,))
        assert 'Texto privado anterior.' in await manager.aformat_context_for_prompt('texto')
        await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
        assert await manager.aformat_context_for_prompt('texto') == ''
        fresh = manager.source_for_capture(str(uuid4()), clock.now, 'test_room')
        assert fresh.conversation_id != source.conversation_id
        await manager.arecord_interaction('Intercambio posterior.', 'Respuesta.', sources=(fresh,))
        assert 'Intercambio posterior.' in await manager.aformat_context_for_prompt('texto')


async def test_standalone_cancelled_disk_read_remains_owned_until_close(tmp_path, monkeypatch):
    persister = ContextPersister(tmp_path/'contexts')
    manager = ContextManager(persister=persister)
    entered, release = threading.Event(), threading.Event()

    def delayed(user_id):
        entered.set()
        assert release.wait(3)
        return None

    monkeypatch.setattr(persister, 'load', delayed)
    reading = asyncio.create_task(manager.aget_or_create('test_user'))
    try:
        await wait_for(entered.is_set)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        assert len(manager._io_tasks) == 1
        closing = asyncio.create_task(manager.aclose())
        await asyncio.sleep(.01)
        assert not closing.done()
        release.set()
        await closing
        assert not manager._io_tasks
    finally:
        release.set()
        await manager.aclose()


async def test_final_compaction_keeps_context_reactivated_while_waiting(tmp_path):
    async with graph(tmp_path) as (manager, persister, _, clock, reasoner):
        original = await manager.aget_or_create('test_user')
        manager.add_turn('test_user', 'user', 'Primer intercambio.', sources=(memory_source(now=clock.now),))
        reasoner.release.clear()
        cleaning = asyncio.create_task(manager._snapshot_and_remove('test_user'))
        try:
            await asyncio.wait_for(reasoner.entered.wait(), 1)
            unchanged_wall_time = original.last_active
            assert await manager.aget_or_create('test_user') is original
            manager.add_turn('test_user', 'user', 'Mensaje nuevo en curso.', sources=(memory_source(now=clock.now),))
            # Reactivation is a revision, even if the wall clock repeats.
            original.last_active = unchanged_wall_time
            reasoner.release.set()
            assert await cleaning is False
            assert manager.get('test_user') is original
            assert any(t.content == 'Mensaje nuevo en curso.' for t in original.conversation_history)
            assert not (persister.base_path/'test_user.json').exists()
        finally:
            reasoner.release.set()
            await cleaning


async def test_failed_atomic_write_and_unlink_leave_a_cleanup_retry(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    async with graph(tmp_path) as (manager, persister, _, clock, _):
        ctx, _ = await add_summary(manager, clock)
        unlink = Path.unlink

        def failed_replace(*args):
            raise OSError('synthetic file promotion failure')

        def failed_unlink(path, *args, **kwargs):
            if path.name.startswith('.snapshot-'):
                raise OSError('synthetic temporary cleanup failure')
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, 'replace', failed_replace)
            patch.setattr(Path, 'unlink', failed_unlink)
            with pytest.raises(OSError):
                await manager._offload(persister.save, ctx)
        assert len(list(persister.base_path.glob('.snapshot-*'))) == 1
        assert await manager._offload(persister.load, 'test_user') is None
        assert await manager._offload(persister.purge_expired) == 1
        assert list(persister.base_path.iterdir()) == []


async def test_cached_context_needs_no_native_slot_when_all_workers_are_occupied(tmp_path):
    async with graph(tmp_path) as (manager, _, memory, _, _):
        existing = await manager.aget_or_create('test_user')
        await wait_for(lambda: memory.resource_status()['operations'] == 0)
        release = threading.Event()
        entered = []

        def occupied():
            entered.append(True)
            assert release.wait(3)

        tasks = [asyncio.create_task(memory.acall(occupied)) for _ in range(8)]
        try:
            await wait_for(lambda: len(entered) == 8)
            assert await manager.aget_or_create('test_user') is existing
            assert memory.resource_status()['operations'] == 8
        finally:
            release.set()
            await asyncio.gather(*tasks)
