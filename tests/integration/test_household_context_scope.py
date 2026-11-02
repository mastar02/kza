"""Household context keys cannot recover or mutate personal conversation state."""

from contextlib import AsyncExitStack
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import numpy as np
import pytest

from src.application import ApplicationFactories
from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.orchestrator.context_manager import ContextManager, UserContext
from src.orchestrator.context_persister import ContextPersister
from src.orchestrator.priority_queue import Priority, PriorityRequestQueue, RequestStatus
from src.pipeline.command_event import CommandEvent
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config, offline_home
from tests.mocks.audio_output import AudioBackend
from tests.mocks.llm_http import Endpoint, SSEBody, event
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_actual_factory_anonymous_voice_cannot_read_eligible_personal_data(tmp_path, monkeypatch, orchestrated):
    import src.main as main_module
    import src.tts.piper_tts as output_module
    import src.vectordb.chroma_sync as index_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, 'SentenceTransformer', lambda *a, **kw: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(output_module, 'sd', backend)
    endpoint = Endpoint(SSEBody([event('Respuesta sintética sobre el cielo.'), event(finish='stop')]))
    clients = []
    original_sync, original_async = httpx.Client, httpx.AsyncClient

    def models(request):
        assert request.url.host == '127.0.0.1' and request.url.path.endswith('/models')
        return httpx.Response(200, json={'object': 'list', 'data': [
            {'id': 'configured-model', 'object': 'model', 'created': 0, 'owned_by': 'test'}]})

    class SyncClient(original_sync):
        def __init__(self, *args, **kwargs):
            kwargs['transport'] = httpx.MockTransport(models)
            super().__init__(*args, **kwargs)
            clients.append(self)

    class AsyncClient(original_async):
        def __init__(self, *args, **kwargs):
            kwargs['transport'] = httpx.MockTransport(endpoint.handle)
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(httpx, 'Client', SyncClient)
    monkeypatch.setattr(httpx, 'AsyncClient', AsyncClient)
    real_build_memory = main_module.build_memory_services
    memory_client = MemoryClient(tmp_path / 'memory-index')

    async def build_memory(config, **kwargs):
        return await real_build_memory(config, chroma_client=memory_client, **kwargs)

    monkeypatch.setattr(main_module, 'build_memory_services', build_memory)
    stt, tts = OfflineSTT(text='Nexa explicame el cielo'), OfflineTTS()
    factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    try:
        async with offline_home() as (url, observations):
            config = offline_config(tmp_path, url)
            config['reasoner'].update(destination='local', http_base_url='http://127.0.0.1:1/v1',
                http_model='configured-model', api_style='chat')
            config['memory'].update(enabled=True, preferences_path=str(tmp_path / 'preferences.json'))
            config['orchestrator'].update(enabled=orchestrated, context={
                'compaction': {'enabled': True, 'model': 'configured-model'},
                'persistence': {'enabled': True, 'base_path': str(tmp_path / 'contexts')}})
            async with AsyncExitStack() as stack:
                stack.callback(memory_client.close)
                graph = await build_application(config, shutdown=stack, factories=factories)
                router = graph.pipeline.request_router
                manager = router.memory
                source = manager.source_for_capture(str(uuid4()), time.time(), 'private-fixture')
                markers = ('PERSONAL_FACT_572', 'PERSONAL_PREFERENCE_638', 'PERSONAL_HISTORY_391',
                           'PERSONAL_SUMMARY_824', 'PERSONAL_NAME_739')
                await manager.long_term.astore_fact(markers[0], 'fact', sources=(source,))
                proposal = await manager.long_term.acall(manager.long_term.propose, markers[1], (source,), category='preference')
                await manager.long_term.acall(manager.long_term.confirm, proposal['id'], MemoryReviewer('fixture-owner'))
                await manager.arecord_interaction(markers[2], 'Respuesta privada.', sources=(source,))
                eligible = await manager.aprompt_context('cielo')
                assert all(marker in eligible.text for marker in markers[:3])
                orchestrator = router._orchestrator
                persister = orchestrator._persister if orchestrator else ContextPersister(
                    tmp_path / 'contexts', source_policy=manager.long_term.ledger)
                assert persister is not None
                for user_id in ('unknown', 'known-profile'):
                    persister.save(UserContext(user_id, markers[4], compacted_summary=markers[3],
                        summary_sources=(source,), _source_policy=manager.long_term.ledger))
                    assert persister.load(user_id) is not None
                before = {p.name: p.read_bytes() for p in persister.base_path.glob('*.json')}
                assert len(before) == 2
                recalls = AsyncMock(wraps=manager.aprompt_context)
                monkeypatch.setattr(manager, 'aprompt_context', recalls)
                loads, saves = [], []
                original_load, original_save = persister.load, persister.save_payload

                def load(*args, **kwargs):
                    loads.append(True)
                    return original_load(*args, **kwargs)

                def save(*args, **kwargs):
                    saves.append(True)
                    return original_save(*args, **kwargs)

                monkeypatch.setattr(persister, 'load', load)
                monkeypatch.setattr(persister, 'save_payload', save)
                if orchestrator:
                    await orchestrator.start()
                capture = CommandEvent(np.full(16000, .1, dtype=np.float32), room_id='')
                result = await graph.pipeline.process_command(capture)
                assert result['success'] and result['intent'] == 'conversation'
                posts = [request for request in endpoint.requests if request.method == 'POST']
                assert len(posts) == 1
                assert all(marker not in posts[0].content.decode() for marker in markers)
                assert tts.texts and all(marker not in ' '.join(tts.texts) for marker in markers)
                assert len(backend.heard()) > 0 and observations['service_calls'] == 0
                recalls.assert_not_awaited()
                assert loads == [] and saves == []
                assert {p.name: p.read_bytes() for p in persister.base_path.glob('*.json')} == before
                turns = manager.short_term.get_recent()
                assert len(turns) == 2 and turns[-1].sources[0].capture_id == capture.capture_id
                assert turns[-1].sources[0].fact_dependencies == ()
                if orchestrator:
                    ctx = orchestrator.context_manager.get_household('')
                    assert ctx.ephemeral and ctx.conversation_history == []
                    assert orchestrator.context_manager.get('unknown') is None
            assert saves == []
            assert {p.name: p.read_bytes() for p in persister.base_path.glob('*.json')} == before
    finally:
        for client in clients:
            if isinstance(client, original_async):
                await client.aclose()
            else:
                client.close()


async def test_household_context_never_loads_compacts_or_saves_personal_snapshots(tmp_path, monkeypatch):
    persister = ContextPersister(tmp_path / 'contexts')
    private = UserContext('unknown', 'Synthetic private profile name',
        compacted_summary='Synthetic private persisted summary.',
        summary_sources=(memory_source(),), preserved_ids=['light.private'])
    persister.save(private)
    before = {p.name: p.read_bytes() for p in persister.base_path.glob('*.json')}
    assert len(before) == 1
    loads, saves = [], []
    original_load, original_save = persister.load, persister.save_payload

    def load(*args, **kwargs):
        loads.append(True)
        return original_load(*args, **kwargs)

    def save(*args, **kwargs):
        saves.append(True)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(persister, 'load', load)
    monkeypatch.setattr(persister, 'save_payload', save)
    compactor = AsyncMock()
    async with AsyncExitStack() as stack:
        manager = ContextManager(persister=persister, compactor=compactor, compaction_threshold=1)
        stack.push_async_callback(manager.aclose)
        ctx = manager.get_household('synthetic-room')
        manager.add_turn(ctx.user_id, 'user', 'Synthetic transient microphone content.', household_only=True)
        assert ctx.ephemeral and ctx.conversation_history == [] and ctx.compacted_summary is None
        assert loads == []
        await manager._compact_background(ctx.user_id)
        assert await manager._snapshot_and_remove(ctx.user_id)
        compactor.compact.assert_not_called()
        assert saves == []
        with pytest.raises(ValueError, match='household_context_is_not_persistent'):
            persister.build_payload(ctx)
        assert {p.name: p.read_bytes() for p in persister.base_path.glob('*.json')} == before


async def test_forged_personal_context_key_cannot_select_or_write_personal_history():
    manager = ContextManager()
    try:
        private = manager.get_or_create('known-profile', 'Synthetic private name',
            preferences={'private': 'Synthetic private preference.'})
        private.add_turn('user', 'Synthetic private history.', sources=(memory_source(),))
        private.compacted_summary = 'Synthetic private summary.'
        private.summary_sources = (memory_source(),)
        history = list(private.conversation_history)
        selected = manager.build_household_prompt_context('known-profile', 'Current public question')
        assert 'Synthetic private' not in selected.prompt.text
        assert selected.prompt.sources == () and selected.context is None
        assert not manager.prompt_available('known-profile', selected)
        manager.add_turn('known-profile', 'user', 'Unauthorized injection', household_only=True)
        assert private.conversation_history == history
        assert private.compacted_summary == 'Synthetic private summary.'
    finally:
        await manager.aclose()


async def test_poisoned_household_fields_cannot_enter_any_prompt_or_snapshot():
    manager = ContextManager()
    try:
        ctx = manager.get_household('room-a')
        old = manager.build_household_prompt_context(ctx.user_id, 'First question')
        private = UserContext('seed', 'Synthetic private profile')
        private.add_turn('user', 'Synthetic private history.', sources=(memory_source(),))
        ctx.conversation_history = private.conversation_history
        ctx.compacted_summary = 'Synthetic private summary.'
        ctx.summary_sources = (memory_source(),)
        ctx.preferences = {'private': 'Synthetic private preference.'}
        ctx.user_name = 'Synthetic private name'
        ctx.pending_confirmation = {'private': 'Synthetic private action.'}
        selected = manager.build_household_prompt_context(ctx.user_id, 'Second question')
        assert 'Synthetic private' not in selected.prompt.text and selected.prompt.sources == ()
        assert not manager.prompt_available(ctx.user_id, old)
        assert ctx.conversation_history == [] and ctx.pending_confirmation is None
        assert ctx.preferences == {} and ctx.user_name == 'Hogar' and ctx.permission_level == 0
        ctx.conversation_history = private.conversation_history
        ctx.compacted_summary = 'Synthetic private summary.'
        assert 'Synthetic private' not in str(manager.build_chat_messages(ctx.user_id, 'Third question'))
        assert ctx.conversation_history == [] and ctx.compacted_summary is None
        with pytest.raises(PermissionError, match='household_confirmation_requires_bounded_structure'):
            manager.set_pending_confirmation(ctx.user_id, {'private': 'Unbounded action'})
    finally:
        await manager.aclose()


async def test_household_rooms_expire_and_new_instances_cannot_hydrate_old_keys():
    first, second = ContextManager(inactive_timeout=1), ContextManager()
    try:
        room = first.get_household('room-a')
        other = first.get_household('room-b')
        missing = first.get_household()
        literal_default = first.get_household('default')
        assert len({c.user_id for c in (room, other, missing, literal_default)}) == 4
        with pytest.raises(PermissionError, match='household_context_cannot_be_persistent'):
            first.get_or_create(room.user_id)
        snapshot = first.build_household_prompt_context(room.user_id, 'Current question')
        room.last_active = time.time() - 2
        replacement = first.get_household('room-a')
        assert replacement is not room and not first.prompt_available(room.user_id, snapshot)
        rebuilt = second.get_household('room-a')
        assert rebuilt.user_id != replacement.user_id
        assert second.build_household_prompt_context(replacement.user_id, 'New question').context is None
    finally:
        await first.aclose()
        await second.aclose()


async def test_queue_room_ownership_cannot_cancel_another_room_or_backend_request():
    manager, queue = ContextManager(), PriorityRequestQueue()
    try:
        room_a, room_b = manager.get_household('room-a'), manager.get_household('room-b')
        first = queue.enqueue('unknown', 'First room request', Priority.LOW,
            context_key=room_a.user_id, zone_id='destination', memory_sources=(memory_source(),))
        assert queue.dequeue() is first
        second = queue.enqueue('unknown', 'Other room request', Priority.LOW,
            context_key=room_b.user_id, memory_sources=(memory_source(),))
        backend = queue.enqueue('unknown', 'Trusted backend request', Priority.LOW, memory_sources=(memory_source(),))
        assert first.status is RequestStatus.PROCESSING and not first.is_cancelled
        assert queue.cancel_user_request('unknown', context_key=room_b.user_id)
        assert second.is_cancelled and not first.is_cancelled and not backend.is_cancelled
        replacement = queue.enqueue('different-acoustic-label', 'Replacement from room A', Priority.LOW,
            context_key=room_a.user_id, memory_sources=(memory_source(),))
        assert first.is_cancelled and not replacement.is_cancelled and not backend.is_cancelled
        assert queue.cancel_user_request('unknown') and backend.is_cancelled
        assert not replacement.is_cancelled
        assert 'context_key' not in replacement.to_dict()
        assert room_a.user_id not in repr(replacement)
        assert all('context_key' not in row for row in queue.get_queue_status())
        assert manager.get_active_users() == []
        assert manager.get_user_stats(room_a.user_id) is None
        assert room_a.user_id not in str(manager.get_stats())
    finally:
        queue.discard_private()
        await manager.aclose()
