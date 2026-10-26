"""Privacy authority survives disabled recall, late capture work and native drain."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer
from src.memory.long_term import MemoryCloseTimeout
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.memory.services import build_memory_services
from src.orchestrator.context_persister import ContextPersister
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import Priority, RequestStatus
from tests.integration.test_prompt_provenance import request_router, command_event
from tests.integration.test_transient_privacy import until
from tests.mocks.llm_http import event
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


def configuration(tmp_path, *, enabled=False):
    return {'memory': {'enabled': enabled, 'privacy_db_path': str(tmp_path/'privacy.db'),
                       'preferences_path': str(tmp_path/'preferences.json')}}


async def test_disabled_recall_keeps_private_authority_and_scope_fences_across_restart(tmp_path):
    clock = Clock()
    config = configuration(tmp_path)
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        client = MagicMock()
        assert await build_memory_services(config, shutdown=stack, privacy=privacy, chroma_client=client) is None
        first = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        same = privacy.source_for_capture(str(uuid4()), clock.now+1, 'origin')
        assert first.conversation_id == same.conversation_id
        await privacy.acall(privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        assert not privacy.sources_available((first,))
        with pytest.raises(PermissionError, match='precedes_conversation_revocation'):
            privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        clock.now += 2
        fresh = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        assert fresh.conversation_id != first.conversation_id and privacy.sources_available((fresh,))
        client.get_or_create_collection.assert_not_called()
        assert not (tmp_path/'preferences.json').exists()
        assert Path(privacy.ledger.path).stat().st_mode & 0o777 == 0o600
    assert privacy.maintenance.done() and privacy.resource_status()['closed']
    assert not privacy.resource_status()['writer_lease']
    async with AsyncExitStack() as stack:
        restored = await build_privacy_services(config, shutdown=stack, clock=clock)
        assert not restored.sources_available((first,))
        assert restored.ledger.conversation_deleted_at(first.conversation_id) == clock.now-2


async def test_enabled_memory_borrows_the_same_policy_without_releasing_its_lease(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        config = configuration(tmp_path, enabled=True)
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        memory = await build_memory_services(config, shutdown=stack, privacy=privacy,
                                              chroma_client=MemoryClient(tmp_path/'chroma'))
        assert memory.manager.long_term.ledger is privacy.ledger
        capture = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        same = memory.manager.source_for_capture(str(uuid4()), clock.now+1, 'origin')
        assert capture.conversation_id == same.conversation_id
        await memory.aclose()
        assert privacy.resource_status()['writer_lease']
        await privacy.acall(privacy.ledger.delete_conversation, capture.conversation_id, ACTOR)
        assert not privacy.sources_available((capture,))
        competitor = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
        try:
            with pytest.raises(RuntimeError, match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            await competitor.aclose()
    assert not privacy.resource_status()['writer_lease']


async def test_blocked_semantic_default_executor_cannot_starve_privacy_revocation(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        config = configuration(tmp_path, enabled=True)
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        memory = await build_memory_services(config, shutdown=stack, privacy=privacy,
                                              chroma_client=MemoryClient(tmp_path/'chroma'))
        source = memory_source(now=clock.now)
        loop = asyncio.get_running_loop()
        original = loop._default_executor
        single = ThreadPoolExecutor(max_workers=1, thread_name_prefix='synthetic-blocked-semantic')
        loop.set_default_executor(single)
        entered, release = threading.Event(), threading.Event()
        def native():
            entered.set()
            assert release.wait(3)
        blocked = asyncio.create_task(memory.manager.long_term.acall(native))
        try:
            await until(entered.is_set)
            await asyncio.wait_for(privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR), .5)
            assert not privacy.sources_available((source,)) and not blocked.done()
        finally:
            release.set()
            try:
                await blocked
            finally:
                loop.set_default_executor(original)
                await asyncio.to_thread(single.shutdown, wait=True)


async def test_shared_authority_keeps_lease_until_timed_out_native_borrower_actually_finishes(tmp_path):
    clock = Clock()
    config = configuration(tmp_path, enabled=True)
    privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=.02)
    await privacy.acall(privacy.initialize)
    entered, release = threading.Event(), threading.Event()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(privacy.aclose)
        memory = await build_memory_services(config, shutdown=stack, privacy=privacy,
                                              chroma_client=MemoryClient(tmp_path/'chroma'))
        memory.manager.long_term.shutdown_timeout_s = .02
        def native():
            entered.set()
            assert release.wait(3)
        blocked = asyncio.create_task(memory.manager.long_term.acall(native))
        try:
            await until(entered.is_set)
            with pytest.raises(MemoryCloseTimeout):
                await memory.aclose()
            with pytest.raises(WorkCloseTimeout, match='privacy_resources_still_running'):
                await privacy.aclose()
            assert privacy.resource_status()['writer_lease'] and privacy.ledger._db is not None
            competitor = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
            try:
                with pytest.raises(RuntimeError, match='writer_already_active'):
                    await competitor.acall(competitor.initialize)
            finally:
                await competitor.aclose()
            assert not privacy.sources_available((memory_source(now=clock.now),))
        finally:
            release.set()
            try:
                await blocked
            finally:
                await privacy.aclose()
    assert privacy.ledger._db is None and not privacy.resource_status()['writer_lease']


async def test_disabled_memory_still_revokes_queue_context_and_snapshot(tmp_path, monkeypatch):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        handler, _, _, endpoint, _ = voice_graph(monkeypatch, parts=[event(''), asyncio.Event()])
        stack.push_async_callback(handler.aclose)
        persister = ContextPersister(tmp_path/'contexts', source_policy=privacy.ledger, clock=clock)
        owner = MultiUserOrchestrator(None, None, None, llm=handler.llm, privacy=privacy, persister=persister)
        stack.push_async_callback(owner.stop)
        ctx = await owner.context_manager.aget_or_create('household')
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        ctx.compacted_summary, ctx.summary_sources = 'Resumen privado.', (source,)
        await privacy.acall(persister.save, ctx)
        owner._running = True
        owner._processor_task = asyncio.create_task(owner._process_queue())
        receipt = owner.queue.enqueue('household', 'Consulta nueva.', Priority.LOW,
                                      memory_sources=(memory_source(now=clock.now),))
        await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
        await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await until(lambda: receipt.status is RequestStatus.CANCELLED and endpoint.body.closed)
        assert ctx.to_dict()['compacted_summary'] is None
        assert await privacy.acall(persister.load, 'household') is None
        assert await privacy.acall(persister.purge_expired) == 1
        assert not owner._processor_task.done()


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_disabled_memory_voice_routes_reject_revoked_capture_before_processing(tmp_path, monkeypatch, orchestrated):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        handler, engine, _, _, _ = voice_graph(monkeypatch)
        stack.push_async_callback(handler.aclose)
        owner = MultiUserOrchestrator(None, None, None, privacy=privacy) if orchestrated else None
        if owner:
            stack.push_async_callback(owner.stop)
        router = request_router(None, handler, owner=owner, privacy=privacy)
        stack.push_async_callback(router.aclose)
        event = command_event(clock)
        await privacy.acall(privacy.ledger.delete_conversation, event.conversation_id, ACTOR)
        result = await router.process_command(event)
        assert result['intent'] == 'cancelled' and result['text'] == result['response'] == ''
        router.command_processor.process_command.assert_not_awaited()
        assert not engine.calls


async def test_disabled_memory_ambient_reads_and_late_attachment_follow_the_same_tombstone(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        audio = tmp_path/'audio'
        audio.mkdir()
        store = AmbientStore(str(tmp_path/'ambient.db'), audio_dir=str(audio), clock=clock, source_policy=privacy.ledger)
        await store.init()
        stack.push_async_callback(store.close)
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin', origin=MemoryOrigin.AMBIENT)
        utt = AmbientUtterance('origin', clock.now, clock.now+1, text='Texto ambiental privado.', vad_prob=.9,
            capture_id=source.capture_id, conversation_id=source.conversation_id, expires_at=source.expires_at)
        row = await store.add(utt)
        path = audio/f'{row}.flac'
        path.write_bytes(b'synthetic private audio')
        assert await store.set_audio_path(row, str(path))
        assert await store.read_audio(row) == b'synthetic private audio'
        await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        assert await store.utterances_between('origin', clock.now-1, clock.now+2) == []
        assert await store.undistilled_live() == [] and await store.read_audio(row) is None
        assert await store.purge_expired() == 1 and not path.exists()
        late = audio/'late.flac'
        late.write_bytes(b'late private audio')
        assert not await store.set_audio_path(row, str(late)) and not late.exists()
        with pytest.raises(PermissionError, match='conversation_deleted'):
            await store.add(AmbientUtterance('origin', clock.now, clock.now+1, text='Resultado tardío.',
                                            conversation_id=source.conversation_id))


async def test_privacy_only_factory_does_not_import_chroma_or_load_preferences(tmp_path):
    code = textwrap.dedent('''
        import asyncio
        import importlib.abc
        import sys
        from contextlib import AsyncExitStack
        class NoSemanticModules(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'chromadb' or fullname.startswith('chromadb.'):
                    raise AssertionError('Disabled memory imported Chroma')
        sys.meta_path.insert(0, NoSemanticModules())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.services import build_memory_services
        async def check():
            config = {'memory': {'enabled': False, 'privacy_db_path': sys.argv[1]}}
            async with AsyncExitStack() as stack:
                privacy = await build_privacy_services(config, shutdown=stack)
                assert await build_memory_services(config, shutdown=stack, privacy=privacy) is None
                assert privacy.ledger._db is not None
            assert privacy.ledger._db is None and not privacy.resource_status()['writer_lease']
        asyncio.run(check())
    ''')
    completed = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', code, str(tmp_path/'privacy.db')],
        check=False, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('change', ['revoke', 'close'])
async def test_ambient_capture_is_bound_before_stt_and_late_text_does_not_reappear(tmp_path, change):
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.ambient.transcriber import AmbientTranscriber
    from src.ambient.types import RawSegment
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        store = AmbientStore(str(tmp_path/'ambient.db'), clock=clock, source_policy=privacy.ledger)
        await store.init()
        stack.push_async_callback(store.close)
        original = privacy.source_for_capture(str(uuid4()), clock.now, 'origin', origin=MemoryOrigin.AMBIENT)
        entered, release = asyncio.Event(), asyncio.Event()
        async def transcribe(audio):
            entered.set()
            await release.wait()
            return SimpleNamespace(text='Texto privado tardío.', avg_logprob=-.1, no_speech_prob=.01)
        tagger = SimpleNamespace(tag=AsyncMock(return_value=('unknown', .0)))
        transcriber = AmbientTranscriber(None, None, SimpleNamespace(transcribe=transcribe), tagger,
            SimpleNamespace(estimate=lambda audio: None), SimpleNamespace(classify=lambda **kw: 'live'),
            store, ['origin'], privacy=privacy)
        segment = RawSegment(clock.now, clock.now+1, np.zeros(16000, dtype=np.float32), vad_prob=.9)
        pending = asyncio.create_task(transcriber._handle_segment('origin', segment))
        await asyncio.wait_for(entered.wait(), 1)
        if change == 'revoke':
            await privacy.acall(privacy.ledger.delete_conversation, original.conversation_id, ACTOR)
        else:
            await privacy.aclose()
        release.set()
        await pending
        tagger.tag.assert_not_awaited()
        assert await store.utterances_between('origin', clock.now-1, clock.now+2) == []


async def test_ambient_cursor_cannot_return_text_retracted_while_sql_read_was_pending(tmp_path, monkeypatch):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        store = AmbientStore(str(tmp_path/'ambient.db'), clock=clock, source_policy=privacy.ledger)
        await store.init()
        stack.push_async_callback(store.close)
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin', origin=MemoryOrigin.AMBIENT)
        await store.add(AmbientUtterance('origin', clock.now, clock.now+1, text='Texto por revocar.',
                                        conversation_id=source.conversation_id))
        ready, release = asyncio.Event(), asyncio.Event()
        execute = store._db.execute
        async def delayed(sql, *args, **kwargs):
            cursor = await execute(sql, *args, **kwargs)
            if sql.startswith('SELECT * FROM utterances WHERE room_id='):
                ready.set()
                await release.wait()
            return cursor
        monkeypatch.setattr(store._db, 'execute', delayed)
        reading = asyncio.create_task(store.utterances_between('origin', clock.now-1, clock.now+2))
        await asyncio.wait_for(ready.wait(), 1)
        await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        release.set()
        assert await reading == []


async def test_revoked_clip_cleanup_failure_stays_durable_after_restart(tmp_path, monkeypatch):
    clock = Clock()
    config = configuration(tmp_path)
    audio = tmp_path/'audio'
    audio.mkdir()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        store = AmbientStore(str(tmp_path/'ambient.db'), audio_dir=str(audio), clock=clock, source_policy=privacy.ledger)
        await store.init()
        stack.push_async_callback(store.close)
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin', origin=MemoryOrigin.AMBIENT)
        row = await store.add(AmbientUtterance('origin', clock.now, clock.now+1, text='Texto privado.', conversation_id=source.conversation_id))
        path = audio/f'{row}.flac'
        path.write_bytes(b'synthetic audio')
        assert await store.set_audio_path(row, str(path))
        unlink = Path.unlink
        def denied(target, *args, **kwargs):
            if target == path:
                raise PermissionError('synthetic unlink unavailable')
            return unlink(target, *args, **kwargs)
        with monkeypatch.context() as patcher:
            patcher.setattr(Path, 'unlink', denied)
            await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            assert await store.purge_expired() == 1
            assert (await store.audio_cleanup_status())['pending'] == 1 and path.exists()
        await store.close()
    async with AsyncExitStack() as stack:
        restored = await build_privacy_services(config, shutdown=stack, clock=clock)
        store = AmbientStore(str(tmp_path/'ambient.db'), audio_dir=str(audio), clock=clock, source_policy=restored.ledger)
        await store.init()
        stack.push_async_callback(store.close)
        assert await store.read_audio(row) is None
        await store.purge_expired()
        assert not path.exists() and (await store.audio_cleanup_status())['pending'] == 0


async def test_cancelled_initialization_keeps_cleanup_registered_until_native_exit(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    instances = []
    initialize = PrivacyServices.initialize
    def blocked(self):
        instances.append(self)
        started.set()
        assert release.wait(3)
        return initialize(self)
    monkeypatch.setattr(PrivacyServices, 'initialize', blocked)
    stack = AsyncExitStack()
    building = asyncio.create_task(build_privacy_services(configuration(tmp_path), shutdown=stack))
    await until(started.is_set)
    building.cancel()
    with pytest.raises(asyncio.CancelledError):
        await building
    privacy = instances[0]
    privacy._close_timeout_s = .02
    try:
        with pytest.raises(WorkCloseTimeout):
            await stack.aclose()
        assert privacy.resource_status()['operations'] == 1
    finally:
        release.set()
        await privacy.aclose()
    assert privacy.ledger._db is None and not privacy.resource_status()['writer_lease']


async def test_uninitialized_authority_cannot_be_bypassed_by_optional_memory(tmp_path):
    policy = PrivacyServices(str(tmp_path/'privacy.db'))
    client = MagicMock()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(policy.aclose)
        with pytest.raises(RuntimeError, match='authority_unavailable'):
            await build_memory_services(configuration(tmp_path, enabled=True), shutdown=stack,
                                        privacy=policy, chroma_client=client)
        client.get_or_create_collection.assert_not_called()
        assert not (tmp_path/'privacy.db').exists()


async def test_closed_authority_does_not_authorize_stale_receipts_or_treat_unknown_state_as_erased(tmp_path):
    import sqlite3
    from src.analytics.services import build_event_logger
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        events = await build_event_logger({'analytics': {'events_db': str(tmp_path/'events.db')}},
            shutdown=stack, source_policy=privacy.ledger, clock=clock)
        await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto privado.', sources=(source,))
        await privacy.aclose()
        assert not privacy.ledger.sources_available((source,))
        with pytest.raises(sqlite3.OperationalError):
            await events.acall(events.get_events)
        with pytest.raises(sqlite3.OperationalError):
            await events.acall(events.cleanup_old_events)
        def count():
            with sqlite3.connect(tmp_path/'events.db') as db:
                return db.execute('SELECT COUNT(*) FROM events_policy_v1').fetchone()[0]
        assert await asyncio.to_thread(count) == 1


async def test_runtime_privacy_factory_cannot_disable_durable_tombstones_with_an_in_memory_path():
    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match='requires_persistent_file'):
            await build_privacy_services({'memory': {'enabled': False, 'privacy_db_path': ':memory:'}}, shutdown=stack)


async def test_post_revocation_scope_rotation_does_not_relabel_older_capture_as_new(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        before = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        await privacy.acall(privacy.ledger.delete_conversation, before.conversation_id, ACTOR)
        clock.now += 1
        fresh = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        assert fresh.conversation_id != before.conversation_id
        with pytest.raises(PermissionError, match='precedes_conversation_revocation'):
            privacy.source_for_capture(str(uuid4()), before.captured_at, 'origin')
        clock.now += 1
        later = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        assert later.conversation_id == fresh.conversation_id


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_disabling_semantic_recall_keeps_fresh_conversation_and_single_voice_delivery(tmp_path, monkeypatch, orchestrated):
    clock = Clock()
    async with AsyncExitStack() as stack:
        config = configuration(tmp_path)
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        assert await build_memory_services(config, shutdown=stack, privacy=privacy) is None
        handler, engine, _, _, _ = voice_graph(monkeypatch)
        stack.push_async_callback(handler.aclose)
        owner = MultiUserOrchestrator(None, None, None, llm=handler.llm,
            response_handler=handler, privacy=privacy) if orchestrated else None
        if owner is not None:
            stack.push_async_callback(owner.stop)
            owner._running = True
            owner._processor_task = asyncio.create_task(owner._process_queue())
        router = request_router(None, handler, owner=owner, privacy=privacy)
        stack.push_async_callback(router.aclose)
        result = await router.process_command(command_event(clock))
        assert result['success'] and result['response'] == 'Una respuesta completa.'
        assert len(engine.calls) == 1
        assert await privacy.acall(privacy.ledger.list) == []
        if owner is not None:
            ctx = owner.context_manager.get('unknown')
            assert len(ctx.conversation_history) == 2
            assert owner.context_manager.source_policy is privacy.ledger


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_disabling_semantic_recall_keeps_authorized_local_control(tmp_path, orchestrated):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.home_assistant.voice_resolver import HomeVoiceResolver
    from src.pipeline.request_router import RequestRouter
    from src.rooms.room_context import RoomConfig
    from tests.integration import test_typed_home_operations as typed
    from tests.mocks.responses import response_handler_mock
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        ha, actions = typed.typed_graph.__wrapped__()
        rooms = {'origin': RoomConfig('origin', 'Origin', 'salón', default_climate='climate.test_ac')}
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='Nexa poné el aire a 22 grados', user=None, emotion=None, timings={}, stt_confidence=None)))
        owner = MultiUserOrchestrator(None, ha, None, privacy=privacy, home_actions=actions) if orchestrated else None
        if owner is not None:
            stack.push_async_callback(owner.stop)
        router = RequestRouter(processor, response_handler_mock(), MagicMock(), privacy=privacy,
            orchestrator=owner, orchestrator_enabled=orchestrated, ha_client=ha, home_actions=actions,
            home_voice_resolver=HomeVoiceResolver(actions, rooms), wake_acoustically_confirmed=True)
        stack.push_async_callback(router.aclose)
        result = await router.process_command(command_event(clock))
        assert result['success'] and len(ha.sends) == 1
        assert ha.sends[0] == ('climate', 'set_temperature', 'climate.test_ac', {'temperature': 22.0})


@pytest.mark.parametrize('operation', ['load', 'purge_expired'])
async def test_snapshot_unknown_authority_is_unavailable_and_preserves_the_existing_file(tmp_path, operation):
    from src.orchestrator.context_persister import PERSISTED_VERSION
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(configuration(tmp_path), shutdown=stack, clock=clock)
        source = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        persister = ContextPersister(tmp_path/'contexts', source_policy=privacy.ledger, clock=clock)
        payload = {'version': PERSISTED_VERSION, 'user_id': 'household', 'user_name': 'Household',
                   'compacted_summary': 'Resumen privado.', 'preserved_ids': [], 'summary_sources': [source.to_dict()]}
        await privacy.acall(persister.save_payload, payload)
        path = persister.base_path/'household.json'
        assert path.exists()
        await privacy.aclose()
        with pytest.raises(RuntimeError, match='authority_unavailable'):
            await asyncio.to_thread(getattr(persister, operation), *(['household'] if operation == 'load' else []))
        assert path.exists()


async def test_unbound_preboot_capture_cannot_receive_a_fresh_conversation_after_restart(tmp_path):
    clock = Clock()
    config = configuration(tmp_path)
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        original = privacy.source_for_capture(str(uuid4()), clock.now, 'origin')
        await privacy.acall(privacy.ledger.delete_conversation, original.conversation_id, ACTOR)
    clock.now += 1
    async with AsyncExitStack() as stack:
        restored = await build_privacy_services(config, shutdown=stack, clock=clock)
        with pytest.raises(PermissionError, match='precedes_privacy_start'):
            restored.source_for_capture(original.capture_id, original.captured_at, 'origin')
        fresh = restored.source_for_capture(str(uuid4()), clock.now, 'origin')
        assert restored.sources_available((fresh,))
