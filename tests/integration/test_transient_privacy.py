"""Real queue, classifier and vector-cache ownership under source revocation."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.memory.services import build_memory_services
from src.memory.transient import TransientCache
from src.memory.fact_ledger import MemoryReviewer
from src.nlu.llm_router import LLMCommandRouter
from src.orchestrator.cancellation import CancellationReason
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import PriorityRequestQueue, Priority, RequestStatus
from src.pipeline.command_event import CommandEvent
from src.pipeline.request_router import RequestRouter
from tests.mocks.home_control import selected_actions
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source
from tests.mocks.llm_http import event as sse_event
from tests.mocks.transient_work import Classifier, Index
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        services = await build_memory_services({'memory': {'privacy_db_path': str(tmp_path/'privacy.db'),
            'preferences_path': str(tmp_path/'preferences.json')}}, shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        yield services.manager, services.manager.long_term, clock, stack


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


@pytest.mark.parametrize('terminal', ['complete', 'fail', 'cancel'])
async def test_terminal_receipt_expires_without_renewing_and_releases_callback_closures(tmp_path, terminal):
    async with graph(tmp_path) as (_, memory, clock, _):
        source = memory_source(now=clock.now, ttl=60)
        queue = PriorityRequestQueue(source_policy=memory.ledger)
        notified = []
        request = queue.enqueue('synthetic-user', 'Texto privado.', memory_sources=(source,),
            on_complete=lambda _: notified.append('complete'), on_fail=lambda _: notified.append('fail'),
            on_cancel=lambda _: notified.append('cancel'))
        if terminal == 'complete':
            assert request.complete('Respuesta privada.')
        elif terminal == 'fail':
            request.fail('Error con contenido privado.')
        else:
            request.cancel()
        assert notified == [terminal] and request.on_complete is request.on_fail is request.on_cancel is None
        request.error_response = 'Otra copia privada.'
        clock.now += 59
        assert request.to_dict()['text'] == 'Texto privado.'
        clock.now += 1
        queue.prune_private()
        assert request.text == '' and request.result is request.error is request.error_response is None
        assert queue._user_requests == {} and queue.get_queue_status() == []
        assert not request.complete('Resultado tardío.') and notified == [terminal]


async def test_revoked_pending_request_is_cancelled_once_without_disturbing_another_capture(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        source, other = memory_source(now=clock.now), memory_source(now=clock.now)
        queue = PriorityRequestQueue(source_policy=memory.ledger)
        notices = []
        revoked = queue.enqueue('first', 'Texto por borrar.', memory_sources=(source,), on_cancel=lambda _: notices.append(True))
        live = queue.enqueue('other', 'Texto vigente.', memory_sources=(other,))
        await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
        assert [row['text'] for row in queue.get_queue_status()] == ['Texto vigente.']
        assert revoked.is_cancelled and revoked.cancellation_token.reason == CancellationReason.PRIVACY
        assert revoked.text == '' and not revoked.complete('No debe volver.') and notices == [True]
        assert live.private_available()


@pytest.mark.parametrize('action', ['expire', 'revoke'])
async def test_active_generation_cancels_and_scrubs_its_receipt_after_source_invalidation(tmp_path, monkeypatch, action):
    async with graph(tmp_path) as (manager, memory, clock, stack):
        hold = asyncio.Event()
        handler, _, _, endpoint, _ = voice_graph(monkeypatch, parts=[sse_event(''), hold])
        stack.push_async_callback(handler.aclose)
        owner = MultiUserOrchestrator(None, None, None, llm=handler.llm, response_handler=handler, memory_manager=manager)
        stack.push_async_callback(owner.stop)
        owner.context_manager.get_or_create('household')
        source = memory_source(now=clock.now, ttl=60)
        owner._running = True
        worker = asyncio.create_task(owner._process_queue())
        pending = asyncio.create_task(owner.dispatcher._slow_path('Texto privado.', 'household', None, 'zone_origin',
            Priority.LOW, None, 2, memory_sources=(source,)))
        try:
            await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
            receipt = owner.queue.get_current()
            if action == 'expire':
                clock.now += 60
            else:
                await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            result = await asyncio.wait_for(pending, 1)
            assert result.intent == 'cancelled' and result.response == ''
            await until(lambda: endpoint.body.closed and endpoint.client.is_closed())
            assert receipt.status == RequestStatus.CANCELLED and receipt.cancellation_token.reason == CancellationReason.PRIVACY
            assert receipt.text == '' and receipt.result is receipt.error_response is None
            assert owner.context_manager.get('household').conversation_history == []
        finally:
            hold.set()
            owner._running = False
            worker.cancel()
            await asyncio.gather(worker, pending, return_exceptions=True)


async def test_query_cache_keeps_original_sources_and_never_renews_by_access(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        source = memory_source(now=clock.now, ttl=60)
        cache = TransientCache(2, source_policy=memory.ledger)
        original = {'private': ['Contenido derivado.']}
        assert cache.put('Frase privada como clave.', original, (source,))
        original['private'].append('Mutación externa.')
        item = cache['Frase privada como clave.']; item['private'].append('Mutación del consumidor.')
        assert cache['Frase privada como clave.'] == {'private': ['Contenido derivado.']}
        clock.now += 59
        assert cache.get('Frase privada como clave.')
        clock.now += 1
        assert cache.get('Frase privada como clave.') is None and cache._entries == {}
        fresh = memory_source(now=clock.now)
        assert cache.put('Nueva clave.', {}, (fresh,))
        await memory.acall(memory.delete_conversation, fresh.conversation_id, MemoryReviewer('test-admin'))
        assert cache.get('Nueva clave.') is None and not cache.put('Resultado tardío.', {}, (fresh,))
        cache.close()
        assert not cache.put('Después del cierre.', {}, (memory_source(now=clock.now),))


async def test_late_cancellation_listener_runs_outside_token_lock_after_private_scrub(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, _):
        source = memory_source(now=clock.now, ttl=1)
        queue = PriorityRequestQueue(source_policy=memory.ledger)
        receipt = queue.enqueue('synthetic-user', 'Texto privado.', memory_sources=(source,))
        clock.now += 1
        queue.prune_private()
        observed = []

        def late(token):
            unlocked = token._lock.acquire(blocking=False)
            observed.append(unlocked)
            if unlocked:
                token._lock.release()
                receipt.prune_private()

        receipt.cancellation_token.on_cancel(late)
        assert observed == [True] and not receipt.cancellation_token._callbacks


async def test_classifier_history_uses_capture_time_and_cannot_be_recreated_after_its_window(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, stack):
        native = Classifier()
        classifier = LLMCommandRouter(native, source_policy=memory.ledger)
        stack.push_async_callback(classifier.aclose)
        source = memory_source(now=clock.now-119)
        classifier.record_command('Texto de una captura anterior.', 'turn_on', sources=(source,))
        assert 'captura anterior' in classifier._build_history_block()
        clock.now += 1
        assert 'captura anterior' not in classifier._build_history_block() and not classifier._history
        classifier.record_command('Texto de una captura anterior.', 'turn_on', sources=(source,))
        assert not classifier._history
        fresh = memory_source(now=clock.now)
        classifier.record_command('Texto nuevo.', 'turn_on', sources=(fresh,))
        await memory.acall(memory.delete_conversation, fresh.conversation_id, MemoryReviewer('test-admin'))
        assert 'Texto nuevo.' not in classifier._build_history_block() and not classifier._history


@pytest.mark.parametrize('revoked_dependency', ['input', 'history'])
async def test_classifier_discards_result_revoked_during_native_generation(tmp_path, revoked_dependency):
    async with graph(tmp_path) as (_, memory, clock, stack):
        native = Classifier(); native.release.clear()
        classifier = LLMCommandRouter(native, source_policy=memory.ledger)
        stack.push_async_callback(classifier.aclose)
        source = memory_source(now=clock.now)
        historical = memory_source(now=clock.now)
        classifier.record_command('Un comando anterior privado.', 'turn_on', sources=(historical,))
        pending = asyncio.create_task(classifier.classify('Texto privado.', sources=(source,)))
        try:
            await until(native.entered.is_set)
            revoked = source if revoked_dependency == 'input' else historical
            await memory.acall(memory.delete_conversation, revoked.conversation_id, MemoryReviewer('test-admin'))
            native.release.set()
            result = await pending
            assert result.is_command is None and result.raw_response == ''
            classifier.record_command('Resultado tardío.', 'turn_on', sources=(revoked,))
            assert 'Resultado tardío.' not in classifier._build_history_block()
        finally:
            native.release.set()
            await pending


async def test_classifier_timeout_retains_actual_worker_and_close_waits_for_it(tmp_path):
    async with graph(tmp_path) as (_, memory, clock, stack):
        native = Classifier(); native.release.clear()
        classifier = LLMCommandRouter(native, source_policy=memory.ledger, timeout_s=.01)
        stack.push_async_callback(classifier.aclose)
        source = memory_source(now=clock.now)
        try:
            result = await classifier.classify('Texto privado.', sources=(source,))
            assert result.is_command is None and native.entered.is_set()
            assert classifier.resource_status()['operations'] == 1
            closing = asyncio.create_task(classifier.aclose())
            await asyncio.sleep(.01)
            assert not closing.done()
            native.release.set()
            await closing
            assert classifier.resource_status()['operations'] == 0
            classifier.record_command('Después del cierre.', 'turn_on', sources=(source,))
            assert not classifier._history
        finally:
            native.release.set()


async def test_idle_router_maintenance_prunes_cache_history_and_receipts_without_reads(tmp_path):
    async with graph(tmp_path) as (manager, memory, clock, stack):
        classifier = LLMCommandRouter(Classifier(), source_policy=memory.ledger)
        orchestrator = MultiUserOrchestrator(None, None, None, memory_manager=manager)
        router = RequestRouter(None, None, None, memory_manager=manager,
            orchestrator=orchestrator, llm_command_router=classifier)
        stack.push_async_callback(router.aclose)
        source = memory_source(now=clock.now, ttl=60)
        router._memory_source.set(source)
        router._add_to_cache('Clave privada.', {'private': 'Valor privado.'})
        classifier.record_command('Comando privado.', 'turn_on', sources=(source,))
        receipt = orchestrator.queue.enqueue('synthetic-user', 'Petición privada.', memory_sources=(source,))
        assert receipt.complete('Respuesta privada.')
        router.start(); await asyncio.sleep(.01)
        clock.now += 60
        await until(lambda: not router._query_cache._entries and not classifier._history and not orchestrator.queue._user_requests)
        assert receipt.text == '' and receipt.result is None
        await router.aclose()
        assert router._private_maintenance_task.done() and router.resource_status()['closed']
        assert (await router.process_command(None))['intent'] == 'cancelled'


@pytest.mark.parametrize('orchestrated', [False, True])
async def test_successful_typed_control_records_its_actual_capture_once_in_either_route(tmp_path, orchestrated):
    from src.home_assistant.voice_resolver import HomeVoiceResolver
    from src.rooms.room_context import RoomConfig
    from tests.integration import test_typed_home_operations as typed
    from tests.mocks.responses import response_handler_mock
    async with graph(tmp_path) as (manager, memory, clock, stack):
        ha, actions = typed.typed_graph.__wrapped__()
        classifier = LLMCommandRouter(Classifier(), source_policy=memory.ledger)
        rooms = {'test_room': RoomConfig('test_room', 'Test room', 'salón', default_climate='climate.test_ac')}
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='Nexa poné el aire a 22 grados', user=None, emotion=None, timings={}, stt_confidence=None)))
        owner = MultiUserOrchestrator(None, ha, None, memory_manager=manager, home_actions=actions)
        router = RequestRouter(processor, response_handler_mock(), MagicMock(),
            memory_manager=manager, llm_command_router=classifier, orchestrator=owner,
            orchestrator_enabled=orchestrated, ha_client=ha, home_actions=actions,
            home_voice_resolver=HomeVoiceResolver(actions, rooms), wake_acoustically_confirmed=True)
        stack.push_async_callback(router.aclose)
        source = memory_source(now=clock.now, ttl=60)
        event = CommandEvent(np.zeros(16000, dtype=np.float32), 'test_room', timestamp=clock.now,
            captured_at=clock.now, capture_id=source.capture_id, conversation_id=source.conversation_id, source_expires_at=source.expires_at)
        result = await router.process_command(event)
        assert result['success'] and len(classifier._history) == 1
        entry = classifier._history[0]
        assert entry.sources[0].capture_id == event.capture_id and entry.timestamp == event.captured_at
        assert entry.expires_at == source.expires_at
        await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
        assert 'Poné el aire' not in classifier._build_history_block() and not classifier._history


@pytest.mark.parametrize('action', ['revoke', 'cancel', 'close'])
async def test_late_vector_lookup_cannot_refill_cache_or_issue_an_action(tmp_path, monkeypatch, action):
    async with graph(tmp_path) as (manager, memory, clock, stack):
        owner, engine, _, _, _ = voice_graph(monkeypatch)
        stack.push_async_callback(owner.aclose)
        index = Index()
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='nexa prende la luz', user=None, emotion=None, timings={}, stt_confidence=None)))
        audio = MagicMock(); audio.detect_source_zone.return_value = 'zone_origin'
        ha = MagicMock()
        router = RequestRouter(processor, owner, audio, orchestrator_enabled=False,
            memory_manager=manager, chroma_sync=index, ha_client=ha,
            home_actions=selected_actions(ha, ('light.synthetic',)),
            routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
        stack.push_async_callback(router.aclose)
        source = memory_source(now=clock.now)
        event = CommandEvent(np.zeros(16000, dtype=np.float32), None, timestamp=clock.now,
            captured_at=clock.now, capture_id=source.capture_id, conversation_id=source.conversation_id)
        pending = asyncio.create_task(router.process_command(event))
        closing = None
        try:
            await until(index.entered.is_set)
            if action == 'revoke':
                await memory.acall(memory.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            elif action == 'cancel':
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert router.resource_status()['queries']['operations'] == 1
            else:
                closing = asyncio.create_task(router.aclose())
                await asyncio.sleep(.01)
                assert not closing.done()
            index.release.set()
            if action != 'cancel':
                result = await pending
                assert result['intent'] == 'cancelled' and result['text'] == result['response'] == ''
            if closing:
                await closing
            assert not router._query_cache._entries and not engine.calls
            ha.call_service_once.assert_not_awaited()
        finally:
            index.release.set()
            await asyncio.gather(pending, *([closing] if closing else []), return_exceptions=True)
