"""Actual queue workers publish prompt affinity to finite response receivers."""

import asyncio
from contextlib import AsyncExitStack, ExitStack
import json
import threading
from uuid import uuid4
from weakref import ref

import pytest

from src.audio.zone_manager import Zone
from src.memory.producer_scopes import ProducerScopes
from src.memory.services import build_memory_services
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import Priority
from tests.integration.test_capture_receiver_scopes import graph as capture_graph, install_poll
from tests.integration.test_prompt_provenance import REVIEWER, owner_graph, reference, request_router, stored
from tests.integration.test_transient_privacy import until
from tests.mocks.llm_http import event
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


class Receiver:
    def snapshot(self):
        return ()


async def dispatch(owner, source):
    return await owner.dispatcher._slow_path('Consulta privada sintética.', 'household', None,
        'origin', Priority.LOW, None, 2, memory_sources=(source,))


async def test_independent_queue_worker_publishes_fact_sources_to_receiver_before_response(tmp_path, monkeypatch):
    hold = asyncio.Event()
    async with owner_graph(tmp_path, monkeypatch, parts=[event(''), hold, event('Respuesta completa.'), event(finish='stop')]) as (owner, _, memory, clock, endpoint, _, _, _):
        origin, fact = await stored(memory, clock)
        capture = memory_source(now=clock.now)
        receiver = Receiver()
        scopes = ProducerScopes(receiver.snapshot)
        ready, finish = asyncio.Event(), asyncio.Event()
        receipts = []

        async def consume():
            with scopes.operation((capture,)) as scope:
                receipts.append(scope)
                result = await dispatch(owner, capture)
                ready.set()
                await finish.wait()
                return result

        pending = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(endpoint.body.waiting.wait(), 2)
            assert reference(fact) in receipts[0].sources[0].fact_dependencies
            assert owner.conversation_busy(capture.conversation_id)
            assert not owner.conversation_busy(str(uuid4()))
            assert not scopes.busy(str(uuid4()), memory.ledger)
            await memory.acall(memory.ledger.delete_conversation, origin.conversation_id, REVIEWER)
            assert scopes.busy(origin.conversation_id, memory.ledger)
            assert owner.conversation_busy(origin.conversation_id)
            hold.set()
            await asyncio.wait_for(ready.wait(), 2)
        finally:
            hold.set()
            finish.set()
            result = await pending
        assert not result.success and result.response == ''
        await until(lambda: owner.resource_status()['source_scopes']['scopes'] == 0)
        assert scopes.resource_status()['scopes'] == 0


@pytest.mark.parametrize('ending', ['cancel', 'revoke'])
async def test_unresolved_native_read_keeps_worker_and_receiver_scopes_after_queue_cycle_exits(tmp_path, monkeypatch, ending):
    async with owner_graph(tmp_path, monkeypatch) as (owner, manager, memory, clock, endpoint, _, _, _):
        origin, fact = await stored(memory, clock)
        capture = memory_source(now=clock.now)
        receiver = Receiver()
        scopes = ProducerScopes(receiver.snapshot, capacity=1)
        entered, release = threading.Event(), threading.Event()
        references, errors = [], []
        original = manager.prompt_context

        def select(query):
            result = original(query)
            references.append(ref(result))
            entered.set()
            assert release.wait(3)
            return result

        monkeypatch.setattr(manager, 'prompt_context', select)

        async def consume():
            with scopes.operation((capture,)):
                return await dispatch(owner, capture)

        pending = asyncio.create_task(consume())
        try:
            await until(entered.is_set)
            assert owner.conversation_busy(str(uuid4()))
            assert scopes.busy(str(uuid4()), memory.ledger)
            if ending == 'cancel':
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError as error:
                    errors.append(error)
            else:
                await memory.acall(memory.ledger.delete_conversation, capture.conversation_id, REVIEWER)
                assert not (await pending).success
            await until(lambda: owner.queue.get_current() is None)
            assert owner.resource_status()['source_scopes']['unknown'] == 1
            assert scopes.resource_status()['unknown'] == 1
            assert references[0]() is not None and not endpoint.requests
            with pytest.raises(RuntimeError, match='producer_scope_capacity'):
                with scopes.operation((capture,)):
                    raise AssertionError('Native read released receiver capacity')
        finally:
            release.set()
            await until(lambda: memory.resource_status()['operations'] == 0)
            await asyncio.gather(pending, return_exceptions=True)
        assert references[0]() is None
        assert owner.resource_status()['source_scopes']['scopes'] == 0
        assert scopes.resource_status()['scopes'] == 0
        assert not owner.conversation_busy(origin.conversation_id)
        assert 'Consulta privada' not in json.dumps(owner.resource_status())
        assert capture.conversation_id not in json.dumps(owner.resource_status())


async def test_idle_permanent_queue_controller_retains_no_completed_request(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, _, _, clock, _, _, _, _):
        result = await dispatch(owner, memory_source(now=clock.now))
        assert result.success
        await until(lambda: owner.resource_status()['source_scopes']['scopes'] == 0)
        frame = owner._processor_task.get_coro().cr_frame
        assert 'request' not in frame.f_locals
        assert isinstance(frame.f_locals['cycle'], asyncio.Task)
        assert not owner.conversation_busy(str(uuid4()))


@pytest.mark.parametrize('transport', ['single', 'multi'])
@pytest.mark.parametrize('orchestrated', [False, True])
async def test_real_voice_result_cannot_postprocess_after_fact_revocation(tmp_path, monkeypatch, transport, orchestrated):
    async with capture_graph(tmp_path, monkeypatch, transport) as (g, room):
        async with AsyncExitStack() as stack:
            services = await build_memory_services({'memory': {'preferences_path': str(tmp_path/'preferences.json')}},
                shutdown=stack, chroma_client=MemoryClient(tmp_path/'chroma'), privacy=g.privacy, clock=g.clock)
            manager, memory = services.manager, services.manager.long_term
            handler, _, _, _, _ = voice_graph(monkeypatch)
            handler.zone_manager.add_zone(Zone('synthetic-a', 'Synthetic room', 2, 3))
            stack.push_async_callback(handler.aclose)
            owner = MultiUserOrchestrator(None, None, None, llm=handler.llm, memory_manager=manager,
                response_handler=handler, privacy=g.privacy)
            stack.push_async_callback(owner.stop)
            owner._running = True
            owner._processor_task = asyncio.create_task(owner._process_queue())
            router = request_router(manager, handler, owner=owner if orchestrated else None, privacy=g.privacy)
            stack.push_async_callback(router.aclose)
            origin, _ = await stored(memory, g.clock)
            install_poll(room, monkeypatch)
            entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
            posts = []

            async def command(captured):
                result = await router.process_command(captured)
                assert result['success'] and result['response'] == 'Una respuesta completa.'
                entered.set()
                while not release.is_set():
                    try:
                        await asyncio.shield(release.wait())
                    except asyncio.CancelledError:
                        cancelled.set()
                return result

            async def post(*args):
                posts.append(True)

            g.loop.on_command(command)
            g.loop.on_post_command(post)
            g.feed()
            try:
                await asyncio.wait_for(entered.wait(), 3)
                await g.privacy.acall(g.privacy.ledger.delete_conversation, origin.conversation_id, REVIEWER)
                assert g.loop.conversation_busy(origin.conversation_id)
                await asyncio.wait_for(cancelled.wait(), 2)
                assert not posts
            finally:
                release.set()
                tasks = g.loop._dispatches if transport == 'single' else g.loop._dispatch_tasks
                await until(lambda: not tasks)
            assert not posts and not owner._processor_task.done()


async def test_native_context_handoff_enriches_receiver_before_worker_resumes(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, manager, memory, clock, endpoint, _, _, _):
        origin, fact = await stored(memory, clock)
        capture = memory_source(now=clock.now)
        receiver = Receiver()
        scopes = ProducerScopes(receiver.snapshot)
        entered, release = asyncio.Event(), asyncio.Event()
        receipts, contexts = [], []
        original = manager.aprompt_context

        async def held(query):
            selected = None
            try:
                selected = await original(query)
                contexts.append(ref(selected))
                entered.set()
                await release.wait()
                return selected
            finally:
                selected = None

        monkeypatch.setattr(manager, 'aprompt_context', held)

        async def consume():
            with scopes.operation((capture,)) as scope:
                receipts.append(scope)
                return await dispatch(owner, capture)

        pending = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            request = owner.queue.get_current()
            assert all(not source.fact_dependencies for source in request.memory_sources)
            assert reference(fact) in receipts[0].sources[0].fact_dependencies
            assert scopes.resource_status()['unknown'] == 0
            assert not endpoint.requests
            await memory.acall(memory.ledger.delete_conversation, origin.conversation_id, REVIEWER)
            assert owner.conversation_busy(origin.conversation_id)
            assert scopes.busy(origin.conversation_id, memory.ledger)
        finally:
            release.set()
            result = await pending
        assert not result.success and not endpoint.requests
        await until(lambda: owner.resource_status()['source_scopes']['scopes'] == 0)
        assert contexts[0]() is None


async def test_scope_admission_failure_is_terminal_and_does_not_hang_the_waiter(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, _, _, clock, endpoint, _, _, _):
        capture = memory_source(now=clock.now)
        with ExitStack() as scopes:
            for _ in range(owner.resource_status()['source_scopes']['capacity']):
                scopes.enter_context(owner._scopes.operation((capture,)))
            result = await asyncio.wait_for(dispatch(owner, capture), 1)
            assert not result.success and result.intent == 'error'
            assert result.error == 'request_processing_failed'
            assert owner.queue.get_current() is None and not endpoint.requests
            assert not owner._processor_task.done()
        assert owner.resource_status()['source_scopes']['scopes'] == 0


async def test_queue_rejects_a_receiver_that_is_not_metadata(tmp_path, monkeypatch):
    async with owner_graph(tmp_path, monkeypatch) as (owner, _, _, clock, _, _, _, _):
        with pytest.raises(ValueError, match='invalid_request_producer_receiver'):
            owner.queue.enqueue('household', 'Texto privado.', memory_sources=(memory_source(now=clock.now),),
                                producer_receiver=object())
        assert owner.queue.get_current() is None and owner.queue.get_queue_status() == []


@pytest.mark.parametrize('failure', ['provider', 'incomplete'])
async def test_failed_generation_retires_selected_context_without_waiting_for_gc(tmp_path, monkeypatch, failure):
    provider_error = ValueError('Synthetic provider failure.')
    parts = [provider_error] if failure == 'provider' else [event('Respuesta parcial.'), event(finish='length')]
    async with owner_graph(tmp_path, monkeypatch, parts=parts) as (owner, manager, memory, clock, _, _, _, _):
        await stored(memory, clock)
        contexts = []
        original = manager.aprompt_context

        async def select(query):
            result = await original(query)
            contexts.append(ref(result))
            return result

        monkeypatch.setattr(manager, 'aprompt_context', select)
        result = await dispatch(owner, memory_source(now=clock.now))
        assert not result.success and result.intent == 'error'
        await until(lambda: owner.resource_status()['source_scopes']['scopes'] == 0)
        assert contexts and contexts[0]() is None
