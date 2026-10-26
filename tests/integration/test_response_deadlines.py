"""Deadline authority across the real router, worker, generation and output."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from contextlib import AsyncExitStack
import json
import sys

import numpy as np
import pytest

from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.orchestrator.priority_queue import Priority
from src.core.voice_delivery import DeliveryState, ResponseBudgets, ResponseStage
from src.pipeline.voice_services import build_voice_services
from src.pipeline.command_event import CommandEvent
from src.pipeline.request_router import RequestRouter
from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio

TIMEOUT_TEXT = 'Lo siento, tarde demasiado. Intenta de nuevo.'
ERROR_TEXT = 'No pude procesar eso ahora mismo, probá de nuevo.'


def budgets(**changes):
    return ResponseBudgets(**({'queue_wait_s': .1, 'generation_s': .3,
        'delivery_s': 2, 'feedback_s': .3, 'cleanup_s': .05} | changes))


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


def routed(monkeypatch, parts, *, timeout=.3, orchestrated=True, **voice_options):
    handler, engine, backend, endpoint, _ = voice_graph(monkeypatch, parts=parts, **voice_options)
    orchestrator = MultiUserOrchestrator(None, None, None, llm=handler.llm,
        response_handler=handler, response_timeout_s=timeout) if orchestrated else None
    processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
        text='Nexa explicame el cielo', user=None, emotion=None, timings={}, stt_confidence=None)))
    audio_manager = MagicMock()
    audio_manager.detect_source_zone.return_value = 'zone_origin'
    router = RequestRouter(processor, handler, audio_manager, orchestrator=orchestrator,
        orchestrator_enabled=orchestrated, llm_reasoner=handler.llm,
        chroma_sync=SimpleNamespace(search_command=lambda *a, **kw: None),
        routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
    worker = None
    if orchestrator is not None:
        orchestrator._running = True
        worker = asyncio.create_task(orchestrator._process_queue())
    return router, orchestrator, worker, handler, engine, backend, endpoint


async def test_outer_deadline_cannot_repeat_an_error_whose_device_tail_is_draining(monkeypatch):
    router, orchestrator, worker, handler, engine, backend, _ = routed(monkeypatch,
        [event('incomplete'), event(finish='length')])
    backend.release_tail.clear()
    request = asyncio.create_task(router.process_command(CommandEvent(
        np.zeros(16000, dtype=np.float32), room_id='origin')))
    try:
        await until(lambda: backend.outputs and backend.outputs[0].draining.is_set())
        await asyncio.sleep(.35)
        backend.release_tail.set()
        result = await asyncio.wait_for(request, 2)
        assert not result['success']
        assert engine.calls == ['No pude procesar eso ahora mismo, probá de nuevo.']
        assert orchestrator.context_manager.get('unknown').conversation_history == []
    finally:
        backend.release_tail.set()
        orchestrator._running = False
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await handler.aclose()


async def test_provider_stall_expires_once_and_closes_native_resources(monkeypatch):
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch,
        parts=[event('pending'), asyncio.Event()], response_budgets=budgets())
    try:
        result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 2)
        assert not result.success and result.timeout_stage is ResponseStage.GENERATION
        assert result.text == TIMEOUT_TEXT and engine.calls == [TIMEOUT_TEXT]
        assert result.delivery is DeliveryState.COMPLETE
        assert endpoint.body.closed and endpoint.client.is_closed()
        np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio(TIMEOUT_TEXT))
    finally:
        await owner.aclose()


async def test_total_deadline_during_partial_delivery_discards_old_tail_and_speaks_one_error(monkeypatch):
    router, orchestrator, worker, owner, engine, backend, endpoint = routed(monkeypatch,
        [event('Primera oración suficientemente larga. Un resto pendiente'), asyncio.Event()],
        timeout=.3, response_budgets=budgets(generation_s=5))
    backend.release_tail.clear()
    task = asyncio.create_task(router.process_command(CommandEvent(
        np.zeros(16000, dtype=np.float32), room_id='origin')))
    try:
        await until(lambda: backend.outputs and backend.outputs[0].draining.is_set())
        await until(lambda: len(backend.outputs) == 2)
        assert backend.outputs[0].aborted.is_set()
        backend.release_tail.set()
        result = await asyncio.wait_for(task, 2)
        assert not result['success'] and result['intent'] == 'timeout'
        assert result['response'] == TIMEOUT_TEXT
        assert result['timeout_stage'] == 'generation' and result['voice_delivery'] == 'complete'
        assert engine.calls == ['Primera oración suficientemente larga.', TIMEOUT_TEXT]
        assert endpoint.body.closed and endpoint.client.is_closed()
        assert orchestrator.context_manager.get('unknown').conversation_history == []
        assert all(output.closed for output in backend.outputs)
    finally:
        backend.release_tail.set()
        orchestrator._running = False
        worker.cancel()
        await asyncio.gather(worker, task, return_exceptions=True)
        await owner.aclose()


async def test_feedback_has_its_own_deadline_and_is_never_retried(monkeypatch):
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch,
        parts=[event('incomplete'), event(finish='length')], response_budgets=budgets(feedback_s=.1))
    backend.release_tail.clear()
    try:
        result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 2)
        assert not result.success and result.delivery is DeliveryState.TIMED_OUT
        assert result.timeout_stage is ResponseStage.FEEDBACK
        assert engine.calls == [ERROR_TEXT] and len(backend.outputs) == 1
        assert backend.outputs[0].aborted.is_set() and endpoint.client.is_closed()
    finally:
        backend.release_tail.set()
        await owner.aclose()


async def test_delivery_budget_stops_completed_generation_before_one_error(monkeypatch):
    owner, engine, backend, endpoint, amplifier = voice_graph(monkeypatch,
        response_budgets=budgets(delivery_s=.15, feedback_s=.5))
    backend.release_tail.clear()
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    try:
        await until(lambda: len(backend.outputs) == 2)
        backend.release_tail.set()
        result = await asyncio.wait_for(task, 1)
        assert not result.success and result.timeout_stage is ResponseStage.DELIVERY
        assert result.delivery is DeliveryState.COMPLETE and result.text == TIMEOUT_TEXT
        assert engine.calls == ['Una respuesta completa.', TIMEOUT_TEXT]
        assert backend.outputs[0].aborted.is_set() and all(o.closed for o in backend.outputs)
        assert endpoint.client.is_closed()
        assert [call.args for call in amplifier.select_zone.call_args_list] == [(1,), (1,)]
    finally:
        backend.release_tail.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


async def test_stalled_synthesis_retains_ownership_after_work_and_feedback_timeouts(monkeypatch):
    owner, engine, backend, _, _ = voice_graph(monkeypatch,
        response_budgets=budgets(delivery_s=.1))
    engine.release.clear()
    try:
        result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 1)
        assert result.timeout_stage is ResponseStage.FEEDBACK and not result.success
        assert owner.get_delivery_status()['workers'] == 1
        assert engine.calls == ['Una respuesta completa.'] and backend.outputs == []
        engine.release.set()
        await until(lambda: not owner.is_responding)
        assert backend.outputs == []
        assert (await owner.aspeak('new', zone_id='zone_origin')).success
        assert engine.peak_active == 1 and engine.calls == ['Una respuesta completa.', 'new']
        np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio('new'))
    finally:
        engine.release.set()
        await owner.aclose()


async def test_queue_and_feedback_wait_are_bounded_without_starting_a_second_engine_or_provider(monkeypatch):
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch, response_budgets=budgets())
    engine.release.clear()
    old = asyncio.create_task(owner.aspeak('occupied', zone_id='zone_origin'))
    try:
        await until(engine.entered.is_set)
        result = await asyncio.wait_for(owner.aspeak('queued', zone_id='zone_origin'), .5)
        assert result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is ResponseStage.QUEUE
        result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 1)
        assert result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is ResponseStage.FEEDBACK
        assert endpoint.requests == [] and engine.calls == ['occupied'] and backend.outputs == []
        assert owner.get_delivery_status()['workers'] == 1
    finally:
        engine.release.set()
        await old
        await owner.aclose()


async def test_downstream_backpressure_does_not_spend_provider_budget(monkeypatch):
    phrases = [f'Oración número {i} suficientemente larga.' for i in range(12)]
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch,
        parts=[*(event(text + ' ') for text in phrases), event(finish='stop')],
        response_budgets=budgets(generation_s=.3))
    engine.release.clear()
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    try:
        await until(lambda: owner._owner is not None and len(owner._owner.queue) == 2)
        await asyncio.sleep(.4)
        assert not task.done() and engine.calls == [phrases[0]]
        engine.release.set()
        result = await asyncio.wait_for(task, 2)
        assert result.success and engine.calls == phrases
        assert endpoint.body.closed and endpoint.client.is_closed()
        np.testing.assert_array_equal(backend.heard()[:, 0], np.concatenate([engine.audio(t) for t in phrases]))
    finally:
        engine.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


async def test_cancel_during_transition_to_error_does_not_start_late_feedback(monkeypatch):
    owner, engine, backend, _, _ = voice_graph(monkeypatch,
        parts=[event('incomplete'), event(finish='length')], response_budgets=budgets())
    entered, release = asyncio.Event(), asyncio.Event()
    stop = owner._stop_session

    async def paused_stop(session):
        await stop(session)
        entered.set()
        await release.wait()

    monkeypatch.setattr(owner, '_stop_session', paused_stop)
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        cancelling = asyncio.create_task(owner.cancel('zone_origin'))
        await asyncio.sleep(.01)
        release.set()
        assert await cancelling
        assert (await asyncio.wait_for(task, 1)).delivery is DeliveryState.CANCELLED
        assert engine.calls == [] and backend.outputs == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize('cancel', ['barge_in', 'caller', 'shutdown'])
async def test_cancel_while_error_is_playing_stays_silent(monkeypatch, cancel):
    owner, engine, backend, _, _ = voice_graph(monkeypatch,
        parts=[event('incomplete'), event(finish='length')], response_budgets=budgets())
    backend.release_tail.clear()
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    try:
        await until(lambda: backend.outputs and backend.outputs[0].draining.is_set())
        if cancel == 'caller':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            if cancel == 'shutdown':
                await owner.aclose()
            else:
                assert await owner.cancel('zone_origin')
            assert (await task).delivery is DeliveryState.CANCELLED
        assert engine.calls == [ERROR_TEXT] and len(backend.outputs) == 1
        assert backend.outputs[0].aborted.is_set()
    finally:
        backend.release_tail.set()
        await owner.aclose()


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_new_request_cancels_old_error_without_repeating_it_in_both_routers(monkeypatch, orchestrated):
    router, orchestrator, worker, owner, engine, backend, endpoint = routed(monkeypatch,
        [event('incomplete'), event(finish='length')], timeout=2, orchestrated=orchestrated,
        response_budgets=budgets(feedback_s=1))
    backend.release_tail.clear()
    old = asyncio.create_task(router.process_command(CommandEvent(
        np.zeros(16000, dtype=np.float32), room_id='origin')))
    new = None
    try:
        await until(lambda: backend.outputs and backend.outputs[-1].draining.is_set())
        assert engine.calls == [ERROR_TEXT]
        endpoint.body.parts = [event('Una respuesta nueva.'), event(finish='stop')]
        new = asyncio.create_task(router.process_command(CommandEvent(
            np.zeros(16000, dtype=np.float32), room_id='origin')))
        await until(lambda: len(backend.outputs) == 2)
        backend.release_tail.set()
        assert not (await old)['success'] and (await new)['success']
        assert engine.calls == [ERROR_TEXT, 'Una respuesta nueva.']
        assert backend.outputs[0].aborted.is_set()
        np.testing.assert_array_equal(np.concatenate(backend.outputs[1].frames)[:, 0], engine.audio('Una respuesta nueva.'))
        if orchestrator:
            history = orchestrator.context_manager.get('unknown').conversation_history
            assert len(history) == 2 and history[-1].content == 'Una respuesta nueva.'
    finally:
        backend.release_tail.set()
        if worker:
            orchestrator._running = False
            worker.cancel()
        await asyncio.gather(*(t for t in (old, new, worker) if t is not None), return_exceptions=True)
        await owner.aclose()


@pytest.mark.parametrize('resource', ['client', 'body'])
async def test_native_cleanup_stall_keeps_resources_owned_and_blocks_new_connections(monkeypatch, resource):
    owner, engine, _, endpoint, _ = voice_graph(monkeypatch,
        parts=[event('Una respuesta completa.'), event(finish='stop')], response_budgets=budgets())
    release = asyncio.Event()
    build = endpoint.build

    def delayed_close(base_url):
        client = build(base_url)
        close = client.close

        async def blocked_close():
            await release.wait()
            await close()

        monkeypatch.setattr(client, 'close', blocked_close)
        return client

    if resource == 'client':
        monkeypatch.setattr(owner.llm, '_new_async_client', delayed_close)
    else:
        close_body = endpoint.body.aclose

        async def blocked_body_close():
            await release.wait()
            await close_body()

        monkeypatch.setattr(endpoint.body, 'aclose', blocked_body_close)
    try:
        result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 2)
        assert not result.success and result.timeout_stage is ResponseStage.GENERATION
        assert not endpoint.client.is_closed() and owner.get_delivery_status()['generations'] >= 1
        count = len(endpoint.requests)
        for _ in range(3):
            result = await owner.arespond('another', zone_id='zone_origin')
            assert not result.success and result.error == 'previous_generation_still_running'
            assert len(endpoint.requests) == count
        assert engine.peak_active == 1
    finally:
        release.set()
        await owner.aclose()
        assert endpoint.body.closed and endpoint.client.is_closed()


async def test_long_complete_uncached_response_ten_times(record_property, tmp_path):
    # Isolate graph measurements from the suite's torch/model imports and
    # capture buffers. The child's GC remains enabled. Full runtime latency
    # still needs H4/H6 acceptance, including stop-the-world GC pauses.
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'tests.mocks.voice_metrics', str(tmp_path / 'metrics.json'),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    assert process.returncode == 0, stderr.decode()
    metrics = json.loads(await asyncio.to_thread((tmp_path / 'metrics.json').read_text))
    record_property('long_complete_voice_metrics', json.dumps(metrics))
    assert metrics['loop_lag_p95_ms'] <= 25 and metrics['loop_lag_max_ms'] <= 75


async def test_application_factory_consumes_stage_budgets(monkeypatch):
    owner, engine, _, _, _ = voice_graph(monkeypatch)
    selected = budgets()
    async with AsyncExitStack() as stack:
        services = build_voice_services({'tts': {'streaming': {'response_timeouts': {
            name: getattr(selected, name) for name in selected.__dataclass_fields__
        }}}, 'alerts': {'enabled': False}}, tts=engine, shutdown=stack, zone_manager=owner.zone_manager)
        assert services.responses.budgets == selected


@pytest.mark.parametrize('enabled', [True, False])
async def test_application_factory_respects_filler_selection(monkeypatch, enabled):
    release = asyncio.Event()
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch,
        parts=[release, event('Una respuesta completa.'), event(finish='stop')])
    async with AsyncExitStack() as stack:
        services = build_voice_services({
            'llm': {'buffering': {'preset': 'smooth', 'use_filler': enabled, 'filler_phrases': ['Dejame ver...']}},
            'alerts': {'enabled': False},
        }, tts=engine, llm=owner.llm, shutdown=stack, zone_manager=owner.zone_manager)
        speaking = asyncio.create_task(services.responses.arespond('synthetic', zone_id='zone_origin'))
        try:
            await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
            if enabled:
                await until(lambda: engine.calls == ['Dejame ver...'])
            else:
                await asyncio.sleep(.15)
                assert engine.calls == []
            release.set()
            result = await speaking
        finally:
            release.set()
            if not speaking.done():
                speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
        expected = (['Dejame ver...'] if enabled else []) + ['Una respuesta completa.']
        assert result.success and engine.calls == expected
        assert services.responses.llm_buffer_preset == 'smooth'
        np.testing.assert_array_equal(backend.heard()[:, 0], np.concatenate([engine.audio(t) for t in expected]))
        assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('cancel', [False, True])
async def test_queue_announcement_shares_request_deadline_and_cancellation(monkeypatch, cancel):
    owner, _, backend, endpoint, _ = voice_graph(monkeypatch)
    orchestrator = MultiUserOrchestrator(None, None, None, llm=owner.llm, response_handler=owner)
    orchestrator.queue.enqueue(user_id='other', text='first', priority=Priority.LOW)
    backend.release_tail.clear()

    async def announce(result):
        await owner.aspeak(result.response, zone_id='zone_origin')

    task = asyncio.create_task(orchestrator.dispatcher._slow_path(
        text='second', user_id='requester', user_name=None, zone_id='zone_origin',
        priority=Priority.LOW, on_response=announce, timeout=.15))
    try:
        await until(lambda: backend.outputs and backend.outputs[-1].draining.is_set())
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await asyncio.wait_for(task, .5)
            assert not result.success and result.intent == 'timeout'
        assert all(r['user_id'] != 'requester' for r in orchestrator.queue.get_queue_status())
        assert endpoint.requests == [] and backend.outputs[0].aborted.is_set()
    finally:
        backend.release_tail.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()
