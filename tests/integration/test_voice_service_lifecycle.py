"""Real voice owner cleanup, including stalled native/legacy producers."""

import asyncio
import threading
from contextlib import AsyncExitStack

import pytest

from src.core.voice_delivery import DeliveryState
from src.llm.generation_stream import GenerationStream
from src.pipeline.response_handler import ResponseCloseTimeout
from src.pipeline.voice_services import build_voice_services
from tests.mocks.voice_graph import voice_graph
from tests.mocks.llm_http import event

pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def assert_drained(owner):
    status = owner.get_delivery_status()
    assert status['closed']
    assert all(status[key] == 0 for key in ('controllers', 'workers', 'generations', 'blocked_generations', 'pending'))


async def test_close_stops_active_and_pending_output_and_rejects_new_admission(monkeypatch):
    owner, _, backend, _, _ = voice_graph(monkeypatch)
    backend.stall = True
    old = asyncio.create_task(owner.aspeak('active', zone_id='zone_origin'))
    await until(backend.created.is_set)
    pending = asyncio.create_task(owner.aspeak('pending', zone_id='zone_origin'))
    await until(lambda: owner.get_delivery_status()['pending'] == 1)
    await owner.aclose()
    assert (await old).delivery is DeliveryState.CANCELLED
    assert (await pending).delivery is DeliveryState.CANCELLED
    assert_drained(owner)
    assert backend.outputs[0].closed and backend.outputs[0].aborted.is_set()
    with pytest.raises(RuntimeError, match='voice_delivery_closed'):
        await owner.aspeak('new')
    await owner.aclose()


async def test_close_waits_for_native_stream_cleanup_after_output_worker_exits(monkeypatch):
    owner, _, _, endpoint, _ = voice_graph(monkeypatch, parts=[event('pending'), asyncio.Event()])
    entered, release = asyncio.Event(), asyncio.Event()
    build = endpoint.build

    def delayed_close(base_url):
        client = build(base_url)
        original = client.close
        async def close():
            entered.set()
            await release.wait()
            await original()
        monkeypatch.setattr(client, 'close', close)
        return client

    monkeypatch.setattr(owner.llm, '_new_async_client', delayed_close)
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
    closing = asyncio.create_task(owner.aclose())
    await asyncio.wait_for(entered.wait(), 1)
    await until(lambda: owner.get_delivery_status()['workers'] == 0)
    assert not closing.done() and owner.get_delivery_status()['generations'] >= 1
    release.set()
    await closing
    assert (await task).delivery is DeliveryState.CANCELLED
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert_drained(owner)


async def test_repeated_close_cancellation_finishes_cleanup_before_propagating(monkeypatch):
    owner, engine, backend, _, _ = voice_graph(monkeypatch)
    engine.release.clear()
    task = asyncio.create_task(owner.aspeak('pending', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    closing = asyncio.create_task(owner.aclose())
    await until(lambda: owner.get_delivery_status()['closed'])
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    engine.release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert (await task).delivery is DeliveryState.CANCELLED
    assert backend.outputs == []
    assert_drained(owner)


async def test_uncooperative_synthesis_reports_retained_resource_and_cannot_emit_late_pcm(monkeypatch):
    owner, engine, backend, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=0.03)
    engine.release.clear()
    task = asyncio.create_task(owner.aspeak('blocked', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    try:
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert owner.get_delivery_status()['workers'] == 1
        assert owner.get_delivery_status()['closed']
        with pytest.raises(RuntimeError, match='voice_delivery_closed'):
            await owner.aspeak('another')
    finally:
        engine.release.set()
    assert (await task).delivery is DeliveryState.CANCELLED
    await until(lambda: not owner.is_responding)
    await owner.aclose()
    assert backend.outputs == []
    assert_drained(owner)


async def test_close_accounts_for_quarantined_legacy_generation(monkeypatch):
    from types import SimpleNamespace
    owner, _, _, _, _ = voice_graph(monkeypatch, shutdown_timeout_s=0.03)
    entered, release = threading.Event(), threading.Event()

    def generate(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        yield {'token': 'late'}

    source = GenerationStream(SimpleNamespace(generate_stream=generate))
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin', generation=source))
    await until(entered.is_set)
    try:
        with pytest.raises(ResponseCloseTimeout):
            await owner.aclose()
        assert owner.get_delivery_status()['blocked_generations'] == 1
    finally:
        release.set()
    assert (await task).delivery is DeliveryState.CANCELLED
    await until(lambda: not source.sync_busy)
    await owner.aclose()
    assert_drained(owner)


async def test_application_factory_owns_cleanup_after_partial_feature_start_failure(monkeypatch):
    old, engine, backend, _, _ = voice_graph(monkeypatch)
    stack = AsyncExitStack()
    services = build_voice_services({'alerts': {'enabled': False}}, tts=engine,
        zone_manager=old.zone_manager, shutdown=stack)
    backend.stall = True
    speaking = asyncio.create_task(services.responses.aspeak('active', zone_id='zone_origin'))
    await until(backend.created.is_set)

    async def broken_load():
        raise RuntimeError('synthetic startup failure')

    monkeypatch.setattr(services.features.intercom, '_load_zones', broken_load)
    with pytest.raises(RuntimeError, match='synthetic startup failure'):
        await services.features.start()
    assert not services.features.timer_manager._running
    assert not services.features.intercom._running
    assert services.features.intercom._process_task.done()
    await stack.aclose()
    assert (await speaking).delivery is DeliveryState.CANCELLED
    assert_drained(services.responses)
    assert backend.outputs[0].closed


async def test_application_close_deadline_ten_times(monkeypatch, record_property):
    import json
    elapsed = []
    for _ in range(10):
        original, engine, backend, _, _ = voice_graph(monkeypatch)
        stack = AsyncExitStack()
        services = build_voice_services({'alerts': {'scheduler': {'enabled': False}}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager)
        await services.features.start()
        backend.stall = True
        speaking = asyncio.create_task(services.responses.aspeak('active', zone_id='zone_origin'))
        await until(backend.created.is_set)
        start = asyncio.get_running_loop().time()
        await stack.aclose()
        elapsed.append((asyncio.get_running_loop().time()-start)*1000)
        assert (await speaking).delivery is DeliveryState.CANCELLED
        assert backend.outputs[0].closed
        assert_drained(services.responses)
        assert services.features.timer_manager._update_task.done()
        assert services.features.intercom._process_task.done()
        assert services.features.notifications._process_task.done()
    ordered = sorted(elapsed)
    record_property('voice_application_close_ms', json.dumps({
        'repetitions': 10, 'p50': ordered[4], 'p95': ordered[9], 'max': max(elapsed),
        'cooperative_cleanup_limit_ms': 1000,
    }))
    assert max(elapsed) <= 1000


async def test_cache_build_cancellation_retains_native_worker_and_never_publishes_late_array(monkeypatch):
    from src.tts.response_cache import ResponseCache
    owner, engine, backend, _, _ = voice_graph(monkeypatch)
    engine.release.clear()
    cache = ResponseCache(engine, synthesize_callback=owner.asynthesize)
    building = asyncio.create_task(cache.build())
    await until(engine.entered.is_set)
    closing = asyncio.create_task(owner.aclose())
    await until(lambda: owner.get_delivery_status()['closed'])
    with pytest.raises(asyncio.CancelledError):
        await building
    assert owner.get_delivery_status()['workers'] == 1 and not closing.done()
    assert cache.size() == 0 and backend.outputs == []
    engine.release.set()
    await closing
    assert_drained(owner)
    assert cache.size() == 0 and backend.outputs == []


async def test_owned_startup_warmup_keeps_loop_available_and_synthesizes_without_output(monkeypatch):
    from src.main import _warmup_models
    owner, engine, backend, _, _ = voice_graph(monkeypatch)
    engine.release.clear()
    task = asyncio.create_task(_warmup_models(None, engine, None, None, None, voice_output=owner))
    await until(engine.entered.is_set)
    assert owner.get_delivery_status()['workers'] == 1 and not owner.is_speaking
    await asyncio.sleep(.2)
    assert not task.done()
    engine.release.set()
    await task
    assert engine.calls == ['hola'] and backend.outputs == []
    await owner.aclose()
    assert_drained(owner)
