"""Owned generation-to-output graph, using real zones/player and synthetic I/O."""

import asyncio

import numpy as np
import pytest

from src.pipeline.response_handler import DeliveryState, GenerationState
from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph as graph
from unittest.mock import MagicMock
from src.tts.response_cache import CachedAudio

pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)




@pytest.mark.parametrize('stream', [True, False])
async def test_complete_pcm_and_fixed_origin_through_full_and_streamed_synthesis(monkeypatch, stream):
    owner, engine, backend, _, amp = graph(monkeypatch)
    engine.release.clear()
    task = asyncio.create_task(owner.aspeak('Respuesta', zone_id='zone_origin', stream=stream))
    await until(engine.entered.is_set)
    assert owner.is_responding and not owner.is_speaking
    owner.set_active_zone('zone_other')
    engine.release.set()
    result = await task
    assert result.success and result.spoken_segments == 1
    np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio('Respuesta'))
    amp.select_zone.assert_called_once_with(1)
    assert backend.outputs[0].device == 7
    assert not owner.is_responding and not owner.is_speaking


async def test_cached_audio_uses_owned_player_and_never_synthesizes(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    audio = np.linspace(-1, 1, 24000, dtype=np.float32)
    owner._response_cache = MagicMock()
    owner._response_cache.get.return_value = CachedAudio(audio, 24000, 1, 'Listo')
    result = await owner.aspeak('Listo', zone_id='zone_origin')
    assert result.success and engine.calls == []
    np.testing.assert_array_equal(backend.heard()[:, 0], audio)
    assert backend.outputs[0].closed


async def test_earcon_is_cancelable_through_the_same_owner(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    backend.stall = True
    owner.set_earcon(np.ones(24000, dtype=np.float32), 24000)
    task = asyncio.create_task(owner.aplay_earcon(zone_id='zone_origin'))
    await until(backend.created.is_set)
    assert owner.is_speaking
    assert await owner.cancel('zone_origin')
    result = await task
    assert result.delivery is DeliveryState.CANCELLED and engine.calls == []
    await until(lambda: not owner.is_responding)
    assert backend.outputs[0].aborted.is_set()
    assert not await owner.cancel()


async def test_late_synthesis_keeps_slot_and_cannot_start_old_audio(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    engine.release.clear()
    old = asyncio.create_task(owner.aspeak('old', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    assert await owner.cancel()
    assert (await old).delivery is DeliveryState.CANCELLED
    new = asyncio.create_task(owner.aspeak('new', zone_id='zone_origin'))
    await asyncio.sleep(0.02)
    assert engine.calls == ['old'] and backend.outputs == []
    engine.release.set()
    assert (await new).success
    assert engine.peak_active == 1 and engine.calls == ['old', 'new']
    np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio('new'))


async def test_generation_and_playback_stream_once_with_bounded_sentences(monkeypatch):
    text = 'Primera oración larga. Segunda oración larga. Tercera oración larga.'
    owner, engine, backend, endpoint, _ = graph(monkeypatch, parts=[event(text), event(finish='stop')])
    result = await owner.arespond('synthetic', zone_id='zone_origin')
    assert result.success and result.text == text
    assert engine.calls == ['Primera oración larga.', 'Segunda oración larga.', 'Tercera oración larga.']
    expected = np.concatenate([engine.audio(text) for text in engine.calls])
    np.testing.assert_array_equal(backend.heard()[:, 0], expected)
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_generation_failure_after_partial_voice_delivers_one_error(monkeypatch):
    release = asyncio.Event()
    prefix = 'Una primera oración completa. '
    owner, engine, backend, endpoint, _ = graph(monkeypatch, parts=[event(prefix + 'un tail pendiente'), release])
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    await until(lambda: backend.outputs and backend.outputs[0].delivered.is_set())
    release.set()
    result = await task
    assert result.generation is GenerationState.FAILED and result.delivery is DeliveryState.COMPLETE
    assert not result.success and result.speech_handled
    assert engine.calls == [prefix.strip(), 'No pude procesar eso ahora mismo, probá de nuevo.']
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_cancel_pending_generation_stops_stream_without_error_voice(monkeypatch):
    owner, engine, backend, endpoint, _ = graph(monkeypatch, parts=[event('pending'), asyncio.Event()])
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
    assert owner.is_responding and not owner.is_speaking
    assert await owner.cancel('zone_origin')
    result = await asyncio.wait_for(task, 1)
    assert result.delivery is DeliveryState.CANCELLED
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert engine.calls == [] and backend.outputs == []


async def test_audio_failure_cancels_pending_network_without_retry_or_context_success(monkeypatch):
    owner, engine, backend, endpoint, _ = graph(monkeypatch,
        parts=[event('Primera oración suficientemente larga. mas texto esperando'), asyncio.Event()])
    backend.fail_start = True
    result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 2)
    assert result.delivery is DeliveryState.FAILED and not result.success
    assert len(backend.outputs) == 1 and endpoint.body.closed and endpoint.client.is_closed()
    assert len(engine.calls) == 1


async def test_async_caller_cancel_during_blocked_tts_does_not_release_engine_early(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    engine.release.clear()
    task = asyncio.create_task(owner.aspeak('old', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owner.is_responding
    engine.release.set()
    await until(lambda: not owner.is_responding)
    assert backend.outputs == []


async def test_bounded_queue_applies_backpressure_and_cancellation_wakes_waiting_producer(monkeypatch):
    owner, engine, backend, endpoint, _ = graph(monkeypatch, parts=[
        *(event(f'Oración número {i} suficientemente larga. ') for i in range(100)), event(finish='stop'),
    ])
    engine.release.clear()
    task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    session = owner._owner
    await until(lambda: len(session.queue) == 2)
    assert session.peak_queue == owner.sentence_queue_capacity == 2
    assert engine.calls == ['Oración número 0 suficientemente larga.']
    assert await owner.cancel()
    assert (await asyncio.wait_for(task, 1)).delivery is DeliveryState.CANCELLED
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert backend.outputs == []
    engine.release.set()
    await until(lambda: not owner.is_responding)


async def test_unconfigured_zone_closes_source_even_before_first_iteration(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)

    class Source:
        closed = 0
        def __iter__(self):
            return self
        def __next__(self):
            return engine.audio('old')
        def close(self):
            self.closed += 1

    source = Source()
    monkeypatch.setattr(engine, 'synthesize_stream', lambda _: source)
    result = await owner.aspeak('old', zone_id='not_configured')
    assert result.delivery is DeliveryState.FAILED and source.closed == 1
    assert backend.outputs == []


async def test_other_zone_cancellation_does_not_interrupt_origin(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    engine.release.clear()
    task = asyncio.create_task(owner.aspeak('origin', zone_id='zone_origin'))
    await until(engine.entered.is_set)
    assert not await owner.cancel('zone_other')
    engine.release.set()
    assert (await task).success
    np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio('origin'))


async def test_broadcast_preserves_owned_output_and_abort_contract(monkeypatch):
    owner, engine, backend, _, amp = graph(monkeypatch)
    assert (await owner.aspeak_to_all_zones('Anuncio')).success
    amp.select_all_zones.assert_called_once_with()
    np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio('Anuncio'))
    backend.stall = True
    task = asyncio.create_task(owner.aspeak_to_all_zones('Segundo anuncio'))
    await until(lambda: len(backend.outputs) == 2)
    assert await owner.cancel('zone_other')
    assert (await task).delivery is DeliveryState.CANCELLED


async def test_pending_responses_are_bounded_without_extra_synthesis_workers(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch, max_pending_responses=1)
    engine.release.clear()
    old = asyncio.create_task(owner.aspeak('old'))
    await until(engine.entered.is_set)
    waiting = asyncio.create_task(owner.aspeak('waiting'))
    await until(lambda: len(owner._pending) == 1)
    with pytest.raises(RuntimeError, match='response_queue_full'):
        await owner.aspeak('excess')
    assert engine.calls == ['old'] and engine.peak_active == 1
    await owner.cancel()
    assert (await waiting).delivery is DeliveryState.CANCELLED
    assert (await old).delivery is DeliveryState.CANCELLED
    engine.release.set()
    await until(lambda: not owner.is_responding)
    assert backend.outputs == []


async def test_hook_cannot_redirect_a_captured_origin(monkeypatch):
    from src.hooks import HookRegistry, RewriteResult, TtsCall
    owner, engine, backend, _, amp = graph(monkeypatch)
    owner.hooks = HookRegistry()
    owner.hooks.register_before('before_tts_speak', lambda call: RewriteResult(
        TtsCall(text=call.text, voice=None, lang='es', user_id=None, zone_id='zone_other'), rule_name='redirect'), priority=10)
    result = await owner.aspeak('Texto', zone_id='zone_origin')
    assert result.delivery is DeliveryState.BLOCKED
    assert engine.calls == [] and backend.outputs == []
    amp.select_zone.assert_not_called()


async def test_voice_graph_loop_and_cancellation_limits(monkeypatch, record_property):
    import json
    lags, cancels = [], []
    loop = asyncio.get_running_loop()
    for _ in range(10):
        owner, engine, backend, endpoint, _ = graph(monkeypatch,
            parts=[0.2, event('Respuesta completa de prueba.'), event(finish='stop')])
        task = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
        deadline = loop.time() + 0.005
        while not task.done():
            await asyncio.sleep(max(0, deadline - loop.time()))
            lags.append(max(0, loop.time() - deadline) * 1000)
            deadline = loop.time() + 0.005
        assert (await task).success and endpoint.body.closed
        expected = engine.audio('Respuesta completa de prueba.')
        np.testing.assert_array_equal(backend.heard()[:, 0], expected)
        backend.stall = True
        task = asyncio.create_task(owner.aspeak('Cancelar', zone_id='zone_origin'))
        await until(lambda: len(backend.outputs) == 2)
        started = loop.time()
        await owner.cancel('zone_origin')
        assert (await task).delivery is DeliveryState.CANCELLED
        await until(lambda: not owner.is_responding)
        cancels.append((loop.time() - started) * 1000)
        assert backend.outputs[-1].closed and backend.outputs[-1].aborted.is_set()
    ordered = sorted(lags)
    p95 = ordered[int(.95 * (len(ordered) - 1))]
    record_property('owned_voice_metrics', json.dumps({
        'repetitions': 10, 'stall_ms': 200, 'heartbeat_interval_ms': 5, 'heartbeat_samples': len(lags),
        'loop_lag_p95_ms': p95, 'loop_lag_max_ms': max(lags),
        'cancel_cleanup_max_ms': max(cancels), 'exact_pcm_runs': 10,
    }))
    assert p95 <= 25 and max(lags) <= 75
    assert max(cancels) <= 100


async def test_decimal_split_across_tokens_is_not_spoken_as_separate_sentences(monkeypatch):
    from types import SimpleNamespace
    from src.llm.generation_stream import GenerationStream
    owner, engine, backend, _, _ = graph(monkeypatch)
    source = GenerationStream(SimpleNamespace(generate_stream=lambda *a, **kw: iter([
        {'token': 'La medida es 22.'}, {'token': '5 grados. '}, {'token': 'Es correcto.'},
    ])))
    result = await owner.arespond('synthetic', zone_id='zone_origin', generation=source)
    assert result.success
    assert engine.calls == ['La medida es 22.5 grados.', 'Es correcto.']
    expected = np.concatenate([engine.audio(t) for t in engine.calls])
    np.testing.assert_array_equal(backend.heard()[:, 0], expected)


async def test_explicit_zone_without_zone_manager_cannot_fall_back_to_default_device(monkeypatch):
    owner, engine, backend, _, _ = graph(monkeypatch)
    owner.zone_manager = None
    result = await owner.aspeak('Texto', zone_id='zone_origin')
    assert result.delivery is DeliveryState.FAILED and result.error == 'output_zone_unavailable'
    assert engine.calls == [] and backend.outputs == []
