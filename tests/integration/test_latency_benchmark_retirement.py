"""Baseline resource assertions; execute only through the root runner."""

import importlib.util
import asyncio
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import pytest_asyncio


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "tools/benchmark_latency.py").is_file())


@pytest_asyncio.fixture
async def latency(monkeypatch, tmp_path):
    from src.stt import whisper_fast
    from src.home_assistant import ha_client
    from src.tts import piper_tts
    from src.users import emotion_detector
    from src.llm import reasoner
    from tests.integration.test_ha_transport_ownership import _Transport, _Response, _Session, _rest_state
    import httpx

    state = SimpleNamespace(
        adapters=[], models=[], workers=[], audio=[], calls=0, benchmarks=[],
        pipers=[], piper_models=[], native_pcm=[], emotion=[], emotion_models=[],
        routers=[], ha_clients=[], http_clients=[], requests=[], router_local=False,
        gates={}, hooks={}, failures={}, tasks=[], external=[], owners=[], receipts=[],
        payloads=[], config=tmp_path / "synthetic.yaml", actor_entered={},
    )

    def hold(phase):
        result = threading.Event(), threading.Event()
        state.gates[phase] = result
        return result

    def hit(phase):
        if phase in state.gates:
            entered, release = state.gates[phase]
            entered.set()
            assert release.wait(5), f"synthetic {phase} not released"
        if phase in state.hooks:
            state.hooks[phase]()
        if phase in state.failures:
            raise state.failures[phase]()

    def observe_load(kind, resource):
        if state.owners:
            owner = next(owner for owner in state.owners if owner.resource is resource)
            from src.core import resource_retirement as resources
            assert owner in resources._OWNERS
            if kind == "stt":
                assert resource._worker_thread is threading.current_thread()
                assert any(lease.active for lease in owner._operations)
                # Pinned STT has the generic owner's actual-thread guard.
                with pytest.raises(RuntimeError, match="latency_stt_close_from_operation"):
                    owner.require_close_caller()
            else:
                assert owner._caller_guard is not None
        state.receipts.append(kind)

    class WhisperModel:
        def __init__(self, model, **options):
            assert model == "distil-whisper/distil-small.en"
            assert options == {"device": "cuda", "device_index": 0, "compute_type": "float16"}
            state.models.append(weakref.ref(self))
            self.worker = threading.current_thread()
            state.workers.append(self.worker)
            observe_load("stt", state.adapters[-1])
            hit("stt-load")

        def transcribe(self, audio, **options):
            assert threading.current_thread() is self.worker
            state.calls += 1
            state.audio.append(weakref.ref(audio))
            hit("stt-infer")
            segment = SimpleNamespace(text=" synthetic transcript ", no_speech_prob=.1,
                                      avg_logprob=-.2, compression_ratio=1.2)
            return iter([segment]), SimpleNamespace()

        def __del__(self):
            hit("stt-drop")

    sdk = ModuleType("faster_whisper")
    sdk.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", sdk)
    actual_stt = whisper_fast.FastWhisperSTT

    def construct(*args, **kwargs):
        adapter = actual_stt(*args, **kwargs)
        state.adapters.append(adapter)
        return adapter

    monkeypatch.setattr(whisper_fast, "FastWhisperSTT", construct)

    class NativePCM:
        def __init__(self):
            state.native_pcm.append(weakref.ref(self))

        def __array__(self, dtype=None, copy=None):
            import numpy as np
            hit("piper-convert")
            array = np.arange(8, dtype=np.float32)
            state.audio.append(weakref.ref(array))
            return array.astype(dtype) if dtype is not None else array

    class Voice:
        @classmethod
        def load(cls, path):
            assert path == "es_ES-davefx-medium.onnx"
            model = cls()
            state.piper_models.append(weakref.ref(model))
            observe_load("piper", state.pipers[-1])
            hit("piper-load")
            return model

        def synthesize(self, text):
            hit("piper-warmup" if text == "Hola" else "piper-infer")
            return NativePCM()

        def synthesize_stream_raw(self, text):
            import numpy as np
            yield np.array([0, 16384, -16384], dtype=np.int16).tobytes()

        def __del__(self):
            hit("piper-drop")

    piper = ModuleType("piper")
    piper.PiperVoice = Voice
    monkeypatch.setitem(sys.modules, "piper", piper)
    actual_piper = piper_tts.PiperTTS

    def create_piper(*args, **kwargs):
        adapter = actual_piper(*args, **kwargs)
        state.pipers.append(adapter)
        return adapter

    monkeypatch.setattr(piper_tts, "PiperTTS", create_piper)
    transformers = ModuleType("transformers")

    class Pipeline:
        def __call__(self, audio, **options):
            values = audio if isinstance(audio, list) else [audio]
            for array in values:
                state.audio.append(weakref.ref(array))
            hit("emotion-batch" if isinstance(audio, list) else "emotion-infer")
            prediction = [{"label": "neutral", "score": 1.0}]
            return [prediction for _ in audio] if isinstance(audio, list) else prediction

        def __del__(self):
            hit("emotion-drop")

    def pipeline(task, **options):
        assert task == "audio-classification"
        assert options == {"model": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "device": 0}
        model = Pipeline()
        state.emotion_models.append(weakref.ref(model))
        observe_load("emotion", state.emotion[-1])
        hit("emotion-load")
        return model

    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    actual_emotion = emotion_detector.EmotionDetector

    def create_emotion(*args, **kwargs):
        adapter = actual_emotion(*args, **kwargs)
        state.emotion.append(adapter)
        return adapter

    monkeypatch.setattr(emotion_detector, "EmotionDetector", create_emotion)

    actual_http_client = httpx.Client

    def http_handler(request):
        state.requests.append((request.method, request.url.path))
        hit("router-request")
        if request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"},
                {"id": "qwen2.5-7b-instruct", "object": "model"},
            ]})
        return httpx.Response(200, json={
            "id": "synthetic", "object": "text_completion", "created": 0,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "choices": [{"index": 0, "text": "hola", "message": {"role": "assistant", "content": "hola"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    class Transport(httpx.MockTransport):
        def close(self):
            hit("router-close")
            return super().close()

    def http_client(*args, **kwargs):
        observe_load("router", state.routers[-1])
        client = actual_http_client(*args, transport=Transport(http_handler), **kwargs)
        state.http_clients.append(client)
        hit("router-client")
        return client

    class HttpxProxy:
        Client = staticmethod(http_client)

        def __getattr__(self, name):
            return getattr(httpx, name)

    monkeypatch.setattr(reasoner, "httpx", HttpxProxy())
    actual_router = reasoner.FastRouter

    def create_router(*args, **kwargs):
        if state.router_local:
            assert "destination" not in kwargs and not kwargs.get("cloud_consent", False)
            kwargs = {**kwargs, "destination": "local"}
        adapter = actual_router(*args, **kwargs)
        state.routers.append(adapter)
        return adapter

    monkeypatch.setattr(reasoner, "FastRouter", create_router)
    transport = _Transport()
    state.transport, state.Response, state.rest_state = transport, _Response, _rest_state
    state.ha_close_entered, state.ha_close_release = asyncio.Event(), asyncio.Event()
    state.ha_close_release.set()
    transport.cleanup_gates.append(state.ha_close_release)
    original_session_close = _Session.close

    async def session_close(session):
        await original_session_close(session)
        state.ha_close_entered.set()
        await state.ha_close_release.wait()

    monkeypatch.setattr(_Session, "close", session_close)
    actual_ha = ha_client.HomeAssistantClient

    def create_ha(*args, **kwargs):
        client = actual_ha(*args, **kwargs)
        state.ha_clients.append(client)
        return client

    monkeypatch.setattr(ha_client, "HomeAssistantClient", create_ha)

    def session(*args, **kwargs):
        assert state.ha_clients
        assert any(getattr(benchmark, "_owned_ha", None) is state.ha_clients[-1]
                   for benchmark in state.benchmarks), "HA must be registered before first transport"
        state.receipts.append("ha")
        return transport.session_factory(*args, **kwargs)

    monkeypatch.setattr(ha_client.aiohttp, "ClientSession", session)
    spec = importlib.util.spec_from_file_location("synthetic_latency", _root() / "tools/benchmark_latency.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    if hasattr(module, "ResourceOwner"):
        actual_owner = module.ResourceOwner

        def resource_owner(*args, **kwargs):
            owner = actual_owner(*args, **kwargs)
            state.owners.append(owner)
            original_run = owner._closer.run
            entered = threading.Event()
            state.actor_entered[owner] = entered

            def run():
                entered.set()
                original_run()
                hit(f"{owner.error_prefix}-postlude")
                hit("sync-postlude")

            owner._closer.run = run
            return owner

        monkeypatch.setattr(module, "ResourceOwner", resource_owner)
    config = tmp_path / "synthetic.yaml"
    config.write_text("home_assistant:\n  url: http://synthetic.invalid\n  token: synthetic-token\n")
    state.module = module

    def benchmark():
        value = module.LatencyBenchmark(config_path=str(config))
        state.benchmarks.append(value)
        return value

    state.benchmark = benchmark
    state.hold, state.hit = hold, hit
    state.actual = {"stt": actual_stt, "piper": actual_piper, "emotion": actual_emotion,
                    "router": actual_router, "ha": actual_ha}

    def spawn(awaitable):
        task = asyncio.create_task(awaitable)
        state.tasks.append(task)
        return task

    state.spawn = spawn
    yield state
    failures = []
    for _, release in state.gates.values():
        release.set()
    transport.release_all()
    for task in state.tasks:
        if not task.done():
            task.cancel()
    if state.tasks:
        await asyncio.gather(*state.tasks, return_exceptions=True)
    for benchmark in state.benchmarks:
        try:
            close = getattr(benchmark, "aclose", None)
            if close is not None:
                await close()
        except BaseException as error:
            failures.append(error)
    for client in state.ha_clients:
        try:
            await client.close()
        except BaseException as error:
            failures.append(error)
    for adapter in [*state.adapters, *state.pipers, *state.emotion, *state.routers, *state.external]:
        try:
            close = getattr(adapter, "aclose", None)
            if close is not None:
                await close()
            else:
                adapter.close()
        except BaseException as error:
            failures.append(error)
    for worker in state.workers:
        try:
            worker._handle.join(3)
            assert worker._handle.is_done(), "actual pinned worker did not retire in fixture cleanup"
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("latency synthetic SDK cleanup failed", failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("report_failure", [False, True])
async def test_full_latency_return_or_later_report_failure_retires_actual_pinned_stt(latency, monkeypatch, report_failure):
    benchmark = latency.benchmark()
    reported = []
    if report_failure:
        def report(_summaries):
            reported.append(True)
            raise RuntimeError("synthetic later report failure")

        monkeypatch.setattr(benchmark, "_print_summary", report)
        with pytest.raises(RuntimeError):
            await benchmark.run_full_benchmark(iterations=2)
        assert reported == [True]
    else:
        summaries = await benchmark.run_full_benchmark(iterations=2)
        assert summaries["stt"].success_rate == 100
        assert summaries["stt"].iterations == 2
    assert len(latency.adapters) == len(latency.models) == len(latency.workers) == 1
    adapter, worker = latency.adapters[0], latency.workers[0]
    assert latency.calls >= 2
    assert worker._handle.is_done() and adapter._model is None and latency.models[0]() is None, {
        "actual_wrapper_retained": True,
        "actual_native_worker_exited": worker._handle.is_done(),
        "actual_backend_retained": latency.models[0]() is not None,
        "later_report_failure": report_failure,
    }


def _retired(latency, *, owned_pcm=True):
    assert all(adapter._model is None for adapter in latency.adapters)
    assert all(adapter._voice is None for adapter in latency.pipers)
    assert all(adapter._model is None and adapter._processor is None for adapter in latency.emotion)
    assert all(adapter._client is None for adapter in latency.routers)
    for references in (latency.models, latency.piper_models, latency.emotion_models, latency.native_pcm, latency.payloads):
        assert all(reference() is None for reference in references)
    if owned_pcm:
        assert all(reference() is None for reference in latency.audio)
    assert all(worker._handle.is_done() for worker in latency.workers)
    assert all(client.is_closed for client in latency.http_clients)
    assert all(session.closed for session in latency.transport.sessions)
    for client in latency.ha_clients:
        assert client._closed and client._close_task is not None and client._close_task.done()
    for owner in latency.owners:
        assert owner._confirmed and owner.resource is None and owner._caller_guard is None
        assert owner._target_exited.is_set() and owner._handle.is_done()


def _error_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _error_leaves(child)]
    return [error]


async def _caught(awaitable):
    try:
        await awaitable
    except BaseException as error:
        return error
    raise AssertionError("synthetic failure did not reach its owning boundary")


@pytest.mark.asyncio
async def test_full_latency_retires_visible_real_wrappers_and_preserves_default_router_denial(latency):
    benchmark = latency.benchmark()
    summaries = await benchmark.run_full_benchmark(iterations=2)
    assert set(summaries) == {"stt", "stt_vad", "tts", "emotion", "emotion_batch", "ha"}
    assert all(summary.success_rate == 100 for summary in summaries.values())
    assert summaries["emotion_batch"].iterations == 1
    assert len(latency.adapters) == len(latency.pipers) == len(latency.emotion) == len(latency.routers) == 1
    assert not latency.routers[0].policy.allowed
    assert latency.http_clients == [] and latency.requests == []
    assert {"stt", "piper", "emotion", "ha"} <= set(latency.receipts)
    assert len(latency.owners) >= 4
    _retired(latency)


@pytest.mark.asyncio
async def test_explicit_local_router_uses_real_openai_httpx_and_closes_owned_transport(latency):
    latency.router_local = True
    summaries = await latency.benchmark().run_full_benchmark(iterations=2)
    assert summaries["router"].success_rate == 100
    assert latency.routers[0].policy.allowed
    assert len(latency.http_clients) == 1
    assert sum(method == "GET" for method, _ in latency.requests) == 1
    assert sum(method == "POST" for method, _ in latency.requests) == 2
    _retired(latency)


@pytest.mark.asyncio
async def test_direct_latency_methods_reuse_owned_stt_until_explicit_close(latency):
    benchmark = latency.benchmark()
    assert await benchmark.benchmark_stt_with_vad(iterations=1) is None
    assert await benchmark.benchmark_emotion_batch(iterations=1) is None
    assert (await benchmark.benchmark_stt(iterations=1)).success_rate == 100
    assert (await benchmark.benchmark_stt_with_vad(iterations=1)).success_rate == 100
    assert len(latency.models) == len(latency.workers) == 1
    assert not latency.workers[0]._handle.is_done() and latency.models[0]() is not None
    await benchmark.aclose()
    _retired(latency)


@pytest.mark.asyncio
async def test_preinstalled_real_stt_and_caller_pcm_remain_borrowed(latency):
    import numpy as np

    peer = latency.actual["stt"](model="distil-whisper/distil-small.en", device="cuda:0", language="es")
    latency.adapters.append(peer)
    peer.load()
    benchmark = latency.benchmark()
    benchmark._stt = peer
    audio = np.ones(16000, dtype=np.float32)
    assert (await benchmark.benchmark_stt(audio, iterations=1)).success_rate == 100
    await benchmark.aclose()
    assert latency.models[0]() is peer._model and not latency.workers[0]._handle.is_done()
    np.testing.assert_array_equal(audio, np.ones(16000, dtype=np.float32))
    assert peer.transcribe(audio)[0].strip() == "synthetic transcript"
    peer.close()
    _retired(latency, owned_pcm=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["empty", "unavailable-empty", "unavailable-old"])
async def test_ha_actual_snapshot_distinguishes_available_empty_and_unavailable_previous_states(latency, monkeypatch, case):
    benchmark = latency.benchmark()
    observations = []
    actual_type = latency.actual["ha"]
    original_read = actual_type.get_state_snapshot

    async def observe(client, *args, **kwargs):
        snapshot = await original_read(client, *args, **kwargs)
        observations.append((snapshot.available, len(snapshot.states)))
        return snapshot

    monkeypatch.setattr(actual_type, "get_state_snapshot", observe)
    if case == "unavailable-old":
        latency.transport.responses.append(latency.Response([latency.rest_state()]))
        assert (await benchmark.benchmark_ha_call(iterations=1)).success_rate == 100
    latency.transport.responses.append(latency.Response([], status=200 if case == "empty" else 503))
    summary = await benchmark.benchmark_ha_call(iterations=1)
    assert summary.success_rate == (100 if case == "empty" else 0)
    assert observations[-1] == (case == "empty", 1 if case == "unavailable-old" else 0)
    assert latency.transport.service_posts == 0
    assert latency.transport.state_reads == (2 if case == "unavailable-old" else 1)
    await benchmark.aclose()
    _retired(latency)


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [0, -1])
async def test_zero_negative_iterations_keep_empty_outcomes_and_retire_lazy_models(latency, iterations):
    benchmark = latency.benchmark()
    stt = await benchmark.benchmark_stt(iterations=iterations)
    emotion = await benchmark.benchmark_emotion(iterations=iterations)
    batch = await benchmark.benchmark_emotion_batch(iterations=iterations)
    assert all(summary.iterations == 0 and summary.success_rate == 0 and summary.avg_ms == 0
               for summary in (stt, emotion, batch))
    assert len(latency.models) == len(latency.emotion_models) == 1
    await benchmark.aclose()
    _retired(latency)


@pytest.mark.asyncio
async def test_outer_measurement_clock_is_not_backend_elapsed_and_failed_samples_stay_zero(latency, monkeypatch):
    benchmark = latency.benchmark()
    original_time = latency.module.time
    ticks = iter([0., .010, 1., 1.020, 2., 2.040, 3.])

    class Clock:
        @staticmethod
        def perf_counter():
            return next(ticks)

        def __getattr__(self, name):
            return getattr(original_time, name)

    def fourth_fails():
        if latency.calls == 4:
            raise RuntimeError("synthetic fourth sample fails")

    latency.hooks["stt-infer"] = fourth_fails
    monkeypatch.setattr(latency.module, "time", Clock())
    summary = await benchmark.benchmark_stt(iterations=4)
    assert list(ticks) == []
    assert summary.iterations == 4 and summary.success_rate == 75
    assert summary.min_ms == pytest.approx(10) and summary.max_ms == pytest.approx(40)
    assert summary.avg_ms == pytest.approx(70 / 3)
    assert summary.median_ms == pytest.approx(20) and summary.p95_ms == pytest.approx(40)
    assert summary.std_ms == pytest.approx((700 / 3)**.5)
    await benchmark.aclose()
    _retired(latency)


class _RawError(RuntimeError):
    pass


class _RawInterrupt(KeyboardInterrupt):
    pass


class _RawExit(SystemExit):
    pass


def _fresh_payload_error(latency, kind):
    import numpy as np

    payload = np.full(32, .375, dtype=np.float32)
    latency.payloads.append(weakref.ref(payload))
    return {"ordinary": _RawError, "interrupt": _RawInterrupt, "exit": _RawExit}[kind](payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["stt-load", "stt-infer", "piper-load", "piper-infer", "piper-convert", "emotion-load", "emotion-infer", "emotion-batch"])
@pytest.mark.parametrize("kind", ["interrupt", "exit"])
async def test_full_latency_control_failure_is_fresh_and_does_not_retain_owned_payloads(latency, phase, kind):
    latency.failures[phase] = lambda: _fresh_payload_error(latency, kind)
    retained = await _caught(latency.benchmark().run_full_benchmark(iterations=2))
    leaves = _error_leaves(retained)
    assert leaves
    if kind == "interrupt":
        assert all(type(error) is KeyboardInterrupt and error.args == () for error in leaves)
    else:
        assert all(type(error) is SystemExit and error.code == 1 for error in leaves)
    _retired(latency)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase, component, success", [
    ("stt-infer", "stt", 0), ("piper-warmup", "tts", 100),
    ("piper-infer", "tts", 0), ("emotion-infer", "emotion", 100),
    ("emotion-batch", "emotion_batch", 100),
])
async def test_full_latency_ordinary_failures_preserve_component_fallback_outcomes_and_retire_payloads(latency, phase, component, success):
    latency.failures[phase] = lambda: _fresh_payload_error(latency, "ordinary")
    summaries = await latency.benchmark().run_full_benchmark(iterations=2)
    assert summaries[component].success_rate == success
    _retired(latency)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ordinary", "interrupt", "exit"])
async def test_later_report_failure_retires_every_component_and_releases_raw_payload(latency, monkeypatch, kind):
    benchmark = latency.benchmark()

    def report(_summaries):
        raise _fresh_payload_error(latency, kind)

    monkeypatch.setattr(benchmark, "_print_summary", report)
    retained = await _caught(benchmark.run_full_benchmark(iterations=2))
    leaves = _error_leaves(retained)
    expected = {"ordinary": RuntimeError, "interrupt": KeyboardInterrupt, "exit": SystemExit}[kind]
    assert all(type(error) is expected for error in leaves)
    if kind == "exit":
        assert all(error.code == 1 for error in leaves)
    _retired(latency)


@pytest.mark.asyncio
async def test_full_latency_starts_independent_closers_and_ha_before_held_stt_model_drop(latency, monkeypatch):
    benchmark = latency.benchmark()
    entered, release = latency.hold("stt-drop")
    rendered = []
    original_print = benchmark._print_summary

    def report(summaries):
        rendered.append(True)
        return original_print(summaries)

    monkeypatch.setattr(benchmark, "_print_summary", report)
    task = latency.spawn(benchmark.run_full_benchmark(iterations=2))
    assert await asyncio.to_thread(entered.wait, 5), "real pinned backend destructor did not enter"
    for actor_entered in latency.actor_entered.values():
        assert await asyncio.to_thread(actor_entered.wait, 5)
    await asyncio.wait_for(latency.ha_close_entered.wait(), 5)
    assert all(reference() is None for reference in [*latency.piper_models, *latency.emotion_models])
    assert latency.ha_clients[0]._close_task is not None
    assert not task.done() and rendered == []
    release.set()
    await task
    assert rendered == [True]
    _retired(latency)


@pytest.mark.asyncio
async def test_full_latency_cannot_render_after_target_receipts_before_native_thread_exit(latency, monkeypatch):
    benchmark = latency.benchmark()
    entered, release = latency.hold("sync-postlude")
    rendered = []
    monkeypatch.setattr(benchmark, "_print_summary", lambda summaries: rendered.append(True))
    task = latency.spawn(benchmark.run_full_benchmark(iterations=2))
    assert await asyncio.to_thread(entered.wait, 5)
    for owner, actor_entered in latency.actor_entered.items():
        assert await asyncio.to_thread(actor_entered.wait, 5)
        assert not owner._handle.is_done()
    assert any(owner._target_exited.is_set() for owner in latency.owners)
    assert not task.done() and rendered == []
    release.set()
    await task
    assert rendered == [True]
    _retired(latency)


@pytest.mark.asyncio
async def test_full_latency_cancellation_keeps_actual_ha_close_until_session_postlude_finishes(latency, monkeypatch):
    benchmark = latency.benchmark()
    latency.ha_close_release.clear()
    rendered = []
    monkeypatch.setattr(benchmark, "_print_summary", lambda summaries: rendered.append(True))
    task = latency.spawn(benchmark.run_full_benchmark(iterations=2))
    await asyncio.wait_for(latency.ha_close_entered.wait(), 5)
    actual_close = latency.ha_clients[0]._close_task
    assert actual_close is not None and not actual_close.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not actual_close.cancelled() and rendered == []
    latency.ha_close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert actual_close.done() and latency.transport.sessions[0].close_calls == 1
    assert rendered == []
    _retired(latency)


"""Appendable support/tests for the quiet 39-case candidate; not executed here."""


@pytest_asyncio.fixture(autouse=True)
async def _observe_owned_output_aliases(latency, monkeypatch):
    """Observe final copies as well as native PCM; retain only weakrefs."""
    actual_piper = latency.actual["piper"]
    original_synthesize = actual_piper.synthesize
    original_generate = latency.module.LatencyBenchmark._generate_test_audio
    original_np = latency.module.np

    def synthesize(adapter, *args, **kwargs):
        result = original_synthesize(adapter, *args, **kwargs)
        latency.audio.append(weakref.ref(result[0]))
        return result

    def generate(benchmark, *args, **kwargs):
        audio = original_generate(benchmark, *args, **kwargs)
        latency.audio.append(weakref.ref(audio))
        return audio

    class NumpyObserver:
        def __getattr__(self, name):
            return getattr(original_np, name)

        def zeros(self, *args, **kwargs):
            result = original_np.zeros(*args, **kwargs)
            latency.audio.append(weakref.ref(result))
            return result

        def concatenate(self, *args, **kwargs):
            result = original_np.concatenate(*args, **kwargs)
            latency.audio.append(weakref.ref(result))
            return result

    monkeypatch.setattr(actual_piper, "synthesize", synthesize)
    monkeypatch.setattr(latency.module.LatencyBenchmark, "_generate_test_audio", generate)
    monkeypatch.setattr(latency.module, "np", NumpyObserver())
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_cli_defaults_and_explicit_options_keep_real_component_order_and_batch_division(latency, monkeypatch, explicit):
    import io

    expected_n = 3 if explicit else 10
    expected_config = str(latency.config) if explicit else "config/settings.yaml"
    actual_type = latency.module.LatencyBenchmark
    original_init = actual_type.__init__
    constructed, order = [], []

    def initialize(benchmark, *args, **kwargs):
        original_init(benchmark, *args, **kwargs)
        constructed.append(benchmark.config_path)
        latency.benchmarks.append(benchmark)

    monkeypatch.setattr(actual_type, "__init__", initialize)
    names = ["benchmark_stt", "benchmark_stt_with_vad", "benchmark_tts", "benchmark_router",
             "benchmark_emotion", "benchmark_emotion_batch", "benchmark_ha_call"]
    for name in names:
        original = getattr(actual_type, name)

        async def observe(benchmark, *args, _name=name, _original=original, **kwargs):
            order.append((_name, kwargs["iterations"]))
            return await _original(benchmark, *args, **kwargs)

        monkeypatch.setattr(actual_type, name, observe)

    def config_leaf(path, *args, **kwargs):
        assert path == expected_config
        return io.StringIO("home_assistant:\n  url: http://synthetic.invalid\n  token: synthetic-token\n")

    monkeypatch.setattr(latency.module, "open", config_leaf, raising=False)
    argv = ["benchmark_latency.py"]
    if explicit:
        argv.extend(["-n", "3", "-c", expected_config])
    monkeypatch.setattr(sys, "argv", argv)
    await latency.module.main()
    assert constructed == [expected_config]
    assert order == [(name, expected_n // 2 if name == "benchmark_emotion_batch" else expected_n)
                     for name in names]
    _retired(latency)


@pytest.mark.asyncio
async def test_cli_does_not_add_docstring_only_audio_option(latency, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py", "--audio", "synthetic.wav"])
    with pytest.raises(SystemExit) as caught:
        await latency.module.main()
    assert caught.value.code == 2
    assert latency.adapters == latency.pipers == latency.emotion == latency.routers == []


"""Appendable focused public-wrapper cases; use the real latency fixture."""


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["piper", "emotion"])
async def test_retained_wrapper_is_terminal_after_full_retirement_without_mutating_borrowed_pcm(latency, kind):
    import numpy as np

    benchmark = latency.benchmark()
    await benchmark.run_full_benchmark(iterations=2)
    audio = np.ones(16000, dtype=np.float32)
    model_counts = len(latency.piper_models), len(latency.emotion_models)
    if kind == "piper":
        adapter = latency.pipers[0]
        with pytest.raises(RuntimeError, match="tts_closed"):
            adapter.load()
        with pytest.raises(RuntimeError, match="tts_closed"):
            adapter.synthesize("late synthetic text")
        stream = adapter.synthesize_stream("late synthetic stream")
        try:
            with pytest.raises(RuntimeError, match="tts_closed"):
                next(stream)
        finally:
            stream.close()
    else:
        adapter = latency.emotion[0]
        with pytest.raises(RuntimeError, match="emotion_closed"):
            adapter.load()
        with pytest.raises(RuntimeError, match="emotion_closed"):
            adapter.detect(audio)
        with pytest.raises(RuntimeError, match="emotion_closed"):
            adapter.batch_detect([audio, audio])
    np.testing.assert_array_equal(audio, np.ones(16000, dtype=np.float32))
    assert model_counts == (len(latency.piper_models), len(latency.emotion_models))
    _retired(latency)


@pytest.mark.asyncio
async def test_owned_piper_retirement_waits_for_real_suspended_stream_delegate_close(latency, monkeypatch):
    import numpy as np

    benchmark = latency.benchmark()
    assert (await benchmark.benchmark_tts(iterations=1)).success_rate == 100
    adapter = latency.pipers[0]
    voice = weakref.ref(adapter._voice)
    delegates = []

    class Delegate:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            assert not self.closed
            return np.array([0, 16384, -16384], dtype=np.int16).tobytes()

        def close(self):
            self.close_calls += 1
            self.closed = True

    def native_stream(_voice, text):
        delegate = Delegate()
        delegates.append(delegate)
        return delegate

    monkeypatch.setattr(type(adapter._voice), "synthesize_stream_raw", native_stream)
    stream = adapter.synthesize_stream("synthetic suspended stream")
    pull_errors = []

    def consume_in_a():
        try:
            np.testing.assert_array_equal(next(stream), [0.0, 0.5, -0.5])
        except BaseException as error:
            pull_errors.append((type(error).__name__, str(error)))

    consumer = threading.Thread(target=consume_in_a, daemon=True, name="piper-independent-consumer-A")
    closing = None
    try:
        consumer.start()
        consumer._handle.join(2)
        assert consumer._handle.is_done() and pull_errors == []
        lease = next(lease for lease in adapter._adapter_leases if lease.active)
        assert lease.thread is consumer and not lease.on_stack
        owner = next(owner for owner in latency.owners if owner.resource is adapter)
        # The pytest loop is Thread B, an independent close observer. A caller
        # that last pulled the stream itself is covered by the isolated RED.
        closing = latency.spawn(benchmark.aclose())
        assert await asyncio.to_thread(latency.actor_entered[owner].wait, 5)
        assert not closing.done() and not owner._handle.is_done()
        assert voice() is not None and delegates[0].close_calls == 0
    finally:
        stream.close()
        consumer._handle.join(2)
        assert consumer._handle.is_done()
    if closing is not None:
        await closing
    assert delegates[0].closed and delegates[0].close_calls == 1
    assert voice() is None
    _retired(latency)


"""Appendable owning-flow control-tree proof using actual closer actors."""


@pytest.mark.asyncio
async def test_full_latency_preserves_body_exit_and_post_start_closer_interrupt_after_real_drain(latency, monkeypatch):
    from src.core import resource_retirement as resources

    original_init = resources.ResourceOwner.__init__
    injected = []

    def initialize(owner, *args, **kwargs):
        original_init(owner, *args, **kwargs)
        if not injected:
            injected.append(owner)
            actual_start = owner._closer.start

            def start_then_interrupt():
                actual_start()
                assert owner._closer._handle.ident and owner._closer._started.is_set()
                raise KeyboardInterrupt

            owner._closer.start = start_then_interrupt

    monkeypatch.setattr(resources.ResourceOwner, "__init__", initialize)
    latency.failures["piper-infer"] = lambda: SystemExit(7)
    entered, release = latency.hold("sync-postlude")
    benchmark = latency.benchmark()
    rendered = []
    monkeypatch.setattr(benchmark, "_print_summary", lambda summaries: rendered.append(True))
    async def observe():
        return await _caught(benchmark.run_full_benchmark(iterations=2))

    task = latency.spawn(observe())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert injected and injected[0]._closer._handle.ident
        for owner, actor_entered in latency.actor_entered.items():
            assert await asyncio.to_thread(actor_entered.wait, 5)
            assert not owner._handle.is_done()
        assert not task.done() and rendered == []
    finally:
        release.set()
    retained = await task
    _retired(latency)
    assert benchmark._closing.done()
    # Its actual resources/handles are already confirmed. Keep all adapter and
    # native-handle teardown active, but do not replay this observed terminal
    # error before asserting the preserved control tree below.
    latency.benchmarks.remove(benchmark)
    leaves = _error_leaves(retained)
    assert any(type(error) is KeyboardInterrupt for error in leaves), leaves
    assert any(type(error) is SystemExit and error.code == 7 for error in leaves), leaves
    assert not any(type(error) is BaseException for error in leaves)
    assert len(injected) == 1 and rendered == []
