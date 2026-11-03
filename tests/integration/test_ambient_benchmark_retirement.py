"""Ambient benchmark ownership with real wrappers and synthetic SDK leaves."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

from src.ambient import segmenter
from src.stt import whisper_fast
from tools import benchmark_ambient as bench


_REAL_THREAD = threading.Thread


class _Outcome:
    def __init__(self):
        self.done = threading.Event()
        self.value = self.error = None


class _Leaves:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.stts = []
        self.predictors = []
        self.threads = []
        self.native_workers = []
        self.whisper_models = []
        self.silero_models = []
        self.wav_refs = []
        self.infer_refs = []
        self.windows = []
        self.events = []
        self.failures = {}
        self.hooks = {}
        self.blocks = {}
        self.whisper_calls = []
        self.silero_calls = []
        self.silero_resets = 0
        self.wav = np.column_stack((np.full(16000, .25, dtype=np.float32),
                                    np.full(16000, .75, dtype=np.float32)))
        self.wav_rate = 16000
        self.whisper_options = None
        self.cuda_values = []
        self.logs = []
        self.text = " synthetic ambient text "
        leaves = self

        class WhisperModel:
            def __init__(self, model, **kwargs):
                leaves.whisper_models.append(weakref.ref(self))
                leaves.whisper_options = (model, kwargs)
                if threading.current_thread() not in leaves.native_workers:
                    leaves.native_workers.append(threading.current_thread())
                leaves.phase("whisper-load")

            def transcribe(self, audio, **kwargs):
                worker = threading.current_thread()
                if worker not in leaves.native_workers:
                    leaves.native_workers.append(worker)
                leaves.infer_refs.append(weakref.ref(audio))
                leaves.whisper_calls.append((len(audio), audio.dtype,
                    float(audio[0]) if len(audio) else None, dict(kwargs)))
                leaves.phase("whisper-warmup" if len(leaves.whisper_calls) == 1 else "whisper-infer")
                item = SimpleNamespace(text=leaves.text, no_speech_prob=.1,
                                       avg_logprob=-.2, compression_ratio=1.1)
                return iter([item]), SimpleNamespace()

            def __del__(self):
                leaves.phase("whisper-drop")

        class Window:
            def __init__(self, array):
                self.array = array
                leaves.windows.append(weakref.ref(self))

            def __del__(self):
                leaves.phase("tensor-drop")

        class SileroModel:
            def __init__(self):
                leaves.silero_models.append(weakref.ref(self))

            def eval(self):
                leaves.phase("silero-eval")

            def __call__(self, window, rate):
                leaves.silero_calls.append((window.array.shape, window.array.dtype, rate))
                leaves.infer_refs.append(weakref.ref(window.array))
                leaves.phase("silero-warmup" if len(leaves.silero_calls) <= 2 else "silero-infer")
                return SimpleNamespace(item=lambda: .8)

            def reset_states(self):
                leaves.silero_resets += 1

            def __del__(self):
                leaves.phase("silero-drop")

        def load_silero(**kwargs):
            assert kwargs == {"repo_or_dir": "snakers4/silero-vad",
                              "model": "silero_vad", "trust_repo": True}
            model = SileroModel()
            leaves.phase("silero-load")
            return model, ()

        def synchronize(index):
            leaves.cuda_values.append(("synchronize", index, bool(leaves.stts and leaves.stts[-1]._model is not None)))
            leaves.phase("cuda-sync")

        def memory_allocated(index):
            leaves.cuda_values.append(("memory", index, bool(leaves.stts and leaves.stts[-1]._model is not None)))
            leaves.phase("cuda-memory")
            return 4 * 2**20

        torch = ModuleType("torch")
        torch.hub = SimpleNamespace(load=load_silero)
        torch.from_numpy = Window
        torch.no_grad = nullcontext
        torch.cuda = SimpleNamespace(synchronize=synchronize, memory_allocated=memory_allocated)
        monkeypatch.setitem(sys.modules, "torch", torch)
        faster_whisper = ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
        sf = ModuleType("soundfile")
        sf.read = self.read
        monkeypatch.setitem(sys.modules, "soundfile", sf)
        actual_stt = whisper_fast.FastWhisperSTT

        def construct_stt(*args, **kwargs):
            stt = actual_stt(*args, **kwargs)
            leaves.stts.append(stt)
            return stt

        monkeypatch.setattr(whisper_fast, "FastWhisperSTT", construct_stt)
        actual_factory = segmenter.make_silero_predictor

        def construct_predictor(*args, **kwargs):
            predictor = None
            try:
                predictor = actual_factory(*args, **kwargs)
                leaves.predictors.append(predictor)
                return predictor
            finally:
                # Observation must not add callback/PCM loans to a retained
                # factory failure traceback after the real factory clears them.
                args = ()
                kwargs.clear()
                predictor = None

        monkeypatch.setattr(segmenter, "make_silero_predictor", construct_predictor)
        original_info = bench.logger.info

        def report(message, *args, **kwargs):
            leaves.logs.append(str(message))
            leaves.phase("report")
            return original_info(message, *args, **kwargs)

        monkeypatch.setattr(bench.logger, "info", report)

    def phase(self, stage):
        self.events.append(stage)
        if stage in self.hooks:
            self.hooks[stage]()
        if stage in self.blocks:
            entered, release = self.blocks[stage]
            entered.set()
            assert release.wait(10), "synthetic benchmark gate not released"
        if stage in self.failures:
            raise self.failures[stage]("synthetic ambient SDK payload")

    def hold(self, stage):
        result = threading.Event(), threading.Event()
        self.blocks[stage] = result
        return result

    def read(self, _path, **kwargs):
        assert kwargs == {"dtype": "float32"}
        self.phase("wav-read")
        audio = self.wav.copy()
        self.wav_refs.append(weakref.ref(audio))
        return audio, self.wav_rate

    def stt(self, **kwargs):
        options = {"model": "synthetic-whisper", "device": "cpu", "wav": None, "iterations": 2}
        options.update(kwargs)
        return bench.bench_stt(**options)

    def start(self, function):
        outcome = _Outcome()

        def invoke():
            try:
                outcome.value = function()
            except BaseException as error:
                outcome.error = error
            finally:
                outcome.done.set()

        worker = _REAL_THREAD(target=invoke, daemon=True)
        self.threads.append(worker)
        worker.start()
        return outcome

    def release(self):
        errors = []
        for _entered, release in self.blocks.values():
            release.set()
        for worker in self.threads:
            try:
                worker._handle.join(3)
                assert worker._handle.is_done(), "benchmark observer still alive"
            except BaseException as error:
                errors.append(error)
        for resource in (*self.predictors, *self.stts):
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("synthetic benchmark teardown failed", errors)


@pytest.fixture
def leaves(monkeypatch):
    sdk = _Leaves(monkeypatch)
    try:
        yield sdk
    finally:
        sdk.release()


def test_stt_benchmark_retires_real_pinned_worker_with_wrapper_retained(leaves):
    """Baseline RED: normal public bench_stt return leaves its native worker."""
    assert leaves.stt() is None
    assert len(leaves.stts) == 1 and len(leaves.whisper_calls) == 3
    assert leaves.native_workers[0]._handle.is_done()
    assert leaves.whisper_models[0]() is None


def test_vad_benchmark_retires_real_predictor_with_callable_retained(leaves):
    """Baseline RED: a retained actual predictor still owns its loaded model."""
    assert bench.bench_vad(2) is None
    assert len(leaves.predictors) == 1 and callable(leaves.predictors[0])
    assert len(leaves.silero_calls) == 6
    assert leaves.silero_models[0]() is None
    assert leaves.predictors[0]._model is None
    assert leaves.predictors[0]._torch is None


def _retired(leaves):
    assert all(model() is None for model in (*leaves.whisper_models, *leaves.silero_models))
    assert all(worker._handle.is_done() for worker in leaves.native_workers)
    assert all(stt._model is None for stt in leaves.stts)
    assert all(predictor._model is None and predictor._torch is None for predictor in leaves.predictors)


def _errors(error):
    nested = getattr(error, "exceptions", None)
    return [error] if nested is None else [item for group in nested for item in _errors(group)]


@pytest.mark.parametrize("stage", [
    "whisper-load", "whisper-warmup", "whisper-infer", "report", "cuda-sync", "cuda-memory",
])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_stt_failure_retires_real_wrapper_and_detaches_owned_pcm(leaves, stage, error_type):
    leaves.failures[stage] = error_type
    device = "cuda:0" if stage.startswith("cuda") else "cpu"
    with pytest.raises(BaseException) as failure:
        leaves.stt(wav="synthetic.wav", device=device)
    if error_type is KeyboardInterrupt:
        assert isinstance(failure.value, KeyboardInterrupt)
    else:
        assert any(isinstance(error, RuntimeError) for error in _errors(failure.value))
    _retired(leaves)
    assert all(reference() is None for reference in (*leaves.wav_refs, *leaves.infer_refs)), (
        "retained failure owns benchmark PCM or native input"
    )


@pytest.mark.parametrize("stage", [
    "silero-load", "silero-eval", "silero-warmup", "silero-infer", "report",
])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_vad_failure_retires_actual_predictor_and_native_tensor(leaves, stage, error_type):
    leaves.failures[stage] = error_type
    with pytest.raises(BaseException) as failure:
        bench.bench_vad(2)
    if error_type is KeyboardInterrupt:
        assert isinstance(failure.value, KeyboardInterrupt)
    else:
        assert any(isinstance(error, RuntimeError) for error in _errors(failure.value))
    _retired(leaves)
    assert all(reference() is None for reference in leaves.infer_refs)
    assert all(window() is None for window in leaves.windows)


@pytest.mark.parametrize("kind", ["stt", "vad"])
@pytest.mark.parametrize("exit_code", [7, "synthetic exit payload"])
def test_model_control_preserves_safe_exit_code_with_failure_retained(leaves, kind, exit_code):
    stage = "whisper-warmup" if kind == "stt" else "silero-warmup"
    leaves.failures[stage] = lambda _message: SystemExit(exit_code)
    with pytest.raises(SystemExit) as failure:
        leaves.stt(wav="synthetic.wav") if kind == "stt" else bench.bench_vad(2)
    assert failure.value.code == (exit_code if type(exit_code) is int else 1)
    _retired(leaves)
    assert all(reference() is None for reference in (*leaves.wav_refs, *leaves.infer_refs))


def test_stt_wav_uses_first_column_and_keeps_constructor_and_warmup_contract(leaves):
    original = leaves.wav.copy()
    assert leaves.stt(wav="synthetic.wav") is None
    assert leaves.whisper_options == ("synthetic-whisper", {
        "device": "cpu", "device_index": 0, "compute_type": "int8_float16",
    })
    assert [item[0] for item in leaves.whisper_calls] == [16000, 16000, 16000]
    assert all(item[1] == np.dtype("float32") and item[2] == .25 for item in leaves.whisper_calls)
    assert leaves.whisper_calls[0][3]["beam_size"] == 1
    assert leaves.whisper_calls[0][3]["initial_prompt"] is None
    assert leaves.whisper_calls[0][3]["vad_filter"] is False
    np.testing.assert_array_equal(leaves.wav, original)
    _retired(leaves)


def test_synthetic_stt_warmup_is_two_seconds_and_full_audio_is_eight(leaves):
    assert leaves.stt() is None
    assert [item[0] for item in leaves.whisper_calls] == [32000, 128000, 128000]
    assert any("WER no medible" in message for message in leaves.logs)
    _retired(leaves)


def test_non_16k_wav_keeps_diagnostic_and_releases_decoded_audio(leaves):
    leaves.wav_rate = 8000
    with pytest.raises(SystemExit) as failure:
        leaves.stt(wav="synthetic.wav")
    assert str(failure.value) == "wav debe ser 16kHz (es 8000)"
    assert leaves.stts == [] and leaves.whisper_models == []
    assert all(reference() is None for reference in leaves.wav_refs)


def test_vad_keeps_1280_sample_chunks_warmup_and_recurrent_state(leaves):
    assert bench.bench_vad(2) is None
    assert len(leaves.silero_calls) == 6
    assert all(shape == (512,) and dtype == np.dtype("float32") and rate == 16000
        for shape, dtype, rate in leaves.silero_calls)
    assert leaves.silero_resets == 0
    _retired(leaves)


def test_cuda_metrics_are_sampled_before_model_retirement(leaves):
    assert leaves.stt(wav="synthetic.wav", device="cuda:0") is None
    assert leaves.cuda_values == [("synchronize", 0, True),
        ("synchronize", 0, True), ("memory", 0, True)]
    assert any("VRAM torch allocated: 4 MiB" in message for message in leaves.logs)
    assert leaves.events.index("cuda-memory") < leaves.events.index("whisper-drop")
    _retired(leaves)


def test_stt_external_timing_and_text_summary_are_preserved(leaves, monkeypatch):
    values = iter([1.0, 1.2, 2.0, 2.4])

    class Clock:
        def perf_counter(self):
            return next(values)

        def __getattr__(self, name):
            return getattr(time, name)

    monkeypatch.setattr(bench, "time", Clock())
    leaves.text = "text" * 30
    leaves.stt(wav="synthetic.wav")
    assert "  latencia mediana=300ms p95=390ms" in leaves.logs
    assert "  RTF=0.300  (continuo viable si <0.5; cómodo si <0.2)" in leaves.logs
    assert "  texto: " + repr(leaves.text[:80]) in leaves.logs
    _retired(leaves)


def test_vad_external_timing_format_is_preserved(leaves, monkeypatch):
    values = iter([1.0, 1.032])

    class Clock:
        def perf_counter(self):
            return next(values)

        def __getattr__(self, name):
            return getattr(time, name)

    monkeypatch.setattr(bench, "time", Clock())
    bench.bench_vad(2)
    assert leaves.logs == ["Silero VAD: 16.00 ms/chunk de 80ms (20.0% de un core — continuo viable si <50%)"]
    _retired(leaves)


def test_doa_retains_seed_shape_warmup_and_log_format(leaves, monkeypatch):
    from src.ambient.doa import DoAEstimator

    references, first_values = [], []

    def estimate(_estimator, audio):
        references.append(weakref.ref(audio))
        assert audio.shape == (48000, 6) and audio.dtype == np.float32
        first_values.append(audio.ravel()[:6].copy())

    values = iter([1.0, 1.02])

    class Clock:
        def perf_counter(self):
            return next(values)

        def __getattr__(self, name):
            return getattr(time, name)

    monkeypatch.setattr(DoAEstimator, "estimate", estimate)
    monkeypatch.setattr(bench, "time", Clock())
    assert bench.bench_doa(2) is None
    assert len(references) == 3
    expected = np.random.default_rng(7).normal(0, .1, size=(48000, 6)).astype(np.float32)
    for value in first_values:
        np.testing.assert_array_equal(value, expected.ravel()[:6])
    assert all(reference() is None for reference in references)
    assert leaves.logs == ["DoA GCC-PHAT: 10.0 ms por segmento de 3s (CPU)"]


@pytest.mark.parametrize("smoke", [True, False])
def test_main_preserves_flag_order_smoke_counts_and_device(leaves, monkeypatch, smoke):
    calls = []
    monkeypatch.setattr(bench, "bench_doa", lambda iterations: calls.append(("doa", iterations)))
    monkeypatch.setattr(bench, "bench_vad", lambda iterations: calls.append(("vad", iterations)))
    monkeypatch.setattr(bench, "bench_stt", lambda **kwargs: calls.append(("stt", kwargs)))
    argv = ["benchmark_ambient", "--smoke"] if smoke else [
        "benchmark_ambient", "--stt", "--vad", "--doa", "--iterations", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    assert bench.main() is None
    assert calls == [("doa", 3 if smoke else 4), ("vad", 10 if smoke else 4), ("stt", {
        "model": "./models/whisper-v3-turbo", "device": "cpu" if smoke else "cuda:0",
        "wav": None, "iterations": 2 if smoke else 4,
    })]


def test_stt_waits_real_pinned_worker_postlude_after_model_release(leaves, monkeypatch):
    from src.stt import pinned_executor

    entered, release = threading.Event(), threading.Event()
    original_run = pinned_executor._PinnedWorker.run

    def held_postlude(worker):
        try:
            return original_run(worker)
        finally:
            if worker in leaves.native_workers:
                entered.set()
                assert release.wait(10), "native worker postlude release missing"

    monkeypatch.setattr(pinned_executor._PinnedWorker, "run", held_postlude)
    outcome = leaves.start(lambda: leaves.stt(wav="synthetic.wav"))
    try:
        assert entered.wait(3)
        assert leaves.whisper_models[0]() is None
        assert not leaves.native_workers[0]._handle.is_done()
        assert not outcome.done.wait(.1)
        release.set()
        assert outcome.done.wait(3)
        assert outcome.error is None
    finally:
        release.set()
    _retired(leaves)


def test_vad_control_waits_actual_model_finalizer(leaves):
    leaves.failures["report"] = KeyboardInterrupt
    entered, release = leaves.hold("silero-drop")
    outcome = leaves.start(lambda: bench.bench_vad(2))
    try:
        assert entered.wait(3)
        assert not outcome.done.wait(.1)
        release.set()
        assert outcome.done.wait(3)
        assert isinstance(outcome.error, KeyboardInterrupt)
    finally:
        release.set()
    _retired(leaves)


def _record_resource_owners(monkeypatch):
    from src.core.resource_retirement import ResourceOwner

    owners = []
    initialize = ResourceOwner.__init__

    def remember(owner, *args, **kwargs):
        initialize(owner, *args, **kwargs)
        owners.append(owner)

    monkeypatch.setattr(ResourceOwner, "__init__", remember)
    return owners


@pytest.mark.parametrize("kind", ["stt", "vad"])
def test_external_retirement_keeps_real_benchmark_operation_owned(leaves, monkeypatch, kind):
    owners = _record_resource_owners(monkeypatch)
    stage = "whisper-warmup" if kind == "stt" else "silero-warmup"
    entered, release = leaves.hold(stage)
    outcome = leaves.start(lambda: leaves.stt(wav="synthetic.wav") if kind == "stt" else bench.bench_vad(2))
    try:
        assert entered.wait(3)
        assert len(owners) == 1
        closer = owners[0].begin_close()
        assert closer._handle.ident != 0 and not closer._handle.is_done()
        assert not outcome.done.wait(.1)
        models = leaves.whisper_models if kind == "stt" else leaves.silero_models
        assert models[0]() is not None
        release.set()
        assert outcome.done.wait(3)
        assert outcome.error is None
        assert closer._handle.is_done()
    finally:
        release.set()
    _retired(leaves)


def test_silero_factory_registers_returned_predictor_before_loading(leaves):
    registered = []

    def register(predictor):
        assert predictor._model is None and not predictor._ready
        assert leaves.silero_models == []
        leaves.events.append("predictor-registered")
        registered.append(predictor)

    predictor = segmenter.make_silero_predictor(register=register)
    assert registered == [predictor]
    assert leaves.events.index("predictor-registered") < leaves.events.index("silero-load")
    predictor.close()
    _retired(leaves)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_silero_registration_failure_retires_empty_wrapper_without_loading(leaves, error_type):
    registered = []

    def register(predictor):
        registered.append(predictor)
        assert predictor._model is None and not predictor._ready
        raise error_type("synthetic registration payload")

    with pytest.raises(error_type) as failure:
        segmenter.make_silero_predictor(register=register)
    assert len(registered) == 1
    assert registered[0]._silero_close_done.is_set()
    assert registered[0]._model is None and registered[0]._torch is None
    assert leaves.silero_models == []
    assert "synthetic registration payload" not in str(failure.value)
    assert failure.value.__cause__ is None and failure.value.__context__ is None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_retained_factory_failure_does_not_own_registration_callback_or_pcm(leaves, error_type):
    registered = []

    class Registration:
        def __init__(self):
            self.pcm = np.ones(64, dtype=np.float32)

        def __call__(self, predictor):
            registered.append(predictor)
            assert predictor._model is None and not predictor._ready
            raise error_type("synthetic callback payload")

    callback = Registration()
    callback_reference = weakref.ref(callback)
    pcm_reference = weakref.ref(callback.pcm)
    with pytest.raises(error_type) as failure:
        try:
            segmenter.make_silero_predictor(register=callback)
        finally:
            callback = None
    assert len(registered) == 1 and registered[0]._silero_close_done.is_set()
    assert registered[0]._model is None and registered[0]._torch is None
    assert leaves.silero_models == []
    assert failure.value.__cause__ is None and failure.value.__context__ is None
    assert callback_reference() is None
    assert pcm_reference() is None


def _run_owner_probe(mode):
    script = r'''
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import weakref

import pytest
from src.core import resource_retirement as core

namespace = runpy.run_path(sys.argv[1])
mode = sys.argv[2]
leaves = namespace['_Leaves'](pytest.MonkeyPatch())
owners = namespace['_record_resource_owners'](pytest.MonkeyPatch())
guards = []
start_calls = []
if mode == 'uncertain':
    class FailedStart(threading.Thread):
        def start(self):
            start_calls.append('start')
            raise RuntimeError('synthetic unconfirmed resource start')

    class ThreadingProxy:
        Thread = FailedStart

        def __getattr__(self, name):
            return getattr(threading, name)

    # Patch the real extracted helper. A detached STT-tool alias is not a seam.
    core.threading = ThreadingProxy()
    function = lambda: leaves.stt(wav='synthetic.wav')
else:
    _kind, kind, stage = mode.split(':')

    def reenter():
        try:
            owners[0].close()
        except RuntimeError as error:
            guards.append(str(error))

    leaves.hooks[stage] = reenter
    function = (lambda: leaves.stt(wav='synthetic.wav')) if kind == 'stt' else (
        lambda: namespace['bench'].bench_vad(2))
outcome = leaves.start(function)
if not outcome.done.wait(2):
    print(json.dumps({'reason': 'resource_owner_probe_hung', 'mode': mode}), flush=True)
    os._exit(1)
leaves.threads[0]._handle.join(1)
assert leaves.threads[0]._handle.is_done()
assert len(owners) == 1
if mode == 'uncertain':
    assert outcome.error is not None
    messages = [str(error) for error in namespace['_errors'](outcome.error)]
    assert any('close_start_failed' in message for message in messages)
    assert any('close_unconfirmed' in message for message in messages)
    assert len(start_calls) == 1
    assert not owners[0]._confirmed and owners[0]._handle.ident == 0
    owner = weakref.ref(owners[0])
    adapter = weakref.ref(leaves.stts[0])
    model = leaves.whisper_models[0]
    worker = leaves.native_workers[0]
    assert model() is not None and not worker._handle.is_done()
    outcome.error = None
    outcome.value = None
    leaves.stts.clear()
    leaves.predictors.clear()
    leaves.native_workers.clear()
    leaves.threads.clear()
    owners.clear()
    gc.collect()
    assert owner() is not None and adapter() is not None and model() is not None
    assert not worker._handle.is_done()
    reason = 'unconfirmed_start_retains_real_resource'
else:
    assert outcome.error is None
    assert guards and all('close_from_operation' in message for message in guards)
    namespace['_retired'](leaves)
    reason = 'native_reentry_rejected_before_dispatch'
print(json.dumps({'reason': reason, 'mode': mode}), flush=True)
# An uncertain owner deliberately remains live until process termination.
# Never reset its flags, retry a failed close or fabricate a native receipt.
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
        capture_output=True, text=True, timeout=15)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    return json.loads(detail)


@pytest.mark.parametrize(("kind", "stage"), [
    ("stt", "whisper-load"), ("stt", "whisper-warmup"), ("stt", "whisper-infer"),
    ("vad", "silero-load"), ("vad", "silero-eval"),
    ("vad", "silero-warmup"), ("vad", "silero-infer"),
])
def test_actual_native_callback_rejects_its_resource_owner_close(kind, stage):
    result = _run_owner_probe(f"guard:{kind}:{stage}")
    assert result["reason"] == "native_reentry_rejected_before_dispatch"


def test_uncertain_shared_owner_start_keeps_actual_whisper_worker_until_process_exit():
    result = _run_owner_probe("uncertain")
    assert result["reason"] == "unconfirmed_start_retains_real_resource"


def test_shared_operation_notification_interrupt_does_not_lose_native_retirement():
    script = r'''
import json
import os
import threading

from src.core.resource_retirement import ResourceOwner

body_entered = threading.Event()
body_release = threading.Event()
producer_done = threading.Event()
closer_waiting = threading.Event()
adapter_closed = threading.Event()
observer_done = threading.Event()
state = {'notify_interrupted': False, 'producer_control': False, 'observer_error': None}

class Adapter:
    def close(self):
        adapter_closed.set()

owner = ResourceOwner(Adapter(), owns_resource=True)
condition = owner._condition
original_wait, original_notify = condition.wait, condition.notify_all

def observed_wait(*args, **kwargs):
    if threading.current_thread() is owner._closer:
        closer_waiting.set()
    return original_wait(*args, **kwargs)

def interrupted_notify():
    if not state['notify_interrupted'] and not any(lease.active for lease in owner._operations):
        state['notify_interrupted'] = True
        raise KeyboardInterrupt('synthetic operation notification interrupt')
    return original_notify()

condition.wait = observed_wait
condition.notify_all = interrupted_notify

def produce():
    try:
        with owner.operation():
            body_entered.set()
            assert body_release.wait(3)
    except KeyboardInterrupt:
        state['producer_control'] = True
    finally:
        producer_done.set()

producer = threading.Thread(target=produce, daemon=True)
producer.start()
assert body_entered.wait(1)
owner.begin_close()
assert closer_waiting.wait(1), 'closer never waited for the admitted producer'

def observe():
    try:
        owner.wait_closed()
    except BaseException as error:
        state['observer_error'] = type(error).__name__
    finally:
        observer_done.set()

observer = threading.Thread(target=observe, daemon=True)
observer.start()
body_release.set()
assert producer_done.wait(1)
producer._handle.join(1)
assert producer._handle.is_done()
assert state['notify_interrupted'] and state['producer_control']
if not observer_done.wait(1):
    print(json.dumps({'reason': 'operation_notification_lost_wakeup'}), flush=True)
    os._exit(1)
observer._handle.join(1)
assert observer._handle.is_done() and owner._handle.is_done()
assert state['observer_error'] is None and adapter_closed.is_set()
assert owner._confirmed and owner.resource is None
print(json.dumps({'reason': 'inactive_lease_retires_without_notification'}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    assert json.loads(detail)["reason"] == "inactive_lease_retires_without_notification"


@pytest.mark.parametrize(("kind", "iterations"), [
    ("stt", 0), ("stt", -1), ("vad", 0), ("doa", 0),
])
def test_original_empty_iteration_failure_still_retires_resources(leaves, monkeypatch, kind, iterations):
    references = []
    if kind == "doa":
        from src.ambient.doa import DoAEstimator

        def estimate(_estimator, audio):
            references.append(weakref.ref(audio))

        monkeypatch.setattr(DoAEstimator, "estimate", estimate)
    with pytest.raises(Exception) as failure:
        if kind == "stt":
            leaves.stt(iterations=iterations)
        elif kind == "vad":
            bench.bench_vad(iterations)
        else:
            bench.bench_doa(iterations)
    assert failure.value is not None
    if kind == "stt":
        assert len(leaves.whisper_calls) == 1
    elif kind == "vad":
        assert len(leaves.silero_calls) == 2
    else:
        assert len(references) == 1
    _retired(leaves)
    assert all(reference() is None for reference in (*leaves.infer_refs, *references))


@pytest.mark.parametrize("kind", ["vad", "doa"])
def test_negative_vad_and_doa_iterations_keep_original_successful_summary(leaves, monkeypatch, kind):
    references = []
    if kind == "doa":
        from src.ambient.doa import DoAEstimator

        def estimate(_estimator, audio):
            references.append(weakref.ref(audio))

        monkeypatch.setattr(DoAEstimator, "estimate", estimate)
    values = iter([1.0, 1.032])

    class Clock:
        def perf_counter(self):
            return next(values)

        def __getattr__(self, name):
            return getattr(time, name)

    monkeypatch.setattr(bench, "time", Clock())
    if kind == "vad":
        assert bench.bench_vad(-1) is None
        assert len(leaves.silero_calls) == 2
        assert leaves.logs == ["Silero VAD: -32.00 ms/chunk de 80ms (-40.0% de un core — continuo viable si <50%)"]
    else:
        assert bench.bench_doa(-1) is None
        assert len(references) == 1
        assert leaves.logs == ["DoA GCC-PHAT: -32.0 ms por segmento de 3s (CPU)"]
    _retired(leaves)
    assert all(reference() is None for reference in (*leaves.infer_refs, *references))


@pytest.mark.parametrize("operation", ["call", "initialize"])
def test_interrupted_operation_reservation_does_not_leave_phantom_work(operation):
    script = r'''
import json
import os
import sys
import threading
from types import SimpleNamespace

from tools import benchmark_stt_ab as module

body_entered = threading.Event()
adapter_closed = threading.Event()

class Adapter:
    def load(self):
        body_entered.set()

    def transcribe_with_confidence(self, audio):
        body_entered.set()
        return SimpleNamespace(text='metadata-only result', elapsed_ms=1.0)

    def close(self):
        adapter_closed.set()

    def request_close(self):
        pass

adapter = Adapter()
engine = module._tc(adapter, owns_stt=True)
condition = engine._condition
state = {'fired': False, 'published_count': None, 'published_worker': False}

class InterruptOperationExit:
    def __enter__(self):
        condition.__enter__()
        return self

    def __exit__(self, *args):
        result = condition.__exit__(*args)
        operations = getattr(engine, '_operations', None)
        if operations is None:
            count = getattr(engine, '_admitted', 0)
            worker = getattr(engine, '_workers', {}).get(threading.current_thread(), 0)
        else:
            active = tuple(lease for lease in operations if lease.active)
            count = len(active)
            worker = sum(lease.owner is threading.current_thread() for lease in active)
        if count == 1 and worker == 1 and not state['fired']:
            state.update(fired=True, published_count=count, published_worker=True)
            raise KeyboardInterrupt('synthetic operation reservation exit interrupt')
        return result

    def __getattr__(self, name):
        return getattr(condition, name)

engine._condition = InterruptOperationExit()
try:
    if sys.argv[1] == 'call':
        engine(module.np.zeros(8, dtype=module.np.float32))
    else:
        engine.initialize()
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('operation fault did not propagate as control')
assert state['fired'] and state['published_count'] == 1 and state['published_worker']
assert not body_entered.is_set(), 'adapter ran before the interrupted reservation resolved'
done = threading.Event()
outcome = {}

def close_from_peer():
    try:
        engine.close()
        outcome['kind'] = 'returned'
    except BaseException as error:
        outcome['kind'] = type(error).__name__
    finally:
        done.set()

observer = threading.Thread(target=close_from_peer, daemon=True)
observer.start()
if not done.wait(1):
    print(json.dumps({'reason': 'phantom_operation_blocks_retirement',
                      'operation': sys.argv[1],
                      'body_entered': body_entered.is_set(),
                      'published_count': state['published_count'],
                      'admitted_after_control': getattr(engine, '_admitted', None),
                      'native_closer_started': engine._handle.ident != 0}), flush=True)
    # Never clear the operation, change owner flags or set a missing receipt.
    os._exit(1)
observer._handle.join(1)
assert observer._handle.is_done()
assert outcome['kind'] == 'returned'
assert adapter_closed.is_set()
assert engine.stt is None and engine._confirmed
print(json.dumps({'reason': 'operation_reservation_resolved',
                  'operation': sys.argv[1],
                  'body_entered': False, 'observer_native_exit': True}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, operation],
        capture_output=True, text=True, timeout=10)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    assert json.loads(detail)["reason"] == "operation_reservation_resolved"


@pytest.mark.parametrize("owner_api", ["stt", "resource"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_direct_close_defers_notification_error_until_actual_native_exit(error_type, owner_api):
    script = r'''
import json
import os
import sys
import threading

from tools import benchmark_stt_ab as module

error_type = {'RuntimeError': RuntimeError, 'KeyboardInterrupt': KeyboardInterrupt}[sys.argv[1]]
postlude_entered = threading.Event()
postlude_release = threading.Event()
adapter_closed = threading.Event()
done = threading.Event()
state = {'starts': 0, 'set_calls': 0, 'error_types': []}

class Adapter:
    def close(self):
        adapter_closed.set()

adapter = Adapter()
if sys.argv[2] == 'resource':
    from src.core.resource_retirement import ResourceOwner
    engine = ResourceOwner(adapter, owns_resource=True)
else:
    engine = module._tc(adapter, owns_stt=True)

def resource_link():
    return engine.resource if sys.argv[2] == 'resource' else engine.stt
original_run = engine._closer.run
original_start = engine._closer.start
original_event = engine._admission_done

def run_with_postlude():
    try:
        return original_run()
    finally:
        postlude_entered.set()
        assert postlude_release.wait(10), 'native closer postlude release missing'

def counted_start():
    state['starts'] += 1
    return original_start()

class NotificationFault:
    def set(self):
        state['set_calls'] += 1
        if state['set_calls'] == 1:
            raise error_type('synthetic notification payload')
        return original_event.set()

    def __getattr__(self, name):
        return getattr(original_event, name)

engine._closer.run = run_with_postlude
engine._closer.start = counted_start
engine._admission_done = NotificationFault()

def collect_types(error):
    nested = getattr(error, 'exceptions', None)
    return [type(error).__name__] if nested is None else [
        item for child in nested for item in collect_types(child)]

def observe():
    try:
        engine.close()
    except BaseException as error:
        state['error_types'] = collect_types(error)
    finally:
        done.set()

observer = threading.Thread(target=observe, daemon=True)
observer.start()
assert postlude_entered.wait(3), 'the actual closer did not reach its postlude'
assert adapter_closed.is_set()
assert not engine._handle.is_done()
if done.wait(.1):
    print(json.dumps({'reason': 'direct_close_returned_before_native_exit',
                      'error_type': sys.argv[1], 'starts': state['starts'],
                      'confirmed': engine._confirmed,
                      'native_exit': engine._handle.is_done()}), flush=True)
    # The terminal failure preserves the bug. Do not repair the notification
    # or clear any owner flag to rescue the intentionally failed subprocess.
    os._exit(1)
assert resource_link() is adapter and not engine._confirmed
postlude_release.set()
if not done.wait(1):
    print(json.dumps({'reason': 'direct_close_lost_notification'}), flush=True)
    os._exit(1)
observer._handle.join(1)
assert observer._handle.is_done() and engine._handle.is_done()
assert engine._confirmed and resource_link() is None
assert state['starts'] == 1 and state['set_calls'] == 1
assert not engine._admission_done.is_set()
assert sys.argv[1] in state['error_types']
print(json.dumps({'reason': 'notification_error_after_native_exit',
                  'error_type': sys.argv[1], 'native_starts': 1,
                  'confirmed': True}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, error_type.__name__, owner_api],
        capture_output=True, text=True, timeout=10)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    assert json.loads(detail)["reason"] == "notification_error_after_native_exit"


def test_external_silero_lease_uses_selected_caller_guard_before_owner_dispatch():
    script = r'''
import json
import os
import runpy
import sys
import threading

import numpy as np
import pytest

namespace = runpy.run_path(sys.argv[1])
leaves = namespace['_Leaves'](pytest.MonkeyPatch())
owners = namespace['_record_resource_owners'](pytest.MonkeyPatch())
module = namespace['bench']
clock_ready, clock_release = threading.Event(), threading.Event()
actual_time = module.time
state = {'clock_calls': 0, 'external_ident': None, 'starts': 0,
         'guard_messages': [], 'external_error': None}

def fail(reason):
    print(json.dumps({'reason': reason, 'native_owner_starts': state['starts'],
                      'guard_messages': state['guard_messages'],
                      'external_error': state['external_error']}), flush=True)
    # Never release a stuck owner by resetting leases, flags or notifications.
    os._exit(1)

class Clock:
    def perf_counter(self):
        state['clock_calls'] += 1
        if state['clock_calls'] == 1:
            # This is bench_vad's first clock read, after its warmup returns.
            clock_ready.set()
            assert clock_release.wait(10), 'benchmark clock release missing'
        return actual_time.perf_counter()

    def __getattr__(self, name):
        return getattr(actual_time, name)

module.time = Clock()
benchmark = leaves.start(lambda: module.bench_vad(2))
assert clock_ready.wait(3), 'benchmark did not finish its warmup'
assert len(owners) == 1 and len(leaves.predictors) == 1
owner, predictor = owners[0], leaves.predictors[0]
assert owner.resource is predictor
assert predictor._silero_active == 0
active = tuple(lease for lease in owner._operations if lease.active)
assert len(active) == 1 and active[0].owner is leaves.threads[0]
original_start = owner._closer.start

def counted_start():
    state['starts'] += 1
    return original_start()

owner._closer.start = counted_start

def close_from_external_native_call():
    if threading.get_ident() != state['external_ident']:
        return
    # The generic M operation belongs to a different thread. This call is
    # admitted only by the actual selected Silero predictor's native lease.
    assert not any(lease.active and lease.owner is threading.current_thread()
                   for lease in owner._operations)
    assert predictor._silero_current() is not None
    try:
        owner.close()
    except RuntimeError as error:
        state['guard_messages'].append(str(error))

leaves.hooks['silero-infer'] = close_from_external_native_call
external_done = threading.Event()

def external_call():
    state['external_ident'] = threading.get_ident()
    try:
        assert predictor(np.ones(1280, dtype=np.float32)) == .8
    except BaseException as error:
        state['external_error'] = type(error).__name__
    finally:
        external_done.set()

external = namespace['_REAL_THREAD'](target=external_call, daemon=True)
external.start()
if not external_done.wait(1):
    fail('external_silero_owner_close_deadlocked')
external._handle.join(1)
assert external._handle.is_done()
if state['external_error'] is not None or state['guard_messages'] != [
        'silero_close_from_worker', 'silero_close_from_worker']:
    fail('selected_silero_guard_not_observed')
assert state['starts'] == 0 and owner._handle.ident == 0
assert not owner._sealed and not owner._confirmed
assert predictor._silero_active == 0 and predictor._model is not None
assert not benchmark.done.is_set()

# Only after proving the selected guard do we release the real M producer.
clock_release.set()
if not benchmark.done.wait(1):
    fail('benchmark_did_not_retire_after_external_guard')
leaves.threads[0]._handle.join(1)
assert leaves.threads[0]._handle.is_done()
assert benchmark.error is None
namespace['_retired'](leaves)
assert owner._handle.is_done() and owner._confirmed
assert owner.resource is None and owner._caller_guard is None
assert state['starts'] == 1
print(json.dumps({'reason': 'selected_silero_guard_precedes_owner_dispatch',
                  'external_native_exit': True, 'benchmark_native_exit': True,
                  'owner_retired': True}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=15)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    assert json.loads(detail)["reason"] == "selected_silero_guard_precedes_owner_dispatch"


def _run_selected_silero_admission_probe(mode):
    script = r'''
import json
import os
import runpy
import sys
import threading

import pytest
from src.ambient import segmenter
from src.core import resource_retirement as core

namespace = runpy.run_path(sys.argv[1])
mode = sys.argv[2]
leaves = namespace['_Leaves'](pytest.MonkeyPatch())
owners = namespace['_record_resource_owners'](pytest.MonkeyPatch())
original_factory = segmenter.make_silero_predictor
captured = []
state = {'fired': False, 'active_at_fault': None, 'loaded_at_fault': None,
         'close_started_at_fault': None}

def active_count(predictor):
    leases = getattr(predictor, '_silero_leases', None)
    if leases is not None:
        return sum(lease.active for lease in leases)
    return predictor._silero_active

def install_fault(predictor):
    captured.append(predictor)
    condition = predictor._silero_condition

    class ConditionExitFault:
        def __enter__(self):
            condition.__enter__()
            return self

        def __exit__(self, *args):
            result = condition.__exit__(*args)
            count = active_count(predictor)
            closing = predictor._silero_close_started
            trigger = closing if mode == 'close' else count == 1
            if trigger and not state['fired']:
                state.update(fired=True, active_at_fault=count,
                    loaded_at_fault=predictor._model is not None,
                    close_started_at_fault=closing)
                raise KeyboardInterrupt('synthetic selected Silero metadata interrupt')
            return result

        def __getattr__(self, name):
            return getattr(condition, name)

    predictor._silero_condition = ConditionExitFault()

def factory(*args, register=None, **kwargs):
    if mode == 'load':
        def before_load(predictor):
            if register is not None:
                register(predictor)
            install_fault(predictor)
        return original_factory(*args, register=before_load, **kwargs)
    predictor = original_factory(*args, register=register, **kwargs)
    install_fault(predictor)
    return predictor

segmenter.make_silero_predictor = factory
benchmark = leaves.start(lambda: namespace['bench'].bench_vad(2))

def fail(reason):
    predictor = captured[0] if captured else None
    print(json.dumps({'reason': reason, 'mode': mode, 'fault': state,
        'models_created': len(leaves.silero_models),
        'native_prediction_calls': len(leaves.silero_calls),
        'active_now': active_count(predictor) if predictor is not None else None,
        'close_done': predictor._silero_close_done.is_set() if predictor is not None else None}), flush=True)
    # Preserve the stuck owner: no flag reset, synthetic notification, native
    # actor retry or fixture cleanup may rescue this intentionally failed child.
    os._exit(1)

if not benchmark.done.wait(1):
    fail('selected_silero_' + mode + '_admission_never_resolved')
leaves.threads[0]._handle.join(1)
assert leaves.threads[0]._handle.is_done()
assert state['fired'] and len(captured) == 1 and len(owners) == 1
predictor, owner = captured[0], owners[0]
if mode in ('load', 'prediction'):
    assert isinstance(benchmark.error, KeyboardInterrupt)
    assert state['active_at_fault'] == 1
    assert not state['close_started_at_fault']
    assert state['loaded_at_fault'] is (mode == 'prediction')
    assert leaves.silero_calls == [], 'an interrupted reservation entered the model body'
    assert len(leaves.silero_models) == (0 if mode == 'load' else 1)
    assert owner._handle.is_done() and owner._confirmed
    assert owner.resource is None and owner._caller_guard is None
    assert predictor._silero_close_done.is_set()
    namespace['_retired'](leaves)
    reason = 'selected_silero_operation_admission_resolved'
else:
    assert any(isinstance(error, (KeyboardInterrupt, RuntimeError))
               for error in namespace['_errors'](benchmark.error))
    assert state['active_at_fault'] == 0 and state['close_started_at_fault']
    assert state['loaded_at_fault']
    assert owner._handle.is_done() and not owner._confirmed
    assert owner.resource is predictor and predictor._model is not None
    assert owner in core._FAILED_OWNERS
    assert leaves.events.count('silero-drop') == 0
    second_done = threading.Event()
    second = {}

    def observe_failure():
        try:
            predictor.close()
            second['kind'] = 'returned'
        except BaseException as error:
            second['kind'] = type(error).__name__
            second['code'] = str(error)
        finally:
            second_done.set()

    observer = namespace['_REAL_THREAD'](target=observe_failure, daemon=True)
    observer.start()
    if not second_done.wait(1):
        fail('selected_silero_close_admission_never_resolved')
    observer._handle.join(1)
    assert observer._handle.is_done()
    assert second['kind'] == 'RuntimeError' and second['code'] == 'silero_close_failed'
    assert not owner._confirmed and owner.resource is predictor
    assert predictor._model is not None and leaves.silero_models[0]() is not None
    assert predictor._silero_close_done.is_set()
    assert leaves.events.count('silero-drop') == 0, 'unconfirmed close was retried'
    reason = 'selected_silero_close_admission_is_finite_failure'
print(json.dumps({'reason': reason, 'mode': mode, 'fault': state}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
        capture_output=True, text=True, timeout=15)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    return json.loads(detail)


@pytest.mark.parametrize("mode", ["load", "prediction"])
def test_selected_silero_operation_admission_interrupt_has_no_phantom_work(mode):
    result = _run_selected_silero_admission_probe(mode)
    assert result["reason"] == "selected_silero_operation_admission_resolved"


def test_selected_silero_close_admission_interrupt_is_finite_failed_receipt():
    result = _run_selected_silero_admission_probe("close")
    assert result["reason"] == "selected_silero_close_admission_is_finite_failure"


def _run_selected_silero_lifecycle_case(mode):
    script = r'''
import json
import os
import runpy
import sys
import threading

import pytest
from src.ambient import segmenter
from src.core import resource_retirement as core

namespace = runpy.run_path(sys.argv[1])
mode = sys.argv[2]
leaves = namespace['_Leaves'](pytest.MonkeyPatch())
owners = namespace['_record_resource_owners'](pytest.MonkeyPatch())
original_factory = segmenter.make_silero_predictor
captured = []
state = {'fired': False, 'nested_guard': None, 'parent_active': None}

def tls_fault(predictor):
    captured.append(predictor)
    local = predictor._silero_local

    class LocalFault:
        def __getattr__(self, name):
            return getattr(local, name)

        def __setattr__(self, name, value):
            publishing = value is not None and getattr(value, 'active', False)
            trigger = publishing if mode.endswith('publish') else value is None
            if name == 'lease' and trigger and not state['fired']:
                state['fired'] = True
                raise KeyboardInterrupt('synthetic selected Silero TLS interrupt')
            setattr(local, name, value)

    predictor._silero_local = LocalFault()

def factory(*args, register=None, **kwargs):
    if mode.startswith('load_'):
        def before_load(predictor):
            if register is not None:
                register(predictor)
            tls_fault(predictor)
        return original_factory(*args, register=before_load, **kwargs)
    predictor = original_factory(*args, register=register, **kwargs)
    if mode.startswith('prediction_'):
        tls_fault(predictor)
    else:
        captured.append(predictor)
    if mode == 'nested':
        def nested_borrower():
            if state['fired']:
                return
            state['fired'] = True
            parent = predictor._silero_current()
            assert parent is not None and parent.active
            try:
                with predictor._silero_admit():
                    assert predictor._silero_current() is parent
                    raise RuntimeError('synthetic nested metadata body')
            except RuntimeError:
                pass
            state['parent_active'] = parent.active and predictor._silero_current() is parent
            try:
                predictor.close()
            except RuntimeError as error:
                state['nested_guard'] = str(error)
        leaves.hooks['silero-warmup'] = nested_borrower
    elif mode == 'close_notification':
        event = predictor._silero_close_done

        class NotificationFault:
            def set(self):
                if not state['fired']:
                    state['fired'] = True
                    raise KeyboardInterrupt('synthetic selected close notification interrupt')
                return event.set()

            def __getattr__(self, name):
                return getattr(event, name)

        predictor._silero_close_done = NotificationFault()
    return predictor

segmenter.make_silero_predictor = factory
benchmark = leaves.start(lambda: namespace['bench'].bench_vad(2))
if not benchmark.done.wait(1):
    print(json.dumps({'reason': 'selected_silero_lifecycle_hung', 'mode': mode,
                      'fault_fired': state['fired']}), flush=True)
    os._exit(1)
leaves.threads[0]._handle.join(1)
assert leaves.threads[0]._handle.is_done()
assert state['fired'] and len(captured) == 1 and len(owners) == 1
predictor, owner = captured[0], owners[0]
assert all(not lease.active for lease in predictor._silero_leases)
assert owner._handle.is_done()
if mode == 'close_notification':
    assert isinstance(benchmark.error, RuntimeError)
    assert not owner._confirmed and owner.resource is predictor
    assert owner in core._FAILED_OWNERS
    assert predictor._silero_close_admission.finished
    assert predictor._silero_close_admission.failure == 'silero_close_failed'
    assert not predictor._silero_close_done.is_set()
    assert leaves.events.count('silero-drop') == 1
    second = leaves.start(predictor.close)
    if not second.done.wait(1):
        print(json.dumps({'reason': 'selected_close_notification_lost_receipt'}), flush=True)
        os._exit(1)
    leaves.threads[1]._handle.join(1)
    assert leaves.threads[1]._handle.is_done()
    assert isinstance(second.error, RuntimeError) and str(second.error) == 'silero_close_failed'
    assert leaves.events.count('silero-drop') == 1
    assert predictor._model is None and predictor._torch is None
    assert not owner._confirmed
else:
    if mode == 'nested':
        assert benchmark.error is None
        assert state['parent_active']
        assert state['nested_guard'] == 'silero_close_from_worker'
        assert len(leaves.silero_calls) == 6
    else:
        assert isinstance(benchmark.error, KeyboardInterrupt)
        assert len(leaves.silero_models) == (0 if mode == 'load_publish' else 1)
        assert len(leaves.silero_calls) == (2 if mode == 'prediction_restore' else 0)
    assert owner._confirmed and owner.resource is None and owner._caller_guard is None
    assert predictor._silero_close_admission.finished
    assert predictor._silero_close_admission.failure is None
    namespace['_retired'](leaves)
print(json.dumps({'reason': 'selected_silero_lifecycle_resolved', 'mode': mode}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
        capture_output=True, text=True, timeout=15)
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    return json.loads(detail)


@pytest.mark.parametrize("mode", [
    "load_publish", "load_restore", "prediction_publish", "prediction_restore",
])
def test_selected_silero_tls_control_resolves_real_operation_before_retirement(mode):
    result = _run_selected_silero_lifecycle_case(mode)
    assert result["reason"] == "selected_silero_lifecycle_resolved"


def test_selected_silero_nested_borrower_keeps_parent_native_lease_active():
    result = _run_selected_silero_lifecycle_case("nested")
    assert result["reason"] == "selected_silero_lifecycle_resolved"


def test_selected_silero_close_notification_failure_is_finite_and_never_retries():
    result = _run_selected_silero_lifecycle_case("close_notification")
    assert result["reason"] == "selected_silero_lifecycle_resolved"
