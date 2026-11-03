"""Standalone STT benchmarks retire real adapters with synthetic SDK leaves."""

from __future__ import annotations

import builtins
from contextvars import ContextVar, copy_context
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

from src.core import resource_retirement as resources
from src.stt.parakeet_stt import ParakeetSTT
from src.stt.whisper_fast import FastWhisperSTT
from tools import benchmark_stt_ab as bench


_REAL_THREAD = threading.Thread
_CONTEXT = ContextVar("benchmark_test_context", default="empty")


class _Outcome:
    def __init__(self):
        self.done = threading.Event()
        self.value = self.error = None


class _SDK:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.adapters = []
        self.engines = []
        self.threads = []
        self.closers = []
        self.models = {"parakeet": [], "whisper": []}
        self.native_workers = []
        self.inferences = {"parakeet": 0, "whisper": 0}
        self.options = {}
        self.audio = []
        self.wav_audio = []
        self.events = []
        self.fail = {}
        self.hooks = {}
        self.blocks = {}
        self.wav_reads = 0
        self.fail_wav_read = None
        self.captured = [Path("synthetic-captured/a.wav")]
        self.rejected = [Path("synthetic-rejected/tv.wav")]
        self.hold_closer_postlude = False
        self.postlude_entered = threading.Event()
        self.postlude_release = threading.Event()
        self.bootstrap_pending = False
        self.bootstrap_entered = threading.Event()
        self.bootstrap_release = threading.Event()
        self.bootstrap_exited = threading.Event()
        self.fail_start_before = self.fail_start_after = False
        self.interrupt_joins = 0
        self.join_interrupts = 0
        self.join_interrupted = threading.Event()
        self.original_wavs = bench._wavs
        sdk = self

        class ParakeetModel:
            def __init__(self, model_name):
                sdk.models["parakeet"].append(weakref.ref(self))
                sdk.options["parakeet"] = model_name
                sdk.before("parakeet-load")

            def recognize(self, audio, **kwargs):
                sdk.infer("parakeet", audio, kwargs)
                return " synthetic transcript "

            def __del__(self):
                sdk.before("parakeet-drop")

        class WhisperModel:
            def __init__(self, model, **kwargs):
                sdk.models["whisper"].append(weakref.ref(self))
                sdk.options["whisper"] = (model, kwargs)
                sdk.before("whisper-load")

            def transcribe(self, audio, **kwargs):
                sdk.infer("whisper", audio, kwargs)
                segment = SimpleNamespace(text=" synthetic transcript ",
                    no_speech_prob=.1, avg_logprob=-.2, compression_ratio=1.2)
                return iter([segment]), SimpleNamespace()

        onnx = ModuleType("onnx_asr")
        onnx.load_model = ParakeetModel
        whisper = ModuleType("faster_whisper")
        whisper.WhisperModel = WhisperModel
        monkeypatch.setitem(sys.modules, "onnx_asr", onnx)
        monkeypatch.setitem(sys.modules, "faster_whisper", whisper)
        original_tc = bench._tc

        def remember(stt, *args, **kwargs):
            self.adapters.append(stt)
            engine = original_tc(stt, *args, **kwargs)
            self.engines.append(engine)
            return engine

        monkeypatch.setattr(bench, "_tc", remember)
        monkeypatch.setattr(bench, "_wavs", self.wavs)
        monkeypatch.setattr(bench.sf, "read", self.read)

        class Closer(_REAL_THREAD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                sdk.closers.append(self)

            def start(self):
                if sdk.fail_start_before:
                    raise RuntimeError("synthetic unconfirmed start")
                if sdk.bootstrap_pending:
                    with threading._active_limbo_lock:
                        threading._limbo[self] = self
                    threading._start_joinable_thread(
                        self._bootstrap, handle=self._handle, daemon=self.daemon)
                    assert sdk.bootstrap_entered.wait(3), "bootstrap did not enter"
                else:
                    super().start()
                if sdk.fail_start_after:
                    raise RuntimeError("synthetic partial start")

            def _bootstrap(self):
                if sdk.bootstrap_pending:
                    sdk.bootstrap_entered.set()
                    assert sdk.bootstrap_release.wait(10), "bootstrap release missing"
                try:
                    return super()._bootstrap()
                finally:
                    sdk.bootstrap_exited.set()

            def run(self):
                try:
                    return super().run()
                finally:
                    if sdk.hold_closer_postlude:
                        sdk.postlude_entered.set()
                        assert sdk.postlude_release.wait(10), "postlude release missing"

            def join(self, *args, **kwargs):
                if sdk.join_interrupts < sdk.interrupt_joins:
                    sdk.join_interrupts += 1
                    if sdk.join_interrupts == sdk.interrupt_joins:
                        sdk.join_interrupted.set()
                    raise KeyboardInterrupt("synthetic close observer interrupt")
                return super().join(*args, **kwargs)

        class ThreadingProxy:
            Thread = Closer

            def __getattr__(self, name):
                return getattr(threading, name)

        monkeypatch.setattr(resources, "threading", ThreadingProxy())

    def before(self, stage):
        self.events.append(stage)
        hook = self.hooks.get(stage)
        if hook:
            hook()
        if stage in self.blocks:
            entered, release = self.blocks[stage]
            entered.set()
            assert release.wait(10), "synthetic SDK gate not released"
        if stage in self.fail:
            raise self.fail[stage]("synthetic SDK payload must not escape")

    def infer(self, kind, audio, options):
        self.inferences[kind] += 1
        if kind == "whisper" and threading.current_thread() not in self.native_workers:
            self.native_workers.append(threading.current_thread())
        self.audio.append((kind, weakref.ref(audio), audio.shape, audio.dtype,
                           int(np.count_nonzero(audio)), dict(options)))
        self.before(kind + ("-warmup" if self.inferences[kind] == 1 else "-infer"))

    def hold(self, stage):
        gate = (threading.Event(), threading.Event())
        self.blocks[stage] = gate
        return gate

    def wavs(self, directory, limit):
        paths = self.rejected if "rejected" in directory else self.captured
        return paths[:limit] if limit else list(paths)

    def read(self, path, **kwargs):
        assert kwargs == {"dtype": "float32", "always_2d": False}
        self.wav_reads += 1
        if self.wav_reads == self.fail_wav_read:
            raise RuntimeError("synthetic WAV read failed")
        audio = np.full(16000, .25, dtype=np.float32)
        self.wav_audio.append(weakref.ref(audio))
        return audio, 16000

    def main(self, *, whisper=False):
        argv = ["benchmark_stt_ab", "--captured-dir", "synthetic-captured",
                "--rejected-dir", "synthetic-rejected"]
        if whisper:
            argv.append("--with-whisper")
        self.monkeypatch.setattr(sys, "argv", argv)
        return bench.main()

    def build(self, kind, **kwargs):
        if kind == "parakeet":
            return bench.build_parakeet("synthetic-parakeet", "es", **kwargs)
        return bench.build_whisper("synthetic-whisper", "cpu", "int8", "es", **kwargs)

    def start(self, function):
        result = _Outcome()

        def invoke():
            try:
                result.value = function()
            except BaseException as error:
                result.error = error
            finally:
                result.done.set()

        worker = _REAL_THREAD(target=invoke, daemon=True)
        self.threads.append(worker)
        worker.start()
        return result

    def release(self):
        self.bootstrap_release.set()
        self.postlude_release.set()
        for _entered, release in self.blocks.values():
            release.set()
        errors = []

        def attempt(operation):
            try:
                operation()
            except BaseException as error:
                errors.append(error)

        def join(worker):
            worker._handle.join(3)
            assert worker._handle.is_done()

        for worker in self.threads:
            attempt(lambda worker=worker: join(worker))
        for engine in self.engines:
            close = getattr(engine, "close", None)
            if close and not getattr(engine, "_confirmed", False):
                attempt(close)
        for adapter in self.adapters:
            attempt(adapter.close)
        for worker in self.closers:
            if worker._handle.ident:
                attempt(lambda worker=worker: join(worker))
        if errors:
            raise BaseExceptionGroup("synthetic SDK cleanup failed", errors)


@pytest.fixture
def sdk(monkeypatch):
    leaves = _SDK(monkeypatch)
    try:
        yield leaves
    finally:
        leaves.release()


def test_main_retires_parakeet_backend_with_callable_retained(sdk):
    """Baseline RED: completed main/report leaves the real adapter model live."""
    assert sdk.main() == 0
    assert len(sdk.engines) == 1 and callable(sdk.engines[0])
    assert sdk.inferences["parakeet"] == 3
    assert sdk.models["parakeet"][0]() is None


def test_main_retires_whisper_native_worker_with_callable_retained(sdk):
    """Baseline RED: the real pinned worker survives normal main/report return."""
    assert sdk.main(whisper=True) == 0
    worker = sdk.native_workers[0]
    assert worker._handle.is_done()
    assert sdk.models["whisper"][0]() is None


def _until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "test actor did not reach its gate"
        time.sleep(.001)


def _retired(sdk):
    assert all(model() is None for group in sdk.models.values() for model in group)
    assert all(worker._handle.is_done() for worker in sdk.native_workers)
    assert all(engine.stt is None for engine in sdk.engines)


def _errors(error):
    nested = getattr(error, "exceptions", None)
    return [error] if nested is None else [item for child in nested for item in _errors(child)]


@pytest.mark.parametrize("whisper", [False, True])
def test_main_no_input_does_not_construct_any_engine(sdk, whisper, capsys):
    sdk.captured = sdk.rejected = []
    assert sdk.main(whisper=whisper) == 1
    assert sdk.engines == sdk.adapters == []
    assert sdk.models == {"parakeet": [], "whisper": []}
    assert "ERROR: no se encontraron wavs" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
@pytest.mark.parametrize("stage", ["load", "warmup"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_builder_registers_before_partial_initialization_and_retires(
    sdk, kind, stage, error_type,
):
    registered = []
    sdk.fail[kind + "-" + stage] = error_type
    with pytest.raises(BaseException) as failure:
        with resources.ResourceSession(error_prefix="benchmark") as session:
            def register(engine):
                assert engine.stt._model is None
                if kind == "whisper":
                    assert engine.stt._executor is None
                registered.append(engine)
                session.register(engine)

            sdk.build(kind, register=register)
    assert registered == sdk.engines
    assert len(registered) == 1
    if error_type is KeyboardInterrupt:
        assert isinstance(failure.value, KeyboardInterrupt)
    else:
        assert any("benchmark_initialize_failed" in str(error) for error in _errors(failure.value))
        assert "synthetic SDK payload" not in str(failure.value)
    _retired(sdk)


@pytest.mark.parametrize("mode", ["second-builder", "report", "wav", "final-print"])
def test_main_failures_retire_all_created_engines(sdk, monkeypatch, mode):
    body_failure = RuntimeError("synthetic report failure")
    if mode == "second-builder":
        sdk.fail["whisper-load"] = RuntimeError
    elif mode == "report":
        original = bench._print_report

        def report(title, result):
            if result["name"] == "whisper":
                raise body_failure
            original(title, result)

        monkeypatch.setattr(bench, "_print_report", report)
    elif mode == "wav":
        # The fifth read fails after a previous row in the same Whisper run.
        # Retain the propagated error while verifying its old PCM was dropped.
        sdk.captured.append(Path("synthetic-captured/b.wav"))
        sdk.fail_wav_read = 5
    else:
        def output(*args, **kwargs):
            if args and "Gate de decisión" in str(args[0]):
                _retired(sdk)
                raise body_failure
            builtins.print(*args, **kwargs)

        monkeypatch.setattr(bench, "print", output, raising=False)
    with pytest.raises(RuntimeError) as failure:
        sdk.main(whisper=True)
    if mode in {"report", "final-print"}:
        assert failure.value is body_failure
    _retired(sdk)
    assert len(sdk.engines) == 2
    assert all(audio() is None for audio in sdk.wav_audio)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_close_waits_admitted_native_conversion_and_does_not_mutate_input(sdk, kind):
    engine = sdk.build(kind)
    entered, release = sdk.hold(kind + "-infer")
    audio = np.full(16000, .25, dtype=np.float32)
    original = audio.copy()
    inference = sdk.start(lambda: engine(audio))
    assert entered.wait(3)
    closed = sdk.start(engine.close)
    _until(lambda: engine._sealed)
    assert not closed.done.wait(.1)
    assert sdk.models[kind][0]() is not None
    assert engine.stt is sdk.adapters[0]
    release.set()
    assert inference.done.wait(3) and closed.done.wait(3)
    assert inference.error is None and closed.error is None
    assert inference.value[0] == "synthetic transcript"
    np.testing.assert_array_equal(audio, original)
    _retired(sdk)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
@pytest.mark.parametrize("stage", ["load", "warmup", "infer"])
def test_self_close_is_rejected_from_load_warmup_and_native_worker(sdk, kind, stage):
    errors = []

    def reenter():
        try:
            sdk.engines[-1].close()
        except RuntimeError as error:
            errors.append(str(error))

    sdk.hooks[kind + "-" + stage] = reenter
    engine = sdk.build(kind)
    if stage == "infer":
        engine(np.ones(16000, dtype=np.float32))
    assert errors == ["benchmark_close_from_operation"]
    assert not engine._sealed
    engine.close()
    _retired(sdk)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_admitted_load_finishes_warmup_after_external_seal(sdk, kind):
    entered, release = sdk.hold(kind + "-load")
    built = sdk.start(lambda: sdk.build(kind))
    assert entered.wait(3)
    engine = sdk.engines[0]
    closed = sdk.start(engine.close)
    _until(lambda: engine._sealed)
    assert not closed.done.wait(.1)
    release.set()
    assert built.done.wait(3) and closed.done.wait(3)
    assert built.error is None and closed.error is None
    assert built.value is engine
    assert sdk.inferences[kind] == 1
    assert sdk.audio[0][2:5] == ((16000,), np.dtype("float32"), 0)
    _retired(sdk)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_repeated_close_observer_interrupts_wait_for_native_postlude(sdk, kind):
    engine = sdk.build(kind)
    sdk.hold_closer_postlude = True
    sdk.interrupt_joins = 2
    closed = sdk.start(engine.close)
    assert sdk.postlude_entered.wait(3)
    assert sdk.join_interrupted.wait(3)
    assert not closed.done.wait(.1)
    assert engine.stt is sdk.adapters[0]
    assert not sdk.closers[0]._handle.is_done()
    sdk.postlude_release.set()
    assert closed.done.wait(3)
    assert isinstance(closed.error, KeyboardInterrupt)
    assert sdk.join_interrupts == 2
    assert sdk.closers[0]._handle.is_done()
    _retired(sdk)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_partial_start_with_pending_bootstrap_waits_native_handle(sdk, kind):
    engine = sdk.build(kind)
    sdk.bootstrap_pending = sdk.fail_start_after = True
    closed = sdk.start(engine.close)
    assert sdk.bootstrap_entered.wait(3)
    closer = sdk.closers[0]
    assert closer.ident is None and not closer._started.is_set()
    assert closer._handle.ident != 0 and not closer._handle.is_done()
    assert not closed.done.wait(.1)
    assert sdk.models[kind][0]() is not None
    sdk.bootstrap_release.set()
    assert closed.done.wait(3)
    assert isinstance(closed.error, RuntimeError)
    assert str(closed.error).startswith("benchmark_close_start_failed:")
    assert closer._handle.is_done() and sdk.bootstrap_exited.is_set()
    _retired(sdk)


def test_closer_context_is_empty_and_reentry_cannot_wait_on_itself(sdk):
    engine = sdk.build("parakeet")
    observed = []

    def from_drop():
        observed.append(_CONTEXT.get())
        try:
            engine.close()
        except RuntimeError as error:
            observed.append(str(error))

    sdk.hooks["parakeet-drop"] = from_drop
    token = _CONTEXT.set("caller")
    try:
        context = copy_context()
        closed = sdk.start(lambda: context.run(engine.close))
        assert closed.done.wait(3)
        assert closed.error is None
    finally:
        _CONTEXT.reset(token)
    assert observed == ["empty", "benchmark_close_from_owner"]
    _retired(sdk)


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_borrowed_callable_does_not_close_adapter_or_independent_peer(sdk, kind):
    def adapter():
        if kind == "parakeet":
            return ParakeetSTT(model_name="synthetic-borrowed", language="es")
        return FastWhisperSTT(model="synthetic-borrowed", device="cpu", compute_type="int8")

    external, peer = adapter(), adapter()
    sdk.adapters.append(peer)
    external.load()
    peer.load()
    engine = bench._tc(external)
    pcm = np.full(16000, .25, dtype=np.float32)
    expected = pcm.copy()
    assert engine(pcm)[0] == "synthetic transcript"
    engine.close()
    assert external.transcribe_with_confidence(pcm).text == "synthetic transcript"
    assert peer.transcribe_with_confidence(pcm).text == "synthetic transcript"
    assert external._model is not None and peer._model is not None
    assert engine.stt is None
    np.testing.assert_array_equal(pcm, expected)


def test_result_text_conversion_is_admitted_and_preserves_exact_values(sdk):
    errors = []
    calls = []
    engine = None

    class Result:
        @property
        def text(self):
            try:
                engine.close()
            except RuntimeError as error:
                errors.append(str(error))
            return "  exact result  "

        elapsed_ms = 7.25

    class Adapter:
        def transcribe_with_confidence(self, audio):
            return Result()

        def request_close(self):
            calls.append("seal")

        def close(self):
            calls.append("close")

    engine = bench._tc(Adapter())
    assert engine(np.ones(16, dtype=np.float32)) == ("  exact result  ", 7.25)
    assert errors == ["benchmark_close_from_operation"]
    assert not engine._sealed
    engine.close()
    assert calls == []


def _rejected_audio(engine):
    audio = np.ones(128, dtype=np.float32)
    reference = weakref.ref(audio)
    error = None
    try:
        engine(audio)
    except BaseException as caught:
        error = caught
    finally:
        audio = None
    return error, reference


def test_late_rejection_traceback_does_not_retain_caller_pcm(sdk):
    engine = sdk.build("parakeet")
    engine.close()
    error, audio = _rejected_audio(engine)
    assert isinstance(error, RuntimeError)
    assert str(error) == "benchmark_engine_closed"
    assert audio() is None
    _retired(sdk)


def test_statistics_samples_and_report_format_are_preserved(sdk, capsys):
    paths = [Path(f"synthetic/{index}.wav") for index in range(10)]
    calls = []

    def transcribe(_audio):
        index = len(calls)
        calls.append(index)
        return (" " if index % 2 else "text" * 20), (index + 1) * 20.0

    result = bench._run_engine("synthetic", transcribe, paths)
    assert result == {
        "name": "synthetic", "n": 10, "p50": 110.0, "p95": pytest.approx(191.0),
        "max": 200.0, "empty_pct": 50.0, "rtf": pytest.approx(.11),
        "samples": [(f"{index}.wav", " " if index % 2 else ("text" * 20)[:60])
                    for index in range(8)],
    }
    assert all(audio() is None for audio in sdk.wav_audio)
    bench._print_report("SYNTHETIC", result)
    output = capsys.readouterr().out
    assert "── SYNTHETIC · synthetic (n=10) ──" in output
    assert "p50=110ms  p95=191ms  max=200ms  RTF=0.110" in output
    assert "vacío     50.0%" in output


def test_wav_normalization_and_selection_remain_unchanged(sdk, monkeypatch):
    stereo = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    original = stereo.copy()
    monkeypatch.setattr(bench.sf, "read", lambda *args, **kwargs: (stereo, 8000))
    actual = bench.load_wav_mono16k(Path("synthetic.wav"))
    np.testing.assert_allclose(actual, np.array([1, 7/3, 11/3, 5], dtype=np.float32))
    np.testing.assert_array_equal(stereo, original)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: iter([Path("b.wav"), Path("a.wav")]))
    assert sdk.original_wavs("synthetic", 1) == [Path("a.wav")]
    assert sdk.original_wavs("synthetic", 0) == [Path("a.wav"), Path("b.wav")]


@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
def test_builder_options_and_single_float32_warmup_are_preserved(sdk, kind):
    engine = sdk.build(kind)
    assert sdk.inferences[kind] == 1
    assert sdk.audio[0][2:5] == ((bench.TARGET_SR,), np.dtype("float32"), 0)
    if kind == "parakeet":
        assert sdk.options[kind] == "synthetic-parakeet"
    else:
        assert sdk.options[kind] == ("synthetic-whisper", {
            "device": "cpu", "device_index": 0, "compute_type": "int8",
        })
        assert sdk.audio[0][5]["beam_size"] == 1
        assert sdk.audio[0][5]["vad_filter"] is False
    engine.close()
    _retired(sdk)


def test_uncertain_closer_start_keeps_models_and_native_worker_until_process_exit():
    script = r'''
import gc
import json
import os
import runpy
import sys
import weakref
import pytest

namespace = runpy.run_path(sys.argv[1])
leaves = namespace['_SDK'](pytest.MonkeyPatch())
engine = leaves.build('whisper')
adapter = weakref.ref(leaves.adapters[0])
model = leaves.models['whisper'][0]
worker = leaves.native_workers[0]
owner = weakref.ref(engine)
leaves.fail_start_before = True
try:
    engine.close()
except (RuntimeError, ExceptionGroup) as error:
    errors = namespace['_errors'](error)
    assert all(type(item) is RuntimeError for item in errors)
    assert any('benchmark_close_start_failed' in str(item) for item in errors)
    assert any('benchmark_close_unconfirmed:start' in str(item) for item in errors)
    assert not engine._confirmed
else:
    raise AssertionError('unconfirmed start was reported as successful retirement')
assert not worker._handle.is_done()
assert model() is not None
leaves.adapters.clear()
leaves.engines.clear()
leaves.closers.clear()
leaves.threads.clear()
leaves.native_workers.clear()
engine = None
gc.collect()
assert owner() is not None
assert adapter() is not None and model() is not None
assert not worker._handle.is_done()
print(json.dumps({'owner_retained': True, 'backend_retained': True,
                  'native_exit_unconfirmed': True}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "owner_retained": True, "backend_retained": True, "native_exit_unconfirmed": True,
    }

@pytest.mark.parametrize("kind", ["parakeet", "whisper"])
@pytest.mark.parametrize("stage", ["load", "warmup"])
@pytest.mark.parametrize("exit_code", [7, "synthetic private exit payload"])
def test_initialization_system_exit_preserves_safe_code_without_retaining_pcm(
    sdk, kind, stage, exit_code,
):
    sdk.fail[kind + "-" + stage] = lambda _message: SystemExit(exit_code)
    with pytest.raises(SystemExit) as failure:
        with resources.ResourceSession(error_prefix="benchmark") as session:
            sdk.build(kind, register=session.register)
    assert failure.value.code == (exit_code if type(exit_code) is int else 1)
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert "synthetic private exit payload" not in str(failure.value)
    _retired(sdk)
    assert all(reference() is None for _kind, reference, *_metadata in sdk.audio), (
        "retained control exception still owns warmup PCM"
    )


def test_external_close_waits_actual_result_text_conversion(sdk, monkeypatch):
    from src.stt import parakeet_stt

    entered, release = threading.Event(), threading.Event()
    hold = False

    class Result:
        def __init__(self, text, elapsed_ms):
            self._text, self.elapsed_ms = text, elapsed_ms

        @property
        def text(self):
            if hold:
                entered.set()
                assert release.wait(10), "result getter release missing"
            return self._text

    monkeypatch.setattr(parakeet_stt, "STTResult", Result)
    engine = sdk.build("parakeet")
    adapter = sdk.adapters[0]
    hold = True
    inference = sdk.start(lambda: engine(np.ones(16000, dtype=np.float32)))
    try:
        assert entered.wait(3)
        # The adapter has returned; only the benchmark's result conversion is
        # still admitted and must keep the external closer from retiring it.
        assert adapter._parakeet_active == 0
        closed = sdk.start(engine.close)
        _until(lambda: engine._sealed)
        assert not closed.done.wait(.1)
        assert engine.stt is adapter
        assert sdk.models["parakeet"][0]() is not None
        release.set()
        assert inference.done.wait(3) and closed.done.wait(3)
        assert inference.error is None and closed.error is None
        assert inference.value[0] == "synthetic transcript"
    finally:
        release.set()
    _retired(sdk)
    assert all(reference() is None for _kind, reference, *_metadata in sdk.audio)


def test_session_starts_both_closers_before_joining_first_held_model_drop(sdk):
    session = resources.ResourceSession(error_prefix="benchmark")
    session.__enter__()
    sdk.build("parakeet", register=session.register)
    sdk.build("whisper", register=session.register)
    whisper_worker = sdk.native_workers[0]
    entered, release = sdk.hold("parakeet-drop")
    finished = sdk.start(lambda: session.__exit__(None, None, None))
    try:
        assert entered.wait(3)
        assert not finished.done.wait(.1)
        # The first close is still blocked in the genuine Parakeet release.
        # The independently owned Whisper worker and closer must already exit.
        _until(lambda: whisper_worker._handle.is_done())
        _until(lambda: sdk.closers[1]._handle.ident != 0
            and sdk.closers[1]._handle.is_done())
        assert sdk.models["whisper"][0]() is None
        assert not sdk.closers[0]._handle.is_done()
        assert not finished.done.is_set()
        release.set()
        assert finished.done.wait(3)
        assert finished.error is None
    finally:
        release.set()
    _retired(sdk)


def test_benchmark_waits_real_whisper_worker_native_postlude(sdk, monkeypatch):
    from src.stt import pinned_executor

    entered, release = threading.Event(), threading.Event()
    targets = []
    original_run = pinned_executor._PinnedWorker.run

    def run_with_postlude(worker):
        try:
            return original_run(worker)
        finally:
            if worker in targets:
                entered.set()
                assert release.wait(10), "native worker postlude release missing"

    monkeypatch.setattr(pinned_executor._PinnedWorker, "run", run_with_postlude)
    try:
        engine = sdk.build("whisper")
        worker = sdk.native_workers[0]
        targets.append(worker)
        closed = sdk.start(engine.close)
        assert entered.wait(3)
        assert sdk.models["whisper"][0]() is None
        assert not worker._handle.is_done()
        assert not closed.done.wait(.1)
        assert engine.stt is sdk.adapters[0]
        release.set()
        assert closed.done.wait(3)
        assert closed.error is None
        assert worker._handle.is_done()
    finally:
        release.set()
    _retired(sdk)


def test_failed_model_close_keeps_ownership_after_closer_native_exit():
    script = r'''
import gc
import json
import os
import runpy
import sys
import weakref
import pytest

namespace = runpy.run_path(sys.argv[1])
leaves = namespace['_SDK'](pytest.MonkeyPatch())
engine = leaves.build('whisper')
adapter = leaves.adapters[0]
adapter_ref = weakref.ref(adapter)
model = leaves.models['whisper'][0]
worker = leaves.native_workers[0]
owner = weakref.ref(engine)

def fail_close():
    raise OSError('synthetic native close failure')

# Do not retain a bound original method in a monkeypatch undo stack: the
# resource must remain alive through its real failed owner, not test aliases.
adapter.close = fail_close
try:
    engine.close()
except BaseException as error:
    errors = namespace['_errors'](error)
    assert all(type(item) is RuntimeError for item in errors)
    assert any('benchmark_engine_close_failed:OSError' in str(item) for item in errors)
    assert all(item.__cause__ is None and item.__context__ is None for item in errors)
else:
    raise AssertionError('failed native close was reported as success')
assert leaves.closers[0]._handle.is_done()
assert not engine._confirmed
assert not worker._handle.is_done()
assert model() is not None
errors = None
leaves.adapters.clear()
leaves.engines.clear()
leaves.closers.clear()
leaves.threads.clear()
leaves.native_workers.clear()
adapter = engine = None
gc.collect()
assert owner() is not None
assert adapter_ref() is not None and model() is not None
assert owner().stt is adapter_ref()
assert not worker._handle.is_done()
print(json.dumps({'failed_owner_retained': True, 'closer_exited': True,
                  'model_worker_unconfirmed': True}), flush=True)
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "failed_owner_retained": True, "closer_exited": True, "model_worker_unconfirmed": True,
    }

def _run_benchmark_admission_fault(fault):
    script = r'''
import json
import os
import runpy
import sys
import threading

import pytest

namespace = runpy.run_path(sys.argv[1])
fault = sys.argv[2]
leaves = namespace['_SDK'](pytest.MonkeyPatch())
module = namespace['bench']
resources = namespace['resources']
session = resources.ResourceSession(error_prefix="benchmark")
engine = leaves.build('parakeet', register=session.register)
adapter = engine.stt
model = leaves.models['parakeet'][0]
state = {'fired': False, 'start_calls': 0, 'reentry_error': None}
original_start = engine._closer.start

def counted_start():
    state['start_calls'] += 1
    return original_start()

engine._closer.start = counted_start

# Arm faults only after wrapper, model and registry initialization succeeded.
if fault in ('registry', 'reentry'):
    original_lock = resources._OWNERS_LOCK

    class RegistryFault:
        def __enter__(self):
            original_lock.__enter__()
            if not state['fired']:
                state['fired'] = True
                if fault == 'registry':
                    original_lock.__exit__(None, None, None)
                    raise KeyboardInterrupt('synthetic registry admission interrupt')
                try:
                    engine.close()
                except RuntimeError as error:
                    state['reentry_error'] = str(error)
            return self

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    resources._OWNERS_LOCK = RegistryFault()
elif fault == 'condition':
    original_condition = engine._condition

    class ConditionExitFault:
        def __enter__(self):
            original_condition.__enter__()
            return self

        def __exit__(self, *args):
            result = original_condition.__exit__(*args)
            if engine._start_called and not state['fired']:
                state['fired'] = True
                raise KeyboardInterrupt('synthetic reservation exit interrupt')
            return result

        def __getattr__(self, name):
            return getattr(original_condition, name)

    engine._condition = ConditionExitFault()
elif fault == 'notification':
    original_event = engine._admission_done
    leaves.hold_closer_postlude = True

    class NotificationFault:
        def set(self):
            if not state['fired']:
                state['fired'] = True
                raise KeyboardInterrupt('synthetic notification interrupt')
            return original_event.set()

        def __getattr__(self, name):
            return getattr(original_event, name)

    engine._admission_done = NotificationFault()
else:
    raise AssertionError('unknown synthetic admission fault')

done = threading.Event()
outcome = {}

def observe():
    try:
        session.__exit__(None, None, None)
        outcome['kind'] = 'returned'
    except BaseException as error:
        outcome['kind'] = type(error).__name__
    finally:
        done.set()

observer = namespace['_REAL_THREAD'](target=observe, daemon=True)
observer.start()
if fault == 'notification':
    assert leaves.postlude_entered.wait(3), 'closer never reached native postlude'
    assert not done.wait(.05), 'observer returned before actual closer exit'
    leaves.postlude_release.set()

if not done.wait(1):
    print(json.dumps({'reason': 'admission_observer_hung', 'fault': fault,
                      'start_called': engine._start_called,
                      'native_start_calls': state['start_calls'],
                      'notification_set': engine._admission_done.is_set()}), flush=True)
    # This intentional terminal result records the deadlock. Never repair the
    # admission Event, clear a flag, cancel an owner or invoke fixture rescue.
    os._exit(1)
observer._handle.join(1)
assert observer._handle.is_done()
assert state['fired'] and engine._start_called
if fault in ('registry', 'condition'):
    assert outcome['kind'] == 'KeyboardInterrupt'
    assert state['start_calls'] == 0
    assert engine._handle.ident == 0
    assert not engine._confirmed
    assert engine.stt is adapter and model() is not None
    assert engine in resources._OWNERS
    assert engine in resources._FAILED_OWNERS
    reason = 'admission_resolved_unconfirmed'
else:
    assert state['start_calls'] == 1
    assert engine._handle.is_done()
    assert engine._confirmed and engine.stt is None and model() is None
    if fault == 'notification':
        assert outcome['kind'] == 'KeyboardInterrupt'
        assert not engine._admission_done.is_set()
        reason = 'notification_resolved_after_native_exit'
    else:
        assert outcome['kind'] == 'returned'
        assert state['reentry_error'] == 'benchmark_close_from_admission'
        reason = 'admission_reentry_rejected'
print(json.dumps({'reason': reason, 'fault': fault,
                  'native_start_calls': state['start_calls'],
                  'observer_native_exit': True,
                  'confirmed': engine._confirmed}), flush=True)
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), fault],
        capture_output=True, text=True, timeout=10,
    )
    detail = result.stdout.splitlines()[-1] if result.stdout.splitlines() else result.stderr[-2000:]
    assert result.returncode == 0, detail
    return json.loads(detail)


@pytest.mark.parametrize("fault", ["registry", "condition"])
def test_pre_dispatch_interrupt_resolves_reserved_close_without_starting_actor(fault):
    """Baseline RED: the session waits forever after reservation loses its owner."""
    result = _run_benchmark_admission_fault(fault)
    assert result["reason"] == "admission_resolved_unconfirmed"
    assert result["native_start_calls"] == 0
    assert result["confirmed"] is False


def test_interrupted_admission_notification_still_waits_actual_native_exit():
    result = _run_benchmark_admission_fault("notification")
    assert result["reason"] == "notification_resolved_after_native_exit"
    assert result["observer_native_exit"] is True
    assert result["confirmed"] is True


def test_reentry_from_active_start_admission_is_rejected_before_waiting():
    result = _run_benchmark_admission_fault("reentry")
    assert result["reason"] == "admission_reentry_rejected"
    assert result["confirmed"] is True
