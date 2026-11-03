"""Real adapter ownership, synthetic SDK leaves; no models/ORT/network.

Event deadlines only guard failures; no C03 or native pool teardown claim.
"""

import asyncio
from contextvars import ContextVar
import logging
import sys
import threading
from types import ModuleType
import weakref

import numpy as np
import pytest

import src.stt.parakeet_stt as parakeet_module
from src.stt.parakeet_stt import ParakeetSTT
from src.stt.whisper_fast import STTResult

pytestmark = pytest.mark.asyncio
CANARY = "SYNTHETIC_PRIVATE_PARAKEET_CANARY_71A6"


class NativeText(str):
    def __new__(cls, probe):
        value = super().__new__(cls, f"  {CANARY}  ")
        value.probe = probe
        return value

    def strip(self, *args, **kwargs):
        self.probe.strip_entered.set()
        assert self.probe.strip_release.wait(5), "synthetic strip not released"
        return super().strip(*args, **kwargs)


class Recognizer:
    def __init__(self, probe):
        self.probe = probe

    def recognize(self, audio, *, language):
        probe = self.probe
        probe.calls += 1
        probe.last_audio, probe.last_language = audio, language
        if probe.on_recognize is not None:
            probe.on_recognize()
        probe.recognize_entered.set()
        assert probe.recognize_release.wait(5), "synthetic recognize not released"
        if probe.error is not None:
            raise probe.error
        return None if probe.empty else NativeText(probe)

    def close(self):
        # A decoy is not a reviewed SDK lifecycle API.
        self.probe.sdk_close_calls += 1

    def __del__(self):
        callback = self.probe.on_drop
        if callback is not None:
            callback()


class SDKProbe:
    def __init__(self, monkeypatch):
        self.ctor_entered, self.ctor_release = threading.Event(), threading.Event()
        self.recognize_entered, self.recognize_release = threading.Event(), threading.Event()
        self.strip_entered, self.strip_release = threading.Event(), threading.Event()
        self.release_all()
        self.models = []
        self.calls = self.sdk_close_calls = 0
        self.last_audio = self.last_language = None
        self.empty, self.error = False, None
        self.on_drop = self.on_recognize = None
        sdk = ModuleType("onnx_asr")
        sdk.load_model = self.construct
        monkeypatch.setitem(sys.modules, "onnx_asr", sdk)

    def construct(self, *args, **kwargs):
        assert args == ("synthetic-parakeet",) and kwargs == {}
        model = Recognizer(self)
        self.models.append(weakref.ref(model))
        self.ctor_entered.set()
        assert self.ctor_release.wait(5), "synthetic constructor not released"
        return model

    def release_all(self):
        self.ctor_release.set()
        self.recognize_release.set()
        self.strip_release.set()


def adapter():
    return ParakeetSTT(model_name="synthetic-parakeet", language="es")


def pcm():
    return np.zeros(1600, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(probe, models, *, tasks=(), releases=()):
    # Always release native leaves before awaiting callers or closers.
    probe.release_all()
    for release in releases:
        release.set()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await asyncio.gather(*(model.aclose() for model in models), return_exceptions=True)


async def test_concurrent_load_is_once_and_close_retains_constructor(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.ctor_release.clear()
    model = adapter()
    first = second = None
    try:
        first = asyncio.create_task(asyncio.to_thread(model.load))
        await until(probe.ctor_entered.is_set)
        second = asyncio.create_task(asyncio.to_thread(model.load))
        await until(lambda: model._parakeet_active == 2)
        assert len(probe.models) == 1 and probe.models[0]() is not None
        assert model._model is None
        owner = model.begin_close()
        assert owner is model.begin_close()
        assert model._parakeet_sealed and not owner.done()
        assert not first.done() and not second.done()
        probe.ctor_release.set()
        assert await asyncio.gather(first, second) == [None, None]
        await asyncio.shield(owner)
        assert model._model is None and probe.models[0]() is None
        assert len(probe.models) == 1 and probe.sdk_close_calls == 0
        with pytest.raises(RuntimeError, match="^parakeet_closed$"):
            await asyncio.to_thread(model.load)
        assert len(probe.models) == 1
    finally:
        await finish(probe, [model], tasks=[first, second])


@pytest.mark.parametrize("phase", ["recognize", "strip", "result"])
async def test_recognize_strip_and_result_construction_remain_owned(monkeypatch, phase):
    probe = SDKProbe(monkeypatch)
    model = adapter()
    result_entered, result_release = threading.Event(), threading.Event()
    result_release.set()
    if phase == "recognize":
        probe.recognize_release.clear()
        entered = probe.recognize_entered
    elif phase == "strip":
        probe.strip_release.clear()
        entered = probe.strip_entered
    else:
        result_release.clear()
        entered = result_entered

        def construct_result(**kwargs):
            result_entered.set()
            assert result_release.wait(5), "synthetic result not released"
            return STTResult(**kwargs)

        monkeypatch.setattr(parakeet_module, "STTResult", construct_result)
    caller = None
    try:
        await asyncio.to_thread(model.load)
        await asyncio.to_thread(model.load)
        assert len(probe.models) == 1
        caller = asyncio.create_task(asyncio.to_thread(model.transcribe_with_confidence, pcm()))
        await until(entered.is_set)
        owner = model.begin_close()
        assert model._parakeet_active == 1
        assert not caller.done() and not owner.done()
        assert probe.models[0]() is not None
        probe.release_all()
        result_release.set()
        result = await caller
        assert isinstance(result, STTResult) and result.text == CANARY
        assert result.no_speech_prob is result.avg_logprob is result.compression_ratio is None
        await asyncio.shield(owner)
        assert model._model is None and probe.models[0]() is None
        assert probe.sdk_close_calls == 0
    finally:
        await finish(probe, [model], tasks=[caller], releases=[result_release])


async def test_nested_transcribe_admitted_before_seal_can_load_after_seal(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model = adapter()
    entered, release = threading.Event(), threading.Event()
    actual = model.transcribe_with_confidence

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5), "synthetic nested call not released"
        return actual(*args, **kwargs)

    monkeypatch.setattr(model, "transcribe_with_confidence", delayed)
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(model.transcribe, pcm()))
        await until(entered.is_set)
        assert model._parakeet_active == 1 and probe.models == []
        owner = model.begin_close()
        assert not owner.done()
        release.set()
        result = await caller
        assert type(result) is tuple and len(result) == 2
        assert result[0] == CANARY and type(result[1]) is float
        await asyncio.shield(owner)
        assert len(probe.models) == 1 and probe.models[0]() is None
        assert probe.calls == 1
    finally:
        await finish(probe, [model], tasks=[caller], releases=[release])


async def test_cancelled_call_and_repeated_close_observer_keep_native_owner(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.recognize_release.clear()
    model = adapter()
    caller = observer = None
    started = asyncio.Event()

    async def observe():
        started.set()
        await model.aclose()

    try:
        caller = asyncio.create_task(asyncio.to_thread(model.transcribe_with_confidence, pcm()))
        await until(probe.recognize_entered.is_set)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        owner = model.begin_close()
        observer = asyncio.create_task(observe())
        await started.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        assert owner is model.begin_close()
        assert not owner.done() and not observer.done()
        assert model._parakeet_active == 1 and probe.models[0]() is not None
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await progressed.wait()
        probe.recognize_release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.shield(owner)
        assert model._parakeet_active == 0
        assert model._model is None and probe.models[0]() is None
    finally:
        await finish(probe, [model], tasks=[caller, observer])


@pytest.mark.parametrize("dtype,values,expected", [
    (np.int16, [-16384, 16384], [-0.5, 0.5]),
    (np.int32, [-1073741824, 1073741824], [-0.5, 0.5]),
    (np.float64, [-0.5, 0.5], [-0.5, 0.5]),
    (np.float32, [-2.0, 2.0], [-2.0, 2.0]),
])
async def test_data_contract_normalization_language_and_ignored_sample_rate(monkeypatch, dtype, values, expected):
    probe = SDKProbe(monkeypatch)
    model = adapter()
    audio = np.array(values, dtype=dtype)
    try:
        result = await asyncio.to_thread(model.transcribe_with_confidence, audio, 8000)
        assert isinstance(result, STTResult) and type(result.text) is str
        assert result.text == CANARY and type(result.elapsed_ms) is float
        assert result.elapsed_ms >= 0
        assert result.no_speech_prob is result.avg_logprob is result.compression_ratio is None
        assert probe.last_language == "es"
        assert probe.last_audio.dtype == np.float32
        assert np.array_equal(probe.last_audio, expected)
        if dtype == np.float32:
            assert probe.last_audio is audio
        probe.empty = True
        text, elapsed = await asyncio.to_thread(model.transcribe, audio, 8000)
        assert text == "" and type(elapsed) is float
        assert len(probe.models) == 1
    finally:
        await finish(probe, [model])


async def test_peer_stays_usable_and_closed_adapter_cannot_reload(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model, peer = adapter(), adapter()
    try:
        await asyncio.to_thread(model.load)
        await asyncio.to_thread(peer.load)
        assert len(probe.models) == 2
        await model.aclose()
        assert model._model is None and probe.models[0]() is None
        assert peer._model is not None and probe.models[1]() is not None
        assert not peer._parakeet_sealed
        for function in (model.load,
                         lambda: model.transcribe(pcm()),
                         lambda: model.transcribe_with_confidence(pcm())):
            with pytest.raises(RuntimeError, match="^parakeet_closed$"):
                await asyncio.to_thread(function)
        text, elapsed = await asyncio.to_thread(peer.transcribe, pcm())
        assert text == CANARY and type(elapsed) is float
        assert len(probe.models) == 2 and probe.sdk_close_calls == 0
    finally:
        await finish(probe, [model, peer])


async def test_sdk_error_releases_admission_preserves_error_and_does_not_log_text(monkeypatch, caplog):
    probe = SDKProbe(monkeypatch)
    model = adapter()
    probe.error = ValueError("synthetic recognize failure")
    caplog.set_level(logging.DEBUG, logger=parakeet_module.__name__)
    try:
        with pytest.raises(ValueError, match="^synthetic recognize failure$"):
            await asyncio.to_thread(model.transcribe_with_confidence, pcm())
        assert model._parakeet_active == 0
        probe.error = None
        result = await asyncio.to_thread(model.transcribe_with_confidence, pcm())
        assert result.text == CANARY and len(probe.models) == 1
        assert any('Parakeet STT completed in ' in record.getMessage() for record in caplog.records)
        assert not any(CANARY in record.getMessage() for record in caplog.records)
        await model.aclose()
        assert model._model is None and probe.sdk_close_calls == 0
        # The caller's SDK exception may own its traceback; no all-copy claim.
    finally:
        await finish(probe, [model])


async def test_native_and_closer_reentry_are_rejected_and_close_context_is_empty(monkeypatch):
    probe = SDKProbe(monkeypatch)
    model = adapter()
    worker_errors, closer_observed = [], []
    marker = ContextVar("synthetic_parakeet_closer_context", default=None)

    def from_native():
        for function in (model.wait_closed, model.close, model.begin_close):
            try:
                function()
            except RuntimeError as error:
                worker_errors.append(str(error))

    def from_drop():
        closer_observed.append(marker.get())
        try:
            model.wait_closed()
        except RuntimeError as error:
            closer_observed.append(str(error))

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    token = marker.set("synthetic caller context")
    try:
        probe.on_recognize = from_native
        result = await asyncio.to_thread(model.transcribe_with_confidence, pcm())
        assert result.text == CANARY
        assert worker_errors == ["parakeet_close_from_worker"] * 3
        assert not model._parakeet_sealed
        probe.on_recognize = None
        probe.on_drop = from_drop
        loop.set_task_factory(asyncio.eager_task_factory)
        owner = model.begin_close()
        assert owner is model.begin_close()
        await asyncio.shield(owner)
        assert closer_observed == [None, "parakeet_close_from_owner"]
        assert model._model is None and probe.models[0]() is None
        assert probe.sdk_close_calls == 0
    finally:
        loop.set_task_factory(previous_factory)
        marker.reset(token)
        probe.on_recognize = probe.on_drop = None
        await finish(probe, [model])
