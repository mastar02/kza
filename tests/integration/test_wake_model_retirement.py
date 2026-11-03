"""Real wrapper, synthetic SDK leaves; no models/hardware/native-pool claim."""

import asyncio
from contextvars import ContextVar
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

import src.wakeword.detector as detector_module
from src.wakeword.detector import WakeWordDetector

pytestmark = pytest.mark.asyncio


class OWWModel:
    def __init__(self, probe):
        self.probe = probe

    def predict(self, audio):
        self.probe.predicts += 1
        self.probe.last_audio = audio
        self.probe.predict_entered.set()
        assert self.probe.predict_release.wait(5), "synthetic predict not released"
        return dict(self.probe.scores)

    def reset(self):
        self.probe.reset_entered.set()
        assert self.probe.reset_release.wait(5), "synthetic reset not released"
        self.probe.resets += 1

    def close(self):
        self.probe.sdk_closes += 1  # Decoy, not a reviewed SDK API.

    def __del__(self):
        if self.probe.on_drop is not None:
            self.probe.on_drop()


class CustomSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]


class SDKProbe:
    def __init__(self, monkeypatch):
        self.ctor_entered, self.ctor_release = threading.Event(), threading.Event()
        self.predict_entered, self.predict_release = threading.Event(), threading.Event()
        self.reset_entered, self.reset_release = threading.Event(), threading.Event()
        self.release_all()
        self.models, self.sessions, self.options = [], [], []
        self.predicts = self.resets = self.sdk_closes = 0
        self.last_audio = self.on_drop = None
        self.scores = {"hey_jarvis": 0.9}
        sdk = ModuleType("openwakeword")
        sdk.Model = self.construct
        monkeypatch.setitem(sys.modules, "openwakeword", sdk)
        ort = ModuleType("onnxruntime")
        ort.InferenceSession = self.custom_session
        monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    def construct(self, **options):
        self.options.append(options)
        model = OWWModel(self)
        self.models.append(weakref.ref(model))
        self.ctor_entered.set()
        assert self.ctor_release.wait(5), "synthetic constructor not released"
        return model

    def custom_session(self, path):
        assert path.endswith("custom.onnx")
        session = CustomSession()
        self.sessions.append(weakref.ref(session))
        return session

    def release_all(self):
        self.ctor_release.set()
        self.predict_release.set()
        self.reset_release.set()


class Callback:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def __call__(self, name, confidence):
        self.calls.append((name, confidence))
        self.entered.set()
        assert self.release.wait(5), "synthetic callback not released"


def detector():
    return WakeWordDetector(models=["hey_jarvis"], threshold=0.5, refractory_period=2.0)


def pcm():
    return np.zeros(1280, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def finish(probe, detectors, *, tasks=(), releases=()):
    probe.release_all()
    for release in releases:
        release.set()
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
    await asyncio.gather(*(item.aclose() for item in detectors), return_exceptions=True)


def assert_closed_metadata(item):
    assert item.get_active_models() == []
    info = item.get_info()
    assert set(info) == {"models", "threshold", "refractory_period", "inference_framework",
                         "loaded", "pretrained_count", "custom_count"}
    assert info["models"] == [] and info["loaded"] is False
    assert info["pretrained_count"] == info["custom_count"] == 0


async def test_concurrent_load_retains_constructor_and_publishes_detached_metadata(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.ctor_release.clear()
    item = detector()
    first = second = metadata = None
    try:
        first = asyncio.create_task(asyncio.to_thread(item.load))
        await until(probe.ctor_entered.is_set)
        second = asyncio.create_task(asyncio.to_thread(item.load))
        await until(lambda: item._wake_active == 2)
        metadata = asyncio.create_task(asyncio.to_thread(item.get_info))
        await until(metadata.done)
        assert metadata.result()["loaded"] is False
        assert metadata.result()["models"] == []
        assert len(probe.models) == 1 and probe.models[0]() is not None
        owner = item.begin_close()
        assert owner is item.begin_close() and not owner.done()
        assert not first.done() and not second.done()
        probe.ctor_release.set()
        assert await asyncio.gather(first, second) == [None, None]
        await asyncio.shield(owner)
        assert len(probe.models) == 1 and probe.models[0]() is None
        assert probe.options == [{"wakeword_models": ["hey_jarvis"], "inference_framework": "onnx"}]
        assert probe.sdk_closes == 0
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item], tasks=[first, second, metadata])


@pytest.mark.parametrize("operation", ["predict", "reset"])
async def test_native_operation_is_retained_while_metadata_bypasses_native_lock(monkeypatch, operation):
    probe = SDKProbe(monkeypatch)
    item = detector()
    caller = metadata = None
    try:
        await asyncio.to_thread(item.load)
        if operation == "predict":
            probe.predict_release.clear()
            entered = probe.predict_entered
            def call():
                return item.predict(pcm())
        else:
            probe.reset_release.clear()
            entered = probe.reset_entered
            call = item.reset
        caller = asyncio.create_task(asyncio.to_thread(call))
        await until(entered.is_set)
        item.models.append("alexa")  # The actual public configuration remains live.
        metadata = asyncio.create_task(asyncio.to_thread(lambda: (item.get_info(), item.get_active_models())))
        await until(metadata.done)
        info, names = metadata.result()
        assert info["loaded"] is True and info["pretrained_count"] == 1
        assert info["models"] == names == ["hey_jarvis", "alexa"]
        info["models"].append("detached-only")
        assert item.get_active_models() == ["hey_jarvis", "alexa"]
        assert not caller.done() and item._wake_active == 1
        owner = item.begin_close()
        assert not owner.done() and probe.models[0]() is not None
        probe.release_all()
        result = await caller
        assert result == ({"hey_jarvis": 0.9} if operation == "predict" else None)
        await asyncio.shield(owner)
        assert probe.models[0]() is None and probe.sdk_closes == 0
        assert item.models == ["hey_jarvis", "alexa"]
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item], tasks=[caller, metadata])


async def test_inline_callback_and_backend_are_retained_until_actual_callback_exit(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = detector()
    callback = Callback()
    callback_ref = weakref.ref(callback)
    entered, release, calls = callback.entered, callback.release, callback.calls
    item.on_detection(callback)
    callback = None
    caller = metadata = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(item.detect, pcm()))
        await until(entered.is_set)
        metadata = asyncio.create_task(asyncio.to_thread(item.get_info))
        await until(metadata.done)
        assert metadata.result()["models"] == ["hey_jarvis"]
        owner = item.begin_close()
        assert item._wake_active == 1 and not owner.done()
        assert callback_ref() is not None and probe.models[0]() is not None
        release.set()
        assert await caller == ("hey_jarvis", 0.9)
        await asyncio.shield(owner)
        assert calls == [("hey_jarvis", 0.9)]
        assert callback_ref() is None and probe.models[0]() is None
        assert item._on_detection is None and item._last_detection_time == {}
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item], tasks=[caller, metadata], releases=[release])


async def test_admitted_detect_can_enter_nested_predict_and_load_after_seal(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = detector()
    entered, release = threading.Event(), threading.Event()
    actual = item.predict

    def delayed(audio):
        entered.set()
        assert release.wait(5), "synthetic nested predict not released"
        return actual(audio)

    monkeypatch.setattr(item, "predict", delayed)
    caller = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(item.detect, pcm()))
        await until(entered.is_set)
        assert item._wake_active == 1 and probe.models == []
        owner = item.begin_close()
        assert not owner.done()
        release.set()
        assert await caller == ("hey_jarvis", 0.9)
        await asyncio.shield(owner)
        assert len(probe.models) == 1 and probe.models[0]() is None
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item], tasks=[caller], releases=[release])


async def test_reset_and_refractory_remain_reusable_without_reconstructing_backend(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = detector()
    clock = [100.0]
    monkeypatch.setattr(detector_module, "time", SimpleNamespace(time=lambda: clock[0]))
    try:
        assert await asyncio.to_thread(item.detect, pcm()) == ("hey_jarvis", 0.9)
        clock[0] += 0.1
        assert await asyncio.to_thread(item.detect, pcm()) is None
        await asyncio.to_thread(item.reset_refractory)
        assert probe.resets == 0
        assert await asyncio.to_thread(item.detect, pcm()) == ("hey_jarvis", 0.9)
        await asyncio.to_thread(item.reset)
        assert probe.resets == 1 and not item._wake_sealed
        assert item.get_info()["loaded"] is True
        assert await asyncio.to_thread(item.detect, pcm()) == ("hey_jarvis", 0.9)
        await asyncio.to_thread(item.set_threshold, 0.95, "hey_jarvis")
        assert await asyncio.to_thread(item.detect, pcm()) is None
        assert item.get_info()["threshold"] == 0.95
        assert len(probe.models) == 1
    finally:
        await finish(probe, [item])


async def test_callback_may_reset_after_seal_but_cannot_wait_for_own_retirement(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = detector()
    errors = []

    def callback(_name, _score):
        for function in (item.wait_closed, item.close, item.begin_close):
            try:
                function()
            except RuntimeError as error:
                errors.append(str(error))
        item.request_close()
        item.reset()
        item.reset_refractory()
        item.set_threshold(0.8)

    item.on_detection(callback)
    try:
        assert await asyncio.to_thread(item.detect, pcm()) == ("hey_jarvis", 0.9)
        assert errors == ["wake_close_from_worker"] * 3
        assert probe.resets == 1 and item._wake_sealed
        await item.aclose()
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item])


async def test_cancelled_call_and_repeated_close_observer_preserve_actual_native_exit(monkeypatch):
    probe = SDKProbe(monkeypatch)
    probe.predict_release.clear()
    item = detector()
    caller = observer = None
    started = asyncio.Event()

    async def observe():
        started.set()
        await item.aclose()

    try:
        caller = asyncio.create_task(asyncio.to_thread(item.predict, pcm()))
        await until(probe.predict_entered.is_set)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        owner = item.begin_close()
        observer = asyncio.create_task(observe())
        await started.wait()
        observer.cancel()
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        assert owner is item.begin_close()
        assert not owner.done() and not observer.done()
        assert item._wake_active == 1 and probe.models[0]() is not None
        progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(progressed.set)
        await progressed.wait()
        probe.predict_release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.shield(owner)
        assert probe.models[0]() is None and probe.sdk_closes == 0
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item], tasks=[caller, observer])


async def test_close_retires_own_custom_session_and_preserves_peer(monkeypatch, tmp_path):
    probe = SDKProbe(monkeypatch)
    item, peer = detector(), detector()
    path = tmp_path / "custom.onnx"
    path.write_bytes(b"synthetic-not-an-onnx-model")
    try:
        await asyncio.to_thread(item.load)
        await asyncio.to_thread(peer.load)
        await asyncio.to_thread(item._load_custom_model, str(path))
        assert len(probe.sessions) == 1 and probe.sessions[0]() is not None
        assert item.get_info()["custom_count"] == 1
        assert item.get_active_models() == ["hey_jarvis", "custom"]

        def failed_custom(_model, _audio):
            raise RuntimeError("synthetic custom prediction failure")

        monkeypatch.setattr(item, "_predict_custom", failed_custom)
        assert await asyncio.to_thread(item.predict, pcm()) == {"hey_jarvis": 0.9, "custom": 0.0}
        await item.aclose()
        assert probe.models[0]() is None and probe.sessions[0]() is None
        assert probe.models[1]() is not None and not peer._wake_sealed
        for function in (item.load,
                         lambda: item.predict(pcm()),
                         lambda: item.detect(pcm()),
                         item.reset, item.reset_refractory,
                         lambda: item.set_threshold(0.4),
                         lambda: item.on_detection(None)):
            with pytest.raises(RuntimeError, match="^wake_closed$"):
                await asyncio.to_thread(function)
        assert await asyncio.to_thread(peer.predict, pcm()) == {"hey_jarvis": 0.9}
        assert len(probe.models) == 2 and probe.sdk_closes == 0
        assert_closed_metadata(item)
    finally:
        await finish(probe, [item, peer])


@pytest.mark.parametrize("dtype,values,expected_dtype,expected", [
    (np.float32, [-1.0, 0.0, 0.5, 1.0], np.int16, [-32767, 0, 16383, 32767]),
    (np.float64, [-1.5, 1.5], np.int16, [-32767, 32767]),
    (np.int16, [-100, 100], np.int16, [-100, 100]),
    (np.float32, [-100.0, 100.0], np.float32, [-100.0, 100.0]),
    (np.float32, [], np.int16, []),
])
async def test_normalization_and_detection_keep_original_return_contracts(monkeypatch, dtype, values, expected_dtype, expected):
    probe = SDKProbe(monkeypatch)
    probe.scores = {"hey_jarvis": 0.6, "alexa": 0.9}
    item = detector()
    audio = np.array(values, dtype=dtype)
    try:
        result = await asyncio.to_thread(item.predict, audio)
        assert type(result) is dict and result == probe.scores
        assert probe.last_audio.dtype == expected_dtype
        assert np.array_equal(probe.last_audio, np.array(expected, dtype=expected_dtype))
        # Detection uses the first qualifying entry, not the maximum score.
        assert await asyncio.to_thread(item.detect, audio) == ("hey_jarvis", 0.6)
        assert len(probe.models) == 1
    finally:
        await finish(probe, [item])


async def test_onnx_path_stays_in_oww_route_with_framework_option_unchanged(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = WakeWordDetector(models=["synthetic-custom.onnx"], inference_framework="tflite")
    try:
        await asyncio.to_thread(item.load)
        assert probe.options == [{"wakeword_models": ["synthetic-custom.onnx"],
                                  "inference_framework": "tflite"}]
        assert probe.sessions == []  # The existing custom list remains empty.
        assert item.get_active_models() == []
        assert item.get_info()["loaded"] is True
        assert item.get_info()["pretrained_count"] == 1
    finally:
        await finish(probe, [item])


async def test_closer_is_context_empty_and_own_destructor_cannot_wait_on_itself(monkeypatch):
    probe = SDKProbe(monkeypatch)
    item = detector()
    marker = ContextVar("synthetic_wake_closer_context", default=None)
    observed = []

    def from_drop():
        observed.append(marker.get())
        try:
            item.wait_closed()
        except RuntimeError as error:
            observed.append(str(error))

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    token = marker.set("synthetic caller context")
    try:
        await asyncio.to_thread(item.load)
        probe.on_drop = from_drop
        loop.set_task_factory(asyncio.eager_task_factory)
        owner = item.begin_close()
        assert owner is item.begin_close()
        await asyncio.shield(owner)
        assert observed == [None, "wake_close_from_owner"]
        assert probe.models[0]() is None and probe.sdk_closes == 0
        assert_closed_metadata(item)
    finally:
        loop.set_task_factory(previous_factory)
        marker.reset(token)
        probe.on_drop = None
        await finish(probe, [item])
