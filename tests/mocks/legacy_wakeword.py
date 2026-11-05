"""Synthetic six-channel SDK leaves; real Threads provide exit receipts."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np

from tests.mocks.wakeword_capture import CaptureCall, TrainingAuthorityFactory


class _CallbackAudio(np.ndarray):
    def __new__(cls, values, leaves):
        value = np.asarray(values).view(cls)
        value.leaves = leaves
        return value

    def __array_finalize__(self, source):
        self.leaves = getattr(source, "leaves", None)

    def copy(self, order="C"):
        if self.leaves is not None:
            self.leaves.checkpoint("callback")
        return np.asarray(self).copy(order=order)


class LegacyLeaves:
    """Own only fixture gates and SDK actors, never production admission flags."""

    def __init__(self, monkeypatch):
        self.events = []
        self.entered = {}
        self.releases = {}
        self.hold = set()
        self.faults = {}
        self.streams = []
        self.detectors = []
        self.threads = []
        self.monitor_threads = []
        self.weak_audio = []
        self.weak_chunks = []
        self.weak_detectors = []
        self.stream_values = [.25]
        self.read_frames = []
        self.score_rows = [{"synthetic": .8}]
        self.predictions = []
        self.predict_calls = 0
        self.emit_callbacks = False
        self.callback_errors = []
        self.normal_interrupt = True
        self.trace = []
        self.outputs = []
        leaves = self

        class InputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.stop_kwargs = []
                self.close_kwargs = []
                self.worker = None
                self.native_stop = threading.Event()
                self.index = len(leaves.streams)
                leaves.streams.append(self)
                # For the unknown constructor case the SDK has really started
                # work but the caller never receives this Python handle.
                if "construct" in leaves.faults:
                    self._spawn(False)
                leaves.checkpoint("construct")

            def _spawn(self, callbacks):
                if self.worker is not None:
                    return

                def native():
                    try:
                        if callbacks:
                            for _ in range(len(leaves.score_rows) + 1):
                                frames = self.kwargs.get("blocksize", 1280)
                                audio = _CallbackAudio(leaves.audio(frames, self.index), leaves)
                                leaves.weak_audio.append(weakref.ref(audio))
                                try:
                                    self.kwargs["callback"](audio, frames, None, None)
                                except BaseException as error:
                                    leaves.callback_errors.append(type(error).__name__)
                                    break
                                finally:
                                    audio = None
                            leaves.event("callbacks-returned").set()
                        self.native_stop.wait()
                    finally:
                        leaves.event("native-returned").set()

                self.worker = threading.Thread(target=native, daemon=True)
                leaves.threads.append(self.worker)
                self.worker.start()

            def start(self, **kwargs):
                self._spawn(leaves.emit_callbacks and "callback" in self.kwargs)
                leaves.checkpoint("start")
                return self

            def read(self, frames):
                leaves.read_frames.append(frames)
                leaves.checkpoint("read")
                audio = leaves.audio(frames, self.index)
                leaves.weak_audio.append(weakref.ref(audio))
                return audio, False

            def stop(self, **kwargs):
                self.stop_kwargs.append(kwargs)
                assert kwargs == {"ignore_errors": False}
                leaves.checkpoint("stop")
                self.native_stop.set()

            def close(self, **kwargs):
                self.close_kwargs.append(kwargs)
                assert kwargs == {"ignore_errors": False}
                self.native_stop.set()
                if self.worker is not None:
                    leaves.join(self.worker)
                leaves.checkpoint("close")
                leaves.event("stream-closed-returned").set()

            def __enter__(self):
                return self.start()

            def __exit__(self, *args):
                self.stop()
                self.close()
                return False

        class Detector:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.closed = False
                self.native_stop = threading.Event()
                self.worker = None
                leaves.detectors.append(self)
                leaves.weak_detectors.append(weakref.ref(self))
                if "detector-construct" in leaves.faults:
                    self.worker = threading.Thread(target=self.native_stop.wait, daemon=True)
                    leaves.threads.append(self.worker)
                    self.worker.start()
                leaves.checkpoint("detector-construct")

            def load(self):
                leaves.checkpoint("detector-load")

            def get_active_models(self):
                return ["synthetic"]

            def predict(self, audio):
                assert audio.ndim == 1 and audio.dtype == np.float32
                leaves.weak_chunks.append(weakref.ref(audio))
                leaves.predictions.append(audio.copy())
                leaves.predict_calls += 1
                leaves.checkpoint("predict")
                index = leaves.predict_calls - 1
                if index >= len(leaves.score_rows):
                    if leaves.normal_interrupt:
                        raise KeyboardInterrupt()
                    return {}
                return dict(leaves.score_rows[index])

            def request_close(self):
                leaves.events.append("detector-seal")

            def close(self):
                leaves.checkpoint("detector-close")
                self.native_stop.set()
                if self.worker is not None:
                    leaves.join(self.worker)
                self.closed = True
                leaves.event("detector-closed-returned").set()

        sounddevice = ModuleType("sounddevice")
        sounddevice.InputStream = InputStream
        sounddevice.rec = sounddevice.wait = sounddevice.stop = self.forbidden
        detector = ModuleType("src.wakeword.detector")
        detector.WakeWordDetector = Detector
        monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
        monkeypatch.setitem(sys.modules, "src.wakeword.detector", detector)

    @staticmethod
    def forbidden(*args, **kwargs):
        raise AssertionError("global audio or implicit service operation is forbidden")

    def audio(self, frames, stream_index=0):
        value = self.stream_values[min(stream_index, len(self.stream_values) - 1)]
        return np.column_stack([
            np.full(frames, value if channel == 0 else -.1 * channel, np.float32)
            for channel in range(6)
        ])

    def event(self, name):
        return self.entered.setdefault(name, threading.Event())

    def release_event(self, name):
        return self.releases.setdefault(name, threading.Event())

    def checkpoint(self, name):
        self.events.append(name)
        self.event(name).set()
        if name in self.hold:
            assert self.release_event(name).wait(15), f"fixture gate not released: {name}"
        failure = self.faults.get(name)
        if failure is not None:
            raise failure()

    @staticmethod
    def join(worker):
        worker.join(5)
        assert not worker.is_alive() and worker._handle.is_done(), "actual native exit missing"

    def release(self):
        self.hold.clear()
        for event in self.releases.values():
            event.set()
        for stream in self.streams:
            stream.native_stop.set()
        for detector in self.detectors:
            detector.native_stop.set()
        for thread in self.threads:
            self.join(thread)

    def strict_receipts(self):
        for stream in self.streams:
            assert stream.stop_kwargs == [{"ignore_errors": False}]
            assert stream.close_kwargs == [{"ignore_errors": False}]
            if stream.worker is not None:
                assert stream.worker._handle.is_done()
        assert all(thread._handle.is_done() for thread in self.threads)


def load_alias(path, monkeypatch, root):
    name = "legacy_test_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(
        run=LegacyLeaves.forbidden, Popen=LegacyLeaves.forbidden), raising=False)
    for name in ("_voice_service_running", "_stop_voice_service", "_start_voice_service"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, LegacyLeaves.forbidden)
    return module


@asynccontextmanager
async def bound_legacy(tmp_path, monkeypatch, dashboard_auth, alias="train_nexa"):
    from pathlib import Path
    from tests.integration.test_capture_browser_owners import browser_app
    from tests.integration.test_global_capture_control import controlled
    from tests.mocks.calibration_authority import serve_capture_app

    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            async with serve_capture_app(app) as url:
                with monkeypatch.context() as patch:
                    from src.audio.native_capture_authority import (
                        NativeCaptureAuthorityClient, WakewordTrainingAuthorityClient,
                    )

                    leaves = LegacyLeaves(patch)
                    module = load_alias(Path("scripts") / (alias + ".py"), patch, tmp_path)
                    credential = dashboard_auth["Authorization"].split(" ", 1)[1]
                    factory = TrainingAuthorityFactory(url, credential)
                    calls = []
                    clients = []
                    initialize = WakewordTrainingAuthorityClient.__init__
                    monitor = NativeCaptureAuthorityClient._monitor

                    def observe(client, *args, **kwargs):
                        initialize(client, *args, **kwargs)
                        clients.append(client)

                    def monitor_exit(client):
                        leaves.monitor_threads.append(threading.current_thread())
                        try:
                            monitor(client)
                        finally:
                            leaves.checkpoint("monitor-exit")

                    patch.setattr(WakewordTrainingAuthorityClient, "__init__", observe)
                    patch.setattr(NativeCaptureAuthorityClient, "_monitor", monitor_exit)

                    def start(operation):
                        call = CaptureCall(operation)
                        calls.append(call)
                        return call

                    state = SimpleNamespace(graph=graph, app=app, url=url, credential=credential,
                        factory=factory, leaves=leaves, module=module, start=start, patch=patch,
                        clients=clients)
                    try:
                        yield state
                    finally:
                        if any(not call.done.is_set() for call in calls):
                            from tests.integration.test_global_capture_control import actor

                            snapshot = graph.control.snapshot()
                            if not snapshot["target_muted"]:
                                graph.control.request(True, snapshot["revision"], actor())
                        await asyncio.to_thread(leaves.release)
                        for call in calls:
                            await asyncio.to_thread(call.join)
                        for client in clients:
                            await asyncio.to_thread(client.close)


def error_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in error_leaves(child)]
    return [error]
