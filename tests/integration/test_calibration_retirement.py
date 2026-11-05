"""Standalone calibration retains its synthetic workers until actual exit."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import weakref
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.wakeword.detector import WakeWordDetector as _ActualWakeWordDetector
from tools import acoustic_calibration as ac
from tests.mocks.calibration_authority import positive_authority


_REAL_THREAD = threading.Thread


class _Failure(RuntimeError):
    pass


def _contains(error: BaseException, expected: BaseException) -> bool:
    if error is expected:
        return True
    return any(_contains(child, expected) for child in getattr(error, "exceptions", ()))


class _Harness:
    def __init__(self, monkeypatch, out_dir: Path, *, authority=None):
        self.out_dir = out_dir
        self.authority = positive_authority(monkeypatch) if authority is None else authority
        self.events: list[str] = []
        self.faults: set[str] = set()
        self.error = _Failure("synthetic calibration failure")
        self.usb_available = False
        self.hold_usb = False
        self.hold_callback = False
        self.emit_callback = False
        self.interrupt_join = False
        self.interrupt_callback_wait = False
        self.delay_bootstrap = False
        self.bootstrap_entered = threading.Event()
        self.bootstrap_release = threading.Event()
        self.bootstrap_exited = threading.Event()
        self.join_interrupted = threading.Event()
        self.callback_wait_interrupted = threading.Event()
        self.join_interrupt_error = KeyboardInterrupt("synthetic join interrupt")
        self.callback_wait_interrupt_error = KeyboardInterrupt("synthetic callback wait interrupt")
        self.caller_ident = None
        self.usb_entered = threading.Event()
        self.usb_release = threading.Event()
        self.usb_exited = threading.Event()
        self.predict_entered = threading.Event()
        self.predict_release = threading.Event()
        self.stream_stopped = threading.Event()
        self.stream_close_called = threading.Event()
        self.detector_closed = threading.Event()
        self.controller_invalidated = threading.Event()
        self.done = threading.Event()
        self.result: Path | None = None
        self.failure: BaseException | None = None
        self.callback_failure: BaseException | None = None
        self.callback = None
        self.predict_calls = 0
        self.stream_kwargs: dict = {}
        self.stop_kwargs: list[dict] = []
        self.close_kwargs: list[dict] = []
        self.threads: list[threading.Thread] = []
        self.pollers: list[threading.Thread] = []
        self.detectors: list = []
        self.controllers: list = []
        harness = self

        class Detector:
            def __init__(self, **kwargs):
                harness.detectors.append(self)
                self.models = kwargs["models"]
                self.chunk_ref = None

            def load(self):
                harness.mark("detector-load")

            def predict(self, chunk):
                self.chunk_ref = weakref.ref(chunk)
                harness.events.append("predict-enter")
                harness.predict_calls += 1
                harness.predict_entered.set()
                if harness.hold_callback:
                    assert harness.predict_release.wait(10), "callback release missing"
                harness.events.append("predict-exit")
                return {"first": 0.4, "second": 0.8}

            def request_close(self):
                harness.events.append("detector-seal")

            def close(self):
                harness.mark("detector-close")
                harness.detector_closed.set()

        class Controller:
            def __init__(self, **kwargs):
                harness.controllers.append(self)
                self.usb_port = kwargs["usb_port"]

            def open(self):
                harness.mark("controller-open")
                return harness.usb_available

            def read_spenergy(self):
                harness.events.append("usb-enter")
                harness.usb_entered.set()
                try:
                    if harness.hold_usb:
                        assert harness.usb_release.wait(10), "USB release missing"
                    return [0.0, 0.0, 0.0, 9999.0]
                finally:
                    harness.events.append("usb-exit")
                    harness.usb_exited.set()

            def request_stop(self):
                harness.events.append("controller-stop")

            def wait_stopped(self, *args, **kwargs):
                harness.events.append("controller-wait")
                return True

            def invalidate_device(self):
                harness.mark("controller-invalidate")
                harness.controller_invalidated.set()

        class InputStream:
            def __init__(self, **kwargs):
                harness.stream_kwargs = kwargs
                harness.callback = kwargs["callback"]
                harness.mark("stream-construct")

            def __enter__(self):
                harness.mark("stream-enter")
                harness.mark("stream-start")
                if harness.usb_available:
                    assert harness.usb_entered.wait(3), "poller never entered read"
                if harness.emit_callback:
                    if harness.hold_callback:
                        worker = _REAL_THREAD(target=harness.emit, daemon=True)
                        harness.threads.append(worker)
                        worker.start()
                        assert harness.predict_entered.wait(3), "callback never entered predict"
                    else:
                        harness.emit()
                return self

            def stop(self, **kwargs):
                harness.stream_stopped.set()
                harness.stop_kwargs.append(kwargs)
                harness.mark("stream-stop")

            def close(self, **kwargs):
                harness.stream_close_called.set()
                harness.close_kwargs.append(kwargs)
                harness.mark("stream-close")

            def __exit__(self, *args):
                self.stop()
                self.close()
                return False

        base_poller = getattr(ac, "_CapturePoller", _REAL_THREAD)

        class PollerThread(base_poller):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                harness.threads.append(self)
                harness.pollers.append(self)

            def start(self):
                harness.mark("poller-start")
                if harness.delay_bootstrap:
                    # Match CPython 3.13 Thread.start bookkeeping. The normal
                    # bootstrap will move this entry to _active itself.
                    with threading._active_limbo_lock:
                        threading._limbo[self] = self
                    threading._start_joinable_thread(
                        self._bootstrap, handle=self._handle, daemon=self.daemon,
                    )
                    assert harness.bootstrap_entered.wait(3), "bootstrap never entered"
                    harness.mark("poller-start-after")
                    return
                super().start()
                harness.mark("poller-start-after")

            def _bootstrap(self):
                if harness.delay_bootstrap:
                    harness.bootstrap_entered.set()
                    assert harness.bootstrap_release.wait(10), "bootstrap release missing"
                try:
                    return super()._bootstrap()
                finally:
                    harness.bootstrap_exited.set()

            def join(self, *args, **kwargs):
                if harness.interrupt_join and not harness.join_interrupted.is_set():
                    harness.join_interrupted.set()
                    raise harness.join_interrupt_error
                return super().join(*args, **kwargs)

        class CallbackCondition(threading.Condition):
            def wait(self, *args, **kwargs):
                if (
                    harness.interrupt_callback_wait
                    and not harness.callback_wait_interrupted.is_set()
                    and harness.predict_entered.is_set()
                    and not harness.predict_release.is_set()
                    and threading.get_ident() == harness.caller_ident
                ):
                    harness.callback_wait_interrupted.set()
                    raise harness.callback_wait_interrupt_error
                return super().wait(*args, **kwargs)

        class ThreadingProxy:
            Thread = PollerThread
            Condition = CallbackCondition

            def __getattr__(self, name):
                return getattr(threading, name)

        for name, attr, value in (
            ("sounddevice", "InputStream", InputStream),
            ("src.wakeword.detector", "WakeWordDetector", Detector),
            ("src.audio.xvf_controller", "XvfController", Controller),
        ):
            module = ModuleType(name)
            setattr(module, attr, value)
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(ac, "threading", ThreadingProxy())
        monkeypatch.setattr(ac, "_CapturePoller", PollerThread, raising=False)

    def mark(self, name: str):
        self.events.append(name)
        if name in self.faults:
            raise self.error

    def emit(self):
        try:
            assert self.callback is not None
            pcm = np.empty((64, 2), dtype=np.float32)
            pcm[:, 0] = 0.125
            pcm[:, 1] = 0.25
            self.callback(pcm, len(pcm), None, None)
        except BaseException as exc:
            self.callback_failure = exc

    def capture(self, *, duration_s: float = 0.0, channel: int = 1):
        self.caller_ident = threading.get_ident()
        try:
            self.result = ac.run_capture(
                "silencio", duration_s, 17, channel, "synthetic-model.onnx",
                self.out_dir, usb_port="5-5.4", authority=self.authority,
            )
        except BaseException as exc:
            self.failure = exc
        finally:
            self.done.set()

    def start(self, **kwargs):
        worker = _REAL_THREAD(target=lambda: self.capture(**kwargs), daemon=True)
        self.threads.append(worker)
        worker.start()
        return worker

    def rows(self):
        assert self.result is not None
        return [json.loads(row) for row in self.result.read_text().splitlines()]

    def release(self):
        self.bootstrap_release.set()
        self.usb_release.set()
        self.predict_release.set()
        for worker in self.threads:
            handle = getattr(worker, "_handle", None)
            if handle is not None and handle.ident != 0:
                handle.join(3)
                assert handle.is_done(), "test worker has not actually exited"
            elif worker.ident is not None:
                worker.join(3)
        assert all(not worker.is_alive() for worker in self.threads)


@pytest.fixture
def capture(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path / "capture")
    try:
        yield harness
    finally:
        harness.release()


def test_stream_constructor_failure_retires_loaded_detector(capture):
    """Baseline RED: InputStream construction was outside the cleanup scope."""
    capture.faults.add("stream-construct")
    capture.capture()
    assert capture.failure is not None and _contains(capture.failure, capture.error)
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not capture.out_dir.exists()


def test_success_waits_actual_usb_read_and_discards_post_seal_row(capture):
    """Baseline RED: the old one-second join reported success while read lived."""
    capture.usb_available = capture.hold_usb = True
    capture.start()
    assert capture.usb_entered.wait(3)
    assert capture.stream_stopped.wait(3)
    assert not capture.done.wait(1.15), "capture completed before actual USB exit"
    assert not capture.usb_exited.is_set()
    assert not capture.controller_invalidated.is_set()
    assert not capture.detector_closed.is_set()
    assert not capture.out_dir.exists()
    capture.usb_release.set()
    assert capture.done.wait(3)
    assert capture.failure is None
    assert capture.usb_exited.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.detector_closed.is_set()
    assert not [row for row in capture.rows() if row.get("kind") == "spenergy"]
    assert capture.events.index("usb-exit") < capture.events.index("controller-invalidate")


@pytest.mark.parametrize("fault", [
    "detector-load", "controller-open", "stream-enter", "stream-start",
    "stream-stop", "stream-close",
])
def test_partial_start_or_stream_failure_observes_other_cleanup(capture, fault):
    capture.faults.add(fault)
    capture.capture()
    assert capture.failure is not None and _contains(capture.failure, capture.error)
    assert capture.detector_closed.is_set()
    if capture.controllers:
        assert capture.controller_invalidated.is_set()
    if fault not in ("detector-load", "controller-open"):
        assert capture.stream_close_called.is_set()
    assert not capture.out_dir.exists()


def test_partial_poller_start_retains_actual_thread_until_read_exit(capture):
    capture.usb_available = capture.hold_usb = True
    capture.faults.add("poller-start-after")
    capture.start()
    assert capture.usb_entered.wait(3)
    assert not capture.done.wait(0.1)
    assert not capture.controller_invalidated.is_set()
    capture.usb_release.set()
    assert capture.done.wait(3)
    assert capture.failure is not None and _contains(capture.failure, capture.error)
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.usb_exited.is_set()
    assert capture.events.index("usb-exit") < capture.events.index("controller-invalidate")
    assert not capture.out_dir.exists()


def test_pending_bootstrap_waits_native_handle_without_thread_ident(capture):
    """CPython published a native handle before Thread._bootstrap ran."""
    capture.usb_available = capture.delay_bootstrap = True
    capture.faults.add("poller-start-after")
    capture.start()
    assert capture.bootstrap_entered.wait(3)
    poller = capture.pollers[0]
    assert poller.ident is None
    assert not poller._started.is_set()
    assert poller._handle.ident != 0
    assert not poller._handle.is_done()
    assert not capture.done.wait(0.1)
    assert not capture.detector_closed.is_set()
    assert not capture.controller_invalidated.is_set()
    assert not capture.usb_entered.is_set()
    assert not capture.out_dir.exists()
    capture.bootstrap_release.set()
    assert capture.done.wait(3)
    assert capture.failure is not None and _contains(capture.failure, capture.error)
    assert capture.bootstrap_exited.is_set()
    assert poller._handle.is_done()
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not capture.usb_entered.is_set()
    assert not capture.out_dir.exists()


def test_callback_exit_precedes_model_retirement_and_late_admission_is_denied(capture):
    capture.emit_callback = capture.hold_callback = True
    capture.start()
    assert capture.predict_entered.wait(3)
    assert capture.stream_close_called.wait(3)
    assert not capture.done.wait(0.1)
    assert not capture.detector_closed.is_set()
    assert not capture.out_dir.exists()
    capture.emit()  # A retained callback after stream close must not touch the model.
    assert capture.predict_calls == 1
    capture.predict_release.set()
    assert capture.done.wait(3)
    assert capture.failure is None
    assert capture.callback_failure is None
    assert capture.detector_closed.is_set()
    assert not [row for row in capture.rows() if row.get("kind") == "wake"]
    assert capture.events.index("predict-exit") < capture.events.index("detector-close")
    capture.release()
    assert capture.detectors[0].chunk_ref() is None


@pytest.mark.parametrize("fault", ["detector-close", "controller-invalidate"])
def test_cleanup_failure_does_not_skip_other_resources_or_publish_success(capture, fault):
    capture.faults.add(fault)
    capture.capture()
    assert capture.failure is not None and _contains(capture.failure, capture.error)
    assert "detector-close" in capture.events
    assert "controller-invalidate" in capture.events
    assert capture.stream_close_called.is_set()
    assert not capture.out_dir.exists()


def test_keyboard_interrupt_still_drains_workers_before_return(capture, monkeypatch):
    capture.usb_available = capture.hold_usb = True

    def interrupted_sleep(_seconds):
        raise KeyboardInterrupt()

    class TimeProxy:
        sleep = staticmethod(interrupted_sleep)

        def __getattr__(self, name):
            return getattr(__import__("time"), name)

    monkeypatch.setattr(ac, "time", TimeProxy())
    capture.start(duration_s=60.0)
    assert capture.usb_entered.wait(3)
    assert capture.stream_stopped.wait(3)
    assert not capture.done.wait(0.1)
    capture.usb_release.set()
    assert capture.done.wait(3)
    assert isinstance(capture.failure, KeyboardInterrupt)
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not capture.out_dir.exists()


def test_interrupt_during_join_keeps_actual_usb_receipt_owned(capture):
    capture.usb_available = capture.hold_usb = capture.interrupt_join = True
    capture.start()
    assert capture.usb_entered.wait(3)
    assert capture.join_interrupted.wait(3)
    assert not capture.done.wait(0.1)
    assert not capture.controller_invalidated.is_set()
    assert not capture.detector_closed.is_set()
    assert not capture.out_dir.exists()
    capture.usb_release.set()
    assert capture.done.wait(3)
    assert capture.failure is not None and _contains(capture.failure, capture.join_interrupt_error)
    assert capture.usb_exited.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.detector_closed.is_set()
    assert not capture.out_dir.exists()


def test_interrupt_during_callback_wait_keeps_real_callback_owned(capture):
    capture.emit_callback = capture.hold_callback = capture.interrupt_callback_wait = True
    capture.start()
    assert capture.predict_entered.wait(3)
    assert capture.callback_wait_interrupted.wait(3)
    assert not capture.done.wait(0.1)
    assert not capture.detector_closed.is_set()
    assert not capture.out_dir.exists()
    capture.predict_release.set()
    assert capture.done.wait(3)
    assert capture.failure is not None and _contains(capture.failure, capture.callback_wait_interrupt_error)
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not capture.out_dir.exists()


@pytest.mark.parametrize("failure_at", ["mkdir", "write"])
def test_output_failure_occurs_only_after_resource_retirement(capture, monkeypatch, failure_at):
    original_mkdir = Path.mkdir
    original_write = Path.write_text

    def fail_mkdir(path, *args, **kwargs):
        if path == capture.out_dir:
            assert capture.detector_closed.is_set()
            assert capture.controller_invalidated.is_set()
            raise capture.error
        return original_mkdir(path, *args, **kwargs)

    def fail_write(path, *args, **kwargs):
        if path.parent == capture.out_dir:
            assert capture.detector_closed.is_set()
            assert capture.controller_invalidated.is_set()
            raise capture.error
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_mkdir if failure_at == "mkdir" else original_mkdir)
    monkeypatch.setattr(Path, "write_text", fail_write if failure_at == "write" else original_write)
    capture.capture()
    assert capture.failure is capture.error
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not list(capture.out_dir.glob("*.jsonl"))


def test_success_preserves_rms_wake_metadata_and_device_binding(capture):
    capture.emit_callback = True
    capture.capture()
    assert capture.failure is None
    assert capture.callback_failure is None
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    rows = capture.rows()
    assert rows[0] == {
        "meta": True, "condition": "silencio", "device": 17,
        "channel": 1, "model": "synthetic-model.onnx", "duration_s": 0.0,
        "started_at": rows[0]["started_at"],
    }
    assert [(row["kind"], row["value"]) for row in rows[1:]] == [
        ("rms", 0.25), ("wake", 0.8),
    ]
    assert capture.stream_kwargs["device"] == 17
    assert capture.stream_kwargs["channels"] == 2
    assert capture.stop_kwargs == [{"ignore_errors": False}]
    assert capture.close_kwargs == [{"ignore_errors": False}]
    assert capture.controllers[0].usb_port == "5-5.4"
    assert capture.detectors[0].models == ["synthetic-model.onnx"]


def test_real_detector_keeps_native_model_until_callback_exit(capture, monkeypatch):
    """Use the real H lifecycle and load wrapper, replacing only the SDK leaf."""
    models = []
    detectors = []
    native_chunks = []

    class Model:
        def __init__(self, **kwargs):
            models.append(weakref.ref(self))

        def predict(self, chunk):
            native_chunks.append(weakref.ref(chunk))
            capture.predict_entered.set()
            assert capture.predict_release.wait(10), "native model release missing"
            return {"synthetic": 0.8}

        def reset(self):
            pass

    def make_detector(**kwargs):
        detector = _ActualWakeWordDetector(**kwargs)
        detectors.append(detector)
        return detector

    module = ModuleType("openwakeword")
    module.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", module)
    monkeypatch.setattr(sys.modules["src.wakeword.detector"], "WakeWordDetector", make_detector)
    capture.emit_callback = capture.hold_callback = True
    capture.start()
    assert capture.predict_entered.wait(3)
    assert capture.stream_close_called.wait(3)
    assert not capture.done.wait(0.1)
    assert detectors[0]._oww_model is models[0]()
    assert detectors[0]._loaded
    capture.predict_release.set()
    assert capture.done.wait(3)
    assert capture.failure is None
    assert capture.callback_failure is None
    capture.release()
    assert detectors[0]._oww_model is None
    assert detectors[0]._custom_models == {}
    assert not detectors[0]._loaded
    assert models[0]() is None
    assert all(chunk() is None for chunk in native_chunks)
    assert not [row for row in capture.rows() if row.get("kind") == "wake"]


def test_unconfirmed_start_keeps_ownership_until_process_exit(tmp_path):
    """A generic startup exception proves neither dispatch nor its absence."""
    script = r'''
import gc
import os
from pathlib import Path
import runpy
import sys
import time
import weakref
import pytest

namespace = runpy.run_path(sys.argv[1])
harness = namespace['_Harness'](pytest.MonkeyPatch(), Path(sys.argv[2]) / 'capture')
harness.usb_available = True
harness.faults.add('poller-start')
caller = harness.start()
deadline = time.monotonic() + 3
while 'poller-start' not in harness.events and time.monotonic() < deadline:
    time.sleep(0.005)
assert 'poller-start' in harness.events
time.sleep(0.1)
assert not harness.out_dir.exists()
assert not harness.detector_closed.is_set()
assert not harness.controller_invalidated.is_set()
if harness.done.is_set():
    assert harness.failure is not None
    assert namespace['_contains'](harness.failure, harness.error)
    caller.join(3)
    assert not caller.is_alive()
controller = weakref.ref(harness.controllers[0])
detector = weakref.ref(harness.detectors[0])
harness.controllers.clear()
harness.detectors.clear()
harness.threads.clear()
harness.failure = None
harness.error = None
caller = None
gc.collect()
assert detector() is not None, 'uncertain owner lost loaded detector'
assert controller() is not None, 'uncertain owner lost controller'
print('uncertain-start-retained', flush=True)
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path)],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "uncertain-start-retained" in result.stdout


def test_unsupported_runtime_is_rejected_before_native_start(capture, monkeypatch):
    class SysProxy:
        version_info = (3, 12, 0)

        def __getattr__(self, name):
            return getattr(sys, name)

    monkeypatch.setattr(ac, "sys", SysProxy())
    capture.usb_available = True
    capture.capture()
    assert isinstance(capture.failure, RuntimeError)
    assert str(capture.failure) == "calibration_poller_unsupported_runtime"
    assert "poller-start" not in capture.events
    assert len(capture.pollers) == 1
    assert capture.pollers[0]._handle.ident == 0
    assert not capture.pollers[0]._started.is_set()
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert "stream-construct" not in capture.events
    assert not capture.out_dir.exists()


def test_unsupported_handle_layout_is_rejected_before_native_start(capture, monkeypatch):
    class UnsupportedHandle:
        pass

    # Override the module's proxy only. Real Thread.__init__ still allocates
    # its genuine CPython handle; the guard must reject this layout mismatch.
    monkeypatch.setattr(ac.threading, "_ThreadHandle", UnsupportedHandle)
    capture.usb_available = True
    capture.capture()
    assert isinstance(capture.failure, RuntimeError)
    assert str(capture.failure) == "calibration_poller_unsupported_runtime"
    assert "poller-start" not in capture.events
    assert len(capture.pollers) == 1
    assert type(capture.pollers[0]._handle) is threading._ThreadHandle
    assert capture.pollers[0]._handle.ident == 0
    assert not capture.pollers[0]._started.is_set()
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert "stream-construct" not in capture.events
    assert not capture.out_dir.exists()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_callback_failure_absorbed_by_sdk_prevents_success(capture, monkeypatch, error_type):
    error = error_type("synthetic callback failure")

    def failed_predict(_self, _chunk):
        capture.predict_calls += 1
        capture.predict_entered.set()
        raise error

    detector_type = sys.modules["src.wakeword.detector"].WakeWordDetector
    monkeypatch.setattr(detector_type, "predict", failed_predict)
    capture.emit_callback = True
    capture.capture()
    assert capture.predict_calls == 1
    assert isinstance(capture.failure, error_type)
    assert capture.failure is not error
    assert capture.failure.__cause__ is None
    assert capture.failure.__context__ is None
    if error_type is RuntimeError:
        assert str(capture.failure) == "calibration_callback_failed"
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.stream_close_called.is_set()
    assert not capture.out_dir.exists()

@pytest.mark.parametrize(("exit_code", "expected"), [(7, 7), ("synthetic-exit-payload", 1)])
def test_callback_system_exit_keeps_status_without_retaining_payload(
    capture, monkeypatch, exit_code, expected,
):
    error = SystemExit(exit_code)

    def failed_predict(_self, _chunk):
        raise error

    detector_type = sys.modules["src.wakeword.detector"].WakeWordDetector
    monkeypatch.setattr(detector_type, "predict", failed_predict)
    capture.emit_callback = True
    capture.capture()
    assert isinstance(capture.failure, SystemExit)
    assert capture.failure is not error
    assert capture.failure.code == expected
    assert capture.failure.__cause__ is None
    assert capture.failure.__context__ is None
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert not capture.out_dir.exists()

@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_retained_callback_failure_does_not_retain_owned_pcm(capture, monkeypatch, error_type):
    chunks = []

    def failed_predict(_self, chunk):
        chunks.append(weakref.ref(chunk))
        # Do not retain this raw exception in a closure or test variable.
        raise error_type("synthetic SDK payload must be discarded")

    detector_type = sys.modules["src.wakeword.detector"].WakeWordDetector
    monkeypatch.setattr(detector_type, "predict", failed_predict)
    capture.emit_callback = True
    capture.capture()
    assert isinstance(capture.failure, error_type)
    assert capture.failure.__cause__ is None
    assert capture.failure.__context__ is None
    assert capture.callback_failure is None
    if error_type is RuntimeError:
        assert capture.failure.args == ("calibration_callback_failed",)
    else:
        assert capture.failure.args == ()
    # Keep capture.failure, wrapper and caller visible while checking that the
    # synthetic SDK traceback no longer owns the callback's private PCM copy.
    assert chunks and all(chunk() is None for chunk in chunks)
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.stream_close_called.is_set()
    assert not capture.out_dir.exists()


@pytest.mark.parametrize(("error_type", "body_fault"), [
    (OSError, False), (KeyboardInterrupt, False), (OSError, True),
])
def test_poller_failure_is_detached_and_preserves_body_failure(
    capture, monkeypatch, error_type, body_fault,
):
    def failed_poller(_controller, _rows, _stop, **kwargs):
        capture.usb_entered.set()
        try:
            # Fresh error: no test variable or closure retains its traceback.
            raise error_type("synthetic USB payload must be discarded")
        finally:
            capture.usb_exited.set()

    monkeypatch.setattr(ac, "_spenergy_poller", failed_poller)
    capture.usb_available = True
    if body_fault:
        capture.faults.add("stream-enter")
    capture.capture()
    assert capture.usb_entered.is_set()
    assert capture.usb_exited.is_set()
    assert capture.failure is not None
    if body_fault:
        assert isinstance(capture.failure, ExceptionGroup)
        assert capture.error in capture.failure.exceptions
        descriptors = [error for error in capture.failure.exceptions if error is not capture.error]
        assert len(descriptors) == 1
        failure = descriptors[0]
    else:
        failure = capture.failure
    if error_type is OSError:
        assert type(failure) is RuntimeError
        assert failure.args == ("calibration_poller_failed",)
    else:
        assert isinstance(failure, KeyboardInterrupt)
        assert failure.args == ()
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert capture.detector_closed.is_set()
    assert capture.controller_invalidated.is_set()
    assert capture.stream_close_called.is_set()
    assert all(poller._handle.is_done() for poller in capture.pollers)
    assert not capture.out_dir.exists()
