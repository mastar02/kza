"""Real Piper stream self-wait probes; root alone executes these subprocesses."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType
import weakref

import pytest


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "src/tts/piper_tts.py").is_file())


def _emit(mode, reason, **scalars):
    print("PIPER_SELF_WAIT=" + json.dumps({"mode": mode, "reason": reason, **scalars}), flush=True)


def _probe(mode):
    import numpy as np

    models = []
    source_exited = threading.Event()
    backend_exited = threading.Event()
    pulls = []

    class Voice:
        @classmethod
        def load(cls, model_path):
            assert model_path == "es_ES-davefx-medium.onnx"
            model = cls()
            models.append(weakref.ref(model))
            return model

        def synthesize(self, text):
            return np.zeros(3, dtype=np.float32)

        def synthesize_stream_raw(self, text):
            # This is the actual SDK seam used by PiperTTS, not a fake wrapper.
            try:
                for ordinal in range(3):
                    pulls.append(ordinal)
                    yield np.array([0, 16384, -16384], dtype=np.int16).tobytes()
            finally:
                source_exited.set()

        def __del__(self):
            backend_exited.set()

    sdk = ModuleType("piper")
    sdk.PiperVoice = Voice
    patch = pytest.MonkeyPatch()
    patch.setitem(sys.modules, "piper", sdk)
    from src.tts.piper_tts import PiperTTS

    adapter = PiperTTS()
    adapter.load(warmup=False)
    assert len(models) == 1 and models[0]() is adapter._voice
    stream = adapter.synthesize_stream("synthetic stream")
    errors = []
    close_attempted = threading.Event()

    def pull():
        np.testing.assert_array_equal(next(stream), [0.0, 0.5, -0.5])

    def consume_then_close():
        try:
            pull()
            close_attempted.set()
            try:
                adapter.close()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
            else:
                errors.append(("returned", ""))
        except BaseException as error:
            errors.append(("fixture_failure", type(error).__name__))

    first = None
    if mode == "migrate":
        first_errors = []

        def first_pull():
            try:
                pull()
            except BaseException as error:
                first_errors.append((type(error).__name__, str(error)))

        first = threading.Thread(target=first_pull, daemon=True, name="piper-consumer-A")
        first.start()
        first._handle.join(2)
        assert first._handle.is_done() and first_errors == []

    consumer = threading.Thread(
        target=consume_then_close, daemon=True,
        name="piper-consumer-B" if mode == "migrate" else "piper-consumer-A",
    )
    assert first is None or first is not consumer
    consumer.start()
    assert close_attempted.wait(2), f"consumer did not reach actual adapter.close: {errors}"
    consumer._handle.join(1)
    leases = [lease for lease in adapter._adapter_leases if lease.active]
    if not consumer._handle.is_done():
        _emit(
            mode, "self_wait_hang", actual_native_exited=False,
            active_leases=len(leases), on_stack=[lease.on_stack for lease in leases],
            lease_threads=[lease.thread.name for lease in leases],
            last_consumer=consumer.name,
            first_consumer_native_exited=first is None or first._handle.is_done(),
            sdk_pulls=len(pulls), sdk_source_exited=source_exited.is_set(),
            backend_retained=models[0]() is not None,
            adapter_sealed=adapter._adapter_sealed,
        )
        # Deliberately no stream.close, flag reset, Event rescue or disposal retry.
        os._exit(1)

    assert errors == [("RuntimeError", "tts_close_from_worker")], errors
    assert not adapter._adapter_sealed and adapter._adapter_close_admission is None
    assert len(leases) == 1 and not leases[0].on_stack
    assert leases[0].thread is consumer, "suspended ownership must follow the latest actual consumer Thread"
    assert len(pulls) == (2 if mode == "migrate" else 1)
    assert not source_exited.is_set() and models[0]() is adapter._voice

    # A real generator close resolves the suspended producer before another
    # native Thread is allowed to retire the wrapper/backend.
    stream.close()
    assert source_exited.is_set()
    assert not any(lease.active for lease in adapter._adapter_leases)
    assert adapter._adapter_retained_iterators == {}
    close_errors = []

    def external_close():
        try:
            adapter.close()
        except BaseException as error:
            close_errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=external_close, daemon=True, name="piper-external-closer")
    closer.start()
    closer._handle.join(2)
    if not closer._handle.is_done():
        _emit(mode, "external_close_hang", actual_native_exited=False)
        os._exit(1)
    assert close_errors == []
    assert adapter._voice is None and models[0]() is None and backend_exited.is_set()
    admission = adapter._adapter_close_admission
    assert admission.finished and admission.failure is None
    _emit(mode, "confirmed", actual_native_exited=True, source_exited=True,
          backend_released=True, consumer_rejected_before_seal=True)
    patch.undo()


def _run(mode):
    script = (
        "import os, runpy, sys; "
        "module = runpy.run_path(sys.argv[1]); "
        "module['_probe'](sys.argv[2]); os._exit(0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
        cwd=_root(), capture_output=True, text=True, timeout=15,
    )
    receipts = [json.loads(line.removeprefix("PIPER_SELF_WAIT="))
                for line in result.stdout.splitlines() if line.startswith("PIPER_SELF_WAIT=")]
    assert receipts, f"fixture failed before a native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{receipts[-1]}\n{result.stderr}"
    assert receipts[-1]["reason"] == "confirmed", receipts[-1]


def test_same_consumer_cannot_wait_on_its_suspended_real_piper_stream():
    _run("same")


def test_latest_consumer_cannot_wait_on_migrated_suspended_real_piper_stream():
    _run("migrate")
