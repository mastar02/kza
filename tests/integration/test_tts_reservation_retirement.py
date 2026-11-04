"""Genuine legacy TTS reservation REDs; root alone executes the children."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


def _root():
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "tests/integration/test_tts_model_retirement.py").is_file())


def _active(adapter):
    values = vars(adapter)
    for name in ("_tts_leases", "_adapter_leases"):
        if name in values:
            return sum(lease.active for lease in values[name])
    return values["_tts_active"]


class _ReservationExitFault:
    def __init__(self, condition, adapter):
        self.condition, self.adapter = condition, adapter
        self.fired = False

    def __enter__(self):
        self.condition.__enter__()
        return self

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if not self.fired and _active(self.adapter) == 1:
            self.fired = True
            raise KeyboardInterrupt
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)


def _emit(kind, operation, reason, **scalars):
    print("TTS_RESERVATION=" + json.dumps({"kind": kind, "operation": operation,
          "reason": reason, **scalars}), flush=True)


def _probe(kind, operation):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    if operation != "load":
        adapter.load()
        assert len(probe.models) == 1
    before_calls = probe.calls
    condition_name = "_adapter_condition" if "_adapter_condition" in vars(adapter) else "_tts_condition"
    fault = _ReservationExitFault(getattr(adapter, condition_name), adapter)
    setattr(adapter, condition_name, fault)
    stream = None
    interrupted = False
    try:
        if operation == "load":
            adapter.load()
        elif operation == "synthesize":
            adapter.synthesize(fixture["CANARY"])
        else:
            stream = adapter.synthesize_stream(fixture["CANARY"])
            next(stream)
    except KeyboardInterrupt:
        interrupted = True
    assert interrupted and fault.fired, "fault did not reach actual admission publication"
    assert probe.calls == before_calls == 0, "SDK body entered before reservation returned"
    assert len(probe.models) == (0 if operation == "load" else 1)
    errors = []
    entered = threading.Event()

    def observe_close():
        entered.set()
        try:
            adapter.close()
        except BaseException as error:
            errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=observe_close, daemon=True, name="tts-real-close-observer")
    closer.start()
    assert entered.wait(2), "actual close Thread was not dispatched"
    closer._handle.join(1)
    if not closer._handle.is_done():
        _emit(kind, operation, "phantom_reservation_hang", actual_native_exited=False,
              active=_active(adapter), sdk_calls=probe.calls, fault_fired=fault.fired,
              constructor_calls=len(probe.models), backend_retained=any(ref() is not None for ref in probe.models))
        # No stream close, counter/flag reset, Event rescue or alternate disposer.
        os._exit(1)
    assert errors == [], errors
    assert _active(adapter) == 0
    selected = fixture["child"](adapter)
    assert selected is None or selected._model is None
    assert all(reference() is None for reference in probe.models)
    if stream is not None:
        stream.close()
    _emit(kind, operation, "confirmed", actual_native_exited=True,
          fault_fired=True, sdk_calls=probe.calls, backend_released=True)
    patch.undo()


def _run(kind, operation):
    script = (
        "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); "
        "m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    )
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, operation],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    receipts = [json.loads(line.removeprefix("TTS_RESERVATION=")) for line in result.stdout.splitlines()
                if line.startswith("TTS_RESERVATION=")]
    assert receipts, f"fixture failed before actual close receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{receipts[-1]}\n{result.stderr}"
    assert receipts[-1]["reason"] == "confirmed", receipts[-1]


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("operation", ["load", "synthesize", "stream"])
def test_tts_reservation_exit_interrupt_does_not_leave_a_producerless_active_lease(kind, operation):
    _run(kind, operation)
