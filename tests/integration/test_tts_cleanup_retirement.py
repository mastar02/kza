"""Actual TTS cleanup-acquire/lost-notify probes, isolated without flag rescue."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


def _root():
    return next(p for p in Path(__file__).resolve().parents
                if (p / "tests/integration/test_tts_model_retirement.py").is_file())


def _active(adapter):
    for name in ("_adapter_leases", "_tts_leases"):
        if name in vars(adapter):
            return sum(lease.active for lease in vars(adapter)[name])
    return vars(adapter)["_tts_active"]


class _ConditionFault:
    def __init__(self, condition, adapter, mode):
        self.condition, self.adapter, self.mode = condition, adapter, mode
        self.armed = self.fired = False
        self.wait_entered = threading.Event()

    def __enter__(self):
        if self.mode == "acquire" and self.armed and not self.fired:
            self.fired = True
            raise KeyboardInterrupt
        self.condition.__enter__()
        return self

    def __exit__(self, *args):
        return self.condition.__exit__(*args)

    def wait(self, timeout=None):
        self.wait_entered.set()
        return self.condition.wait(timeout)

    def notify_all(self):
        if self.mode == "notify" and not self.fired and _active(self.adapter) == 0:
            self.fired = True
            raise KeyboardInterrupt
        return self.condition.notify_all()

    def __getattr__(self, name):
        return getattr(self.condition, name)


class _RestoreArmsAcquire:
    def __init__(self, local, fault):
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "fault", fault)
        object.__setattr__(self, "stores", 0)

    def __getattr__(self, name):
        return getattr(self.local, name)

    def __setattr__(self, name, value):
        setattr(self.local, name, value)
        if name == "lease":
            object.__setattr__(self, "stores", self.stores + 1)
            if self.stores == 2:
                self.fault.armed = True


def _probe(kind, mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    condition_name = "_adapter_condition" if "_adapter_condition" in vars(adapter) else "_tts_condition"
    fault = _ConditionFault(getattr(adapter, condition_name), adapter, mode)
    patch.setattr(adapter, condition_name, fault)
    if mode == "acquire":
        local_name = "_adapter_local" if "_adapter_local" in vars(adapter) else "_tts_local"
        patch.setattr(adapter, local_name, _RestoreArmsAcquire(getattr(adapter, local_name), fault))
    else:
        probe.infer_release.clear()
    caller_errors, close_errors = [], []

    def synthesize():
        try:
            adapter.synthesize(fixture["CANARY"])
        except BaseException as error:
            caller_errors.append((type(error).__name__, str(error)))

    def close():
        try:
            adapter.close()
        except BaseException as error:
            close_errors.append((type(error).__name__, str(error)))

    caller = threading.Thread(target=synthesize, daemon=True, name="tts-actual-api-caller")
    caller.start()
    if mode == "notify":
        assert probe.infer_entered.wait(2), "actual SDK inference did not enter"
        closer = threading.Thread(target=close, daemon=True, name="tts-actual-waiting-closer")
        closer.start()
        assert fault.wait_entered.wait(2), "actual closer did not wait on admitted synthesis"
        probe.infer_release.set()
    else:
        caller._handle.join(2)
        assert caller._handle.is_done(), "actual public synthesis caller failed to exit"
        closer = threading.Thread(target=close, daemon=True, name="tts-actual-external-closer")
        closer.start()
    caller._handle.join(2)
    assert caller._handle.is_done() and fault.fired
    assert caller_errors == [("KeyboardInterrupt", "")], caller_errors
    assert probe.calls == probe.iterator_exits == 1
    closer._handle.join(1)
    if not closer._handle.is_done():
        print("TTS_CLEANUP=" + json.dumps({"kind": kind, "mode": mode,
            "reason": "cleanup_progress_lost", "caller_native_exited": True, "closer_native_exited": False,
            "active": _active(adapter), "sdk_calls": probe.calls, "sdk_iterator_exits": probe.iterator_exits,
            "backend_retained": probe.models[0]() is not None, "fault_fired": True}), flush=True)
        os._exit(1)
    assert close_errors == [], close_errors
    assert _active(adapter) == 0 and all(reference() is None for reference in probe.models)
    print("TTS_CLEANUP=" + json.dumps({"kind": kind, "mode": mode, "reason": "confirmed",
        "caller_native_exited": True, "closer_native_exited": True, "fault_fired": True}), flush=True)
    patch.undo()


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("mode", ["acquire", "notify"])
def test_tts_logical_cleanup_precedes_fallible_acquire_and_does_not_require_notify(kind, mode):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_CLEANUP=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_CLEANUP=")]
    assert rows, f"fixture failed before cleanup/native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
