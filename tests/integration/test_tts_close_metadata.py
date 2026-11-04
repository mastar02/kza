"""Legacy TTS close publisher faults, with actual native observer receipts."""

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


def _reserved(adapter):
    values = vars(adapter)
    for name in ("_adapter_close_admission", "_tts_close_admission"):
        if name in values:
            return values[name] is not None
    return values["_tts_close_started"]


class _ElectionExitFault:
    def __init__(self, condition, adapter):
        self.condition, self.adapter = condition, adapter
        self.fired = False

    def __enter__(self):
        self.condition.__enter__()
        return self

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if not self.fired and _reserved(self.adapter):
            self.fired = True
            raise KeyboardInterrupt
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)


class _CompletionFault:
    def __init__(self, event):
        self.event, self.fired = event, False

    def set(self):
        if not self.fired:
            self.fired = True
            raise KeyboardInterrupt
        return self.event.set()

    def __getattr__(self, name):
        return getattr(self.event, name)


def _emit(kind, mode, reason, **scalars):
    print("TTS_CLOSE_METADATA=" + json.dumps({"kind": kind, "mode": mode, "reason": reason, **scalars}), flush=True)


def _probe(kind, mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    assert len(probe.models) == 1 and probe.calls == 0
    release_calls = []
    release_name = "_adapter_release_resources" if hasattr(adapter, "_adapter_release_resources") else "_tts_release_resources"
    real_release = getattr(adapter, release_name)

    def dispose():
        release_calls.append(threading.current_thread().name)
        return real_release()

    patch.setattr(adapter, release_name, dispose)
    if mode == "election":
        name = "_adapter_condition" if "_adapter_condition" in vars(adapter) else "_tts_condition"
        fault = _ElectionExitFault(getattr(adapter, name), adapter)
    else:
        name = "_adapter_close_done" if "_adapter_close_done" in vars(adapter) else "_tts_close_done"
        fault = _CompletionFault(getattr(adapter, name))
    patch.setattr(adapter, name, fault)

    def observer(errors):
        try:
            adapter.close()
        except BaseException as error:
            errors.append((type(error).__name__, str(error)))

    first_errors, follower_errors = [], []
    first = threading.Thread(target=observer, args=(first_errors,), daemon=True, name="tts-elected-closer")
    first.start()
    first._handle.join(1)
    if not first._handle.is_done():
        _emit(kind, mode, "elected_observer_hang", first_native_exited=False)
        os._exit(1)
    assert fault.fired and first_errors, "actual close metadata fault did not become observable"
    assert len(release_calls) == (0 if mode == "election" else 1)
    follower = threading.Thread(target=observer, args=(follower_errors,), daemon=True, name="tts-following-closer")
    follower.start()
    follower._handle.join(1)
    if not follower._handle.is_done():
        _emit(kind, mode, "close_publisher_lost", first_native_exited=True, follower_native_exited=False,
              actual_dispose_calls=len(release_calls), backend_retained=probe.models[0]() is not None,
              first_error_type=first_errors[0][0], fault_fired=True)
        os._exit(1)  # No event/flag rescue, second disposer or forced model release.
    expected = [("RuntimeError", "tts_close_failed")]
    assert first_errors == follower_errors == expected, (first_errors, follower_errors)
    assert len(release_calls) == (0 if mode == "election" else 1)
    selected = fixture["child"](adapter)
    assert (selected._model is not None) == (mode == "election")
    assert (probe.models[0]() is not None) == (mode == "election")
    _emit(kind, mode, "confirmed_failed_retirement", first_native_exited=True, follower_native_exited=True,
          actual_dispose_calls=len(release_calls), backend_retained=mode == "election")
    # Uncertain pre-disposal ownership is deliberately retained until OS exit.
    patch.undo()


def _run(kind, mode):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_CLOSE_METADATA=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_CLOSE_METADATA=")]
    assert rows, f"fixture failed before native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed_failed_retirement", rows[-1]


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("mode", ["election", "completion"])
def test_tts_close_metadata_interrupt_resolves_followers_without_disposal_retry(kind, mode):
    _run(kind, mode)
