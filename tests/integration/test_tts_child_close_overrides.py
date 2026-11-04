"""Dual retirement preserves reviewed children with public close overrides."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


def _root():
    return next(p for p in Path(__file__).resolve().parents
                if (p / "tests/integration/test_tts_model_retirement.py").is_file())


def _probe(scope):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    parent = fixture["adapter"]("dual")
    parent.load()
    selected = fixture["child"](parent)
    assert selected._model is not None and selected._tts_active == parent._tts_active == 0
    assert len(probe.models) == 1 and probe.models[0]() is selected._model
    original_close = selected.close
    calls, errors = [], []

    if scope == "class":
        def public_close(child):
            assert child is selected
            calls.append(threading.current_thread())
            return original_close()

        patch.setattr(type(selected), "close", public_close)
        assert "close" not in vars(selected)
    else:
        def public_close():
            calls.append(threading.current_thread())
            return original_close()

        patch.setattr(selected, "close", public_close)
        assert "close" in vars(selected)

    def close_parent():
        try:
            parent.close()
        except BaseException as error:
            errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=close_parent, daemon=True, name="tts-owned-public-override-closer")
    closer.start()
    closer._handle.join(3)
    assert closer._handle.is_done(), "actual parent closer did not retire"
    assert errors == [] and calls == [closer]
    assert selected._model is None and all(reference() is None for reference in probe.models)
    assert parent._tts_sealed and selected._tts_sealed
    assert parent._owned_tts_engines == ()
    assert parent._adapter_close_admission.finished and parent._adapter_close_admission.failure is None
    assert selected._adapter_close_admission.finished and selected._adapter_close_admission.failure is None
    assert probe.calls == probe.iterator_exits == 0
    print("TTS_CHILD_CLOSE_OVERRIDE=" + json.dumps({"scope": scope,
        "reason": "confirmed", "public_override_calls": len(calls),
        "actual_native_closer_exited": True, "actual_model_retired": True}), flush=True)
    patch.undo()


@pytest.mark.parametrize("scope", ["class", "instance"])
def test_owned_child_public_close_override_runs_once_and_retires_real_model(scope):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), scope],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_CHILD_CLOSE_OVERRIDE=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_CHILD_CLOSE_OVERRIDE=")]
    assert rows, f"fixture failed before override receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
