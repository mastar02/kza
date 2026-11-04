"""Actual admission/TLS and nested parent proofs; baseline may already pass."""

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


def _local_name(adapter):
    return "_adapter_local" if "_adapter_local" in vars(adapter) else "_tts_local"


def _active(adapter):
    for name in ("_adapter_leases", "_tts_leases"):
        if name in vars(adapter):
            return sum(lease.active for lease in vars(adapter)[name])
    return vars(adapter)["_tts_active"]


class _TLSFault:
    def __init__(self, local, phase):
        for name, value in (("local", local), ("phase", phase), ("fired", False), ("armed", False)):
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return getattr(self.local, name)

    def arm(self):
        object.__setattr__(self, "armed", True)

    def __setattr__(self, name, value):
        if name == "lease" and not self.fired:
            selected = ((self.phase == "publication" and value is not None)
                        or (self.phase == "restoration" and value is None)
                        or (self.phase == "nested" and self.armed and value is not None))
            if selected:
                object.__setattr__(self, "fired", True)
                raise KeyboardInterrupt
        setattr(self.local, name, value)


def _external_close(adapter):
    errors = []

    def close():
        try:
            adapter.close()
        except BaseException as error:
            errors.append((type(error).__name__, str(error)))

    thread = threading.Thread(target=close, daemon=True, name="tts-tls-external-closer")
    thread.start()
    thread._handle.join(1)
    return thread._handle.is_done(), errors


def _probe(kind, scope, phase):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    name = _local_name(adapter)
    fault = _TLSFault(getattr(adapter, name), phase)
    patch.setattr(adapter, name, fault)
    interrupted, stream = False, None
    try:
        if scope == "regular":
            adapter.synthesize(fixture["CANARY"])
        else:
            stream = adapter.synthesize_stream(fixture["CANARY"])
            next(stream)
    except KeyboardInterrupt:
        interrupted = True
    assert interrupted and fault.fired
    assert adapter._tts_current() is None and _active(adapter) == 0
    if stream is not None:
        stream.close()
    exited, errors = _external_close(adapter)
    if not exited:
        print("TTS_TLS=" + json.dumps({"reason": "tls_cleanup_hang", "kind": kind, "scope": scope, "phase": phase}), flush=True)
        os._exit(1)
    assert errors == [], errors
    assert all(reference() is None for reference in probe.models)
    expected_calls = int(phase == "restoration" and (scope == "regular" or kind == "kokoro"))
    # Dual restores its first parent scope after constructing the unstarted
    # delegated generator, before its first native synthesis pull.
    assert probe.calls == probe.iterator_exits == expected_calls
    print("TTS_TLS=" + json.dumps({"reason": "confirmed", "kind": kind, "scope": scope,
          "phase": phase, "actual_close_native_exited": True}), flush=True)
    patch.undo()


def _nested():
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    parent = fixture["adapter"]("dual")
    name = _local_name(parent)
    fault = _TLSFault(getattr(parent, name), "nested")
    patch.setattr(parent, name, fault)
    selected = fixture["child"](parent)
    original_synthesize, original_construct = selected.synthesize, sys.modules["kokoro"].KPipeline
    leases, controls = [], []

    def construct(**options):
        model = original_construct(**options)
        fault.arm()
        return model

    def synthesize(text):
        lease = parent._tts_current()
        assert lease is not None and lease.active and _active(parent) == 1
        leases.append(lease)
        try:
            # This deliberate delegate hook exercises real nested parent.load;
            # ordinary Dual.synthesize directly asks its selected child instead.
            parent.load()
        except KeyboardInterrupt:
            controls.append(True)
            assert parent._tts_current() is lease and lease.active and _active(parent) == 1
            with pytest.raises(RuntimeError, match="^tts_close_from_worker$"):
                parent.close()
            parent.request_close()
            assert not fixture["child"](parent)._tts_sealed
        return original_synthesize(text)

    patch.setattr(sys.modules["kokoro"], "KPipeline", construct)
    patch.setattr(selected, "synthesize", synthesize)
    audio, elapsed, engine = parent.synthesize(fixture["CANARY"])
    fixture["np"].testing.assert_array_equal(audio, fixture["np"].arange(8192, dtype=fixture["np"].float32))
    assert elapsed >= 0 and engine == "kokoro" and controls == [True] and fault.fired
    assert len(leases) == 1 and not leases[0].active and _active(parent) == 0
    exited, errors = _external_close(parent)
    if not exited:
        print("TTS_TLS=" + json.dumps({"reason": "nested_parent_hang"}), flush=True)
        os._exit(1)
    assert errors == [] and all(reference() is None for reference in probe.models)
    assert probe.calls == probe.iterator_exits == 1
    print("TTS_TLS=" + json.dumps({"reason": "confirmed", "nested_parent": True,
          "actual_close_native_exited": True}), flush=True)
    patch.undo()


def _run(function, *args):
    script = "import json,os,runpy,sys; m=runpy.run_path(sys.argv[1]); m[sys.argv[2]](*json.loads(sys.argv[3])); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), function, json.dumps(args)],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_TLS=")) for line in result.stdout.splitlines() if line.startswith("TTS_TLS=")]
    assert rows, f"fixture failed before TLS/native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("scope", ["regular", "stream"])
@pytest.mark.parametrize("phase", ["publication", "restoration"])
def test_actual_tts_tls_interrupt_keeps_cleanup_and_control_observable(kind, scope, phase):
    _run("_probe", kind, scope, phase)


def test_nested_dual_load_restore_interrupt_keeps_parent_identity_live_until_outer_exit():
    _run("_nested")
