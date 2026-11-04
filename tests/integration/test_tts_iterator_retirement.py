"""Exhausted nonclosable iterables and truthful failed delegated retirement."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace
import weakref

import pytest


def _root():
    return next(p for p in Path(__file__).resolve().parents
                if (p / "tests/integration/test_tts_model_retirement.py").is_file())


def _probe(kind, mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    selected = fixture["child"](adapter)
    source_refs, close_calls, leaf_calls = [], [], []
    sdk_error = RuntimeError("tts_backend_close_unsupported")

    class FailingSource:
        def __init__(self, model):
            self.model, self.index = model, 0
            self.audio = np.arange(8192, dtype=np.float32)
            source_refs.append(weakref.ref(self))

        def __iter__(self):
            return self

        def __next__(self):
            if self.index:
                raise StopIteration
            self.index += 1
            return SimpleNamespace(audio=self.audio)

        def close(self):
            close_calls.append(True)
            raise sdk_error  # The original SDK error belongs to the live caller.

    def generate(model, text, voice):
        leaf_calls.append(True)
        if mode == "failed-close":
            return FailingSource(model)
        if mode == "empty-list":
            return []  # Neither list nor its exhausted iterator offers close().
        return [SimpleNamespace(audio=np.arange(4096, dtype=np.float32)),
                SimpleNamespace(audio=np.arange(4096, 8192, dtype=np.float32))]

    patch.setattr(probe, "generate", generate)
    retained = None
    try:
        result = adapter.synthesize(fixture["CANARY"])
    except BaseException as error:
        retained = error
    assert leaf_calls == [True]
    if mode == "failed-close":
        assert retained is sdk_error
        assert close_calls == [True]
        assert selected._tts_sealed
        assert len(selected._tts_retained_iterators) == 1
        assert source_refs[0]() is not None and selected._model is not None
        try:
            adapter.synthesize(fixture["CANARY"])
        except RuntimeError as error:
            assert str(error) == "tts_closed"
        else:
            raise AssertionError("failed delegated close allowed another synthesis")
        assert leaf_calls == [True]
    else:
        assert retained is None
        expected = np.zeros(0, dtype=np.float32) if mode == "empty-list" else np.arange(8192, dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected)
        assert result[0].dtype == np.float32 and result[1] >= 0
        if kind == "dual":
            assert result[2] == "kokoro"

    errors = []

    def close():
        try:
            (selected if mode == "failed-close" else adapter).close()
        except BaseException as error:
            errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=close, daemon=True, name="tts-iterator-positive-closer")
    closer.start()
    closer._handle.join(3)
    assert closer._handle.is_done(), "actual native closer did not exit"
    if mode == "failed-close":
        assert errors == [("RuntimeError", "tts_iterator_close_failed")]
        assert close_calls == [True]  # Retirement does not retry delegated close.
        assert selected._model is not None and source_refs[0]() is not None
        assert len(selected._tts_retained_iterators) == 1
        assert all(reference() is not None for reference in probe.models)
        # Deliberately failed ownership stays live until this subprocess exits.
    else:
        assert errors == [] and selected._model is None
        assert all(reference() is None for reference in probe.models)
    print("TTS_ITERATOR_POSITIVE=" + json.dumps({"kind": kind, "mode": mode,
        "reason": "confirmed", "native_closer_exited": True,
        "failed_source_retained": mode == "failed-close", "delegated_close_calls": len(close_calls)}), flush=True)
    patch.undo()


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("mode", ["empty-list", "distinct-list-iterator", "failed-close"])
def test_actual_tts_preserves_exhaustion_and_fixed_failed_iterator_label(kind, mode):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_ITERATOR_POSITIVE=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_ITERATOR_POSITIVE=")]
    assert rows, f"fixture failed before iterator receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
