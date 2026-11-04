"""Actual wrapper raw-source ownership, with a real synthetic SDK actor."""

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


class _IterConstructionFailure(RuntimeError):
    pass


def _probe(surface, mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"]("kokoro" if surface == "kokoro-synth" else "dual")
    adapter.load()
    selected = fixture["child"](adapter)
    actors, source_closes, iterator_closes = [], [], []

    class Iterator:
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.inner)

        def close(self):
            iterator_closes.append(True)
            self.inner.close()
            self.inner = None

    class Source:
        def __init__(self, model, inner):
            self.model, self.inner = model, inner
            self.private_pcm = np.zeros(32, dtype=np.float32)
            self.release = threading.Event()
            self.worker = threading.Thread(target=self.retain_sdk_resource, daemon=True, name="synthetic-raw-sdk-owner")
            actors.append(self.worker)
            self.worker.start()

        def retain_sdk_resource(self):
            # Bound target retains actual synthetic SDK ownership until the
            # PROVIDED iterable.close protocol retires this actor.
            self.release.wait()

        def __iter__(self):
            if mode == "iter-fails":
                raise _IterConstructionFailure("synthetic raw __iter__ failure")
            return Iterator(self.inner)

        def close(self):
            source_closes.append(True)
            self.inner.close()
            self.release.set()
            self.worker._handle.join(2)
            assert self.worker._handle.is_done()
            self.model = self.inner = self.private_pcm = None

    if surface == "kokoro-synth":
        actual_generate = probe.generate

        def generate(model, text, voice):
            return Source(model, actual_generate(model, text, voice))

        patch.setattr(probe, "generate", generate)
    else:
        actual_stream = selected.synthesize_stream

        def stream(text):
            return Source(selected._model, actual_stream(text))

        patch.setattr(selected, "synthesize_stream", stream)
    outer = None
    retained = None
    try:
        if surface == "kokoro-synth":
            audio, elapsed = adapter.synthesize(fixture["CANARY"])
            np.testing.assert_array_equal(audio, np.arange(8192, dtype=np.float32))
            assert elapsed >= 0
        else:
            outer = adapter.synthesize_stream(fixture["CANARY"])
            np.testing.assert_array_equal(next(outer), np.arange(4096, dtype=np.float32))
            outer.close()
    except BaseException as error:
        retained = error
    if mode == "iter-fails":
        assert type(retained) is _IterConstructionFailure, type(retained).__name__
    else:
        assert retained is None, type(retained).__name__
    if source_closes != [True] or any(not worker._handle.is_done() for worker in actors):
        print("TTS_RAW_SOURCE=" + json.dumps({"surface": surface, "mode": mode,
            "reason": "raw_sdk_source_not_retired", "source_close_calls": len(source_closes),
            "iterator_close_calls": len(iterator_closes), "actual_sdk_native_exited": [worker._handle.is_done() for worker in actors]}), flush=True)
        os._exit(1)  # No forced source close, actor release or metadata rescue.
    assert iterator_closes == ([] if mode == "iter-fails" else [True])
    close_errors = []

    def close():
        try:
            adapter.close()
        except BaseException as error:
            close_errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=close, daemon=True, name="tts-raw-external-closer")
    closer.start()
    closer._handle.join(2)
    if not closer._handle.is_done():
        print("TTS_RAW_SOURCE=" + json.dumps({"reason": "external_closer_hang"}), flush=True)
        os._exit(1)
    assert close_errors == [] and selected._model is None
    assert all(reference() is None for reference in probe.models)
    assert probe.calls == probe.iterator_exits == (0 if mode == "iter-fails" else 1)
    print("TTS_RAW_SOURCE=" + json.dumps({"surface": surface, "mode": mode,
        "reason": "confirmed", "source_close_calls": 1,
        "iterator_close_calls": len(iterator_closes), "actual_sdk_native_exited": True,
        "actual_close_native_exited": True}), flush=True)
    patch.undo()


@pytest.mark.parametrize("surface", ["kokoro-synth", "dual-stream"])
@pytest.mark.parametrize("mode", ["iter-fails", "distinct"])
def test_actual_tts_retains_raw_sdk_source_before_iter_and_retires_distinct_handles(surface, mode):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), surface, mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_RAW_SOURCE=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_RAW_SOURCE=")]
    assert rows, f"fixture failed before raw-source receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
