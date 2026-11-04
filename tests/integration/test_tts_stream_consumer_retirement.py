"""Real Kokoro/Dual suspended-stream consumer self-wait baseline probes."""

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
    for name in ("_tts_leases", "_adapter_leases"):
        if name in vars(adapter):
            return sum(lease.active for lease in vars(adapter)[name])
    return vars(adapter)["_tts_active"]


def _emit(kind, mode, reason, **scalars):
    print("TTS_SELF_WAIT=" + json.dumps({"kind": kind, "mode": mode, "reason": reason, **scalars}), flush=True)


def _probe(kind, mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    stream = adapter.synthesize_stream(fixture["CANARY"])
    pulls, errors = [], []
    close_entered = threading.Event()

    def pull(ordinal):
        value = next(stream)
        assert value.shape == (4096,) and value.dtype == np.float32
        np.testing.assert_array_equal(value, np.arange(ordinal * 4096, (ordinal + 1) * 4096, dtype=np.float32))
        pulls.append(ordinal)

    first = None
    if mode == "migrate":
        first_errors = []

        def first_pull():
            try:
                pull(0)
            except BaseException as error:
                first_errors.append((type(error).__name__, str(error)))

        first = threading.Thread(target=first_pull, daemon=True, name="tts-consumer-A")
        first.start()
        first._handle.join(2)
        assert first._handle.is_done() and first_errors == []
        assert probe.calls == probe.iterator_exits == 1

    def consume_then_close():
        try:
            pull(1 if mode == "migrate" else 0)
            assert probe.calls == probe.iterator_exits == 1
            close_entered.set()
            try:
                adapter.close()
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
            else:
                errors.append(("returned", ""))
        except BaseException as error:
            errors.append(("fixture_failure", type(error).__name__))

    consumer = threading.Thread(target=consume_then_close, daemon=True,
        name="tts-consumer-B" if mode == "migrate" else "tts-consumer-A")
    assert first is None or first is not consumer
    consumer.start()
    assert close_entered.wait(2), f"consumer failed before actual close: {errors}"
    consumer._handle.join(1)
    if not consumer._handle.is_done():
        _emit(kind, mode, "suspended_consumer_self_wait", actual_native_exited=False,
              active=_active(adapter), sdk_calls=probe.calls, sdk_iterator_exits=probe.iterator_exits,
              pcm_slice_ordinals=pulls, backend_retained=probe.models[0]() is not None,
              latest_consumer=consumer.name, first_consumer_native_exited=first is None or first._handle.is_done())
        os._exit(1)  # No stream, counter, flag or notification rescue.
    assert errors == [("RuntimeError", "tts_close_from_worker")], errors
    selected = fixture["child"](adapter)
    assert not adapter._tts_sealed and not selected._tts_sealed
    assert _active(adapter) == 1 and selected._model is probe.models[0]()
    stream.close()
    assert _active(adapter) == _active(selected) == 0
    final_errors = []

    def external_close():
        try:
            adapter.close()
        except BaseException as error:
            final_errors.append((type(error).__name__, str(error)))

    closer = threading.Thread(target=external_close, daemon=True, name="tts-independent-closer")
    closer.start()
    closer._handle.join(2)
    if not closer._handle.is_done():
        _emit(kind, mode, "external_closer_hang", actual_native_exited=False)
        os._exit(1)
    assert final_errors == []
    assert selected._model is None and all(ref() is None for ref in probe.models)
    assert probe.calls == probe.iterator_exits == 1
    _emit(kind, mode, "confirmed", actual_native_exited=True, backend_released=True,
          sdk_calls=1, sdk_iterator_exits=1, pcm_slice_ordinals=pulls)
    patch.undo()


def _run(kind, mode):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_SELF_WAIT=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_SELF_WAIT=")]
    assert rows, f"fixture failed before native receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("mode", ["same", "migrate"])
def test_latest_real_tts_pcm_consumer_cannot_wait_on_its_suspended_stream(kind, mode):
    _run(kind, mode)
