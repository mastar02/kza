"""Actual SDK-result / playback-PCM finalization remains inside admission."""

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
    leases = getattr(adapter, "_adapter_leases", None)
    return (sum(lease.active for lease in leases)
            if leases is not None else adapter._tts_active)


def _probe(kind, payload):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    adapter.load()
    selected = fixture["child"](adapter)
    target = selected if payload == "sdk-result" else adapter
    finalizer_entered, finalizer_release, finalizer_exited = (threading.Event() for _ in range(3))
    close_progress, close_waiting, dispose_entered = (threading.Event() for _ in range(3))
    output, caller_errors, close_errors, played = [], [], [], []

    class SDKResult:
        def __init__(self, audio, model):
            self.audio, self.model = audio, model

        def __del__(self):
            finalizer_entered.set()
            finalizer_release.wait()
            self.audio = self.model = None
            finalizer_exited.set()

    class PlaybackPCM(np.ndarray):
        def __del__(self):
            finalizer_entered.set()
            finalizer_release.wait()
            finalizer_exited.set()

    original_generate = probe.generate

    def generate(model, text, voice):
        for item in original_generate(model, text, voice):
            if payload == "sdk-result":
                yield SDKResult(item.audio, model)
            else:
                item.audio = item.audio.view(PlaybackPCM)
                yield item

    patch.setattr(probe, "generate", generate)

    def play(audio, *, samplerate):
        assert samplerate == adapter.sample_rate
        assert audio.dtype == np.float32 and len(audio) == 8192
        assert audio[0] == 0 and audio[-1] == 8191
        played.append(True)  # The playback leaf borrows no PCM after return.

    module = sys.modules[fixture["KokoroTTS"].__module__]
    patch.setattr(module.sd, "play", play)
    patch.setattr(module.sd, "wait", lambda: None)
    condition_name = "_adapter_condition" if hasattr(target, "_adapter_condition") else "_tts_condition"
    original_condition = getattr(target, condition_name)

    class ObservedCondition:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            return original_condition.__exit__(*args)

        def __getattr__(self, name):
            return getattr(original_condition, name)

        def wait(self, *args, **kwargs):
            close_waiting.set()
            close_progress.set()
            return original_condition.wait(*args, **kwargs)

    patch.setattr(target, condition_name, ObservedCondition())
    original_release = target._tts_release_resources

    def release_resources():
        dispose_entered.set()
        close_progress.set()
        return original_release()

    patch.setattr(target, "_tts_release_resources", release_resources)

    def invoke():
        try:
            if payload == "sdk-result":
                output.append(adapter.synthesize(fixture["CANARY"]))
            else:
                adapter.speak(fixture["CANARY"])
        except BaseException as error:
            caller_errors.append((type(error).__name__, str(error)))

    def close():
        try:
            target.close()
        except BaseException as error:
            close_errors.append((type(error).__name__, str(error)))

    caller = threading.Thread(target=invoke, daemon=True, name="tts-payload-api-caller")
    closer = threading.Thread(target=close, daemon=True, name="tts-payload-model-closer")
    caller.start()
    assert finalizer_entered.wait(3), "actual API payload cleanup was not reached"
    assert probe.calls == probe.iterator_exits == 1
    assert not caller._handle.is_done() and not finalizer_exited.is_set()
    closer.start()
    assert close_progress.wait(3), "closer reached neither actual wait nor actual disposal"
    if dispose_entered.is_set() or _active(target) == 0:
        print("TTS_PAYLOAD_FINALIZER=" + json.dumps({"kind": kind, "payload": payload,
            "reason": "model_disposal_admitted_before_api_payload_cleanup", "active": _active(target),
            "actual_disposer_entered": dispose_entered.is_set(), "actual_closer_waiting": close_waiting.is_set(),
            "actual_caller_native_exited": caller._handle.is_done(), "actual_finalizer_exited": finalizer_exited.is_set()}), flush=True)
        os._exit(1)  # Do not release the retained payload to manufacture success.
    assert close_waiting.is_set() and _active(target) >= 1
    assert selected._model is not None and not closer._handle.is_done()
    finalizer_release.set()  # Release only the test's real SDK/PCM cleanup gate.
    caller._handle.join(3)
    closer._handle.join(3)
    assert caller._handle.is_done() and closer._handle.is_done()
    assert finalizer_exited.is_set() and caller_errors == close_errors == []
    assert selected._model is None and all(reference() is None for reference in probe.models)
    if payload == "sdk-result":
        np.testing.assert_array_equal(output[0][0], np.arange(8192, dtype=np.float32))
        assert output[0][1] >= 0
        if kind == "dual":
            assert output[0][2] == "kokoro"
    else:
        assert played == [True] and output == []
    adapter.close()
    print("TTS_PAYLOAD_FINALIZER=" + json.dumps({"kind": kind, "payload": payload,
        "reason": "confirmed", "actual_api_payload_cleanup": True,
        "actual_caller_native_exited": True, "actual_closer_native_exited": True}), flush=True)
    patch.undo()


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
@pytest.mark.parametrize("payload", ["sdk-result", "playback-pcm"])
def test_actual_tts_model_waits_for_in_api_payload_finalizer(kind, payload):
    script = "import os,runpy,sys; m=runpy.run_path(sys.argv[1]); m['_probe'](sys.argv[2],sys.argv[3]); os._exit(0)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind, payload],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_PAYLOAD_FINALIZER=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_PAYLOAD_FINALIZER=")]
    assert rows, f"fixture failed before payload receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
