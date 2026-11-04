"""The async bridge preserves an explicit public wait_closed override."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
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


async def _probe(scope):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"]("dual")
    selected = fixture["child"](adapter)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-public-wait-override")
    asyncio.get_running_loop().set_default_executor(executor)
    await asyncio.to_thread(adapter.load)
    assert selected._model is not None and selected._tts_active == adapter._tts_active == 0
    original_wait_closed = adapter.wait_closed
    calls = []

    if scope == "class":
        def class_wait_closed(owner):
            assert owner is adapter
            calls.append(threading.current_thread())
            return original_wait_closed()

        patch.setattr(type(adapter), "wait_closed", class_wait_closed)
        assert "wait_closed" not in vars(adapter)
    else:
        def instance_wait_closed():
            calls.append(threading.current_thread())
            return original_wait_closed()

        patch.setattr(adapter, "wait_closed", instance_wait_closed)
        assert "wait_closed" in vars(adapter)
    await adapter.aclose()
    assert adapter._adapter_closing.done() and adapter._adapter_closing.result() is None
    assert selected._model is None and all(reference() is None for reference in probe.models)
    assert adapter._tts_sealed and selected._tts_sealed
    assert adapter._owned_tts_engines == ()
    assert adapter._adapter_close_admission.finished and adapter._adapter_close_admission.failure is None
    assert selected._adapter_close_admission.finished and selected._adapter_close_admission.failure is None
    assert probe.calls == probe.iterator_exits == 0
    workers = tuple(executor._threads)
    executor.shutdown(wait=True)
    assert len(workers) == 1 and all(worker._handle.is_done() for worker in workers)
    correct = calls == [workers[0]]
    print("TTS_ASYNC_WAIT_OVERRIDE=" + json.dumps({"scope": scope,
        "reason": "confirmed" if correct else "public_wait_closed_override_skipped",
        "public_override_calls": len(calls), "actual_native_workers_exited": True,
        "actual_model_retired": True}), flush=True)
    patch.undo()
    return 0 if correct else 1


@pytest.mark.parametrize("scope", ["class", "instance"])
def test_async_close_preserves_public_wait_closed_override_and_real_retirement(scope):
    script = "import asyncio,os,runpy,sys; m=runpy.run_path(sys.argv[1]); code=asyncio.run(m['_probe'](sys.argv[2])); os._exit(code)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), scope],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_ASYNC_WAIT_OVERRIDE=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_ASYNC_WAIT_OVERRIDE=")]
    assert rows, f"fixture failed before public-wait receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
