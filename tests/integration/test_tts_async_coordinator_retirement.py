"""Original and owned-child callers keep their meaning across async retirement."""

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


async def _probe(kind):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    adapter = fixture["adapter"](kind)
    selected = fixture["child"](adapter)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-reused-consumer")
    loop.set_default_executor(executor)
    await asyncio.to_thread(adapter.load)
    stream = adapter.synthesize_stream(fixture["CANARY"])
    chunk = await asyncio.to_thread(next, stream)
    np.testing.assert_array_equal(chunk, np.arange(4096, dtype=np.float32))
    assert probe.calls == probe.iterator_exits == 1
    workers = tuple(executor._threads)
    assert len(workers) == 1
    worker = workers[0]
    assert worker is not threading.current_thread()
    assert any(lease.active and lease.thread is worker for lease in adapter._adapter_leases)
    assert selected._tts_active == 1 and selected._model is not None
    progress, actually_waiting, failed_guard = (threading.Event() for _ in range(3))
    wait_threads, guard_threads, stream_errors = [], [], []
    original_condition = adapter._adapter_condition
    original_guard = adapter._adapter_require_close_caller

    class ObservedCondition:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            return original_condition.__exit__(*args)

        def __getattr__(self, name):
            return getattr(original_condition, name)

        def wait(self, *args, **kwargs):
            wait_threads.append(threading.current_thread())
            actually_waiting.set()
            progress.set()
            return original_condition.wait(*args, **kwargs)

    def observed_guard():
        try:
            return original_guard()
        except RuntimeError as error:
            if str(error) == "tts_close_from_worker":
                guard_threads.append(threading.current_thread())
                failed_guard.set()
                progress.set()
            raise

    patch.setattr(adapter, "_adapter_condition", ObservedCondition())
    patch.setattr(adapter, "_adapter_require_close_caller", observed_guard)

    async def observe_close():
        try:
            await adapter.aclose()  # The original loop caller passes the guard.
        except BaseException as error:
            return type(error).__name__, str(error)
        return None

    closing = asyncio.create_task(observe_close())

    async def observed_progress():
        while not progress.is_set():
            await asyncio.sleep(.001)  # Never queue an Event.wait in the sole worker.

    await asyncio.wait_for(observed_progress(), 3)
    false_guard = failed_guard.is_set()
    if false_guard:
        assert guard_threads == [worker]
        assert not actually_waiting.is_set()
    else:
        assert actually_waiting.is_set() and all(thread is worker for thread in wait_threads)
        assert not closing.done()
    assert selected._model is not None and selected._tts_active == 1

    def close_stream():
        try:
            stream.close()  # An independent real consumer retires the real lease.
        except BaseException as error:
            stream_errors.append((type(error).__name__, str(error)))

    consumer = threading.Thread(target=close_stream, daemon=True, name="tts-independent-stream-closer")
    consumer.start()
    consumer._handle.join(3)
    assert consumer is not worker and consumer._handle.is_done()
    assert stream_errors == []
    close_result = await asyncio.wait_for(asyncio.shield(closing), 3)
    if false_guard:
        assert close_result == ("RuntimeError", "tts_close_from_worker")
        # Ordinary public cleanup after the independently closed stream does
        # not erase the recorded failed coordinator. No flags/events are reset.
        cleanup_errors = []

        def cleanup():
            try:
                adapter.close()
            except BaseException as error:
                cleanup_errors.append((type(error).__name__, str(error)))

        cleanup_thread = threading.Thread(target=cleanup, daemon=True, name="tts-red-external-cleanup")
        cleanup_thread.start()
        cleanup_thread._handle.join(3)
        assert cleanup_thread._handle.is_done() and cleanup_errors == []
    else:
        assert close_result is None
    assert selected._model is None and selected._tts_active == 0
    assert all(reference() is None for reference in probe.models)
    np.testing.assert_array_equal(chunk, np.arange(4096, dtype=np.float32))
    executor.shutdown(wait=True)
    assert all(thread._handle.is_done() for thread in workers)
    print("TTS_COORDINATOR_REUSE=" + json.dumps({"kind": kind,
        "reason": "authorized_coordinator_rejected_as_consumer" if false_guard else "confirmed",
        "single_actual_worker_reused": True, "real_condition_wait": actually_waiting.is_set(),
        "failed_public_guard": false_guard, "stream_closer_native_exited": True,
        "pool_native_workers_exited": True, "actual_model_retired": True,
        "caller_pcm_unchanged": True}), flush=True)
    patch.undo()
    return 1 if false_guard else 0


@pytest.mark.parametrize("kind", ["kokoro", "dual"])
def test_authorized_async_close_reuses_suspended_consumer_worker_without_false_guard(kind):
    script = "import asyncio,os,runpy,sys; m=runpy.run_path(sys.argv[1]); code=asyncio.run(m['_probe'](sys.argv[2])); os._exit(code)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), kind],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_COORDINATOR_REUSE=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_COORDINATOR_REUSE=")]
    assert rows, f"fixture failed before coordinator receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]


async def _owned_probe(mode):
    fixture = runpy.run_path(str(_root() / "tests/integration/test_tts_model_retirement.py"))
    np = fixture["np"]
    patch = pytest.MonkeyPatch()
    probe = fixture["PipelineProbe"](patch)
    parent = fixture["adapter"]("dual")
    selected = fixture["child"](parent)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-owned-child-consumer")
    loop.set_default_executor(executor)
    await asyncio.to_thread(parent.load)
    stream = selected.synthesize_stream(fixture["CANARY"])
    chunk = await asyncio.to_thread(next, stream)
    workers = tuple(executor._threads)
    assert len(workers) == 1
    worker = workers[0]
    assert parent._tts_active == 0 and selected._tts_active == 1
    assert any(lease.active and lease.thread is worker for lease in selected._adapter_leases)
    assert probe.calls == probe.iterator_exits == 1
    np.testing.assert_array_equal(chunk, np.arange(4096, dtype=np.float32))
    progress, waited, guard_failed = (threading.Event() for _ in range(3))
    wait_threads, guard_threads = [], []
    original_condition, original_guard = selected._adapter_condition, selected._adapter_require_close_caller

    class ObservedCondition:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, *args):
            return original_condition.__exit__(*args)

        def __getattr__(self, name):
            return getattr(original_condition, name)

        def wait(self, *args, **kwargs):
            wait_threads.append(threading.current_thread())
            waited.set()
            progress.set()
            return original_condition.wait(*args, **kwargs)

    def observed_guard():
        try:
            return original_guard()
        except RuntimeError as error:
            if str(error) == "tts_close_from_worker":
                guard_threads.append(threading.current_thread())
                guard_failed.set()
                progress.set()
            raise

    patch.setattr(selected, "_adapter_condition", ObservedCondition())
    patch.setattr(selected, "_adapter_require_close_caller", observed_guard)

    async def observe_async_close():
        try:
            await parent.aclose()
        except BaseException as error:
            return type(error).__name__, str(error)
        return None

    def observe_public_close():
        assert threading.current_thread() is worker
        try:
            parent.close()
        except BaseException as error:
            return type(error).__name__, str(error)
        return None

    if mode == "async-parent":
        closing = asyncio.create_task(observe_async_close())

        async def observed_progress():
            while not progress.is_set():
                await asyncio.sleep(.001)

        await asyncio.wait_for(observed_progress(), 3)
        false_guard = guard_failed.is_set()
        if false_guard:
            assert guard_threads == [worker] and not waited.is_set()
        else:
            assert waited.is_set() and all(thread is worker for thread in wait_threads)
            assert not closing.done()
        correct = not false_guard
        assert selected._tts_active == 1 and selected._model is not None
        stream.close()  # Direct loop caller is independent of the blocked pool.
        close_result = await asyncio.wait_for(asyncio.shield(closing), 3)
        assert close_result == (None if correct else ("RuntimeError", "tts_close_failed"))
    else:
        close_result = await asyncio.to_thread(observe_public_close)
        correct = (close_result == ("RuntimeError", "tts_close_from_worker")
                   and not parent._tts_sealed and not selected._tts_sealed
                   and parent._adapter_close_admission is None)
        assert selected._tts_active == 1 and selected._model is not None
        stream.close()
        if correct:
            errors = []

            def legitimate_close():
                try:
                    parent.close()
                except BaseException as error:
                    errors.append((type(error).__name__, str(error)))

            closer = threading.Thread(target=legitimate_close, daemon=True, name="tts-owned-child-external-close")
            closer.start()
            closer._handle.join(3)
            assert closer._handle.is_done() and errors == []
    assert selected._tts_active == 0
    if correct:
        assert selected._model is None and all(reference() is None for reference in probe.models)
    else:
        # The failed elected parent attempt is terminal. Do not retry it or
        # clear its token to make the retained model disappear in the fixture.
        admission = parent._adapter_close_admission
        assert admission is not None and admission.finished
        assert admission.failure == "tts_close_failed"
        assert selected._model is not None and all(reference() is not None for reference in probe.models)
    np.testing.assert_array_equal(chunk, np.arange(4096, dtype=np.float32))
    executor.shutdown(wait=True)
    assert all(thread._handle.is_done() for thread in workers)
    print("TTS_OWNED_COORDINATOR=" + json.dumps({"mode": mode,
        "reason": "confirmed" if correct else "owned_child_caller_confused_or_parent_sealed",
        "real_child_condition_wait": waited.is_set(), "actual_pool_workers_exited": True,
        "parent_sealed": parent._tts_sealed, "model_retired": selected._model is None,
        "terminal_failed_ownership_retained": not correct, "caller_pcm_unchanged": True}), flush=True)
    patch.undo()
    return 0 if correct else 1


@pytest.mark.parametrize("mode", ["async-parent", "public-parent-consumer"])
def test_owned_child_stream_preserves_original_parent_caller_authorization(mode):
    script = "import asyncio,os,runpy,sys; m=runpy.run_path(sys.argv[1]); code=asyncio.run(m['_owned_probe'](sys.argv[2])); os._exit(code)"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
                            cwd=_root(), capture_output=True, text=True, timeout=15)
    rows = [json.loads(line.removeprefix("TTS_OWNED_COORDINATOR=")) for line in result.stdout.splitlines()
            if line.startswith("TTS_OWNED_COORDINATOR=")]
    assert rows, f"fixture failed before owned-child receipt: {result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{rows[-1]}\n{result.stderr}"
    assert rows[-1]["reason"] == "confirmed", rows[-1]
