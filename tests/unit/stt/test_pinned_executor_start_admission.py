"""Actual executor start receipts with isolated, non-spawning fault leaves."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _run_case(role: str) -> dict:
    script = (
        "import json,os,runpy,sys,traceback\n"
        "try:\n"
        " m=runpy.run_path(sys.argv[1]);m['_child'](sys.argv[2])\n"
        "except BaseException as error:\n"
        " traceback.print_exc()\n"
        " print('STT_START_CONTRACT='+json.dumps({'outcome':'failed','error_type':type(error).__name__}),flush=True)\n"
        " os._exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', script, str(Path(__file__).resolve()), role],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=12,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
    rows = [json.loads(line.split('=', 1)[1]) for line in result.stdout.splitlines()
            if line.startswith('STT_START_CONTRACT=')]
    assert result.returncode == 0 and len(rows) == 1 and rows[0]['outcome'] == 'confirmed'
    return rows[0]


def _child(role: str) -> None:
    import os
    import weakref
    from src.stt import pinned_executor as implementation

    attempts, registered, calls = [], [], []
    interruption = KeyboardInterrupt()

    def fail_start(thread):
        assert type(thread) is implementation._PinnedWorker
        assert thread.ident is None and thread._handle.ident == 0
        assert thread._handle.is_done() is False and not thread._started.is_set()
        attempts.append(thread)
        raise interruption  # No original Thread.start call and no OS worker.

    implementation._PinnedWorker.start = fail_start
    if role == 'whisper':
        import numpy as np
        from src.stt.whisper_fast import FastWhisperSTT

        model = FastWhisperSTT(model='synthetic', device='cpu')
        sentinel = object()
        model._model = sentinel

        def release_model():
            calls.append('release')

        model._release_model = release_model
        try:
            model.transcribe_with_confidence(np.zeros(64, dtype=np.float32))
        except BaseException as error:
            assert error is interruption
        else:
            raise AssertionError('original start fault did not reach caller')
        executor, worker = model._executor, attempts[0]
        failures = []
        for unused in range(2):
            try:
                model.close()
            except BaseException as error:
                assert isinstance(error, RuntimeError)
                failures.append(str(error))
            else:
                raise AssertionError('uncertain model retirement returned success')
        assert failures == ['pinned_executor_start_unconfirmed', 'whisper_shutdown_failed']
        assert len(attempts) == 1 and calls == []
        assert model._model is sentinel and model._executor is executor
        assert model._close_finished.is_set() and model._close_failed
        assert worker in executor._threads
        assert implementation._pool._threads_queues[worker] is executor._work_queue
        assert worker._handle.is_done() is False
        import asyncio
        from src.application import STTExecutorOwner

        async def observe_owner_failure():
            owner = STTExecutorOwner()
            owner.add_model(model)
            closing = model.begin_close()
            with pytest.raises(RuntimeError, match='^whisper_shutdown_failed$'):
                await closing
            with pytest.raises(RuntimeError, match='^whisper_shutdown_failed$'):
                await owner.aclose()
            assert model.begin_close() is closing
            assert owner._models[id(model)] is model
            assert model._model is sentinel and attempts == [worker]

        asyncio.run(observe_owner_failure())
        result = {'model_reference_retained': True, 'close_failed': True,
                  'graph_model_owner_reports_failure': True}
    else:
        assert role in {'registration', 'unconfirmed'}
        executor = implementation.PinnedThreadPoolExecutor()
        if role == 'registration':
            original_set = weakref.WeakKeyDictionary.__setitem__

            def reject_registration(mapping, key, value):
                if mapping is implementation._pool._threads_queues:
                    assert type(key) is implementation._PinnedWorker
                    registered.append(key)
                    raise interruption  # Before start is invoked, not after.
                return original_set(mapping, key, value)

            weakref.WeakKeyDictionary.__setitem__ = reject_registration
        try:
            executor.submit(lambda: calls.append('work'))
        except BaseException as error:
            assert error is interruption
        else:
            raise AssertionError('selected admission fault was not invoked')
        if role == 'registration':
            assert len(registered) == 1 and attempts == []
            assert registered[0] not in executor._threads
            assert registered[0] not in implementation._pool._threads_queues
            executor.shutdown(wait=True, cancel_futures=True)
            assert calls == [] and registered[0]._handle.is_done() is False
            result = {'start_never_invoked': True, 'shutdown_confirmed_without_actor': True}
        else:
            worker = attempts[0]
            with pytest.raises(RuntimeError):
                executor.submit(lambda: calls.append('successor'))
            with pytest.raises(RuntimeError, match='^pinned_executor_start_unconfirmed$'):
                executor.shutdown(wait=True, cancel_futures=True)
            assert attempts == [worker] and calls == []
            assert worker in executor._threads
            assert implementation._pool._threads_queues[worker] is executor._work_queue
            assert worker._handle.is_done() is False
            result = {'actual_thread_and_queue_retained': True, 'successor_rejected': True}
    print('STT_START_CONTRACT=' + json.dumps({
        'outcome': 'confirmed', 'role': role, 'start_invocations': len(attempts),
        'pinned_os_worker_created': False, **result,
    }), flush=True)
    os._exit(0)  # Fault containment only; no native exit is manufactured.


@pytest.mark.parametrize('role', ['unconfirmed', 'registration'])
def test_real_start_receipt_distinguishes_uncertainty_from_pre_dispatch_absence(role):
    _run_case(role)
