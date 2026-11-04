"""Retain real executor threads when startup fails around Thread.start."""

import asyncio
import threading

import numpy as np
import pytest

from src.stt.whisper_fast import FastWhisperSTT, STTResult
from src.stt.pinned_executor import PinnedThreadPoolExecutor


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault_point", ["before_start", "after_start"])
async def test_partial_submit_keeps_model_and_real_thread_owned_until_actual_exit(
    monkeypatch, fault_point,
):
    if fault_point == "before_start":
        # An invoked start without a positive receipt cannot grant a retry.
        from tests.unit.stt.test_pinned_executor_start_admission import _run_case
        await asyncio.to_thread(_run_case, "whisper")
        return
    model = FastWhisperSTT(model="synthetic", device="cpu")
    entered, release, releasing_model = threading.Event(), threading.Event(), threading.Event()
    attempts, faulted = [], []
    original_start, original_release = threading.Thread.start, model._release_model

    def start(thread):
        if thread.name.startswith("whisper-cpu_"):
            attempts.append(thread)
            if not faulted:
                faulted.append(True)
                if fault_point == "after_start":
                    original_start(thread)
                raise RuntimeError("synthetic partial Thread.start failure")
        return original_start(thread)

    def load():
        if model._model is None:
            model._model = object()

    def infer(_audio):
        entered.set()
        assert release.wait(5), "synthetic inference not released"
        return STTResult("Synthetic partial submit", 1.0)

    def release_model():
        original_release()
        releasing_model.set()

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(model, "_load_impl", load)
    monkeypatch.setattr(model, "_transcribe_impl", infer)
    monkeypatch.setattr(model, "_release_model", release_model)
    caller = closing = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(
            model.transcribe_with_confidence, np.ones(16000, dtype=np.float32),
        ))
        with pytest.raises(RuntimeError, match="synthetic partial Thread.start failure"):
            await asyncio.wait_for(asyncio.shield(caller), 3)
        assert faulted == [True]
        if fault_point == "after_start":
            await until(entered.is_set)
            assert attempts[0].is_alive()
        closing = model.begin_close()
        # The confirmed native worker drains its admitted job before disposal.
        await until(entered.is_set)
        released_early = await asyncio.to_thread(releasing_model.wait, 0.15)
        if released_early:
            # Record the false success before finally releases the lost worker.
            await until(closing.done)
            assert closing.exception() is None
            assert model._model is None and attempts[0].is_alive()
            assert not release.is_set()
        assert not released_early
        assert not closing.done() and not closing.cancelled()
        assert model._model is not None
        assert any(thread.is_alive() for thread in attempts)
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert releasing_model.is_set() and model._model is None
        assert all(not thread.is_alive() for thread in attempts)
        assert closing.exception() is None
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        if model._executor is not None:
            await asyncio.to_thread(model._executor.shutdown, wait=True)
        for thread in attempts:
            if thread.ident is not None:
                await asyncio.to_thread(thread.join, 3)
                assert not thread.is_alive()


@pytest.mark.asyncio
async def test_native_handle_remains_owned_before_python_bootstrap_receipt(monkeypatch):
    entered, release, ran = threading.Event(), threading.Event(), threading.Event()
    original_start = threading.Thread.start
    original_bootstrap = threading.Thread._bootstrap_inner
    attempts = []

    def bootstrap(thread):
        if thread.name.startswith("whisper-native-gap_"):
            entered.set()
            assert release.wait(5), "synthetic native bootstrap not released"
        original_bootstrap(thread)

    def start(thread):
        if not thread.name.startswith("whisper-native-gap_"):
            return original_start(thread)
        attempts.append(thread)
        original_wait = thread._started.wait

        def failed_receipt(*_args, **_kwargs):
            raise RuntimeError("synthetic Python startup receipt failure")

        thread._started.wait = failed_receipt
        try:
            return original_start(thread)
        finally:
            thread._started.wait = original_wait

    monkeypatch.setattr(threading.Thread, "_bootstrap_inner", bootstrap)
    monkeypatch.setattr(threading.Thread, "start", start)
    pool = PinnedThreadPoolExecutor(thread_name_prefix="whisper-native-gap")
    caller = closing = None
    try:
        caller = asyncio.create_task(asyncio.to_thread(pool.submit, ran.set))
        await until(lambda: entered.is_set() and caller.done())
        with pytest.raises(RuntimeError, match="synthetic Python startup receipt failure"):
            await caller
        worker = attempts[0]
        assert tuple(pool._threads) == (worker,)
        assert worker.ident is None and worker._handle.ident > 0
        assert not worker._started.is_set() and not worker._handle.is_done()
        closing = asyncio.create_task(asyncio.to_thread(pool.shutdown, wait=True))
        await until(lambda: pool._shutdown)
        assert not closing.done() and not worker._handle.is_done()
        # No SDK submission lock may be held while awaiting this native exit.
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "healthy peer"), 1) == "healthy peer"
        assert not closing.done() and not ran.is_set()
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert ran.is_set() and worker._handle.is_done()
        assert not worker.is_alive() and worker._started.is_set()
    finally:
        release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await asyncio.to_thread(pool.shutdown, wait=True)
        for worker in attempts:
            if worker._handle.ident:
                await asyncio.to_thread(worker._handle.join)
                assert worker._handle.is_done()


def test_unsupported_private_layout_is_rejected_before_any_worker(monkeypatch):
    from concurrent.futures import thread as sdk

    attempts = []
    monkeypatch.setattr(sdk, "_worker", lambda: None)
    monkeypatch.setattr(threading.Thread, "start", lambda thread: attempts.append(thread))
    pool = None
    try:
        with pytest.raises(RuntimeError, match="^pinned_executor_unsupported_runtime$"):
            pool = PinnedThreadPoolExecutor()
        assert attempts == [] and pool is None
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
