"""Real pinned executors/threads; only model and constructor leaves are synthetic.

Root owns the separate graph factory/runtime baseline tests. No capture, model
loading, graph.run or shutdown of the process-wide default executor happens here.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
import threading

import numpy as np
import pytest

import src.stt.whisper_fast as whisper_module
from src.stt.whisper_fast import FastWhisperSTT, STTResult


pytestmark = pytest.mark.asyncio


def synthetic_stt(monkeypatch):
    stt = FastWhisperSTT(model="synthetic-only", device="cpu", compute_type="float32")
    loaded = []

    def load():
        if stt._model is None:
            worker = threading.current_thread()
            assert stt._run_pinned(threading.current_thread) is worker
            loaded.append(worker)
            stt._model = object()

    monkeypatch.setattr(stt, "_load_impl", load)
    monkeypatch.setattr(stt, "_transcribe_impl", lambda _audio: STTResult("Synthetic STT", 1.0))
    return stt, loaded


def held_executor_constructor(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    created = []

    class HeldExecutor(whisper_module.ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
            entered.set()
            assert release.wait(5), "synthetic executor constructor not released"

    monkeypatch.setattr(whisper_module, "ThreadPoolExecutor", HeldExecutor)
    return entered, release, created


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def final_cleanup(stts, pools, releases, tasks):
    """Safety only after assertions, including a failed/RED run."""
    for release in releases:
        release.set()
    for stt in stts:
        request = getattr(stt, "request_close", None)
        if callable(request):
            request()
    unique = {id(pool): pool for pool in pools if pool is not None}
    for stt in stts:
        pool = stt._executor
        if pool is not None:
            unique[id(pool)] = pool
    # No manual shutdown is used by the behavior under test, only this finally helper.
    for pool in unique.values():
        await asyncio.to_thread(pool.shutdown, wait=True)
    await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)


async def test_pinned_inference_and_cancelled_close_observers_retain_thread_and_other_instance(monkeypatch):
    stt, loaded = synthetic_stt(monkeypatch)
    peer, peer_loaded = synthetic_stt(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    nested, refused = [], []
    pools, observers = [], []
    caller = closing = None

    def infer(_audio):
        worker = threading.current_thread()
        entered.set()
        assert release.wait(5), "synthetic inference not released"
        # Already-admitted native work may reenter after the close seal.
        nested.append(stt._run_pinned(threading.current_thread))
        try:
            stt.begin_close()
        except RuntimeError as error:
            refused.append(str(error))
        else:
            raise AssertionError("worker was allowed to own its own close")
        assert nested[-1] is worker
        return STTResult("Synthetic retained inference", 1.0)

    monkeypatch.setattr(stt, "_transcribe_impl", infer)
    try:
        await asyncio.to_thread(stt.load)
        await asyncio.to_thread(peer.load)
        pool, peer_pool = stt._executor, peer._executor
        pools.extend((pool, peer_pool))
        threads, peer_threads = tuple(pool._threads), tuple(peer_pool._threads)
        assert len(threads) == len(peer_threads) == 1
        assert threads[0] is loaded[0] and peer_threads[0] is peer_loaded[0]
        assert threads[0] is not peer_threads[0] and threads[0].is_alive()
        caller = asyncio.create_task(asyncio.to_thread(
            stt.transcribe_with_confidence, np.ones(16000, dtype=np.float32),
        ))
        await until(entered.is_set)
        caller.cancel()
        result = await asyncio.gather(caller, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        closing = stt.begin_close()
        assert stt.begin_close() is closing
        assert not closing.done() and not closing.cancelled()
        for _ in range(2):
            observer = asyncio.create_task(stt.aclose())
            observers.append(observer)
            await asyncio.sleep(0)
            observer.cancel()
            await asyncio.sleep(0)
            observer.cancel()
            assert not observer.done()
            assert not closing.done() and not closing.cancelled()
            assert threads[0].is_alive()
        with pytest.raises(RuntimeError, match="^whisper_closed$"):
            await asyncio.wait_for(asyncio.to_thread(stt.load), 3)
        assert not closing.done() and threads[0].is_alive()
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert all(isinstance(value, asyncio.CancelledError) for value in results)
        assert nested == [threads[0]] and refused == ["whisper_close_from_worker"]
        assert not threads[0].is_alive() and pool._shutdown
        assert stt._model is None and closing.exception() is None
        assert stt.begin_close() is closing
        value = await asyncio.to_thread(peer.transcribe_with_confidence, np.ones(16000, dtype=np.float32))
        assert value.text == "Synthetic STT"
        assert peer._executor is peer_pool and peer_threads[0].is_alive()
    finally:
        await final_cleanup((stt, peer), pools, (release,), (caller, closing, *observers))


@pytest.mark.parametrize("entry", ["load", "transcribe"])
async def test_close_retains_first_executor_creation_after_cancelled_admission(monkeypatch, entry):
    stt, _loaded = synthetic_stt(monkeypatch)
    entered, release, created = held_executor_constructor(monkeypatch)
    caller = closing = None
    try:
        function = stt.load if entry == "load" else lambda: stt.transcribe_with_confidence(
            np.ones(16000, dtype=np.float32),
        )
        caller = asyncio.create_task(asyncio.to_thread(function))
        await until(entered.is_set)
        assert len(created) == 1
        pool = created[0]
        assert isinstance(pool, RealThreadPoolExecutor) and not pool._shutdown
        caller.cancel()
        result = await asyncio.gather(caller, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        closing = stt.begin_close()
        assert stt.begin_close() is closing
        for _ in range(20):
            await asyncio.sleep(0)
        assert not closing.done() and not closing.cancelled()
        assert not pool._shutdown
        with pytest.raises(RuntimeError, match="^whisper_closed$"):
            await asyncio.wait_for(asyncio.to_thread(stt.load), 3)
        assert len(created) == 1 and not closing.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert pool._shutdown and all(not thread.is_alive() for thread in pool._threads)
        assert closing.exception() is None and stt._model is None
        with pytest.raises(RuntimeError, match="^whisper_closed$"):
            await asyncio.wait_for(asyncio.to_thread(stt.load), 3)
        assert len(created) == 1
    finally:
        await final_cleanup((stt,), created, (release,), (caller, closing))


async def test_concurrent_first_admissions_share_one_executor_and_reenter_on_its_worker(monkeypatch):
    stt, loaded = synthetic_stt(monkeypatch)
    entered, release, created = held_executor_constructor(monkeypatch)
    barrier = threading.Barrier(2)
    callers = []

    def together():
        callers.append(threading.current_thread())
        barrier.wait(timeout=3)
        stt.load()

    first = asyncio.create_task(asyncio.to_thread(together))
    second = asyncio.create_task(asyncio.to_thread(together))
    try:
        await until(lambda: entered.is_set() and len(callers) == 2)
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(created) == 1
        release.set()
        assert await asyncio.gather(first, second) == [None, None]
        assert len(created) == 1
        pool = created[0]
        threads = tuple(pool._threads)
        assert len(threads) == 1 and loaded == [threads[0]]
        assert all(caller is not threads[0] for caller in callers)
        assert threads[0].is_alive() and stt._executor is pool
        value = await asyncio.to_thread(stt.transcribe_with_confidence, np.ones(16000, dtype=np.float32))
        assert value.text == "Synthetic STT" and loaded == [threads[0]]
        await asyncio.to_thread(stt.close)
        assert pool._shutdown and not threads[0].is_alive()
        assert stt._model is None
        with pytest.raises(RuntimeError, match="^whisper_closed$"):
            await asyncio.wait_for(asyncio.to_thread(stt.load), 3)
        assert len(created) == 1
    finally:
        barrier.abort()
        await final_cleanup((stt,), created, (release,), (first, second))
