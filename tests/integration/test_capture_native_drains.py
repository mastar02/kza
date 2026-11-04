"""Real native receipts for reusable capture drains, with no SDK/model load."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.application import STTExecutorOwner, _ModelResourceOwner
from src.core.owned_work import OwnedBlockingWork
from src.memory.capture_native import build_capture_native_drains
from src.pipeline.response_handler import ResponseHandler
from src.stt.parakeet_stt import ParakeetSTT


pytestmark = pytest.mark.integration


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@pytest.fixture
def submissions(monkeypatch):
    """Forward SDK submission and identify the actual Future from its worker."""
    original = ThreadPoolExecutor.submit
    local = threading.local()
    futures, lock = {}, threading.Lock()

    def submit(executor, function, /, *args, **kwargs):
        key = object()

        def invoke():
            local.key = key
            try:
                return function(*args, **kwargs)
            finally:
                del local.key

        future = original(executor, invoke)
        with lock:
            futures[key] = future
        return future

    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)

    class Receipts:
        @staticmethod
        def current():
            return local.key

        @staticmethod
        def future(key):
            with lock:
                return futures.get(key)

    return Receipts()


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_returned_model_lease_does_not_retire_native_postlude(monkeypatch, submissions, cancel_observer):
    model = ParakeetSTT()
    backend = object()
    model._model = backend  # Public load takes its existing-model path; no onnx import.
    cohort = STTExecutorOwner()
    cohort.add_model(model)
    work = OwnedBlockingWork(isolate_context=True)
    cohort.add_borrower(work.begin_close)
    drain_entered = asyncio.Event()
    original_drain = work.drain_admitted

    async def observed_drain():
        drain_entered.set()
        await original_drain()

    monkeypatch.setattr(work, "drain_admitted", observed_drain)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    keys = []

    def native():
        try:
            model.load()
            keys.append(submissions.current())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test_native_release_missing")
        finally:
            returned.set()

    producer = asyncio.create_task(work.run(native))
    draining = observer = None
    try:
        await until(entered.is_set)
        await until(lambda: keys and submissions.future(keys[0]) is not None)
        future = submissions.future(keys[0])
        assert future.running() and not future.done()
        with model._parakeet_condition:
            assert not any(lease.active for lease in model._parakeet_leases)
        drains, supported = build_capture_native_drains(model_owners=(cohort,))
        assert supported
        draining = asyncio.create_task(drains[0]())
        if cancel_observer:
            async def observe():
                return await asyncio.shield(draining)
            observer = asyncio.create_task(observe())
            await asyncio.sleep(0)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        await asyncio.wait_for(drain_entered.wait(), 3)
        assert not draining.done() and not draining.cancelled()
        assert not returned.is_set() and not future.done()
        assert model._model is backend and not model._parakeet_sealed
        assert not work._closed
        release.set()
        await asyncio.wait_for(asyncio.shield(draining), 3)
        await producer
        assert returned.is_set() and future.done() and not future.cancelled()
        assert model._model is backend and not model._parakeet_sealed
        model.load()
        assert model._model is backend
    finally:
        release.set()
        await asyncio.gather(producer, return_exceptions=True)
        if draining is not None:
            await asyncio.gather(asyncio.shield(draining), return_exceptions=True)
        await work.aclose()
        await model.aclose()


async def test_response_native_postlude_outlives_released_worker_slot(monkeypatch, submissions):
    class TTSLeaf:
        def synthesize(self, text):
            return object()

    handler = ResponseHandler(TTSLeaf())
    entered, release = threading.Event(), threading.Event()
    keys = []
    original = handler._worker_finished
    drain_entered = asyncio.Event()
    original_drain = handler._native_work.drain_admitted

    async def observed_drain():
        drain_entered.set()
        await original_drain()

    monkeypatch.setattr(handler._native_work, "drain_admitted", observed_drain)

    def finished(session):
        original(session)
        keys.append(submissions.current())
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test_response_release_missing")

    monkeypatch.setattr(handler, "_worker_finished", finished)
    producer = asyncio.create_task(handler.asynthesize("synthetic"))
    draining = None
    try:
        await until(entered.is_set)
        await until(lambda: keys and submissions.future(keys[0]) is not None)
        future = submissions.future(keys[0])
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        assert handler.get_delivery_status()["workers"] == 0
        assert handler.get_delivery_status()["native_operations"] == 1
        assert future.running() and not future.done()
        drains, supported = build_capture_native_drains(responses=handler)
        assert supported
        draining = asyncio.create_task(drains[0]())
        await asyncio.wait_for(drain_entered.wait(), 3)
        assert not draining.done() and not future.done()
        assert not handler._closed and not handler._native_work._closed
        release.set()
        await asyncio.wait_for(asyncio.shield(draining), 3)
        assert future.done()
        assert handler.get_delivery_status()["native_calls"] == 0
        assert not handler._closed and not handler._native_work._closed
        assert await handler.asynthesize("still reusable") is not None
    finally:
        release.set()
        await asyncio.gather(producer, return_exceptions=True)
        if draining is not None:
            await asyncio.gather(asyncio.shield(draining), return_exceptions=True)
        await handler.wait_closed()


async def test_unknown_model_is_not_reported_as_supported():
    cohort = _ModelResourceOwner((object,))
    foreign = object()
    cohort.add_model(foreign)
    drains, supported = build_capture_native_drains(model_owners=(cohort,))
    assert not supported
    with pytest.raises(RuntimeError, match="^capture_native_retirement_unconfirmed$"):
        await drains[0]()
    assert cohort._models[id(foreign)] is foreign
    assert cohort._closing is None


async def test_model_lease_without_borrower_future_is_not_a_native_receipt():
    model = ParakeetSTT()
    backend = object()
    model._model = backend
    cohort = STTExecutorOwner()
    cohort.add_model(model)
    try:
        drains, supported = build_capture_native_drains(model_owners=(cohort,))
        assert not supported
        with pytest.raises(RuntimeError, match="^capture_native_retirement_unconfirmed$"):
            await drains[0]()
        assert model._model is backend and not model._parakeet_sealed
    finally:
        await model.aclose()
