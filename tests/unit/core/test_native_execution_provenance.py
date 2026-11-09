"""Original native tasks retain provenance without making copied contexts grants."""

import asyncio
from contextvars import copy_context
import gc
import sqlite3
import threading
import weakref

import pytest

from src.core.owned_work import (
    NativeExecution, OwnedBlockingWork, current_native_execution,
    observe_native, validate_native_execution,
)


async def _until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [False, True], ids=["run", "native_step_after_seal"])
async def test_original_native_entry_retains_actual_task_through_cancel_and_drain(sequence):
    entered, release = threading.Event(), threading.Event()
    sequence_entered, dispatch = asyncio.Event(), asyncio.Event()
    seen = []

    def native(argument):
        binding = current_native_execution()
        assert binding is not None and binding.work is work and binding.function is native
        assert validate_native_execution(binding, work=work, function=native)
        assert (binding.parent_sequence is not None) is sequence
        clone = NativeExecution(binding.work, binding.function, binding.task, binding.parent_sequence)
        assert not validate_native_execution(clone)
        db = sqlite3.connect(":memory:")
        try:
            db.execute("CREATE TABLE observations(value TEXT)")
            db.execute("INSERT INTO observations VALUES (?)", (argument,))
            seen.append((binding, db.execute("SELECT value FROM observations").fetchone()[0]))
            entered.set()
            assert release.wait(3)
            assert validate_native_execution(binding, work=work, function=native)
        finally:
            db.close()
        return argument

    async def worker(function, *args, **kwargs):
        assert function is native and args == ("synthetic",) and kwargs == {}
        return await asyncio.to_thread(observe_native, function, *args, **kwargs)

    work = OwnedBlockingWork(worker=worker, capacity=1)

    async def admitted_sequence():
        assert current_native_execution() is None
        sequence_entered.set()
        await dispatch.wait()
        return await work.native_step(native, "synthetic")

    observing = asyncio.create_task(
        work.run_async(admitted_sequence) if sequence else work.run(native, "synthetic"),
    )
    try:
        if sequence:
            await sequence_entered.wait()
            closing = work.begin_close()
            dispatch.set()
        else:
            await _until(entered.is_set)
            closing = work.begin_close()
        await _until(entered.is_set)
        observing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observing
        assert not closing.done() and work.status()["operations"] == 1
        with pytest.raises(RuntimeError, match="blocking_work_closed"):
            await work.run(native, "rejected")
    finally:
        release.set()
        dispatch.set()
        await work.begin_close()
        await asyncio.gather(observing, return_exceptions=True)
    assert len(seen) == 1 and seen[0][1] == "synthetic"
    assert not validate_native_execution(seen[0][0])
    assert current_native_execution() is None


@pytest.mark.asyncio
async def test_copied_context_is_not_live_native_execution_on_foreign_task_or_thread():
    entered, release = threading.Event(), threading.Event()
    captured = []

    def native():
        binding = current_native_execution()
        assert binding is not None
        captured.append((copy_context(), binding))
        entered.set()
        assert release.wait(3)
        assert current_native_execution() is binding

    work = OwnedBlockingWork()
    observing = asyncio.create_task(work.run(native))
    try:
        await _until(entered.is_set)
        context, binding = captured[0]
        assert context.copy().run(current_native_execution) is None
        assert await asyncio.to_thread(context.copy().run, current_native_execution) is None

        async def inherited_task():
            return current_native_execution(), validate_native_execution(binding)

        task = asyncio.create_task(inherited_task(), context=context.copy())
        assert await task == (None, False)
    finally:
        release.set()
        await observing
        await work.aclose()
    assert await asyncio.to_thread(captured[0][0].copy().run, current_native_execution) is None


@pytest.mark.asyncio
async def test_pure_async_sequence_and_unobserved_worker_do_not_mint_native_proof():
    async def sequence():
        return current_native_execution()

    async def unobserved_worker(function, *args, **kwargs):
        return function(*args, **kwargs)

    work = OwnedBlockingWork(worker=unobserved_worker)
    try:
        assert await work.run_async(sequence) is None
        assert await work.run(current_native_execution) is None
    finally:
        await work.aclose()


@pytest.mark.asyncio
async def test_foreign_asyncio_task_on_same_native_thread_cannot_inherit_proof():
    def native():
        binding = current_native_execution()
        assert binding is not None
        native_thread = threading.get_ident()

        async def foreign_task():
            assert threading.get_ident() == native_thread
            assert current_native_execution() is None
            assert not validate_native_execution(binding, work=work, function=native)

        asyncio.run(foreign_task())
        assert current_native_execution() is binding
        assert validate_native_execution(binding, work=work, function=native)

    work = OwnedBlockingWork()
    try:
        await work.run(native)
    finally:
        await work.aclose()


@pytest.mark.asyncio
async def test_observed_sync_worker_keeps_original_retained_task_proof():
    def native():
        binding = current_native_execution()
        assert binding is not None and binding.task is asyncio.current_task()
        assert validate_native_execution(binding, work=work, function=native)
        return "observed"

    async def worker(function, *args, **kwargs):
        assert function is native
        return observe_native(function, *args, **kwargs)

    work = OwnedBlockingWork(worker=worker)
    try:
        assert await work.run(native) == "observed"
    finally:
        await work.aclose()


@pytest.mark.asyncio
async def test_foreign_loop_handle_on_same_native_thread_cannot_inherit_proof():
    def native():
        binding = current_native_execution()
        assert binding is not None
        original_context = copy_context()
        native_thread = threading.get_ident()

        async def foreign_loop():
            loop = asyncio.get_running_loop()
            finished = loop.create_future()

            def replayed_sync_helper():
                try:
                    assert threading.get_ident() == native_thread
                    assert asyncio.current_task() is None
                    assert current_native_execution() is None
                    assert not validate_native_execution(binding, work=work, function=native)
                except BaseException as error:
                    finished.set_exception(error)
                else:
                    finished.set_result(None)

            loop.call_soon(replayed_sync_helper, context=original_context.copy())
            await finished

        asyncio.run(foreign_loop())
        assert current_native_execution() is binding
        assert validate_native_execution(binding, work=work, function=native)

    work = OwnedBlockingWork()
    try:
        await work.run(native)
    finally:
        await work.aclose()



class _PrivateInput:
    """Weak-referenceable stand-in for private native input."""


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence", [False, True], ids=["run", "native_step"])
async def test_cancelled_observer_late_native_failure_releases_owned_task_and_input(sequence):
    """A finished owned task must not be retained through its native binding.

    The observer is cancelled while the original native call is blocked; the
    owner keeps the task, then the call fails inside ``asyncio.to_thread``.
    The owned task stores that exception, whose traceback holds the worker
    frames, their private arguments and the copied thread ``Context``. Once
    the owner drops the task, reference counting alone must release task and
    input. Automatic collection is suspended so an incidental collector pass
    cannot hide a reference cycle; nothing is collected explicitly.
    """
    entered, release = threading.Event(), threading.Event()
    payload = _PrivateInput()
    payload_ref = weakref.ref(payload)
    task_refs = []

    def native(value):
        binding = current_native_execution()
        assert binding is not None and validate_native_execution(binding, work=work)
        assert value is payload_ref()
        task_refs.append(weakref.ref(binding.task))
        if sequence:
            task_refs.append(weakref.ref(binding.parent_sequence))
        # The callback frame is in the traceback; only the copied context and
        # owner frames are under test, so it must not keep the binding.
        del binding, value
        entered.set()
        assert release.wait(3)
        raise RuntimeError("synthetic_late_native_failure")

    async def admitted_sequence(value):
        return await work.native_step(native, value)

    work = OwnedBlockingWork(capacity=1)
    collector_was_enabled = gc.isenabled()
    gc.disable()
    try:
        if sequence:
            observer = asyncio.create_task(work.run_async(admitted_sequence, payload))
        else:
            observer = asyncio.create_task(work.run(native, payload))
        del payload
        await _until(entered.is_set)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert payload_ref() is not None and work._tasks
        release.set()
        await _until(lambda: not work._tasks)
        observer = None
        assert len(task_refs) == (2 if sequence else 1)
        assert all(task_ref() is None for task_ref in task_refs)
        assert payload_ref() is None
    finally:
        release.set()
        if collector_was_enabled:
            gc.enable()
        await work.aclose()
