"""One retained owner covers async preparation and the actual native completion."""

import asyncio
from contextvars import ContextVar
import gc
import threading
import weakref

import pytest

from src.core.owned_work import OwnedBlockingWork


async def test_cancelled_sequence_finishes_its_native_step_after_admission_closes():
    owner = OwnedBlockingWork(capacity=1, isolate_context=True)
    marker = ContextVar('synthetic_sequence_context', default=None)
    prepared, proceed = asyncio.Event(), asyncio.Event()
    entered, release = threading.Event(), threading.Event()

    class Payload:
        pass

    value = Payload()
    reference = weakref.ref(value)

    def persist(payload):
        try:
            assert marker.get() is None
            assert isinstance(payload, Payload)
            entered.set()
            assert release.wait(5)
            return 'saved'
        finally:
            payload = None

    async def sequence(payload):
        try:
            assert marker.get() is None
            prepared.set()
            await proceed.wait()
            return await owner.native_step(persist, payload)
        finally:
            payload = None

    token = marker.set('synthetic private context')
    try:
        observer = asyncio.create_task(owner.run_async(sequence, value))
    finally:
        marker.reset(token)
    retained_error = None
    try:
        await asyncio.wait_for(prepared.wait(), 2)
        observer.cancel()
        try:
            await observer
        except asyncio.CancelledError as error:
            retained_error = error
        value = None
        closing = owner.begin_close()
        assert owner.status() == {'closed': True, 'operations': 1, 'capacity': 1}
        with pytest.raises(RuntimeError, match='blocking_work_closed'):
            await owner.run_async(sequence, None)
        proceed.set()
        assert await asyncio.to_thread(entered.wait, 2)
        assert not closing.done()
        assert reference() is not None
        [timing] = owner.diagnostics()['active']
        assert timing['phase'] == 'sequence' and timing['kind'] == 'sequence'
        assert timing['native_ms'] is None
    finally:
        proceed.set()
        release.set()
        await asyncio.shield(owner.begin_close())
    await asyncio.sleep(0)
    gc.collect()
    assert retained_error is not None
    assert reference() is None
    assert owner.status()['operations'] == 0


async def test_native_step_rejects_an_unowned_task_even_with_an_inherited_context():
    owner = OwnedBlockingWork(capacity=1)
    calls = []

    def native():
        calls.append(True)

    with pytest.raises(RuntimeError, match='blocking_work_step_not_owned'):
        await owner.native_step(native)

    async def sequence():
        with pytest.raises(RuntimeError, match='blocking_work_step_not_owned'):
            await asyncio.create_task(owner.native_step(native))

    await owner.run_async(sequence)
    await owner.aclose()
    assert calls == []


async def test_eager_task_factory_registers_sequence_before_its_first_native_step():
    owner = OwnedBlockingWork(capacity=1)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    async def sequence():
        return await owner.native_step(lambda: 'saved')

    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        assert await owner.run_async(sequence) == 'saved'
    finally:
        loop.set_task_factory(previous)
        await owner.aclose()


async def test_cancellation_of_the_retained_sequence_cannot_abandon_an_actual_native_worker():
    owner = OwnedBlockingWork(capacity=1)
    entered, release = threading.Event(), threading.Event()

    def native():
        entered.set()
        assert release.wait(5)

    async def sequence():
        await owner.native_step(native)

    observer = asyncio.create_task(owner.run_async(sequence))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        retained = next(iter(owner._tasks))
        for _ in range(2):
            retained.cancel()
            await asyncio.sleep(0)
        closing = owner.begin_close()
        assert not retained.done()
        assert not closing.done()
        assert owner.status()['operations'] == 1
    finally:
        release.set()
        await asyncio.gather(observer, asyncio.shield(owner.begin_close()), return_exceptions=True)
    assert observer.cancelled()
    assert owner.status()['operations'] == 0


@pytest.mark.parametrize('asynchronous', [False, True])
async def test_capacity_rejection_traceback_does_not_keep_unadmitted_arguments(asynchronous):
    owner = OwnedBlockingWork(capacity=1)
    started, release = asyncio.Event(), asyncio.Event()

    class Payload:
        pass

    async def hold():
        started.set()
        await release.wait()

    async def unused_async(payload):
        raise AssertionError('not admitted')

    def unused_native(payload):
        raise AssertionError('not admitted')

    running = asyncio.create_task(owner.run_async(hold))
    await started.wait()
    payload = Payload()
    reference = weakref.ref(payload)
    operation = owner.run_async if asynchronous else owner.run
    rejected = asyncio.create_task(operation(unused_async if asynchronous else unused_native, payload))
    retained_error = None
    try:
        try:
            await rejected
        except RuntimeError as error:
            retained_error = error
        payload = None
        gc.collect()
        assert retained_error is not None
        assert reference() is None
    finally:
        release.set()
        await running
        await owner.aclose()
