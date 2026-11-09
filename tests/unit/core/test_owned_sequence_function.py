"""Actual admitted sequence function observation lives only as long as its task."""
import asyncio
import gc
import weakref

import pytest

from src.core.owned_work import OwnedBlockingWork


class _Owner:
    """Bound-method owner whose lifetime must not be extended by observation."""

    def __init__(self, observed):
        self.observed = observed

    async def sequence(self, payload, outcome):
        # Weak: a strong task here would form a test-made task/traceback cycle.
        self.observed.append(weakref.ref(asyncio.current_task()))
        assert payload is not None
        await asyncio.sleep(0)
        if outcome == "error":
            raise ValueError("synthetic sequence failure")
        if outcome == "cancel":
            await asyncio.Event().wait()
        return "ok"


class _Payload:
    pass


@pytest.mark.parametrize("outcome", ["ok", "error", "cancel"])
async def test_sequence_function_is_exact_while_retained_and_released_after_terminal(outcome):
    gc.disable()
    try:
        work = OwnedBlockingWork(capacity=2)
        observed = []
        owner, payload = _Owner(observed), _Payload()
        owner_ref, payload_ref = weakref.ref(owner), weakref.ref(payload)
        function = owner.sequence
        pending = asyncio.create_task(work.run_async(function, payload, outcome))
        while not observed:
            await asyncio.sleep(0)
        task = observed[0]()
        assert task in work._tasks
        assert work._retained_sequence_function(task) is function
        assert work._retained_sequence_function(asyncio.current_task()) is None
        if outcome == "cancel":
            task.cancel()
        result = await asyncio.gather(pending, return_exceptions=True)
        assert task.done() and task not in work._tasks
        assert work._retained_sequence_function(task) is None
        assert work._sequence_functions == {}
        assert (result[0] == "ok") if outcome == "ok" else isinstance(result[0], BaseException)
        function = owner = payload = pending = result = observed = None
        task = None
        await asyncio.sleep(0)
        assert owner_ref() is None and payload_ref() is None
    finally:
        gc.enable()


async def test_native_run_publishes_no_sequence_function():
    work = OwnedBlockingWork(capacity=1)
    seen = []

    def native():
        seen.append(dict(work._sequence_functions))
        return 1

    assert await work.run(native) == 1
    assert seen == [{}] and work._sequence_functions == {}


class _FailingMap(dict):
    def __setitem__(self, key, value):
        raise MemoryError("synthetic observation failure")


async def test_observation_publication_failure_keeps_task_owned_and_never_enters():
    work = OwnedBlockingWork(capacity=1)
    entered = []

    async def sequence():
        entered.append(True)

    work._sequence_functions = _FailingMap()
    with pytest.raises(MemoryError):
        await work.run_async(sequence)
    retained = tuple(work._tasks)
    assert len(retained) == 1 and work.status()["operations"] == 1
    await asyncio.gather(*retained, return_exceptions=True)
    await asyncio.sleep(0)
    assert entered == [] and retained[0].cancelled()
    assert work.status()["operations"] == 0 and dict(work._sequence_functions) == {}
    await work.aclose()


async def test_capacity_rejection_creates_no_sequence_observation():
    work = OwnedBlockingWork(capacity=1)
    release = asyncio.Event()

    async def held():
        await release.wait()

    first = asyncio.create_task(work.run_async(held))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="capacity"):
        await work.run_async(held)
    assert len(work._sequence_functions) == 1
    release.set()
    await first
    assert work._sequence_functions == {}
