"""Backpressure, resource ownership and late legacy producer isolation."""

import asyncio
import threading
import time
from contextlib import aclosing

import pytest

from src.llm.generation_stream import GenerationBusyError, GenerationStream

pytestmark = pytest.mark.asyncio


async def wait_idle(owner):
    async with asyncio.timeout(1):
        while owner.sync_busy:
            await asyncio.sleep(0.001)


async def collect(owner):
    async with aclosing(owner.astream("synthetic")) as source:
        return [chunk["token"] async for chunk in source]


async def test_blocking_sync_generation_does_not_block_loop():
    class Legacy:
        def generate_stream(self, prompt):
            time.sleep(0.2)
            yield {"token": "complete", "text": "discard accumulated text"}

    owner = GenerationStream(Legacy())
    task = asyncio.create_task(collect(owner))
    beats = 0
    while not task.done():
        await asyncio.sleep(0.005)
        beats += 1
    assert await task == ["complete"]
    assert beats >= 20 and not owner.sync_busy


async def test_slow_consumer_bounds_queue_and_closes_source():
    produced = []
    closed = threading.Event()

    class Legacy:
        def generate_stream(self, prompt):
            try:
                for number in range(1000):
                    produced.append(number)
                    yield {"token": str(number)}
            finally:
                closed.set()

    owner = GenerationStream(Legacy(), queue_capacity=2)
    async with aclosing(owner.astream("synthetic")) as source:
        assert (await anext(source))["token"] == "0"
        await asyncio.sleep(0.02)
        # One consumed, two buffered and at most one waiting in the producer.
        assert len(produced) <= 4
    await wait_idle(owner)
    assert closed.is_set()


async def test_blocked_cancelled_producer_stays_quarantined_until_real_exit():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Legacy:
        calls = 0

        def generate_stream(self, prompt):
            self.calls += 1
            try:
                if self.calls == 1:
                    entered.set()
                    assert release.wait(1)
                    yield {"token": "late old answer"}
                else:
                    yield {"token": "new answer"}
            finally:
                closed.set()

    legacy = Legacy()
    owner = GenerationStream(legacy)
    task = asyncio.create_task(collect(owner))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        for _ in range(20):
            with pytest.raises(GenerationBusyError):
                await collect(owner)
        assert owner.sync_busy and legacy.calls == 1
    finally:
        release.set()
    await wait_idle(owner)
    assert closed.is_set()
    assert await collect(owner) == ["new answer"]


async def test_producer_exception_is_not_normal_eof():
    class Legacy:
        def generate_stream(self, prompt):
            yield {"token": "partial"}
            raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        await collect(GenerationStream(Legacy()))


async def test_sync_full_generation_uses_same_worker_and_limit():
    class Legacy:
        def generate(self, prompt):
            return "complete answer"

    assert await collect(GenerationStream(Legacy())) == ["complete answer"]
    with pytest.raises(ValueError, match="generation_text_limit"):
        await collect(GenerationStream(Legacy(), max_chars=5))


async def test_native_async_source_closed_on_early_exit():
    closed = False

    class Native:
        async def astream(self, prompt, *, max_chars):
            nonlocal closed
            assert max_chars == 123
            try:
                yield {"token": "complete answer"}
                await asyncio.Event().wait()
            finally:
                closed = True

    owner = GenerationStream(Native(), max_chars=123)
    async with aclosing(owner.astream("synthetic")) as source:
        assert (await anext(source))["token"] == "complete answer"
    assert closed and not owner.sync_busy
