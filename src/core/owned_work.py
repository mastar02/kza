"""Bounded blocking operations whose ownership survives caller cancellation."""

import asyncio
import math
from collections.abc import Callable


class WorkCloseTimeout(RuntimeError):
    """Native operations still own their resources after the close deadline."""


class OwnedBlockingWork:
    """Retain actual worker completion, including delegated native operations."""

    def __init__(self, *, worker: Callable | None = None, capacity: int = 8, close_timeout_s: float = 1):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("invalid_blocking_work_capacity")
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("invalid_blocking_work_close_timeout")
        self._worker = worker or asyncio.to_thread
        self._capacity = capacity
        self._close_timeout_s = close_timeout_s
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self._closing: asyncio.Task | None = None

    async def run(self, function: Callable, *args, **kwargs):
        """Admit bounded work and shield its actual worker from caller cancellation."""
        if self._closed:
            raise RuntimeError("blocking_work_closed")
        if len(self._tasks) >= self._capacity:
            raise RuntimeError("blocking_work_capacity")
        task = asyncio.create_task(self._worker(function, *args, **kwargs))
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    def status(self) -> dict:
        """Report actual retained operations, not cancelled callers."""
        return {"closed": self._closed, "operations": len(self._tasks), "capacity": self._capacity}

    async def aclose(self) -> None:
        """Close admission and wait for actual work, retaining timed-out cleanup."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            self._closing = asyncio.create_task(finish())
        cancelled = False
        while not self._closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout("blocking_work_still_running") from error
        self._closing.result()
        if cancelled:
            raise asyncio.CancelledError
