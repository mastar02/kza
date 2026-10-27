"""Bounded blocking operations whose ownership survives caller cancellation."""

import asyncio
from collections import deque
from contextvars import Context, ContextVar
from dataclasses import dataclass
from enum import StrEnum
import math
from collections.abc import Callable
import time


class ClosePhase(StrEnum):
    UNKNOWN = 'unknown'
    IDLE = 'idle'
    DRAIN = 'drain'
    MAINTENANCE = 'maintenance'
    WRITES = 'writes'
    RETENTION = 'retention'
    RESOURCES = 'resources'
    SOCKET = 'socket'
    STORE = 'store'
    LEDGER = 'ledger'
    EXECUTOR = 'executor'
    LEASE = 'lease'
    COMPLETE = 'complete'


@dataclass
class WorkTiming:
    """Only monotonic timestamps cross the native observation boundary."""

    admitted: float
    dispatched: float | None = None
    native_started: float | None = None
    native_finished: float | None = None
    finished: float | None = None

    def snapshot(self, now: float) -> dict:
        """Report numeric waits without retaining an operation or its context."""
        phase = ('complete' if self.finished is not None else 'after_native' if self.native_finished is not None
            else 'native' if self.native_started is not None else 'worker' if self.dispatched is not None else 'queued')
        def duration(start, end):
            return None if start is None else max(0.,((now if end is None else end)-start)*1000)
        return {'phase':phase,'age_ms':duration(self.admitted,self.finished),
            'dispatch_ms':duration(self.admitted,self.dispatched),
            'native_queue_ms':duration(self.dispatched,self.native_started) if self.native_started is not None else None,
            'native_ms':duration(self.native_started,self.native_finished),
            'after_native_ms':duration(self.native_finished,self.finished)}


_TIMING: ContextVar[WorkTiming | None] = ContextVar('owned_work_timing',default=None)


def observe_native(function: Callable, *args, **kwargs):
    """Observe one real synchronous invocation without changing its arguments."""
    timing = _TIMING.get()
    if timing is None or timing.native_started is not None:
        return function(*args,**kwargs)
    timing.native_started = time.monotonic()
    try:
        return function(*args,**kwargs)
    finally:
        timing.native_finished = time.monotonic()
        timing = function = args = kwargs = None


async def _observed_thread(function, *args, **kwargs):
    return await asyncio.to_thread(observe_native,function,*args,**kwargs)


class WorkCloseTimeout(RuntimeError):
    """Native operations still own their resources after the close deadline."""

    def __init__(self, message: str, *, phase: ClosePhase = ClosePhase.UNKNOWN, diagnostics: dict | None = None):
        super().__init__(message)
        self.phase = phase
        self.diagnostics = diagnostics


class OwnedBlockingWork:
    """Retain actual worker completion, including delegated native operations."""

    def __init__(self, *, worker: Callable | None = None, capacity: int = 8, close_timeout_s: float = 1):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("invalid_blocking_work_capacity")
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("invalid_blocking_work_close_timeout")
        self._worker = worker or _observed_thread
        self._capacity = capacity
        self._close_timeout_s = close_timeout_s
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self._closing: asyncio.Task | None = None
        self._timings: dict[asyncio.Task,WorkTiming] = {}
        self._history = deque(maxlen=capacity)

    async def _execute(self, timing, function, args, kwargs):
        token = _TIMING.set(timing)
        timing.dispatched = time.monotonic()
        try:
            return await self._worker(function,*args,**kwargs)
        finally:
            timing.finished = time.monotonic()
            _TIMING.reset(token)
            function, args, kwargs = None, (), {}

    async def run(self, function: Callable, *args, **kwargs):
        """Admit bounded work and shield its actual worker from caller cancellation."""
        if self._closed:
            raise RuntimeError("blocking_work_closed")
        if len(self._tasks) >= self._capacity:
            raise RuntimeError("blocking_work_capacity")
        timing = WorkTiming(time.monotonic())
        task = asyncio.create_task(self._execute(timing,function,args,kwargs))
        self._tasks.add(task)
        self._timings[task] = timing

        def finished(done):
            self._tasks.discard(done)
            observation = self._timings.pop(done,None)
            if observation is not None:
                self._history.append(observation.snapshot(time.monotonic()))
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished,context=Context())
        try:
            return await asyncio.shield(task)
        finally:
            # A cancelled observer can retain this frame through its exception.
            # The owner set, not that traceback, keeps actual native work alive.
            # Drop the observer's task/argument links so a later result cannot
            # remain reachable after the native completion callback releases it.
            task, function, args, kwargs = None, None, (), {}

    def status(self) -> dict:
        """Report actual retained operations, not cancelled callers."""
        return {"closed": self._closed, "operations": len(self._tasks), "capacity": self._capacity}

    def diagnostics(self) -> dict:
        """Observe bounded native progress; uninstrumented custom workers stay unknown."""
        now = time.monotonic()
        return {'active':[timing.snapshot(now) for timing in tuple(self._timings.values())],
            'recent':[dict(row) for row in self._history]}

    def begin_close(self) -> asyncio.Task:
        """Close admission and return the actual native drain task for an owner."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            self._closing = asyncio.create_task(finish(),context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Close admission and wait for actual work, retaining timed-out cleanup."""
        self.begin_close()
        cancelled = False
        while not self._closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout("blocking_work_still_running",phase=ClosePhase.DRAIN,
                    diagnostics=self.diagnostics()) from error
        self._closing.result()
        if cancelled:
            raise asyncio.CancelledError
