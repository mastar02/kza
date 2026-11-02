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


class WorkKind(StrEnum):
    """Distinguish native timings from an owned asynchronous coordinator."""

    NATIVE = "native"
    SEQUENCE = "sequence"


@dataclass
class WorkTiming:
    """Only monotonic timestamps and a closed work kind cross the boundary."""

    admitted: float
    dispatched: float | None = None
    native_started: float | None = None
    native_finished: float | None = None
    finished: float | None = None
    kind: WorkKind = WorkKind.NATIVE

    def snapshot(self, now: float) -> dict:
        """Report numeric waits without retaining an operation or its context."""
        phase = ('complete' if self.finished is not None else 'sequence' if self.kind is WorkKind.SEQUENCE
            else 'after_native' if self.native_finished is not None else 'native' if self.native_started is not None
            else 'worker' if self.dispatched is not None else 'queued')
        def duration(start, end):
            return None if start is None else max(0.,((now if end is None else end)-start)*1000)
        return {'phase':phase,'kind':self.kind.value,'age_ms':duration(self.admitted,self.finished),
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

    def __init__(self, *, worker: Callable | None = None, capacity: int = 8, close_timeout_s: float = 1,
                 isolate_context: bool = False):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("invalid_blocking_work_capacity")
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("invalid_blocking_work_close_timeout")
        if type(isolate_context) is not bool:
            raise ValueError("invalid_blocking_work_context_policy")
        self._worker = worker or _observed_thread
        self._isolate_context = isolate_context
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

    async def _execute_async(self, timing, function, args, kwargs):
        # A coordinator is not a native invocation. Its native timing is unknown;
        # nested service owners retain their own actual invocation observations.
        token = _TIMING.set(None)
        timing.dispatched = time.monotonic()
        try:
            # Even an eager task factory must register ownership before a step.
            await asyncio.sleep(0)
            return await function(*args, **kwargs)
        finally:
            timing.finished = time.monotonic()
            _TIMING.reset(token)
            function, args, kwargs = None, (), {}

    def _retain(self, function: Callable, args: tuple, kwargs: dict, *, asynchronous: bool = False) -> asyncio.Task:
        operation = task = None
        try:
            if self._closed:
                raise RuntimeError("blocking_work_closed")
            if len(self._tasks) >= self._capacity:
                raise RuntimeError("blocking_work_capacity")
            timing = WorkTiming(time.monotonic(), kind=WorkKind.SEQUENCE if asynchronous else WorkKind.NATIVE)
            execute = self._execute_async if asynchronous else self._execute
            operation = execute(timing, function, args, kwargs)
            try:
                task = asyncio.create_task(operation, context=Context() if self._isolate_context else None)
            except BaseException:
                operation.close()
                raise
            self._tasks.add(task)
            self._timings[task] = timing

            def finished(done):
                self._tasks.discard(done)
                observation = self._timings.pop(done, None)
                if observation is not None:
                    self._history.append(observation.snapshot(time.monotonic()))
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished, context=Context())
            return task
        finally:
            operation = task = function = None
            args, kwargs = (), {}

    async def run_async(self, function: Callable, *args, **kwargs) -> object:
        """Retain one bounded async sequence through observer cancellation and close."""
        task = None
        try:
            task = self._retain(function, args, kwargs, asynchronous=True)
            return await asyncio.shield(task)
        finally:
            task = function = None
            args, kwargs = (), {}

    async def native_step(self, function: Callable, *args, **kwargs) -> object:
        """Run native work only inside an admitted sequence, including after sealing.

        Await this method directly. A separate caller task does not own a step.
        Cancellation during a step waits for its actual worker before propagating.
        """
        task = result = None
        try:
            if asyncio.current_task() not in self._tasks:
                raise RuntimeError("blocking_work_step_not_owned")
            task = asyncio.create_task(self._worker(function, *args, **kwargs))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            result = task.result()
            if cancelled:
                raise asyncio.CancelledError
            return result
        finally:
            task = result = function = None
            args, kwargs = (), {}

    async def run(self, function: Callable, *args, **kwargs):
        """Admit native work and release rejected/cancelled observer input links."""
        task = None
        try:
            task = self._retain(function, args, kwargs)
            return await asyncio.shield(task)
        finally:
            task = function = None
            args, kwargs = (), {}

    def status(self) -> dict:
        """Report actual retained operations, not cancelled callers."""
        return {"closed": self._closed, "operations": len(self._tasks), "capacity": self._capacity}

    def diagnostics(self) -> dict:
        """Observe bounded native progress; uninstrumented custom workers stay unknown."""
        now = time.monotonic()
        return {'active':[timing.snapshot(now) for timing in tuple(self._timings.values())],
            'recent':[dict(row) for row in self._history]}

    async def drain_admitted(self) -> None:
        """Wait for the current admitted snapshot without closing or cancelling work."""
        admitted = tuple(self._tasks)
        try:
            if admitted:
                await asyncio.shield(asyncio.gather(*admitted, return_exceptions=True))
        finally:
            admitted = None

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
