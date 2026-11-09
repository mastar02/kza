"""Bounded blocking operations whose ownership survives caller cancellation."""

import asyncio
from collections import deque
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
import math
from collections.abc import Callable
import time
import threading
import traceback
from typing import NoReturn


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


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class NativeExecution:
    """Identity of one actual invocation; fields alone never grant admission."""

    work: "OwnedBlockingWork" = field(repr=False)
    function: Callable = field(repr=False)
    task: asyncio.Task = field(repr=False)
    parent_sequence: asyncio.Task | None = field(default=None, repr=False)

    @property
    def retained_task(self) -> asyncio.Task:
        """Return the original task whose admission owns this invocation."""
        return self.task if self.parent_sequence is None else self.parent_sequence

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("native_execution_not_serializable")


@dataclass
class _NativeState:
    binding: NativeExecution
    thread_id: int | None = None
    executing_loop: asyncio.AbstractEventLoop | None = None
    executing_task: asyncio.Task | None = None
    entered: bool = False


@dataclass(eq=False, slots=True)
class _NativeSlot:
    """Shared, revocable carrier of one binding inside every copied context.

    Contexts copied while a native invocation runs (``asyncio.to_thread``,
    executors, handles) are immutable snapshots that can outlive the call, for
    example through a stored exception traceback. Holding the binding directly
    in them formed an unreachable cycle: copied context -> binding -> finished
    owned task -> stored exception -> traceback frames -> copied context, which
    kept private native arguments and results alive until the cyclic
    collector ran. Every copy shares this slot, so the owner revokes the
    binding from all of them at the point where the binding stops validating.
    """

    binding: NativeExecution | None


_NATIVE_EXECUTION: ContextVar[_NativeSlot | None] = ContextVar(
    "owned_work_native_execution", default=None,
)
_NATIVE_LOCK = threading.RLock()


def _context_binding() -> NativeExecution | None:
    slot = _NATIVE_EXECUTION.get()
    return None if slot is None else slot.binding


def _revoke_slot(slot: _NativeSlot | None, binding: NativeExecution | None = None) -> None:
    """Clear ``slot`` (only if it still carries ``binding``, when given)."""
    if slot is None:
        return
    with _NATIVE_LOCK:
        if binding is None or slot.binding is binding:
            slot.binding = None


def _executing_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _executing_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def validate_native_execution(
    binding: NativeExecution | None, *, work: "OwnedBlockingWork | None" = None,
    function: Callable | None = None,
) -> bool:
    """Validate actual task/work/thread membership, not a copied context token."""
    if type(binding) is not NativeExecution or _context_binding() is not binding:
        return False
    with _NATIVE_LOCK:
        owner = binding.work
        state = owner._native_executions.get(id(binding))
        return bool(
            state is not None and state.binding is binding and state.entered
            and state.thread_id == threading.get_ident()
            and state.executing_loop is _executing_loop()
            and state.executing_task is _executing_task()
            and (work is None or owner is work)
            and (function is None or binding.function is function)
            and binding.retained_task in owner._tasks
            and not binding.task.done() and not binding.retained_task.done()
        )


def current_native_execution() -> NativeExecution | None:
    """Locate the actual current invocation while its native callback runs."""
    binding = _context_binding()
    return binding if validate_native_execution(binding) else None


def _enter_native(binding, function):
    if type(binding) is not NativeExecution or binding.function is not function:
        return False
    with _NATIVE_LOCK:
        state = binding.work._native_executions.get(id(binding))
        executing_task = _executing_task()
        if (state is None or state.binding is not binding or state.entered
                or executing_task is not None and executing_task is not binding.task
                or binding.retained_task not in binding.work._tasks
                or binding.task.done() or binding.retained_task.done()):
            return False
        state.thread_id = threading.get_ident()
        state.executing_loop = _executing_loop()
        state.executing_task = executing_task
        state.entered = True
        return True


def _leave_native(binding):
    if type(binding) is NativeExecution:
        with _NATIVE_LOCK:
            state = binding.work._native_executions.get(id(binding))
            if state is not None and state.binding is binding:
                binding.work._native_executions.pop(id(binding), None)


def _release_failed_frames(error: BaseException) -> None:
    """Drop locals of the finished native frames held by ``error``'s tracebacks.

    The owned task stores ``error`` and the binding keeps that task as its
    completion witness, so any finished native frame (also in a chained
    ``__cause__``/``__context__`` or an exception-group member) that still
    holds the binding closes a cycle which also retains the native arguments.
    Tracebacks, line numbers and the exception objects are unchanged.

    Relies on CPython >= 3.13 (the runtime artifact target): ``frame.clear()``
    raises ``RuntimeError`` for executing or suspended frames, which
    ``traceback.clear_frames`` skips, so no live generator or other thread's
    frame is closed or altered. Older interpreters would close suspended
    generators here.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending += [current.__cause__, current.__context__]
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def observe_native(function: Callable, *args, **kwargs):
    """Observe one real synchronous invocation without changing its arguments."""
    timing = _TIMING.get()
    owns_timing = timing is not None and timing.native_started is None
    slot = _NATIVE_EXECUTION.get()
    binding = None if slot is None else slot.binding
    owns_execution = _enter_native(binding, function)
    failure = None
    if owns_timing:
        timing.native_started = time.monotonic()
    try:
        # Entered provenance is already live, and _NATIVE_LOCK is released.
        # This hook must finish before the original callback can acquire a
        # ledger or SQLite lock. Its failure still runs original cleanup.
        if owns_execution and binding.work._native_preflight is not None:
            binding.work._native_preflight(binding)
        return function(*args,**kwargs)
    except BaseException as error:
        failure = error
        raise
    finally:
        if owns_timing:
            timing.native_finished = time.monotonic()
        if owns_execution:
            _leave_native(binding)
            # The left binding can no longer validate; revoke it from every
            # copied context before the result or exception propagates.
            _revoke_slot(slot, binding)
            # Only after the admission window closed: finalizers of released
            # locals then run outside it, as they did when frames died later.
            if failure is not None:
                _release_failed_frames(failure)
        timing = slot = binding = function = args = kwargs = failure = None


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
        self._native_executions: dict[int, _NativeState] = {}
        self._native_preflight_owner: object | None = None
        self._native_preflight: Callable[[NativeExecution], None] | None = None
        # Exact admitted coroutine function per retained sequence task; never
        # its arguments. Cleared by the original finished callback.
        self._sequence_functions: dict[asyncio.Task, Callable] = {}

    def _install_native_preflight(
        self, owner: object, callback: Callable[[NativeExecution], None],
    ) -> None:
        """Install one observer on this original owner, without changing dispatch."""
        if self._native_preflight_owner is owner:
            return
        if self._native_preflight_owner is not None or self._closed:
            raise RuntimeError("native_preflight_owner_changed")
        self._native_preflight_owner, self._native_preflight = owner, callback

    def _native_binding(self, function, task, parent_sequence=None):
        retained = task if parent_sequence is None else parent_sequence
        with _NATIVE_LOCK:
            if retained not in self._tasks or retained.done():
                raise RuntimeError("native_execution_not_retained")
            if len(self._native_executions) >= self._capacity:
                raise RuntimeError("native_execution_capacity")
            binding = NativeExecution(self, function, task, parent_sequence)
            self._native_executions[id(binding)] = _NativeState(binding)
            return binding

    async def _execute(self, timing, function, args, kwargs):
        token = _TIMING.set(timing)
        binding = native_token = slot = None
        try:
            # _retain must publish ownership before even an eager worker starts.
            await asyncio.sleep(0)
            binding = self._native_binding(function, asyncio.current_task())
            slot = _NativeSlot(binding)
            native_token = _NATIVE_EXECUTION.set(slot)
            timing.dispatched = time.monotonic()
            # Preserve the exact callable seen by existing private workers.
            return await self._worker(function,*args,**kwargs)
        finally:
            _leave_native(binding)
            _revoke_slot(slot)
            if native_token is not None:
                _NATIVE_EXECUTION.reset(native_token)
            timing.finished = time.monotonic()
            _TIMING.reset(token)
            binding, slot, function, args, kwargs = None, None, None, (), {}

    async def _execute_async(self, timing, function, args, kwargs):
        # A coordinator is not a native invocation. Its native timing is unknown;
        # nested service owners retain their own actual invocation observations.
        token = _TIMING.set(None)
        native_token = _NATIVE_EXECUTION.set(None)
        timing.dispatched = time.monotonic()
        try:
            # Even an eager task factory must register ownership before a step.
            await asyncio.sleep(0)
            return await function(*args, **kwargs)
        finally:
            timing.finished = time.monotonic()
            _TIMING.reset(token)
            _NATIVE_EXECUTION.reset(native_token)
            function, args, kwargs = None, (), {}

    async def _execute_native_step(self, parent_sequence, function, args, kwargs):
        binding = self._native_binding(function, asyncio.current_task(), parent_sequence)
        slot = _NativeSlot(binding)
        token = _NATIVE_EXECUTION.set(slot)
        try:
            return await self._worker(function, *args, **kwargs)
        finally:
            _leave_native(binding)
            _revoke_slot(slot)
            _NATIVE_EXECUTION.reset(token)
            # This frame stays in the traceback stored by the parent sequence;
            # keeping the parent here would make that sequence retain itself.
            parent_sequence = binding = slot = function = None
            args, kwargs = (), {}

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
                self._sequence_functions.pop(done, None)
                observation = self._timings.pop(done, None)
                if observation is not None:
                    self._history.append(observation.snapshot(time.monotonic()))
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished, context=Context())
            if asynchronous:
                try:
                    # _execute_async yields before entering ``function``, so a
                    # failed observation can still stop the retained task.
                    self._sequence_functions[task] = function
                except BaseException:
                    task.cancel()
                    raise
            return task
        finally:
            operation = task = function = None
            args, kwargs = (), {}

    def _retained_sequence_function(self, task: asyncio.Task) -> Callable | None:
        """Observe the exact function admitted for a still-retained live sequence.

        This is provenance only, never a permission or dispatch token.
        """
        if task not in self._tasks or task.done():
            return None
        return self._sequence_functions.get(task)

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
            task = asyncio.create_task(self._execute_native_step(
                asyncio.current_task(), function, args, kwargs,
            ))
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
