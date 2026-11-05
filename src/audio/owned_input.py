"""Retain actual input-driver construction, start and close after cancellation."""

import asyncio
from collections.abc import Callable
from contextvars import Context
from enum import StrEnum
import logging

from src.core.owned_work import OwnedBlockingWork

logger = logging.getLogger(__name__)


_OWNED_INPUTS: set = set()


def _input_failures(error: BaseException) -> tuple[tuple[str, int | None], ...]:
    if isinstance(error, BaseExceptionGroup):
        return tuple(item for child in error.exceptions for item in _input_failures(child))
    if isinstance(error, KeyboardInterrupt):
        return (("interrupt", None),)
    if isinstance(error, SystemExit):
        code = error.code
        return (("exit", code if code is None or type(code) is int else 1),)
    if isinstance(error, InputRetirementError):
        return error.failures
    return (("error" if isinstance(error, Exception) else "control", None),)


class InputNotAdmitted(RuntimeError):
    """Signal that a composite factory did not enter its native constructor."""


class InputRetirementError(RuntimeError):
    """A failed native retirement, with control information stored as data."""

    def __init__(self, failures=()):
        super().__init__("capture_input_retirement_unconfirmed")
        self.failures = tuple(failures) or (("error", None),)


def _raise_open_failure(failures) -> None:
    errors = []
    for kind, code in failures:
        if kind == "interrupt":
            errors.append(KeyboardInterrupt())
        elif kind == "exit":
            errors.append(SystemExit(code))
        elif kind == "control":
            errors.append(BaseException("capture_input_not_open"))
        else:
            errors.append(RuntimeError("capture_input_not_open"))
    if not errors:
        errors.append(RuntimeError("capture_input_not_open"))
    if len(errors) == 1 and isinstance(errors[0], Exception):
        raise errors[0]
    # asyncio only reraises bare KeyboardInterrupt/SystemExit globally.
    # A fresh group preserves those leaves without escaping an owner Task.
    raise BaseExceptionGroup("capture_input_open_failed", errors)


class InputState(StrEnum):
    NEW = 'new'
    OPENING = 'opening'
    OPEN = 'open'
    CLOSING = 'closing'
    CLOSED = 'closed'
    UNKNOWN = 'unknown'


class OwnedInputStream:
    """Keep even a late-created stream until its native close really succeeds."""

    def __init__(self, factory: Callable, retire: Callable[[], None], *, configure: Callable | None = None) -> None:
        self._factory, self._retire = factory, retire
        self._configure = configure
        self._work = OwnedBlockingWork(capacity=1)
        self._stream = None
        self._state = InputState.NEW
        self._closed = False
        self._admitted = False
        self._constructor_entered = False
        self._constructor_returned = False
        self._close_attempted = False
        self._retired = False
        self._unknown = False
        self._open_failures: tuple[tuple[str, int | None], ...] = ()
        self._failures: list[tuple[str, int | None]] = []
        self._close_failures = 0
        self._closing: asyncio.Task | None = None
        self._close_call: asyncio.Task | None = None
        _OWNED_INPUTS.add(self)

    def _open(self) -> bool:
        if self._closed:
            return False
        self._state = InputState.OPENING
        try:
            self._constructor_entered = True
            try:
                self._stream = self._factory()
            except InputNotAdmitted:
                self._constructor_entered = False
                return False
            self._constructor_returned = True
            if self._stream is None:
                self._unknown = True
                raise RuntimeError('capture_input_constructor_unconfirmed')
            if self._closed:
                return False
            if self._configure is not None:
                self._configure(self._stream)
            if self._closed:
                return False
            self._stream.start()
            if self._closed:
                return False
            self._state = InputState.OPEN
            return True
        except BaseException as error:
            self._open_failures = _input_failures(error)
            if not self._constructor_returned:
                self._unknown = True
            logger.warning('Capture input open failed: %s', type(error).__name__)
            return False

    async def open(self) -> None:
        """Admit one native open; observer cancellation never retires its work."""
        if self._closed or self._admitted:
            raise RuntimeError('capture_input_admission_closed')
        self._admitted = True
        try:
            opened = await self._work.run(self._open)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._unknown = True
            self._open_failures = _input_failures(error)
            opened = False
        if not opened:
            _raise_open_failure(self._open_failures)

    @property
    def stream(self) -> object | None:
        """Expose only a successfully opened, still admitted native handle."""
        return self._stream if self._state is InputState.OPEN and not self._closed and not self._unknown else None

    @property
    def retired(self) -> bool:
        """True only after an actual successful disposal or no acquisition."""
        return self._retired and not self._unknown

    @property
    def retirement_unknown(self) -> bool:
        """A failed disposal/acquisition stays unknown after its Task ends."""
        return self._unknown

    def _close(self) -> bool:
        self._state = InputState.CLOSING
        stream = self._stream
        if stream is None:
            self._unknown |= self._constructor_entered and not self._constructor_returned
            self._retired = not self._unknown
            self._state = InputState.CLOSED if self._retired else InputState.UNKNOWN
            return self._retired
        if self._close_attempted:
            self._state = InputState.CLOSED if self._retired else InputState.UNKNOWN
            return self._retired
        self._close_attempted = True
        try:
            try:
                stream.stop(ignore_errors=False)
            except BaseException as error:
                self._failures.extend(_input_failures(error))
            try:
                stream.close(ignore_errors=False)
            except BaseException as error:
                self._close_failures += 1
                self._unknown = True
                self._failures.extend(_input_failures(error))
            else:
                self._stream = None
                self._retired = not self._unknown
            self._state = InputState.CLOSED if self._retired else InputState.UNKNOWN
            return self._retired
        finally:
            stream = None

    def begin_close(self) -> asyncio.Task:
        """Seal once; observe one strict native close without retrying failure."""
        self._closed = True
        if self._closing is None:
            try:
                self._retire()
            except Exception as error:
                logger.warning('Capture input retirement failed: %s', type(error).__name__)
            except BaseException as error:
                self._failures.extend(_input_failures(error))

            async def finish():
                async def observe(task):
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except BaseException as error:
                            if not task.done():
                                self._failures.extend(_input_failures(error))
                    try:
                        return task.result()
                    except BaseException as error:
                        self._failures.extend(_input_failures(error))
                        self._unknown = True
                        return None

                try:
                    # Each retained Task awaits the actual borrowed-executor
                    # call, including late constructor/start return.
                    for task in tuple(self._work._tasks):
                        await observe(task)
                    self._close_call = asyncio.create_task(
                        self._work.run(self._close), context=Context())
                    await observe(self._close_call)
                except BaseException as error:
                    self._failures.extend(_input_failures(error))
                    self._unknown = True
                finally:
                    try:
                        # A late constructor can publish a new callback holder.
                        self._retire()
                    except Exception as error:
                        logger.warning('Capture input final retirement failed: %s', type(error).__name__)
                    except BaseException as error:
                        self._failures.extend(_input_failures(error))
                    # Native uncertainty does not leave Python admission open.
                    await observe(self._work.begin_close())
                if self._unknown:
                    self._retired = False
                    self._state = InputState.UNKNOWN
                if self._retired:
                    _OWNED_INPUTS.discard(self)
                if not self._retired or self._failures:
                    raise InputRetirementError((*self._open_failures, *self._failures))
            self._closing = asyncio.create_task(finish(), context=Context())
            self._closing.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        return self._closing

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual pending calls, separately from persistent uncertainty."""
        tasks = set(self._work._tasks)
        tasks.update(task for task in (self._close_call, self._closing)
                     if task is not None and not task.done())
        return tuple(tasks)

    def resource_status(self) -> dict:
        """Describe admission, actual work and native retirement independently."""
        return {'state': self._state.value, 'closed': self._closed,
                'operations': len(self.snapshot_operations()), 'close_failures': self._close_failures,
                'retirement_confirmed': self.retired, 'retirement_unknown': self.retirement_unknown,
                'failure_count': len(self._open_failures) + len(self._failures)}
