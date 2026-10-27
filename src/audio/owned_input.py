"""Retain actual input-driver construction, start and close after cancellation."""

import asyncio
from collections.abc import Callable
from enum import StrEnum
import logging

from src.core.owned_work import OwnedBlockingWork

logger = logging.getLogger(__name__)


class InputState(StrEnum):
    NEW = 'new'
    OPENING = 'opening'
    OPEN = 'open'
    CLOSING = 'closing'
    CLOSED = 'closed'


class OwnedInputStream:
    """Keep even a late-created stream until its native close really succeeds."""

    def __init__(self, factory: Callable, retire: Callable[[], None]) -> None:
        self._factory, self._retire = factory, retire
        self._work = OwnedBlockingWork(capacity=1)
        self._stream = None
        self._state = InputState.NEW
        self._closed = False
        self._admitted = False
        self._close_failures = 0
        self._closing: asyncio.Task | None = None

    def _open(self) -> bool:
        if self._closed:
            return False
        self._state = InputState.OPENING
        try:
            self._stream = self._factory()
            if self._closed:
                return False
            self._stream.start()
            if self._closed:
                return False
            self._state = InputState.OPEN
            return True
        except Exception as error:
            logger.warning('Capture input open failed: %s', type(error).__name__)
            return False

    async def open(self) -> None:
        """Admit one native open; its handle remains owned after observer cancel."""
        if self._closed or self._admitted:
            raise RuntimeError('capture_input_admission_closed')
        self._admitted = True
        if not await self._work.run(self._open):
            raise RuntimeError('capture_input_not_open')

    def _close(self) -> bool:
        self._state = InputState.CLOSING
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception as error:
                # close() still owns the driver release when stop() failed.
                logger.debug('Capture input stop deferred to close: %s', type(error).__name__)
            try:
                stream.close()
            except Exception as error:
                self._close_failures += 1
                logger.warning('Capture input close deferred: %s', type(error).__name__)
                return False
            finally:
                stream = None
        self._stream = None
        self._state = InputState.CLOSED
        return True

    def begin_close(self) -> asyncio.Task:
        """Retire callbacks now; drain open then retry one actual close at a time."""
        self._closed = True
        try:
            self._retire()
        except Exception as error:
            # Retiring a callback must never prevent releasing its driver.
            logger.warning('Capture input retirement failed: %s', type(error).__name__)
        if self._closing is None:
            async def finish():
                await asyncio.gather(*(asyncio.shield(t) for t in tuple(self._work._tasks)), return_exceptions=True)
                delay = .05
                while not await self._work.run(self._close):
                    await asyncio.sleep(delay)
                    delay = min(1., delay * 2)
                await self._work.begin_close()
            self._closing = asyncio.create_task(finish())
        return self._closing

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose native driver calls and a retained retry/drain controller."""
        tasks = set(self._work._tasks)
        if self._closing is not None and not self._closing.done():
            tasks.add(self._closing)
        return tuple(tasks)

    def resource_status(self) -> dict:
        """Describe ownership without handles, callbacks or device information."""
        return {'state': self._state.value, 'closed': self._closed,
                'operations': len(self.snapshot_operations()), 'close_failures': self._close_failures}
