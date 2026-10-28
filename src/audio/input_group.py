"""Retained native microphone lifetimes and a reusable all-input admission gate."""

import asyncio
from collections.abc import Callable, Iterable
from contextvars import Context

from src.audio.owned_input import OwnedInputStream
from src.core.owned_work import OwnedBlockingWork


class InputGroup:
    """Own each driver before opening it; seal before waiting for native exit.

    This is a driver primitive. The caller must separately retire command and
    ambient processing before reporting a completed product-level total mute.
    """

    def __init__(self, keys: Iterable[str], *, factory: Callable, configure: Callable,
                 retire: Callable, callbacks: Callable) -> None:
        self._keys = tuple(keys)
        if len(set(self._keys)) != len(self._keys):
            raise ValueError('invalid_input_group')
        self._factory, self._configure = factory, configure
        self._retire, self._callbacks = retire, callbacks
        self._owners: dict[str, OwnedInputStream] = {}
        self._opening: dict[str, asyncio.Task] = {}
        self._catalog = OwnedBlockingWork(capacity=1)
        self._catalog_calls: set[asyncio.Task] = set()
        self._paused = self._closed = False
        self._pausing = self._closing = None
        self.streams: dict[str, object] = {}

    @property
    def accepting(self) -> bool:
        """Read the driver fence before clocks, array work or a native open."""
        return not self._paused and not self._closed

    def _retire_key(self, key):
        self.streams.pop(key, None)
        self._retire(key)

    async def open(self, key: str) -> object | None:
        """Deduplicate one retained open per configured room, even after cancel."""
        if key not in self._keys:
            raise ValueError('unknown_input_key')
        if not self.accepting:
            return None
        task = self._opening.get(key)
        if task is None or task.done():
            task = self._opening[key] = asyncio.create_task(self._open(key), context=Context())
            def finished(done):
                if self._opening.get(key) is done:
                    self._opening.pop(key)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def _open(self, key):
        if not self.accepting:
            return None
        owner = self._owners.get(key)
        if owner is not None:
            if owner.stream is not None:
                return owner.stream
            await asyncio.shield(owner.begin_close())
        if not self.accepting:
            return None
        owner = OwnedInputStream(lambda: self._factory(key), lambda: self._retire_key(key),
                                 configure=lambda stream: self._configure(key, stream))
        self._owners[key] = owner
        try:
            await owner.open()
        except RuntimeError:
            await asyncio.shield(owner.begin_close())
            return None
        if not self.accepting or owner.stream is None:
            await asyncio.shield(owner.begin_close())
            return None
        self.streams[key] = owner.stream
        return owner.stream

    async def retire(self, key: str) -> None:
        """Retire a failed room without losing its actual native close."""
        owner = self._owners.get(key)
        if owner is not None:
            await asyncio.shield(owner.begin_close())

    async def catalog_call(self, function: Callable) -> None:
        """Retain shared PortAudio catalog work through observer cancellation."""
        def admitted():
            if self.accepting:
                function()
        if self.accepting:
            task = asyncio.create_task(self._catalog.run(admitted), context=Context())
            self._catalog_calls.add(task)
            def finished(done):
                self._catalog_calls.discard(done)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished, context=Context())
            await asyncio.shield(task)

    def begin_pause(self) -> asyncio.Task:
        """Seal immediately and await native opens, closes and callback release."""
        self._paused = True
        for key in self._keys:
            self._retire_key(key)
        closes = [owner.begin_close() for owner in self._owners.values()]
        if self._pausing is None:
            async def finish():
                await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._opening.values())),
                                     return_exceptions=True)
                await asyncio.gather(*(asyncio.shield(task) for task in closes))
                await asyncio.gather(*(asyncio.shield(task) for task in self._catalog_calls | set(self._catalog._tasks)),
                                     return_exceptions=True)
                while any(self._callbacks(key).status()['active'] for key in self._keys):
                    await asyncio.sleep(.005)
            self._pausing = asyncio.create_task(finish(), context=Context())
        return self._pausing

    def resume(self) -> None:
        """Allow fresh driver owners only after the actual pause has completed."""
        if self._closed or self._pausing is not None and not self._pausing.done():
            raise RuntimeError('input_group_still_owned')
        if self._pausing is not None:
            self._pausing.result()
        if any(owner.stream is not None or owner.snapshot_operations() for owner in self._owners.values()):
            raise RuntimeError('input_group_still_owned')
        self._paused = False
        self._pausing = None

    def begin_close(self) -> asyncio.Task:
        """Keep terminal shutdown ownership after bounded callers stop waiting."""
        self._closed = True
        pause = self.begin_pause()
        if self._closing is None:
            async def finish():
                await asyncio.shield(pause)
                await self._catalog.begin_close()
            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Describe real driver work without input handles or callback values."""
        tasks = set(self._opening.values()) | set(self._catalog._tasks) | self._catalog_calls
        for owner in self._owners.values():
            tasks.update(owner.snapshot_operations())
        tasks.update(task for task in (self._pausing, self._closing) if task is not None)
        return tuple(task for task in tasks if not task.done())

    def resource_status(self) -> dict:
        """Project fixed counts and admission, never devices or household keys."""
        return {'admission_sealed': not self.accepting, 'closed': self._closed,
                'pause_complete': self._pausing is not None and self._pausing.done()
                    and not self._pausing.cancelled() and self._pausing.exception() is None,
                'streams': len(self.streams), 'operations': len(self.snapshot_operations()),
                'callbacks': sum(self._callbacks(key).status()['active'] for key in self._keys)}
