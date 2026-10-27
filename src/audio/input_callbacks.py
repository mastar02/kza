"""Bounded metadata ownership of admitted native input callback invocations."""

import asyncio
from collections import deque
from dataclasses import dataclass

from src.memory.fact_ledger import FactLedger, MemorySource
from src.memory.sources import CaptureBoundary


@dataclass(eq=False)
class _CallbackSlot:
    timestamp: float | None = None
    source: MemorySource | None = None
    active: bool = False
    boundary: CaptureBoundary | None = None
    branch: object | None = None
    branch_bound: bool = False


class InputCallbacks:
    """Reserve fixed slots without locks, authority queries or raw input values.

    CPython deque popleft/append reserve and release a slot atomically. A slot
    absent from the free pool is owned even before its metadata is stamped or
    after its fields are cleared. Only the control loop waits for that pool.
    """

    def __init__(self, capacity: int = 8):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('invalid_input_callback_capacity')
        self._slots = tuple(_CallbackSlot() for _ in range(capacity))
        self._free = deque(self._slots)
        self._closed = False
        self._drops = 0
        self._errors = 0
        self._closing = None

    def enter(self, boundary: CaptureBoundary, *, branch: object | None = None) -> _CallbackSlot | None:
        """Reserve before clocks, array methods, copies or any other input work."""
        try:
            slot = self._free.popleft()
        except IndexError:
            self._drops += 1
            return None
        slot.active = True
        slot.boundary = boundary
        slot.branch = branch
        slot.branch_bound = True
        return slot

    def release_branch(self, slot: _CallbackSlot) -> None:
        """Retire a secondary delivery before continuing independent input work."""
        slot.branch = None

    def branch_pending(self, branch: object) -> bool:
        """Include the reservation gap before a callback can publish its branch."""
        free = set(self._free)
        return any(slot not in free and (not slot.branch_bound or slot.branch is branch)
                   for slot in self._slots)

    @property
    def accepting(self) -> bool:
        """Sealed callbacks still retire their arguments before releasing a slot."""
        return not self._closed

    def stamp(self, slot: _CallbackSlot, timestamp: float, source: MemorySource | None) -> None:
        """Publish acquisition metadata, never permission or a new source ID."""
        slot.timestamp, slot.source = timestamp, source

    def leave(self, slot: _CallbackSlot) -> None:
        """Release only after the callback has retired its own raw arguments."""
        if not slot.active:
            raise RuntimeError('input_callback_not_owned')
        slot.timestamp = slot.source = None
        slot.boundary = None
        slot.branch = None
        slot.branch_bound = False
        slot.active = False
        self._free.append(slot)

    def failed(self) -> None:
        """Record an input failure without retaining its exception or arguments."""
        self._errors += 1

    def busy(self, conversation_id: str, ledger: FactLedger, boundary: CaptureBoundary) -> bool:
        """Unknown or retired input stays pending until its actual callback exits."""
        free = set(self._free)
        for slot in self._slots:
            if slot in free:
                continue
            timestamp, source = slot.timestamp, slot.source
            if timestamp is None or source is None or slot.boundary != boundary or not boundary.allows(timestamp):
                return True
            if source.conversation_id == conversation_id or ledger.dependencies_committed_unavailable(source.fact_dependencies):
                return True
        return False

    def status(self) -> dict:
        """Expose counts only; slots contain no arrays, callbacks or exceptions."""
        return {'closed': self._closed, 'capacity': len(self._slots),
                'active': len(self._slots)-len(self._free), 'drops': self._drops, 'errors': self._errors}

    def begin_close(self) -> asyncio.Task:
        """Seal admission and await the exit of all previously admitted callbacks."""
        self._closed = True
        if self._closing is None:
            async def finish():
                while len(self._free) != len(self._slots):
                    await asyncio.sleep(.005)
            self._closing = asyncio.create_task(finish())
        return self._closing
