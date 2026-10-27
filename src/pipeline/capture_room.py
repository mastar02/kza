"""One serialized native owner for command capture and wake state in one room."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from contextvars import ContextVar
from enum import StrEnum
import logging
from uuid import uuid4
from typing import TYPE_CHECKING
from threading import Lock

import numpy as np

from src.core.owned_work import OwnedBlockingWork
from src.audio.input_callbacks import InputCallbacks
from src.memory.fact_ledger import MemoryOrigin, MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT, producer_context
from src.memory.producer_scopes import ProducerScope, ProducerScopes
from src.memory.prompt_context import SourcesInvalidated
from src.memory.sources import CaptureBoundary

if TYPE_CHECKING:
    from src.memory.privacy_services import PrivacyServices
    from src.pipeline.command_event import CommandEvent

logger = logging.getLogger(__name__)
_CAPTURE_CHECK = ContextVar('native_capture_check', default=None)


def require_capture_live() -> None:
    """Guard nested native wake handoffs under the admitted room owner."""
    check = _CAPTURE_CHECK.get()
    if check is not None:
        check()


@dataclass(frozen=True)
class CapturePacket:
    """A copied input frame with its original acquisition boundary."""

    timestamp: float
    audio: np.ndarray
    frames: int


class _AuxiliaryError(StrEnum):
    """Fixed outcomes of a private native callback."""

    INVALIDATED = 'capture_auxiliary_invalidated'
    FAILED = 'capture_auxiliary_failed'


@dataclass(frozen=True)
class _AuxiliaryResult:
    """Return data or a fixed error without transporting an exception."""

    value: object = None
    error: _AuxiliaryError | None = None


@dataclass
class _AuxiliaryCall:
    """Mutable input indirection; transport tuples retain only the emptied box."""

    function: Callable | None
    args: tuple
    sources: tuple[MemorySource, ...] | None
    admitted: bool = False

    def admit(self) -> None:
        self.admitted = True

    def clear(self) -> None:
        self.function, self.args = None, ()


class CaptureRoom:
    """Serialize wake, capture, snapshots and reset without blocking PortAudio.

    The callback only appends to a bounded deque. Boundary publication detaches
    old deques; native workers recheck authority before and after processing.
    Actual native completion survives cancellation of a controller or observer.
    """

    def __init__(self, room_id: str, *, process: Callable, poll: Callable,
                 clear: Callable, listening: Callable[[], bool],
                 privacy: PrivacyServices | None = None) -> None:
        self.room_id = room_id
        self._process, self._poll, self._clear = process, poll, clear
        self._privacy = privacy
        self._listening = listening
        self._queue = deque(maxlen=32)
        self._boundary = CaptureBoundary(None, float('-inf'))
        self._source: MemorySource | None = None
        self._oldest = None
        self._work = OwnedBlockingWork(capacity=1)
        self._lock = asyncio.Lock()
        self._pending: asyncio.Task | None = None
        self._controllers: set[asyncio.Task] = set()
        self._scopes = ProducerScopes(self.snapshot_operations, capacity=16)
        self._closed = False
        self._closing: asyncio.Task | None = None
        self._last_packet = None
        self._failed = False
        self._input_drops = 0
        self.callbacks = InputCallbacks()
        self._boundary_lock = Lock()  # Only control publishers acquire it; never PortAudio.

    @property
    def source(self) -> MemorySource | None:
        """Expose metadata for finite capture and early-STT lineage."""
        return self._source

    @property
    def input_boundary(self) -> CaptureBoundary:
        """Expose the immutable published admission epoch without authority I/O."""
        return self._boundary

    def push(self, timestamp: float, audio: np.ndarray, frames: int, *, boundary: CaptureBoundary | None = None) -> None:
        """Copy input without locks, model calls or authority queries on callback."""
        queue = self._queue  # Snapshot queue before fence; never revive a detached queue.
        admitted, current = boundary, self._boundary
        try:
            if not self._closed and not self._failed and (admitted is None or admitted == current) and current.allows(timestamp):
                queue.append(CapturePacket(timestamp, audio.copy(), frames))
        except Exception:
            self._input_drops += 1
        finally:
            audio = queue = None

    def accepts(self, timestamp: float, *, boundary: CaptureBoundary | None = None) -> bool:
        """Read only the published fence on the input callback thread."""
        return not self._closed and not self._failed and (boundary is None or boundary == self._boundary) and self._boundary.allows(timestamp)

    def publish_boundary(self) -> None:
        """Tighten callback admission, then detach retired queued audio."""
        if self._privacy is None:
            return
        boundary = self._privacy.scopes.buffer_boundary(self.room_id, origin=MemoryOrigin.VOICE)
        with self._boundary_lock:
            boundary = self._boundary.tighten(boundary)
            if boundary != self._boundary:
                self._boundary = boundary
                queue = self._queue
                self._queue = deque((p for p in tuple(queue) if self._boundary.allows(p.timestamp)), maxlen=32)
                # An in-flight copy may hold an empty old deque. It must never
                # publish into the current queue after the stricter fence.
                queue.clear()

    def retire_stream(self, timestamp: float) -> None:
        """Fence a retired driver's frames and detach its old queue before replacement."""
        with self._boundary_lock:
            self._boundary = self._boundary.tighten(CaptureBoundary(None, timestamp))
            self._queue = deque(maxlen=32)

    def _current_boundary(self) -> CaptureBoundary:
        boundary = self._boundary
        if self._privacy is not None:
            boundary = boundary.tighten(self._privacy.scopes.buffer_boundary(self.room_id, origin=MemoryOrigin.VOICE))
        return self._boundary.tighten(boundary)

    def available(self) -> bool:
        return not self._closed and not self._failed and (self._privacy is None or self._source is not None and
            self._privacy.sources_available((self._source,))) and (
                self._oldest is None or self._current_boundary().allows(self._oldest))

    def require_live(self) -> None:
        """Guard native handoffs, including model waiters and later stages."""
        if not self.available():
            raise SourcesInvalidated('capture_sources_invalidated')

    def _reset(self):
        try:
            self._clear()
        except Exception as error:
            self._failed = True
            self._queue = deque(maxlen=32)
            logger.warning('Command capture reset failed: %s', type(error).__name__)
        else:
            self._failed = False
            self._source = self._oldest = self._last_packet = None

    async def _call_locked(self, function, *args, sources=None, on_admit=None):
        task = None
        try:
            if self._pending is not None:
                await asyncio.gather(asyncio.shield(self._pending), return_exceptions=True)
            source = self._source
            lineage = ((source,) if source is not None else ()) if sources is None else sources
            with producer_context(self.room_id, lineage):
                task = asyncio.create_task(self._work.run(function, *args))
            self._pending = task
            if on_admit is not None:
                on_admit()

            def finished(done):
                if self._pending is done:
                    self._pending = None
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
            return await asyncio.shield(task)
        finally:
            function = args = task = on_admit = None

    async def call(self, function: Callable, *args, sources: tuple[MemorySource, ...] | None = None) -> object:
        """Run finite state work; explicit lineage describes retained auxiliary inputs."""
        task = None
        try:
            lineage = sources if sources is not None else PRODUCER_CONTEXT.get().sources
            if not lineage and sources is None:
                lineage = (self._source,) if self._source is not None else None
            with self._scopes.operation(lineage):
                task = self._start_controller(self._call_owned(function, args, lineage))
                return await task
        finally:
            function = args = task = None

    def _start_controller(self, operation):
        if self._closed:
            operation.close()
            raise SourcesInvalidated('capture_closed')
        if len(self._controllers) >= 8:
            operation.close()
            raise RuntimeError('capture_controller_capacity')
        task = asyncio.create_task(operation)
        self._controllers.add(task)
        def finished(done):
            self._controllers.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    async def _call_owned(self, function, args, sources):
        outcome = job = None
        try:
            async with self._lock:
                if self._closed:
                    raise SourcesInvalidated('capture_closed')
                job = _AuxiliaryCall(function, args, sources)
                function = args = None
                outcome = await self._call_locked(self._invoke_auxiliary, job,
                                                  sources=sources or (), on_admit=job.admit)
                if outcome.error is _AuxiliaryError.INVALIDATED:
                    raise SourcesInvalidated(outcome.error.value)
                if outcome.error is not None:
                    raise RuntimeError(outcome.error.value)
                return outcome.value
        finally:
            if job is not None and not job.admitted:
                job.clear()
            function = args = outcome = job = None

    def _invoke_auxiliary(self, job):
        # No exception crosses the executor: its transport frames also own
        # argument tuples even after the callback's raw locals are cleared.
        result = None
        try:
            if self._closed or job.sources and self._privacy is not None and not self._privacy.sources_available(job.sources):
                raise SourcesInvalidated('capture_auxiliary_invalidated')
            result = job.function(*job.args)
            if self._closed or job.sources and self._privacy is not None and not self._privacy.sources_available(job.sources):
                raise SourcesInvalidated('capture_auxiliary_invalidated')
            return _AuxiliaryResult(result)
        except SourcesInvalidated:
            return _AuxiliaryResult(error=_AuxiliaryError.INVALIDATED)
        except Exception:
            return _AuxiliaryResult(error=_AuxiliaryError.FAILED)
        finally:
            job.clear()
            job = result = None

    def _advance(self, packets):
        event = None
        token = _CAPTURE_CHECK.set(self.require_live)
        try:
            for packet in packets:
                self.require_live()
                if not self._current_boundary().allows(packet.timestamp):
                    continue
                if self._oldest is not None and packet.timestamp < self._oldest:
                    continue
                self._process(packet.audio, packet.frames, packet.timestamp, None)
                self.require_live()
                self._last_packet = packet.timestamp
            if self._oldest is not None:
                self.require_live()
            event = self._poll()
            if event is not None:
                self.require_live()
                source = self._source
                if source is not None:
                    event.capture_id, event.conversation_id = source.capture_id, source.conversation_id
                    event.captured_at, event.source_expires_at = source.captured_at, source.expires_at
                self._reset()
            return event
        except SourcesInvalidated:
            self._reset()
            return None
        except Exception as error:
            logger.warning('Command capture reset after native failure: %s', type(error).__name__)
            self._reset()
            return None
        finally:
            _CAPTURE_CHECK.reset(token)
            packets = event = packet = None

    async def step(self, *, handoff: ProducerScope | None = None) -> CommandEvent | None:
        """Consume input, binding an exclusive receiver before raw result delivery."""
        if handoff is not None and not isinstance(handoff, ProducerScope):
            raise ValueError('invalid_capture_handoff')
        if self._closed:
            if handoff is not None:
                handoff.bind(())
            return None
        task = None
        with self._scopes.operation(()):
            try:
                task = self._start_controller(self._step_owned(handoff))
                return await task
            finally:
                task = None

    def _bind_step(self, sources, handoff):
        self._scopes.replace_sources(sources)
        if handoff is not None:
            handoff.bind(sources)

    async def prune_private(self) -> None:
        """Retire invalid idle state without consuming input or polling a command."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        self.publish_boundary()
        task = None
        sources = (self._source,) if self._source is not None else ()
        with self._scopes.operation(sources):
            try:
                task = self._start_controller(self._prune_owned())
                await task
            finally:
                task = None

    async def _prune_owned(self):
        async with self._lock:
            if self._closed:
                return
            if self._pending is not None:
                await asyncio.gather(asyncio.shield(self._pending), return_exceptions=True)
            self.publish_boundary()
            if self._failed or self._oldest is not None and not self.available():
                await self._call_locked(self._reset)

    async def _step_owned(self, handoff):
        packets = event = None
        try:
            async with self._lock:
                if self._closed or self._failed:
                    self._bind_step((), handoff)
                    return None
                if self._pending is not None:
                    await asyncio.gather(asyncio.shield(self._pending), return_exceptions=True)
                self._bind_step((self._source,) if self._source is not None else (), handoff)
                self.publish_boundary()
                if self._oldest is not None and not self.available():
                    await self._call_locked(self._reset)
                queue = self._queue
                packets = []
                while queue and len(packets) < 8:
                    # An idle reset owns its known epoch. Only acquiring new
                    # raw packets introduces input whose lineage is unresolved.
                    self._bind_step(None, handoff)
                    packet = queue.popleft()
                    if self._boundary.allows(packet.timestamp):
                        packets.append(packet)
                if self._last_packet is not None and packets and packets[0].timestamp - self._last_packet > .5:
                    await self._call_locked(self._reset)
                if self._source is not None and self._privacy is not None and not self._listening() and self._privacy.ledger.clock() - self._source.captured_at > 3600:
                    await self._call_locked(self._reset)
                if not packets and self._oldest is None and not self._listening():
                    self._bind_step((), handoff)
                    return None
                if self._failed:
                    self._bind_step((), handoff)
                    return None
                if packets and self._oldest is None:
                    self._oldest = min(packet.timestamp for packet in packets)
                    if self._privacy is not None:
                        try:
                            self._source = self._privacy.source_for_capture(
                                str(uuid4()), self._oldest, self.room_id, origin=MemoryOrigin.VOICE)
                        except (PermissionError, ValueError, RuntimeError):
                            self._oldest = None
                            self._bind_step((), handoff)
                            return None
                self._bind_step((self._source,) if self._source is not None else None, handoff)
                event = await self._call_locked(self._advance, packets)
                return event
        finally:
            packets = event = packet = queue = None

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual finite native scopes, including cancelled observers."""
        tasks = self._controllers | set(self._work._tasks)
        if self._pending is not None:
            tasks.add(self._pending)
        return tuple(tasks)

    def conversation_busy(self, conversation_id: str) -> bool:
        """Include actual callbacks, queued input and retained idle capture state."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        if self._failed:
            return True
        if self._scopes.busy(conversation_id, self._privacy.ledger):
            return True
        if self.callbacks.busy(conversation_id, self._privacy.ledger, self._current_boundary()):
            return True
        if self._oldest is not None or self._queue:
            return self._source is None or self._source.conversation_id == conversation_id
        return False

    def resource_status(self) -> dict:
        """Report counts without room, source identifiers, audio or text."""
        return {'closed': self._closed, 'reset_failed': self._failed, 'queued': len(self._queue), 'input_drops': self._input_drops,
                'native': self._work.status(), 'controllers': len(self._controllers),
                'source_scopes': self._scopes.resource_status(), 'has_state': self._oldest is not None,
                'callbacks': self.callbacks.status()}

    def begin_close(self) -> asyncio.Task:
        """Detach input and retain native reset after the last actual operation."""
        self._closed = True
        self._queue = deque(maxlen=32)
        callbacks = self.callbacks.begin_close()
        if self._closing is None:
            for task in self._controllers:
                task.cancel()
            async def finish():
                await asyncio.gather(*self._controllers, return_exceptions=True)
                async with self._lock:
                    delay = .05
                    while True:
                        await self._call_locked(self._reset)
                        if not self._failed:
                            break
                        await asyncio.sleep(delay)
                        delay = min(1., delay * 2)
                    await self._work.begin_close()
                await callbacks
            self._closing = asyncio.create_task(finish())
        return self._closing
