"""A serialized native owner for one room's VAD, preroll and pending segments."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import logging

from src.core.owned_work import OwnedBlockingWork
from src.memory.producer_context import producer_context
from src.memory.sources import CaptureBoundary
from src.memory.fact_ledger import FactLedger, MemorySource
from src.ambient.tap import TapPacket

logger = logging.getLogger(__name__)


class RoomProducerError(StrEnum):
    FEED = "ambient_segmentation_failed"
    RESET = "ambient_segmenter_reset_failed"
    NATIVE = "ambient_room_native_failed"


@dataclass
class _RoomCall:
    function: Callable | None
    args: tuple
    admitted: bool = False
    outcome: object = None
    receiver_retired: bool = False

    def clear(self):
        self.function, self.args = None, ()

    def publish(self, outcome):
        self.outcome = outcome
        if self.receiver_retired:
            outcome.clear()

    def retire_receiver(self):
        self.receiver_retired = True
        if self.outcome is not None:
            self.outcome.clear()
            self.outcome = None


@dataclass
class _RoomOutcome:
    value: object = None
    error: RoomProducerError | None = None

    def clear(self):
        self.value = None


def _invoke_room(call):
    outcome = None
    try:
        try:
            outcome = _RoomOutcome(call.function(*call.args))
        except (Exception, asyncio.CancelledError):
            outcome = _RoomOutcome(error=RoomProducerError.NATIVE)
        call.publish(outcome)
        return outcome
    finally:
        call.clear()
        call = outcome = None


class RoomProducer:
    """Serialize actual inference/reset and retain it after observer cancellation."""

    def __init__(
        self, room_id: str, factory, *, require_cleanup: bool = False, boundary_fn=None,
        source_fn: Callable[[str, float], MemorySource] | None = None,
        epoch_available: Callable[[object, str], bool] | None = None,
    ):
        if source_fn is not None and not callable(epoch_available):
            raise ValueError('ambient_admission_validator_required')
        self.room_id, self._factory = room_id, factory
        self._require_cleanup = require_cleanup
        self._segmenter = None
        self._oldest = None
        self._boundary = CaptureBoundary(None, float("-inf"))
        self.ready = []
        self._work = OwnedBlockingWork(worker=self._offload, capacity=1)
        self._lock = asyncio.Lock()
        self._pending = None
        self._batches = set()
        self._closing = None
        self._closed = False
        self._error = None
        self._boundary_fn = boundary_fn
        self._source_fn = source_fn
        self._source: MemorySource | None = None
        self._reset_pending = False
        self._epoch_available = epoch_available
        self._epoch = None

    def publish_boundary(self, boundary: CaptureBoundary) -> None:
        """Publish metadata; native work observes it before and after each chunk."""
        self._boundary = self._boundary.tighten(boundary)
        boundary = self._boundary
        self.ready[:] = [
            segment for segment in self.ready if boundary.allows(segment.t0)
            and self._admitted(segment.input_epoch)
        ]

    def needs_prune(self) -> bool:
        """Report a stale retained native state without reading private arrays."""
        return self._reset_pending or getattr(self._segmenter, 'needs_reset', False) is True or (
            self._oldest is not None and not self._boundary.allows(self._oldest)) or (
            self._source is not None and not self._boundary.allows(self._source.captured_at))

    def _discard(self):
        segmenter = self._segmenter
        if segmenter is not None:
            try:
                reset = getattr(segmenter, "reset", None)
                if callable(reset):
                    reset()
            except Exception as error:
                self._reset_pending = True
                self._error = RoomProducerError.RESET
                logger.warning("Ambient state reset deferred: %s", type(error).__name__)
                return False
        if self._reset_pending:
            self._error = None
        self._segmenter = self._oldest = self._source = self._epoch = None
        self._reset_pending = False
        return True

    def _prune(self):
        boundary = self._current_boundary()
        if self._reset_pending or getattr(self._segmenter, 'needs_reset', False) is True or (
            self._oldest is not None and not boundary.allows(self._oldest)) or (
            self._source is not None and not boundary.allows(self._source.captured_at)) or (
            self._epoch is not None and not self._admitted(self._epoch)):
            return self._discard()
        return True

    def _admitted(self, epoch):
        return self._epoch_available is None or self._epoch_available(epoch, self.room_id)

    def _current_boundary(self):
        if self._boundary_fn is not None:
            return self._boundary.tighten(self._boundary_fn(self.room_id))
        return self._boundary

    def _ensure_segmenter(self):
        if self._segmenter is None:
            segmenter = self._factory()
            if self._require_cleanup and (
                not callable(getattr(segmenter, "reset", None))
                or not hasattr(segmenter, "oldest_timestamp")
            ):
                raise ValueError("unsupported_segmenter_cleanup")
            self._segmenter = segmenter

    def _feed(self, items):
        segments = []
        try:
            if self._closed:
                return []
            if not self._prune():
                return []
            for item in items:
                if isinstance(item, TapPacket):
                    timestamp, chunk, tts_active, epoch = item.timestamp, item.audio, item.tts_active, item.epoch
                else:
                    timestamp, chunk, tts_active = item
                    epoch = None
                if self._closed:
                    segments.clear()
                    self._discard()
                    break
                if not self._prune():
                    segments.clear()
                    break
                if not self._admitted(epoch) or not self._current_boundary().allows(timestamp):
                    continue
                if self._source is None and self._source_fn is not None:
                    self._source = self._source_fn(self.room_id, timestamp)
                    self._epoch = epoch
                if not self._admitted(epoch):
                    self._discard()
                    continue
                self._ensure_segmenter()
                if not self._prune():
                    segments.clear()
                    break
                if self._closed or not self._current_boundary().allows(timestamp):
                    continue
                if self._segmenter is None:
                    continue  # A fence changed during factory/reset; bridge on a fresh chunk.
                self._oldest = (
                    timestamp if self._oldest is None else min(timestamp, self._oldest)
                )
                input_source = self._source
                input_epoch = self._epoch
                produced = self._segmenter.feed(
                    ts=timestamp, chunk=chunk, tts_active=tts_active
                )
                if not self._admitted(epoch):
                    segments.clear()
                    self._discard()
                    break
                for segment in produced:
                    segment.input_source = input_source
                    segment.input_epoch = input_epoch
                self._oldest = getattr(
                    self._segmenter, "oldest_timestamp", self._oldest
                )
                if self._oldest is None:
                    self._source = self._epoch = None
                segments.extend(produced)
                if not self._prune():
                    segments.clear()
                    break
            if not self._reset_pending:
                self._error = None
            boundary = self._current_boundary()
            return [
                segment
                for segment in segments
                if not self._closed and boundary.allows(segment.t0)
            ]
        except Exception as error:
            self._error = RoomProducerError.FEED
            self._discard()
            logger.warning("Ambient segmentation deferred: %s", type(error).__name__)
            return []

    async def _offload(self, function, call):
        call.admitted = True
        try:
            return await asyncio.to_thread(function, call)
        except Exception:
            call.clear()
            return _RoomOutcome(error=RoomProducerError.NATIVE)
        finally:
            function = call = None

    async def _call(self, function, *args, closing=False):
        task = call = outcome = None
        try:
            async with self._lock:
                if self._pending is not None:
                    await asyncio.gather(
                        asyncio.shield(self._pending), return_exceptions=True
                    )
                if self._closed and not closing:
                    raise RuntimeError("ambient_room_closed")
                with producer_context(self.room_id):
                    call = _RoomCall(function, args)
                    task = asyncio.create_task(self._work.run(_invoke_room, call))
                self._pending = task

                def finished(done):
                    if self._pending is done:
                        self._pending = None
                    if not done.cancelled():
                        done.exception()

                task.add_done_callback(finished)
                outcome = await asyncio.shield(task)
                if outcome.error is not None:
                    self._error = outcome.error
                    self._reset_pending = self._segmenter is not None
                    raise RuntimeError(outcome.error.value)
                return outcome.value
        finally:
            if call is not None:
                # The transport may keep its future/result after actual native
                # exit. Its carrier must become empty on handoff or abandonment.
                call.retire_receiver()
            if call is not None and not call.admitted:
                # A queued owned task may not have started its transport yet.
                # Its owner, not a cancelled observer, must retire that box.
                if task is None or task.done():
                    call.clear()
            task, function, args, call, outcome = None, None, (), None, None

    async def feed(self, items) -> None:
        """Feed one bounded tap batch and publish only currently admissible segments."""
        task = None
        try:
            if self._closed:
                raise RuntimeError("ambient_room_closed")
            if self._batches:
                raise RuntimeError("ambient_room_batch_capacity")
            with producer_context(self.room_id):
                task = asyncio.create_task(self._accept_feed(items))
            self._batches.add(task)
            task.add_done_callback(self._batches.discard)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    task.cancel()
                    raise
        finally:
            task = items = None

    async def _accept_feed(self, items):
        produced = None
        try:
            produced = await self._call(self._feed, items)
            self.ready.extend(
                segment
                for segment in produced
                if self._boundary.allows(segment.t0) and not self._closed
                and self._admitted(segment.input_epoch)
            )
        finally:
            items = produced = None

    async def prune(self) -> None:
        """Wait for real VAD completion before retiring stale native buffers."""
        await self._call(self._prune)

    async def discard_private(self) -> bool:
        """Reset actual VAD/preroll after its worker exits, keeping it reusable."""
        self.ready.clear()
        return await self._call(self._discard)

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Observe actual VAD/reset work and its shielded native completion."""
        tasks = set(self._work._tasks)
        tasks.update(self._batches)
        if self._pending is not None:
            tasks.add(self._pending)
        if self._closing is not None and not self._closing.done():
            tasks.add(self._closing)
        return tuple(tasks)

    def status(self) -> dict:
        """Return counts and fixed failures, never room IDs, timestamps or audio."""
        return {
            "closed": self._closed,
            "ready": len(self.ready),
            "native_operations": self._work.status()["operations"],
            "batches": len(self._batches),
            "handoff_pending": self._pending is not None,
            "state_retained": self._segmenter is not None,
            "input_state_retained": self._oldest is not None,
            "error": self._error.value if self._error is not None else None,
            "reset_pending": self._reset_pending,
        }

    def conversation_busy(self, conversation_id: str, ledger: FactLedger) -> bool:
        """Retain reset failures and known idle VAD inputs until actual cleanup."""
        if self._reset_pending or getattr(self._segmenter, 'needs_reset', False) is True:
            return True
        for segment in self.ready:
            source = segment.input_source
            if source is None or source.conversation_id == conversation_id or ledger.dependencies_committed_unavailable(source.fact_dependencies):
                return True
        if self._oldest is None:
            return False
        return self._source is None or self._source.conversation_id == conversation_id or ledger.dependencies_committed_unavailable(self._source.fact_dependencies)

    def begin_close(self) -> asyncio.Task:
        """Detach ready audio and retain the actual native reset before exit."""
        self._closed = True
        self.ready.clear()
        if self._closing is None:
            for task in self._batches:
                task.cancel()

            async def finish():
                await asyncio.gather(*self._batches, return_exceptions=True)
                delay = .05
                while True:
                    try:
                        if await self._call(self._discard, closing=True):
                            break
                    except Exception as error:
                        logger.warning('Ambient room close deferred: %s', type(error).__name__)
                    await asyncio.sleep(delay)
                    delay = min(1., delay * 2)
                await self._work.begin_close()

            self._closing = asyncio.create_task(finish())
        return self._closing
