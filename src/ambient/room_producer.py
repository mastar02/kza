"""A serialized native owner for one room's VAD, preroll and pending segments."""

import asyncio
from enum import StrEnum
import logging

from src.core.owned_work import OwnedBlockingWork
from src.memory.producer_context import producer_context
from src.memory.sources import CaptureBoundary

logger = logging.getLogger(__name__)


class RoomProducerError(StrEnum):
    FEED = "ambient_segmentation_failed"
    RESET = "ambient_segmenter_reset_failed"


class RoomProducer:
    """Serialize actual inference/reset and retain it after observer cancellation."""

    def __init__(
        self, room_id: str, factory, *, require_cleanup: bool = False, boundary_fn=None
    ):
        self.room_id, self._factory = room_id, factory
        self._require_cleanup = require_cleanup
        self._segmenter = None
        self._oldest = None
        self._boundary = CaptureBoundary(None, float("-inf"))
        self.ready = []
        self._work = OwnedBlockingWork(capacity=1)
        self._lock = asyncio.Lock()
        self._pending = None
        self._batches = set()
        self._closing = None
        self._closed = False
        self._error = None
        self._boundary_fn = boundary_fn

    def publish_boundary(self, boundary: CaptureBoundary) -> None:
        """Publish metadata; native work observes it before and after each chunk."""
        self._boundary = self._boundary.tighten(boundary)
        boundary = self._boundary
        self.ready[:] = [
            segment for segment in self.ready if boundary.allows(segment.t0)
        ]

    def needs_prune(self) -> bool:
        """Report a stale retained native state without reading private arrays."""
        return self._oldest is not None and not self._boundary.allows(self._oldest)

    def _discard(self):
        segmenter, self._segmenter = self._segmenter, None
        self._oldest = None
        if segmenter is not None:
            try:
                reset = getattr(segmenter, "reset", None)
                if callable(reset):
                    reset()
            except Exception as error:
                self._error = RoomProducerError.RESET
                logger.warning("Ambient state reset deferred: %s", type(error).__name__)

    def _prune(self):
        if self._oldest is not None and not self._current_boundary().allows(
            self._oldest
        ):
            self._discard()

    def _current_boundary(self):
        if self._boundary_fn is not None:
            return self._boundary.tighten(self._boundary_fn(self.room_id))
        return self._boundary

    def _ensure_segmenter(self):
        if self._segmenter is None:
            self._segmenter = self._factory()
            if self._require_cleanup and (
                not callable(getattr(self._segmenter, "reset", None))
                or not hasattr(self._segmenter, "oldest_timestamp")
            ):
                raise ValueError("unsupported_segmenter_cleanup")

    def _feed(self, items):
        segments = []
        try:
            if self._closed:
                return []
            self._prune()
            for timestamp, chunk, tts_active in items:
                if self._closed:
                    segments.clear()
                    self._discard()
                    break
                self._prune()
                if not self._current_boundary().allows(timestamp):
                    continue
                self._ensure_segmenter()
                self._prune()
                if self._closed or not self._current_boundary().allows(timestamp):
                    continue
                if self._segmenter is None:
                    continue  # A fence changed during factory/reset; bridge on a fresh chunk.
                self._oldest = (
                    timestamp if self._oldest is None else min(timestamp, self._oldest)
                )
                produced = self._segmenter.feed(
                    ts=timestamp, chunk=chunk, tts_active=tts_active
                )
                self._oldest = getattr(
                    self._segmenter, "oldest_timestamp", self._oldest
                )
                segments.extend(produced)
                self._prune()
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

    async def _call(self, function, *args, closing=False):
        task = None
        try:
            async with self._lock:
                if self._pending is not None:
                    await asyncio.gather(
                        asyncio.shield(self._pending), return_exceptions=True
                    )
                if self._closed and not closing:
                    raise RuntimeError("ambient_room_closed")
                with producer_context(self.room_id):
                    task = asyncio.create_task(self._work.run(function, *args))
                self._pending = task

                def finished(done):
                    if self._pending is done:
                        self._pending = None
                    if not done.cancelled():
                        done.exception()

                task.add_done_callback(finished)
                return await asyncio.shield(task)
        finally:
            task, function, args = None, None, ()

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
            )
        finally:
            items = produced = None

    async def prune(self) -> None:
        """Wait for real VAD completion before retiring stale native buffers."""
        await self._call(self._prune)

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
        }

    def begin_close(self) -> asyncio.Task:
        """Detach ready audio and retain the actual native reset before exit."""
        self._closed = True
        self.ready.clear()
        if self._closing is None:
            for task in self._batches:
                task.cancel()

            async def finish():
                await asyncio.gather(*self._batches, return_exceptions=True)
                await self._call(self._discard, closing=True)
                await self._work.begin_close()

            self._closing = asyncio.create_task(finish())
        return self._closing
