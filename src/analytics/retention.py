"""Always-on, bounded event TTL maintenance over registered historical databases."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import logging
import math

from src.analytics.event_storage import (
    EventSweep,
    sweep_events,
    validate_event_descriptor,
)
from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices

logger = logging.getLogger(__name__)


class EventRetentionError(StrEnum):
    FAILED = "event_retention_failed"
    UNSUPPORTED = "unsupported_event_retention_target"


@dataclass(frozen=True)
class SweepOutcome:
    """An actual result cannot retain a private exception or traceback."""

    receipt: EventSweep | None = None
    error: EventRetentionError | None = None


class EventRetention:
    """Retain two database batches with their own native pool and deadline wakeups."""

    def __init__(self, privacy: PrivacyServices, *, retention_days: float = 90):
        if (
            type(retention_days) not in (int, float)
            or not math.isfinite(retention_days)
            or retention_days <= 0
        ):
            raise ValueError("invalid_event_retention")
        self.privacy, self.retention_days = privacy, retention_days
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="event-retention"
        )
        self._work = OwnedBlockingWork(worker=self._offload)
        self._targets: dict[str, dict] = {}
        self._active: dict[str, asyncio.Task] = {}
        self._due: dict[str, float] = {}
        self._cursors: dict[str, int] = {}
        self._deadlines: dict[str, float] = {}
        self._attempts: dict[str, int] = {}
        self._sequence = 0
        self._last_started: dict[str, int] = {}
        self._status: dict[str, dict] = {}
        self._dirty: set[str] = set()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cycle: asyncio.Task | None = None
        self._maintenance: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._closed = False

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, partial(function, *args, **kwargs)
        )

    def notify(self, database: str) -> None:
        """Wake a newly earlier deadline without letting shutdown break a write."""
        if self._closed or self._loop is None or self._loop.is_closed():
            return

        def changed():
            self._dirty.add(database)
            self._wake.set()

        try:
            self._loop.call_soon_threadsafe(changed)
        except RuntimeError:
            logger.debug("Event retention wake deferred during loop shutdown")

    async def _run(self, target: dict, cursor: int) -> SweepOutcome:
        try:
            receipt = await self._work.run(
                sweep_events,
                target["descriptor"],
                cursor,
                self.privacy.ledger,
                self.retention_days,
            )
            return SweepOutcome(receipt)
        except Exception:
            return SweepOutcome(error=EventRetentionError.FAILED)

    async def _reconcile(self) -> None:
        now = self.privacy.ledger.clock()
        for key, task in list(self._active.items()):
            if not task.done():
                continue
            outcome = task.result()
            del self._active[key]
            if outcome.error is not None:
                self._attempts[key] = self._attempts.get(key, 0) + 1
                self._due[key] = now + min(60, 2 ** min(self._attempts[key] - 1, 6))
                self._status[key] = {
                    "id": key,
                    "error": outcome.error.value,
                    "next_at": self._due[key],
                }
                continue
            receipt = outcome.receipt
            self._attempts[key] = 0
            self._cursors[key] = receipt.cursor or 0
            if receipt.next_at is not None:
                self._deadlines[key] = min(
                    self._deadlines.get(key, float("inf")), receipt.next_at
                )
            if receipt.cursor is not None:
                self._due[key] = receipt.checked_at
            else:
                deadline = self._deadlines.pop(key, receipt.checked_at + 30)
                self._due[key] = min(receipt.checked_at + 30, deadline)
                if self._targets[key]["descriptor"]["path"] in self._dirty:
                    self._due[key] = 0
            self._status[key] = {
                "id": key,
                "error": None,
                "checked_at": receipt.checked_at,
                "next_at": self._due[key],
                "cursor_pending": receipt.cursor is not None,
                "deleted": receipt.deleted,
                "scrubbed": receipt.scrubbed,
            }
        if self._closed:
            return
        targets = await self.privacy.acall(
            self.privacy.ledger.erasures.targets, ErasureParticipant.EVENTS
        )
        for target in targets:
            key = target["id"]
            try:
                if target["version"] != 1:
                    raise ValueError("unsupported_event_target")
                validate_event_descriptor(target["descriptor"])
            except ValueError:
                self._status[key] = {
                    "id": key,
                    "error": EventRetentionError.UNSUPPORTED.value,
                }
                continue
            self._targets[key] = target
            self._due.setdefault(key, 0)
        now = self.privacy.ledger.clock()
        for key, target in sorted(
            self._targets.items(),
            key=lambda item: (self._due[item[0]], self._last_started.get(item[0], 0)),
        ):
            if self._closed or len(self._active) >= 2:
                break
            if key in self._active:
                continue
            dirty = target["descriptor"]["path"] in self._dirty
            if self._due[key] > now and not (dirty and not self._attempts.get(key)):
                continue
            if self._cursors.get(key, 0) == 0:
                self._dirty.discard(target["descriptor"]["path"])
            self._sequence += 1
            self._last_started[key] = self._sequence
            self._active[key] = asyncio.create_task(
                self._run(target, self._cursors.get(key, 0))
            )

    async def reconcile(self) -> None:
        """Observe a shielded batch handoff without cancelling actual native work."""
        if self._closed:
            raise RuntimeError("event_retention_closed")
        if self._cycle is None or self._cycle.done():
            self._cycle = asyncio.create_task(self._reconcile())
            self._cycle.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        await asyncio.shield(self._cycle)

    def start(self) -> None:
        """Start independently of event producers, analytics and semantic recall."""
        if self._closed or self._maintenance is not None:
            return
        self._loop = asyncio.get_running_loop()

        async def maintain():
            while not self._closed:
                self._wake.clear()
                failed = False
                try:
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("Event retention deferred: %s", type(error).__name__)
                    failed = True
                delay = (
                    30
                    if failed
                    else max(
                        0.01,
                        min(
                            30,
                            min(
                                (
                                    due - self.privacy.ledger.clock()
                                    for key, due in self._due.items()
                                    if key not in self._active
                                ),
                                default=30,
                            ),
                        ),
                    )
                )
                if self._active and not failed:
                    delay = 0.05
                try:
                    async with asyncio.timeout(delay):
                        await self._wake.wait()
                except TimeoutError:
                    pass

        self._maintenance = asyncio.create_task(maintain())

    def status(self) -> dict:
        """Return opaque locations and fixed codes, never paths or event documents."""
        return {
            "closed": self._closed,
            "active": len(self._active),
            "targets": list(self._status.values()),
        }

    def begin_close(self) -> asyncio.Task:
        """Retain actual batches, then drain their native executor before lease exit."""
        self._closed = True
        if self._closing is None:

            async def finish():
                if self._maintenance is not None:
                    self._maintenance.cancel()
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                if self._cycle is not None:
                    await asyncio.gather(self._cycle, return_exceptions=True)
                await asyncio.gather(*self._active.values(), return_exceptions=True)
                self._active.clear()
                await self._work.begin_close()
                await asyncio.to_thread(self._executor.shutdown, wait=True)

            self._closing = asyncio.create_task(finish())
        return self._closing
