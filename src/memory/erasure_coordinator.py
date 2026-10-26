"""Retained cleanup ownership over the durable, versioned erasure journal."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import math

from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_journal import ErasureError, ErasureParticipant, ErasureResult, ErasureStep
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices

logger = logging.getLogger(__name__)
Cleanup = Callable[[ErasureStep], Awaitable[ErasureResult]]


@dataclass(frozen=True)
class CleanupDriver:
    """Explicit cleanup versions understood by one runtime participant."""

    callback: Cleanup
    versions: frozenset[int]


@dataclass(frozen=True)
class ActiveCleanup:
    """Keep both the exact durable claim and actual callback until acknowledged."""

    step: ErasureStep
    task: asyncio.Task


class ErasureCoordinator:
    """Reconcile bounded cleanup without treating observer cancellation as exit.

    Drivers must return only after their physical resources have finished. A
    missing driver/version remains blocked, including historical locations.
    The runtime does not expose broad erasure in the UI until every required
    participant is implemented and its producers have the same source policy.
    """

    def __init__(self, privacy: PrivacyServices, *, capacity: int = 4, close_timeout_s: float = 1):
        if type(capacity) is not int or not 1 <= capacity <= 32:
            raise ValueError("invalid_erasure_capacity")
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError("invalid_erasure_close_timeout")
        privacy.require_available()
        self.privacy, self.journal = privacy, privacy.ledger.erasures
        self._capacity, self._close_timeout_s = capacity, close_timeout_s
        self._drivers: dict[ErasureParticipant, CleanupDriver] = {}
        self._resources: list[Callable[[], asyncio.Task]] = []
        self._active: dict[tuple, ActiveCleanup] = {}
        self._cycle: asyncio.Task | None = None
        self._maintenance: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._closed = False
        self._wake = asyncio.Event()
        privacy.add_borrower(self.begin_close)

    def register(self, participant: ErasureParticipant, callback: Cleanup, *, versions: frozenset[int],
                 begin_close: Callable[[], asyncio.Task] | None = None) -> None:
        """Register a driver without pretending unavailable participants succeeded."""
        if self._closed:
            raise RuntimeError("erasure_closed")
        if not isinstance(participant, ErasureParticipant) or not callable(callback) or not versions or any(type(v) is not int or v < 1 for v in versions):
            raise ValueError("invalid_erasure_driver")
        if participant in self._drivers:
            raise ValueError("duplicate_erasure_driver")
        self._drivers[participant] = CleanupDriver(callback, frozenset(versions))
        if begin_close is not None:
            self._resources.append(begin_close)
        self._wake.set()

    async def request(self, conversation_id: str, actor: MemoryReviewer) -> dict:
        """Record atomic revocation/intent; broad completion requires all receipts."""
        if self._closed:
            raise RuntimeError("erasure_closed")
        receipt = await self.privacy.acall(self.journal.request, conversation_id, actor)
        self._wake.set()
        return receipt

    async def _harvest(self) -> None:
        for key, active in list(self._active.items()):
            if not active.task.done():
                continue
            result, error = None, None
            try:
                result = active.task.result()
                if not isinstance(result, ErasureResult):
                    result, error = None, ErasureError.INVALID_RESULT
            except asyncio.CancelledError:
                error = ErasureError.FAILED
            except Exception:
                error = ErasureError.FAILED
            # A failed acknowledgement keeps the completed task. Retry the
            # receipt, never rerun physical cleanup for the same live claim.
            await self.privacy.acall(self.journal.finish, active.step, result, error)
            del self._active[key]

    async def _reconcile(self) -> None:
        await self._harvest()
        if self._closed:
            return
        rows = await self.privacy.acall(self.journal.due, 128)
        for row in rows:
            if self._closed or len(self._active) >= self._capacity:
                break
            participant = ErasureParticipant(row["participant"])
            driver = self._drivers.get(participant)
            if driver is None or row["version"] not in driver.versions:
                await self.privacy.acall(self.journal.unavailable, row)
                continue
            step = await self.privacy.acall(self.journal.claim, row)
            if step is None:
                continue
            # The whole cycle is shielded, including the claim -> task handoff.
            # Closing waits for it before draining callbacks and their resources.
            async def cleanup(claim=step, callback=driver.callback):
                return await callback(claim)
            task = asyncio.create_task(cleanup())
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            self._active[(step.operation_id, step.participant, step.target_id)] = ActiveCleanup(step, task)
        await self._harvest()

    async def reconcile(self) -> None:
        """Observe one owned cycle; cancellation cannot orphan its durable claim."""
        if self._closed:
            raise RuntimeError("erasure_closed")
        if self._cycle is None or self._cycle.done():
            self._cycle = asyncio.create_task(self._reconcile())
            self._cycle.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        await asyncio.shield(self._cycle)

    def start(self) -> None:
        """Start one lightweight poller; slow native cleanup stays separately owned."""
        if self._closed:
            raise RuntimeError("erasure_closed")
        if self._maintenance is not None:
            return

        async def maintain():
            while not self._closed:
                self._wake.clear()
                try:
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("Erasure reconciliation deferred: %s", type(error).__name__)
                try:
                    async with asyncio.timeout(.25 if self._active else 1):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        self._maintenance = asyncio.create_task(maintain())

    def resource_status(self) -> dict:
        """Describe retained work without private content or storage locations."""
        return {"closed": self._closed, "active": len(self._active), "capacity": self._capacity,
                "cycle_running": self._cycle is not None and not self._cycle.done(),
                "closing": self._closing is not None and not self._closing.done()}

    def begin_close(self) -> asyncio.Task:
        """Close admission, then drain actual callbacks before their resources."""
        self._closed = True
        if self._closing is None:
            async def finish():
                if self._maintenance is not None:
                    self._maintenance.cancel()
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                if self._cycle is not None:
                    await asyncio.gather(self._cycle, return_exceptions=True)
                await asyncio.gather(*(a.task for a in self._active.values()), return_exceptions=True)
                try:
                    await self._harvest()
                except Exception as error:
                    # If authority admission has already closed, unacknowledged
                    # claims are safely retried after the next leased recovery.
                    logger.warning("Erasure receipts deferred until restart: %s", type(error).__name__)
                await asyncio.gather(*(close() for close in self._resources))
                self._active.clear()
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound close observation while retaining the actual shared-lease drain."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout("erasure_resources_still_running") from error
        closing.result()
        if cancelled:
            raise asyncio.CancelledError
