"""Always-on deletion reconciliation without loading a semantic encoder."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging

from src.memory.index_deletions import IndexCleanupError, IndexDeletionBatch
from src.memory.privacy_services import PrivacyServices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexCleanupOutcome:
    """A retained native result cannot contain exception text or a traceback."""

    error: IndexCleanupError | None = None


class IndexMaintenance:
    """Own two target attempts, exclusive claims and receipt-only retries."""

    def __init__(self, privacy: PrivacyServices, clean_batch: Callable[[IndexDeletionBatch], Awaitable[None]]):
        self.privacy, self._clean_batch = privacy, clean_batch
        self.queue = privacy.ledger.index_deletions
        self._active: dict[str, tuple[IndexDeletionBatch, asyncio.Task]] = {}
        self._cycle: asyncio.Task | None = None
        self._maintenance: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._status: list[dict] = []
        self._next_at = 0.0

    def notify(self) -> None:
        """Accept ledger-worker wakeups without letting shutdown break a commit."""
        if not self._closed and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._wake.set)

    async def _run(self, batch: IndexDeletionBatch) -> IndexCleanupOutcome:
        if batch.version != 1:
            return IndexCleanupOutcome(IndexCleanupError.UNSUPPORTED)
        try:
            await self._clean_batch(batch)
            return IndexCleanupOutcome()
        except Exception:
            return IndexCleanupOutcome(IndexCleanupError.FAILED)

    async def _reconcile(self) -> None:
        for target_id, (batch, task) in list(self._active.items()):
            if task.done():
                outcome = task.result()
                # If this commit fails, keep the completed native outcome and
                # retry only its receipt; never abandon or duplicate the claim.
                await self.privacy.acall(self.queue.finish, batch, outcome.error)
                del self._active[target_id]
        if self._closed:
            return
        await self.privacy.acall(self.privacy.ledger.expire)
        targets = await self.privacy.acall(self.queue.pending_targets)
        now = self.privacy.ledger.clock()
        for target in sorted(targets, key=lambda row: row['due'] if row['due'] is not None else float('inf')):
            if self._closed or len(self._active) >= 2:
                break
            if target['target_id'] in self._active or target['due'] is None or target['due'] > now:
                continue
            batch = await self.privacy.acall(self.queue.claim, target['target_id'])
            if batch is not None:
                self._active[batch.target_id] = (batch, asyncio.create_task(self._run(batch)))
        self._status = targets
        def next_deadline():
            checked_at = self.privacy.ledger.clock()
            return checked_at+min(30, self.privacy.ledger.next_maintenance_delay())
        deadline = await self.privacy.acall(next_deadline)
        deadlines = [row['due'] for row in targets if row['due'] is not None and row['target_id'] not in self._active]
        self._next_at = min([deadline, *deadlines])

    async def reconcile(self) -> None:
        """Observe a shielded scheduling/claim handoff cycle."""
        if self._closed:
            raise RuntimeError('index_maintenance_closed')
        if self._cycle is None or self._cycle.done():
            self._cycle = asyncio.create_task(self._reconcile())
            self._cycle.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        await asyncio.shield(self._cycle)

    def start(self) -> None:
        """Wake at catalog expiry or deletion without enabling semantic recall."""
        if self._maintenance is not None or self._closed:
            return
        self._loop = asyncio.get_running_loop()
        self.queue.set_listener(self.notify)
        async def maintain():
            while not self._closed:
                self._wake.clear()
                failed = False
                try:
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning('Index cleanup deferred: %s', type(error).__name__)
                    failed = True
                delay = 30 if failed else max(.01, min(30, self._next_at-self.privacy.ledger.clock()))
                if not failed and self._active:
                    delay = .05
                try:
                    async with asyncio.timeout(delay):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        self._maintenance = asyncio.create_task(maintain())

    def status(self) -> dict:
        """Return counters and opaque target IDs without index paths or documents."""
        return {'closed': self._closed, 'active': len(self._active), 'targets': list(self._status)}

    def begin_close(self) -> asyncio.Task:
        """Close scheduling, then retain actual native attempts for the writer lease."""
        self._closed = True
        if self._closing is None:
            async def finish():
                if self._maintenance is not None:
                    self._maintenance.cancel()
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                if self._cycle is not None:
                    await asyncio.gather(self._cycle, return_exceptions=True)
                await asyncio.gather(*(task for _, task in self._active.values()), return_exceptions=True)
                deferred = 0
                for batch, task in self._active.values():
                    try:
                        await self.privacy.acall(self.queue.finish, batch, task.result().error)
                    except Exception:
                        # Native completion remains true; recovery retries any
                        # unacknowledged claim after the next leased startup.
                        deferred += 1
                if deferred:
                    logger.debug('Index cleanup receipts deferred to leased recovery: %d', deferred)
                self._active.clear()
            self._closing = asyncio.create_task(finish())
        return self._closing
