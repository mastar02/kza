"""Always-on, bounded proposal TTL maintenance over registered historical sidecars."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import logging
import math

from src.analytics.suggestion_sweep import (
    SuggestionSweep,
    SweepPosition,
    sweep_suggestions,
)
from src.analytics.suggestion_storage import (
    RUNTIME_TARGET,
    validate_suggestion_descriptor,
)
from src.analytics.event_storage import validate_event_descriptor
from src.analytics.event_evidence import event_target_id
from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices

logger = logging.getLogger(__name__)


class SuggestionRetentionError(StrEnum):
    FAILED = "suggestion_retention_failed"
    UNSUPPORTED = "unsupported_suggestion_retention_target"


@dataclass(frozen=True)
class SweepOutcome:
    """An actual result cannot retain a private exception or traceback."""

    receipt: SuggestionSweep | None = None
    error: SuggestionRetentionError | None = None


class SuggestionRetention:
    """Retain two database batches with their own native pool and deadline wakeups."""

    def __init__(
        self, privacy: PrivacyServices, *, runtime_sweep, retention_days: float = 90
    ):
        if (
            type(retention_days) not in (int, float)
            or not math.isfinite(retention_days)
            or retention_days <= 0
        ):
            raise ValueError("invalid_event_retention")
        self.privacy, self._runtime_sweep = privacy, runtime_sweep
        self.retention_days = retention_days
        self._event_targets = {}
        self._runtime_task = None
        self._runtime_due = 0.0
        self._runtime_dirty = False
        self._runtime_status = {}
        self._pending_counts = {}
        self._registry_task = None
        self._registry_due = 0.0
        self._registry_error = None
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="suggestion-retention"
        )
        self._work = OwnedBlockingWork(worker=self._offload)
        self._targets: dict[str, dict] = {}
        self._active: dict[str, asyncio.Task] = {}
        self._due: dict[str, float] = {}
        self._cursors: dict[str, SweepPosition | None] = {}
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

    def notify(self, database: str | None = None) -> None:
        """Wake a newly earlier deadline without letting shutdown break a write."""
        if self._closed or self._loop is None or self._loop.is_closed():
            return

        def changed():
            if database is not None:
                self._dirty.add(database)
            self._registry_due = 0
            self._runtime_dirty = True
            self._runtime_due = 0
            self._wake.set()

        try:
            self._loop.call_soon_threadsafe(changed)
        except RuntimeError:
            logger.debug("Suggestion retention wake deferred during loop shutdown")

    async def _read_registry(self) -> tuple:
        try:

            def read():
                rows = self.privacy.ledger.erasures.targets(
                    ErasureParticipant.SUGGESTIONS
                )
                events = self.privacy.ledger.erasures.targets(ErasureParticipant.EVENTS)
                return self.privacy.ledger.clock(), (rows, events)

            return await self.privacy.acall(read)
        except Exception:
            return self.privacy.ledger.clock(), None

    def _accept_registry(self, checked_at, targets) -> None:
        self._registry_due = checked_at + 30
        self._registry_error = (
            None if targets is not None else "suggestion_registry_unavailable"
        )
        if targets is None:
            return
        targets, events = targets
        for target in events:
            try:
                if target["version"] != 1:
                    continue
                validate_event_descriptor(target["descriptor"])
                self._event_targets[event_target_id(target["descriptor"])] = target[
                    "descriptor"
                ]
            except ValueError:
                continue
        for target in targets:
            key = target["id"]
            if target["version"] == 1 and target["descriptor"] == RUNTIME_TARGET:
                continue
            try:
                if target["version"] != 1:
                    raise ValueError("unsupported_suggestion_target")
                validate_suggestion_descriptor(target["descriptor"])
            except ValueError:
                self._status[key] = {
                    "id": key,
                    "error": SuggestionRetentionError.UNSUPPORTED.value,
                }
                continue
            self._targets[key] = target
            self._due.setdefault(key, 0)

    async def refresh(self) -> None:
        """Load known roots before optional writers can hold the source ledger."""
        checked_at, targets = await self._read_registry()
        self._accept_registry(checked_at, targets)
        if targets is None:
            raise RuntimeError("suggestion_registry_unavailable")

    async def _run_runtime(self) -> dict:
        try:
            return await self._runtime_sweep()
        except Exception:
            return {
                "error": "suggestion_runtime_retention_failed",
                "next_at": self.privacy.ledger.clock() + 30,
            }

    async def _run(self, target: dict, cursor) -> SweepOutcome:
        try:
            receipt = await self._work.run(
                sweep_suggestions,
                self.privacy.ledger,
                target["descriptor"],
                cursor,
                dict(self._event_targets),
                self.retention_days,
            )
            return SweepOutcome(receipt)
        except Exception:
            return SweepOutcome(error=SuggestionRetentionError.FAILED)

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
            self._cursors[key] = receipt.cursor
            self._pending_counts[key] = (
                self._pending_counts.get(key, 0) + receipt.pending
            )
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
                "error": "unknown_suggestion_entries"
                if self._pending_counts.get(key, 0)
                else None,
                "checked_at": receipt.checked_at,
                "next_at": self._due[key],
                "cursor_pending": receipt.cursor is not None,
                "deleted": receipt.deleted,
                "pending": self._pending_counts.get(key, 0),
            }
        if self._runtime_task is not None and self._runtime_task.done():
            outcome = self._runtime_task.result()
            self._runtime_task = None
            self._runtime_status = outcome
            self._runtime_due = 0 if self._runtime_dirty else outcome["next_at"]
        if self._closed:
            return
        if self._runtime_task is None and (
            self._runtime_dirty or self._runtime_due <= self.privacy.ledger.clock()
        ):
            self._runtime_dirty = False
            self._runtime_task = asyncio.create_task(self._run_runtime())
        if self._registry_task is not None and self._registry_task.done():
            checked_at, targets = self._registry_task.result()
            self._registry_task = None
            self._accept_registry(checked_at, targets)
        if (
            self._registry_task is None
            and self._registry_due <= self.privacy.ledger.clock()
        ):
            self._registry_task = asyncio.create_task(self._read_registry())
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
            if self._cursors.get(key) is None:
                self._pending_counts[key] = 0
                self._dirty.discard(target["descriptor"]["path"])
            self._sequence += 1
            self._last_started[key] = self._sequence
            self._active[key] = asyncio.create_task(
                self._run(target, self._cursors.get(key))
            )

    async def reconcile(self) -> None:
        """Observe a shielded batch handoff without cancelling actual native work."""
        if self._closed:
            raise RuntimeError("suggestion_retention_closed")
        if self._cycle is None or self._cycle.done():
            self._cycle = asyncio.create_task(self._reconcile())
            self._cycle.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        await asyncio.shield(self._cycle)

    def start(self) -> None:
        """Start independently of proposal producers, analytics and semantic recall."""
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
                    logger.warning(
                        "Suggestion retention deferred: %s", type(error).__name__
                    )
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
                delay = (
                    min(
                        delay,
                        max(0.01, self._runtime_due - self.privacy.ledger.clock()),
                    )
                    if self._runtime_task is None
                    else 0.05
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
        """Return opaque locations and fixed codes, never paths or proposal text."""
        return {
            "closed": self._closed,
            "active": len(self._active),
            "targets": list(self._status.values()),
            "runtime": dict(self._runtime_status),
            "runtime_active": self._runtime_task is not None,
            "registry_pending": self._registry_task is not None,
            "registry_error": self._registry_error,
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
                if self._runtime_task is not None:
                    await asyncio.gather(self._runtime_task, return_exceptions=True)
                if self._registry_task is not None:
                    await asyncio.gather(self._registry_task, return_exceptions=True)
                await self._work.begin_close()
                await asyncio.to_thread(self._executor.shutdown, wait=True)

            self._closing = asyncio.create_task(finish())
        return self._closing
