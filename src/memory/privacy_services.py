"""Always-on privacy authority, independent of semantic memory availability."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import copy_context
from functools import partial
import fcntl
import logging
import os
from pathlib import Path
import threading
from weakref import WeakMethod

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource
from src.memory.sources import CaptureScopes

logger = logging.getLogger(__name__)


class PrivacyServices:
    """Own the sole privacy writer until its native users actually finish."""

    def __init__(self, path: str, *, clock=None, automatic_days: float = 30,
                 close_timeout_s: float = 1):
        self.ledger = FactLedger(path, clock=clock, automatic_days=automatic_days)
        self.scopes = CaptureScopes(self.ledger, admission_started_at=self.ledger.clock())
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kza-privacy")
        self._work = OwnedBlockingWork(worker=self._offload, close_timeout_s=close_timeout_s)
        self._close_timeout_s = close_timeout_s
        self._lease: int | None = None
        self._lock = threading.RLock()
        self._borrowers: list[WeakMethod] = []
        self._closed = False
        self._ready = False
        self._closing: asyncio.Task | None = None
        self.maintenance: asyncio.Task | None = None

    def initialize(self) -> None:
        """Acquire the private writer lease before opening or migrating SQLite."""
        with self._lock:
            if self._closed:
                raise RuntimeError("privacy_closed")
            if self.ledger.path != ":memory:" and self._lease is None:
                path = Path(self.ledger.path)
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(str(path)+".lock", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise RuntimeError("privacy_writer_already_active") from None
                self._lease = descriptor
            self.ledger.initialize()
            if not self._ready:
                self.ledger.erasures.recover()
                self.ledger.index_deletions.recover()
                self._ready = True

    def require_available(self) -> None:
        """Reject consumers until the leased authority is ready, or once closing."""
        if self._closed or not self._ready or not self.ledger.authority_available:
            raise RuntimeError("privacy_authority_unavailable")

    def add_borrower(self, begin_close: Callable[[], asyncio.Task]) -> None:
        """Register actual native-user cleanup without retaining its private caches."""
        with self._lock:
            self.require_available()
            if any(reference() == begin_close for reference in self._borrowers):
                return
            self._borrowers.append(WeakMethod(begin_close))

    async def _offload(self, function, *args, **kwargs):
        context = copy_context()
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            context.run, partial(function, *args, **kwargs))

    async def acall(self, function, *args, **kwargs):
        """Run privacy I/O in its own bounded, retained worker scope."""
        if function != self.initialize:
            self.require_available()
        return await self._work.run(function, *args, **kwargs)

    def sources_available(self, sources: tuple[MemorySource, ...]) -> bool:
        """Reject admission after the authority begins closing."""
        return not self._closed and self._ready and self.ledger.sources_available(sources)

    def source_for_capture(self, capture_id: str, captured_at: float, room_id: str | None,
                           *, conversation_id: str | None = None, expires_at: float | None = None,
                           origin: MemoryOrigin = MemoryOrigin.VOICE) -> MemorySource:
        """Preserve capture scopes even while semantic memory is disabled."""
        self.require_available()
        return self.scopes.source_for_capture(capture_id, captured_at, room_id,
            conversation_id=conversation_id, expires_at=expires_at, origin=origin)

    def start(self) -> None:
        """Maintain fact deadlines independently of optional index reconciliation."""
        if self.maintenance is not None:
            return
        async def maintain():
            while not self._closed:
                try:
                    await self.acall(self.ledger.expire)
                    delay = await self.acall(self.ledger.next_maintenance_delay)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("Privacy maintenance deferred: %s", type(error).__name__)
                    delay = 30
                await asyncio.sleep(min(delay, 30))
        self.maintenance = asyncio.create_task(maintain())

    def resource_status(self) -> dict:
        """Report actual ownership, including pending optional-memory shutdown."""
        return {**self._work.status(), "writer_lease": self._lease is not None,
                "borrowers": sum(reference() is not None for reference in self._borrowers),
                "closing": self._closing is not None and not self._closing.done()}

    def _close_resources(self):
        with self._lock:
            self.ledger.close()
            self._executor.shutdown(wait=True)
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None

    def begin_close(self) -> asyncio.Task:
        """Close admission and retain the actual drain/SQLite/lease cleanup task."""
        with self._lock:
            self._closed = True
            if self._closing is None:
                drains = [self._work.begin_close()]
                for reference in self._borrowers:
                    begin = reference()
                    if begin is not None:
                        drains.append(begin())
                async def finish():
                    if self.maintenance is not None:
                        self.maintenance.cancel()
                        await asyncio.gather(self.maintenance, return_exceptions=True)
                    await asyncio.gather(*drains)
                    await asyncio.to_thread(self._close_resources)
                self._closing = asyncio.create_task(finish())
            return self._closing

    async def aclose(self) -> None:
        """Bound observation while retaining the shared authority and writer lease."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError as error:
                raise WorkCloseTimeout("privacy_resources_still_running") from error
        closing.result()
        if cancelled:
            raise asyncio.CancelledError


async def build_privacy_services(config: dict, *, shutdown: AsyncExitStack, clock=None) -> PrivacyServices:
    """Register ownership before unconditional initialization; never load Chroma."""
    settings = config.get("memory", {})
    path = os.fspath(settings.get("privacy_db_path", "./data/memory_privacy.db"))
    if not path or path == ":memory:" or path.startswith("file:"):
        raise ValueError("privacy_requires_persistent_file")
    services = PrivacyServices(path,
        clock=clock, automatic_days=settings.get("automatic_retention_days", 30))
    shutdown.push_async_callback(services.aclose)
    await services.acall(services.initialize)
    services.start()
    return services
