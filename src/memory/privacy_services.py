"""Always-on privacy authority, independent of semantic memory availability."""

from src.memory import output_registration_v2 as output_registration
import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from contextvars import Context, copy_context
from functools import partial
import fcntl
import logging
import os
from pathlib import Path
import threading
import time
from weakref import WeakMethod

from src.core.owned_work import ClosePhase, OwnedBlockingWork, WorkCloseTimeout, observe_native
from src.core.runtime_admission import verify_privacy_admission
from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource
from src.memory.sources import CaptureScopes

logger = logging.getLogger(__name__)

# Unconditional retired-owner prune cadence (review R1): one owner-row read under the
# writer lock per hour bounds how long restored or stranded rows of a removed account
# survive once the authority is usable, without taking the lock on every 30 s tick.
RETIRED_OWNER_PRUNE_INTERVAL_S = 3600.0
# While the prune keeps failing, repeat the warning at most this often (review R6).
RETIRED_OWNER_PRUNE_LOG_S = 600.0


class PrivacyServices:
    """Own the sole privacy writer until its native users actually finish."""

    def __init__(self, path: str, *, clock=None, automatic_days: float = 30,
                 close_timeout_s: float = 1):
        self.ledger = FactLedger(path, clock=clock, automatic_days=automatic_days)
        self.scopes = CaptureScopes(self.ledger, admission_started_at=self.ledger.clock())
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kza-privacy")
        self._work = OwnedBlockingWork(worker=self._offload, close_timeout_s=close_timeout_s)
        self._close_work = OwnedBlockingWork(capacity=1,close_timeout_s=close_timeout_s)
        self._close_phase = ClosePhase.IDLE
        self._close_timeout_s = close_timeout_s
        self._lease: int | None = None
        self._lock = threading.RLock()
        self._borrowers: list[WeakMethod] = []
        self._closed = False
        self._ready = False
        self._closing: asyncio.Task | None = None
        self.maintenance: asyncio.Task | None = None
        self._retired_owner_prune: tuple[Callable, Callable] | None = None
        self._retired_owner_prune_pending = False
        self._retired_owner_prune_at = 0.0
        self._retired_owner_prune_failed_at: float | None = None

    def initialize(self) -> None:
        """Acquire the private writer lease before opening or migrating SQLite."""
        with self._lock:
            if self._closed:
                raise RuntimeError("privacy_closed")
            output_registration.before_native(self, self.ledger.path)
            verify_privacy_admission(self.ledger.path)
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
                if not output_registration.native_ready(self):
                    self._ready = True

    def require_available(self) -> None:
        """Reject consumers until the leased authority is ready, or once closing."""
        if self._closed or not self._ready or not self.ledger.authority_available:
            raise RuntimeError("privacy_authority_unavailable")

    def add_borrower(self, begin_close: Callable[[], asyncio.Task]) -> None:
        """Register actual native-user cleanup without retaining its private caches."""
        # A closing native callback may hold _lock. Closed admission must reject
        # on the loop before attempting that lock, just as begin_close returns.
        self.require_available()
        with self._lock:
            self.require_available()
            if any(reference() == begin_close for reference in self._borrowers):
                return
            self._borrowers.append(WeakMethod(begin_close))

    async def _offload(self, function, *args, **kwargs):
        context = copy_context()
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            context.run, partial(observe_native,function, *args, **kwargs))

    async def acall(self, function, *args, **kwargs):
        """Run privacy I/O in its own bounded, retained worker scope."""
        if function != self.initialize:
            self.require_available()
        if function == self.initialize:
            return await output_registration.initialize_owned(self, self._work, function, *args, **kwargs)
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

    def attach_retired_owner_prune(self, retired: Callable[[frozenset[str]], bool],
                                   probe: Callable[[frozenset[str]], bool | None]) -> None:
        """Register the retired-owner prune for the maintenance tick (U3b-1b F2, review R1).

        ``retired`` is the durable proof (DashboardAuthority.users_retired) and
        ``probe`` the tri-state check that answers None when the authority is
        unusable (DashboardAuthority.memory_owner_retired). Called at boot, right
        before the boot prune itself, so the next unconditional run is one
        interval later.
        """
        if not callable(retired) or not callable(probe):
            raise PermissionError("memory_owner_retirement_required")
        self._retired_owner_prune = (retired, probe)
        self._retired_owner_prune_at = time.monotonic()

    def defer_retired_owner_prune(self) -> None:
        """Mark the prune pending after a failure: retried on every tick until it succeeds.

        Until then the ledger reports ``retired_owner_prune_pending``, a
        content-free health signal shown to operators.
        """
        if self._retired_owner_prune is None:
            raise PermissionError("memory_owner_retirement_required")
        if not self._retired_owner_prune_pending:
            self._retired_owner_prune_failed_at = None
        self._retired_owner_prune_pending = True
        self.ledger.retired_owner_prune_pending = True

    async def _retry_retired_owner_prune(self) -> None:
        """Run the prune while pending (every tick) or, unconditionally, once per interval."""
        if self._retired_owner_prune is None:
            return
        now = time.monotonic()
        if not self._retired_owner_prune_pending and now - self._retired_owner_prune_at < RETIRED_OWNER_PRUNE_INTERVAL_S:
            return
        self._retired_owner_prune_at = now
        retired, probe = self._retired_owner_prune
        try:
            deleted = await self.acall(self.ledger.prune_retired_owners, retired, probe=probe)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Log on the transition to failing, then at most once per interval of quiet (R6).
            last = self._retired_owner_prune_failed_at
            if not self._retired_owner_prune_pending or last is None or now - last >= RETIRED_OWNER_PRUNE_LOG_S:
                logger.warning("Retired-owner prune pending: %s", type(error).__name__)
                self._retired_owner_prune_failed_at = now
            self._retired_owner_prune_pending = True
            self.ledger.retired_owner_prune_pending = True
            return
        if self._retired_owner_prune_pending:
            logger.info("Retired-owner prune completed on the maintenance tick (%d deleted)", deleted)
        self._retired_owner_prune_pending = False
        self._retired_owner_prune_failed_at = None
        self.ledger.retired_owner_prune_pending = False

    def start(self) -> None:
        """Maintain fact deadlines independently of optional index reconciliation."""
        if self.maintenance is not None:
            return
        async def maintain():
            while not self._closed:
                try:
                    await self._retry_retired_owner_prune()
                    await self.acall(self.ledger.expire)
                    delay = await self.acall(self.ledger.next_maintenance_delay)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("Privacy maintenance deferred: %s", type(error).__name__)
                    delay = 30
                await asyncio.sleep(min(delay, 30))
        self.maintenance = asyncio.create_task(maintain(),context=Context())

    def resource_status(self) -> dict:
        """Report actual ownership, including pending optional-memory shutdown."""
        return {**self._work.status(), "writer_lease": self._lease is not None,
                "borrowers": sum(reference() is not None for reference in self._borrowers),
                "closing": self._closing is not None and not self._closing.done(),
                "close_phase":self._close_phase.value,"io":self._work.diagnostics(),
                "close_io":self._close_work.diagnostics()}

    def _close_resources(self):
        with self._lock:
            self._close_phase = ClosePhase.LEDGER
            self.ledger.close()
            self._close_phase = ClosePhase.EXECUTOR
            self._executor.shutdown(wait=True)
            self._close_phase = ClosePhase.LEASE
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None

    def begin_close(self) -> asyncio.Task:
        """Close admission and retain the actual drain/SQLite/lease cleanup task."""
        # Admission and the retained task are loop-owned. The native initializer
        # and close callback may hold _lock; waiting for it here stalls capture.
        self._closed = True
        if self._closing is None:
            drains = [self._work.begin_close()]
            for reference in self._borrowers:
                begin = reference()
                if begin is not None:
                    drains.append(Context().run(begin))
            async def finish():
                self._close_phase = ClosePhase.MAINTENANCE
                if self.maintenance is not None:
                    self.maintenance.cancel()
                    await asyncio.gather(self.maintenance, return_exceptions=True)
                self._close_phase = ClosePhase.DRAIN
                await asyncio.gather(*drains)
                self._close_phase = ClosePhase.RESOURCES
                publication_failure = None
                try:
                    await output_registration.final_registry(self)
                except BaseException as error:
                    publication_failure = error
                try:
                    await self._close_work.run(self._close_resources)
                finally:
                    await self._close_work.begin_close()
                await output_registration.owner_closed(self)
                self._close_phase = ClosePhase.COMPLETE
                if publication_failure is not None:
                    raise publication_failure
            self._closing = asyncio.create_task(finish(),context=Context())
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
                raise WorkCloseTimeout("privacy_resources_still_running",phase=self._close_phase,
                    diagnostics=self.resource_status()) from error
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
    await output_registration.attach_output(services, "privacy_authority")
    await services.acall(services.initialize)
    services.start()
    return services
