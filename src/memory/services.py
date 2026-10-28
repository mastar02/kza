"""Application construction and lifecycle for policy-owned memory and SDKs."""

import asyncio
from contextlib import AsyncExitStack
from contextvars import Context
from dataclasses import dataclass, field
from threading import Lock

from src.core.owned_work import ClosePhase, OwnedBlockingWork
from src.memory.memory_manager import MemoryManager
from src.memory.index_erasure import register_policy_index_target
from src.memory.long_term import MemoryCloseTimeout


@dataclass
class MemoryServices:
    """Retain actual construction, native borrowers and an app-created client."""

    manager: MemoryManager | None = None
    maintenance: asyncio.Task | None = None
    _client: object = field(default=None, init=False, repr=False)
    _owns_client: bool = field(default=False, init=False)
    _client_closed: bool = field(default=False, init=False)
    _client_failed: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _building: asyncio.Task | None = field(default=None, init=False, repr=False)
    _initializing: asyncio.Task | None = field(default=None, init=False, repr=False)
    _closing: asyncio.Task | None = field(default=None, init=False, repr=False)
    _phase: ClosePhase = field(default=ClosePhase.IDLE, init=False)
    _client_lock: object = field(default_factory=Lock, init=False, repr=False)
    _close_work: OwnedBlockingWork = field(default_factory=lambda: OwnedBlockingWork(capacity=1), init=False, repr=False)

    def start(self) -> None:
        """Start one maintenance owner after successful construction."""
        if self._closed or self.manager is None:
            raise RuntimeError('memory_services_unavailable')
        if self.maintenance is None:
            self.maintenance = asyncio.create_task(self.manager.long_term.maintain(), context=Context())

    def _close_client(self) -> None:
        """Close only our SDK handle, once, before releasing authority leases."""
        failed = False
        with self._client_lock:
            if not self._owns_client or self._client_closed or self._client is None:
                return
            if self._client_failed:
                raise RuntimeError('memory_sdk_close_failed')
            self._phase = ClosePhase.RESOURCES
            try:
                close = getattr(self._client, 'close', None)
                if not callable(close):
                    raise RuntimeError('unsupported_memory_sdk_close')
                close()
            except Exception:
                # A failed SDK may mark itself closed before physical release.
                # Retrying an idempotent no-op cannot prove that release.
                self._client_failed = failed = True
            else:
                self._client_closed = True
        if failed:
            raise RuntimeError('memory_sdk_close_failed')

    def begin_close(self) -> asyncio.Task:
        """Retain late construction and SDK shutdown after observer timeout."""
        self._closed = True
        if self._closing is None:
            self._phase = ClosePhase.DRAIN
            draining = self.manager.long_term.begin_close() if self.manager is not None else None
            async def finish():
                if self._building is not None:
                    await asyncio.gather(asyncio.shield(self._building), return_exceptions=True)
                    self._building = None
                if self._initializing is not None:
                    self._initializing.cancel()
                    await asyncio.gather(self._initializing, return_exceptions=True)
                    self._initializing = None
                if self.maintenance is not None:
                    self.maintenance.cancel()
                    await asyncio.gather(self.maintenance, return_exceptions=True)
                if self.manager is not None:
                    # Its hook closes our SDK before its own ledger and lease.
                    await asyncio.shield(draining if draining is not None else self.manager.long_term.begin_close())
                elif self._owns_client:
                    await self._close_work.run(self._close_client)
                await self._close_work.begin_close()
                self._phase = ClosePhase.COMPLETE
            self._closing = asyncio.create_task(finish(), context=Context())
            self._closing.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Observe the existing memory close budget while retaining live SDKs."""
        closing = self.begin_close()
        budget = self.manager.long_term.shutdown_timeout_s if self.manager is not None else 1
        try:
            async with asyncio.timeout(budget):
                await asyncio.shield(closing)
        except TimeoutError:
            raise MemoryCloseTimeout('memory_services_still_running') from None

    def resource_status(self) -> dict:
        """Project ownership and phase without paths, profiles or SDK reprs."""
        return {'closed': self._closed, 'owns_client': self._owns_client,
                'client_closed': self._client_closed, 'client_close_failed': self._client_failed,
                'building': self._building is not None and not self._building.done(),
                'closing': self._closing is not None and not self._closing.done(), 'close_phase': self._phase.value}


async def build_memory_services(config: dict, *, shutdown: AsyncExitStack, chroma_client=None,
                                clock=None, privacy=None) -> MemoryServices | None:
    """Register ownership before construction; preserve caller-owned clients."""
    settings = config.get('memory', {})
    if not settings.get('enabled', True):
        return None
    descriptor = None
    if privacy is not None:
        privacy.require_available()
        descriptor = await register_policy_index_target(config, privacy)
    config = None
    services = MemoryServices()
    services._owns_client = chroma_client is None
    if privacy is not None:
        privacy.add_borrower(services.begin_close)
    shutdown.push_async_callback(services.aclose)

    def construct():
        if services._closed:
            return
        client = chroma_client
        if client is None:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=descriptor['path'] if descriptor is not None else settings.get('chroma_path', './data/memory_db'),
                settings=Settings(anonymized_telemetry=False))
            services._client = client
        if services._closed:
            return
        services.manager = MemoryManager(client,
            preferences_path=settings.get('preferences_path', './data/preferences.json'),
            max_short_term_turns=settings.get('short_term_size', 10),
            collection_name=settings.get('long_term_collection', 'user_memories'),
            ledger_path=settings.get('privacy_db_path', './data/memory_privacy.db'),
            automatic_days=settings.get('automatic_retention_days', 30), clock=clock, privacy=privacy, index_descriptor=descriptor,
            close_chroma_client=services._close_client if services._owns_client else None)

    services._building = asyncio.create_task(asyncio.to_thread(construct), context=Context())
    building = services._building
    cancelled = False
    while not building.done():
        try:
            await asyncio.shield(building)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    failed = building.exception() is not None
    building = construct = None
    if failed or cancelled or services._closed:
        await asyncio.shield(services.begin_close())
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError('memory_services_construction_failed') from None
    services._initializing = asyncio.create_task(services.manager.ainitialize(), context=Context())
    failed = cancelled = False
    try:
        await asyncio.shield(services._initializing)
    except asyncio.CancelledError:
        cancelled = True
    except Exception:
        failed = True
    if failed or cancelled:
        await asyncio.shield(services.begin_close())
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError('memory_services_initialization_failed') from None
    if services._closed:
        await asyncio.shield(services.begin_close())
        raise RuntimeError('memory_services_closed')
    services.start()
    return services
