"""Always-on capture retention over the private registry of storage locations."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import hashlib
import logging
import math
import os
from pathlib import Path
from weakref import WeakSet

import aiosqlite

from src.ambient.store import AmbientStore
from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices

logger = logging.getLogger(__name__)


class RetentionState(StrEnum):
    RUNNING = 'running'
    CURRENT = 'current'
    PENDING = 'pending'
    UNAVAILABLE = 'unavailable'


@dataclass(frozen=True)
class CaptureLocation:
    """Group historical archive roots by their actual database path."""

    key: str
    database: str
    roots: tuple[str, ...]


@dataclass(frozen=True)
class SweepReceipt:
    """Content-free completion of actual SQLite and filesystem work."""

    state: RetentionState
    checked_at: float
    next_delay: float = 60
    rows_deleted: int = 0
    pending_files: int = 0
    error: str | None = None


class CaptureRetention:
    """Own two independent sweeps, isolated native I/O and deadline wakeups.

    SQLite deadlines, write intents and pending unlinks provide durable restart
    state. Observation cancellation never cancels an actual sweep or its disk
    worker. Status deliberately cannot assert global conversation erasure.
    """

    def __init__(self, privacy: PrivacyServices, *, audio_inflight: Callable[[str], bool], archive_contains: Callable[[str], bool], retention_hours: float = 48):
        if type(retention_hours) not in (int, float) or not math.isfinite(retention_hours) or not 0 < retention_hours <= 48:
            raise ValueError('invalid_capture_retention_hours')
        self.privacy = privacy
        self._retention_hours = retention_hours
        self._audio_inflight = audio_inflight
        self._archive_contains = archive_contains
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='capture-retention')
        self._disk = OwnedBlockingWork(worker=self._offload, capacity=8)
        self._locations: dict[str, CaptureLocation] = {}
        self._registry_errors: list[dict] = []
        self._watched: dict[str, WeakSet] = {}
        self._receipts: dict[str, SweepReceipt] = {}
        self._due: dict[str, float] = {}
        self._active: dict[str, tuple[CaptureLocation, asyncio.Task]] = {}
        self._dirty: set[str] = set()
        self._orphan_due: dict[str, float] = {}
        self._wake = asyncio.Event()
        self._cycle: asyncio.Task | None = None
        self._maintenance: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._closed = False

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            copy_context().run, partial(function, *args, **kwargs))

    async def refresh(self) -> None:
        """Refresh registered locators off-loop, preserving historical locations."""
        targets = await self.privacy.acall(self.privacy.ledger.erasures.targets, ErasureParticipant.CAPTURES)
        grouped: dict[str, set[str]] = {}
        errors = []
        for target in targets:
            descriptor = target['descriptor']
            if target['version'] != 1 or set(descriptor) != {'db_path', 'audio_dir'} or any(type(p) is not str or not Path(p).is_absolute() for p in descriptor.values()):
                errors.append({'id': target['id'], 'error': 'unsupported_capture_retention_target'})
                continue
            grouped.setdefault(descriptor['db_path'], set()).add(descriptor['audio_dir'])
        locations = {database: CaptureLocation(hashlib.sha256(database.encode()).hexdigest(), database, tuple(sorted(roots)))
                     for database, roots in grouped.items()}
        for database, location in locations.items():
            if self._locations.get(database) != location:
                self._dirty.add(database)
                self._orphan_due.pop(database, None)
        self._locations = locations
        self._registry_errors = errors
        for database, location in locations.items():
            for store in self._watched.get(database, ()):
                store._cleanup_audio_roots = tuple(Path(root) for root in location.roots)

    def notify(self, database: str) -> None:
        """Wake one target after accepted capture, new locator or native cleanup."""
        if not self._closed:
            self._dirty.add(database)
            self._wake.set()

    async def watch_store(self, store: AmbientStore, database: str) -> None:
        """Bind active stores to the same file guard and isolated native owner."""
        await self.refresh()
        self._watched.setdefault(database, WeakSet()).add(store)
        store._disk_worker = self._disk.run
        store.file_guard = partial(self.guard_files, database)
        store.retention_notify = partial(self.notify, database)
        store.retention_hours = min(store.retention_hours, self._retention_hours)
        # Active transcriber purges must also retain the historical roots.
        store._cleanup_audio_roots = tuple(Path(root) for root in self._locations[database].roots)
        self.notify(database)

    def _canonical(self, paths):
        return {path: str(Path(path).resolve()) for path in paths}

    async def _live_references(self, location: CaptureLocation) -> tuple[set[str], set[int]]:
        """Read metadata only, without upgrading an unrelated peer database."""
        def locate():
            try:
                os.stat(location.database)
            except FileNotFoundError:
                return None
            return Path(location.database).as_uri()+'?mode=ro'
        uri = await self._disk.run(locate)
        if uri is None:
            return set(), set()
        connection = None
        try:
            connection = await aiosqlite.connect(uri, uri=True)
            connection.row_factory = aiosqlite.Row
            cursor = await connection.execute('PRAGMA table_info(utterances)')
            columns = {row['name'] for row in await cursor.fetchall()}
            required = {'id', 't0', 'created_at', 'audio_path', 'expires_at', 'conversation_id'}
            if not required <= columns:
                raise RuntimeError('unknown_capture_reference_schema')
            cursor = await connection.execute('SELECT id, t0, created_at, audio_path, expires_at, conversation_id FROM utterances')
            now = self.privacy.ledger.clock()
            paths, ids = set(), set()
            for row in await cursor.fetchall():
                deadline = min(row['t0'], row['created_at'])+self._retention_hours*3600
                if row['expires_at'] is not None:
                    deadline = min(deadline, row['expires_at'])
                conversation = row['conversation_id']
                if now < deadline and (conversation is None or not self.privacy.ledger.conversation_revoked(conversation)):
                    ids.add(row['id'])
                    if row['audio_path'] is not None:
                        paths.add(row['audio_path'])
            cursor = await connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_write_intents'")
            if await cursor.fetchone() is not None:
                cursor = await connection.execute('SELECT path, expires_at, conversation_id FROM audio_write_intents')
                paths.update(row['path'] for row in await cursor.fetchall() if now < row['expires_at'] and not self.privacy.ledger.conversation_revoked(row['conversation_id']))
            return set((await self._disk.run(self._canonical, paths)).values()), ids
        finally:
            if connection is not None:
                await connection.close()

    async def guard_files(self, database: str, paths: list[str]) -> set[str]:
        """Protect references in every registered database sharing a clip root."""
        if not self.privacy.ledger.authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        if database not in self._locations:
            raise RuntimeError('unregistered_capture_retention_store')
        if self._registry_errors:
            return set()
        canonical = await self._disk.run(self._canonical, paths)
        allowed = set(paths)
        # The cached registry is refreshed before admission and every scheduler
        # tick; it remains usable during a leased close after privacy I/O closes.
        for location in tuple(self._locations.values()):
            relevant = {path for path, resolved in canonical.items()
                        if any(Path(resolved).is_relative_to(root) for root in location.roots)}
            if not relevant:
                continue
            try:
                references, ids = await self._live_references(location)
            except Exception:
                allowed.difference_update(relevant)
                continue
            for path in relevant:
                resolved = canonical[path]
                if resolved in references or Path(path).stem.isdecimal() and int(Path(path).stem) in ids:
                    allowed.discard(path)
        if not self.privacy.ledger.authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        return allowed

    def _legacy_orphans(self, roots: tuple[str, ...], cutoff: float, queued: set[str]) -> list[str]:
        candidates = []
        def visit(directory):
            try:
                os.stat(directory)
            except FileNotFoundError:
                return
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise RuntimeError('capture_archive_symlink')
                    if entry.is_dir(follow_symlinks=False):
                        visit(entry.path)
                    elif entry.path not in queued and entry.name.endswith('.flac') and Path(entry.name).stem.isdecimal() and entry.stat().st_mtime < cutoff:
                        candidates.append(entry.path)
                        if len(candidates) >= 256:
                            return
                    if len(candidates) >= 256:
                        return
        for root in roots:
            visit(root)
            if len(candidates) >= 256:
                break
        return candidates

    async def _sweep(self, location: CaptureLocation) -> SweepReceipt:
        store = AmbientStore(location.database, audio_dir=location.roots[0], clock=self.privacy.ledger.clock, retention_hours=self._retention_hours,
            source_policy=self.privacy.ledger, audio_inflight=self._audio_inflight,
            cleanup_audio_roots=location.roots, disk_worker=self._disk.run,
            file_guard=partial(self.guard_files, location.database))
        try:
            if not await store.init(existing_only=True):
                contains = await self._disk.run(lambda: any(self._archive_contains(root) for root in location.roots))
                return SweepReceipt(RetentionState.UNAVAILABLE, self.privacy.ledger.clock(), error='capture_archive_without_database') if contains else SweepReceipt(RetentionState.CURRENT, self.privacy.ledger.clock())
            deleted = await store.purge_expired(sweep_orphans=False)
            candidates = []
            now = self.privacy.ledger.clock()
            if now >= self._orphan_due.get(location.database, 0):
                self._orphan_due[location.database] = now+60
                cursor = await store._db.execute('SELECT path FROM pending_audio_deletions')
                queued = set((await self._disk.run(self._canonical, [row['path'] for row in await cursor.fetchall()])).values())
                candidates = await self._disk.run(self._legacy_orphans, location.roots, now-self._retention_hours*3600, queued)
                if candidates:
                    await store.queue_orphan_audio(candidates)
                if len(candidates) >= 256:
                    self._orphan_due[location.database] = now+.25
            pending = (await store.audio_cleanup_status())['pending']
            delay = await store.next_expiration_delay()
            delay = min(delay, max(.05, self._orphan_due.get(location.database, now)-now))
            return SweepReceipt(RetentionState.PENDING if pending else RetentionState.CURRENT,
                                self.privacy.ledger.clock(), min(60, max(.05, delay)), deleted, pending)
        finally:
            await store.close()

    async def _run_sweep(self, location: CaptureLocation) -> SweepReceipt:
        try:
            return await self._sweep(location)
        except Exception:
            # Retained task outcomes cannot hold private driver exceptions.
            return SweepReceipt(RetentionState.UNAVAILABLE, self.privacy.ledger.clock(), error='capture_cleanup_failed')

    async def _reconcile(self) -> None:
        await self.refresh()
        for database, (location, task) in list(self._active.items()):
            if task.done():
                receipt = task.result()
                self._receipts[database] = receipt
                self._due[database] = receipt.checked_at+receipt.next_delay
                if self._locations.get(database) != location:
                    self._dirty.add(database)
                    self._orphan_due.pop(database, None)
                del self._active[database]
        now = self.privacy.ledger.clock()
        for database in sorted(self._locations, key=lambda db: self._due.get(db, 0)):
            if self._closed or len(self._active) >= 2:
                break
            if database in self._active or database not in self._dirty and self._due.get(database, 0) > now:
                continue
            self._dirty.discard(database)
            location = self._locations[database]
            self._active[database] = (location, asyncio.create_task(self._run_sweep(location)))

    async def reconcile(self) -> None:
        """Observe one bounded scheduling cycle without cancelling actual work."""
        if self._closed:
            raise RuntimeError('capture_retention_closed')
        if self._cycle is None or self._cycle.done():
            self._cycle = asyncio.create_task(self._reconcile())
            self._cycle.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        await asyncio.shield(self._cycle)

    def start(self) -> None:
        """Maintain deadlines even while capture, archiving and recall are off."""
        if self._maintenance is not None or self._closed:
            return
        async def maintain():
            while not self._closed:
                self._wake.clear()
                failed = False
                try:
                    await self.reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning('Capture retention deferred: %s', type(error).__name__)
                    failed = True
                now = self.privacy.ledger.clock()
                delay = min([30, *[max(.05, due-now) for db, due in self._due.items() if db not in self._active]])
                if failed:
                    delay = 30
                elif self._active:
                    delay = min(delay, .05)
                try:
                    async with asyncio.timeout(delay):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        self._maintenance = asyncio.create_task(maintain())

    def status(self) -> dict:
        """Report content-free actual ownership and per-target cleanup state."""
        return {'closed': self._closed, 'active': len(self._active), 'disk': self._disk.status(), 'registry_errors': self._registry_errors,
                'targets': [{'id': location.key, 'state': RetentionState.RUNNING.value if db in self._active else self._receipts.get(db, SweepReceipt(RetentionState.PENDING, 0)).state.value,
                             'pending_files': self._receipts.get(db, SweepReceipt(RetentionState.PENDING, 0)).pending_files,
                             'error': self._receipts.get(db, SweepReceipt(RetentionState.PENDING, 0)).error}
                            for db, location in self._locations.items()]}

    def stop_admission(self) -> None:
        """Stop scheduling, leaving the disk scope open for producer cleanup."""
        self._closed = True
        self._wake.set()
        if self._maintenance is not None:
            self._maintenance.cancel()

    def begin_close(self) -> asyncio.Task:
        """Retain actual sweep, peer-read, unlink and executor completion."""
        self.stop_admission()
        if self._closing is None:
            async def finish():
                if self._maintenance is not None:
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                if self._cycle is not None:
                    await asyncio.gather(self._cycle, return_exceptions=True)
                await asyncio.gather(*(task for _, task in self._active.values()), return_exceptions=True)
                self._active.clear()
                await self._disk.begin_close()
                await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._closing = asyncio.create_task(finish())
        return self._closing
