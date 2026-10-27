"""Durable locators and independent retention for sourced wake-training WAVs."""

import asyncio
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from enum import StrEnum
import json
import logging
import math
import os
from pathlib import Path
import threading
from uuid import UUID
from weakref import WeakSet

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemorySource

logger = logging.getLogger(__name__)


class ClipRetentionState(StrEnum):
    """Local managed-file maintenance, never a global erasure receipt."""

    PENDING = 'pending'
    CURRENT = 'current'
    UNAVAILABLE = 'unavailable'


_SCHEMA = """CREATE TABLE IF NOT EXISTS wake_clips_v1 (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
    target_id TEXT NOT NULL REFERENCES erasure_targets(id), bucket TEXT NOT NULL,
    room_id TEXT NOT NULL, score REAL NOT NULL, sources_json TEXT NOT NULL,
    captured_at REAL NOT NULL, expires_at REAL NOT NULL, published INTEGER NOT NULL DEFAULT 0)"""


@dataclass(frozen=True)
class ClipTarget:
    """Canonical root registered before a writer can create any file."""

    id: str
    root: str


@contextmanager
def clip_directory(root: str, *, create: bool = False):
    """Open a private directory using descriptors; never follow a replaced link."""
    path = Path(root)
    if not path.is_absolute() or str(path) != root or '..' in path.parts:
        raise ValueError('invalid_wake_clip_root')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in (*path.parts[1:], 'policy_v1'):
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


def clip_sources(encoded: str) -> tuple[MemorySource, ...]:
    """Reject unknown metadata rather than claiming an unattributed file erased."""
    values = json.loads(encoded)
    if type(values) is not list or not 1 <= len(values) <= 64:
        raise ValueError('invalid_wake_clip_sources')
    return tuple(MemorySource.from_dict(value) for value in values)


class WakeClipStore:
    """Own locators, actual file cleanup and writer lifetimes independently of capture.

    This registers collector targets but leaves COLLECTORS without a cleanup
    driver: conversations, corrections and other training copies are still open.
    Local clip cleanup must never masquerade as global collector completion.
    """

    def __init__(self, privacy, *, close_timeout_s: float = 1):
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError('invalid_wake_clip_store_close_timeout')
        privacy.require_available()
        self.privacy = privacy
        self.ledger = privacy.ledger
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='wake-clip-storage')
        self._work = OwnedBlockingWork(worker=self._offload, capacity=4)
        self._files = threading.RLock()
        self._writers = WeakSet()
        self._closed = False
        self._closing = self._maintenance = None
        self._close_timeout_s = close_timeout_s
        self._next = 0.
        self._authority_version = -1
        self._changes = self._seen_changes = 0
        self._cursor = 0
        self._cycle_deadline = 0.
        self._cycle_pending = 0
        self._sweep = None
        self._status = {'state': ClipRetentionState.PENDING, 'removed': 0, 'pending': 0}
        privacy.add_borrower(self.begin_close)

    async def _offload(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, copy_context().run, partial(function, *args))

    def initialize(self) -> None:
        """Create private metadata without importing or deleting legacy files."""
        self.privacy.require_available()
        with self.ledger._transaction() as db:
            db.execute(_SCHEMA)
            db.execute('CREATE INDEX IF NOT EXISTS wake_clip_expiry ON wake_clips_v1(expires_at)')

    def register(self, directory) -> ClipTarget:
        """Persist a canonical locator even when the corresponding feature is off."""
        if self._closed:
            raise RuntimeError('wake_clip_store_closed')
        self.privacy.require_available()
        root = str(Path(directory).resolve())
        key = self.ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,
                                                 {'kind': 'wake_clips', 'root': root})
        return ClipTarget(key, root)

    def attach(self, writer) -> None:
        """Keep an actual native writer under this shared privacy lease."""
        if self._closed or writer._store is not self:
            raise RuntimeError('wake_clip_store_unavailable')
        self._writers.add(writer)

    def _registered(self, target: ClipTarget) -> bool:
        return any(t['id'] == target.id and t['version'] == 1 and t['descriptor'] == {'kind': 'wake_clips', 'root': target.root}
                   for t in self.ledger.erasures.targets(ErasureParticipant.COLLECTORS))

    def changed(self) -> None:
        """Publish a metadata-only wakeup from a native writer."""
        self._changes += 1

    def reserve(self, target: ClipTarget, job) -> None:
        """Commit an exact file intent before opening either part or final WAV."""
        if self._closed or not self._registered(target):
            raise RuntimeError('wake_clip_store_unavailable')
        with self.ledger._transaction() as db:
            if not self.ledger._sources_live(db, job.sources):
                raise PermissionError('wake_clip_sources_invalidated')
            db.execute('''INSERT INTO wake_clips_v1
                (id,target_id,bucket,room_id,score,sources_json,captured_at,expires_at)
                VALUES (?,?,?,?,?,?,?,?)''', (job.id, target.id, job.bucket, job.room_id, job.score,
                json.dumps([s.to_dict() for s in job.sources], separators=(',', ':')),
                min(s.captured_at for s in job.sources), min(s.expires_at for s in job.sources)))

        self.changed()

    def publish(self, target: ClipTarget, job, live) -> None:
        """Atomically admit final rename before a tombstone can commit."""
        with self._files, self.ledger._transaction() as db:
            if not live() or not self.ledger._sources_live(db, job.sources):
                raise PermissionError('wake_clip_sources_invalidated')
            with clip_directory(target.root) as directory:
                os.replace(job.id + '.part', job.id + '.wav', src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            if not live() or not self.ledger._sources_live(db, job.sources):
                raise PermissionError('wake_clip_sources_invalidated')
            db.execute('UPDATE wake_clips_v1 SET published=1 WHERE id=? AND target_id=?', (job.id, target.id))

    def _unlink(self, target: ClipTarget, identifier: str) -> None:
        if str(UUID(identifier)) != identifier:
            raise ValueError('invalid_wake_clip_id')
        try:
            with clip_directory(target.root) as directory:
                for extension in ('.part', '.wav'):
                    try:
                        os.unlink(identifier + extension, dir_fd=directory)
                    except FileNotFoundError:
                        pass
                os.fsync(directory)
        except FileNotFoundError:
            pass

    def discard(self, target: ClipTarget, identifier: str) -> None:
        """Remove files before releasing their restart-safe locator."""
        with self._files:
            self._unlink(target, identifier)
            with self.ledger._transaction() as db:
                db.execute('DELETE FROM wake_clips_v1 WHERE target_id=? AND id=?', (target.id, identifier))

    def _writing(self, identifier: str) -> bool:
        return any(w.is_writing(identifier) for w in tuple(self._writers))

    def rotate(self, target: ClipTarget, bucket: str, cap: int) -> None:
        """Rotate only this policy's published files, preserving all legacy material."""
        with self._files, self.ledger._transaction() as db:
            rows = db.execute('''SELECT id FROM wake_clips_v1 WHERE target_id=? AND bucket=? AND published=1
                ORDER BY sequence DESC LIMIT -1 OFFSET ?''', (target.id, bucket, cap)).fetchall()
            for row in rows:
                if not self._writing(row['id']):
                    self._unlink(target, row['id'])
                    db.execute('DELETE FROM wake_clips_v1 WHERE id=?', (row['id'],))

    def _sweep_native(self):
        removed = pending = 0
        now = self.ledger.clock()
        version, changes = self.ledger._authority_version, self._changes
        if version != self._authority_version or changes != self._seen_changes:
            self._cursor = 0
        if not self._cursor:
            self._cycle_deadline = now + 60
            self._cycle_pending = 0
        targets = self.ledger.erasures.targets(ErasureParticipant.COLLECTORS)
        roots = {t['id']: ClipTarget(t['id'], t['descriptor']['root']) for t in targets
                 if t['version'] == 1 and set(t['descriptor']) == {'kind', 'root'} and t['descriptor']['kind'] == 'wake_clips'}
        with self._files, self.ledger._transaction() as db:
            if db.execute('SELECT 1 FROM wake_clips_v1 WHERE sequence<=0 LIMIT 1').fetchone():
                raise ValueError('unrecognized_wake_clip_sequence')
            rows = db.execute('SELECT * FROM wake_clips_v1 WHERE sequence>? ORDER BY sequence LIMIT 128', (self._cursor,)).fetchall()
            for row in rows:
                sources = clip_sources(row['sources_json'])
                target = roots.get(row['target_id'])
                if target is None or not 0 < row['expires_at'] <= min(s.expires_at for s in sources):
                    raise ValueError('unrecognized_wake_clip_locator')
                deadline = min(row['expires_at'], *(s.expires_at for s in sources))
                if self._writing(row['id']):
                    pending += 1
                    self._cycle_deadline = min(self._cycle_deadline, now + .05)
                    continue
                if not row['published'] or deadline <= now or self.ledger.sources_committed_unavailable(sources):
                    self._unlink(target, row['id'])
                    db.execute('DELETE FROM wake_clips_v1 WHERE id=?', (row['id'],))
                    removed += 1
                else:
                    self._cycle_deadline = min(self._cycle_deadline, deadline)
        self._cursor = rows[-1]['sequence'] if len(rows) == 128 else 0
        self._cycle_pending += pending
        return ({'state': ClipRetentionState.PENDING if self._cycle_pending or self._cursor else ClipRetentionState.CURRENT, 'removed': removed, 'pending': self._cycle_pending},
                now + .01 if self._cursor else self._cycle_deadline, version, changes)

    async def reconcile(self) -> None:
        """Observe a retained physical sweep without cancelling its actual worker."""
        if self._closed:
            raise RuntimeError('wake_clip_store_closed')
        if self._sweep is None or self._sweep.done():
            self._sweep = asyncio.create_task(self._work.run(self._sweep_native))
            self._sweep.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            self._status, self._next, self._authority_version, self._seen_changes = await asyncio.shield(self._sweep)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._status = {'state': ClipRetentionState.UNAVAILABLE, 'removed': 0, 'pending': 1}
            self._next = self.ledger.clock() + 1
            logger.warning('Wake clip maintenance deferred: %s', type(error).__name__)

    def start(self) -> None:
        """Prune queued audio and schedule physical TTL even with capture disabled."""
        if self._closed:
            raise RuntimeError('wake_clip_store_closed')
        if self._maintenance is None:
            async def maintain():
                while not self._closed:
                    for writer in tuple(self._writers):
                        writer.prune()
                    if self.ledger.clock() >= self._next or self._authority_version != self.ledger._authority_version or self._seen_changes != self._changes:
                        await self.reconcile()
                    await asyncio.sleep(.05)
            self._maintenance = asyncio.create_task(maintain())

    def begin_close(self) -> asyncio.Task:
        """Retain real writer exits, sweeps and executor shutdown before lease release."""
        self._closed = True
        if self._closing is None:
            writers = [w.begin_close() for w in tuple(self._writers)]
            async def finish():
                if self._maintenance is not None:
                    self._maintenance.cancel()
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                await asyncio.gather(*writers)
                if self._sweep is not None:
                    await asyncio.gather(self._sweep, return_exceptions=True)
                await self._work.begin_close()
                # Admission and submitted work are already drained. Signal the
                # private pool without joining on-loop or borrowing a possibly
                # saturated general executor; retain its actual thread exit.
                self._executor.shutdown(wait=False)
                while any(thread.is_alive() for thread in tuple(self._executor._threads)):
                    await asyncio.sleep(.005)
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound close observation while retaining actual filesystem ownership."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('wake_clip_storage_still_running') from None

    def resource_status(self) -> dict:
        """Expose counts only, without paths, rooms, source IDs or raw data."""
        return {**self._status, 'scope': 'managed_wake_clips_only', 'legacy_policy': 'quarantined',
                'closed': self._closed, 'writers': len(self._writers),
                'operations': self._work.status()['operations']}


async def build_wake_clip_store(config: dict, *, privacy, shutdown) -> WakeClipStore:
    """Construct retention before the optional writer and register its close early."""
    store = WakeClipStore(privacy)
    shutdown.push_async_callback(store.aclose)
    await store._work.run(store.initialize)
    clips = config.get('rooms', {}).get('wake_word', {}).get('capture_clips', {}) or {}
    await store._work.run(store.register, clips.get('dir', 'data/wakeword_training/captured'))
    store.start()
    return store
