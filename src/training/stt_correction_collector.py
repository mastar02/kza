"""Sourced STT corrections sharing durable file intents and raw capture TTL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import logging
import math
import os
from pathlib import Path
import stat
import threading
from uuid import UUID, uuid4
from weakref import WeakSet, ref

import numpy as np

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.fact_ledger import MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT
from src.memory.prompt_context import merge_sources
from src.wakeword.clip_store import CollectorKind, ClipTarget, WakeClipStore, clip_directory, clip_sources
from src.wakeword.wake_clip_writer import encode_wav

logger = logging.getLogger(__name__)
_RAW_FIELDS = frozenset({'audio_path', 'original_text', 'corrected_text', 'timestamp', 'user_id', 'confidence'})


@dataclass(eq=False, repr=False)
class STTCorrection:
    """A managed read view; original sources gate every raw field access.

    Standalone DTOs remain usable for synthetic training adapters. Real reads
    carry a weak owner receipt and are scrubbed on expiry, deletion or close.
    """

    audio_path: str
    original_text: str
    corrected_text: str
    timestamp: str
    user_id: str | None = None
    confidence: float = 0.
    used: bool = False
    id: str = ''
    sources: tuple[MemorySource, ...] = ()
    expires_at: float = 0.
    _owner: object = field(default=None, repr=False)
    _retired: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def available(self) -> bool:
        """Check authority without disk or renewing the capture deadline."""
        owner = self._owner() if self._owner is not None else None
        return bool(owner is not None and not self._retired.is_set() and owner._live(self.sources)
                    and owner._store.ledger.clock() < min(self.expires_at, owner._store.deadline_for(self.id, self.expires_at)))

    def clear(self) -> None:
        """Retire all copied raw fields while retaining only source metadata."""
        self._retired.set()
        for name in _RAW_FIELDS:
            object.__setattr__(self, name, None if name == 'user_id' else 0. if name == 'confidence' else '')

    def __getattribute__(self, name):
        if name in _RAW_FIELDS and object.__getattribute__(self, '_owner') is not None:
            if not object.__getattribute__(self, 'available'):
                object.__getattribute__(self, 'clear')()
        return object.__getattribute__(self, name)

    def __repr__(self):
        return 'STTCorrection(managed=%s, available=%s)' % (self._owner is not None, self.available)


class CorrectionState(StrEnum):
    QUEUED = 'queued'
    RUNNING = 'running'
    FINISHED = 'finished'


@dataclass
class CorrectionJob:
    """One actual native operation, including original and feedback captures."""

    id: str
    audio: np.ndarray | None
    sources: tuple[MemorySource, ...]
    original_text: str
    corrected_text: str
    sample_rate: int
    user_id: str | None
    score: float
    room_id: str
    bucket: str = 'stt_correction'
    cancelled: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)
    state: CorrectionState = CorrectionState.QUEUED
    deadline: float = 0.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.deadline = min(source.expires_at for source in self.sources)

    def clear(self) -> None:
        """Release queued or completed data; a running native call keeps its inputs."""
        self.audio = None
        self.original_text = self.corrected_text = ''
        self.user_id = None


class STTCorrectionCollector:
    """Own actual correction writes and expiring read views, without legacy import."""

    def __init__(self, data_dir: str, max_audio_duration: float = 30., *,
                 store: WakeClipStore, target: ClipTarget, close_timeout_s: float = 1):
        if str(data_dir) != target.root or target.kind is not CollectorKind.STT_CORRECTIONS:
            raise ValueError('stt_correction_target_mismatch')
        if type(max_audio_duration) not in (int, float) or not math.isfinite(max_audio_duration) or not 0 < max_audio_duration <= 30:
            raise ValueError('invalid_stt_correction_duration')
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError('invalid_stt_correction_close_timeout')
        self.data_dir, self.max_audio_duration = Path(data_dir), max_audio_duration
        self._store, self._target = store, target
        self._work = OwnedBlockingWork(worker=store._offload, capacity=8)
        self._closed = False
        self._closing = None
        self._close_timeout_s = close_timeout_s
        self._active: dict[str, CorrectionJob] = {}
        self._jobs: dict[str, CorrectionJob] = {}
        self._views = WeakSet()
        self._views_lock = threading.Lock()
        self._errors = 0
        store.attach(self)

    def _invoke(self, function, args):
        failure = None
        try:
            return function(*args)
        except Exception as error:
            failure = type(error).__name__
        finally:
            function = args = None
        self._errors += 1
        logger.warning('STT correction operation failed: %s', failure)
        raise RuntimeError('stt_correction_operation_failed')

    async def _native(self, function, *args):
        try:
            return await self._work.run(self._invoke, function, args)
        finally:
            function = args = None

    def _live(self, sources):
        return not self._closed and not self._store._closed and self._store.privacy.sources_available(sources)

    def _job_live(self, job):
        if self._store.ledger.clock() >= min(job.deadline, self._store.deadline_for(job.id, job.deadline)) or not self._live(job.sources):
            job.cancelled.set()
        return not job.cancelled.is_set()

    async def add_correction(self, audio: np.ndarray, original_text: str, corrected_text: str,
                             sample_rate: int = 16000, user_id: str | None = None, confidence: float = 0., *,
                             sources: tuple[MemorySource, ...] = (),
                             correction_sources: tuple[MemorySource, ...] | None = None) -> str | None:
        """Save original audio and feedback only with both sets of capture sources."""
        job = None
        copied = None
        try:
            feedback = PRODUCER_CONTEXT.get().sources if correction_sources is None else correction_sources
            if type(sources) is not tuple or type(feedback) is not tuple or not sources or not feedback or not self._live(sources) or not self._live(feedback):
                return None
            if len(self._jobs) >= 8 or self._work.status()['operations'] >= 8:
                return None
            lineage = merge_sources(sources, feedback)
            if len(lineage) > 64 or type(sample_rate) is not int or not 8000 <= sample_rate <= 48000:
                return None
            if not 0 < len(audio) <= sample_rate*self.max_audio_duration:
                return None
            if not isinstance(original_text, str) or not isinstance(corrected_text, str) or not corrected_text.strip():
                return None
            if max(len(original_text), len(corrected_text)) > 4096 or user_id is not None and (type(user_id) is not str or len(user_id) > 128):
                return None
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                return None
            copied = np.asarray(audio, dtype=np.float32).copy()
            if copied.ndim != 1:
                return None
            job = CorrectionJob(str(uuid4()), copied, lineage, original_text, corrected_text,
                                sample_rate, user_id, confidence, PRODUCER_CONTEXT.get().room_id or '')
            self._jobs[job.id] = job
            copied = audio = original_text = corrected_text = user_id = None
            return await self._native(self._save, job)
        finally:
            audio = copied = job = original_text = corrected_text = user_id = None

    def _encode(self, job):
        encode_wav(self._target, job.id, job.audio, job.sample_rate)

    def _save(self, job):
        identifier = job.id
        published = False
        failure = None
        try:
            with job.lock:
                if not self._job_live(job):
                    return None
                job.state = CorrectionState.RUNNING
                self._active[identifier] = job
            self._store.reserve(self._target, job)
            with self._store.ledger._transaction() as db:
                if not self._job_live(job) or not self._store.ledger._sources_live(db, job.sources):
                    return None
                db.execute('INSERT INTO stt_corrections_v1(id,original_text,corrected_text,user_id,sample_rate) VALUES (?,?,?,?,?)',
                           (identifier, job.original_text, job.corrected_text, job.user_id, job.sample_rate))
            self._encode(job)
            self._store.publish(self._target, job, lambda: self._job_live(job))
            published = True
            self._store.rotate(self._target, job.bucket, 1000)
        except Exception as error:
            failure = type(error).__name__
        finally:
            try:
                if not published:
                    self._store.discard(self._target, identifier)
            except Exception as error:
                failure = type(error).__name__
            self._active.pop(identifier, None)
            self._jobs.pop(identifier, None)
            job.state = CorrectionState.FINISHED
            job.clear()
            job = None
        if failure is not None:
            self._errors += 1
            logger.warning('STT correction write deferred: %s', failure)
        return identifier if published else None

    def is_writing(self, identifier: str) -> bool:
        """Keep the file locator while its actual write can still create a WAV."""
        return identifier in self._active

    def retire_record(self, identifier: str) -> None:
        """Invalidate existing views when managed files are physically retired."""
        with self._views_lock:
            for view in tuple(self._views):
                if view.id == identifier:
                    view.clear()
                    self._views.discard(view)

    def prune(self) -> None:
        """Latch invalid writes and clear read views without waiting for native I/O."""
        for job in tuple(self._jobs.values()):
            if job.lock.acquire(blocking=False):
                try:
                    if not self._job_live(job) and job.state is CorrectionState.QUEUED:
                        job.clear()
                finally:
                    job.lock.release()
        if self._views_lock.acquire(blocking=False):
            try:
                for view in tuple(self._views):
                    if not view.available:
                        view.clear()
                        self._views.discard(view)
            finally:
                self._views_lock.release()

    def _read(self, limit, *, count_only=False):
        if self._closed:
            raise RuntimeError('stt_correction_closed')
        if not count_only:
            with self._views_lock:
                for view in tuple(self._views):
                    if not view.available:
                        view.clear()
                        self._views.discard(view)
                if len(self._views) + limit > 256:
                    raise RuntimeError('stt_correction_view_capacity')
        views = []
        count = 0
        with self._store.ledger._transaction() as db:
            rows = db.execute('''SELECT a.*,c.original_text,c.corrected_text,c.user_id,c.sample_rate,c.used
                FROM wake_clips_v1 a JOIN stt_corrections_v1 c ON c.id=a.id
                WHERE a.target_id=? AND a.published=1 AND c.used=0 AND a.expires_at>?
                ORDER BY a.sequence LIMIT ?''', (self._target.id, self._store.ledger.clock(), limit))
            for row in rows:
                sources = clip_sources(row['sources_json'])
                if not 0 < row['expires_at'] <= min(s.expires_at for s in sources) or str(UUID(row['id'])) != row['id']:
                    raise ValueError('invalid_stt_correction_metadata')
                if not self._live(sources) or self.is_writing(row['id']):
                    continue
                try:
                    with clip_directory(self._target.root) as directory:
                        if not stat.S_ISREG(os.stat(row['id']+'.wav', dir_fd=directory, follow_symlinks=False).st_mode):
                            continue
                except FileNotFoundError:
                    continue
                count += 1
                if not count_only:
                    self._store.note_deadline(row['id'], row['expires_at'])
                    view = STTCorrection(str(self.data_dir/'policy_v1'/(row['id']+'.wav')),
                        row['original_text'], row['corrected_text'], datetime.fromtimestamp(row['captured_at'], timezone.utc).isoformat(),
                        row['user_id'], row['score'], False, row['id'], sources, row['expires_at'], ref(self))
                    with self._views_lock:
                        if len(self._views) >= 256:
                            view.clear()
                            raise RuntimeError('stt_correction_view_capacity')
                        self._views.add(view)
                    views.append(view)
        return count if count_only else views

    async def get_training_pairs(self, *, limit: int = 128) -> list[STTCorrection]:
        """Return bounded, revocable views; legacy JSON never authorizes training."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('invalid_stt_correction_read_limit')
        return await self._native(self._read, limit)

    async def get_corrections_count(self) -> int:
        """Count available, unused records within this collector's 1000-record cap."""
        return await self._native(lambda: self._read(1000, count_only=True))

    def _mark(self, receipts):
        marked = set()
        with self._store.ledger._transaction() as db:
            for identifier, sources in receipts:
                row = db.execute('SELECT sources_json,expires_at FROM wake_clips_v1 WHERE target_id=? AND id=? AND published=1',
                                 (self._target.id, identifier)).fetchone()
                if row is not None and clip_sources(row['sources_json']) == sources and self._live(sources) and row['expires_at'] > self._store.ledger.clock():
                    db.execute('UPDATE stt_corrections_v1 SET used=1 WHERE id=?', (identifier,))
                    marked.add(identifier)
        return marked

    async def mark_used(self, corrections: list[STTCorrection]) -> None:
        """Use actual record receipts; caller-supplied file paths are never write authority."""
        receipts = ()
        try:
            if len(corrections) > 1000:
                raise ValueError('invalid_stt_correction_batch')
            receipts = tuple((view.id, view.sources) for view in corrections
                             if isinstance(view, STTCorrection) and view._owner is not None and view._owner() is self and view.available)
            marked = await self._native(self._mark, receipts)
            for view in corrections:
                if isinstance(view, STTCorrection) and view.id in marked:
                    view.used = True
        finally:
            corrections = receipts = None

    def _tighten_receipts(self, seconds):
        with self._views_lock:
            for view in tuple(self._views):
                view.expires_at = min(view.expires_at, min(s.captured_at for s in view.sources)+seconds)
                self._store.note_deadline(view.id, view.expires_at)
        for job in tuple(self._jobs.values()):
            job.deadline = min(job.deadline, min(s.captured_at for s in job.sources)+seconds)
            self._store.note_deadline(job.id, job.deadline)

    async def cleanup_old(self, days: float = 2) -> None:
        """Shorten raw deadlines if requested; never extend the 48-hour policy."""
        if type(days) not in (int, float) or not math.isfinite(days) or not 0 < days <= 2:
            raise ValueError('stt_correction_retention_must_be_at_most_48_hours')
        def shorten():
            with self._store.ledger._transaction() as db:
                db.execute('UPDATE wake_clips_v1 SET expires_at=MIN(expires_at,captured_at+?) WHERE target_id=?', (days*86400, self._target.id))
                for row in db.execute('SELECT id,expires_at FROM wake_clips_v1 WHERE target_id=?', (self._target.id,)):
                    self._store.note_deadline(row['id'], row['expires_at'])
                for owner in tuple(self._store._writers):
                    if isinstance(owner, STTCorrectionCollector) and owner._target.id == self._target.id:
                        owner._tighten_receipts(days*86400)
            self._store.changed()
        await self._native(shorten)
        await self._store.reconcile()

    def begin_close(self) -> asyncio.Task:
        """Close admission and keep actual native writes and reads under the lease."""
        self._closed = True
        self.prune()
        if self._closing is None:
            drain = self._work.begin_close()
            async def finish():
                await drain
                with self._views_lock:
                    for view in tuple(self._views):
                        view.clear()
                    self._views.clear()
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation without cancelling the actual native drain."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('stt_correction_native_work_still_running') from None

    def resource_status(self) -> dict:
        """Report ownership counts without transcription, user names or file paths."""
        return {'closed': self._closed, 'operations': self._work.status()['operations'],
                'capacity': self._work.status()['capacity'], 'active_writes': len(self._active),
                'pending_writes': len(self._jobs),
                'read_views': len(self._views), 'view_capacity': 256, 'errors': self._errors}


async def build_stt_correction_collector(config: dict, *, store: WakeClipStore) -> STTCorrectionCollector | None:
    """Prepare the source-aware collector without enabling a training consumer."""
    settings = config.get('training', {}).get('stt_corrections', {}) or {}
    if not settings.get('enabled', False):
        return None
    target = await store._work.run(store.register, settings.get('data_dir', './data/stt_corrections'), CollectorKind.STT_CORRECTIONS)
    return STTCorrectionCollector(target.root, settings.get('max_audio_duration', 30.), store=store, target=target)
