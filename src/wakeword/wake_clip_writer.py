"""Bounded wake clip collection with original source and actual native ownership."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
import logging
import math
import os
import threading
from uuid import uuid4
import wave

import numpy as np

from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT
from src.wakeword.clip_store import CollectorKind, ClipTarget, WakeClipStore, clip_directory

logger = logging.getLogger(__name__)


class ClipBucket(StrEnum):
    """Independent policy datasets sharing the same source deadlines."""

    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    MISSED = 'missed'


@dataclass(frozen=True)
class ClipJob:
    """An owned waveform with original, immutable source metadata."""

    id: str
    room_id: str
    score: float
    audio: np.ndarray
    sources: tuple[MemorySource, ...]
    bucket: ClipBucket
    cancelled: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)


def encode_wav(target: ClipTarget, identifier: str, audio: np.ndarray, sample_rate: int) -> None:
    """Encode owned PCM to a registered private part file, entirely off loop."""
    pcm = file = None
    try:
        pcm = (np.clip(audio, -1., 1.) * 32767.).astype(np.int16)
        with clip_directory(target.root, create=True) as directory:
            descriptor = os.open(identifier + '.part', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            with os.fdopen(descriptor, 'wb') as file:
                with wave.open(file, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(pcm.tobytes())
                file.flush()
                os.fsync(file.fileno())
    finally:
        audio = pcm = file = None


class WakeClipWriter:
    """Copy bounded inputs and retain the real thread until files are finished.

    Construction has no disk I/O. The store and canonical target are initialized
    by the application before capture. An unsourced input is never collected.
    """

    def __init__(self, directory, sample_rate: int = 16000, max_files: int = 2000,
                 max_rejected_files: int = 4000, max_missed_files: int = 300,
                 queue_size: int = 8, *, store: WakeClipStore, target: ClipTarget,
                 close_timeout_s: float = 1):
        if str(directory) != target.root or target.kind is not CollectorKind.WAKE_CLIPS:
            raise ValueError('wake_clip_target_mismatch')
        if any(type(n) is not int or n <= 0 for n in
               (sample_rate, max_files, max_rejected_files, max_missed_files, queue_size)):
            raise ValueError('invalid_wake_clip_capacity')
        if type(close_timeout_s) not in (int, float) or not math.isfinite(close_timeout_s) or close_timeout_s <= 0:
            raise ValueError('invalid_wake_clip_close_timeout')
        self._store, self._target = store, target
        self._sample_rate, self._capacity = sample_rate, queue_size
        self._caps = {ClipBucket.ACCEPTED: max_files, ClipBucket.REJECTED: max_rejected_files, ClipBucket.MISSED: max_missed_files}
        self._queue = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._busy = None
        self._running = True
        self._closing = None
        self._close_timeout_s = close_timeout_s
        self._dropped = self._written = self._errors = 0
        self._worker = threading.Thread(target=self._run, daemon=True, name='wake-clip-writer')
        store.attach(self)
        self._worker.start()

    def _live(self, sources) -> bool:
        return self._running and not self._store._closed and self._store.privacy.sources_available(sources)

    def _job_live(self, job) -> bool:
        if not self._live(job.sources):
            job.cancelled.set()
        return not job.cancelled.is_set()

    def submit(self, room_id: str, score: float, audio, accepted: bool = True, *, sources=None) -> bool:
        """Copy an accepted/rejected wake with its inherited acquisition lineage."""
        try:
            return self._enqueue(room_id, score, audio, ClipBucket.ACCEPTED if accepted else ClipBucket.REJECTED, sources)
        finally:
            audio = None

    def submit_missed(self, room_id: str, score: float, audio, *, sources=None) -> bool:
        """Copy a textual wake's original sources without minting a new capture."""
        try:
            return self._enqueue(room_id, score, audio, ClipBucket.MISSED, sources)
        finally:
            audio = None

    def _enqueue(self, room_id, score, audio, bucket, sources):
        clip = None
        admitted = False
        try:
            sources = PRODUCER_CONTEXT.get().sources if sources is None else sources
            if type(sources) is not tuple or not 1 <= len(sources) <= 64 or not self._live(sources):
                return False
            if type(room_id) is not str or len(room_id) > 200 or not -1 <= float(score) <= 1 or not math.isfinite(float(score)):
                return False
            if not 0 < len(audio) <= self._sample_rate * 30:
                return False
            clip = np.asarray(audio, dtype=np.float32).copy()
            if clip.ndim != 1 or not self._live(sources) or not self._lock.acquire(blocking=False):
                return False
            try:
                if not self._live(sources) or len(self._queue) >= self._capacity:
                    return False
                self._queue.append(ClipJob(str(uuid4()), room_id, float(score), clip, sources, bucket))
                self._wake.set()
                admitted = True
                return True
            finally:
                self._lock.release()
        except Exception:
            # Submission may come from a device callback; logging is disk I/O.
            self._errors += 1
            return False
        finally:
            if not admitted:
                self._dropped += 1
            audio = clip = None

    def prune(self) -> None:
        """Drop expired/revoked queued waves without waiting for a blocked encoder."""
        busy = self._busy
        if busy is not None:
            self._job_live(busy)
        if self._lock.acquire(blocking=False):
            try:
                before = len(self._queue)
                self._queue = deque(j for j in self._queue if self._job_live(j))
                self._dropped += before - len(self._queue)
            finally:
                self._lock.release()

    def _run(self):
        job = None
        try:
            while self._running:
                with self._lock:
                    job = self._queue.popleft() if self._queue else None
                    self._busy = job
                if job is None:
                    self._wake.wait(.05)
                    self._wake.clear()
                    continue
                failure = None
                try:
                    if self._job_live(job):
                        self._write(job)
                    else:
                        self._dropped += 1
                except Exception as error:
                    self._errors += 1
                    failure = type(error).__name__
                finally:
                    job = self._busy = None
                # Drop the exception/traceback and raw job before a diagnostic
                # sink can block. Logging owns only a fixed message and type.
                if failure is not None:
                    logger.warning('Wake clip write deferred: %s', failure)
        finally:
            job = self._busy = None
            self._queue.clear()

    def _encode(self, job):
        encode_wav(self._target, job.id, job.audio, self._sample_rate)

    def _write(self, job):
        published = False
        try:
            self._store.reserve(self._target, job)
            if not self._job_live(job):
                return
            self._encode(job)
            self._store.publish(self._target, job, lambda: self._job_live(job))
            published = True
            self._written += 1
            self._store.rotate(self._target, job.bucket, self._caps[job.bucket])
        finally:
            if not published:
                self._store.discard(self._target, job.id)
            job = None

    def is_writing(self, identifier: str) -> bool:
        """Report only whether a locator still has a real native writer."""
        busy = self._busy
        return busy is not None and busy.id == identifier

    def _retire(self):
        self._running = False
        self._dropped += len(self._queue)
        self._queue = deque()
        self._wake.set()

    def stop(self, timeout: float = 2.) -> None:
        """Synchronous adapter for native callers; timeout never means thread exit."""
        self._retire()
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise WorkCloseTimeout('wake_clip_writer_still_running')

    def begin_close(self) -> asyncio.Task:
        """Reject input, discard queued waves and retain the actual native join."""
        self._retire()
        if self._closing is None:
            self._closing = asyncio.create_task(self._store._work.run(self._worker.join))
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while retaining the writer under the shared lease."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('wake_clip_writer_still_running') from None

    def resource_status(self) -> dict:
        """Report capacity and ownership without sources, rooms, handles or paths."""
        return {'closed': not self._running, 'queued': len(self._queue), 'capacity': self._capacity,
                'native_running': self._worker.is_alive(), 'busy': self._busy is not None,
                'written': self._written, 'dropped': self._dropped, 'errors': self._errors}
