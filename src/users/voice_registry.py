"""Bounded background voice observations under the leased privacy authority."""

import asyncio
from collections import deque
from contextvars import Context
from dataclasses import dataclass,field
from enum import StrEnum
import hashlib
import json
from functools import partial
import threading

import numpy as np

from src.core.owned_work import OwnedBlockingWork,WorkCloseTimeout
from src.memory.erasure_journal import ErasureParticipant,ErasureResult
from src.memory.producer_context import model_admission
from src.memory.fact_ledger import MemoryOrigin,MemorySource
from src.users.voice_ledger import VoiceLedger
from src.users.voice_capture import build_review_capture

VOICE_TARGET = {'kind':'source_bound_voice_registry','schema':1}


class VoiceFailure(StrEnum):
    FAILED = 'voice_operation_failed'


@dataclass(repr=False)
class VoiceObservation:
    """Actual native work owns its copied PCM until the model and commit exit."""

    identifier: object
    lock: object
    sources: tuple
    audio: np.ndarray | None = field(repr=False)
    transcript: str = field(default='',repr=False)
    room_id: str | None = None


class VoiceRegistry:
    """Collect fingerprints without blocking commands or granting voice authority."""

    def __init__(self, privacy):
        self.privacy = privacy
        self.ledger = VoiceLedger(privacy.ledger)
        self._work = OwnedBlockingWork(capacity=6)
        self._disk = OwnedBlockingWork(capacity=2)
        self._free = deque(range(6))
        self._jobs = {}
        self._tasks = set()
        self._io_tasks = set()
        self._encoders = {}
        self._loop = asyncio.get_running_loop()
        self._maintenance = self._closing = None
        self._closed = False
        self._suspended_origins = frozenset()
        self._dropped = self._failed = 0

    def register_encoder(self, identifier) -> threading.Lock:
        """Share the actual inference lock with existing command/ambient owners."""
        if self._closed:
            raise RuntimeError('voices_closed')
        if (type(getattr(identifier,'model_name',None)) is not str or not 1<=len(identifier.model_name)<=256
                or type(getattr(identifier,'sample_rate',None)) is not int or identifier.sample_rate!=16000):
            raise ValueError('invalid_voice_encoder')
        key = id(identifier)
        if key not in self._encoders:
            if len(self._encoders)>=8:
                raise RuntimeError('voice_encoder_capacity')
            self._encoders[key] = (identifier,threading.Lock())
        return self._encoders[key][1]

    def observe(self, identifier, audio: np.ndarray, sources: tuple, *, sample_rate: int = 16000,
            transcript: str = '', room_id: str | None = None) -> bool:
        """Reserve bounded PCM ownership and return without model or disk I/O."""
        registration = self._encoders.get(id(identifier))
        if (self._closed or registration is None or type(sample_rate) is not int or sample_rate!=16000
                or type(audio) is not np.ndarray
                or audio.dtype!=np.float32 or audio.ndim!=1 or not .8*16000<=audio.size<=30*16000
                or type(sources) is not tuple or not 1<=len(sources)<=256
                or type(transcript) is not str or len(transcript)>8192
                or room_id is not None and (type(room_id) is not str or not 1<=len(room_id)<=128)
                or any(type(source) is not MemorySource for source in sources)
                or any(source.origin in self._suspended_origins for source in sources)
                or not np.isfinite(audio).all() or not self.privacy.sources_available(sources)):
            self._dropped+=1
            return False
        try:
            slot = self._free.popleft()
        except IndexError:
            self._dropped+=1
            return False
        if self._closed:
            self._free.append(slot)
            return False
        try:
            job = VoiceObservation(identifier,registration[1],sources,audio.copy(),transcript,room_id)
            self._jobs[slot] = job
            self._loop.call_soon_threadsafe(self._start,slot,job,context=Context())
        except Exception:
            self._jobs.pop(slot,None)
            self._free.append(slot)
            self._dropped+=1
            return False
        return True

    def suspend_origin(self, origin: MemoryOrigin) -> None:
        """Seal one recording lane while keeping command observations available."""
        self._suspended_origins = self._suspended_origins | {origin}

    def pending_origin(self, origin: MemoryOrigin) -> int:
        """Count actual retained PCM/model/commit owners for this lane."""
        return sum(any(source.origin is origin for source in job.sources) for job in self._jobs.values())

    def resume_origin(self, origin: MemoryOrigin) -> None:
        """Old model work must finish before its lane may accept new captures."""
        if self.pending_origin(origin):
            raise RuntimeError('voice_lane_still_owned')
        self._suspended_origins = self._suspended_origins - {origin}

    def _start(self, slot, job):
        task = asyncio.create_task(self._work.run(self._observe_native,job),context=Context())
        self._tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            self._jobs.pop(slot,None)
            self._free.append(slot)
            if done.cancelled() or done.exception() is not None or done.result() is VoiceFailure.FAILED:
                self._failed+=1
        task.add_done_callback(finished,context=Context())

    def _observe_native(self, job):
        vector = review = None
        def require_live():
            if self._closed or any(source.origin in self._suspended_origins for source in job.sources) or not self.privacy.sources_available(job.sources):
                raise RuntimeError('voice_sources_unavailable')
        try:
            with model_admission(job.lock,require_live):
                vector = job.identifier.get_embedding(job.audio)
                require_live()
                descriptor = [job.identifier.model_name,getattr(job.identifier,'_model_type','unknown'),job.identifier.sample_rate]
                model_key = 'enc_'+hashlib.sha256(json.dumps(descriptor).encode()).hexdigest()
                review = build_review_capture(job.audio,job.transcript,job.room_id)
                require_live()
                return self.ledger.put(model_key,vector,job.audio.size/job.identifier.sample_rate,job.sources,review)
        except (Exception,asyncio.CancelledError):
            return VoiceFailure.FAILED
        finally:
            job.audio = None
            job.transcript = ''
            job = vector = review = require_live = None

    async def acall(self, function, *args):
        """Run bounded registry I/O without retaining native exception payloads."""
        if self._closed:
            raise RuntimeError('voices_closed')
        if len(self._io_tasks)>=2:
            raise RuntimeError('voice_operation_capacity')
        task = asyncio.create_task(self._call_native(function,args),context=Context())
        self._tasks.add(task)
        self._io_tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            self._io_tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished,context=Context())
        try:
            return await asyncio.shield(task)
        finally:
            task = function = args = None

    async def _call_native(self, function, args):
        try:
            return await self._disk.run(function,*args)
        finally:
            function = args = None

    async def load_accounts(self, path: str):
        """Read role metadata off-loop while quarantining legacy voice vectors."""
        from src.users.user_manager import UserManager
        return await self.acall(partial(UserManager,path,voice_registry=self))

    async def clean(self, step) -> ErasureResult:
        """A local voice receipt needs both SQLite cleanup and exited model work."""
        if step.participant is not ErasureParticipant.VOICES or step.version!=1 or step.descriptor!=VOICE_TARGET:
            raise ValueError('unsupported_voice_target')
        if not self.privacy.ledger.conversation_committed_revoked(step.conversation_id):
            raise RuntimeError('voice_erasure_requires_committed_authority')
        targets = await self.privacy.acall(self.privacy.ledger.erasures.targets,ErasureParticipant.VOICES)
        if not any(row['id']==step.target_id and row['version']==1 and row['descriptor']==VOICE_TARGET for row in targets):
            raise ValueError('unregistered_voice_target')
        await self.acall(self.ledger.prune)
        pending = await self.acall(self.ledger.pending,step.conversation_id)
        pending+=sum(any(source.conversation_id==step.conversation_id for source in job.sources) for job in self._jobs.values())
        pending+=sum(not task.done() for task in self._io_tasks)
        return ErasureResult(pending==0,pending_count=pending)

    def start(self) -> None:
        """Expire vectors on idle as well as on read and restart."""
        async def maintain():
            while not self._closed:
                try:
                    await self.acall(self.ledger.prune)
                except Exception:
                    self._failed+=1
                await asyncio.sleep(1)
        self._maintenance = asyncio.create_task(maintain(),context=Context())

    def resource_status(self) -> dict:
        """Expose counts, never PCM, vectors, names, models or storage paths."""
        return {'closed':self._closed,'capacity':6,'io_capacity':2,'pending':6-len(self._free),
            'operations':len(self._tasks),'dropped':self._dropped,'failed':self._failed}

    def begin_close(self) -> asyncio.Task:
        """Seal native posts and retain every reserved or actual operation."""
        self._closed = True
        if self._closing is None:
            async def finish():
                if self._maintenance is not None:
                    self._maintenance.cancel()
                    await asyncio.gather(self._maintenance,return_exceptions=True)
                while len(self._free)!=6 or self._tasks:
                    await asyncio.sleep(.005)
                await self._work.begin_close()
                await self._disk.begin_close()
                self._encoders.clear()
            self._closing = asyncio.create_task(finish(),context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Observation timeout cannot release actual model or SQLite ownership."""
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('voice_resources_still_running') from None


async def build_voice_registry(*, privacy, owner, shutdown) -> VoiceRegistry:
    """Bind the local registry to the mandatory authority before audio starts."""
    voices = VoiceRegistry(privacy)
    privacy.add_borrower(voices.begin_close)
    shutdown.push_async_callback(voices.aclose)
    owner.register(ErasureParticipant.VOICES,voices.clean,versions=frozenset({1}),begin_close=voices.begin_close)
    await privacy.acall(privacy.ledger.erasures.register_target,ErasureParticipant.VOICES,1,VOICE_TARGET)
    await privacy.acall(privacy.ledger.erasures.register_target,ErasureParticipant.VOICES,2,{'kind':'unresolved_voice_inventory'})
    await voices.acall(voices.ledger.prune)
    voices.start()
    return voices
