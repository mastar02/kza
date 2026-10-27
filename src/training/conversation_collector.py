"""Sourced conversation records and expiring, durably owned dataset exports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
from uuid import UUID, uuid4
from weakref import WeakSet, ref

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.fact_ledger import MemorySource
from src.memory.producer_context import PRODUCER_CONTEXT
from src.memory.prompt_context import merge_sources
from src.wakeword.clip_store import ClipTarget, CollectorKind, WakeClipStore, clip_directory, clip_sources

logger = logging.getLogger(__name__)
_FIELDS = frozenset({'user_input','assistant_response','intent','entities_used','correction','user_name','latency_ms'})


class ResponseQuality(StrEnum):
    UNMARKED = 'unmarked'
    GOOD = 'good'
    BAD = 'bad'
    CORRECTED = 'corrected'


@dataclass(eq=False)
class ViewScope:
    """Metadata-only receipt, shared by a copied record and its actual owner."""

    id: str
    sources: tuple[MemorySource, ...]
    expires_at: float
    owner: object
    parents: tuple[tuple[str,float], ...] = ()
    retired: threading.Event = field(default_factory=threading.Event)

    @property
    def available(self) -> bool:
        owner = self.owner()
        return bool(owner is not None and not self.retired.is_set() and owner._live(self.sources)
                    and owner._receipts_live(self.id, self.expires_at, self.parents))


@dataclass(eq=False, repr=False)
class ConversationTurn:
    """A revocable raw turn; standalone DTOs support synthetic adapters only."""

    timestamp: float
    user_input: str
    assistant_response: str
    intent: str | None = None
    entities_used: list[str] = field(default_factory=list)
    quality: ResponseQuality = ResponseQuality.UNMARKED
    correction: str | None = None
    user_name: str | None = None
    latency_ms: float | None = None
    _scope: ViewScope | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return self._scope is not None and self._scope.available

    @property
    def id(self) -> str:
        return self._scope.id if self._scope is not None else ''

    @property
    def sources(self) -> tuple[MemorySource, ...]:
        return self._scope.sources if self._scope is not None else ()

    @property
    def expires_at(self) -> float:
        return self._scope.expires_at if self._scope is not None else 0.

    def clear(self) -> None:
        """Scrub raw fields even if the caller keeps this view."""
        if self._scope is not None:
            self._scope.retired.set()
        for name in _FIELDS:
            object.__setattr__(self, name, [] if name == 'entities_used' else '' if name in {'user_input','assistant_response'} else None)

    def __getattribute__(self, name):
        if name in _FIELDS and object.__getattribute__(self, '_scope') is not None:
            if not object.__getattribute__(self, 'available'):
                object.__getattribute__(self, 'clear')()
        return object.__getattribute__(self, name)

    def __repr__(self):
        return 'ConversationTurn(managed=%s, available=%s)' % (self._scope is not None, self.available)


@dataclass
class Conversation:
    """Grouping metadata and guarded turns; no duplicate raw conversation cache."""

    id: str
    started_at: float
    turns: list[ConversationTurn] = field(default_factory=list)
    user_name: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(eq=False, repr=False)
class ConversationExport:
    """An expiring export address; the archive also removes its actual file."""

    _path: str
    exported: int
    good_responses: int
    corrected_responses: int
    format: str
    next_cursor: int
    has_more: bool
    _scope: ViewScope

    @property
    def available(self) -> bool:
        return self._scope.available

    @property
    def output_path(self) -> str:
        if not self.available:
            self.clear()
        return self._path

    @property
    def expires_at(self) -> float:
        return self._scope.expires_at

    @property
    def id(self) -> str:
        return self._scope.id

    def clear(self) -> None:
        self._scope.retired.set()
        self._path = ''


class RecordState(StrEnum):
    QUEUED = 'queued'
    RUNNING = 'running'
    FINISHED = 'finished'


class RecordAction(StrEnum):
    ADD = 'add'
    FEEDBACK = 'feedback'
    EXPORT = 'export'


@dataclass
class RecordJob:
    """One bounded native operation, including all transient raw copies."""

    id: str
    target: ClipTarget
    action: RecordAction
    sources: tuple[MemorySource, ...]
    data: dict
    room_id: str
    group_id: str = ''
    input_sources: tuple[MemorySource, ...] = ()
    quality: ResponseQuality = ResponseQuality.UNMARKED
    parents: tuple[tuple[str,float], ...] = ()
    deadline: float = 0.
    body: bytes = b''
    digest: str = ''
    state: RecordState = RecordState.QUEUED
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    score: float = 0.

    @property
    def bucket(self):
        return 'conversation_export' if self.action is RecordAction.EXPORT else 'conversation_turn'

    def clear(self) -> None:
        self.data.clear()
        self.body = b''


def _valid_data(data):
    if type(data) is not dict or set(data) != _FIELDS:
        raise ValueError('invalid_conversation_payload')
    for name, limit in (('user_input',4096),('assistant_response',8192),('correction',8192),('intent',128),('user_name',128)):
        value = data[name]
        if value is None and name not in {'user_input','assistant_response'}:
            continue
        if type(value) is not str or len(value) > limit:
            raise ValueError('invalid_conversation_text')
    if type(data['entities_used']) is not list or len(data['entities_used']) > 32 or any(type(v) is not str or len(v)>200 for v in data['entities_used']):
        raise ValueError('invalid_conversation_entities')
    latency = data['latency_ms']
    if latency is not None and (type(latency) not in (int,float) or not math.isfinite(latency) or latency < 0):
        raise ValueError('invalid_conversation_latency')


class ConversationCollector:
    """Own sourced immutable turns, feedback replacements and temporary exports."""

    def __init__(self, data_dir: str, auto_save_interval: int = 10, max_conversations_in_memory: int = 100, *,
                 store: WakeClipStore, target: ClipTarget, close_timeout_s: float = 1):
        if str(data_dir) != target.root or target.kind is not CollectorKind.CONVERSATIONS:
            raise ValueError('conversation_target_mismatch')
        if any(type(n) is not int or n <= 0 for n in (auto_save_interval,max_conversations_in_memory)):
            raise ValueError('invalid_conversation_capacity')
        if type(close_timeout_s) not in (int,float) or not math.isfinite(close_timeout_s) or close_timeout_s<=0:
            raise ValueError('invalid_conversation_close_timeout')
        self.data_dir, self._store, self._target = Path(data_dir), store, target
        self.auto_save_interval, self.max_conversations_in_memory = auto_save_interval, max_conversations_in_memory
        self._close_timeout_s = close_timeout_s
        self._work = OwnedBlockingWork(worker=store._offload, capacity=8)
        self._closed, self._closing = False, None
        self._jobs: dict[str,RecordJob] = {}
        self._active = set()
        self._views = WeakSet()
        self._views_lock = threading.Lock()
        self._current_scope = None
        self._errors = 0
        store.attach(self)

    def _live(self, sources):
        return not self._closed and not self._store._closed and self._store.privacy.sources_available(sources)

    def _receipts_live(self, identifier, deadline, parents=()):
        now = self._store.ledger.clock()
        return now < self._store.deadline_for(identifier, deadline) and all(now < self._store.deadline_for(i,d) for i,d in parents)

    def _job_live(self, job):
        if not self._live(job.sources) or not self._receipts_live(job.id, job.deadline, job.parents):
            job.cancelled.set()
        return not job.cancelled.is_set()

    def _invoke(self, function, args):
        failure = None
        with self._store.reading_session():
            try:
                return function(*args)
            except Exception as error:
                failure = type(error).__name__
            finally:
                function = args = None
        self._errors += 1
        logger.warning('Conversation operation failed: %s', failure)
        raise RuntimeError('conversation_operation_failed')

    async def _native(self, function, *args):
        try:
            return await self._work.run(self._invoke, function, args)
        finally:
            function = args = None

    def start_conversation(self, user_name: str | None = None, *, sources: tuple[MemorySource,...] = ()) -> str:
        """Select an existing source scope; never mint identity or retain a name here."""
        try:
            if user_name is not None and (type(user_name) is not str or len(user_name)>128):
                raise ValueError('invalid_conversation_name')
            if not self._live(sources):
                raise PermissionError('conversation_sources_required')
            self._current_scope = (PRODUCER_CONTEXT.get().room_id or '', sources[0].conversation_id)
            return sources[0].conversation_id
        finally:
            user_name = None

    def end_conversation(self) -> None:
        """Drop grouping metadata; each published turn was already durable."""
        self._current_scope = None

    async def add_turn(self, user_input: str, assistant_response: str, intent: str | None = None,
                       entities_used: list[str] | None = None, user_name: str | None = None,
                       latency_ms: float | None = None, *, input_sources: tuple[MemorySource,...] = (),
                       response_sources: tuple[MemorySource,...] = ()) -> str | None:
        """Publish a turn only with explicit input and full response provenance."""
        job = data = None
        try:
            if type(input_sources) is not tuple or type(response_sources) is not tuple or not self._live(input_sources) or not self._live(response_sources):
                return None
            sources = merge_sources(input_sources,response_sources)
            if len(sources)>64:
                return None
            data = dict(user_input=user_input, assistant_response=assistant_response, intent=intent,
                        entities_used=list(entities_used or ()), correction=None, user_name=user_name, latency_ms=latency_ms)
            try:
                _valid_data(data)
            except (TypeError,ValueError):
                data.clear()
                return None
            job = RecordJob(str(uuid4()), self._target, RecordAction.ADD, sources, data,
                            PRODUCER_CONTEXT.get().room_id or '', input_sources[0].conversation_id,
                            input_sources, deadline=min(s.expires_at for s in sources))
            user_input = assistant_response = intent = entities_used = user_name = latency_ms = data = None
            return await self._submit(job)
        finally:
            user_input = assistant_response = intent = entities_used = user_name = latency_ms = data = job = None

    async def _submit(self, job):
        try:
            if not self._live(job.sources) or len(self._jobs)>=8 or self._work.status()['operations']>=8:
                job.clear()
                return None
            self._jobs[job.id] = job
            return await self._native(self._run_job, job)
        finally:
            job = None

    def _metadata(self, *, group=None, room=None, identifier=None, limit=128, after=0):
        with self._store.ledger._transaction() as db:
            rows = db.execute('''SELECT a.*,t.group_id,t.input_sources_json,t.quality,t.digest
                FROM wake_clips_v1 a JOIN conversation_turns_v1 t ON t.id=a.id
                WHERE a.target_id=? AND a.published=1 AND a.expires_at>? AND a.sequence>?
                AND (? IS NULL OR t.group_id=?) AND (? IS NULL OR a.room_id=?)
                AND (? IS NULL OR a.id=?) ORDER BY a.sequence DESC LIMIT ?''',
                (self._target.id,self._store.ledger.clock(),after,group,group,room,room,identifier,identifier,limit)).fetchall()
            return [dict(row) for row in rows]

    def _meta_live(self, row):
        sources = clip_sources(row['sources_json'])
        if not 0 < row['expires_at'] <= min(s.expires_at for s in sources):
            raise ValueError('invalid_conversation_deadline')
        return self._live(sources) and self._receipts_live(row['id'],row['expires_at'])

    def _load(self, row):
        raw = envelope = data = None
        try:
            if not self._meta_live(row):
                return None
            identifier = row['id']
            if str(UUID(identifier)) != identifier:
                raise ValueError('invalid_conversation_id')
            with clip_directory(self._target.root) as directory:
                descriptor = os.open(identifier+'.json',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                with os.fdopen(descriptor,'rb') as stream:
                    raw = stream.read(65537)
            if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=row['digest']:
                raise ValueError('invalid_conversation_file_receipt')
            envelope = json.loads(raw)
            if (set(envelope)!={'version','id','sources','data'} or envelope['version']!=1
                or envelope['id']!=identifier or tuple(MemorySource.from_dict(s) for s in envelope['sources'])!=clip_sources(row['sources_json'])):
                raise ValueError('invalid_conversation_envelope')
            data = envelope['data']
            _valid_data(data)
            with self._store.ledger._transaction() as db:
                current = db.execute('SELECT published,expires_at FROM wake_clips_v1 WHERE id=? AND target_id=?', (identifier,self._target.id)).fetchone()
                if current is None or not current['published'] or current['expires_at']<=self._store.ledger.clock() or not self._meta_live(row):
                    return None
            return data
        except FileNotFoundError:
            return None
        finally:
            raw = envelope = data = None

    def _prepare_feedback(self, job):
        identifier = job.data.get('turn_id')
        rows = self._metadata(group=job.group_id if identifier is None else None,
                              room=job.room_id if identifier is None else None, identifier=identifier,limit=1)
        if not rows:
            return False
        row = rows[0]
        job.parents = ((row['id'],row['expires_at']),)
        job.sources = merge_sources(clip_sources(row['sources_json']),job.sources)
        job.deadline = min(row['expires_at'],*(s.expires_at for s in job.sources))
        self._store.note_deadline(job.id,job.deadline)
        payload = self._load(row)
        if payload is None:
            return False
        correction = job.data.get('correction')
        job.group_id = row['group_id']
        job.input_sources = clip_sources(row['input_sources_json'])
        job.room_id = row['room_id']
        payload['correction'] = correction if job.quality is ResponseQuality.CORRECTED else None
        job.data = payload
        return len(job.sources)<=64

    def _prepare_export(self, job):
        options = job.data
        rows = list(reversed(self._metadata(limit=1000,after=options['after'])))
        selected, entries = [], []
        good = corrected = 0
        try:
            for row in rows:
                if not self._meta_live(row):
                    continue
                quality = ResponseQuality(row['quality'])
                if options['only_marked'] and quality not in (ResponseQuality.GOOD,ResponseQuality.CORRECTED):
                    continue
                if quality is ResponseQuality.CORRECTED and not options['corrections']:
                    continue
                sources = merge_sources(job.sources,clip_sources(row['sources_json']))
                if len(sources)>64:
                    if selected:
                        break
                    raise ValueError('conversation_export_source_limit')
                job.sources = sources
                job.parents = (*job.parents,(row['id'],row['expires_at']))
                job.deadline = min(job.deadline,row['expires_at'])
                data = self._load(row)
                if data is None:
                    return False
                selected.append(row)
                response = data['correction'] if quality is ResponseQuality.CORRECTED else data['assistant_response']
                if options['format']=='alpaca':
                    entry = {'instruction':data['user_input'],'input':'','output':response}
                elif options['format']=='sharegpt':
                    entry = {'conversations':[{'from':'human','value':data['user_input']},{'from':'gpt','value':response}]}
                else:
                    entry = {'user':data['user_input'],'assistant':response,'intent':data['intent'],
                             'timestamp':clip_sources(row['input_sources_json'])[0].captured_at}
                entry['_kza'] = {'record_id':row['id'],'sources':[s.to_dict() for s in clip_sources(row['sources_json'])], 'expires_at':row['expires_at']}
                entries.append(entry)
                good += quality is ResponseQuality.GOOD
                corrected += quality is ResponseQuality.CORRECTED
                if len(entries)>=options['limit']:
                    break
            if not selected:
                return False
            job.parents = tuple((r['id'],r['expires_at']) for r in selected)
            job.deadline = min(r['expires_at'] for r in selected)
            self._store.note_deadline(job.id,job.deadline)
            job.body = ''.join(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n' for e in entries).encode()
            job.data = {'format':options['format'],'exported':len(entries),'good':good,'corrected':corrected,
                        'next_cursor':selected[-1]['sequence'],'has_more':selected[-1]['sequence']<rows[-1]['sequence']}
            return True
        finally:
            options = rows = selected = entries = data = entry = None

    def _encode(self, job):
        with clip_directory(job.target.root,create=True) as directory:
            descriptor = os.open(job.id+'.part',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
            with os.fdopen(descriptor,'wb') as stream:
                stream.write(job.body)
                stream.flush()
                os.fsync(stream.fileno())

    def _parents_live(self, db, job):
        for identifier,deadline in job.parents:
            row = db.execute('SELECT published,expires_at,target_id FROM wake_clips_v1 WHERE id=?', (identifier,)).fetchone()
            if row is None or not row['published'] or row['target_id']!=self._target.id or row['expires_at']<=self._store.ledger.clock() or not self._receipts_live(identifier,deadline):
                raise PermissionError('conversation_parent_invalidated')

    def _commit(self, db, job):
        self._parents_live(db,job)
        if job.action is RecordAction.FEEDBACK:
            old = job.parents[0][0]
            # Transfer the validated predecessor's raw provenance to the new
            # immutable version. Its own retirement must not cancel this job.
            job.parents = ()
            db.execute('UPDATE wake_clips_v1 SET expires_at=MIN(expires_at,?) WHERE id=?', (self._store.ledger.clock(),old))
            self._store.note_deadline(old,0.)

    def _run_job(self, job):
        identifier, published, result, failure = job.id,False,None,None
        try:
            with job.lock:
                if job.cancelled.is_set() or self._closed:
                    return None
                job.state = RecordState.RUNNING
                self._active.add(identifier)
            if job.action is RecordAction.FEEDBACK and not self._prepare_feedback(job):
                return None
            if job.action is RecordAction.EXPORT and not self._prepare_export(job):
                return None
            if not self._job_live(job):
                return None
            if job.action is not RecordAction.EXPORT:
                job.body = json.dumps({'version':1,'id':job.id,'sources':[s.to_dict() for s in job.sources], 'data':job.data},
                                      ensure_ascii=False,separators=(',',':')).encode()
                if len(job.body)>65536:
                    raise ValueError('conversation_record_size')
            if len(job.body)>2*1024*1024:
                raise ValueError('conversation_export_size')
            job.digest = hashlib.sha256(job.body).hexdigest()
            self._store.note_deadline(identifier,job.deadline)
            self._store.reserve(job.target,job)
            with self._store.ledger._transaction() as db:
                self._parents_live(db,job)
                if job.action is RecordAction.EXPORT:
                    db.executemany('INSERT INTO conversation_export_parents_v1 VALUES (?,?)', ((identifier,p) for p,_ in job.parents))
                else:
                    db.execute('INSERT INTO conversation_turns_v1 VALUES (?,?,?,?,?)',
                               (identifier,job.group_id,json.dumps([s.to_dict() for s in job.input_sources]),job.quality.value,job.digest))
            self._encode(job)
            self._store.publish(job.target,job,lambda:self._job_live(job),lambda db:self._commit(db,job))
            published = True
            self._store.changed()
            if job.action is RecordAction.EXPORT:
                scope = ViewScope(identifier,job.sources,job.deadline,ref(self),job.parents)
                result = ConversationExport(str(Path(job.target.root)/'policy_v1'/(identifier+'.jsonl')),
                    job.data['exported'],job.data['good'],job.data['corrected'],job.data['format'],
                    job.data['next_cursor'],job.data['has_more'],scope)
                with self._views_lock:
                    if len(self._views)>=256:
                        published = False
                        raise RuntimeError('conversation_view_capacity')
                    self._views.add(result)
            else:
                self._current_scope = (job.room_id,job.group_id)
                result = identifier
            self._store.rotate(job.target,job.bucket,1000 if job.action is not RecordAction.EXPORT else 32)
        except Exception as error:
            failure = type(error).__name__
        finally:
            try:
                if not published:
                    self._store.discard(job.target,identifier)
            except Exception as error:
                failure = type(error).__name__
            job.state = RecordState.FINISHED
            job.clear()
            self._jobs.pop(identifier,None)
            self._active.discard(identifier)
            job = None
        if failure is not None:
            self._errors += 1
            logger.warning('Conversation write deferred: %s',failure)
        return result

    async def mark_last_response(self, quality: str, correction: str | None = None, *,
                                 sources: tuple[MemorySource,...] | None = None) -> bool:
        """Mark only the latest live turn in this exact room/conversation scope."""
        return await self._feedback(None,quality,correction,sources)

    async def mark_response(self, view: ConversationTurn, quality: str, correction: str | None = None, *,
                            sources: tuple[MemorySource,...] | None = None) -> bool:
        """Use a genuine owned view when feedback comes from a separate interaction."""
        try:
            if not isinstance(view,ConversationTurn) or view._scope is None or view._scope.owner() is not self or not view.available:
                return False
            return await self._feedback(view.id,quality,correction,sources)
        finally:
            view = correction = None

    async def _feedback(self, identifier, quality, correction, sources):
        job = None
        try:
            sources = PRODUCER_CONTEXT.get().sources if sources is None else sources
            if type(sources) is not tuple or not self._live(sources) or quality not in {'good','bad','corrected'}:
                return False
            if quality=='corrected' and (type(correction) is not str or not correction.strip() or len(correction)>8192):
                return False
            job = RecordJob(str(uuid4()),self._target,RecordAction.FEEDBACK,sources,
                {'turn_id':identifier,'correction':correction},PRODUCER_CONTEXT.get().room_id or '',
                sources[0].conversation_id,quality=ResponseQuality(quality),deadline=min(s.expires_at for s in sources))
            correction = None
            return bool(await self._submit(job))
        finally:
            correction = job = None

    def _get_turns(self, limit, group):
        with self._views_lock:
            for view in tuple(self._views):
                if not view.available:
                    view.clear()
                    self._views.discard(view)
            if len(self._views)+limit>256:
                raise RuntimeError('conversation_view_capacity')
        views = []
        for row in self._metadata(group=group,limit=limit):
            with self._store.reading(self._target, clip_sources(row['sources_json'])) as admitted:
                if not admitted:
                    continue
                data = None
                try:
                    data = self._load(row)
                    if data is None:
                        continue
                    with self._store.ledger._transaction() as db:
                        current = db.execute('SELECT published,expires_at FROM wake_clips_v1 WHERE id=?', (row['id'],)).fetchone()
                        if current is None or not current['published'] or current['expires_at']<=self._store.ledger.clock() or not self._meta_live(row):
                            continue
                        scope = ViewScope(row['id'],clip_sources(row['sources_json']),row['expires_at'],ref(self))
                        self._store.note_deadline(scope.id,scope.expires_at)
                        view = ConversationTurn(clip_sources(row['input_sources_json'])[0].captured_at,**data,
                                                quality=ResponseQuality(row['quality']),_scope=scope)
                        with self._views_lock:
                            if len(self._views)>=256:
                                view.clear()
                                raise RuntimeError('conversation_view_capacity')
                            self._views.add(view)
                        views.append(view)
                finally:
                    if data is not None:
                        data.clear()
                    data = None
        return views

    async def get_turns(self, *, limit: int = 128, conversation_id: str | None = None) -> list[ConversationTurn]:
        """Read bounded, revocable views without loading legacy files."""
        if type(limit) is not int or not 1<=limit<=128:
            raise ValueError('invalid_conversation_read_limit')
        return await self._native(self._get_turns,limit,conversation_id)

    async def get_conversations(self) -> list[Conversation]:
        """Group guarded turns without copying names or text into another cache."""
        groups = {}
        for turn in await self.get_turns():
            identifier = turn.sources[0].conversation_id
            groups.setdefault(identifier,Conversation(identifier,turn.timestamp)).turns.append(turn)
        return list(groups.values())

    def _stats(self):
        counts = {quality.value:0 for quality in ResponseQuality}
        groups, current = set(),0
        for row in self._metadata(limit=1000):
            if not self._meta_live(row):
                continue
            quality = ResponseQuality(row['quality'])
            counts[quality.value] += 1
            groups.add(row['group_id'])
            current += self._current_scope==(row['room_id'],row['group_id'])
        return {'total_conversations':len(groups),'total_turns':sum(counts.values()),'current_conversation_turns':current,
                'quality_distribution':counts,'marked_for_training':counts['good']+counts['corrected'],
                'legacy_policy':'quarantined'}

    async def get_stats(self) -> dict:
        """Read numeric policy metadata; statistics never authorize a model input."""
        return await self._native(self._stats)

    async def export_for_training(self, output_dir: str, format: str = 'alpaca', only_marked: bool = True,
                                  include_corrections: bool = True, *, limit: int = 32, after_sequence: int = 0) -> ConversationExport | None:
        """Create a bounded temporary dataset under a registered output directory."""
        if format not in {'alpaca','sharegpt','conversations'} or type(limit) is not int or not 1<=limit<=32 or type(after_sequence) is not int or after_sequence<0:
            raise ValueError('invalid_conversation_export_options')
        if self._closed:
            return None
        target = await self._store._work.run(self._store.register,output_dir,CollectorKind.CONVERSATION_EXPORTS)
        # An export acquires its actual sources inside the owned native reader,
        # before reserve/encode. Empty lineage cannot authorize publication.
        job = RecordJob(str(uuid4()),target,RecordAction.EXPORT,(),
            {'format':format,'only_marked':only_marked,'corrections':include_corrections,'limit':limit,'after':after_sequence},'',deadline=float('inf'))
        try:
            if self._closed or len(self._jobs)>=8 or self._work.status()['operations']>=8:
                return None
            self._jobs[job.id] = job
            return await self._native(self._run_job,job)
        finally:
            job = None

    def is_writing(self, identifier: str) -> bool:
        return identifier in self._active

    def has_conversation_work(self, target_id: str, conversation_id: str) -> bool:
        """Include exports/feedback with their original sources before raw reads."""
        return any(job.target.id == target_id and any(source.conversation_id == conversation_id
                   or source.fact_dependencies and self._store.ledger.dependencies_committed_unavailable(source.fact_dependencies)
                   for source in job.sources)
                   for job in tuple(self._jobs.values()))

    def retire_record(self, identifier: str) -> None:
        """Invalidate turn/export views and providers depending on retired records."""
        for job in tuple(self._jobs.values()):
            if identifier==job.id or any(i==identifier for i,_ in job.parents):
                job.cancelled.set()
        with self._views_lock:
            for view in tuple(self._views):
                if identifier==view.id or any(i==identifier for i,_ in view._scope.parents):
                    view.clear()
                    self._views.discard(view)

    def prune(self) -> None:
        """Clear queued raw data independently of blocked native file work."""
        for job in tuple(self._jobs.values()):
            if job.action is RecordAction.EXPORT and not job.sources and not self._closed:
                continue
            if job.lock.acquire(blocking=False):
                try:
                    if not self._job_live(job) and job.state is RecordState.QUEUED:
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

    def begin_close(self) -> asyncio.Task:
        self._closed = True
        self._current_scope = None
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
        """Bound observation while retaining actual native work under the lease."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('conversation_native_work_still_running') from None

    def resource_status(self) -> dict:
        return {'closed':self._closed,'operations':self._work.status()['operations'],'capacity':8,
                'pending_writes':len(self._jobs),'active_writes':len(self._active),
                'read_views':len(self._views),'view_capacity':256,'errors':self._errors}

    @staticmethod
    def get_training_commands() -> str:
        return 'Feedback: «buena respuesta», «mala respuesta» o «debiste decir …». Las marcas no prolongan el vencimiento.'


async def build_conversation_collector(config: dict, *, store: WakeClipStore) -> ConversationCollector:
    """Prepare policy independently of the unverified nightly-training profile."""
    settings = config.get('training',{})
    target = await store._work.run(store.register,settings.get('conversations_dir','./data/conversations'),CollectorKind.CONVERSATIONS)
    return ConversationCollector(target.root,settings.get('auto_save_interval',10),
        settings.get('max_conversations_in_memory',100),store=store,target=target)


# Revision pineada del modelo base default (commit del branch main,
# verificado 2026-08-02 vía GET
# https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct ->
# campo "sha"). B615 (bandit): from_pretrained/load_dataset sin revision
# resuelve "latest" en cada descarga, no reproducible y no auditable — un
# --base-model distinto del default no hereda este pin (no existe un pin
# "genérico" válido para un repo arbitrario del Hub), por eso LoRATrainer
# exige revision= explícita en ese caso en vez de caer a "sin pinear".
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


class LoRATrainer:
    """
    Entrena el modelo con LoRA usando los datos recolectados.

    LoRA (Low-Rank Adaptation) permite fine-tuning eficiente
    usando solo una fracción de los parámetros del modelo.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        revision: str | None = None,
        output_dir: str = "./models/lora_adapters",
        device: str = "cuda:0"
    ):
        if revision is None:
            if base_model == DEFAULT_BASE_MODEL:
                revision = DEFAULT_BASE_MODEL_REVISION
            else:
                raise ValueError(
                    f"LoRATrainer: base_model='{base_model}' distinto del "
                    f"default ({DEFAULT_BASE_MODEL}) requiere revision= "
                    "explícita — no hay una revisión pineada válida para un "
                    "repo arbitrario del Hub (evita reintroducir B615 sin "
                    "pinear)."
                )

        self.base_model = base_model
        self.revision = revision
        self.output_dir = Path(output_dir)
        self.device = device

    @property
    def training_blocked(self) -> bool:
        """No private dataset/model session has passed release acceptance yet."""
        return True

    def train(
        self,
        training_data_path: str,
        adapter_name: str = "custom_adapter",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ) -> dict:
        """
        Entrenar adapter LoRA.

        Args:
            training_data_path: Ruta a los datos de entrenamiento (.jsonl)
            adapter_name: Nombre del adapter resultante
            epochs: Número de épocas
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
            lora_r: Rango de LoRA (menor = más ligero)
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA

        Returns:
            Información del entrenamiento
        """
        # A copied export loses its canonical path, but not its privacy policy.
        # No backend is admitted until datasets and models can obey the same
        # source lifetime. Keep the legacy implementation below for that review.
        if self.training_blocked:
            raise RuntimeError('private_training_session_unavailable')
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import load_dataset
            import torch

        except ImportError as e:
            logger.error(f"Dependencias no instaladas: {e}")
            return {
                "success": False,
                "error": "Instala: pip install transformers peft datasets accelerate"
            }

        logger.info(f"Iniciando entrenamiento LoRA para {self.base_model}")
        logger.info(f"Datos: {training_data_path}")

        # Cargar dataset ("json" es el builder local built-in de datasets,
        # data_files es un path local: no hay descarga del Hub acá)
        dataset = load_dataset("json", data_files=training_data_path)  # nosec B615 -- builder local "json", data_files es un path local, no descarga del Hub

        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, revision=self.revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Función de tokenización
        def tokenize_function(examples):
            # Formato Alpaca
            prompts = []
            for instruction, output in zip(examples["instruction"], examples["output"]):
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                prompts.append(prompt)

            return tokenizer(
                prompts,
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Cargar modelo base
        logger.info("Cargando modelo base...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            revision=self.revision,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Configurar LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Configurar entrenamiento
        output_path = self.output_dir / adapter_name
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
        )

        # Entrenar
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer
        )

        logger.info("Iniciando entrenamiento...")
        trainer.train()

        # Guardar adapter
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Adapter guardado en: {output_path}")

        return {
            "success": True,
            "adapter_path": str(output_path),
            "epochs": epochs,
            "trainable_params": model.num_parameters(only_trainable=True)
        }

    def list_adapters(self) -> list[dict]:
        """Listar adapters entrenados"""
        if self.training_blocked:
            raise RuntimeError('private_training_session_unavailable')
        adapters = []

        for adapter_dir in self.output_dir.iterdir():
            if adapter_dir.is_dir():
                config_file = adapter_dir / "adapter_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)

                    adapters.append({
                        "name": adapter_dir.name,
                        "path": str(adapter_dir),
                        "lora_r": config.get("r"),
                        "lora_alpha": config.get("lora_alpha"),
                        "base_model": config.get("base_model_name_or_path")
                    })

        return adapters
