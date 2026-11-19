"""Version-three voice REST bodies share the retained socket/ASGI send owner."""

import asyncio
from functools import partial
import json
import math

from starlette.responses import Response

from src.dashboard.rest_delivery import RestOperation,RestResult
from src.users.voice_projection import VoiceProjection, VoiceQuerySummary
from src.dashboard.live_retention_types import RevisionSet, RevisionStamp, VoicePersonFence

CAPTURE_FIELDS = frozenset({'id','group_id','created_at','expires_at','duration_s','model_key','capture_ids',
    'review_available','room_id','suggested_profile','similarity','association','group_revision','group_state',
    'group_reason','group_size','group_similarity','group_expires_at'})
REVIEW_FIELDS = {
    'audio':frozenset({'id','observed_at','expires_at','transcript','room_id','duration_s','clip_duration_s',
        'clip_truncated','transcript_scope','audio_mime','audio_base64','audio_metadata','transcript_metadata'}),
    # QB: the text variant never carries audio, its MIME, metadata or clip fields.
    'text':frozenset({'id','observed_at','expires_at','transcript','room_id','duration_s','transcript_scope',
        'transcript_metadata','variant'}),
}


def _room(value) -> bool:
    return value is None or (type(value) is str and 1<=len(value)<=128
        and not any(ord(char)<32 or ord(char)==127 for char in value))


def _deadline(value, observed) -> bool:
    return value is None or type(value) in (int,float) and math.isfinite(value) and value>observed


class VoiceResponse(Response):
    """Read only within the actual retained outer HTTP scope and page connection."""

    def __init__(self, owner, voices, connection, kind, cursor, actor, *, reader, include_person: bool = False,
                 person_id: str | None = None, variant: str = 'audio', query=None):
        super().__init__(content=b'')
        if type(include_person) is not bool or include_person and kind != 'profiles':
            raise ValueError('invalid_voice_person_projection')
        # T044-U1: 'text' is the QB transcript read on voices:review; a query is a
        # filtered captures page (its filters stay in memory, never in the URL).
        if variant not in REVIEW_FIELDS or variant=='text' and kind!='review' or query is not None and (
                kind!='captures' or include_person):
            raise ValueError('invalid_voice_projection')
        self.variant,self.query = variant,query
        self.owner,self.voices,self.connection = owner,voices,connection
        self.kind,self.cursor,self.actor = kind,cursor,actor
        self.reader = reader
        if person_id is not None and (not include_person or type(person_id) is not str
                or not 1<=len(person_id)<=128 or any(ord(char)<32 or ord(char)==127 for char in person_id)):
            raise ValueError('invalid_voice_person_projection')
        self.include_person = include_person
        self.person_id = person_id

    async def __call__(self, scope, receive, send):
        """An observer cannot retire a native read or actual body callback early."""
        operation = scope.get('kza.rest_operation')
        try:
            if not isinstance(operation,RestOperation):
                await self.owner._error(send,503)
                return
            await self._deliver(operation,send)
        finally:
            self.owner = self.voices = self.connection = self.kind = self.cursor = self.actor = None
            self.reader = self.person_id = self.query = None

    async def _deliver(self, operation, send):
        projection = state = ticket = body = snapshot = row = link = profile_fields = summary = document = None
        started = False
        try:
            if self.include_person:
                self.actor.require_people()
            operation.channel = 'voices:'+self.kind
            state = self.owner.connection(self.connection,reader=self.reader,minimum_protocol=3,channel=operation.channel)
            operation.connection = self.connection
            operation.reader = self.reader
            credential = self.actor._current
            operation.guard = lambda:credential is None or credential() is True
            if (state is None or self.voices.privacy.ledger is not self.owner.retention.policy
                    or self.owner.retention._closed):
                started = True
                await self.owner._error(send,428)
                return RestResult.FAILED
            operation.private_envelope = True
            if self.query is not None:
                projection = await self.voices.acall(self.voices.ledger.query_captures,self.query,self.cursor,self.actor)
            elif self.variant=='text':
                projection = await self.voices.acall(partial(self.voices.ledger.project,variant='text'),self.kind,self.cursor,self.actor)
            elif self.person_id is not None:
                projection = await self.voices.acall(self.voices.ledger.project,self.kind,self.cursor,self.actor,True,self.person_id)
            elif self.include_person:
                projection = await self.voices.acall(self.voices.ledger.project,self.kind,self.cursor,self.actor,True)
            else:
                projection = await self.voices.acall(self.voices.ledger.project,self.kind,self.cursor,self.actor)
            if self.owner.connection(self.connection,reader=self.reader,minimum_protocol=3,channel=operation.channel) is not state:
                raise PermissionError('dashboard_reader_retired')
            if (type(projection) is not VoiceProjection or projection.channel!='voices:'+self.kind
                    or type(projection.rows) is not list or len(projection.rows)>(1 if self.kind=='review' else 50)
                    or bool(projection.profile_refs)!=(self.kind=='profiles' and bool(projection.rows))):
                raise ValueError('invalid_voice_projection')
            profile_fields = {'id','name','revision','confirmed_at','confirmed_by','expires_at'}
            if self.include_person:
                profile_fields.add('person_link')
            if self.kind=='profiles' and any(type(row) is not dict or set(row)!=profile_fields for row in projection.rows):
                raise ValueError('invalid_profile_metadata')
            if self.include_person:
                for row in projection.rows:
                    link = row['person_link']
                    if (type(link) is not dict or set(link)!={'user_id','revision'}
                            or self.person_id is not None and link['user_id'] != self.person_id
                            or type(link['revision']) is not int or link['revision']<0
                            or link['user_id'] is not None and link['revision']<1
                            or link['user_id'] is not None and (type(link['user_id']) is not str
                                or not 1<=len(link['user_id'])<=128
                                or any(ord(char)<32 or ord(char)==127 for char in link['user_id']))):
                        raise ValueError('invalid_voice_person_metadata')
                stamps = projection.revision.stamps if type(projection.revision) is RevisionSet else (projection.revision,)
                ledger_stamps = [stamp for stamp in stamps if type(stamp) is RevisionStamp and stamp.producer() is self.voices.ledger]
                person_stamps = [stamp for stamp in stamps if type(stamp) is VoicePersonFence]
                identities = {row['person_link']['user_id'] for row in projection.rows if row['person_link']['user_id'] is not None}
                if self.person_id is not None:
                    identities.add(self.person_id)
                identities = tuple(sorted(identities))
                if (len(ledger_stamps)!=1 or len(person_stamps)!=bool(identities)
                        or identities and (person_stamps[0].identities!=identities
                            or person_stamps[0].authority() is not self.actor._person_authority())):
                    raise ValueError('invalid_voice_person_fence')
            if self.kind=='profiles' and set(projection.profile_refs)!={(row['id'],row['revision']) for row in projection.rows}:
                raise ValueError('invalid_profile_revisions')
            if self.kind=='profiles' and any(type(row['name']) is not str or not 1<=len(row['name'])<=64
                    or type(row['confirmed_by']) is not str or not 1<=len(row['confirmed_by'])<=256
                    or type(row['confirmed_at']) not in (int,float) or not math.isfinite(row['confirmed_at'])
                    or row['confirmed_at']>projection.observed_at or row['expires_at'] is not None for row in projection.rows):
                raise ValueError('invalid_profile_metadata')
            if (bool(projection.associated_profiles)!=bool(projection.activity_sources)
                    or projection.associated_profiles and self.kind!='captures'):
                raise ValueError('invalid_voice_associations')
            if self.kind=='captures' and (any(type(row) is not dict or set(row)!=CAPTURE_FIELDS
                    or not _room(row['room_id']) for row in projection.rows)
                    or {row['suggested_profile'] for row in projection.rows if row['suggested_profile']}!={identity for identity,_ in projection.associated_profiles}):
                raise ValueError('invalid_association_metadata')
            if self.kind=='review' and any(type(row) is not dict or set(row)!=REVIEW_FIELDS[self.variant]
                    or self.variant=='text' and (row['variant']!='text' or not _room(row['room_id'])) for row in projection.rows):
                raise ValueError('invalid_voice_review_variant')
            summary = projection.summary
            if (summary is None) != (self.query is None):
                raise ValueError('invalid_voice_query_projection')
            if summary is not None:
                stamps = projection.revision.stamps if type(projection.revision) is RevisionSet else ()
                producers = [stamp.producer() for stamp in stamps if type(stamp) is RevisionStamp]
                if (type(summary) is not VoiceQuerySummary or type(summary.scan_complete) is not bool
                        or (summary.matching_count is None) == summary.scan_complete
                        or summary.matching_count is not None and (type(summary.matching_count) is not int
                            or not 0<=summary.matching_count<=4096)
                        or not _deadline(summary.count_expires_at,projection.observed_at)
                        or not _deadline(summary.room_options_expires_at,projection.observed_at)
                        or summary.room_options is not None and (self.query.search or type(summary.room_options) is not tuple
                            or len(summary.room_options)>32 or not all(type(room) is str and _room(room) for room in summary.room_options)
                            or list(summary.room_options)!=sorted(set(summary.room_options)))
                        or (summary.room_options_expires_at is None) != (not summary.room_options)
                        # Every query response is fenced by group, profile and revocation producers.
                        or len(producers)!=3 or producers[0] is not self.voices.ledger.groups
                        or producers[1] is not self.voices.ledger
                        or producers[2] is not self.voices.privacy.ledger.revocations):
                    raise ValueError('invalid_voice_query_projection')
            operation.empty_metadata = self.kind in {'profiles','captures'} and not projection.rows
            operation.rows,projection.rows = projection.rows,[]
            operation.sources = projection.sources
            operation.expires_at = projection.expires_at
            operation.channel = projection.channel
            operation.profile_refs = projection.profile_refs
            operation.revision = projection.revision
            operation.associated_profiles = projection.associated_profiles
            operation.activity_sources = projection.activity_sources
            if operation.rows:
                ticket = await state.browser.before_frame(operation)
                if ticket is None:
                    started = True
                    await self.owner._error(send,503)
                    return RestResult.FAILED
                state.snapshots[operation.event_id] = operation.metadata()
            if self.owner.connection(self.connection,reader=self.reader,minimum_protocol=3,channel=operation.channel) is not state or operation.guard() is not True:
                started = True
                await self.owner._error(send,403)
                return RestResult.FAILED
            started = True
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-kza-source',b'real')]})
            available = (not self.owner.retention._closed and operation.guard() is True
                and self.owner.connection(self.connection,reader=self.reader,minimum_protocol=3,channel=operation.channel) is state
                and (not operation.sources and operation.revision is None
                    or operation.metadata().available(self.owner.retention.policy,self.owner.retention.clock())))
            if ticket is not None and available:
                snapshot = {'event_id':operation.event_id,'expires_at':operation.expires_at,'observed_at':projection.observed_at,
                    'receipt':ticket.to_dict(),'channel':operation.channel,'next_cursor':projection.next_cursor,'rows':operation.rows}
                if summary is not None:
                    snapshot.update({'matching_count':summary.matching_count,'scan_complete':summary.scan_complete,
                        'count_observed_at':projection.observed_at,'count_expires_at':summary.count_expires_at,
                        'room_options':list(summary.room_options) if summary.room_options is not None else None,
                        'room_options_expires_at':summary.room_options_expires_at})
            elif ticket is not None:
                state.snapshot_retired.add(operation.event_id)
                state.wake.set()
            document = {'connection_id':self.connection,'snapshot':snapshot,'channel':operation.channel,
                'next_cursor':projection.next_cursor if available else None}
            if summary is not None:
                # An empty page has no ticket, so nothing revocable leaves without
                # one: only a complete zero count, never room options.
                document['empty_page'] = ({'matching_count':0 if summary.matching_count==0 else None,
                    'scan_complete':summary.scan_complete,'count_observed_at':projection.observed_at}
                    if ticket is None and not operation.rows and available else None)
            body = json.dumps(document,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            document = None
            snapshot = None
            operation.rows = None
            await send({'type':'http.response.body','body':body})
            return RestResult.COMPLETE
        except (Exception,asyncio.CancelledError) as error:
            status = 403 if isinstance(error,PermissionError) else 404 if isinstance(error,LookupError) else 503
            if not started:
                try:
                    await self.owner._error(send,status)
                except (Exception,asyncio.CancelledError):
                    pass
            return RestResult.FAILED
        finally:
            operation.rows = None
            if projection is not None:
                projection.rows = []
            projection = state = ticket = body = snapshot = send = row = link = profile_fields = summary = document = None
