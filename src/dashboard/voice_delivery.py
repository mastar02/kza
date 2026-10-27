"""Version-three voice REST bodies share the retained socket/ASGI send owner."""

import asyncio
import json
import math

from starlette.responses import Response

from src.dashboard.rest_delivery import RestOperation,RestResult
from src.users.voice_projection import VoiceProjection


class VoiceResponse(Response):
    """Read only within the actual retained outer HTTP scope and page connection."""

    def __init__(self, owner, voices, connection, kind, cursor, actor):
        super().__init__(content=b'')
        self.owner,self.voices,self.connection = owner,voices,connection
        self.kind,self.cursor,self.actor = kind,cursor,actor

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

    async def _deliver(self, operation, send):
        projection = state = ticket = body = snapshot = None
        started = False
        try:
            state = self.owner.connection(self.connection,minimum_protocol=3)
            operation.connection = self.connection
            credential = self.actor._current
            operation.guard = lambda:credential is None or credential() is True
            if (state is None or self.voices.privacy.ledger is not self.owner.retention.policy
                    or self.owner.retention._closed):
                started = True
                await self.owner._error(send,428)
                return RestResult.FAILED
            projection = await self.voices.acall(self.voices.ledger.project,self.kind,self.cursor,self.actor)
            if (type(projection) is not VoiceProjection or projection.channel!='voices:'+self.kind
                    or type(projection.rows) is not list or len(projection.rows)>(1 if self.kind=='review' else 50)
                    or bool(projection.profile_refs)!=(self.kind=='profiles' and bool(projection.rows))):
                raise ValueError('invalid_voice_projection')
            if self.kind=='profiles' and any(type(row) is not dict or set(row)!={
                    'id','name','revision','confirmed_at','confirmed_by','expires_at'} for row in projection.rows):
                raise ValueError('invalid_profile_metadata')
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
            if self.kind=='captures' and (any(type(row) is not dict or set(row)!={'id','group_id','created_at','expires_at','duration_s',
                    'model_key','capture_ids','review_available','suggested_profile','similarity','association',
                    'group_revision','group_state','group_reason','group_size','group_similarity','group_expires_at'} for row in projection.rows)
                    or {row['suggested_profile'] for row in projection.rows if row['suggested_profile']}!={identity for identity,_ in projection.associated_profiles}):
                raise ValueError('invalid_association_metadata')
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
            if self.owner.connection(self.connection,minimum_protocol=3) is not state or operation.guard() is not True:
                started = True
                await self.owner._error(send,403)
                return RestResult.FAILED
            started = True
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-kza-source',b'real')]})
            available = (not self.owner.retention._closed and operation.guard() is True
                and self.owner.connection(self.connection,minimum_protocol=3) is state
                and (not operation.sources or operation.metadata().available(self.owner.retention.policy,self.owner.retention.clock())))
            if ticket is not None and available:
                snapshot = {'event_id':operation.event_id,'expires_at':operation.expires_at,'observed_at':projection.observed_at,
                    'receipt':ticket.to_dict(),'channel':operation.channel,'next_cursor':projection.next_cursor,'rows':operation.rows}
            elif ticket is not None:
                state.snapshot_retired.add(operation.event_id)
                state.wake.set()
            body = json.dumps({'connection_id':self.connection,'snapshot':snapshot,'channel':operation.channel,
                'next_cursor':projection.next_cursor},ensure_ascii=False,
                allow_nan=False,separators=(',',':')).encode()
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
            projection = state = ticket = body = snapshot = send = None
