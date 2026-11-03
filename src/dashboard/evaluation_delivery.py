"""A singular evaluation sample retained through its actual outer HTTP send."""

import asyncio
import json

from starlette.responses import Response

from src.dashboard.authority import DashboardAuthority, DashboardLease
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.evaluation_grants import EvaluationGrantCatalog
from src.dashboard.live_retention_types import EvaluationScopeFence, RevisionSet
from src.dashboard.rest_delivery import RestOperation, RestResult
from src.users.voice_evaluation import EvaluationProjection, VoiceEvaluationReader


class EvaluationResponse(Response):
    def __init__(self,owner,voices,connection,actor,*,reader):
        super().__init__(content=b'')
        if type(actor) is not EvaluationReadActor:
            raise ValueError('invalid_evaluation_actor')
        self.owner,self.voices,self.connection,self.actor,self.reader = owner,voices,connection,actor,reader

    async def __call__(self,scope,receive,send):
        operation = scope.get('kza.rest_operation')
        try:
            if not isinstance(operation,RestOperation):
                await self.owner._error(send,503)
                return
            await self._deliver(operation,send)
        finally:
            self.owner = self.voices = self.connection = self.actor = self.reader = None

    async def _deliver(self,operation,send):
        owner,voices,connection,actor,reader = self.owner,self.voices,self.connection,self.actor,self.reader
        projection = state = ticket = body = snapshot = fence = None
        started = False

        def available():
            return (not owner.retention._closed and operation.guard() is True
                and owner.connection(connection,reader=reader,minimum_protocol=3,channel=operation.channel) is state
                and operation.metadata().available(owner.retention.policy,owner.retention.clock()))

        try:
            operation.channel = 'evaluation:sample'
            operation.connection,operation.reader = connection,reader
            operation.guard = lambda a=actor:a.lease.current(frozenset({'evaluation:read'}))
            state = owner.connection(connection,reader=reader,minimum_protocol=3,channel=operation.channel)
            if (state is None or voices.privacy.ledger is not owner.retention.policy or owner.retention._closed
                    or type(voices.evaluation) is not VoiceEvaluationReader
                    or voices.evaluation.ledger is not owner.retention.policy):
                started = True
                await owner._error(send,428)
                return RestResult.FAILED
            operation.private_envelope = True
            projection = await voices.acall(voices.evaluation.read,actor)
            if (type(projection) is not EvaluationProjection or projection.actor is not actor
                    or projection.channel!='evaluation:sample' or type(projection.revision) is not RevisionSet
                    or len(projection.revision.stamps)!=1):
                raise ValueError('invalid_evaluation_projection')
            fence = projection.revision.stamps[0]
            if (type(fence) is not EvaluationScopeFence or fence.actor is not actor
                    or fence.sources!=projection.sources or fence.source_digests!=projection.source_digests
                    or fence.expires_at!=projection.expires_at):
                raise ValueError('invalid_evaluation_fence')
            # Recheck the closed DTO at the delivery boundary before ownership
            # transfer; the native reader's returned mutable rows are not a grant.
            EvaluationProjection.__post_init__(projection)
            operation.rows,projection.rows = projection.rows,[]
            operation.sources,operation.expires_at = projection.sources,projection.expires_at
            operation.revision = projection.revision
            operation.profile_refs = operation.associated_profiles = operation.activity_sources = ()
            operation.empty_metadata = False
            operation.guard = lambda a=actor,d=projection.source_digests,t=projection.expires_at:a.current(d,t)
            if not available():
                raise PermissionError('evaluation_scope_unavailable')
            ticket = await state.browser.before_frame(operation)
            if ticket is None:
                raise RuntimeError('evaluation_ticket_unavailable')
            state.snapshots[operation.event_id] = operation.metadata()
            if not available():
                raise PermissionError('evaluation_scope_unavailable')
            started = True
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-kza-source',b'real')]})
            if available():
                snapshot = {'event_id':operation.event_id,'expires_at':operation.expires_at,
                    'observed_at':projection.observed_at,'receipt':ticket.to_dict(),
                    'channel':operation.channel,'rows':operation.rows}
            else:
                state.snapshot_retired.add(operation.event_id)
                state.wake.set()
            body = json.dumps({'connection_id':connection,'snapshot':snapshot,'channel':operation.channel},
                ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            snapshot = None
            operation.rows = None
            await send({'type':'http.response.body','body':body})
            return RestResult.COMPLETE
        except (Exception,asyncio.CancelledError) as error:
            if ticket is not None and state is not None:
                state.snapshot_retired.add(operation.event_id)
                state.wake.set()
            status = 403 if isinstance(error,PermissionError) else 404 if isinstance(error,LookupError) else 503
            if not started:
                try:
                    await owner._error(send,status)
                except (Exception,asyncio.CancelledError):
                    pass
            return RestResult.FAILED
        finally:
            operation.rows = None
            if type(projection) is EvaluationProjection:
                projection.rows = []
            projection = state = ticket = body = snapshot = fence = send = None


class EvaluationGrantsResponse(Response):
    """Keep the original catalog guard through actual HTTP transmission."""

    def __init__(self, owner, authority: DashboardAuthority, actor: DashboardLease) -> None:
        super().__init__(content=b'')
        self.owner,self.authority,self.actor = owner,authority,actor

    async def __call__(self, scope, receive, send) -> None:
        del receive
        operation = scope.get('kza.rest_operation')
        owner,authority,actor = self.owner,self.authority,self.actor
        catalog = body = failure_send = None
        started = False
        try:
            if type(operation) is not RestOperation:
                raise RuntimeError('evaluation_catalog_owner_required')
            operation.sources = ()
            operation.private_envelope = False
            operation.channel = 'evaluation:grants'
            operation.revision = None
            catalog = authority.evaluation_grants(actor)
            if type(catalog) is not EvaluationGrantCatalog or catalog.actor is not actor:
                raise ValueError('invalid_evaluation_catalog')
            EvaluationGrantCatalog.__post_init__(catalog)
            def guard(c=catalog,o=owner):
                return not o.retention._closed and c.current()
            operation.guard = guard
            operation.expires_at = catalog.expires_at
            scope['kza.dashboard_delivery_guard'] = guard
            if not guard():
                raise PermissionError('evaluation_catalog_retired')
            body = json.dumps(catalog.public(),ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            started = True
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
            if not guard():
                raise PermissionError('evaluation_catalog_retired')
            await send({'type':'http.response.body','body':body})
        except (Exception,asyncio.CancelledError) as error:
            status = 403 if isinstance(error,PermissionError) else 503
            if not started:
                failure_send = send
        finally:
            self.owner = self.authority = self.actor = None
            catalog = body = owner = authority = actor = scope = send = operation = None
        if failure_send is not None:
            try:
                await failure_send({'type':'http.response.start','status':status,'headers':[
                    (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
                await failure_send({'type':'http.response.body','body':b'{"detail":"evaluation_catalog_unavailable"}'})
            except (Exception,asyncio.CancelledError):
                pass
            finally:
                failure_send = None
