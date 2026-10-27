"""Own sourced REST bodies, actual sends and their durable page receipts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import Context
from dataclasses import dataclass, field
from enum import StrEnum
import json
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI
from starlette.responses import Response
from starlette.types import ASGIApp,Receive,Scope,Send

from src.analytics.event_logger import EventLogger
from src.dashboard.conversation_projection import ConversationProjection
from src.dashboard.live_retention_types import SnapshotMetadata
from src.memory.fact_ledger import MemorySource

if TYPE_CHECKING:
    from src.dashboard.live_retention import LiveRetention,LiveSubscriber


class RestResult(StrEnum):
    COMPLETE = 'rest_delivery_complete'
    FAILED = 'rest_delivery_failed'


@dataclass(eq=False, repr=False)
class RestOperation:
    """Unknown native reads block every conversation until lineage is known."""

    sources: tuple[MemorySource,...] | None = None
    rows: list | None = field(default=None, repr=False)
    event_id: str = field(default_factory=lambda:str(uuid4()))
    expires_at: float = 0
    connection: str | None = None
    guard: object | None = field(default=None,repr=False)


class DashboardApplication(FastAPI):
    """Own private REST scopes outside framework middleware body buffers."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Keep actual outer ASGI sends visible after observer cancellation."""
        owner = getattr(self.state,'rest_delivery_owner',None)
        if owner is not None and scope['type']=='http' and scope.get('path')=='/api/conversations':
            await owner.scope(super().__call__,scope,receive,send)
        else:
            await super().__call__(scope,receive,send)


class ConversationResponse(Response):
    """Start private I/O only inside the retained ASGI operation."""

    def __init__(self, owner: RestDeliveryOwner, connection: str, events: EventLogger,
            filters: dict[str,str | None], guard: Callable[[],bool]) -> None:
        super().__init__(content=b'')
        self.owner, self.connection, self.events = owner,connection,events
        self.filters, self.guard = filters,guard

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Observer cancellation never cancels the actual response task."""
        try:
            operation = scope.get('kza.rest_operation')
            if isinstance(operation,RestOperation):
                await self.owner._deliver(operation,self.connection,self.events,self.filters,self.guard,send)
            else:
                await self.owner.run(self.connection,self.events,self.filters,self.guard,send)
        finally:
            self.owner = self.connection = self.events = self.filters = self.guard = None


class RestDeliveryOwner:
    """Bound all native reads and send callbacks before admitting raw content."""

    def __init__(self, retention: LiveRetention) -> None:
        self.retention = retention
        self._tasks = {}
        self._scopes = {}

    async def scope(self, app: ASGIApp, scope: Scope, receive: Receive, send: Send) -> None:
        """Own framework buffering and actual transport callbacks as one scope."""
        if self.retention._closed or len(self._scopes)>=8:
            await self._error(send,503)
            return
        operation = RestOperation()
        task = asyncio.create_task(self._scope(operation,app,scope,receive,send),context=Context())
        self._scopes[task] = operation
        def finished(done):
            self._scopes.pop(done,None)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished,context=Context())
        await asyncio.shield(task)

    async def _scope(self, operation, app, scope, receive, send):
        transmissions = set()
        scope['kza.rest_operation'] = operation
        async def transmit(message):
            try:
                await send(message)
                return RestResult.COMPLETE
            except (Exception,asyncio.CancelledError):
                return RestResult.FAILED
            finally:
                message = None
        async def owned_send(message):
            if len(transmissions)>=2:
                raise RuntimeError('rest_transport_capacity')
            if (message.get('type')=='http.response.body' and message.get('body') and operation.sources
                    and (self.connection(operation.connection) is None
                        or operation.guard is None or operation.guard() is not True
                        or not self.retention.policy.sources_available(operation.sources))):
                message = {'type':'http.response.body','body':b'{"snapshot":null}'}
            task = asyncio.create_task(transmit(message),context=Context())
            message = None
            transmissions.add(task)
            def finished(done):
                transmissions.discard(done)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished,context=Context())
            result = await asyncio.shield(task)
            if result is RestResult.FAILED:
                raise RuntimeError('rest_transport_failed')
        try:
            await app(scope,receive,owned_send)
            return RestResult.COMPLETE
        except (Exception,asyncio.CancelledError):
            return RestResult.FAILED
        finally:
            await asyncio.gather(*tuple(transmissions),return_exceptions=True)
            scope.pop('kza.rest_operation',None)
            operation.rows = None
            operation.guard = None
            app = scope = receive = send = None

    def connection(self, identity: str | None) -> LiveSubscriber | None:
        for state in self.retention._states.values():
            browser = state.browser
            if (not state.closed and browser is not None and browser.ready and browser.protocol == 2
                    and browser.connection == identity and (state.guard is None or state.guard() is True)):
                return state
        return None

    def pending(self, conversation: str) -> int:
        """A committed ACK cannot hide an actual raw callback or unknown read."""
        return sum(operation.sources is None or any(source.conversation_id == conversation
            for source in operation.sources) for operation in (*self._tasks.values(),*self._scopes.values()))

    async def run(self, connection: str, events: EventLogger, filters: dict[str,str | None],
            guard: Callable[[],bool], send: Send) -> None:
        """Keep a bounded context-free task through native work and ASGI sends."""
        if self.retention._closed or len(self._tasks) >= 8:
            await self._error(send,503)
            return
        operation = RestOperation()
        task = asyncio.create_task(self._deliver(operation,connection,events,filters,guard,send),context=Context())
        self._tasks[task] = operation
        def finished(done):
            self._tasks.pop(done,None)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished,context=Context())
        await asyncio.shield(task)

    @staticmethod
    async def _error(send, status):
        await send({'type':'http.response.start','status':status,'headers':[
            (b'content-type',b'application/json'),(b'cache-control',b'no-store')]})
        await send({'type':'http.response.body','body':b'{"detail":"conversation_delivery_unavailable"}'})

    async def _deliver(self, operation, connection, events, filters, guard, send):
        raw = body = document = ticket = state = None
        started = False
        try:
            state = self.connection(connection)
            operation.connection = connection
            operation.guard = guard
            if (state is None or events.source_policy is not self.retention.policy
                    or guard() is not True or self.retention._closed):
                started = True
                await self._error(send,428)
                return RestResult.FAILED
            raw = await events.acall(events.get_events,limit=150)
            operation.rows = []
            sources = set()
            observed = self.retention.clock()
            for event in raw:
                context = event.context or {}
                if (filters.get('user') and (event.user_name or '').lower() != filters['user'].lower()
                        or filters.get('zone') and (context.get('zone') or context.get('room_id')) != filters['zone']
                        or filters.get('path') and context.get('path','fast') != filters['path']):
                    continue
                row = ConversationProjection(event,observed).to_dict()
                if row is None or not self.retention.policy.sources_available(event.sources):
                    continue
                operation.rows.append(row)
                sources.update(event.sources)
                if len(operation.rows) == 50:
                    break
            raw = event = row = context = None
            if len(sources)>256:
                raise ValueError('rest_source_capacity')
            operation.sources = tuple(sources)
            sources = None
            if operation.rows:
                operation.expires_at = min(row['expires_at'] for row in operation.rows)
                ticket = await state.browser.before_frame(operation)
                if ticket is None:
                    started = True
                    await self._error(send,503)
                    return RestResult.FAILED
                state.snapshots[operation.event_id] = SnapshotMetadata(operation.event_id,operation.sources)
            if self.retention._closed or self.connection(connection) is not state or guard() is not True:
                started = True
                await self._error(send,428)
                return RestResult.FAILED
            started = True
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-kza-source',b'real')]})
            # Headers may themselves suspend. Recheck immediately before serializing
            # and handing private bytes to the actual body callback.
            available = (self.connection(connection) is state and guard() is True and not self.retention._closed
                and (not operation.sources or self.retention.policy.sources_available(operation.sources)))
            snapshot = None
            if ticket is not None and available:
                snapshot = {'event_id':operation.event_id,'expires_at':operation.expires_at,'observed_at':observed,
                    'receipt':ticket.to_dict(),'rows':operation.rows}
            elif ticket is not None:
                state.snapshot_retired.add(operation.event_id)
                state.wake.set()
            document = {'connection_id':connection,'snapshot':snapshot}
            body = json.dumps(document,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            document = snapshot = None
            operation.rows = None
            await send({'type':'http.response.body','body':body})
            return RestResult.COMPLETE
        except (Exception,asyncio.CancelledError):
            if not started:
                try:
                    await self._error(send,503)
                except (Exception,asyncio.CancelledError):
                    pass
            return RestResult.FAILED
        finally:
            operation.rows = None
            raw = body = document = snapshot = ticket = state = events = filters = guard = send = None

    async def drain(self) -> None:
        """Shared authority stays leased until all actual REST work has ended."""
        await asyncio.gather(*tuple(self._tasks),*tuple(self._scopes),return_exceptions=True)
