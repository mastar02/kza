"""Authenticated browser handshake and exact retirement capabilities."""

import asyncio
from contextvars import Context
from enum import StrEnum
import json
from uuid import uuid4

from src.dashboard.live_receipts import canonical_id


class BrowserPhase(StrEnum):
    HELLO = 'hello'
    RESET = 'reset'
    READY = 'ready'
    CLOSED = 'closed'


class LiveBrowserSession:
    """Keep metadata work owned after a socket observer or receive loop exits."""

    def __init__(self, receipts, wake):
        self.receipts = receipts
        self.connection = str(uuid4())
        self.phase = BrowserPhase.HELLO
        self._wake = wake
        self._reset = None
        self._reset_sent = False
        self._tickets = {}
        self._tasks = set()
        self._deadline = asyncio.get_running_loop().time()+5
        self.protocol = 1
        self._rest_ready = False

    @property
    def ready(self) -> bool:
        """Private frames require a committed clear acknowledgement."""
        return self.phase is BrowserPhase.READY

    def timed_out(self) -> bool:
        """A page cannot reserve a slot forever without clearing its view."""
        return self._deadline is not None and asyncio.get_running_loop().time() >= self._deadline

    def reset_frame(self) -> dict | None:
        """Select one pending challenge without replaying its wire payload."""
        if self.phase is BrowserPhase.RESET and self._reset is not None and not self._reset_sent:
            self._reset_sent = True
            return self._reset.to_frame()
        return None

    def ready_frame(self) -> dict | None:
        """Advertise REST admission only after the exact reset commit."""
        if self.ready and self._rest_ready:
            self._rest_ready = False
            return {'type':'rest_ready','connection_id':self.connection,'ts':''}
        return None

    @staticmethod
    def _decode(message):
        if type(message) is not dict or message.get('type') != 'websocket.receive':
            raise ValueError('invalid_live_protocol')
        text = message.get('text')
        if type(text) is not str or len(text) > 8192:
            raise ValueError('invalid_live_protocol')
        doc = json.loads(text)
        if type(doc) is not dict:
            raise ValueError('invalid_live_protocol')
        kind = doc.get('type')
        if kind == 'live_hello' and set(doc) == {'type','version','client_id'}:
            if type(doc['version']) is not int or doc['version'] not in (1,2):
                raise ValueError('unsupported_live_protocol')
            return kind,(canonical_id(doc['client_id']),doc['version'])
        if kind == 'live_reset_ack' and set(doc) == {'type','id','nonce'}:
            return kind,(canonical_id(doc['id']),canonical_id(doc['nonce']))
        if kind == 'live_ack' and set(doc) == {'type','receipts'}:
            rows = doc['receipts']
            if type(rows) is not list or not 1<=len(rows)<=16:
                raise ValueError('invalid_live_protocol')
            receipts = []
            for row in rows:
                if type(row) is not dict or set(row) != {'ticket','nonce'}:
                    raise ValueError('invalid_live_protocol')
                receipts.append((canonical_id(row['ticket']),canonical_id(row['nonce'])))
            return kind,(tuple(receipts),)
        raise ValueError('invalid_live_protocol')

    async def handle(self, message) -> None:
        """Decode bounded metadata before creating any retained worker task."""
        kind,args = self._decode(message)
        message = None
        async def execute():
            if self.phase is BrowserPhase.CLOSED:
                return
            if kind == 'live_hello':
                if self.phase is not BrowserPhase.HELLO:
                    raise ValueError('duplicate_live_hello')
                reset = await self.receipts.hello(args[0],self.connection)
                if self.phase is not BrowserPhase.CLOSED:
                    self.protocol = args[1]
                    self._reset, self._reset_sent = reset,False
                    self.phase = BrowserPhase.RESET
            elif kind == 'live_reset_ack':
                if self.phase is not BrowserPhase.RESET or self._reset is None:
                    raise ValueError('unexpected_live_reset_ack')
                if not await self.receipts.reset_ack(self.connection,*args):
                    raise ValueError('invalid_live_reset_ack')
                if self.phase is not BrowserPhase.CLOSED:
                    self._tickets.clear()
                    self._reset = None
                    self.phase = BrowserPhase.READY
                    self._deadline = None
                    self._rest_ready = self.protocol == 2
            elif kind == 'live_ack':
                if self.phase not in {BrowserPhase.READY,BrowserPhase.RESET}:
                    raise ValueError('unexpected_live_ack')
                await self.receipts.ack(self.connection,args[0])
                acknowledged = set(args[0])
                self._tickets = {event:ticket for event,ticket in self._tickets.items()
                    if (ticket.ticket,ticket.nonce) not in acknowledged}
            self._wake.set()
        task = asyncio.create_task(execute(),context=Context())
        self._tasks.add(task)
        def finished(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished,context=Context())
        await asyncio.shield(task)

    async def before_frame(self, record):
        """Persist lineage before private bytes, retaining no payload in SQLite."""
        if not self.ready:
            return None
        ticket = await self.receipts.issue(self.connection,record.event_id,record.expires_at,record.sources)
        if ticket is not None:
            self._tickets[record.event_id] = ticket
            return ticket
        if self.phase is not BrowserPhase.CLOSED:
            self._reset = await self.receipts.reset(self.connection)
            self._reset_sent = False
            self.phase = BrowserPhase.RESET
            self._deadline = asyncio.get_running_loop().time()+5
            self._wake.set()
        return None

    def retirement_frame(self, tokens) -> dict:
        """The client can retire unseen frames too, using the exact capability."""
        return {'type':'retire','payload':{'event_ids':list(tokens),
            'receipts':[{'event_id':token,**self._tickets[token].to_dict()}
                for token in tokens if token in self._tickets]},'ts':''}

    def seal(self) -> None:
        """Prevent late metadata results from reopening a closing session."""
        self.phase = BrowserPhase.CLOSED

    async def finish(self) -> None:
        """Wait for late hello/ACK work before releasing its connection lease."""
        self.seal()
        await asyncio.gather(*tuple(self._tasks),return_exceptions=True)
        await self.receipts.disconnect(self.connection)
        self._tickets.clear()
        self._reset = None
