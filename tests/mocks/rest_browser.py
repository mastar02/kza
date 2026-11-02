"""A real version-two receipt session with a synthetic protocol peer."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from uuid import uuid4
from src.dashboard.live_retention_types import LiveSubscriptionMode


@dataclass
class RestBrowser:
    """Only channel metadata and a controlled receiver are retained here."""

    connection: str
    incoming: asyncio.Queue
    outgoing: asyncio.Queue
    reader: object
    ready_frame: dict | None = None

    async def message(self, document):
        """Deliver an actual decoded ASGI receive message to the protocol owner."""
        await self.incoming.put({'type':'websocket.receive','text':json.dumps(document)})


@asynccontextmanager
async def ready_browser(bus, *, version=2, reader=None, client_id=None, mode=LiveSubscriptionMode.TECHNICAL):
    """Complete hello/reset and wait for the persisted REST-ready control."""
    reader = bus.test_reader if reader is None else reader
    subscription,_ = await bus.subscribe(reader=reader,mode=mode)
    incoming,outgoing = asyncio.Queue(),asyncio.Queue()
    async def send(frame):
        await outgoing.put(frame)
    observing = asyncio.create_task(bus.browser_stream(subscription,send,receiver=incoming.get,reader=reader))
    peer = RestBrowser('',incoming,outgoing,reader)
    try:
        await peer.message({'type':'live_hello','version':version,'client_id':client_id or str(uuid4())})
        async with asyncio.timeout(3):
            reset = await outgoing.get()
        assert reset['type']=='live_reset'
        await peer.message({'type':'live_reset_ack',**reset['payload']})
        async with asyncio.timeout(3):
            ready = await outgoing.get()
        assert ready['type']=='rest_ready'
        peer.connection = ready['connection_id']
        peer.ready_frame = ready
        yield peer
    finally:
        await incoming.put({'type':'websocket.disconnect','code':1000})
        async with asyncio.timeout(3):
            await observing
