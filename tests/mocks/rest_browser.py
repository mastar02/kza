"""A real version-two receipt session with a synthetic protocol peer."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from uuid import uuid4


@dataclass
class RestBrowser:
    """Only channel metadata and a controlled receiver are retained here."""

    connection: str
    incoming: asyncio.Queue
    outgoing: asyncio.Queue

    async def message(self, document):
        """Deliver an actual decoded ASGI receive message to the protocol owner."""
        await self.incoming.put({'type':'websocket.receive','text':json.dumps(document)})


@asynccontextmanager
async def ready_browser(bus):
    """Complete hello/reset and wait for the persisted REST-ready control."""
    subscription,_ = await bus.subscribe()
    incoming,outgoing = asyncio.Queue(),asyncio.Queue()
    async def send(frame):
        await outgoing.put(frame)
    observing = asyncio.create_task(bus.browser_stream(subscription,send,receiver=incoming.get))
    peer = RestBrowser('',incoming,outgoing)
    try:
        await peer.message({'type':'live_hello','version':2,'client_id':str(uuid4())})
        async with asyncio.timeout(3):
            reset = await outgoing.get()
        assert reset['type']=='live_reset'
        await peer.message({'type':'live_reset_ack',**reset['payload']})
        async with asyncio.timeout(3):
            ready = await outgoing.get()
        assert ready['type']=='rest_ready'
        peer.connection = ready['connection_id']
        yield peer
    finally:
        await incoming.put({'type':'websocket.disconnect','code':1000})
        async with asyncio.timeout(3):
            await observing
