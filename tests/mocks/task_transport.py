"""Loopback HA transport for task-receipt tests; never addresses a home server."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from aiohttp import web


@dataclass
class TaskTransport:
    """Observe exact send attempts and control server acceptance/state evidence."""

    mode: str = "confirmed"
    modes: dict = field(default_factory=dict)
    hold: bool = False
    sends: list = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    states: dict = field(default_factory=lambda: {
        "light.living": {"entity_id": "light.living", "state": "off", "attributes": {"friendly_name": "Luz living"}},
        "light.second": {"entity_id": "light.second", "state": "off", "attributes": {}},
        "light.third": {"entity_id": "light.third", "state": "off", "attributes": {}},
        "light.group": {"entity_id": "light.group", "state": "off",
                        "attributes": {"entity_id": ["light.living", "light.second", "light.third"]}},
    })

    @asynccontextmanager
    async def serve(self):
        """Serve the real HA client's GET/POST contract on a disposable port."""
        async def api(request):
            return web.json_response({"message": "API running"})

        async def states(request):
            return web.json_response(list(self.states.values()))

        async def state(request):
            if self.mode == "observation_error" and self.sends:
                return web.Response(status=503)
            return web.json_response(self.states[request.match_info["entity"]])

        async def send(request):
            body = await request.json()
            entity = body["entity_id"]
            self.sends.append((entity, request.match_info["service"], body))
            self.entered.set()
            if self.hold:
                await self.release.wait()
            mode = self.modes.get(entity, self.mode)
            if mode in {"rejected", "uncertain"}:
                return web.Response(status=403 if mode == "rejected" else 503)
            if mode == "confirmed":
                self.states[entity]["state"] = "off" if request.match_info["service"] == "turn_off" else "on"
            return web.json_response([])

        app = web.Application()
        app.add_routes([web.get("/api/", api), web.get("/api/states", states),
                        web.get("/api/states/{entity}", state),
                        web.post("/api/services/{domain}/{service}", send)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        finally:
            self.release.set()
            await runner.cleanup()
