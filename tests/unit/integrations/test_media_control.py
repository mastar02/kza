"""Exercise room policy, HA observations and single POST receipts over local HTTP."""

import asyncio
from dataclasses import dataclass, field
from time import time

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from src.home_assistant.control_policy import ActorKind, HomeActionDenied, HomeActor
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus
from src.integrations.media_control import (
    MediaController,
    MediaSelection,
    MediaState,
    RemoteGrant,
)


PLAYER = "media_player.tv"
REMOTE = "remote.living_room"
ROOM = "living_room"
HOUSE = HomeActor.household()


@dataclass
class Home:
    ha: HomeAssistantClient
    states: dict
    gets: list = field(default_factory=list)
    posts: list = field(default_factory=list)
    service_delay: float = 0
    status: int = 200
    revoke_on: str | None = None
    active: bool = True
    clock_offset: float = 0

    def controller(self, remote=None):
        return MediaController(
            self.ha,
            [MediaSelection(ROOM, PLAYER, remote)],
            max_staleness_s=5,
            clock=lambda: time() + self.clock_offset,
        )

    def actor(self, *permissions):
        return HomeActor(
            ActorKind.DASHBOARD, "session", frozenset(permissions), lambda: self.active
        )


@pytest.fixture
async def home():
    app = web.Application()

    async def observe(request):
        entity = request.match_info["entity"]
        rig.gets.append(entity)
        if rig.revoke_on == entity:
            rig.active = False
        if entity not in rig.states:
            raise web.HTTPNotFound()
        return web.json_response({"entity_id": entity, **rig.states[entity]})

    async def service(request):
        rig.posts.append(
            (
                request.match_info["domain"],
                request.match_info["service"],
                await request.json(),
            )
        )
        if rig.revoke_on == "post":
            rig.active = False
        await asyncio.sleep(rig.service_delay)
        return web.json_response([], status=rig.status)

    app.router.add_get("/api/states/{entity}", observe)
    app.router.add_post("/api/services/{domain}/{service}", service)
    async with TestServer(app) as server:
        client = HomeAssistantClient(str(server.make_url("/")), "test")
        rig = Home(
            client,
            {
                PLAYER: {"state": "idle", "attributes": {}},
                REMOTE: {"state": "on", "attributes": {}},
            },
        )
        try:
            yield rig
        finally:
            await client.close()


@pytest.mark.parametrize("state", ["playing", "paused", "idle", "off", "standby"])
async def test_read_current(home, state):
    home.states[PLAYER] = {
        "state": state,
        "attributes": {
            "supported_features": 4,
            "volume_level": 0,
            "is_volume_muted": False,
            "media_title": "Local TV",
            "app_name": "Tuner",
        },
    }
    result = await home.controller().read(home.actor("home:read"), ROOM, PLAYER)
    assert result == MediaState(state, 4, 0, False, "Local TV", "Tuner")


async def test_absent_attributes(home):
    assert await home.controller().read(HOUSE, ROOM, PLAYER) == MediaState("idle")


@pytest.mark.parametrize("state", ["unavailable", "unknown", "stale", "missing"])
async def test_unusable_player_never_falls_back(home, state):
    if state == "missing":
        del home.states[PLAYER]
    else:
        home.states[PLAYER] = {
            "state": "playing" if state == "stale" else state,
            "attributes": {"supported_features": 128, "media_title": "Historical"},
        }
    home.clock_offset = 6 if state == "stale" else 0
    controller = home.controller(RemoteGrant(REMOTE, turn_on=True))
    expected = "unavailable" if state == "missing" else state
    assert await controller.read(HOUSE, ROOM, PLAYER) == MediaState(expected)
    with pytest.raises(HomeActionDenied, match=f"media_{expected}"):
        await controller.control(HOUSE, ROOM, PLAYER, "turn_on")
    assert home.gets == [PLAYER, PLAYER]
    assert not home.posts


@pytest.mark.parametrize(
    ("op", "bit", "service", "value", "payload"),
    [
        ("play", 16384, "media_play", None, {}),
        ("pause", 1, "media_pause", None, {}),
        ("stop", 4096, "media_stop", None, {}),
        ("turn_on", 128, "turn_on", None, {}),
        ("turn_off", 256, "turn_off", None, {}),
        ("volume_up", 1024, "volume_up", None, {}),
        ("volume_down", 1024, "volume_down", None, {}),
        ("volume_set", 4, "volume_set", 1.0, {"volume_level": 1.0}),
        ("volume_mute", 8, "volume_mute", False, {"is_volume_muted": False}),
    ],
)
async def test_only_advertised_service(home, op, bit, service, value, payload):
    controller = home.controller()
    # Advertise every listed bit except the one requested (including PLAY_MEDIA).
    home.states[PLAYER]["attributes"]["supported_features"] = 65535 ^ bit
    with pytest.raises(HomeActionDenied, match=f"media_unsupported_{op}"):
        await controller.control(HOUSE, ROOM, PLAYER, op, value=value)
    assert not home.posts
    home.states[PLAYER]["attributes"]["supported_features"] = bit
    result = await controller.control(
        home.actor("home:control"), ROOM, PLAYER, op, value=value
    )
    assert result.status is ServiceCallStatus.ACCEPTED
    assert home.posts == [("media_player", service, {"entity_id": PLAYER, **payload})]


@pytest.mark.parametrize("bits", [None, True, -1, "1", 1.5])
async def test_invalid_mask_denies(home, bits):
    home.states[PLAYER]["attributes"]["supported_features"] = bits
    with pytest.raises(HomeActionDenied, match="unsupported_pause"):
        await home.controller().control(HOUSE, ROOM, PLAYER, "pause")
    assert not home.posts


@pytest.mark.parametrize(
    "value", [-0.01, 1.01, float("nan"), float("inf"), "0.5", True, None]
)
async def test_invalid_volume(home, value):
    with pytest.raises(ValueError, match="invalid_volume"):
        await home.controller().control(HOUSE, ROOM, PLAYER, "volume_set", value=value)
    assert not home.gets and not home.posts


@pytest.mark.parametrize(
    ("room", "entity"),
    [("other", PLAYER), (ROOM, "media_player.other"), (ROOM, REMOTE)],
)
async def test_unselected(home, room, entity):
    controller = home.controller(RemoteGrant(REMOTE, turn_on=True))
    with pytest.raises(HomeActionDenied, match="not_selected"):
        await controller.read(HOUSE, room, entity)
    with pytest.raises(HomeActionDenied, match="not_selected"):
        await controller.control(HOUSE, room, entity, "turn_on")
    assert not home.gets and not home.posts


@pytest.mark.parametrize(
    ("code", "status"),
    [(503, ServiceCallStatus.UNCERTAIN), (403, ServiceCallStatus.REJECTED)],
)
async def test_service_failure_without_retry_or_remote_fallback(home, code, status):
    home.status = code
    home.states[PLAYER]["attributes"]["supported_features"] = 128
    outcome = await home.controller(RemoteGrant(REMOTE, turn_on=True)).control(
        HOUSE, ROOM, PLAYER, "turn_on"
    )
    assert (outcome.status, outcome.reason) == (status, f"ha_http_{code}")
    assert home.posts == [("media_player", "turn_on", {"entity_id": PLAYER})]


async def test_remote_only_for_individually_configured_operations(home):
    controller = home.controller(
        RemoteGrant(REMOTE, turn_off=True, volume_up_command="VOL+")
    )
    for op in ("turn_on", "volume_down", "pause"):
        with pytest.raises(HomeActionDenied, match="unsupported"):
            await controller.control(HOUSE, ROOM, PLAYER, op)
    assert REMOTE not in home.gets and not home.posts
    for op in ("turn_off", "volume_up"):
        assert (
            await controller.control(HOUSE, ROOM, PLAYER, op)
        ).status is ServiceCallStatus.ACCEPTED
    assert home.posts == [
        ("remote", "turn_off", {"entity_id": REMOTE}),
        ("remote", "send_command", {"entity_id": REMOTE, "command": "VOL+"}),
    ]
    home.states[REMOTE]["state"] = "unavailable"
    with pytest.raises(HomeActionDenied, match="remote_unavailable"):
        await controller.control(HOUSE, ROOM, PLAYER, "volume_up")
    assert len(home.posts) == 2


async def test_read_and_control_permissions(home):
    controller = home.controller()
    with pytest.raises(HomeActionDenied, match="permission_denied"):
        await controller.read(home.actor("home:control"), ROOM, PLAYER)
    with pytest.raises(HomeActionDenied, match="permission_denied"):
        await controller.control(home.actor("home:read"), ROOM, PLAYER, "play")
    home.active = False
    with pytest.raises(HomeActionDenied):
        await controller.read(home.actor("home:read"), ROOM, PLAYER)
    assert not home.gets and not home.posts


@pytest.mark.parametrize("phase", ["read", PLAYER, REMOTE])
async def test_revocation_at_each_io_boundary(home, phase):
    home.revoke_on = PLAYER if phase == "read" else phase
    controller = home.controller(RemoteGrant(REMOTE, turn_on=True))
    with pytest.raises(HomeActionDenied, match="home_control_denied"):
        if phase == "read":
            await controller.read(home.actor("home:read"), ROOM, PLAYER)
        else:
            await controller.control(
                home.actor("home:control"), ROOM, PLAYER, "turn_on"
            )
    assert home.gets == ([PLAYER, REMOTE] if phase == REMOTE else [PLAYER])
    assert not home.posts


async def test_revocation_after_send_preserves_uncertainty(home):
    home.revoke_on = "post"
    controller = home.controller(RemoteGrant(REMOTE, turn_on=True))
    outcome = await controller.control(home.actor("home:control"), ROOM, PLAYER, "turn_on")
    assert outcome.status is ServiceCallStatus.UNCERTAIN
    assert outcome.reason == "authority_lost_after_dispatch"
    assert home.gets == [PLAYER, REMOTE]
    assert len(home.posts) == 1


async def test_service_timeout_is_uncertain_without_retry(home):
    home.service_delay = 0.2
    home.ha.timeout = 0.05
    home.states[PLAYER]["attributes"]["supported_features"] = 1
    outcome = await home.controller().control(HOUSE, ROOM, PLAYER, "pause")
    assert outcome.status is ServiceCallStatus.UNCERTAIN
    assert outcome.reason == "ha_transport_uncertain"
    assert home.posts == [("media_player", "media_pause", {"entity_id": PLAYER})]
