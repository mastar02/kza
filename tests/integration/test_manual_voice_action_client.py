"""Real loopback Dashboard/HA transports preserve separate action authority."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import socket
import time

from fastapi import Depends, FastAPI
import pytest
import uvicorn

from src.audio.manual_command_action import ManualVoiceActionClient
from src.dashboard.auth import require_dashboard_access
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.dashboard.home_api import register_home_routes
from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant.home_actions import HomeActionService
from tests.mocks.task_transport import TaskTransport


@asynccontextmanager
async def action_server(tmp_path, dashboard_auth, mode="confirmed"):
    transport = TaskTransport(mode)
    async with transport.serve() as ha_url:
        ha = HomeAssistantClient(ha_url, "synthetic-server-ha-secret", state_confirmation_timeout_s=.03)
        actions = HomeActionService(ha, policy=HomeControlPolicy((
            EntityGrant("light.living", frozenset({"turn_on", "turn_off"})),)))
        authority = DashboardAuthority(str(tmp_path / "action-authority.sqlite"))
        await authority.initialize()
        operator = authority.resolve(dashboard_auth["Authorization"])
        grant = await authority.issue_credential(operator, subject="synthetic-manual-home-actor",
            role=DashboardRole.HOME, expires_at=time.time() + 120)
        app = FastAPI(dependencies=[Depends(require_dashboard_access)])
        app.state.dashboard_authority = authority
        register_home_routes(app, catalog=None, actions=actions)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))

        async def started():
            while not server.started:
                if serving.done():
                    serving.result()
                    raise AssertionError("loopback Dashboard did not start")
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(started(), 3)
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            yield url, grant, authority, operator, ha, transport, actions
        finally:
            transport.release.set()
            transport.states_release.set()
            await actions.close()
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(serving), 5)
            finally:
                listener.close()
                await ha.close()
                await authority.aclose()


@pytest.mark.parametrize("mode,expected", [
    ("confirmed", ("confirmed", "accepted", "observed", "ha_state")),
    ("unconfirmed", ("uncertain", "accepted", "unconfirmed", None)),
    ("rejected", ("failed", "rejected", "not_observed", None)),
    ("uncertain", ("uncertain", "uncertain", "uncertain", None)),
])
async def test_actual_typed_dashboard_and_ha_receipt_is_not_a_generic_http_success(
        tmp_path, dashboard_auth, mode, expected):
    async with action_server(tmp_path, dashboard_auth, mode) as (url, grant, _, _, _, transport, _):
        client = ManualVoiceActionClient(url, grant.token)
        try:
            result = await asyncio.to_thread(client.execute, "light", "turn_on", "light.living",
                                             request_id="manual:typed")
            assert (result.status, result.dispatch, result.effect, result.observation_scope) == expected
            assert transport.sends == [("light.living", "turn_on", {"entity_id": "light.living"})]
        finally:
            await asyncio.to_thread(client.close)


async def test_same_actor_request_id_deduplicates_and_conflicting_body_does_not_resend(tmp_path, dashboard_auth):
    async with action_server(tmp_path, dashboard_auth) as (url, grant, _, _, _, transport, _):
        client = ManualVoiceActionClient(url, grant.token)
        try:
            first = await asyncio.to_thread(client.execute, "light", "turn_on", "light.living", request_id="manual:one")
            same = await asyncio.to_thread(client.execute, "light", "turn_on", "light.living", request_id="manual:one")
            conflict = await asyncio.to_thread(client.execute, "light", "turn_off", "light.living", request_id="manual:one")
            assert same == first and first.status == "confirmed"
            assert conflict.status == "failed" and conflict.reason == "manual_voice_action_request_conflict"
            assert len(transport.sends) == 1
        finally:
            await asyncio.to_thread(client.close)


async def test_retirement_capability_is_not_action_auth_and_original_home_actor_can_be_revoked(
        tmp_path, dashboard_auth, monkeypatch):
    async with action_server(tmp_path, dashboard_auth) as (url, grant, authority, operator, ha, transport, _):
        capability_only = ManualVoiceActionClient(url, "a" * 43)
        client = ManualVoiceActionClient(url, grant.token)
        entered, release = asyncio.Event(), asyncio.Event()
        original = ha.get_entity_state_fresh

        async def held(entity):
            value = await original(entity)
            entered.set()
            await release.wait()
            return value

        pending = None
        try:
            rejected = await asyncio.to_thread(capability_only.execute, "light", "turn_on", "light.living",
                                               request_id="manual:cap-only")
            assert rejected.status == "uncertain" and transport.sends == []
            monkeypatch.setattr(ha, "get_entity_state_fresh", held)
            pending = asyncio.create_task(asyncio.to_thread(client.execute, "light", "turn_on", "light.living",
                                                          request_id="manual:original-actor"))
            await asyncio.wait_for(entered.wait(), 3)
            await authority.revoke_credential(operator, grant.identity)
            release.set()
            result = await asyncio.wait_for(asyncio.shield(pending), 3)
            assert result.status == "failed" and result.dispatch == "not_sent"
            assert result.effect == "not_observed" and transport.sends == []
        finally:
            release.set()
            if pending is not None:
                await pending
            await asyncio.to_thread(client.close)
            with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
                await asyncio.to_thread(capability_only.close)


async def test_real_http_timeout_does_not_retire_the_still_running_server_action(tmp_path, dashboard_auth):
    async with action_server(tmp_path, dashboard_auth) as (url, grant, _, _, _, transport, actions):
        transport.hold = True
        client = ManualVoiceActionClient(url, grant.token, timeout_s=.05)
        pending = asyncio.create_task(asyncio.to_thread(client.execute, "light", "turn_on", "light.living",
                                                      request_id="manual:remote-held"))
        server_task = None
        try:
            await asyncio.wait_for(transport.entered.wait(), 3)
            server_task = next(iter(actions._requests.values()))[1]
            assert not server_task.done()
            result = await asyncio.wait_for(asyncio.shield(pending), 3)
            assert (result.status, result.dispatch, result.effect) == ("uncertain", "unknown", "unknown")
            assert not server_task.done() and len(transport.sends) == 1
            with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
                await asyncio.to_thread(client.close)
            assert not server_task.done()
            with pytest.raises(RuntimeError, match="^manual_voice_action_closed$"):
                await asyncio.to_thread(client.execute, "light", "turn_on", "light.living",
                                        request_id="manual:no-retry")
            assert len(transport.sends) == 1
        finally:
            transport.release.set()
            await pending
            if server_task is not None:
                terminal = await asyncio.wait_for(asyncio.shield(server_task), 3)
                assert terminal.status == "confirmed"
            with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
                await asyncio.to_thread(client.close)
