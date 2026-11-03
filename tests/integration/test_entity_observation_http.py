"""Real HA reads keep source deadlines and original diagnostic authority."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import time
from types import SimpleNamespace

from aiohttp import web
from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.api import DashboardAPI
import src.dashboard.api as api_module
import src.dashboard.rest_delivery as delivery_module
from src.dashboard.authority import (
    DashboardAuthority,
    DashboardRole,
    EvaluationModality,
    EvaluationSample,
    source_digest,
)
from src.home_assistant.ha_client import HomeAssistantClient
from tests.integration.test_capability_application import leaves as capability_leaves
from tests.integration.test_home_session_authority import web_graph

pytestmark = pytest.mark.asyncio


@dataclass
class EntitySource:
    status: int = 200
    payload: object = field(default_factory=lambda: {
        "entity_id": "light.synthetic", "state": "on",
        "attributes": {"friendly_name": "Synthetic diagnostic light"},
    })
    reads: int = 0

    async def read(self, request):
        self.reads += 1
        if self.status != 200:
            return web.Response(status=self.status)
        return web.json_response(self.payload)


@pytest.fixture
async def entity_graph(tmp_path, dashboard_auth):
    source = EntitySource()
    server = web.Application()
    server.router.add_get("/api/states/{entity}", source.read)
    runner = web.AppRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    clock = {"now": time.time()}
    try:
        async with AsyncExitStack() as stack:
            authority = DashboardAuthority(tmp_path / "authority.db",
                clock=lambda: clock["now"], session_ttl_s=60)
            await authority.initialize()
            stack.push_async_callback(authority.aclose)
            port = site._server.sockets[0].getsockname()[1]
            ha = HomeAssistantClient(f"http://127.0.0.1:{port}", "synthetic-ha-token")
            stack.push_async_callback(ha.close)
            api = DashboardAPI(ha_client=ha, authority=authority)
            stack.push_async_callback(api.stop)
            client = await stack.enter_async_context(AsyncClient(
                transport=ASGITransport(app=api.app), base_url="http://testserver"))
            token = dashboard_auth["Authorization"].split(" ", 1)[1]
            assert (await client.post("/api/admin/auth/login",
                                      json={"token": token})).status_code == 200
            yield SimpleNamespace(source=source, ha=ha, api=api, client=client,
                                  authority=authority, clock=clock, root_token=token)
    finally:
        await runner.cleanup()


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


async def test_actual_application_factory_awaits_diagnostic_entity_read(
    tmp_path, leaves, dashboard_auth,
):
    async with web_graph(tmp_path, leaves, dashboard_auth) as (graph, client, transport):
        response = await client.get("/api/entities/light.living")
        assert response.status_code == 200
        assert response.json() == transport.states["light.living"]
        assert response.headers["x-kza-source-id"] == "ha-rest-entity"
        assert response.headers["x-kza-observation-state"] == "fresh"
        assert response.headers["cache-control"] == "no-store"
        assert graph.ha_client.get_entity_observation("light.living").state == response.json()
        assert transport.sends == []


async def test_entity_response_preserves_canonical_shape_and_source_deadline(entity_graph):
    graph = entity_graph
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 200
    assert response.json() == graph.source.payload
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-kza-source-id"] == "ha-rest-entity"
    original = graph.ha.get_entity_observation("light.synthetic")
    assert response.headers["x-kza-observed-at"] == datetime.fromtimestamp(original.observed_at, UTC).isoformat()
    assert response.headers["x-kza-expires-at"] == datetime.fromtimestamp(original.expires_at, UTC).isoformat()
    assert graph.source.reads == 1


@pytest.mark.parametrize("status, reason", [(403, "source_denied"), (503, "source_unavailable")])
async def test_upstream_failure_preserves_dates_and_does_not_retire_the_login(entity_graph, status, reason):
    graph = entity_graph
    initial = await graph.client.get("/api/entities/light.synthetic")
    assert initial.status_code == 200
    graph.source.status = status
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 503
    assert response.json() == {"detail": "Estado de entidad no disponible"}
    assert response.headers["x-kza-observation-reason"] == reason
    assert response.headers["x-kza-observed-at"] == initial.headers["x-kza-observed-at"]
    assert response.headers["x-kza-expires-at"] == initial.headers["x-kza-expires-at"]
    assert response.headers["x-kza-source"] == "degraded"
    assert graph.source.reads == 2
    retained = graph.ha.get_entity_observation("light.synthetic")
    assert retained.current(time.time()) is False
    assert retained.reason == reason
    assert (await graph.client.get("/api/admin/auth/whoami")).status_code == 200


@pytest.mark.parametrize("status, reason", [(403, "source_denied"), (503, "source_unavailable")])
async def test_first_failed_source_read_has_origin_but_no_fabricated_observation(
    entity_graph, status, reason,
):
    graph = entity_graph
    graph.source.status = status
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 503
    assert response.headers["x-kza-source-id"] == "ha-rest-entity"
    assert response.headers["x-kza-observation-reason"] == reason
    assert "x-kza-observed-at" not in response.headers
    assert "x-kza-expires-at" not in response.headers
    assert graph.source.reads == 1


async def test_actual_missing_entity_is_current_negative_evidence(entity_graph):
    graph = entity_graph
    graph.source.status = 404
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 404
    assert response.json() == {"detail": "Entidad no encontrada"}
    assert response.headers["x-kza-source-id"] == "ha-rest-entity"
    assert response.headers["x-kza-observation-state"] == "fresh"
    assert "x-kza-observed-at" in response.headers
    assert graph.ha.get_entity_observation("light.synthetic").exists is False


async def test_malformed_source_is_not_a_current_entity_or_a_missing_one(entity_graph):
    graph = entity_graph
    graph.source.payload = {"entity_id": "light.synthetic", "state": 42, "attributes": {}}
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 503
    assert response.headers["x-kza-observation-reason"] == "source_invalid"
    assert "x-kza-observed-at" not in response.headers


@pytest.mark.parametrize("role", [None, DashboardRole.HOME, DashboardRole.EVALUATOR])
async def test_other_authorities_perform_zero_diagnostic_source_reads(entity_graph, dashboard_auth, role):
    graph = entity_graph
    if role is None:
        graph.client.cookies.clear()
    else:
        now = graph.clock["now"]
        samples = ()
        if role is DashboardRole.EVALUATOR:
            samples = (EvaluationSample("vf_" + "a" * 64,
                frozenset({source_digest({"capture_id": "synthetic-evaluation", "expires_at": now + 20})}),
                frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 20),)
        grant = await graph.authority.issue_credential(
            graph.authority.resolve(dashboard_auth["Authorization"]),
            subject="synthetic-limited", role=role, expires_at=now + 120, samples=samples)
        assert (await graph.client.post("/api/admin/auth/login",
                                       json={"token": grant.token})).status_code == 200
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == (401 if role is None else 403)
    assert graph.source.reads == 0


@pytest.mark.parametrize("stage", ["read", "headers"])
@pytest.mark.parametrize("withdrawal", ["logout", "replace", "expiry", "role"])
async def test_original_diagnostic_session_controls_actual_read_and_delivery(
    entity_graph, monkeypatch, dashboard_auth, stage, withdrawal,
):
    graph = entity_graph
    actor = graph.authority.resolve(dashboard_auth["Authorization"])
    grant = None
    if withdrawal == "role":
        grant = await graph.authority.issue_credential(actor, subject="synthetic-operator",
            role=DashboardRole.OPERATOR, expires_at=graph.clock["now"] + 120)
        assert (await graph.client.post("/api/admin/auth/login",
                                       json={"token": grant.token})).status_code == 200
    entered, release = asyncio.Event(), asyncio.Event()
    original = graph.ha.read_entity_observation
    messages = []

    async def held_read(*args, **kwargs):
        value = await original(*args, **kwargs)
        if stage == "read":
            entered.set()
            await release.wait()
        return value

    monkeypatch.setattr(graph.ha, "read_entity_observation", held_read)

    async def gated(scope, receive, send):
        async def actual(message):
            if stage == "headers" and message["type"] == "http.response.start":
                entered.set()
                await release.wait()
            messages.append(message.copy())
            await send(message)
        await graph.api.app(scope, receive, actual)

    async with AsyncClient(transport=ASGITransport(app=gated), base_url="http://testserver",
                           cookies=graph.client.cookies) as reader:
        pending = asyncio.create_task(reader.get("/api/entities/light.synthetic"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if withdrawal == "logout":
                assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
            elif withdrawal == "replace":
                assert (await graph.client.post("/api/admin/auth/login",
                    json={"token": graph.root_token})).status_code == 200
            elif withdrawal == "expiry":
                graph.clock["now"] += 60
            else:
                assert await graph.authority.revise_credential(actor, grant.identity,
                    role=DashboardRole.HOME, expires_at=graph.clock["now"] + 90)
            release.set()
            if stage == "headers":
                with pytest.raises(AssertionError):
                    await pending
                assert [message["type"] for message in messages] == ["http.response.start"]
                assert messages[0]["status"] == 200
            else:
                response = await pending
                assert response.status_code == 403
                assert response.json() == {"detail": "dashboard_session_retired"}
            assert b"Synthetic diagnostic light" not in b"".join(
                message.get("body", b"") for message in messages)
            assert graph.source.reads == 1
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("stage", ["before_headers", "headers"])
async def test_source_expiry_retires_entity_body_without_retiring_the_session(
    entity_graph, monkeypatch, stage, buffered,
):
    graph = entity_graph
    api = DashboardAPI(ha_client=graph.ha, authority=graph.authority)
    if buffered:
        @api.app.middleware("http")
        async def buffering(request, call_next):
            return await call_next(request)

    entered, release = asyncio.Event(), asyncio.Event()
    observed = {}
    delivery_now = {"value": None}

    class DeliveryClock:
        @staticmethod
        def now(tz):
            return delivery_now["value"] or datetime.now(tz)

    monkeypatch.setattr(delivery_module, "datetime", DeliveryClock)
    original = graph.ha.read_entity_observation

    async def held_read(*args, **kwargs):
        value = await original(*args, **kwargs)
        observed["value"] = value
        if stage == "before_headers":
            entered.set()
            await release.wait()
        return value

    monkeypatch.setattr(graph.ha, "read_entity_observation", held_read)
    messages = []

    async def gated(scope, receive, send):
        async def actual(message):
            if stage == "headers" and message["type"] == "http.response.start":
                entered.set()
                await release.wait()
            messages.append(message.copy())
            await send(message)
        await api.app(scope, receive, actual)

    try:
        async with AsyncClient(transport=ASGITransport(app=gated), base_url="http://testserver",
                               cookies=graph.client.cookies) as reader:
            pending = asyncio.create_task(reader.get("/api/entities/light.synthetic"))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                value = observed["value"]
                # Only the actual delivery clock advances. The body was built
                # from a valid real read and now waits at a transport boundary.
                delivery_now["value"] = datetime.fromtimestamp(value.expires_at, UTC)
                release.set()
                if stage == "headers":
                    with pytest.raises(AssertionError):
                        await pending
                    assert [message["type"] for message in messages] == ["http.response.start"]
                    assert messages[0]["status"] == 200
                else:
                    response = await pending
                    assert response.status_code == 503
                    assert response.headers["x-kza-observation-state"] == "stale"
                    assert response.headers["x-kza-observation-reason"] == "observation_expired"
                    assert response.headers["x-kza-observed-at"] == datetime.fromtimestamp(value.observed_at, UTC).isoformat()
                    assert response.headers["x-kza-expires-at"] == datetime.fromtimestamp(value.expires_at, UTC).isoformat()
                assert b"Synthetic diagnostic light" not in b"".join(
                    message.get("body", b"") for message in messages)
                assert (await graph.client.get("/api/admin/auth/whoami")).status_code == 200
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
    finally:
        await api.stop()


@pytest.mark.parametrize("already_expired", [False, True])
async def test_expiry_before_metadata_projection_preserves_its_actual_classification(
    entity_graph, monkeypatch, already_expired,
):
    graph = entity_graph
    clock = {"now": None}
    observed = {}
    original = graph.ha.read_entity_observation

    class AdapterClock:
        @staticmethod
        def now(tz):
            return clock["now"] or datetime.now(tz)

    async def aged_read(*args, **kwargs):
        value = await original(*args, **kwargs)
        observed["value"] = value
        # UTC datetime rounds epoch fractions to microseconds; step past that
        # precision so the simulated adapter time is always after native expiry.
        clock["now"] = datetime.fromtimestamp(value.expires_at, UTC) + timedelta(microseconds=1)
        return value.as_of(value.expires_at) if already_expired else value

    monkeypatch.setattr(api_module, "datetime", AdapterClock)
    monkeypatch.setattr(graph.ha, "read_entity_observation", aged_read)
    response = await graph.client.get("/api/entities/light.synthetic")
    assert response.status_code == 503
    assert response.headers["x-kza-observation-state"] == "stale"
    assert response.headers["x-kza-observation-reason"] == "observation_expired"
    value = observed["value"]
    assert response.headers["x-kza-observed-at"] == datetime.fromtimestamp(value.observed_at, UTC).isoformat()
    assert response.headers["x-kza-expires-at"] == datetime.fromtimestamp(value.expires_at, UTC).isoformat()
    assert response.headers["x-kza-source"] == "degraded"
    assert (await graph.client.get("/api/admin/auth/whoami")).status_code == 200
