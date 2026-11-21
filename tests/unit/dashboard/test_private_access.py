"""Authenticate every registered private route before reading or mutating state."""

from functools import lru_cache
import re
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.dashboard.api import DashboardAPI
from src.dashboard.live_event_bus import LiveEventBus
from tests.mocks.dashboard_sessions import session_client


@pytest.fixture
def _session_client(tmp_path):
    sequence = []
    def create(**kwargs):
        def app_factory(authority):
            api = DashboardAPI(authority=authority)
            return api.app, api.stop
        path = tmp_path / f"app-{len(sequence)}" / "auth.db"
        sequence.append(path)
        return session_client(app_factory, path, **kwargs)
    return create


@lru_cache(maxsize=1)
def _private_routes() -> tuple[tuple[str, str], ...]:
    app = DashboardAPI(event_bus=LiveEventBus()).app
    routes: list[tuple[str, str]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path in {"/", "/api/admin/auth/login"}:
            continue
        path = re.sub(r"\{[^}]+\}", "example", route.path)
        for method in sorted(route.methods):
            if (method, route.path) in {
                ("POST", "/api/capture/recorders/retire"),
                ("POST", "/api/capture/native/calibration/retire"),
                ("POST", "/api/capture/native/wakeword-training/retire"),
                ("POST", "/api/capture/native/wakeword-diagnostics/retire"),
                ("POST", "/api/capture/native/zone-setup/retire"),
                ("POST", "/api/capture/native/manual-voice-command/retire"),
            }:
                continue
            routes.append((method, path))
    return tuple(routes)


def _make_mocked_dashboard() -> tuple[DashboardAPI, MagicMock]:
    services = MagicMock()
    dashboard = DashboardAPI(
        routine_scheduler=services, presence_detector=services, ha_client=services,
        list_manager=services, reminder_manager=services, health_aggregator=services,
        reminder_scheduler=services, event_bus=LiveEventBus(), llm_router=services,
        user_manager=services, alert_manager=services, event_logger=services,
    )
    services.reset_mock()
    return dashboard, services


@pytest.fixture(scope="module")
def _private_http_client():
    """One mocked dashboard per module; the credential is read per request, so tests set it."""
    dashboard, services = _make_mocked_dashboard()
    with TestClient(dashboard.app) as client:
        # The startup window stays inside the oracle: nothing may touch a dependency there either.
        assert services.mock_calls == [], services.mock_calls
        yield client, dashboard, services


@pytest.fixture(scope="module")
def _private_ws_client():
    bus = LiveEventBus()
    dashboard = DashboardAPI(event_bus=bus)
    with TestClient(dashboard.app) as client:
        assert bus.subscriber_count == 0 and dashboard._ws_clients == []
        yield client, dashboard, bus


@pytest.mark.parametrize("method,path", _private_routes())
@pytest.mark.parametrize("credential", ["missing", "wrong", "unconfigured", "foreign_origin"])
def test_all_private_http_routes_reject_before_using_dependencies(
    method, path, credential, monkeypatch, dashboard_auth, _private_http_client,
):
    headers = {}
    expected = 401
    if credential == "wrong":
        headers = {"Authorization": "Bearer not-the-credential"}
    elif credential == "unconfigured":
        monkeypatch.delenv("KZA_DASHBOARD_TOKEN")
        expected = 503
    elif credential == "foreign_origin":
        headers = {**dashboard_auth, "Origin": "https://other.invalid"}
        expected = 403
    client, _dashboard, services = _private_http_client
    client.cookies.clear()
    services.reset_mock()
    response = client.request(method, path, json={} if method not in {"GET", "HEAD"} else None, headers=headers)
    assert response.status_code == expected, (method, path, response.text)
    assert services.mock_calls == [], (method, path, services.mock_calls)


@pytest.mark.parametrize("path", ["/ws", "/ws/live", "/ws/evaluation"])
@pytest.mark.parametrize("credential", ["missing", "wrong", "unconfigured", "foreign_origin"])
def test_private_websockets_reject_before_subscribing(
    path, credential, monkeypatch, dashboard_auth, _private_ws_client,
):
    headers = {}
    if credential == "wrong":
        headers = {"Authorization": "Bearer not-the-credential"}
    elif credential == "unconfigured":
        monkeypatch.delenv("KZA_DASHBOARD_TOKEN")
    elif credential == "foreign_origin":
        headers = {**dashboard_auth, "Origin": "null"}
    client, dashboard, bus = _private_ws_client
    client.cookies.clear()
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(path, headers=headers):
            pytest.fail("Unauthorized socket was accepted")
    assert error.value.code == 1008
    assert bus.subscriber_count == 0
    assert dashboard._ws_clients == []


@pytest.mark.parametrize("origin", ["https://other.invalid", "null", "http://testserver/path", "http://[::1", "http://user@testserver", "http://testserver:0"])
@pytest.mark.parametrize("cookie", [True, False])
def test_invalid_origins_reject_authenticated_reads_and_login(origin, cookie, dashboard_auth, _session_client):
    with _session_client() as client:
        if cookie:
            assert client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"}).status_code == 200
        response = client.get("/api/health", headers={"Origin": origin, **({} if cookie else dashboard_auth)})
        assert response.status_code == 403
        login = client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"}, headers={"Origin": origin})
        assert login.status_code == 403


def test_login_cookie_and_logout_protect_read_and_write_lifecycle(dashboard_auth, _session_client):
    with _session_client() as client:
        assert client.get("/api/health").status_code == 401
        assert client.post("/api/admin/auth/logout").status_code == 401
        assert client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"}).status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.post("/api/admin/auth/logout").status_code == 200
        assert client.get("/api/health").status_code == 401


def test_default_dependency_also_protects_a_new_route(dashboard_auth):
    dashboard = DashboardAPI()
    called = []

    @dashboard.app.get("/future-private-route")
    async def future():
        called.append(True)
        return {"private": True}

    # Static mounts normally precede routes appended after construction. Move
    # this test route first to exercise the inherited dependency itself.
    dashboard.app.router.routes.insert(0, dashboard.app.router.routes.pop())
    with TestClient(dashboard.app) as client:
        assert client.get("/future-private-route").status_code == 401
        assert called == []
        assert client.get("/future-private-route", headers=dashboard_auth).status_code == 200
        assert called == [True]


def test_public_shell_assets_and_unconfigured_login_have_no_private_data(monkeypatch):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    with TestClient(DashboardAPI().app) as client:
        for path in ("/", "/home/home.js", "/obs/", "/obs/src/app.jsx"):
            assert client.get(path).status_code == 200
        assert client.post("/api/admin/auth/login", json={"token": "unused"}).status_code == 503
        assert client.get("/api/health").status_code == 503
        assert all(getattr(route, "path", None) not in {"/docs", "/redoc", "/openapi.json"} for route in client.app.routes)


def test_proxy_origin_uses_effective_https_port(dashboard_auth, _session_client):
    with _session_client(base_url="https://home.test") as client:
        login = client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"})
        assert "Secure" in login.headers["set-cookie"]
        assert client.get("/api/health", headers={**dashboard_auth, "Origin": "https://HOME.test:443"}).status_code == 200
        assert client.get("/api/health", headers={**dashboard_auth, "Origin": "http://home.test"}).status_code == 403


def test_every_task_route_is_a_household_task_not_an_operator_default():
    """T035-U1: lists and reminders need tasks:read/tasks:write, held by every household session."""
    from starlette.requests import Request

    from src.dashboard.auth import _route_permissions
    from src.dashboard.authority import DashboardRole, _permissions

    task_routes = [(method, path) for method, path in _private_routes()
                   if path.startswith(("/api/lists", "/api/reminders"))
                   and path not in {"/api/reminders/status"} and not path.startswith("/api/reminders/notices")]
    assert {path for _, path in task_routes} == {"/api/lists", "/api/lists/example/items",
                                                 "/api/lists/example/items/example", "/api/reminders",
                                                 "/api/reminders/example"}
    for method, path in task_routes:
        needed = _route_permissions(Request({"type": "http", "method": method, "path": path,
                                             "query_string": b"", "headers": []}))
        assert needed == {"tasks:read" if method == "GET" else "tasks:write"}, (method, path)
        assert needed <= _permissions(DashboardRole.HOME, None)
        assert needed <= _permissions(DashboardRole.OPERATOR, None)
        assert not needed & _permissions(DashboardRole.EVALUATOR, None)
