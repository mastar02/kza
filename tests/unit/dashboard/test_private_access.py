"""Authenticate every registered private route before reading or mutating state."""

import re
from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.dashboard.api import DashboardAPI
from src.dashboard.live_event_bus import LiveEventBus


def _private_routes():
    app = DashboardAPI(event_bus=LiveEventBus()).app
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path in {"/", "/api/admin/auth/login"}:
            continue
        path = re.sub(r"\{[^}]+\}", "example", route.path)
        for method in sorted(route.methods):
            yield method, path


@pytest.mark.parametrize("method,path", list(_private_routes()))
@pytest.mark.parametrize("credential", ["missing", "wrong", "unconfigured", "foreign_origin"])
def test_all_private_http_routes_reject_before_using_dependencies(method, path, credential, monkeypatch, dashboard_auth):
    services = MagicMock()
    dashboard = DashboardAPI(
        routine_scheduler=services, presence_detector=services, ha_client=services,
        list_manager=services, reminder_manager=services, health_aggregator=services,
        reminder_scheduler=services, event_bus=LiveEventBus(), llm_router=services,
        user_manager=services, alert_manager=services, event_logger=services,
    )
    services.reset_mock()
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
    with TestClient(dashboard.app) as client:
        response = client.request(method, path, json={} if method not in {"GET", "HEAD"} else None, headers=headers)
    assert response.status_code == expected, (method, path, response.text)
    assert services.mock_calls == [], (method, path, services.mock_calls)


@pytest.mark.parametrize("path", ["/ws", "/ws/live"])
@pytest.mark.parametrize("credential", ["missing", "wrong", "unconfigured", "foreign_origin"])
def test_private_websockets_reject_before_subscribing(path, credential, monkeypatch, dashboard_auth):
    bus = LiveEventBus()
    dashboard = DashboardAPI(event_bus=bus)
    headers = {}
    if credential == "wrong":
        headers = {"Authorization": "Bearer not-the-credential"}
    elif credential == "unconfigured":
        monkeypatch.delenv("KZA_DASHBOARD_TOKEN")
    elif credential == "foreign_origin":
        headers = {**dashboard_auth, "Origin": "null"}
    with TestClient(dashboard.app) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(path, headers=headers):
                pytest.fail("Unauthorized socket was accepted")
    assert error.value.code == 1008
    assert bus.subscriber_count == 0
    assert dashboard._ws_clients == []


@pytest.mark.parametrize("origin", ["https://other.invalid", "null", "http://testserver/path", "http://[::1", "http://user@testserver", "http://testserver:0"])
@pytest.mark.parametrize("cookie", [True, False])
def test_invalid_origins_reject_authenticated_reads_and_login(origin, cookie, dashboard_auth):
    dashboard = DashboardAPI()
    with TestClient(dashboard.app) as client:
        if cookie:
            assert client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"}).status_code == 200
        response = client.get("/api/health", headers={"Origin": origin, **({} if cookie else dashboard_auth)})
        assert response.status_code == 403
        login = client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"}, headers={"Origin": origin})
        assert login.status_code == 403


def test_login_cookie_and_logout_protect_read_and_write_lifecycle(dashboard_auth):
    with TestClient(DashboardAPI().app) as client:
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


def test_proxy_origin_uses_effective_https_port(dashboard_auth):
    with TestClient(DashboardAPI().app, base_url="https://home.test") as client:
        login = client.post("/api/admin/auth/login", json={"token": "unit-test-dashboard-session"})
        assert "Secure" in login.headers["set-cookie"]
        assert client.get("/api/health", headers={**dashboard_auth, "Origin": "https://HOME.test:443"}).status_code == 200
        assert client.get("/api/health", headers={**dashboard_auth, "Origin": "http://home.test"}).status_code == 403
