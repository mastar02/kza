"""Tests para adapters reales (use_mocks=False)."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.observability import register_observability_routes
from src.audio.zone_manager import Zone, ZoneState
from src.users.user_manager import User, PermissionLevel
from src.alerts.alert_manager import Alert, AlertType, AlertPriority


@pytest.fixture
def fake_zone_manager():
    z1 = Zone(id="sala", name="Sala", mic_device_index=0, ma1260_zone=1,
              state=ZoneState.IDLE, volume=42)
    z2 = Zone(id="cocina", name="Cocina", mic_device_index=1, ma1260_zone=3,
              state=ZoneState.IDLE, volume=0)
    zm = MagicMock()
    zm.get_all_zones.return_value = {"sala": z1, "cocina": z2}
    return zm


@pytest.fixture
def fake_user_manager():
    u = User(user_id="u_test", name="Test", permission_level=PermissionLevel.ADULT,
             voice_embedding=np.array([0.1, 0.2], dtype=np.float32))
    um = MagicMock()
    um.get_all_users.return_value = [u]
    return um


@pytest.fixture
def fake_alert_manager():
    a = Alert(alert_id="al_1", alert_type=AlertType.SECURITY,
              priority=AlertPriority.CRITICAL, message="movimiento patio",
              details={"zone": "patio", "body": "cámara nada"})
    am = MagicMock()
    am.get_history.return_value = [a]
    return am


@pytest.fixture
def fake_llm_router():
    ep = SimpleNamespace(id="vllm-7b", priority=1,
                         kind=SimpleNamespace(value="fast"),
                         client=SimpleNamespace(base_url="http://localhost:8100"))
    cd = MagicMock()
    cd.is_available.return_value = True
    return SimpleNamespace(_endpoints=[ep], _cd=cd)


def _client(**kwargs):
    app = FastAPI()
    register_observability_routes(app, use_mocks=False, **kwargs)
    return TestClient(app)


def test_zones_adapter_returns_real_data(fake_zone_manager):
    c = _client(zone_manager=fake_zone_manager)
    r = c.get("/api/zones")
    assert r.status_code == 200
    data = r.json()
    assert {z["id"] for z in data} == {"sala", "cocina"}
    sala = next(z for z in data if z["id"] == "sala")
    assert sala["volume"] == 42
    assert sala["ma1260_zone"] == "A"


def test_users_adapter_real_data(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.get("/api/users")
    assert r.status_code == 200
    [u] = r.json()
    assert u["id"] == "u_test"
    assert u["samples"] == 1
    assert u["permissions"]["lights"] is True


def test_alerts_adapter_real_data(fake_alert_manager):
    c = _client(alert_manager=fake_alert_manager)
    r = c.get("/api/alerts")
    assert r.status_code == 200
    [a] = r.json()
    assert a["id"] == "al_1"
    assert a["priority"] == "critical"
    assert a["zone"] == "patio"


def test_alerts_adapter_filter_active(fake_alert_manager):
    c = _client(alert_manager=fake_alert_manager)
    r = c.get("/api/alerts", params={"status": "active"})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_llm_endpoints_adapter_real(fake_llm_router):
    c = _client(llm_router=fake_llm_router)
    r = c.get("/api/llm/endpoints")
    assert r.status_code == 200
    [e] = r.json()
    assert e["id"] == "vllm-7b"
    assert e["state"] == "healthy"
    assert e["url"] == "http://localhost:8100"


def test_clear_cooldown_real(fake_llm_router):
    c = _client(llm_router=fake_llm_router)
    r = c.post("/api/llm/endpoints/vllm-7b/clear-cooldown")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    fake_llm_router._cd.record_success.assert_called_once_with("vllm-7b")


def test_adapter_failure_falls_back_to_mocks():
    bad = MagicMock()
    bad.get_all_zones.side_effect = RuntimeError("kaboom")
    c = _client(zone_manager=bad)
    r = c.get("/api/zones")
    assert r.status_code == 200
    assert len(r.json()) == 5  # mocks tienen 5
