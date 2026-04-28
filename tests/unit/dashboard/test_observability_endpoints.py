"""Tests para endpoints observability en USE_MOCKS=True (sin servicios inyectados)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard import observability, system_monitor
from src.dashboard.observability import register_observability_routes


@pytest.fixture
def client(monkeypatch):
    # Aislar de la realidad del host: aunque haya nvidia-smi/systemctl en la
    # dev box, los endpoints /api/system/* deben caer a mocks acá.
    monkeypatch.setattr(system_monitor, "gpu_snapshot", lambda: None)
    monkeypatch.setattr(system_monitor, "services_snapshot", lambda: None)
    app = FastAPI()
    register_observability_routes(app, use_mocks=True)
    return TestClient(app)


def test_zones_returns_five_rooms(client):
    r = client.get("/api/zones")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    assert {z["id"] for z in data} == {"sala", "dormitorio", "cocina", "estudio", "patio"}


def test_conversations_filter_by_user(client):
    r = client.get("/api/conversations", params={"user": "Lucía"})
    assert r.status_code == 200
    assert all(c["user"] == "Lucía" for c in r.json())


def test_conversations_filter_by_path(client):
    r = client.get("/api/conversations", params={"path": "slow"})
    assert r.status_code == 200
    assert all(c["path"] == "slow" for c in r.json())


def test_ha_entities_filter_by_domain(client):
    r = client.get("/api/ha/entities", params={"domain": "climate"})
    assert r.status_code == 200
    assert all(e["domain"] == "climate" for e in r.json())


def test_llm_endpoints_shape(client):
    r = client.get("/api/llm/endpoints")
    assert r.status_code == 200
    data = r.json()
    assert any(e["state"] == "cooldown" for e in data)
    for e in data:
        assert {"id", "name", "priority", "role", "state", "failures_7d"} <= set(e.keys())


def test_clear_cooldown_mocked(client):
    r = client.post("/api/llm/endpoints/ik-30b-cpu/clear-cooldown")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "endpoint_id": "ik-30b-cpu", "mocked": True}


def test_alerts_filter_active(client):
    r = client.get("/api/alerts", params={"status": "active"})
    assert r.status_code == 200
    assert all(a["acked"] is False for a in r.json())


def test_alerts_filter_acked(client):
    r = client.get("/api/alerts", params={"status": "acked"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    assert all(a["acked"] is True for a in data)


def test_source_header_is_mock_when_no_services(client):
    r = client.get("/api/zones")
    assert r.headers.get("X-KZA-Source") == "mock"


def test_system_gpus_two_devices(client):
    r = client.get("/api/system/gpus")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert data[0]["vramTotal"] == 8
    assert "procs" in data[0]


def test_system_services_have_kza_voice(client):
    r = client.get("/api/system/services")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "kza-voice" in names
    assert "kza-llm-ik" in names


def test_users_have_pca_and_emotions(client):
    r = client.get("/api/users")
    assert r.status_code == 200
    for u in r.json():
        assert "pca" in u and isinstance(u["pca"], list)
        assert "emotions" in u
        assert "permissions" in u


def test_ha_actions_returned(client):
    r = client.get("/api/ha/actions")
    assert r.status_code == 200
    actions = r.json()
    assert len(actions) >= 1
    assert {"id", "ts", "idem", "service", "ok"} <= set(actions[0].keys())
