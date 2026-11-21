"""Tests para adapters reales (use_mocks=False)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.observability import register_observability_routes
from src.audio.zone_manager import Zone, ZoneState
from src.users.user_manager import User, PermissionLevel
from src.alerts.alert_manager import AlertManager, AlertPriority, AlertType
from src.alerts.ha_alert_source import HAAlertSource
from src.home_assistant.ha_client import HAStatesSnapshot
from tests.unit.alerts.test_ha_alert_source import BATTERY, DOOR, Reader, state


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
    from src.core.account_metadata import AccountMetadata
    um.project_accounts.return_value = (AccountMetadata(u.user_id, u.name, 'adult', True),)
    return um


@pytest.fixture
def fake_llm_router():
    ep = SimpleNamespace(id="vllm-7b", priority=1,
                         kind=SimpleNamespace(value="fast"),
                         client=SimpleNamespace(base_url="http://localhost:8100"))
    cd = MagicMock()
    cd.is_available.return_value = True
    return SimpleNamespace(_endpoints=[ep], _cd=cd)


def _client(dashboard_auth, **kwargs):
    app = FastAPI()
    register_observability_routes(app, use_mocks=False, **kwargs)
    return TestClient(app, headers=dashboard_auth)


def test_zones_adapter_returns_real_data(fake_zone_manager, dashboard_auth):
    c = _client(dashboard_auth, zone_manager=fake_zone_manager)
    r = c.get("/api/zones")
    assert r.status_code == 200
    assert r.headers.get("X-KZA-Source") == "real"
    data = r.json()
    assert {z["id"] for z in data} == {"sala", "cocina"}
    sala = next(z for z in data if z["id"] == "sala")
    assert sala["volume"] == 42
    assert sala["ma1260_zone"] == "A"


def test_users_adapter_real_data(fake_user_manager, dashboard_auth):
    c = _client(dashboard_auth, user_manager=fake_user_manager)
    r = c.get("/api/users")
    assert r.status_code == 200
    [u] = r.json()
    assert u["id"] == "u_test"
    assert u["samples"] is None and u["permissions"] == {}
    assert u["emotions"] == {} and u["pca"] == []
    assert r.headers["X-KZA-Observation-State"] == "partial"
    assert "effective_permissions" in r.headers["X-KZA-Unobserved-Fields"]


async def test_alerts_adapter_keeps_history_with_no_producer(dashboard_auth):
    manager = AlertManager()
    alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, "movimiento patio",
                                       {"zone": "patio", "body": "cámara nada"})
    c = _client(dashboard_auth, alert_manager=manager)
    r = c.get("/api/alerts", params={"status": "active"})
    assert r.status_code == 200
    assert r.headers["X-KZA-Source"] == "degraded"
    data = r.json()
    assert data["available"] is False and data["producers"] == []
    [a] = data["alerts"]
    assert a["id"] == alert.alert_id
    assert a["priority"] == "critical"
    assert a["zone"] == "patio"
    assert c.get("/api/alerts", params={"status": "acked"}).json()["alerts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnected", "stale", "partial", "overflow"])
async def test_alerts_adapter_projects_live_producer_evidence(dashboard_auth, failure):
    manager, reader = AlertManager(), Reader()
    reader.now = datetime.now(UTC).timestamp() - (30 if failure == "stale" else 1)
    source = HAAlertSource(reader, manager, {
        "door_window": {DOOR: {}}, "battery": {BATTERY: {}}, "queue_maxsize": 1,
    }, clock=lambda: reader.now)
    manager.producer_status = source.get_producer_status
    client = _client(dashboard_auth, alert_manager=manager)
    before = client.get("/api/alerts").json()
    assert before["available"] is False
    assert all(row["last_check_at"] is None and not row["connected"] for row in before["producers"])

    await source.start()
    try:
        reader.rest(state(DOOR, "off", reader.now), state(BATTERY, "95", reader.now))
        await source.reconcile()
        if failure != "stale":
            healthy = client.get("/api/alerts").json()
            assert healthy["available"] is True and healthy["alerts"] == []
            assert all(row["connected"] and row["last_check_at"] and row["observed_at"] and row["expires_at"]
                       for row in healthy["producers"])
        if failure == "partial":
            reader.rest(state(DOOR, "off", reader.now))
            await source.reconcile()
        elif failure == "disconnected":
            reader.generation = None
            reader.now += 1
            reader.snapshot = HAStatesSnapshot(False, None, "ha_states_unavailable")
            await source.reconcile()
        elif failure == "stale":
            reader.now += 30
        else:
            reader.publish(DOOR, "off")
            reader.publish(BATTERY, "95")

        response = client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False and data["alerts"] == []
        rows = {row["id"]: row for row in data["producers"]}
        door, battery = rows["door_open_unexpected"], rows["battery_low"]
        assert DOOR not in response.text and BATTERY not in response.text
        if failure == "partial":
            assert response.headers["X-KZA-Observation-State"] == "partial"
            assert door["available"] is True and battery["available"] is False and battery["uncertain"]
        elif failure == "stale":
            assert response.headers["X-KZA-Observation-State"] == "stale"
            assert all(row["state"] == "stale" and row["expires_at"] for row in rows.values())
        elif failure == "disconnected":
            assert all(not row["connected"] and row["state"] == "unavailable" and row["observed_at"]
                       and row["last_check_at"] > row["observed_at"] for row in rows.values())
        else:
            assert all(row["overflow"] == 1 and row["uncertain"] for row in rows.values())
    finally:
        await source.stop()


def test_llm_endpoints_adapter_real(fake_llm_router, dashboard_auth):
    c = _client(dashboard_auth, llm_router=fake_llm_router)
    r = c.get("/api/llm/endpoints")
    assert r.status_code == 200
    [e] = r.json()
    assert e["id"] == "vllm-7b"
    assert e["state"] == "unknown"
    assert e["last_check"] is None and e["failures_7d"] == {}
    assert r.headers["X-KZA-Observation-State"] == "partial"
    assert e["url"] == "http://localhost:8100"


def test_clear_cooldown_real(fake_llm_router, dashboard_auth, tmp_path):
    from src.llm.cooldown import CooldownManager
    from src.llm.types import ErrorKind
    cooldown = CooldownManager(tmp_path / "cooldown.json")
    cooldown.record_failure("vllm-7b", ErrorKind.BILLING)
    fake_llm_router._cd = cooldown
    client = _client(dashboard_auth, llm_router=fake_llm_router)
    response = client.post("/api/llm/endpoints/vllm-7b/clear-cooldown")
    assert response.status_code == 200 and response.json()["ok"] is True
    assert cooldown.get_state("vllm-7b").error_count == 0
    restored = CooldownManager(cooldown.persistence_path)
    assert restored.get_state("vllm-7b").error_count == 0
    assert restored.is_available("vllm-7b")


def test_missing_cooldown_router_cannot_acknowledge_a_demo_effect(dashboard_auth):
    response = _client(dashboard_auth).post("/api/llm/endpoints/vllm-7b/clear-cooldown")
    assert response.status_code == 503
    assert response.json() == {"detail": "cooldown manager no disponible"}


def test_adapter_failure_is_unavailable_without_demo_data(dashboard_auth):
    bad = MagicMock()
    bad.get_all_zones.side_effect = RuntimeError("kaboom")
    c = _client(dashboard_auth, zone_manager=bad)
    r = c.get("/api/zones")
    assert r.status_code == 200
    assert r.headers.get("X-KZA-Source") == "degraded"
    assert r.json() == []
    assert r.headers["X-KZA-Observation-State"] == "unavailable"
    assert "X-KZA-Observed-At" not in r.headers


def test_real_empty_and_failed_reads_have_distinct_metadata_without_renewing_dates(dashboard_auth):
    manager = MagicMock()
    manager.get_all_zones.return_value = {}
    with _client(dashboard_auth, zone_manager=manager) as client:
        first = client.get('/api/zones')
        assert first.json() == [] and first.headers['X-KZA-Source'] == 'real'
        assert first.headers['X-KZA-Source-Id'] == 'zone-manager'
        manager.get_all_zones.side_effect = RuntimeError('synthetic failure')
        failed = client.get('/api/zones')
        assert failed.json() == [] and failed.headers['X-KZA-Source'] == 'degraded'
        assert failed.headers['X-KZA-Observation-State'] == 'unavailable'
        assert failed.headers['X-KZA-Observed-At'] == first.headers['X-KZA-Observed-At']
        assert failed.headers['X-KZA-Expires-At'] == first.headers['X-KZA-Expires-At']
        assert failed.headers['Cache-Control'] == 'no-store'


def test_warn_logged_when_use_mocks_with_real_services_injected(caplog, dashboard_auth):
    from fastapi import FastAPI
    from src.dashboard.observability import register_observability_routes
    from unittest.mock import MagicMock as _M
    app = FastAPI()
    with caplog.at_level("WARNING"):
        register_observability_routes(app, use_mocks=True, zone_manager=_M())
    assert any("use_mocks=True con servicios reales" in m for m in caplog.messages)


async def test_alert_instant_is_explicit_utc_for_naive_alert_manager_timestamps():
    """AlertManager stores naive UTC; clients receive an offset-bearing instant."""
    from src.dashboard.observability import _alerts_adapter
    manager = AlertManager(now=lambda: datetime(2026, 11, 10, 13, 39, 43))
    await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "m")
    [row] = _alerts_adapter(manager, None)
    assert row["at"] == "2026-11-10T13:39:43+00:00"
    assert row["ts"] == "13:39:43"


def test_llm_cooldown_until_is_zone_explicit_and_legacy_field_kept(fake_llm_router):
    from datetime import UTC, datetime
    from src.dashboard.observability import _llm_endpoints_adapter
    eta = datetime(2026, 11, 10, 14, 0, 5, tzinfo=UTC).timestamp()
    fake_llm_router._cd.is_available.return_value = False
    fake_llm_router._cd.next_attempt_at.return_value = eta
    [row] = _llm_endpoints_adapter(fake_llm_router)
    assert row["state"] == "cooldown"
    assert row["cooldown_until"] == "2026-11-10T14:00:05+00:00"
    assert row["cooldown_ends"] == datetime.fromtimestamp(eta).strftime("%H:%M:%S")


def test_ha_action_instant_is_zone_explicit():
    from datetime import UTC, datetime
    from src.dashboard.observability import _ha_actions_adapter
    ts = datetime(2026, 11, 10, 9, 15, 0, tzinfo=UTC).timestamp()
    event = SimpleNamespace(entity_id="light.sala", context={}, timestamp=ts, user_name=None,
                            user_id="u_test", action="turn_on",
                            datetime=datetime.fromtimestamp(ts))
    logger_ = MagicMock()
    logger_.get_events.return_value = [event]
    rows = _ha_actions_adapter(logger_)
    assert rows and rows[0]["at"] == "2026-11-10T09:15:00+00:00"
