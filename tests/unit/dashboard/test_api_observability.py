"""Tests for observability API endpoints: health, metrics, subsystems, failures, reminders/status."""

import asyncio
from datetime import datetime
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from src.alerts.alert_manager import AlertCondition, AlertDelivery, AlertPriority, AlertType
from src.core.household_clock import HouseholdClock
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.monitoring.health_aggregator import HealthAggregator
from src.pipeline.feature_manager import build_feature_manager
from src.reminders.reminder_store import ReminderStore, ReminderScope
from src.reminders.reminder_manager import ReminderManager


class AlertOutputs:
    """Record the low-level output route and return a typed delivery receipt."""

    def __init__(self):
        self.spoken = []

    def playback_output_key(self, zone):
        return zone

    def get_all_zones(self):
        return {}

    async def aspeak(self, text, *, zone_id, timeout_s):
        assert timeout_s > 0
        self.spoken.append(zone_id)
        return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.COMPLETE,
                              spoken_segments=1, output_started=True)


def alert_features(outputs, store=None, clock=None, *, ha_zone="zone_living"):
    return build_feature_manager({
        "rooms": {"living": {}, "kitchen": {}, "bedroom": {}},
        "alerts": {"ha": {"output_zone": ha_zone}},
        "tts": {"streaming": {"default_output_zone": "zone_bedroom"}},
    }, response_handler=outputs, zone_manager=outputs, reminder_store=store, household_clock=clock)


@pytest.mark.asyncio
async def test_alert_zones_persist_and_guide_spoken_output(tmp_path, monkeypatch):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    store_path = str(tmp_path / "alert-zones.db")
    now = [0.0]
    clock = HouseholdClock(time_source=lambda: now[0])
    now[0] = clock.to_epoch(datetime(2026, 11, 21, 7))
    outputs = AlertOutputs()
    store = ReminderStore(store_path)
    await store.initialize()
    features = alert_features(outputs, store, clock)
    features.bind_alert_mute(lambda: False)
    manager = features.alert_manager
    api = DashboardAPI(alert_manager=manager)
    try:
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as guest:
            assert (await guest.get("/api/alerts")).status_code == 503
        authority = DashboardAuthority(tmp_path / "auth.db", expected_token=lambda: "synthetic-root")
        await authority.initialize()
        api.app.state.dashboard_authority = authority
        try:
            transport = ASGITransport(app=api.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/api/alerts")).status_code == 401
                client.headers["Authorization"] = "Bearer synthetic-root"
                initial = (await client.get("/api/alerts/zones")).json()
                assert initial["zones"] == ["zone_living"] and initial["using_default"] and initial["revision"] == 0
                await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "Una alerta")
                assert outputs.spoken == ["zone_living"]
                for invalid in ("zone_missing", "all", "default"):
                    assert (await client.put("/api/alerts/zones", json={"zones": [invalid]})).status_code == 422
                saved = await client.put("/api/alerts/zones", json={
                    "zones": ["zone_kitchen", "bedroom", "kitchen"], "expected_revision": 0,
                })
                selected = ["zone_kitchen", "zone_bedroom"]
                assert saved.status_code == 200 and saved.json()["zones"] == selected
                assert saved.json()["revision"] == 1 and not saved.json()["using_default"]
                assert (await client.put("/api/alerts/zones", json={"zones": [], "expected_revision": 0})).status_code == 409
                await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "Otra alerta")
                assert outputs.spoken == ["zone_living", *selected]
                await store.close()
                await store.initialize()
                assert (await client.get("/api/alerts")).json()["output"]["zones"] == selected
                saved = await client.put("/api/alerts/zones", json={"zones": [], "expected_revision": 1})
                assert saved.status_code == 200 and saved.json()["zones"] == [] and not saved.json()["using_default"]
                silent = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "Solo en la web")
                assert (silent.delivery, silent.delivery_reason) == (AlertDelivery.MUTED, "alert_no_output_zones")
                assert outputs.spoken == ["zone_living", *selected]
                assert any(item["id"] == silent.alert_id for item in (await client.get("/api/alerts")).json()["alerts"])
                reset = await client.put("/api/alerts/zones", json={"zones": None, "expected_revision": 2})
                assert reset.json()["zones"] == ["zone_living"] and reset.json()["using_default"]
        finally:
            await authority.aclose()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_alert_quiet_hours_and_mute_leave_one_notice_per_episode():
    now = [0.0]
    clock = HouseholdClock(time_source=lambda: now[0])
    outputs = AlertOutputs()
    features = alert_features(outputs, clock=clock, ha_zone=None)
    manager = features.alert_manager
    muted = False
    now[0] = clock.to_epoch(datetime(2026, 11, 21, 12))
    no_policy = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "Control sin conectar")
    assert no_policy.delivery_reason == "alert_mute_unavailable" and not outputs.spoken
    features.bind_alert_mute(lambda: muted)

    for hour, minute, expected in [
        (6, 59, "alert_quiet_hours"), (7, 0, None), (22, 59, None), (23, 0, "alert_quiet_hours"),
    ]:
        now[0] = clock.to_epoch(datetime(2026, 11, 21, hour, minute))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, f"Episodio {hour}")
        assert alert.delivery_reason == expected
    assert outputs.spoken == ["zone_bedroom", "zone_bedroom"]
    now[0] = clock.to_epoch(datetime(2026, 11, 22, 12))
    condition = AlertCondition("synthetic", "door_open_unexpected")
    muted = True
    alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "Puerta abierta",
                                       condition=condition, defer_voice=True)
    assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.MUTED, "alert_global_mute")
    muted = False
    await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, "Escalada", condition=condition)
    await manager.deliver_deferred_notice(AlertType.SECURITY, condition, is_current=lambda: True)
    assert outputs.spoken == ["zone_bedroom", "zone_bedroom"]
    assert alert in manager.get_active_alerts()
    await manager.resolve_condition(AlertType.SECURITY, condition)
    await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "Nueva apertura",
                               condition=condition, defer_voice=True)
    await manager.deliver_deferred_notice(AlertType.SECURITY, condition, is_current=lambda: True)
    await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, "Nueva escalada", condition=condition)
    await manager.deliver_deferred_notice(AlertType.SECURITY, condition, is_current=lambda: True)
    assert outputs.spoken == ["zone_bedroom"] * 3
    now[0] = clock.to_epoch(datetime(2026, 11, 22, 22, 59))
    late = AlertCondition("synthetic", "late_notice")
    pending = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "Una apertura de noche",
                                         condition=late, defer_voice=True)
    now[0] = clock.to_epoch(datetime(2026, 11, 22, 23))
    await manager.deliver_deferred_notice(AlertType.SECURITY, late, is_current=lambda: True)
    assert (pending.delivery, pending.delivery_reason) == (AlertDelivery.MUTED, "alert_quiet_hours")
    assert outputs.spoken == ["zone_bedroom"] * 3


@pytest.mark.asyncio
async def test_shared_alerts_visitor_and_revoked_admin(tmp_path, monkeypatch):
    store = ReminderStore(str(tmp_path / "reminders.db"))
    authority = DashboardAuthority(tmp_path / "authority.db", expected_token=lambda: "synthetic-root")
    await store.initialize()
    await authority.initialize()
    manager = alert_features(AlertOutputs(), store).alert_manager
    api = DashboardAPI(alert_manager=manager, authority=authority)
    alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "Alerta compartida")
    ack_path = f"/api/admin/alerts/{alert.alert_id}/ack"
    try:
        root = authority.resolve("Bearer synthetic-root")
        grant = await authority.issue_credential(root, subject="synthetic-visitor", role=DashboardRole.HOME,
                                                 expires_at=time.time() + 60)
        visitor = await authority.login(grant.token)
        admin = await authority.login("synthetic-root")
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
            client.headers["Authorization"] = f"Bearer {visitor.token}"
            assert (await client.get("/api/alerts")).json()["alerts"][0]["id"] == alert.alert_id
            assert (await client.get("/api/alerts/zones")).status_code == 200
            assert (await client.post(ack_path)).status_code == 403
            assert (await client.delete(f"/api/admin/alerts/{alert.alert_id}")).status_code == 403
            assert (await client.put("/api/alerts/zones", json={"zones": []})).status_code == 403
            admitted = asyncio.Event()
            mark_processed = manager.mark_processed

            async def mark_after_signal(*args, **kwargs):
                admitted.set()
                return await mark_processed(*args, **kwargs)

            monkeypatch.setattr(manager, "mark_processed", mark_after_signal)
            client.headers["Authorization"] = f"Bearer {admin.token}"
            async with manager._lock:
                ack = asyncio.create_task(client.post(ack_path))
                await asyncio.wait_for(admitted.wait(), timeout=5)
                assert await authority.logout(admin.lease)
            assert (await ack).status_code == 403
            assert (await client.post(ack_path)).status_code == 401
            assert (await client.put("/api/alerts/zones", json={"zones": []})).status_code == 401
            assert not alert.processed and await store.get_alert_zones() is None
    finally:
        await authority.aclose()
        await store.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ha_client():
    client = MagicMock()
    health = MagicMock()
    health.state = "connected"
    health.error_count = 0
    health.success_count = 10
    health.avg_latency_ms = 30.0
    health.ws_connected = True
    health.last_error_message = ""
    client.get_health_status.return_value = health
    return client


@pytest.fixture
def mock_latency_monitor():
    monitor = MagicMock()
    monitor.get_stats.return_value = {
        "session": {
            "total_commands": 50,
            "success_rate_percent": 96.0,
        },
        "total": {
            "p50_ms": 110.0,
            "p95_ms": 240.0,
            "p99_ms": 285.0,
        },
        "components": {},
    }
    return monitor


@pytest.fixture
def mock_priority_queue():
    queue = MagicMock()
    queue.get_stats.return_value = {
        "queue_size": 2,
        "max_queue_size": 100,
        "total_enqueued": 20,
        "total_processed": 18,
        "total_cancelled": 1,
        "total_timeout": 0,
        "current_processing": None,
    }
    return queue


@pytest.fixture
def mock_reminder_scheduler():
    from src.reminders.reminder_scheduler import NoticeEntry

    scheduler = MagicMock()
    scheduler._running = True
    # T034-U2: status counts come from durable notice receipts, not in-process retry counters.
    scheduler.notice_entries = AsyncMock(return_value=[
        NoticeEntry("r1", 1.0, "uno", "pending", None, 1, 1, 1.0),
        NoticeEntry("r2", 2.0, "dos", "pending", None, 0, 0, None),
        NoticeEntry("r3", 3.0, "tres", "uncertain", "delivery_cancelled", 1, 0, 3.0),
        NoticeEntry("r4", 4.0, "cuatro", "confirmed", None, 2, 1, 4.0),
    ])
    return scheduler


@pytest.fixture
def health_aggregator(mock_ha_client, mock_latency_monitor, mock_priority_queue, mock_reminder_scheduler):
    return HealthAggregator(
        ha_client=mock_ha_client,
        latency_monitor=mock_latency_monitor,
        priority_queue=mock_priority_queue,
        reminder_scheduler=mock_reminder_scheduler,
    )


@pytest_asyncio.fixture
async def api_with_observability(tmp_path, health_aggregator, mock_reminder_scheduler):
    """DashboardAPI wired with observability dependencies."""
    reminder_store = ReminderStore(str(tmp_path / "test_obs.db"))
    await reminder_store.initialize()
    reminder_mgr = ReminderManager(store=reminder_store, config={})

    # Seed a reminder so pending count > 0
    await reminder_store.create(
        user_id="u1",
        text="take medicine",
        trigger_at=time.time() + 3600,
        scope=ReminderScope.SHARED,
    )

    dashboard = DashboardAPI(
        health_aggregator=health_aggregator,
        reminder_manager=reminder_mgr,
        reminder_scheduler=mock_reminder_scheduler,
    )
    yield dashboard
    await reminder_store.close()


@pytest_asyncio.fixture
async def client(api_with_observability, dashboard_auth):
    transport = ASGITransport(app=api_with_observability.app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests — /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_status_field(self, client):
        data = (await client.get("/api/health")).json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_health_has_timestamp(self, client):
        data = (await client.get("/api/health")).json()
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_lists_subsystems(self, client):
        data = (await client.get("/api/health")).json()
        assert "subsystems" in data
        names = [s["name"] for s in data["subsystems"]]
        assert "home_assistant" in names
        assert "latency_monitor" in names
        assert "priority_queue" in names
        assert "reminders" in names

    @pytest.mark.asyncio
    async def test_health_subsystem_shape(self, client):
        data = (await client.get("/api/health")).json()
        for sub in data["subsystems"]:
            assert "name" in sub
            assert "status" in sub
            assert "detail" in sub
            assert "extra" in sub

    @pytest.mark.asyncio
    async def test_health_without_aggregator(self, tmp_path, dashboard_auth):
        """When health_aggregator is None, returns fallback."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
            resp = await c.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests — /api/metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, client):
        resp = await client.get("/api/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_shape(self, client):
        data = (await client.get("/api/metrics")).json()
        assert "latency" in data
        assert "p50_ms" in data["latency"]
        assert "p95_ms" in data["latency"]
        assert "p99_ms" in data["latency"]
        assert "queue_depth" in data
        assert "command_count" in data

    @pytest.mark.asyncio
    async def test_metrics_values(self, client):
        data = (await client.get("/api/metrics")).json()
        assert data["latency"]["p50_ms"] == 110.0
        assert data["latency"]["p95_ms"] == 240.0
        assert data["latency"]["p99_ms"] == 285.0
        assert data["queue_depth"] == 2
        assert data["command_count"] == 50

    @pytest.mark.asyncio
    async def test_metrics_without_aggregator(self, tmp_path, dashboard_auth):
        """Without aggregator, returns zero-valued metrics."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
            data = (await c.get("/api/metrics")).json()
        assert data["latency"]["p50_ms"] == 0.0
        assert data["queue_depth"] == 0


# ---------------------------------------------------------------------------
# Tests — /api/subsystems
# ---------------------------------------------------------------------------

class TestSubsystemsEndpoint:

    @pytest.mark.asyncio
    async def test_subsystems_returns_200(self, client):
        resp = await client.get("/api/subsystems")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_subsystems_shape(self, client):
        data = (await client.get("/api/subsystems")).json()
        assert "subsystems" in data
        for sub in data["subsystems"]:
            assert "name" in sub
            assert "status" in sub
            assert "detail" in sub
            assert "extra" in sub

    @pytest.mark.asyncio
    async def test_subsystems_includes_all(self, client):
        data = (await client.get("/api/subsystems")).json()
        names = [s["name"] for s in data["subsystems"]]
        assert len(names) == 4
        assert "home_assistant" in names

    @pytest.mark.asyncio
    async def test_subsystems_without_aggregator(self, tmp_path, dashboard_auth):
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
            data = (await c.get("/api/subsystems")).json()
        assert data["subsystems"] == []


# ---------------------------------------------------------------------------
# Tests — /api/failures
# ---------------------------------------------------------------------------

class TestFailuresEndpoint:

    @pytest.mark.asyncio
    async def test_failures_returns_200(self, client):
        resp = await client.get("/api/failures")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_failures_empty_initially(self, client):
        data = (await client.get("/api/failures")).json()
        assert data["failures"] == []

    @pytest.mark.asyncio
    async def test_failures_after_recording(self, client, health_aggregator):
        health_aggregator.record_failure("ha", "connection refused", detail="10.0.0.1")

        data = (await client.get("/api/failures")).json()
        assert len(data["failures"]) == 1
        f = data["failures"][0]
        assert f["subsystem"] == "ha"
        assert f["message"] == "connection refused"
        assert f["detail"] == "10.0.0.1"
        assert "timestamp" in f

    @pytest.mark.asyncio
    async def test_failures_limit_param(self, client, health_aggregator):
        for i in range(10):
            health_aggregator.record_failure("test", f"err_{i}")

        data = (await client.get("/api/failures", params={"limit": 3})).json()
        assert len(data["failures"]) == 3

    @pytest.mark.asyncio
    async def test_failures_without_aggregator(self, tmp_path, dashboard_auth):
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
            data = (await c.get("/api/failures")).json()
        assert data["failures"] == []


# ---------------------------------------------------------------------------
# Tests — /api/reminders/status
# ---------------------------------------------------------------------------

class TestRemindersStatusEndpoint:

    @pytest.mark.asyncio
    async def test_status_returns_200(self, client):
        resp = await client.get("/api/reminders/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_shape(self, client):
        data = (await client.get("/api/reminders/status")).json()
        assert "pending_count" in data
        assert "next_trigger_at" in data
        assert "scheduler_running" in data
        assert "notices" in data and "delivery_failures" not in data

    @pytest.mark.asyncio
    async def test_status_values(self, client):
        data = (await client.get("/api/reminders/status")).json()
        assert data["pending_count"] == 1
        assert data["next_trigger_at"] is not None
        assert data["scheduler_running"] is True
        assert data["notices"] == {"pending": 2, "pending_not_delivered": 1, "confirmed": 1, "uncertain": 1}
        assert data["output_zone_configured"] is True

    @pytest.mark.asyncio
    async def test_status_reports_a_missing_output_zone(self, client, mock_reminder_scheduler):
        mock_reminder_scheduler.output_zone = None
        data = (await client.get("/api/reminders/status")).json()
        assert data["output_zone_configured"] is False

    @pytest.mark.asyncio
    async def test_status_without_dependencies(self, tmp_path, dashboard_auth):
        """Without reminder manager/scheduler, returns safe defaults."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test", headers=dashboard_auth) as c:
            data = (await c.get("/api/reminders/status")).json()
        assert data["pending_count"] == 0
        assert data["next_trigger_at"] is None
        assert data["scheduler_running"] is False
        assert data["notices"] == {"pending": 0, "pending_not_delivered": 0, "confirmed": 0, "uncertain": 0}
        assert data["output_zone_configured"] is False
