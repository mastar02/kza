"""Tests for observability API endpoints: health, metrics, subsystems, failures, reminders/status."""

import time
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

from src.dashboard.api import DashboardAPI
from src.monitoring.health_aggregator import HealthAggregator
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager


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
    scheduler = MagicMock()
    scheduler._running = True
    scheduler._retry_counts = {"r1": 1, "r2": 2}
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
    )

    dashboard = DashboardAPI(
        health_aggregator=health_aggregator,
        reminder_manager=reminder_mgr,
        reminder_scheduler=mock_reminder_scheduler,
    )
    yield dashboard
    await reminder_store.close()


@pytest_asyncio.fixture
async def client(api_with_observability):
    transport = ASGITransport(app=api_with_observability.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
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
    async def test_health_without_aggregator(self, tmp_path):
        """When health_aggregator is None, returns fallback."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
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
    async def test_metrics_without_aggregator(self, tmp_path):
        """Without aggregator, returns zero-valued metrics."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
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
    async def test_subsystems_without_aggregator(self, tmp_path):
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
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
    async def test_failures_without_aggregator(self, tmp_path):
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
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
        assert "delivery_failures" in data

    @pytest.mark.asyncio
    async def test_status_values(self, client):
        data = (await client.get("/api/reminders/status")).json()
        assert data["pending_count"] == 1
        assert data["next_trigger_at"] is not None
        assert data["scheduler_running"] is True
        assert data["delivery_failures"] == 3  # 1 + 2 from mock retry_counts

    @pytest.mark.asyncio
    async def test_status_without_dependencies(self, tmp_path):
        """Without reminder manager/scheduler, returns safe defaults."""
        dashboard = DashboardAPI()
        transport = ASGITransport(app=dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            data = (await c.get("/api/reminders/status")).json()
        assert data["pending_count"] == 0
        assert data["next_trigger_at"] is None
        assert data["scheduler_running"] is False
        assert data["delivery_failures"] == 0
