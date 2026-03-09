"""Tests for HealthAggregator — subsystem health collection and failure tracking."""

import time
import pytest
from unittest.mock import MagicMock

from src.monitoring.health_aggregator import (
    HealthAggregator,
    OverallStatus,
    SubsystemHealth,
    SystemHealth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ha_client():
    """Mock HA client with get_health_status()."""
    client = MagicMock()
    health = MagicMock()
    health.state = "connected"
    health.error_count = 0
    health.success_count = 42
    health.avg_latency_ms = 25.5
    health.ws_connected = True
    health.last_error_message = ""
    client.get_health_status.return_value = health
    return client


@pytest.fixture
def mock_latency_monitor():
    """Mock latency monitor with get_stats()."""
    monitor = MagicMock()
    monitor.get_stats.return_value = {
        "session": {
            "total_commands": 100,
            "success_rate_percent": 95.0,
        },
        "total": {
            "p50_ms": 120.0,
            "p95_ms": 250.0,
            "p99_ms": 290.0,
        },
        "components": {},
    }
    return monitor


@pytest.fixture
def mock_priority_queue():
    """Mock priority queue with get_stats()."""
    queue = MagicMock()
    queue.get_stats.return_value = {
        "queue_size": 3,
        "max_queue_size": 100,
        "total_enqueued": 50,
        "total_processed": 47,
        "total_cancelled": 2,
        "total_timeout": 1,
        "current_processing": "abc123",
    }
    return queue


@pytest.fixture
def mock_reminder_scheduler():
    """Mock reminder scheduler."""
    scheduler = MagicMock()
    scheduler._running = True
    return scheduler


@pytest.fixture
def aggregator(mock_ha_client, mock_latency_monitor, mock_priority_queue, mock_reminder_scheduler):
    return HealthAggregator(
        ha_client=mock_ha_client,
        latency_monitor=mock_latency_monitor,
        priority_queue=mock_priority_queue,
        reminder_scheduler=mock_reminder_scheduler,
    )


# ---------------------------------------------------------------------------
# Tests — System health aggregation
# ---------------------------------------------------------------------------

class TestGetSystemHealth:
    """Tests for get_system_health()."""

    def test_all_healthy(self, aggregator):
        """When all subsystems are healthy, overall status is healthy."""
        result = aggregator.get_system_health()

        assert isinstance(result, SystemHealth)
        assert result.status == OverallStatus.HEALTHY
        assert len(result.subsystems) == 4
        assert result.timestamp > 0

    def test_ha_disconnected_makes_unhealthy(self, aggregator, mock_ha_client):
        """When HA client is disconnected, overall status is unhealthy."""
        health = mock_ha_client.get_health_status.return_value
        health.state = "disconnected"

        result = aggregator.get_system_health()
        assert result.status == OverallStatus.UNHEALTHY

        ha_sub = next(s for s in result.subsystems if s.name == "home_assistant")
        assert ha_sub.status == OverallStatus.UNHEALTHY

    def test_ha_auth_error_makes_unhealthy(self, aggregator, mock_ha_client):
        """HA auth_error maps to unhealthy."""
        health = mock_ha_client.get_health_status.return_value
        health.state = "auth_error"

        result = aggregator.get_system_health()
        assert result.status == OverallStatus.UNHEALTHY

    def test_low_success_rate_degrades(self, aggregator, mock_latency_monitor):
        """Success rate between 70-90% degrades the latency subsystem."""
        stats = mock_latency_monitor.get_stats.return_value
        stats["session"]["success_rate_percent"] = 75.0

        result = aggregator.get_system_health()
        lat_sub = next(s for s in result.subsystems if s.name == "latency_monitor")
        assert lat_sub.status == OverallStatus.DEGRADED
        assert result.status == OverallStatus.DEGRADED

    def test_very_low_success_rate_unhealthy(self, aggregator, mock_latency_monitor):
        """Success rate below 70% makes latency subsystem unhealthy."""
        stats = mock_latency_monitor.get_stats.return_value
        stats["session"]["success_rate_percent"] = 50.0

        result = aggregator.get_system_health()
        lat_sub = next(s for s in result.subsystems if s.name == "latency_monitor")
        assert lat_sub.status == OverallStatus.UNHEALTHY

    def test_queue_high_usage_degrades(self, aggregator, mock_priority_queue):
        """Queue between 50-80% full degrades the queue subsystem."""
        stats = mock_priority_queue.get_stats.return_value
        stats["queue_size"] = 60
        stats["max_queue_size"] = 100

        result = aggregator.get_system_health()
        q_sub = next(s for s in result.subsystems if s.name == "priority_queue")
        assert q_sub.status == OverallStatus.DEGRADED

    def test_queue_critical_usage_unhealthy(self, aggregator, mock_priority_queue):
        """Queue above 80% full is unhealthy."""
        stats = mock_priority_queue.get_stats.return_value
        stats["queue_size"] = 90
        stats["max_queue_size"] = 100

        result = aggregator.get_system_health()
        q_sub = next(s for s in result.subsystems if s.name == "priority_queue")
        assert q_sub.status == OverallStatus.UNHEALTHY

    def test_reminder_scheduler_stopped_degrades(self, aggregator, mock_reminder_scheduler):
        """Stopped reminder scheduler degrades."""
        mock_reminder_scheduler._running = False

        result = aggregator.get_system_health()
        r_sub = next(s for s in result.subsystems if s.name == "reminders")
        assert r_sub.status == OverallStatus.DEGRADED

    def test_no_subsystems_configured(self):
        """Aggregator with no subsystems returns degraded."""
        agg = HealthAggregator()
        result = agg.get_system_health()

        assert result.status == OverallStatus.UNHEALTHY
        assert len(result.subsystems) == 4

    def test_subsystem_extra_contains_details(self, aggregator):
        """Subsystem extra dicts contain expected keys."""
        result = aggregator.get_system_health()

        ha_sub = next(s for s in result.subsystems if s.name == "home_assistant")
        assert "error_count" in ha_sub.extra
        assert "ws_connected" in ha_sub.extra

        lat_sub = next(s for s in result.subsystems if s.name == "latency_monitor")
        assert "p50_ms" in lat_sub.extra
        assert "p95_ms" in lat_sub.extra
        assert "p99_ms" in lat_sub.extra

        q_sub = next(s for s in result.subsystems if s.name == "priority_queue")
        assert "queue_size" in q_sub.extra
        assert "total_enqueued" in q_sub.extra


# ---------------------------------------------------------------------------
# Tests — Metrics
# ---------------------------------------------------------------------------

class TestGetMetrics:

    def test_returns_latency_percentiles(self, aggregator):
        """Metrics include p50, p95, p99."""
        metrics = aggregator.get_metrics()

        assert metrics["latency"]["p50_ms"] == 120.0
        assert metrics["latency"]["p95_ms"] == 250.0
        assert metrics["latency"]["p99_ms"] == 290.0

    def test_returns_queue_depth(self, aggregator):
        """Metrics include queue depth from priority queue."""
        metrics = aggregator.get_metrics()
        assert metrics["queue_depth"] == 3

    def test_returns_command_count(self, aggregator):
        """Metrics include total command count."""
        metrics = aggregator.get_metrics()
        assert metrics["command_count"] == 100

    def test_no_subsystems_returns_zeros(self):
        """Without subsystems, metrics are zero-valued."""
        agg = HealthAggregator()
        metrics = agg.get_metrics()

        assert metrics["latency"]["p50_ms"] == 0.0
        assert metrics["queue_depth"] == 0
        assert metrics["command_count"] == 0


# ---------------------------------------------------------------------------
# Tests — Failure tracking
# ---------------------------------------------------------------------------

class TestFailures:

    def test_record_and_retrieve(self, aggregator):
        """Failures can be recorded and retrieved."""
        aggregator.record_failure("home_assistant", "Connection refused")

        failures = aggregator.get_recent_failures()
        assert len(failures) == 1
        assert failures[0]["subsystem"] == "home_assistant"
        assert failures[0]["message"] == "Connection refused"
        assert failures[0]["timestamp"] > 0

    def test_newest_first(self, aggregator):
        """Failures are returned newest first."""
        aggregator.record_failure("ha", "error1")
        aggregator.record_failure("tts", "error2")
        aggregator.record_failure("stt", "error3")

        failures = aggregator.get_recent_failures()
        assert failures[0]["subsystem"] == "stt"
        assert failures[2]["subsystem"] == "ha"

    def test_max_failures_cap(self):
        """Only MAX_FAILURES records are kept."""
        agg = HealthAggregator()
        for i in range(60):
            agg.record_failure("test", f"error_{i}")

        failures = agg.get_recent_failures(limit=100)
        assert len(failures) == 50  # MAX_FAILURES

    def test_limit_parameter(self, aggregator):
        """Limit parameter restricts returned count."""
        for i in range(10):
            aggregator.record_failure("test", f"error_{i}")

        failures = aggregator.get_recent_failures(limit=3)
        assert len(failures) == 3

    def test_failure_includes_detail(self, aggregator):
        """Detail field is included in failure records."""
        aggregator.record_failure("ha", "timeout", detail="after 5s")

        failures = aggregator.get_recent_failures()
        assert failures[0]["detail"] == "after 5s"

    def test_empty_failures(self, aggregator):
        """No failures returns empty list."""
        assert aggregator.get_recent_failures() == []
