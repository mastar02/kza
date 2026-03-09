"""Health Aggregator — collects subsystem health into a unified view.

Queries HA client, latency monitor, priority queue, and reminder
scheduler to produce an overall system health status.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class OverallStatus(StrEnum):
    """Aggregate health status for the system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class SubsystemHealth:
    """Health snapshot for a single subsystem."""
    name: str
    status: OverallStatus = OverallStatus.HEALTHY
    detail: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class FailureRecord:
    """A recorded failure event."""
    timestamp: float
    subsystem: str
    message: str
    detail: str = ""


@dataclass
class SystemHealth:
    """Full system health report."""
    status: OverallStatus
    timestamp: float
    subsystems: list[SubsystemHealth]
    failures: list[dict] = field(default_factory=list)


class HealthAggregator:
    """Aggregates health from multiple subsystems into a single report.

    Collects status from: HA client, latency monitor, priority queue,
    and reminder scheduler.  Stores recent failure records for the
    /api/failures endpoint.
    """

    MAX_FAILURES = 50

    def __init__(
        self,
        ha_client=None,
        latency_monitor=None,
        priority_queue=None,
        reminder_scheduler=None,
    ):
        self._ha_client = ha_client
        self._latency_monitor = latency_monitor
        self._priority_queue = priority_queue
        self._reminder_scheduler = reminder_scheduler
        self._failures: deque[FailureRecord] = deque(maxlen=self.MAX_FAILURES)

    # ------------------------------------------------------------------
    # Failure tracking
    # ------------------------------------------------------------------

    def record_failure(self, subsystem: str, message: str, detail: str = "") -> None:
        """Record a failure event.

        Args:
            subsystem: Name of the subsystem that failed.
            message: Short human-readable description.
            detail: Optional extended information.
        """
        record = FailureRecord(
            timestamp=time.time(),
            subsystem=subsystem,
            message=message,
            detail=detail,
        )
        self._failures.append(record)
        logger.warning("Failure recorded [%s]: %s", subsystem, message)

    def get_recent_failures(self, limit: int = 50) -> list[dict]:
        """Return recent failures as dicts, newest first.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of failure dicts.
        """
        records = list(self._failures)[-limit:]
        records.reverse()
        return [
            {
                "timestamp": r.timestamp,
                "subsystem": r.subsystem,
                "message": r.message,
                "detail": r.detail,
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Subsystem health checks
    # ------------------------------------------------------------------

    def _check_ha(self) -> SubsystemHealth:
        """Check Home Assistant client health."""
        if self._ha_client is None:
            return SubsystemHealth(
                name="home_assistant",
                status=OverallStatus.UNHEALTHY,
                detail="HA client not configured",
            )

        health = self._ha_client.get_health_status()
        state_str = str(health.state)

        if state_str == "connected":
            status = OverallStatus.HEALTHY
        elif state_str == "auth_error":
            status = OverallStatus.UNHEALTHY
        elif state_str == "disconnected":
            status = OverallStatus.UNHEALTHY
        else:
            status = OverallStatus.DEGRADED

        return SubsystemHealth(
            name="home_assistant",
            status=status,
            detail=state_str,
            extra={
                "error_count": health.error_count,
                "success_count": health.success_count,
                "avg_latency_ms": health.avg_latency_ms,
                "ws_connected": health.ws_connected,
                "last_error": health.last_error_message,
            },
        )

    def _check_latency(self) -> SubsystemHealth:
        """Check latency monitor health."""
        if self._latency_monitor is None:
            return SubsystemHealth(
                name="latency_monitor",
                status=OverallStatus.DEGRADED,
                detail="Latency monitor not configured",
            )

        stats = self._latency_monitor.get_stats()
        session = stats.get("session", {})
        success_rate = session.get("success_rate_percent", 100.0)

        if success_rate >= 90:
            status = OverallStatus.HEALTHY
        elif success_rate >= 70:
            status = OverallStatus.DEGRADED
        else:
            status = OverallStatus.UNHEALTHY

        return SubsystemHealth(
            name="latency_monitor",
            status=status,
            detail=f"{success_rate}% under target",
            extra={
                "total_commands": session.get("total_commands", 0),
                "success_rate_percent": success_rate,
                "p50_ms": stats.get("total", {}).get("p50_ms", 0),
                "p95_ms": stats.get("total", {}).get("p95_ms", 0),
                "p99_ms": stats.get("total", {}).get("p99_ms", 0),
            },
        )

    def _check_queue(self) -> SubsystemHealth:
        """Check priority queue health."""
        if self._priority_queue is None:
            return SubsystemHealth(
                name="priority_queue",
                status=OverallStatus.DEGRADED,
                detail="Priority queue not configured",
            )

        stats = self._priority_queue.get_stats()
        queue_size = stats.get("queue_size", 0)
        max_size = stats.get("max_queue_size", 100)
        usage_pct = (queue_size / max_size * 100) if max_size > 0 else 0

        if usage_pct < 50:
            status = OverallStatus.HEALTHY
        elif usage_pct < 80:
            status = OverallStatus.DEGRADED
        else:
            status = OverallStatus.UNHEALTHY

        return SubsystemHealth(
            name="priority_queue",
            status=status,
            detail=f"{queue_size}/{max_size} ({usage_pct:.0f}% full)",
            extra={
                "queue_size": queue_size,
                "max_queue_size": max_size,
                "total_enqueued": stats.get("total_enqueued", 0),
                "total_processed": stats.get("total_processed", 0),
                "total_cancelled": stats.get("total_cancelled", 0),
                "total_timeout": stats.get("total_timeout", 0),
                "current_processing": stats.get("current_processing"),
            },
        )

    def _check_reminders(self) -> SubsystemHealth:
        """Check reminder scheduler health."""
        if self._reminder_scheduler is None:
            return SubsystemHealth(
                name="reminders",
                status=OverallStatus.DEGRADED,
                detail="Reminder scheduler not configured",
            )

        running = getattr(self._reminder_scheduler, "_running", False)

        if running:
            status = OverallStatus.HEALTHY
            detail = "running"
        else:
            status = OverallStatus.DEGRADED
            detail = "stopped"

        return SubsystemHealth(
            name="reminders",
            status=status,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def get_system_health(self) -> SystemHealth:
        """Collect health from all subsystems and compute overall status.

        Returns:
            SystemHealth with per-subsystem detail and overall status.
        """
        subsystems = [
            self._check_ha(),
            self._check_latency(),
            self._check_queue(),
            self._check_reminders(),
        ]

        # Overall status: worst of all subsystems
        statuses = [s.status for s in subsystems]
        if OverallStatus.UNHEALTHY in statuses:
            overall = OverallStatus.UNHEALTHY
        elif OverallStatus.DEGRADED in statuses:
            overall = OverallStatus.DEGRADED
        else:
            overall = OverallStatus.HEALTHY

        return SystemHealth(
            status=overall,
            timestamp=time.time(),
            subsystems=subsystems,
        )

    def get_metrics(self) -> dict:
        """Return key operational metrics for the /api/metrics endpoint.

        Returns:
            Dict with latency percentiles, queue depth, and command count.
        """
        metrics: dict = {
            "latency": {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            },
            "queue_depth": 0,
            "command_count": 0,
            "active_zones": 0,
        }

        if self._latency_monitor is not None:
            stats = self._latency_monitor.get_stats()
            total = stats.get("total", {})
            session = stats.get("session", {})
            metrics["latency"] = {
                "p50_ms": total.get("p50_ms", 0.0),
                "p95_ms": total.get("p95_ms", 0.0),
                "p99_ms": total.get("p99_ms", 0.0),
            }
            metrics["command_count"] = session.get("total_commands", 0)

        if self._priority_queue is not None:
            q_stats = self._priority_queue.get_stats()
            metrics["queue_depth"] = q_stats.get("queue_size", 0)

        return metrics
