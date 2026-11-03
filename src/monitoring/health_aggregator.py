"""Health Aggregator — collects subsystem health into a unified view.

Queries HA client, latency monitor, priority queue, and reminder
scheduler to produce an overall system health status.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from src.home_assistant.ha_client import (
    HAChannelObservation,
    HAConnectionState,
    HAHealthStatus,
)

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
        audio_health_provider=None,
        health_clock=None,
    ):
        self._health_clock = health_clock or time.time
        self._audio_health_provider = audio_health_provider
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

        try:
            health = self._ha_client.get_health_status()
        except Exception as error:
            return SubsystemHealth(name="home_assistant", status=OverallStatus.UNHEALTHY,
                detail=f"HA health unavailable: {type(error).__name__}")

        channels = {name: getattr(health, name, None)
                    for name in ("rest", "calls", "events", "registry")}
        native = any(type(value) is HAChannelObservation for value in channels.values())
        reported = type(health) is HAHealthStatus and any(value is not None for value in channels.values())
        if native or reported:
            return self._check_ha_channels(health, channels)

        # Standalone integrations and older test doubles may report only the
        # original metrics DTO. ws_connected continues to mean calls, never
        # event-subscription readiness.
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

    @staticmethod
    def _ha_channel(value, name: str, now: float) -> dict:
        sources = {"rest": "ha_rest", "calls": "ha_ws_calls",
                   "events": "ha_ws_events", "registry": "ha_ws_registry"}
        if (type(value) is not HAChannelObservation or type(value.state) is not HAConnectionState
                or value.source != sources[name] or type(value.phase) is not str
                or type(value.required) is not bool or type(value.generation) is not int
                or value.generation < 0 or value.reason is not None and type(value.reason) is not str):
            raise ValueError("invalid_ha_channel_observation")
        for instant in (value.observed_at, value.expires_at, value.attempted_at):
            if instant is not None and (type(instant) not in (int, float)
                    or not math.isfinite(instant) or instant < 0):
                raise ValueError("invalid_ha_channel_observation")
        if ((value.observed_at is None) != (value.expires_at is None)
                or value.observed_at is not None and value.expires_at <= value.observed_at):
            raise ValueError("invalid_ha_channel_observation")
        current = value.observed_at is not None and value.observed_at <= now < value.expires_at
        state, phase = str(value.state), value.phase
        if value.phase in {"closed", "stopped"}:
            pass
        elif value.expires_at is not None and now >= value.expires_at:
            state, phase = "unknown", "expired"
        elif value.observed_at is not None and now < value.observed_at:
            state, phase = "unknown", "future"
        elif state == "connected" and not current:
            state = "unknown"
        return {"state": state, "source": value.source, "phase": phase,
            "observed_at": value.observed_at, "expires_at": value.expires_at,
            "reason": value.reason, "attempted_at": value.attempted_at,
            "required": value.required, "generation": value.generation, "current": current}

    def _check_ha_channels(self, health, channels) -> SubsystemHealth:
        try:
            now = self._health_clock()
            if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
                raise ValueError("invalid_ha_health_clock")
            projected = {name: self._ha_channel(value, name, now)
                         for name, value in channels.items() if value is not None}
        except (ValueError, TypeError, AttributeError):
            return SubsystemHealth(name="home_assistant", status=OverallStatus.UNHEALTHY,
                detail="HA transport observation unavailable")

        rest = projected.get("rest")
        unavailable = [name for name, value in projected.items()
                       if value["required"] and (value["state"] != "connected" or not value["current"])]
        if rest is None:
            unavailable.insert(0, "rest")
        if rest is not None and rest["state"] in {"auth_error", "disconnected"}:
            status = OverallStatus.UNHEALTHY
        elif rest is None or not rest["current"] or rest["state"] != "connected" or unavailable:
            status = OverallStatus.DEGRADED
        else:
            status = OverallStatus.HEALTHY
        detail = "; ".join(f"{name}:{value['state']} ({value['phase']})"
                           for name, value in projected.items()) or "rest:unknown"
        if rest is None and projected:
            detail = "rest:unknown; " + detail
        return SubsystemHealth(name="home_assistant", status=status, detail=detail,
            extra={"error_count": health.error_count, "success_count": health.success_count,
                "avg_latency_ms": health.avg_latency_ms, "ws_connected": health.ws_connected,
                "last_error": health.last_error_message, "channels": projected})

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
        if session.get("total_commands", 0) == 0:
            return SubsystemHealth(
                name="latency_monitor", status=OverallStatus.DEGRADED,
                detail="No command samples; latency is unmeasured",
                extra={"total_commands": 0},
            )
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

    def _check_audio(self) -> SubsystemHealth:
        """Expose missing and stale room capture using in-memory metadata."""
        try:
            snapshot = self._audio_health_provider()
            rooms = snapshot.get("rooms", {})
            threshold = snapshot.get("deaf_after_s", 8.0)
            grace = snapshot.get("first_frame_grace_s", 180.0)
            deaf = [room for room, info in rooms.items() if info.get("age_s", 0) > (threshold if info.get("ever") else grace)]
            pending = [room for room, info in rooms.items() if not info.get("ever")]
            if not rooms or deaf:
                status = OverallStatus.UNHEALTHY
                detail = "No audio frames: " + ", ".join(deaf or ["no rooms configured"])
            elif pending:
                status = OverallStatus.DEGRADED
                detail = "Waiting for first audio frames: " + ", ".join(pending)
            else:
                status = OverallStatus.HEALTHY
                detail = "Audio frames received in every configured room"
            return SubsystemHealth(name="audio_capture", status=status, detail=detail, extra={"rooms": rooms})
        except Exception as exc:
            return SubsystemHealth(name="audio_capture", status=OverallStatus.UNHEALTHY, detail=f"Audio health unavailable: {type(exc).__name__}")

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

        if self._audio_health_provider is not None:
            subsystems.append(self._check_audio())

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
