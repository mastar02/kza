"""Primary STT health: sanitized failure codes, alert, reload and fallback (V3).

On 2026-11-16 the primary Whisper (cuda:1) failed 1293 of 1296 captures with a
RuntimeError and nothing told anyone. This tracker counts consecutive primary
failures by a stable, text-free code; after ``failure_threshold`` it asks for
one alert per episode and for a rate-limited, guarded reload of the model.
The command processor uses the shadow STT transcript while the primary fails.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

logger = logging.getLogger(__name__)

# Keyword → stable category. Only the category and a short digest of the
# digit-free message reach logs; the message itself never does.
_CATEGORIES = (
    # Fixed messages of ModelReloadError (whisper_fast), most specific first.
    ("unload_failed", "unload_failed"),
    ("load_failed", "load_failed"),
    ("model_unloaded", "model_unloaded"),
    ("out of memory", "oom"),
    ("cublas", "cublas"),
    ("cudnn", "cudnn"),
    ("cuda", "cuda"),
    ("closed", "closed"),
    ("not loaded", "not_loaded"),
    ("timeout", "timeout"),
)


def stt_error_code(error: BaseException) -> str:
    """Sanitized, stable code for an STT failure (no message text).

    Args:
        error: The raised exception.

    Returns:
        ``<ExceptionType>:<category>:<6 hex>``, where the digest covers the
        lowercase message with digits and hex addresses removed, so the same
        failure keeps the same code across captures.
    """
    name = type(error).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        name = "Exception"
    try:
        message = str(error)[:500].lower()
    except Exception:
        message = ""
    category = next((code for key, code in _CATEGORIES if key in message), "other")
    stable = re.sub(r"0x[0-9a-f]+|\d+", "#", message)
    digest = hashlib.sha256(stable.encode("utf-8", "replace")).hexdigest()[:6]
    return f"{name}:{category}:{digest}"


class SttHealthState(StrEnum):
    """Primary STT condition."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class PrimarySttHealthConfig:
    """``stt.primary_health`` settings.

    Attributes:
        failure_threshold: Consecutive primary failures that start an episode
            (alert + reload).
        fallback_to_shadow: Use the shadow transcript when the primary fails.
        fallback_timeout_s: Wait bound for the shadow transcript.
        reload_enabled: Reload the primary model during an episode.
        reload_min_interval_s: Minimum time between reload attempts.
        alert_priority: AlertPriority name for the episode alert.
    """

    failure_threshold: int = 3
    fallback_to_shadow: bool = True
    fallback_timeout_s: float = 2.0
    reload_enabled: bool = True
    reload_min_interval_s: float = 600.0
    alert_priority: str = "MEDIUM"

    def __post_init__(self) -> None:
        if type(self.failure_threshold) is not int or self.failure_threshold < 1:
            raise ValueError("invalid primary_health.failure_threshold")
        for name in ("fallback_timeout_s", "reload_min_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid primary_health.{name}")
        if str(self.alert_priority).upper() not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            raise ValueError("invalid primary_health.alert_priority")

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "PrimarySttHealthConfig":
        """Build from ``stt.primary_health`` (missing keys keep defaults)."""
        cfg = dict(cfg or {})
        return cls(
            failure_threshold=int(cfg.get("failure_threshold", 3)),
            fallback_to_shadow=bool(cfg.get("fallback_to_shadow", True)),
            fallback_timeout_s=float(cfg.get("fallback_timeout_s", 2.0)),
            reload_enabled=bool(cfg.get("reload_enabled", True)),
            reload_min_interval_s=float(cfg.get("reload_min_interval_s", 600.0)),
            alert_priority=str(cfg.get("alert_priority", "MEDIUM")).upper(),
        )


@dataclass(frozen=True)
class HealthActions:
    """What the caller should start after a failure."""

    alert: bool = False
    reload: bool = False


ALERT_MESSAGE = ("El reconocimiento de voz principal falló varias veces seguidas. "
                 "Mientras tanto uso el reconocimiento de respaldo y estoy recargando el modelo.")
RECOVERY_MESSAGE = "El reconocimiento de voz principal volvió a funcionar."


@dataclass
class _Counters:
    failures: int = 0
    successes: int = 0
    fallbacks_used: int = 0
    fallbacks_missing: int = 0
    episodes: int = 0
    reloads_ok: int = 0
    reloads_failed: int = 0
    codes: dict[str, int] = field(default_factory=dict)


class PrimarySttHealth:
    """Thread-safe consecutive-failure tracker of the primary STT."""

    def __init__(self, config: PrimarySttHealthConfig | None = None, *, alert_manager=None,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config or PrimarySttHealthConfig()
        self._alert_manager = alert_manager
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive = 0
        self._state = SttHealthState.HEALTHY
        self._last_code: str | None = None
        self._last_reload_at: float | None = None
        self._reloading = False
        self._counters = _Counters()

    def attach_alert_manager(self, alert_manager) -> None:
        """Inject the AlertManager after DI (built after the command processor)."""
        self._alert_manager = alert_manager

    @property
    def state(self) -> SttHealthState:
        """Current state."""
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Primary failures since the last success."""
        return self._consecutive

    def record_failure(self, code: str) -> HealthActions:
        """Count one primary failure.

        Args:
            code: Sanitized code from ``stt_error_code``.

        Returns:
            Whether to raise the episode alert and/or start a reload now.
        """
        with self._lock:
            self._consecutive += 1
            self._last_code = code
            self._counters.failures += 1
            self._counters.codes[code] = self._counters.codes.get(code, 0) + 1
            alert = False
            if self._consecutive >= self.config.failure_threshold and self._state is SttHealthState.HEALTHY:
                self._state = SttHealthState.DEGRADED
                self._counters.episodes += 1
                alert = True
            reload = (self._state is SttHealthState.DEGRADED and self.config.reload_enabled
                      and not self._reloading
                      and (self._last_reload_at is None
                           or self._clock() - self._last_reload_at >= self.config.reload_min_interval_s))
            if reload:
                self._reloading = True
                self._last_reload_at = self._clock()
            consecutive = self._consecutive
        logger.warning("[stt-health] code=primary_failure error=%s consecutive=%d state=%s",
                       code, consecutive, self._state.value)
        return HealthActions(alert=alert, reload=reload)

    def record_success(self) -> bool:
        """Count one primary success.

        Returns:
            True when this success ends a degraded episode.
        """
        with self._lock:
            self._counters.successes += 1
            recovered = self._state is SttHealthState.DEGRADED
            failures = self._consecutive
            self._consecutive = 0
            self._state = SttHealthState.HEALTHY
        if recovered:
            logger.info("[stt-health] code=primary_recovered after_failures=%d", failures)
        return recovered

    def record_fallback(self, used: bool) -> None:
        """Count whether the shadow transcript replaced a failed primary."""
        with self._lock:
            if used:
                self._counters.fallbacks_used += 1
            else:
                self._counters.fallbacks_missing += 1

    def reload_finished(self, ok: bool, code: str | None = None) -> None:
        """Record the end of a reload attempt (rate limit already applied)."""
        with self._lock:
            self._reloading = False
            if ok:
                self._counters.reloads_ok += 1
            else:
                self._counters.reloads_failed += 1
        if ok:
            logger.info("[stt-health] code=primary_reloaded")
        else:
            logger.warning("[stt-health] code=primary_reload_failed error=%s", code or "unknown")

    def snapshot(self) -> dict:
        """Codes and counts only (no text)."""
        with self._lock:
            c = self._counters
            return {
                "state": self._state.value, "consecutive_failures": self._consecutive,
                "last_code": self._last_code, "failures": c.failures, "successes": c.successes,
                "fallbacks_used": c.fallbacks_used, "fallbacks_missing": c.fallbacks_missing,
                "episodes": c.episodes, "reloads_ok": c.reloads_ok, "reloads_failed": c.reloads_failed,
                "codes": dict(c.codes), "reloading": self._reloading,
            }

    async def send_alert(self, recovered: bool = False) -> bool:
        """Raise the episode (or recovery) alert through the AlertManager.

        Returns:
            True when an alert was created.
        """
        manager = self._alert_manager
        if manager is None:
            logger.warning("[stt-health] code=alert_unavailable")
            return False
        from src.alerts.alert_manager import AlertPriority, AlertType

        snapshot = self.snapshot()
        priority = AlertPriority.LOW if recovered else AlertPriority[self.config.alert_priority]
        details = {"source": "stt_primary", "code": "primary_recovered" if recovered else "primary_degraded",
                   "last_error": snapshot["last_code"], "consecutive_failures": snapshot["consecutive_failures"]}
        try:
            alert = await manager.create_alert(AlertType.DEVICE, priority,
                                               RECOVERY_MESSAGE if recovered else ALERT_MESSAGE, details)
        except Exception as error:
            logger.warning("[stt-health] code=alert_failed error=%s", type(error).__name__)
            return False
        return alert is not None
