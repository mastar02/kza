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
from collections import deque
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
        probe_interval_s: While degraded (and a fallback exists), commands
            skip the primary and use the shadow; one command per interval
            (and the first after a successful reload) still tries the primary
            so the episode can end.
        window_size: Recent primary attempts considered for intermittent
            failures.
        window_failures: Failures within the window that raise one
            "intermittent" alert even without ``failure_threshold`` in a row
            (the fallback would otherwise hide them). It clears once the
            window holds no failure.
    """

    failure_threshold: int = 3
    fallback_to_shadow: bool = True
    fallback_timeout_s: float = 2.0
    reload_enabled: bool = True
    reload_min_interval_s: float = 600.0
    alert_priority: str = "MEDIUM"
    probe_interval_s: float = 30.0
    window_size: int = 12
    window_failures: int = 4

    def __post_init__(self) -> None:
        if type(self.failure_threshold) is not int or self.failure_threshold < 1:
            raise ValueError("invalid primary_health.failure_threshold")
        if (type(self.window_size) is not int or type(self.window_failures) is not int
                or not 1 <= self.window_failures <= self.window_size):
            raise ValueError("invalid primary_health window")
        for name in ("fallback_timeout_s", "reload_min_interval_s", "probe_interval_s"):
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
            probe_interval_s=float(cfg.get("probe_interval_s", 30.0)),
            window_size=int(cfg.get("window_size", 12)),
            window_failures=int(cfg.get("window_failures", 4)),
        )


@dataclass(frozen=True)
class HealthActions:
    """What the caller should start after a failure."""

    alert: bool = False
    reload: bool = False
    intermittent: bool = False


_ALERT_START = "El reconocimiento de voz principal falló varias veces seguidas."
_ALERT_STATE = {
    (True, True): "Mientras tanto uso el reconocimiento de respaldo y estoy recargando el modelo.",
    (True, False): ("Mientras tanto uso el reconocimiento de respaldo; "
                    "la recarga automática no está disponible ahora."),
    (False, True): "Estoy recargando el modelo; mientras tanto puede que no entienda los comandos de voz.",
    (False, False): ("No hay respaldo ni recarga automática: puede que no entienda los "
                     "comandos de voz hasta que alguien lo revise."),
}


def alert_message(*, fallback: bool, reload: bool) -> str:
    """Spanish episode alert text that states only what will actually happen.

    Args:
        fallback: The shadow transcript will stand in for the primary.
        reload: A reload of the primary model was started now.

    Returns:
        Content-free alert text (no codes, no transcript).
    """
    return f"{_ALERT_START} {_ALERT_STATE[(bool(fallback), bool(reload))]}"


ALERT_MESSAGE = alert_message(fallback=True, reload=True)
RECOVERY_MESSAGE = "El reconocimiento de voz principal volvió a funcionar."
_INTERMITTENT_START = "El reconocimiento de voz principal está fallando de forma intermitente."


def intermittent_message(*, fallback: bool) -> str:
    """Spanish alert text for intermittent primary failures (content-free)."""
    tail = ("Cuando falla, uso el reconocimiento de respaldo." if fallback
            else "Cuando falla, puede que no entienda el comando.")
    return f"{_INTERMITTENT_START} {tail}"


@dataclass
class _Counters:
    failures: int = 0
    successes: int = 0
    fallbacks_used: int = 0
    fallbacks_missing: int = 0
    episodes: int = 0
    reloads_ok: int = 0
    reloads_failed: int = 0
    primary_skipped: int = 0
    primary_not_run: int = 0
    intermittent_episodes: int = 0
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
        self._last_probe_at: float | None = None
        self._probe_after_reload = False
        self._window: deque[bool] = deque(maxlen=self.config.window_size)
        self._intermittent = False
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
            self._window.append(True)
            intermittent = False
            if (self._state is SttHealthState.HEALTHY and not self._intermittent
                    and sum(self._window) >= self.config.window_failures
                    and self._consecutive < self.config.failure_threshold):
                self._intermittent = True
                self._counters.intermittent_episodes += 1
                intermittent = True
            alert = False
            if self._consecutive >= self.config.failure_threshold and self._state is SttHealthState.HEALTHY:
                self._state = SttHealthState.DEGRADED
                self._counters.episodes += 1
                self._last_probe_at = self._clock()
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
        if intermittent:
            logger.warning("[stt-health] code=primary_intermittent window_failures=%d window=%d",
                           self.config.window_failures, self.config.window_size)
        return HealthActions(alert=alert, reload=reload, intermittent=intermittent)

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
            self._window.append(False)
            if self._intermittent and not any(self._window):
                self._intermittent = False
        if recovered:
            logger.info("[stt-health] code=primary_recovered after_failures=%d", failures)
        return recovered

    def try_primary(self) -> bool:
        """Whether this command should call the primary (fallback available).

        False while a reload runs (it holds the model lock for seconds) and
        while degraded, except for one probe per ``probe_interval_s`` and the
        first command after a successful reload.

        Returns:
            True to call the primary, False to go straight to the fallback.
        """
        with self._lock:
            if self._reloading:
                allowed = False
            elif self._state is SttHealthState.HEALTHY:
                return True
            elif self._probe_after_reload:
                self._probe_after_reload = False
                self._last_probe_at = self._clock()
                allowed = True
            elif (self._last_probe_at is None
                  or self._clock() - self._last_probe_at >= self.config.probe_interval_s):
                self._last_probe_at = self._clock()
                allowed = True
            else:
                allowed = False
            if not allowed:
                self._counters.primary_skipped += 1
            return allowed

    def record_not_run(self, code: str) -> None:
        """Count a primary call that never reached the model (capacity, closed).

        Not a model failure: it neither extends nor starts an episode.
        """
        with self._lock:
            self._counters.primary_not_run += 1
        logger.warning("[stt-health] code=primary_not_run error=%s", code)

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
                self._probe_after_reload = True
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
                "primary_skipped": c.primary_skipped, "primary_not_run": c.primary_not_run,
                "intermittent": self._intermittent, "intermittent_episodes": c.intermittent_episodes,
                "window_failures": sum(self._window), "window_size": len(self._window),
            }

    async def send_alert(self, recovered: bool = False, *, fallback: bool = True,
                         reload: bool = True, intermittent: bool = False) -> bool:
        """Raise the episode (or recovery) alert through the AlertManager.

        Args:
            recovered: Send the LOW recovery alert instead of the episode alert.
            fallback: The shadow transcript stands in while the primary fails.
            reload: A reload of the primary was started with this episode.
            intermittent: Send the intermittent-failure alert (windowed rate).

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
        code = "primary_recovered" if recovered else "primary_intermittent" if intermittent else "primary_degraded"
        details = {"source": "stt_primary", "code": code,
                   "last_error": snapshot["last_code"], "consecutive_failures": snapshot["consecutive_failures"]}
        if intermittent and not recovered:
            details.update(fallback=bool(fallback), window_failures=snapshot["window_failures"],
                           window_size=snapshot["window_size"])
            message = intermittent_message(fallback=fallback)
        elif not recovered:
            details.update(fallback=bool(fallback), reload_started=bool(reload))
            message = alert_message(fallback=fallback, reload=reload)
        else:
            message = RECOVERY_MESSAGE
        try:
            alert = await manager.create_alert(AlertType.DEVICE, priority, message, details)
        except Exception as error:
            logger.warning("[stt-health] code=alert_failed error=%s", type(error).__name__)
            return False
        return alert is not None
