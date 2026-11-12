"""Event-loop lag probe (U3b-0d2): the worst loop block over a run, stdlib only.

A task sleeps ``interval_s`` and measures how late it wakes up. The lateness
is the time the loop could not run callbacks, i.e. a blocking call on the
loop. Resolution: a block of B seconds reads as at least B - interval_s.

The run maximum and a per-window maximum (logged once per
``report_interval_s``) are exposed through ``snapshot()``, which the health
aggregator publishes under /api/metrics as ``loop_lag``. The future watchdog
unit sizes WatchdogSec from these figures; this probe never acts on them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import Context
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopLagConfig:
    """Probe settings (config/settings.yaml monitoring.loop_lag)."""

    enabled: bool = True
    interval_s: float = 0.1
    report_interval_s: float = 3600.0
    warn_ms: float = 200.0

    @classmethod
    def from_config(cls, config: dict | None) -> LoopLagConfig:
        """Build from the monitoring.loop_lag section, keeping defaults for missing keys."""
        section = config or {}
        value = cls(enabled=bool(section.get("enabled", cls.enabled)),
                    interval_s=float(section.get("interval_s", cls.interval_s)),
                    report_interval_s=float(section.get("report_interval_s", cls.report_interval_s)),
                    warn_ms=float(section.get("warn_ms", cls.warn_ms)))
        if not (0.01 <= value.interval_s <= 5 and value.report_interval_s >= value.interval_s
                and value.warn_ms > 0):
            raise ValueError("loop_lag_config_invalid")
        return value


class LoopLagProbe:
    """Measure the event loop's worst scheduling delay.

    Args:
        config: Probe settings.
        clock: Monotonic clock (injectable for tests).
    """

    def __init__(self, config: LoopLagConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or LoopLagConfig()
        self._clock = clock
        self._task: asyncio.Task | None = None
        self._samples = 0
        self._over_warn = 0
        self._max_s = 0.0
        self._window_max_s = 0.0
        self._window_over_warn = 0
        self._last_window_max_s: float | None = None
        self._window_started: float | None = None
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        """Whether the probe task is running."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start sampling on the running loop (idempotent)."""
        if not self.config.enabled or self.running:
            return
        self._started_at = self._window_started = self._clock()
        self._task = asyncio.create_task(self._sample(), name="loop-lag-probe", context=Context())

    async def stop(self) -> None:
        """Stop sampling and log the run maximum."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._samples:
            logger.info("[LoopLag] run max block %.0f ms over %d samples (%d >= %.0f ms)",
                        self._max_s * 1000, self._samples, self._over_warn, self.config.warn_ms)

    def record(self, lag_s: float) -> None:
        """Account one wake-up lateness (seconds)."""
        lag_s = max(0.0, lag_s)
        self._samples += 1
        self._max_s = max(self._max_s, lag_s)
        self._window_max_s = max(self._window_max_s, lag_s)
        if lag_s * 1000 >= self.config.warn_ms:
            self._over_warn += 1
            self._window_over_warn += 1
            # Rate-limited: the first block per window warns; the rest are counted
            # and reported once when the window closes (and in snapshot()).
            if self._window_over_warn == 1:
                logger.warning("[LoopLag] event loop blocked for about %.0f ms "
                               "(further blocks this window are counted, not logged)", lag_s * 1000)

    def _close_window(self, now: float) -> None:
        logger.info("[LoopLag] max block %.0f ms in the last %.0f s (%d >= %.0f ms; run max %.0f ms, "
                    "%d samples)", self._window_max_s * 1000, now - self._window_started,
                    self._window_over_warn, self.config.warn_ms, self._max_s * 1000, self._samples)
        self._last_window_max_s = self._window_max_s
        self._window_max_s = 0.0
        self._window_over_warn = 0
        self._window_started = now

    async def _sample(self) -> None:
        interval = self.config.interval_s
        while True:
            expected = self._clock() + interval
            await asyncio.sleep(interval)
            now = self._clock()
            self.record(now - expected)
            if now - self._window_started >= self.config.report_interval_s:
                self._close_window(now)

    def snapshot(self) -> dict:
        """Content-free figures for metrics and the watchdog sizing."""
        uptime = None if self._started_at is None else self._clock() - self._started_at
        return {
            "enabled": self.config.enabled,
            "running": self.running,
            "interval_ms": self.config.interval_s * 1000,
            "samples": self._samples,
            "max_ms": round(self._max_s * 1000, 1),
            "window_max_ms": round(self._window_max_s * 1000, 1),
            "last_window_max_ms": None if self._last_window_max_s is None
            else round(self._last_window_max_s * 1000, 1),
            "over_warn": self._over_warn,
            "window_over_warn": self._window_over_warn,
            "warn_ms": self.config.warn_ms,
            "uptime_s": None if uptime is None else round(uptime, 1),
        }
