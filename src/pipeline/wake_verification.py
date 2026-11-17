"""Second-stage wake verification by transcript (V2, voice-diag 2026-11-17).

An acoustic (openwakeword) wake is confirmed a second time when the command
transcript, which includes the pre-roll with the wake word, contains one of the
configured wake word variants. The rule is the gate's existing missing_wake
rule (``contains_wake_word``), disabled for acoustic engines since 2026-06.

``shadow`` (default) only records what would have been rejected, so the
household can review seven days of counts and a labelled session before
``enforce`` is considered. Records hold codes, room ids, scores and dates;
never transcript text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable

from src.nlu.command_gate import contains_wake_word, normalize_wake_words

logger = logging.getLogger(__name__)

STATUS_SCHEMA = "wake-verification/v1"


class WakeVerificationMode(StrEnum):
    """How the second stage acts on a missing wake word."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class WakeCheckCode(StrEnum):
    """Outcome code of one verification."""

    PRESENT = "wake_present"
    WOULD_REJECT = "would_reject"
    REJECTED = "rejected_missing_wake"


@dataclass(frozen=True)
class WakeVerificationConfig:
    """``rooms.wake_word.verification`` settings.

    Attributes:
        mode: off / shadow / enforce.
        words: Wake word variants (``rooms.wake_word.words``).
        status_path: JSON file with the review counters; None keeps them in memory.
        max_events: Bound of the per-event review list.
        keep_days: Age limit, in days, of both the per-day counters and the
            per-event records (also bounded by ``max_events``).
        flush_interval_s: Minimum time between status writes.
    """

    mode: WakeVerificationMode = WakeVerificationMode.SHADOW
    words: tuple[str, ...] = ("nexa",)
    status_path: str | None = "./data/wake_verification.json"
    max_events: int = 2000
    keep_days: int = 14
    flush_interval_s: float = 60.0

    @classmethod
    def from_dict(cls, cfg: dict | None, words: Iterable[str]) -> "WakeVerificationConfig":
        """Build from config; an unknown mode raises ValueError.

        Args:
            cfg: ``rooms.wake_word.verification`` mapping or None.
            words: The configured wake word variants.

        Returns:
            The validated config.
        """
        cfg = dict(cfg or {})
        mode = WakeVerificationMode(str(cfg.get("mode", "shadow")).lower())
        normalized = normalize_wake_words(words)
        if not normalized:
            raise ValueError("wake verification needs wake words")
        max_events = int(cfg.get("max_events", 2000))
        keep_days = int(cfg.get("keep_days", 14))
        flush = float(cfg.get("flush_interval_s", 60.0))
        if max_events < 1 or keep_days < 1 or not math.isfinite(flush) or flush < 0:
            raise ValueError("invalid wake verification bounds")
        return cls(mode=mode, words=normalized, status_path=cfg.get("status_path", "./data/wake_verification.json"),
                   max_events=max_events, keep_days=keep_days, flush_interval_s=flush)


@dataclass(frozen=True)
class WakeCheck:
    """Result of one verification.

    Attributes:
        code: Outcome code.
        reject: True only in enforce mode with no wake word.
        reason: Gate-style reason for an enforced rejection.
    """

    code: WakeCheckCode
    reject: bool = False
    reason: str = ""


def _event_current(event: dict, cutoff: float) -> bool:
    """An event with a finite timestamp no older than ``cutoff``."""
    ts = event.get("ts") if isinstance(event, dict) else None
    return type(ts) in (int, float) and math.isfinite(ts) and ts >= cutoff


def _fmt(value: float | None) -> str:
    return "na" if value is None or not math.isfinite(value) else f"{value:.2f}"


class WakeVerifier:
    """Counts and (optionally) enforces the transcript wake-word check."""

    def __init__(self, config: WakeVerificationConfig, clock: Callable[[], float] = time.time):
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._days: dict[str, dict[str, dict[str, int]]] = {}
        self._events: deque[dict] = deque(maxlen=config.max_events)
        self._last_flush = 0.0
        self._flush_task = None

    @property
    def mode(self) -> WakeVerificationMode:
        """Configured mode."""
        return self._config.mode

    def evaluate(self, text: str, *, room_id: str | None, wake_score: float,
                 wake_vad: float | None, gate_accept: bool) -> WakeCheck | None:
        """Verify one acoustic-wake transcript.

        Args:
            text: The non-empty command transcript (never logged or stored).
            room_id: Capturing room.
            wake_score: openwakeword score of the wake.
            wake_vad: Silero probability of the wake frame, if measured.
            gate_accept: Whether the CommandGate accepted the transcript.

        Returns:
            The check, or None when the mode is off.
        """
        if self._config.mode is WakeVerificationMode.OFF:
            return None
        present = contains_wake_word(text, self._config.words)
        if present:
            check = WakeCheck(WakeCheckCode.PRESENT)
        elif self._config.mode is WakeVerificationMode.ENFORCE:
            check = WakeCheck(WakeCheckCode.REJECTED, True, f"missing_wake:{self._config.words[0]!r}")
        else:
            check = WakeCheck(WakeCheckCode.WOULD_REJECT)
        room = room_id or "unknown"
        now = self._clock()
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        with self._lock:
            rooms = self._days.setdefault(day, {})
            counts = rooms.setdefault(room, {})
            counts[check.code.value] = counts.get(check.code.value, 0) + 1
            if not gate_accept:
                counts["gate_rejected"] = counts.get("gate_rejected", 0) + 1
            self._prune_locked(now)
            self._events.append({
                "ts": round(now, 3), "room": room, "code": check.code.value,
                "score": None if wake_score is None else round(float(wake_score), 3),
                "vad": None if wake_vad is None or not math.isfinite(wake_vad) else round(float(wake_vad), 3),
                "gate_accept": bool(gate_accept),
            })
        logger.info("[wake-verify] code=%s room=%s score=%.2f vad=%s gate_accept=%s mode=%s",
                    check.code.value, room, float(wake_score), _fmt(wake_vad), gate_accept,
                    self._config.mode.value)
        return check

    def _prune_locked(self, now: float) -> None:
        """Drop day counters and events older than ``keep_days`` (lock held)."""
        cutoff = now - self._config.keep_days * 86400
        cutoff_day = datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d")
        for stale in [day for day in self._days if day < cutoff_day]:
            del self._days[stale]
        for stale in sorted(self._days)[:-self._config.keep_days]:
            del self._days[stale]
        if any(not _event_current(event, cutoff) for event in self._events):
            kept = [event for event in self._events if _event_current(event, cutoff)]
            self._events.clear()
            self._events.extend(kept)

    def snapshot(self) -> dict:
        """Counters for the 7-day review (codes and numbers only)."""
        with self._lock:
            self._prune_locked(self._clock())
            totals: dict[str, int] = {}
            for rooms in self._days.values():
                for counts in rooms.values():
                    for code, value in counts.items():
                        totals[code] = totals.get(code, 0) + value
            return {
                "schema": STATUS_SCHEMA,
                "mode": self._config.mode.value,
                "words": len(self._config.words),
                "updated_at": round(self._clock(), 3),
                "totals": totals,
                "days": json.loads(json.dumps(self._days)),
                "events": list(self._events),
            }

    def load(self) -> None:
        """Resume counters from the status file (blocking; call off-loop)."""
        path = self._config.status_path
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            logger.warning("[wake-verify] status unreadable: %s", type(error).__name__)
            return
        if not isinstance(data, dict) or data.get("schema") != STATUS_SCHEMA:
            return
        days = data.get("days")
        events = data.get("events")
        with self._lock:
            if isinstance(days, dict):
                for day, rooms in days.items():
                    if not isinstance(rooms, dict):
                        continue
                    for room, counts in rooms.items():
                        if not isinstance(counts, dict):
                            continue
                        target = self._days.setdefault(str(day), {}).setdefault(str(room), {})
                        for code, value in counts.items():
                            if type(value) is int and value >= 0:
                                target[str(code)] = target.get(str(code), 0) + value
            if isinstance(events, list):
                kept = [event for event in events if isinstance(event, dict)]
                self._events.extendleft(reversed(kept[-self._config.max_events:]))
            self._prune_locked(self._clock())

    def write_status(self) -> None:
        """Write the counters atomically with mode 0600 (blocking; call off-loop)."""
        path = self._config.status_path
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.snapshot(), sort_keys=True).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=".wake_verification.", dir=str(target.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def schedule_flush(self) -> asyncio.Task | None:
        """Start an off-loop status write at most once per flush interval.

        Call from the event loop; the command path never waits for the write.

        Returns:
            The write task when one was started, else None.
        """
        if not self._config.status_path:
            return None
        now = self._clock()
        if self._flush_task is not None and not self._flush_task.done():
            return None
        if now - self._last_flush < self._config.flush_interval_s:
            return None
        self._last_flush = now
        self._flush_task = asyncio.get_running_loop().create_task(asyncio.to_thread(self._write_quietly))
        return self._flush_task

    def _write_quietly(self) -> None:
        try:
            self.write_status()
        except Exception as error:
            logger.warning("[wake-verify] status write failed: %s", type(error).__name__)

    async def aclose(self) -> None:
        """Final status write on shutdown."""
        if self._flush_task is not None and not self._flush_task.done():
            await asyncio.gather(asyncio.shield(self._flush_task), return_exceptions=True)
        if self._config.status_path:
            await asyncio.to_thread(self._write_quietly)
