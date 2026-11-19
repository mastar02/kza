"""Per-room budget for the XVF3800 firmware reboot of a dead capture stream.

Incident 2026-11-19 (usb_overrun_20261119): after a warm reboot of the host
every reSpeaker XVF3800 kept its isochronous endpoint stuck. PortAudio opened
the stream, but no usable audio ever arrived, and reopening it or resetting
the port from sysfs did not help. The firmware REBOOT vendor command
recovered every microphone remotely. This module decides, per room, when the
audio loop may send that command: only after the ordinary watchdog recovery
failed a configured number of times, at most a configured number of times
per hour, and with an exponential backoff between two reboots of one room.

Pure bookkeeping (no I/O, no asyncio): the audio loop owns the side effects.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# The cap counts reboots inside this sliding window.
REBOOT_WINDOW_S = 3600.0


class XvfRebootDecision(str, Enum):
    """What the watchdog should do with a room whose stream is dead."""

    # Below the failure threshold: keep the ordinary recovery.
    RECOVER = "recover"
    # Threshold reached and budget left: send the firmware reboot.
    REBOOT = "reboot"
    # Hourly cap reached: keep the ordinary recovery and ask for a replug.
    CAPPED = "capped"


@dataclass(frozen=True)
class XvfRebootPolicy:
    """Configuration of the automatic firmware reboot (settings.yaml).

    Attributes:
        enabled: Master switch; False keeps the previous behaviour exactly.
        failures_before_reboot: Ordinary recoveries of one room that must
            fail (the reopened stream is dead again) before the first reboot.
            Every reboot inside the window doubles it (backoff).
        max_per_hour: Reboots of one room inside a sliding hour.
        settle_s: Upper bound to wait for the device to re-enumerate and
            expose its ALSA card again after the reboot.
    """

    enabled: bool = False
    failures_before_reboot: int = 2
    max_per_hour: int = 3
    settle_s: float = 10.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("xvf reboot enabled must be a boolean")
        for name in ("failures_before_reboot", "max_per_hour"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"xvf reboot {name} must be a positive integer")
        if (type(self.settle_s) not in (int, float) or not math.isfinite(self.settle_s)
                or self.settle_s <= 0):
            raise ValueError("xvf reboot settle_s must be a positive finite number")


@dataclass
class _RoomBudget:
    failures: int = 0
    counted_opening: float | None = None
    reboots: deque = field(default_factory=deque)
    cap_reported: bool = False


class XvfRebootBudget:
    """Count failed recoveries and firmware reboots for each room.

    A failure is one ordinary recovery scheduled for a dead stream opening;
    when the watchdog finds the next opening dead too, that recovery failed.
    Each opening counts once, keyed by its ``opened_ts``.
    """

    def __init__(self, policy: XvfRebootPolicy, clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self._clock = clock
        self._rooms: dict[str, _RoomBudget] = {}

    def _room(self, room_id: str) -> _RoomBudget:
        return self._rooms.setdefault(room_id, _RoomBudget())

    def _prune(self, room: _RoomBudget) -> None:
        cutoff = self._clock() - REBOOT_WINDOW_S
        while room.reboots and room.reboots[0] <= cutoff:
            room.reboots.popleft()
        if len(room.reboots) < self.policy.max_per_hour:
            room.cap_reported = False

    def failures(self, room_id: str) -> int:
        """Ordinary recoveries of this room that did not bring audio back."""
        return self._room(room_id).failures

    def reboots_in_window(self, room_id: str) -> int:
        """Firmware reboots of this room within the last hour."""
        room = self._room(room_id)
        self._prune(room)
        return len(room.reboots)

    def required_failures(self, room_id: str) -> int:
        """Failures needed before the next reboot: doubles per recent reboot."""
        return self.policy.failures_before_reboot * 2 ** self.reboots_in_window(room_id)

    def note_audio(self, room_id: str) -> None:
        """The room delivered real audio: its failure streak ends."""
        room = self._room(room_id)
        room.failures = 0
        room.counted_opening = None
        room.cap_reported = False

    def note_recovery(self, room_id: str, opening: float) -> None:
        """An ordinary recovery was admitted for this dead stream opening."""
        room = self._room(room_id)
        if room.counted_opening == opening:
            return
        room.counted_opening = opening
        room.failures += 1

    def decide(self, room_id: str) -> XvfRebootDecision:
        """Choose between the ordinary recovery, a reboot or the cap."""
        if not self.policy.enabled:
            return XvfRebootDecision.RECOVER
        if self.failures(room_id) < self.required_failures(room_id):
            return XvfRebootDecision.RECOVER
        if self.reboots_in_window(room_id) >= self.policy.max_per_hour:
            return XvfRebootDecision.CAPPED
        return XvfRebootDecision.REBOOT

    def charge(self, room_id: str) -> int:
        """Record one reboot attempt; the failure streak starts over.

        Returns:
            The number of reboots of this room inside the window, this one
            included.
        """
        room = self._room(room_id)
        self._prune(room)
        room.reboots.append(self._clock())
        room.failures = 0
        room.counted_opening = None
        return len(room.reboots)

    def report_cap(self, room_id: str) -> bool:
        """True the first time the cap is hit in an episode (log once)."""
        room = self._room(room_id)
        self._prune(room)
        if room.cap_reported:
            return False
        room.cap_reported = True
        return True
