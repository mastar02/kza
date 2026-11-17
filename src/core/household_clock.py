"""Household clock and room namespace shared by voice and job consumers.

P3-F1 (p3/design.json household_conventions.clock and .room_namespace):

* One :class:`HouseholdClock` is built once in ``main`` from
  ``routines.timezone`` (no new configuration key). The ``ZoneInfo`` is
  constructed in the constructor, so no consumer loads the timezone database
  per turn. Storage stays UTC epoch seconds; the clock converts to and from the
  household's local calendar.
* One :class:`RoomNamespace` is built once from the configured room catalog.
  The canonical form is the bare room id (``living``); ``zone_<room>`` is a
  transport form for components that already expect it (router, response
  handler, ``ResponseHandler.cancel``, ``ZoneManager.set_active_zone``).
* The origin room of a voice turn is the configured room of the microphone
  that heard it, else ``None`` (:meth:`RoomNamespace.origin_room`; user answer
  P3F1-R3 option A, 2026-11-17). Bluetooth confirmation only confirms that
  room; Bluetooth alone, the last known room, the room named in the command
  (the target) and ``rooms.fallback_room`` never name it, and neither does
  acoustic zone detection.
"""

import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SUPPORTED_HOUSEHOLD_TIMEZONE = "America/Montevideo"
ZONE_PREFIX = "zone_"
# Keys of the ``rooms`` settings section that configure rooms rather than name one.
ROOM_SECTION_KEYS = frozenset({
    "enabled", "cross_validation", "fallback_room",
    "dedup_window_ms", "ma1260", "wake_word", "detection",
    "telemetry", "barge_in", "stream_watchdog", "ambient_guard", "endpointing",
    "direction_observation",
})


class HouseholdClock:
    """Aware wall clock on the household's local calendar.

    The timezone is limited to the one ``RoutineScheduler`` accepts until the
    household decides otherwise (design household_conventions.clock.constraint).

    Args:
        timezone: IANA name from ``routines.timezone``.
        time_source: Epoch-seconds source; tests inject a frozen value.

    Raises:
        ValueError: The timezone is not the supported household timezone.
    """

    def __init__(self, timezone: str = SUPPORTED_HOUSEHOLD_TIMEZONE, *,
                 time_source: Callable[[], float] = time.time) -> None:
        if timezone != SUPPORTED_HOUSEHOLD_TIMEZONE:
            raise ValueError(f"Household timezone must be {SUPPORTED_HOUSEHOLD_TIMEZONE}")
        self._timezone = timezone
        self._zone = ZoneInfo(timezone)
        self._time_source = time_source

    @property
    def timezone(self) -> str:
        """IANA name of the household timezone."""
        return self._timezone

    @property
    def zone(self) -> ZoneInfo:
        """The ``ZoneInfo`` built once by the constructor."""
        return self._zone

    def epoch(self) -> float:
        """Current time as UTC epoch seconds (the storage form)."""
        value = self._time_source()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Household clock source must return finite epoch seconds")
        return float(value)

    def now_utc(self) -> datetime:
        """Current time, aware in UTC."""
        return datetime.fromtimestamp(self.epoch(), UTC)

    def now_local(self) -> datetime:
        """Current time, aware in the household timezone."""
        return self.to_local(self.epoch())

    def to_local(self, epoch: float) -> datetime:
        """Convert a stored epoch to an aware household wall time."""
        return datetime.fromtimestamp(epoch, self._zone)

    def local_date(self, epoch: float | None = None) -> date:
        """Household calendar date of ``epoch`` (now when omitted)."""
        return self.to_local(self.epoch() if epoch is None else epoch).date()

    def next_local_midnight(self, epoch: float | None = None) -> float:
        """Epoch of the household midnight that ends the day containing ``epoch``."""
        following = self.local_date(epoch) + timedelta(days=1)
        return self.to_epoch(datetime(following.year, following.month, following.day))

    def to_epoch(self, local_dt: datetime) -> float:
        """Convert a household wall time to UTC epoch seconds.

        Args:
            local_dt: Naive household wall time, or an aware datetime.

        Raises:
            ValueError: A naive wall time that does not exist or occurs twice
                in the household timezone (a daylight-saving change): both
                folds then carry different offsets.
        """
        if local_dt.tzinfo is not None and local_dt.utcoffset() is not None:
            return local_dt.timestamp()
        early = local_dt.replace(tzinfo=self._zone, fold=0)
        if early.utcoffset() != local_dt.replace(tzinfo=self._zone, fold=1).utcoffset():
            raise ValueError("Nonexistent or ambiguous household wall time")
        return early.timestamp()

    def format_time(self, epoch: float) -> str:
        """Household ``HH:MM`` for spoken and logged times."""
        return self.to_local(epoch).strftime("%H:%M")


class RoomNamespace:
    """Normalizer between bare room ids and ``zone_<room>`` transport ids.

    Args:
        configured_rooms: Room catalog ids (keys configured under ``rooms``).

    Raises:
        ValueError: A room id is empty, not a string, or already prefixed.
    """

    def __init__(self, configured_rooms: Iterable[str]) -> None:
        rooms = tuple(configured_rooms)
        for room in rooms:
            if type(room) is not str or not room or room != room.strip() or room.startswith(ZONE_PREFIX):
                raise ValueError("Room catalog ids must be bare non-empty names")
        self._rooms = frozenset(rooms)

    @classmethod
    def from_rooms_config(cls, rooms_config: Mapping,
                          reserved_keys: Iterable[str] = ROOM_SECTION_KEYS) -> "RoomNamespace":
        """Build from the ``rooms`` settings section, skipping its reserved keys.

        Args:
            rooms_config: The ``rooms`` section of ``config/settings.yaml``.
            reserved_keys: Section keys that configure rooms rather than name one.
        """
        reserved = frozenset(reserved_keys)
        return cls(key for key, value in rooms_config.items() if key not in reserved and isinstance(value, dict))

    @property
    def rooms(self) -> frozenset[str]:
        """Configured bare room ids."""
        return self._rooms

    def to_room(self, value: str | None) -> str | None:
        """Canonical bare room id for ``living`` or ``zone_living``; None if not configured."""
        if type(value) is not str:
            return None
        room = value[len(ZONE_PREFIX):] if value.startswith(ZONE_PREFIX) else value
        return room if room in self._rooms else None

    def to_zone(self, room: str | None) -> str | None:
        """Transport id ``zone_<room>`` for a configured room; None otherwise."""
        canonical = self.to_room(room)
        return None if canonical is None else ZONE_PREFIX + canonical

    def origin_room(self, mic_room: str | None) -> str | None:
        """Origin room of a turn: only the microphone that heard it (P3F1-R3 option A).

        A room the room context resolved some other way (Bluetooth alone, the
        user's last known room, the room named in the command, or
        ``rooms.fallback_room``) is not an origin; a Bluetooth-confirmed match
        is the microphone's room itself. Without an origin, shared notices go
        to ``reminders.output_zone`` (the living) until SCHEMA-V2.

        Args:
            mic_room: Room of the microphone that captured the turn, bare or
                in ``zone_<room>`` form.

        Returns:
            The microphone's canonical room when configured, else None.
        """
        return self.to_room(mic_room)
