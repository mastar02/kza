"""Read household calendar events from local iCalendar files and Home Assistant."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import stat
from calendar import monthrange
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.sqlite_work import AuthorityGuard

logger = logging.getLogger(__name__)

_CONTENT_LINE = re.compile(r'([a-zA-Z0-9-]+(?:;(?:"[^"]*"|[^:";])*)*):(.*)')
_PARAMETERS = re.compile(r'(?:[^;"]|"[^"]*")+')
_TEXT_ESCAPE = re.compile(r"\\([nN,;\\])")
_DURATION = re.compile(
    r"\+?P(?:(?P<weeks>[0-9]+)W|(?:(?P<days>[0-9]+)D)?"
    r"(?:T(?=[0-9])(?:(?P<hours>[0-9]+)H)?(?:(?P<minutes>[0-9]+)M)?(?:(?P<seconds>[0-9]+)S)?)?)"
)
_WEEKDAYS = {day: index for index, day in enumerate(("MO", "TU", "WE", "TH", "FR", "SA", "SU"))}


class CalendarScope(StrEnum):
    """Audience of a configured calendar source."""

    SHARED = "shared"
    PERSONAL = "personal"


@dataclass(frozen=True)
class _CalendarSource:
    scope: CalendarScope
    owner_user_id: str | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        scope = CalendarScope(self.scope)
        if scope is CalendarScope.PERSONAL:
            if not self.owner_user_id or not self.owner_user_id.strip():
                raise ValueError("Personal calendars require an owner")
        elif self.owner_user_id is not None:
            raise ValueError("Shared calendars cannot have a personal owner")
        object.__setattr__(self, "scope", scope)


@dataclass(frozen=True)
class ICSCalendarSource(_CalendarSource):
    """Local .ics path relative to the configured data root, or inside it."""

    path: Path = field(repr=False)


@dataclass(frozen=True)
class HACalendarSource(_CalendarSource):
    """Explicitly selected Home Assistant calendar entity."""

    entity_id: str = field(repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not re.fullmatch(r"calendar\.[a-z0-9_]+", self.entity_id):
            raise ValueError("Expected an explicit calendar entity ID")


@dataclass(frozen=True)
class CalendarAccess:
    """Trusted caller context for a calendar query.

    Bind user_id and session_current to the same authenticated session. The
    caller derives personal_authorized from that session's permissions.
    """

    shared_only: bool = True
    user_id: str | None = field(default=None, repr=False)
    personal_authorized: bool = False
    session_current: AuthorityGuard | None = field(default=None, repr=False)

    def _allows(self, source: _CalendarSource) -> bool:
        if source.scope is CalendarScope.SHARED:
            return True
        if (self.shared_only or not self.personal_authorized
                or self.user_id != source.owner_user_id or self.session_current is None):
            return False
        try:
            return self.session_current() is True
        except Exception:
            return False


@dataclass(frozen=True)
class CalendarEvent:
    """One occurrence with aware bounds and an exclusive end.

    All-day bounds represent civil dates at midnight in the configured zone.
    """

    uid: str | None = field(repr=False)
    summary: str = field(repr=False)
    location: str | None = field(repr=False)
    status: str | None
    start: datetime
    end: datetime
    all_day: bool
    scope: CalendarScope


@dataclass(frozen=True)
class CalendarRead:
    """Events and whether all authorized sources could be read completely."""

    events: tuple[CalendarEvent, ...] = field(repr=False)
    complete: bool


class HACalendarClient(Protocol):
    """The two HomeAssistantClient methods used for calendar reads."""

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """Return the configured entity's state or None if unavailable."""
        ...

    async def call_service_with_response(
        self, domain: str, service: str, entity_id: str,
        data: dict | None = None, timeout: float = 5.0,
    ) -> dict | None:
        """Return the response for the calendar.get_events read."""
        ...


class CalendarReader:
    """Read visible calendars without creating or modifying remote or local data."""

    def __init__(
        self,
        sources: Sequence[ICSCalendarSource | HACalendarSource] = (),
        *,
        data_root: Path = Path("state/data"),
        timezone: str = "America/Montevideo",
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        ha_client: HACalendarClient | None = None,
        ha_timeout: float = 5.0,
    ) -> None:
        """Configure sources, the trusted data directory, and optional HA transport."""
        if not math.isfinite(ha_timeout) or ha_timeout <= 0:
            raise ValueError("Calendar HA timeout must be positive and finite")
        self._sources = tuple(sources)
        self._data_root = Path(data_root).absolute()
        self._timezone = ZoneInfo(timezone)
        self._clock = clock
        self._ha_client = ha_client
        self._ha_timeout = ha_timeout

    async def get_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        access: CalendarAccess = CalendarAccess(),
    ) -> CalendarRead:
        """List occurrences overlapping an aware, half-open interval in time order.

        The default interval starts at the injected clock and lasts 24 hours.
        Only authorized sources contribute events or affect completeness.
        """
        start = _as_utc(self._clock() if start is None else start)
        end = start + timedelta(days=1) if end is None else _as_utc(end)
        if end <= start:
            raise ValueError("Calendar window end must follow start")

        reads: list[tuple[_CalendarSource, CalendarRead]] = []
        for source in self._sources:
            if not access._allows(source):
                continue
            if isinstance(source, ICSCalendarSource):
                result = await asyncio.to_thread(self._read_ics, source, start, end)
            else:
                result = await self._read_ha(source, start, end, access)
            reads.append((source, result))

        visible = [result for source, result in reads if access._allows(source)]
        events = sorted(
            (event for result in visible for event in result.events),
            key=lambda event: _as_utc(event.start),
        )
        return CalendarRead(tuple(events), all(result.complete for result in visible))

    def _read_ics(self, source: ICSCalendarSource, start: datetime, end: datetime) -> CalendarRead:
        try:
            content = _read_local_ics(self._data_root, Path(source.path))
        except (OSError, ValueError):
            logger.debug("Local calendar could not be read")
            return CalendarRead((), complete=False)

        series: dict[str | int, list[_EventProperties]] = {}
        complete = True
        try:
            for index, properties in enumerate(_vevents(content)):
                uid = _ics_text(properties, "UID")
                series.setdefault(uid or index, []).append(properties)
        except ValueError:
            complete = False

        events = []
        for components in series.values():
            try:
                events.extend(_ics_series(components, source.scope, self._timezone, start, end))
            except (ValueError, OverflowError, ZoneInfoNotFoundError):
                complete = False
        return CalendarRead(tuple(events), complete=complete)

    async def _read_ha(
        self, source: HACalendarSource, start: datetime, end: datetime, access: CalendarAccess,
    ) -> CalendarRead:
        failed = CalendarRead((), complete=False)
        if self._ha_client is None:
            return failed
        entity = source.entity_id
        try:
            async with asyncio.timeout(self._ha_timeout):
                state = await self._ha_client.get_entity_state(entity)
                if not state or state.get("state") in (None, "unknown", "unavailable"):
                    return failed
                if not access._allows(source):
                    return failed
                result = await self._ha_client.call_service_with_response(
                    "calendar", "get_events", entity,
                    data={"start_date_time": start.isoformat(), "end_date_time": end.isoformat()},
                    timeout=self._ha_timeout,
                )
        except Exception:
            logger.debug("Home Assistant calendar request failed")
            return failed

        response = result.get("service_response") if isinstance(result, Mapping) else None
        entry = response.get(entity) if isinstance(response, Mapping) else None
        rows = entry.get("events") if isinstance(entry, Mapping) else None
        if not isinstance(rows, list):
            return failed

        events = []
        complete = True
        for row in rows:
            try:
                event = _ha_event(row, source.scope, self._timezone)
                if event is not None and _overlaps(event, start, end):
                    events.append(event)
            except (TypeError, ValueError, OverflowError):
                complete = False
        return CalendarRead(tuple(events), complete=complete)


def _read_local_ics(root: Path, path: Path) -> str:
    if ".." in path.parts or path.suffix.lower() != ".ics":
        raise ValueError("Invalid local calendar path")
    resolved_root = root.resolve(strict=True)
    if path.is_absolute():
        if path.is_relative_to(root):
            path = path.relative_to(root)
        else:
            path = path.relative_to(resolved_root)
    resolved_path = (resolved_root / path).resolve(strict=True)
    relative = resolved_path.relative_to(resolved_root)
    if not relative.parts:
        raise ValueError("Expected a calendar file")

    # Open the checked path component by component, rejecting symlink swaps
    # between resolution and open. Nonblocking open also rejects FIFOs safely.
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        directory = os.open(resolved_root, dir_flags)
        stack.callback(os.close, directory)
        for component in relative.parts[:-1]:
            directory = os.open(component, dir_flags, dir_fd=directory)
            stack.callback(os.close, directory)
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        stack.callback(os.close, fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Expected a regular calendar file")
        with open(fd, "rb", closefd=False) as stream:
            unfolded = re.sub(rb"\r?\n[ \t]", b"", stream.read())
    return unfolded.decode("utf-8-sig")


@dataclass(frozen=True)
class _Property:
    value: str
    parameters: dict[str, str] = field(default_factory=dict)


type _EventProperties = dict[str, list[_Property]]


def _vevents(content: str) -> Iterator[_EventProperties]:
    components: list[str] = []
    properties: _EventProperties = {}
    found_calendar = False
    for line in content.split("\n"):
        line = line.removesuffix("\r")
        if not line:
            continue
        match = _CONTENT_LINE.fullmatch(line)
        if match is None:
            raise ValueError("Invalid calendar content line")
        header, value = match.groups()
        name, *parameters = _PARAMETERS.findall(header)
        name = name.upper()
        if name == "BEGIN":
            component = value.upper()
            if not components:
                if component != "VCALENDAR":
                    raise ValueError("Expected VCALENDAR")
                found_calendar = True
            components.append(component)
        elif name == "END":
            if not components or components[-1] != value.upper():
                raise ValueError("Mismatched calendar component")
            if components == ["VCALENDAR", "VEVENT"]:
                yield properties
                properties = {}
            components.pop()
        elif components == ["VCALENDAR", "VEVENT"]:
            params = {key.upper(): val.strip('"') for param in parameters
                      for key, sep, val in [param.partition("=")] if sep}
            properties.setdefault(name, []).append(_Property(value, params))
    if not found_calendar or components:
        raise ValueError("Incomplete VCALENDAR")


def _as_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("Calendar window requires aware datetimes")
    return value.astimezone(UTC)


def _ics_time(prop: _Property, local: ZoneInfo) -> tuple[datetime, bool]:
    value = prop.value
    if re.fullmatch(r"[0-9]{8}", value):
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=local), True
    if prop.parameters.get("VALUE", "").upper() == "DATE" or not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z?", value):
        raise ValueError("Invalid iCalendar date or date-time")
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), False
    zone = ZoneInfo(prop.parameters["TZID"]) if "TZID" in prop.parameters else local
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=zone), False


def _ics_text(properties: _EventProperties, name: str) -> str:
    if name not in properties:
        return ""
    return _TEXT_ESCAPE.sub(
        lambda match: "\n" if match[1] in ("n", "N") else match[1], properties[name][0].value,
    )


def _ics_series(
    components: list[_EventProperties], scope: CalendarScope, local: ZoneInfo,
    window_start: datetime, window_end: datetime,
) -> Iterator[CalendarEvent]:
    masters = []
    overrides: dict[datetime, _EventProperties] = {}
    for properties in components:
        if "RDATE" in properties:
            raise ValueError("RDATE is unsupported")
        if "RECURRENCE-ID" not in properties:
            masters.append(properties)
            continue
        identifiers = properties["RECURRENCE-ID"]
        if len(identifiers) != 1 or not _ics_text(properties, "UID") or "RRULE" in properties:
            raise ValueError("Invalid calendar override")
        identifier = identifiers[0]
        if identifier.parameters.keys() - {"TZID", "VALUE"}:
            raise ValueError("Unsupported RECURRENCE-ID parameter")
        recurrence_start, all_day = _ics_time(identifier, local)
        value_type = identifier.parameters.get("VALUE", "DATE" if all_day else "DATE-TIME")
        if value_type.upper() != ("DATE" if all_day else "DATE-TIME"):
            raise ValueError("Invalid RECURRENCE-ID value type")
        original = _as_utc(recurrence_start)
        if original in overrides:
            raise ValueError("Duplicate calendar override")
        overrides[original] = properties

    # Collect overrides regardless of their file order or the query window.
    for properties in masters:
        yield from _ics_occurrences(properties, scope, local, window_start, window_end, overrides.keys())
    for properties in overrides.values():
        yield from _ics_occurrences(properties, scope, local, window_start, window_end)


def _ics_duration(value: str, all_day: bool) -> tuple[timedelta, timedelta]:
    match = _DURATION.fullmatch(value)
    if match is None:
        raise ValueError("Unsupported iCalendar duration")
    weeks, days, hours, minutes, seconds = match.groups()
    if all_day and any((hours, minutes, seconds)):
        raise ValueError("All-day duration must use days or weeks")
    civil = timedelta(weeks=int(weeks or 0), days=int(days or 0))
    elapsed = timedelta(hours=int(hours or 0), minutes=int(minutes or 0), seconds=int(seconds or 0))
    if not civil and not elapsed:
        raise ValueError("Duration must be positive")
    return civil, elapsed


def _add_duration(start: datetime, civil: timedelta, elapsed: timedelta) -> datetime:
    end = start + civil if civil else start
    return (_as_utc(end) + elapsed).astimezone(start.tzinfo) if elapsed else end


def _overlaps(event: CalendarEvent, start: datetime, end: datetime) -> bool:
    begins, finishes = _as_utc(event.start), _as_utc(event.end)
    if begins == finishes:
        return start <= begins < end
    return begins < end and finishes > start


def _ics_occurrences(
    properties: _EventProperties, scope: CalendarScope, local: ZoneInfo,
    window_start: datetime, window_end: datetime,
    suppressed: Set[datetime] = frozenset(),
) -> Iterator[CalendarEvent]:
    first = {name: values[0] for name, values in properties.items()}
    status = first.get("STATUS", _Property("")).value.upper() or None
    if status == "CANCELLED":
        return
    if "DTSTART" not in first:
        raise ValueError("Expected DTSTART")
    start, all_day = _ics_time(first["DTSTART"], local)
    if "DTEND" in first:
        if "DURATION" in first:
            raise ValueError("DTEND and DURATION cannot coexist")
        end, end_all_day = _ics_time(first["DTEND"], local)
        if end_all_day != all_day:
            raise ValueError("Mismatched calendar date types")
        civil = end - start if all_day else timedelta()
        elapsed = timedelta() if all_day else _as_utc(end) - _as_utc(start)
    else:
        civil, elapsed = (
            _ics_duration(first["DURATION"].value, all_day)
            if "DURATION" in first else (timedelta(days=int(all_day)), timedelta())
        )
        end = _add_duration(start, civil, elapsed)
    if _as_utc(end) < _as_utc(start) or (all_day and end == start):
        raise ValueError("Invalid calendar event end")

    event = CalendarEvent(
        _ics_text(properties, "UID") or None, _ics_text(properties, "SUMMARY"),
        _ics_text(properties, "LOCATION") or None, status, start, end, all_day, scope,
    )
    exclusions = suppressed | {
        _as_utc(_ics_time(_Property(value, prop.parameters), local)[0])
        for prop in properties.get("EXDATE", ())
        for value in prop.value.split(",")
    }
    if "RRULE" not in first:
        if _as_utc(start) not in exclusions and _overlaps(event, window_start, window_end):
            yield event
        return

    rule = _parse_rule(first["RRULE"].value, start, local)
    duration = civil + elapsed
    # Leave room for an ongoing event and for offset changes across its span.
    local_window_day = window_start.astimezone(start.tzinfo).date()
    lower = date.fromordinal(max(1, local_window_day.toordinal() - duration.days - 2))
    limit = min(window_end, rule.until) if rule.until is not None else window_end
    upper = limit.astimezone(start.tzinfo).date()
    matched = 0
    for day in _rule_dates(start.date(), lower, upper, rule):
        begins = datetime.combine(day, start.timetz())
        instant = _as_utc(begins)
        if instant >= window_end or (rule.until is not None and instant > rule.until):
            break
        if day != start.date() and instant.astimezone(start.tzinfo) != begins:
            continue  # RFC 5545 ignores generated local times that do not exist.
        matched += 1
        if rule.count is not None and matched > rule.count:
            break
        if day < lower or instant in exclusions:
            continue
        finishes = _add_duration(begins, civil, elapsed).astimezone(end.tzinfo)
        occurrence = replace(event, start=begins, end=finishes)
        if _overlaps(occurrence, window_start, window_end):
            yield occurrence


@dataclass(frozen=True)
class _Rule:
    frequency: str
    interval: int
    count: int | None
    until: datetime | None
    weekdays: tuple[int, ...]


def _parse_rule(value: str, start: datetime, local: ZoneInfo) -> _Rule:
    parts = dict(part.split("=", 1) for part in value.upper().split(";"))
    frequency = parts.get("FREQ")
    if frequency not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
        raise ValueError("Unsupported calendar recurrence frequency")
    allowed = {"FREQ", "INTERVAL", "COUNT", "UNTIL"}
    if frequency == "WEEKLY":
        allowed.add("BYDAY")
    if parts.keys() - allowed:
        raise ValueError("Unsupported calendar recurrence part")

    interval = int(parts.get("INTERVAL", "1"))
    count = int(parts["COUNT"]) if "COUNT" in parts else None
    if interval < 1 or (count is not None and count < 1):
        raise ValueError("Invalid calendar recurrence interval or count")
    until = None
    if "UNTIL" in parts:
        value = parts["UNTIL"]
        until_zone = start.tzinfo.key if isinstance(start.tzinfo, ZoneInfo) else "UTC"
        until = _as_utc(_ics_time(_Property(value, {"TZID": until_zone}), local)[0])
    weekdays = (start.weekday(),)
    if "BYDAY" in parts:
        names = parts["BYDAY"].split(",")
        if any(name not in _WEEKDAYS for name in names):
            raise ValueError("Invalid weekly recurrence day")
        weekdays = tuple(sorted({_WEEKDAYS[name] for name in names}))
    return _Rule(frequency, interval, count, until, weekdays)


def _rule_dates(seed: date, lower: date, upper: date, rule: _Rule) -> Iterator[date]:
    if upper < seed:
        return
    # COUNT includes the seed and occurrences before the window, even those
    # later removed by EXDATE. Open-ended rules can jump near the query.
    first = seed if rule.count is not None else max(seed, lower)
    if first == seed:
        yield seed

    if rule.frequency in ("DAILY", "WEEKLY"):
        seed_day, first_day, last_day = seed.toordinal(), first.toordinal(), upper.toordinal()
        anchor, step, offsets = seed_day, rule.interval, (0,)
        if rule.frequency == "WEEKLY":
            anchor -= seed.weekday()
            step *= 7
            offsets = rule.weekdays
        cycle = (first_day - anchor) // step
        for base in range(anchor + cycle * step, last_day + 1, step):
            for offset in offsets:
                ordinal = base + offset
                if seed_day < ordinal <= last_day and ordinal >= first_day:
                    yield date.fromordinal(ordinal)
        return

    start_month = seed.year * 12 + seed.month - 1
    first_month = first.year * 12 + first.month - 1
    last_month = upper.year * 12 + upper.month - 1
    step = rule.interval * (12 if rule.frequency == "YEARLY" else 1)
    cycle = max(1, (first_month - start_month) // step)
    for month_index in range(start_month + cycle * step, last_month + 1, step):
        year, zero_month = divmod(month_index, 12)
        month = zero_month + 1
        if seed.day <= monthrange(year, month)[1]:
            day = date(year, month, seed.day)
            if first <= day <= upper:
                yield day


def _ha_event(row: object, scope: CalendarScope, local: ZoneInfo) -> CalendarEvent | None:
    if not isinstance(row, Mapping):
        raise ValueError("Invalid Home Assistant calendar response")

    def string(key: str) -> str | None:
        value = row.get(key)
        return value if isinstance(value, str) else None

    def timestamp(key: str) -> tuple[datetime, bool]:
        value = string(key)
        if value is None:
            raise ValueError("Expected Home Assistant calendar timestamp")
        if len(value) == 10:
            return datetime.combine(date.fromisoformat(value), time.min, local), True
        value = datetime.fromisoformat(value)
        return value if value.tzinfo else value.replace(tzinfo=local), False

    status = string("status")
    if status and status.upper() == "CANCELLED":
        return None
    start, all_day = timestamp("start")
    end, end_all_day = timestamp("end")
    if all_day != end_all_day or _as_utc(end) < _as_utc(start):
        raise ValueError("Invalid Home Assistant calendar event bounds")
    if all_day and end == start:
        end += timedelta(days=1)
    return CalendarEvent(string("uid"), string("summary") or "", string("location"), status,
                         start, end, all_day, scope)
