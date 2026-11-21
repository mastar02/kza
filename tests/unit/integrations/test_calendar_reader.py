"""Calendar reader tests with private files and a local HA service fixture."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.home_assistant.ha_client import HomeAssistantClient
from src.integrations import calendar_reader
from src.integrations.calendar_reader import (
    CalendarAccess,
    CalendarRead,
    CalendarReader,
    CalendarScope,
    HACalendarSource,
    ICSCalendarSource,
)

pytestmark = pytest.mark.asyncio

START = datetime(2026, 11, 1, tzinfo=UTC)
END = datetime(2026, 11, 8, tzinfo=UTC)
LOCAL = ZoneInfo("America/Montevideo")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "state" / "data"
    root.mkdir(parents=True)
    return root


def write_ics(path: Path, *events: str) -> None:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0"]
    for event in events:
        lines.extend(["BEGIN:VEVENT", *event.split("\n"), "END:VEVENT"])
    lines.append("END:VCALENDAR")
    path.write_bytes("\r\n".join(lines).encode("utf-8") + b"\r\n")


def shared_ics(path: str) -> ICSCalendarSource:
    return ICSCalendarSource(CalendarScope.SHARED, Path(path))


async def test_reads_folded_text_civil_days_timezones_and_cancelled(data_root):
    path = data_root / "home.ics"
    write_ics(
        path,
        "UID:holiday\nSUMMARY:Día de la\n  familia\\, en casa\n"
        'LOCATION;ALTREP="cid:example;test":Sala\\; jardín\n'
        "DTSTART;VALUE=DATE:20261102\nDTEND;VALUE=DATE:20261104\n"
        "BEGIN:VALARM\nSUMMARY:Ignored alarm\nEND:VALARM",
        'UID:lesson\nSUMMARY:Clase\nSTATUS:TENTATIVE\nDTSTART;TZID="Europe/Madrid":20261102T070000\n'
        "DTEND:20261102T073000Z",
        "UID:cancelled\nSTATUS:CANCELLED\nSUMMARY:Omitir\nDTSTART:20261102T050000Z",
        "UID:floating\nSUMMARY:Cena\nDTSTART:20261102T200000",
    )
    # Folding is removed before decoding even when the break splits a code point.
    path.write_bytes(path.read_bytes().replace(b"\xc3\xad", b"\xc3\r\n\t\xad", 1))

    result = await CalendarReader([shared_ics("home.ics")], data_root=data_root).get_events(START, END)
    assert result.complete
    holiday, lesson, dinner = result.events
    assert (holiday.uid, holiday.summary, holiday.location) == ("holiday", "Día de la familia, en casa", "Sala; jardín")
    assert holiday.all_day
    assert holiday.start == datetime(2026, 11, 2, tzinfo=LOCAL)
    assert holiday.end == datetime(2026, 11, 4, tzinfo=LOCAL)
    assert lesson.start.astimezone(UTC) == datetime(2026, 11, 2, 6, tzinfo=UTC)
    assert lesson.end == datetime(2026, 11, 2, 7, 30, tzinfo=UTC)
    assert lesson.status == "TENTATIVE"
    assert dinner.start == dinner.end == datetime(2026, 11, 2, 20, tzinfo=LOCAL)


@pytest.mark.parametrize(
    ("first", "rule", "query_start", "query_end", "expected"),
    [
        (
            "20261101", "FREQ=DAILY;INTERVAL=2;COUNT=4", "2026-11-02", "2026-11-11",
            ["2026-11-03", "2026-11-05", "2026-11-07"],
        ),
        (
            "20261102", "FREQ=WEEKLY;BYDAY=MO,WE;INTERVAL=2;UNTIL=20261118", "2026-11-03", "2026-12-01",
            ["2026-11-04", "2026-11-16", "2026-11-18"],
        ),
        (
            "20260131", "FREQ=WEEKLY", "2026-02-10", "2026-02-26",
            ["2026-02-14", "2026-02-21"],
        ),
        (
            "20260131", "FREQ=MONTHLY;COUNT=3", "2026-01-01", "2026-08-01",
            ["2026-01-31", "2026-03-31", "2026-05-31"],
        ),
        (
            "20240131", "FREQ=MONTHLY;INTERVAL=2", "2026-01-01", "2026-06-01",
            ["2026-01-31", "2026-03-31", "2026-05-31"],
        ),
        (
            "20240229", "FREQ=YEARLY;INTERVAL=2;COUNT=3", "2025-01-01", "2035-01-01",
            ["2028-02-29", "2032-02-29"],
        ),
    ],
)
async def test_reads_basic_recurrences(data_root, first, rule, query_start, query_end, expected):
    write_ics(data_root / "home.ics", f"UID:series\nDTSTART;VALUE=DATE:{first}\nRRULE:{rule}")
    reader = CalendarReader([shared_ics("home.ics")], data_root=data_root)
    result = await reader.get_events(
        datetime.fromisoformat(query_start).replace(tzinfo=LOCAL),
        datetime.fromisoformat(query_end).replace(tzinfo=LOCAL),
    )
    assert result.complete
    assert [event.start.date().isoformat() for event in result.events] == expected
    assert all(event.all_day and event.end - event.start == timedelta(days=1) for event in result.events)


async def test_exdates_respect_timezone_and_do_not_extend_count(data_root):
    write_ics(
        data_root / "home.ics",
        "UID:series\nDTSTART;TZID=America/Montevideo:20261101T100000\n"
        "DTEND;TZID=America/Montevideo:20261101T110000\nRRULE:FREQ=DAILY;COUNT=4\n"
        "EXDATE:20261101T130000Z,20261103T130000Z\n"
        "EXDATE;TZID=America/Montevideo:20261102T100000",
    )
    result = await CalendarReader([shared_ics("home.ics")], data_root=data_root).get_events(START, END)
    assert result.complete
    assert [event.start for event in result.events] == [datetime(2026, 11, 4, 10, tzinfo=LOCAL)]
    assert result.events[0].end == datetime(2026, 11, 4, 11, tzinfo=LOCAL)


async def test_weekly_rule_keeps_wall_time_and_honors_utc_until(data_root):
    write_ics(
        data_root / "home.ics",
        "UID:dst\nDTSTART;TZID=Europe/Madrid:20261018T083000\n"
        "RRULE:FREQ=WEEKLY;UNTIL=20261101T073000Z",
        "UID:gap\nDTSTART;TZID=America/New_York:20260307T023000\nRRULE:FREQ=DAILY;COUNT=2",
    )
    result = await CalendarReader([shared_ics("home.ics")], data_root=data_root).get_events(
        datetime(2026, 3, 1, tzinfo=UTC), END,
    )
    assert result.complete
    assert [(event.uid, event.start.astimezone(UTC)) for event in result.events] == [
        ("gap", datetime(2026, 3, 7, 7, 30, tzinfo=UTC)),
        ("gap", datetime(2026, 3, 9, 6, 30, tzinfo=UTC)),
        ("dst", datetime(2026, 10, 18, 6, 30, tzinfo=UTC)),
        ("dst", datetime(2026, 10, 25, 7, 30, tzinfo=UTC)),
        ("dst", datetime(2026, 11, 1, 7, 30, tzinfo=UTC)),
    ]


async def test_window_includes_ongoing_events_and_excludes_its_right_boundary(data_root):
    write_ics(
        data_root / "home.ics",
        "UID:ongoing\nDTSTART:20261101T110000Z\nDTEND:20261101T140000Z",
        "UID:finished\nDTSTART:20261101T110000Z\nDTEND:20261101T120000Z",
        "UID:ongoing-series\nDTSTART:20261001T100000Z\nDTEND:20261003T100000Z\n"
        "RRULE:FREQ=MONTHLY",
        "UID:next-window\nDTSTART:20261102T120000Z",
    )

    def clock() -> datetime:
        return datetime(2026, 11, 1, 12, tzinfo=UTC)

    reader = CalendarReader([shared_ics("home.ics")], data_root=data_root, clock=clock)
    result = await reader.get_events()
    assert result.complete
    assert [event.uid for event in result.events] == ["ongoing-series", "ongoing"]


async def test_reads_valid_events_on_a_partially_invalid_file(data_root):
    write_ics(
        data_root / "home.ics",
        "UID:invalid\nDTSTART;TZID=Unknown/Zone:20261101T100000",
        "UID:valid\nDTSTART:20261101T100000Z",
    )
    result = await CalendarReader([shared_ics("home.ics")], data_root=data_root).get_events(START, END)
    assert not result.complete
    assert [event.uid for event in result.events] == ["valid"]


@pytest.mark.parametrize("kind", ["absolute", "traversal", "alias_traversal", "file_symlink", "dir_symlink"])
async def test_rejects_paths_outside_root_and_parent_components(data_root, kind):
    outside = data_root.parent / "outside.ics"
    inside = data_root / "inside.ics"
    for path in (outside, inside):
        write_ics(path, "UID:protected\nDTSTART:20261101T100000Z")
    (data_root / "file-link.ics").symlink_to(outside)
    (data_root / "dir-link").symlink_to(outside.parent, target_is_directory=True)
    paths = {
        "absolute": outside,
        "traversal": Path("../outside.ics"),
        "alias_traversal": Path("../data/inside.ics"),
        "file_symlink": Path("file-link.ics"),
        "dir_symlink": Path("dir-link/outside.ics"),
    }
    source = ICSCalendarSource(CalendarScope.SHARED, paths[kind])
    result = await CalendarReader([source], data_root=data_root).get_events(START, END)
    assert result == CalendarRead((), complete=False)


async def test_accepts_absolute_file_and_internal_symlink(data_root):
    path = data_root / "inside.ics"
    write_ics(path, "UID:home\nDTSTART:20261101T100000Z")
    (data_root / "link.ics").symlink_to(path.name)
    for source in (ICSCalendarSource(CalendarScope.SHARED, path), shared_ics("link.ics")):
        result = await CalendarReader([source], data_root=data_root).get_events(START, END)
        assert result.complete
        assert [event.uid for event in result.events] == ["home"]


@pytest.mark.parametrize("status", [200, 503])
async def test_reads_selected_entity_with_kza_ha_client_and_merges_chronologically(data_root, status):
    write_ics(data_root / "home.ics", "UID:local\nSUMMARY:Casa\nDTSTART:20261101T110000Z")
    calendar = "calendar.home"
    calls: list[tuple[str, object]] = []

    async def entity_state(request):
        calls.append((request.method, request.path))
        return web.json_response({"entity_id": calendar, "state": "off", "attributes": {}})

    async def get_events(request):
        calls.append((request.method, await request.json()))
        assert request.query["return_response"] == "true"
        if status == 503:
            raise web.HTTPServiceUnavailable()
        return web.json_response({"service_response": {
            calendar: {"events": [
                {"summary": "A la tarde", "start": "2026-11-01T13:00:00Z", "end": "2026-11-01T14:00:00Z"},
                {"summary": "Temprano", "start": "2026-11-01T04:00:00-04:00", "end": "2026-11-01T09:00:00Z"},
                {"summary": "Feriado", "start": "2026-11-02", "end": "2026-11-03"},
            ]},
            "calendar.unselected": {"events": [{"summary": "No solicitada"}]},
        }})

    app = web.Application()
    app.router.add_get(f"/api/states/{calendar}", entity_state)
    app.router.add_post("/api/services/calendar/get_events", get_events)
    async with TestServer(app) as server:
        client = HomeAssistantClient(url=str(server.make_url("")).rstrip("/"), token="test-only")
        try:
            reader = CalendarReader(
                [shared_ics("home.ics"), HACalendarSource(CalendarScope.SHARED, calendar)],
                data_root=data_root, ha_client=client,
            )
            result = await reader.get_events(START, END)
        finally:
            await client.close()

    assert calls == [
        ("GET", f"/api/states/{calendar}"),
        ("POST", {"entity_id": calendar, "start_date_time": START.isoformat(), "end_date_time": END.isoformat()}),
    ]
    assert result.complete is (status == 200)
    if status == 200:
        assert [event.summary for event in result.events] == ["Temprano", "Casa", "A la tarde", "Feriado"]
        assert result.events[-1].all_day
        assert result.events[-1].start == datetime(2026, 11, 2, tzinfo=LOCAL)
    else:
        assert [event.summary for event in result.events] == ["Casa"]


class RecordingHA:
    """Calendar-only test transport with optional service failure."""

    def __init__(self, *, state="off", mode="ok"):
        self.state = state
        self.mode = mode
        self.calls: list[tuple[str, str]] = []

    async def get_entity_state(self, entity_id):
        self.calls.append(("state", entity_id))
        return {"state": self.state}

    async def call_service_with_response(self, domain, service, entity_id, data=None, timeout=5.0):
        assert (domain, service) == ("calendar", "get_events")
        self.calls.append(("events", entity_id))
        if self.mode == "network":
            raise ConnectionError("connection failed")
        if self.mode == "timeout":
            await asyncio.Event().wait()
        return {"service_response": {entity_id: {"events": [{
            "summary": "Personal HA", "location": "Private HA location",
            "start": "2026-11-01T14:00:00Z", "end": "2026-11-01T15:00:00Z",
        }]}}}


@pytest.mark.parametrize(("state", "mode"), [("unavailable", "ok"), ("off", "network"), ("off", "timeout")])
async def test_ha_unavailable_network_failure_and_timeout(state, mode):
    client = RecordingHA(state=state, mode=mode)
    reader = CalendarReader(
        [HACalendarSource(CalendarScope.SHARED, "calendar.home")], ha_client=client, ha_timeout=0.02,
    )
    assert await reader.get_events(START, END) == CalendarRead((), complete=False)
    expected = [("state", "calendar.home")]
    if state == "off":
        expected.append(("events", "calendar.home"))
    assert client.calls == expected


async def test_ha_without_configured_entities_and_without_client():
    client = RecordingHA()
    assert await CalendarReader(ha_client=client).get_events(START, END) == CalendarRead((), complete=True)
    assert client.calls == []
    unconnected = CalendarReader([HACalendarSource(CalendarScope.SHARED, "calendar.home")])
    assert await unconnected.get_events(START, END) == CalendarRead((), complete=False)
    with pytest.raises(ValueError, match="explicit calendar"):
        HACalendarSource(CalendarScope.SHARED, "light.kitchen")
    with pytest.raises(ValueError, match="explicit calendar"):
        HACalendarSource(CalendarScope.SHARED, "calendar.*")
    with pytest.raises(ValueError, match="require an owner"):
        ICSCalendarSource(CalendarScope.PERSONAL, Path("personal.ics"))


async def test_personal_sources_require_current_authorized_owner_before_io(data_root, monkeypatch, caplog):
    write_ics(data_root / "home.ics", "UID:home\nSUMMARY:Compartido\nDTSTART:20261101T100000Z")
    write_ics(
        data_root / "personal.ics",
        "UID:personal\nSUMMARY:Personal ICS\nLOCATION:Private ICS location\nDTSTART:20261101T120000Z",
    )
    opens = []
    read_local_ics = calendar_reader._read_local_ics

    def track_open(root, path):
        opens.append(path.name)
        return read_local_ics(root, path)

    monkeypatch.setattr(calendar_reader, "_read_local_ics", track_open)
    client = RecordingHA()
    reader = CalendarReader(
        [
            shared_ics("home.ics"),
            ICSCalendarSource(CalendarScope.PERSONAL, Path("personal.ics"), owner_user_id="owner"),
            ICSCalendarSource(CalendarScope.PERSONAL, Path("missing.ics"), owner_user_id="third"),
            HACalendarSource(CalendarScope.PERSONAL, "calendar.personal", owner_user_id="owner"),
        ],
        data_root=data_root,
        ha_client=client,
    )

    expected = await CalendarReader([shared_ics("home.ics")], data_root=data_root).get_events(START, END)
    for denied in (
        CalendarAccess(),
        CalendarAccess(shared_only=True, user_id="owner", personal_authorized=True, session_current=lambda: True),
        CalendarAccess(shared_only=False, user_id="owner", session_current=lambda: True),
        CalendarAccess(shared_only=False, user_id="owner", personal_authorized=True),
        CalendarAccess(shared_only=False, user_id="owner", personal_authorized=True, session_current=lambda: False),
        CalendarAccess(shared_only=False, user_id="other", personal_authorized=True, session_current=lambda: True),
    ):
        opens.clear()
        assert await reader.get_events(START, END, access=denied) == expected
        assert opens == ["home.ics"]
        assert client.calls == []

    access = CalendarAccess(
        shared_only=False, user_id="owner", personal_authorized=True, session_current=lambda: True,
    )
    opens.clear()
    with caplog.at_level(logging.INFO):
        result = await reader.get_events(START, END, access=access)
    assert result.complete
    assert [event.summary for event in result.events] == ["Compartido", "Personal ICS", "Personal HA"]
    assert opens == ["home.ics", "personal.ics"]
    assert client.calls == [("state", "calendar.personal"), ("events", "calendar.personal")]
    for value in ("Personal ICS", "Private ICS location", "Personal HA", "Private HA location"):
        assert value not in caplog.text


async def test_revoked_session_drops_inflight_personal_ha_result():
    active = True

    class RevokingHA(RecordingHA):
        async def call_service_with_response(self, *args, **kwargs):
            nonlocal active
            result = await super().call_service_with_response(*args, **kwargs)
            active = False
            return result

    reader = CalendarReader(
        [HACalendarSource(CalendarScope.PERSONAL, "calendar.personal", owner_user_id="owner")],
        ha_client=RevokingHA(),
    )
    access = CalendarAccess(
        shared_only=False, user_id="owner", personal_authorized=True, session_current=lambda: active,
    )
    assert await reader.get_events(START, END, access=access) == CalendarRead((), complete=True)
