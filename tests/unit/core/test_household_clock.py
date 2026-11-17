"""P3-F1: household clock (routines.timezone, built once) and room namespace."""

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

import src.core.household_clock as household_module
from src.core.household_clock import (
    ROOM_SECTION_KEYS, SUPPORTED_HOUSEHOLD_TIMEZONE, HouseholdClock, RoomNamespace,
)

MVD = ZoneInfo("America/Montevideo")
# 02:30Z of the 12th is 23:30 of the 11th in Montevideo (UTC-3, no DST since 2015).
BEFORE_LOCAL_MIDNIGHT = datetime(2026, 11, 12, 2, 30, tzinfo=UTC)
# 03:00Z of the 12th is local midnight of the 12th.
LOCAL_MIDNIGHT = datetime(2026, 11, 12, 3, 0, tzinfo=UTC)
SETTINGS = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


@pytest.fixture
def foreign_process_tz(monkeypatch):
    """Run with a process TZ that is not the household's (the server's is Montevideo)."""
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def frozen(at: datetime) -> HouseholdClock:
    return HouseholdClock(SUPPORTED_HOUSEHOLD_TIMEZONE, time_source=lambda: at.timestamp())


def test_frozen_clock_across_the_utc_minus_3_midnight(foreign_process_tz):
    clock = frozen(BEFORE_LOCAL_MIDNIGHT)
    assert clock.now_utc() == BEFORE_LOCAL_MIDNIGHT and clock.now_utc().tzinfo is UTC
    local = clock.now_local()
    assert local.tzinfo is clock.zone
    assert (local.date(), local.hour, local.minute) == (date(2026, 11, 11), 23, 30)
    assert local.utcoffset() == timedelta(hours=-3)
    # The UTC date already is the 12th; the household date is still the 11th.
    assert clock.local_date() == date(2026, 11, 11)
    assert clock.next_local_midnight() == LOCAL_MIDNIGHT.timestamp()
    assert clock.format_time(clock.epoch()) == "23:30"

    after = frozen(LOCAL_MIDNIGHT)
    assert after.local_date() == date(2026, 11, 12)
    assert after.format_time(after.epoch()) == "00:00"
    assert after.next_local_midnight() == (LOCAL_MIDNIGHT + timedelta(days=1)).timestamp()
    # The last instant before local midnight still belongs to the 11th.
    assert after.local_date(LOCAL_MIDNIGHT.timestamp() - 0.001) == date(2026, 11, 11)


def test_epoch_is_storage_form_and_rejects_non_finite_sources():
    assert frozen(LOCAL_MIDNIGHT).epoch() == LOCAL_MIDNIGHT.timestamp()
    assert type(HouseholdClock(time_source=lambda: 5).epoch()) is float
    for bad in (float("nan"), float("inf"), "1", None, True):
        with pytest.raises(ValueError):
            HouseholdClock(time_source=lambda bad=bad: bad).epoch()


def test_to_epoch_reads_naive_values_as_household_wall_time(foreign_process_tz):
    clock = HouseholdClock()
    assert clock.to_epoch(datetime(2026, 11, 11, 23, 30)) == BEFORE_LOCAL_MIDNIGHT.timestamp()
    assert clock.to_epoch(datetime(2026, 11, 12, 0, 0)) == LOCAL_MIDNIGHT.timestamp()
    assert clock.to_epoch(BEFORE_LOCAL_MIDNIGHT) == BEFORE_LOCAL_MIDNIGHT.timestamp()
    assert clock.to_epoch(datetime(2026, 11, 11, 23, 30, tzinfo=MVD)) == BEFORE_LOCAL_MIDNIGHT.timestamp()
    assert clock.to_local(BEFORE_LOCAL_MIDNIGHT.timestamp()) == datetime(2026, 11, 11, 23, 30, tzinfo=MVD)


@pytest.mark.parametrize("wall", [
    datetime(2014, 10, 5, 2, 30),   # skipped when Uruguay's last DST started
    datetime(2015, 3, 8, 1, 30),    # repeated when it ended
])
def test_to_epoch_refuses_a_wall_time_that_does_not_exist_or_repeats(wall):
    with pytest.raises(ValueError):
        HouseholdClock().to_epoch(wall)


def test_only_the_routine_scheduler_timezone_is_supported():
    with pytest.raises(ValueError):
        HouseholdClock("UTC")
    with pytest.raises(ValueError):
        HouseholdClock("America/Argentina/Buenos_Aires")
    clock = HouseholdClock()
    assert clock.timezone == SUPPORTED_HOUSEHOLD_TIMEZONE == clock.zone.key


def test_supported_timezone_is_the_shipped_routines_timezone():
    configured = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))["routines"]["timezone"]
    assert SUPPORTED_HOUSEHOLD_TIMEZONE == configured
    HouseholdClock(configured)


def test_no_timezone_database_load_per_turn(monkeypatch):
    """ZoneInfo is constructed by the constructor only."""
    loads = []
    real = household_module.ZoneInfo

    def counting(key):
        loads.append(key)
        return real(key)

    monkeypatch.setattr(household_module, "ZoneInfo", counting)
    clock = frozen(BEFORE_LOCAL_MIDNIGHT)
    assert loads == [SUPPORTED_HOUSEHOLD_TIMEZONE]
    zone = clock.zone
    for _ in range(3):
        clock.now_utc(), clock.now_local(), clock.local_date(), clock.next_local_midnight()
        clock.to_epoch(datetime(2026, 11, 11, 8, 0)), clock.format_time(clock.epoch())
        clock.to_local(0.0)
    assert loads == [SUPPORTED_HOUSEHOLD_TIMEZONE]
    assert clock.zone is zone


ROOMS = ("living", "hall", "cocina", "escritorio", "bano")


@pytest.mark.parametrize("room", ROOMS)
def test_namespace_round_trip(room):
    namespace = RoomNamespace(ROOMS)
    zone = namespace.to_zone(room)
    assert zone == f"zone_{room}"
    assert namespace.to_room(zone) == room
    assert namespace.to_room(room) == room
    assert namespace.to_zone(zone) == zone
    assert namespace.to_zone(namespace.to_room(zone)) == zone


@pytest.mark.parametrize("value", [
    None, "", "zone_", "patio", "zone_patio", "zone_zone_living", "Living", "zone_Living", " living",
    "living ", "livingzone_", 3, b"living",
])
def test_namespace_refuses_anything_not_configured(value):
    namespace = RoomNamespace(ROOMS)
    assert namespace.to_room(value) is None
    assert namespace.to_zone(value) is None


@pytest.mark.parametrize("rooms", [("zone_living",), ("",), (" living",), (None,), (3,)])
def test_namespace_catalog_ids_must_be_bare_names(rooms):
    with pytest.raises(ValueError):
        RoomNamespace(rooms)


def test_namespace_from_the_shipped_rooms_section():
    rooms = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))["rooms"]
    namespace = RoomNamespace.from_rooms_config(rooms)
    assert namespace.rooms == frozenset(ROOMS)
    assert RoomNamespace.from_rooms_config(rooms, ROOM_SECTION_KEYS).rooms == namespace.rooms


def test_namespace_from_config_skips_reserved_and_non_room_values():
    namespace = RoomNamespace.from_rooms_config({
        "enabled": True, "wake_word": {"engine": "x"}, "fallback_room": "living",
        "living": {"name": "Living"}, "hall": {}, "notes": "texto",
    })
    assert namespace.rooms == frozenset({"living", "hall"})


@pytest.mark.parametrize(("resolved", "mic", "expected"), [
    ("living", "hall", "living"),          # the resolved room wins
    ("zone_living", None, "living"),       # transport form normalized
    (None, "hall", "hall"),                # else the configured mic room
    ("patio", "zone_hall", "hall"),        # an unconfigured resolved room falls back to the mic
    ("patio", "garage", None),             # nothing configured
    (None, None, None),
    ("", "", None),
])
def test_origin_room_of_a_turn(resolved, mic, expected):
    assert RoomNamespace(ROOMS).origin_room(resolved, mic) == expected


def test_empty_catalog_names_no_origin():
    namespace = RoomNamespace(())
    assert namespace.rooms == frozenset()
    assert namespace.origin_room("living", "living") is None
