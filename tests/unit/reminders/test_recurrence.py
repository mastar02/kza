"""Tests for recurrence engine — calculating next trigger dates."""
import pytest
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from src.reminders.recurrence import next_trigger, parse_recurrence, RecurrenceType

# P3-F1: occurrences are counted on the household calendar, whatever the process TZ is.
MVD = ZoneInfo("America/Montevideo")
TOKYO = ZoneInfo("Asia/Tokyo")


def local(*args, zone=MVD) -> float:
    """Epoch of a household wall time."""
    return datetime(*args, tzinfo=zone).timestamp()


def wall(epoch: float, zone=MVD) -> datetime:
    return datetime.fromtimestamp(epoch, zone)


class TestParseRecurrence:
    def test_daily(self):
        r = parse_recurrence("daily")
        assert r == (RecurrenceType.DAILY, None)

    def test_weekdays(self):
        r = parse_recurrence("weekdays")
        assert r == (RecurrenceType.WEEKDAYS, None)

    def test_weekly_monday(self):
        r = parse_recurrence("weekly:1")
        assert r == (RecurrenceType.WEEKLY, 1)

    def test_weekly_friday(self):
        r = parse_recurrence("weekly:5")
        assert r == (RecurrenceType.WEEKLY, 5)

    def test_monthly(self):
        r = parse_recurrence("monthly:15")
        assert r == (RecurrenceType.MONTHLY, 15)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_recurrence("every_3_days")


class TestNextTrigger:
    def test_daily_next_day(self):
        base = local(2026, 3, 2, 9, 0)  # Monday
        nxt = next_trigger(base, "daily", zone=MVD)
        result = wall(nxt)
        assert result.day == 3
        assert result.hour == 9

    def test_weekdays_friday_to_monday(self):
        base = local(2026, 3, 6, 9, 0)  # Friday
        nxt = next_trigger(base, "weekdays", zone=MVD)
        result = wall(nxt)
        assert result.weekday() == 0  # Monday
        assert result.day == 9

    def test_weekdays_monday_to_tuesday(self):
        base = local(2026, 3, 2, 9, 0)  # Monday
        nxt = next_trigger(base, "weekdays", zone=MVD)
        result = wall(nxt)
        assert result.weekday() == 1  # Tuesday

    def test_weekly_next_week(self):
        base = local(2026, 3, 2, 9, 0)
        nxt = next_trigger(base, "weekly:1", zone=MVD)
        result = wall(nxt)
        assert result.day == 9
        assert result.weekday() == 0

    def test_monthly(self):
        base = local(2026, 3, 15, 9, 0)
        nxt = next_trigger(base, "monthly:15", zone=MVD)
        result = wall(nxt)
        assert result.month == 4
        assert result.day == 15

    def test_weekly_aligns_to_correct_day(self):
        # Wednesday -> next Monday (weekly:1)
        base = local(2026, 3, 4, 9, 0)  # Wednesday
        nxt = next_trigger(base, "weekly:1", zone=MVD)
        result = wall(nxt)
        assert result.weekday() == 0  # Monday
        assert result.day == 9  # Next Monday

    def test_weekly_thursday_to_friday(self):
        # Thursday -> next Friday (weekly:5)
        base = local(2026, 3, 5, 9, 0)  # Thursday
        nxt = next_trigger(base, "weekly:5", zone=MVD)
        result = wall(nxt)
        assert result.weekday() == 4  # Friday
        assert result.day == 6  # Next day is Friday

    def test_monthly_end_of_year(self):
        base = local(2026, 12, 15, 9, 0)
        nxt = next_trigger(base, "monthly:15", zone=MVD)
        result = wall(nxt)
        assert result.month == 1
        assert result.year == 2027


class TestHouseholdCalendar:
    """T031-U1 follow-up: the household zone, not the process's local time, counts days."""

    def test_daily_keeps_the_household_wall_time_across_utc_midnight(self):
        # 23:30 local is 02:30Z of the next UTC day.
        nxt = next_trigger(local(2026, 11, 11, 23, 30), "daily", zone=MVD)
        assert wall(nxt) == datetime(2026, 11, 12, 23, 30, tzinfo=MVD)
        assert nxt == datetime(2026, 11, 13, 2, 30, tzinfo=UTC).timestamp()

    def test_weekdays_counts_the_household_friday_not_the_utc_saturday(self):
        # Friday 23:30 in Montevideo is already Saturday in UTC.
        nxt = next_trigger(local(2026, 11, 13, 23, 30), "weekdays", zone=MVD)
        assert wall(nxt) == datetime(2026, 11, 16, 23, 30, tzinfo=MVD)

    @pytest.mark.parametrize(("rule", "expected"), [
        ("weekdays", datetime(2026, 3, 9, 8, 0, tzinfo=TOKYO)),
        ("weekly:5", datetime(2026, 3, 13, 8, 0, tzinfo=TOKYO)),
        ("daily", datetime(2026, 3, 7, 8, 0, tzinfo=TOKYO)),
        ("monthly:6", datetime(2026, 4, 6, 8, 0, tzinfo=TOKYO)),
    ])
    def test_the_zone_argument_decides_the_calendar(self, rule, expected):
        # Friday 08:00 in Tokyo is Thursday in Montevideo and in UTC: only the zone decides.
        nxt = next_trigger(local(2026, 3, 6, 8, 0, zone=TOKYO), rule, zone=TOKYO)
        assert wall(nxt, TOKYO) == expected

    def test_zone_is_required(self):
        with pytest.raises(TypeError):
            next_trigger(local(2026, 3, 2, 9, 0), "daily")
