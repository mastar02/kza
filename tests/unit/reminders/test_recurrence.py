"""Tests for recurrence engine — calculating next trigger dates."""
import pytest
from datetime import datetime

from src.reminders.recurrence import next_trigger, parse_recurrence, RecurrenceType


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
        base = datetime(2026, 3, 2, 9, 0).timestamp()  # Monday
        nxt = next_trigger(base, "daily")
        result = datetime.fromtimestamp(nxt)
        assert result.day == 3
        assert result.hour == 9

    def test_weekdays_friday_to_monday(self):
        base = datetime(2026, 3, 6, 9, 0).timestamp()  # Friday
        nxt = next_trigger(base, "weekdays")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 0  # Monday
        assert result.day == 9

    def test_weekdays_monday_to_tuesday(self):
        base = datetime(2026, 3, 2, 9, 0).timestamp()  # Monday
        nxt = next_trigger(base, "weekdays")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 1  # Tuesday

    def test_weekly_next_week(self):
        base = datetime(2026, 3, 2, 9, 0).timestamp()
        nxt = next_trigger(base, "weekly:1")
        result = datetime.fromtimestamp(nxt)
        assert result.day == 9
        assert result.weekday() == 0

    def test_monthly(self):
        base = datetime(2026, 3, 15, 9, 0).timestamp()
        nxt = next_trigger(base, "monthly:15")
        result = datetime.fromtimestamp(nxt)
        assert result.month == 4
        assert result.day == 15

    def test_weekly_aligns_to_correct_day(self):
        # Wednesday -> next Monday (weekly:1)
        base = datetime(2026, 3, 4, 9, 0).timestamp()  # Wednesday
        nxt = next_trigger(base, "weekly:1")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 0  # Monday
        assert result.day == 9  # Next Monday

    def test_weekly_thursday_to_friday(self):
        # Thursday -> next Friday (weekly:5)
        base = datetime(2026, 3, 5, 9, 0).timestamp()  # Thursday
        nxt = next_trigger(base, "weekly:5")
        result = datetime.fromtimestamp(nxt)
        assert result.weekday() == 4  # Friday
        assert result.day == 6  # Next day is Friday

    def test_monthly_end_of_year(self):
        base = datetime(2026, 12, 15, 9, 0).timestamp()
        nxt = next_trigger(base, "monthly:15")
        result = datetime.fromtimestamp(nxt)
        assert result.month == 1
        assert result.year == 2027
