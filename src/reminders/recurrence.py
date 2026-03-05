"""Recurrence engine — calculates next trigger dates."""

from datetime import datetime, timedelta
from enum import StrEnum


class RecurrenceType(StrEnum):
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


def parse_recurrence(recurrence: str) -> tuple[RecurrenceType, int | None]:
    """Parse recurrence string into (type, param). Raises ValueError if invalid."""
    if recurrence == "daily":
        return RecurrenceType.DAILY, None
    if recurrence == "weekdays":
        return RecurrenceType.WEEKDAYS, None
    if recurrence.startswith("weekly:"):
        day = int(recurrence.split(":")[1])
        return RecurrenceType.WEEKLY, day
    if recurrence.startswith("monthly:"):
        day = int(recurrence.split(":")[1])
        return RecurrenceType.MONTHLY, day
    raise ValueError(f"Invalid recurrence format: {recurrence}")


def next_trigger(last_trigger: float, recurrence: str) -> float:
    """Calculate next trigger time. Returns epoch timestamp."""
    rec_type, param = parse_recurrence(recurrence)
    dt = datetime.fromtimestamp(last_trigger)

    if rec_type == RecurrenceType.DAILY:
        nxt = dt + timedelta(days=1)
    elif rec_type == RecurrenceType.WEEKDAYS:
        nxt = dt + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
    elif rec_type == RecurrenceType.WEEKLY:
        nxt = dt + timedelta(days=7)
    elif rec_type == RecurrenceType.MONTHLY:
        month = dt.month + 1
        year = dt.year
        if month > 12:
            month = 1
            year += 1
        day = min(param, 28)
        nxt = dt.replace(year=year, month=month, day=day)
    else:
        raise ValueError(f"Unknown recurrence type: {rec_type}")

    return nxt.timestamp()
