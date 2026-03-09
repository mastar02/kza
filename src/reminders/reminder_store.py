"""SQLite persistence layer for reminders."""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import aiosqlite

logger = logging.getLogger(__name__)


class ReminderState(StrEnum):
    ACTIVE = "active"
    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass
class Reminder:
    """Data transfer object for a single reminder."""

    id: str
    user_id: str
    text: str
    trigger_at: float
    created_at: float
    recurrence: str | None = None
    recurrence_end: float | None = None
    ha_actions: list[dict] | None = None
    state: ReminderState = ReminderState.ACTIVE
    last_fired_at: float | None = None
    fire_count: int = 0


class ReminderStore:
    """Async SQLite store for reminder CRUD operations."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the reminders table and indexes."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                trigger_at REAL NOT NULL,
                created_at REAL NOT NULL,
                recurrence TEXT,
                recurrence_end REAL,
                ha_actions TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                last_fired_at REAL,
                fire_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_trigger
            ON reminders (trigger_at) WHERE state = 'active'
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_reminders_user
            ON reminders (user_id, state)
        """)
        await self._db.commit()
        logger.info("ReminderStore initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def create(
        self,
        user_id: str,
        text: str,
        trigger_at: float,
        recurrence: str | None = None,
        recurrence_end: float | None = None,
        ha_actions: list[dict] | None = None,
    ) -> Reminder:
        """Create a new reminder and return it."""
        reminder = Reminder(
            id=uuid.uuid4().hex[:12],
            user_id=user_id,
            text=text,
            trigger_at=trigger_at,
            created_at=time.time(),
            recurrence=recurrence,
            recurrence_end=recurrence_end,
            ha_actions=ha_actions,
        )
        ha_json = json.dumps(ha_actions) if ha_actions else None
        await self._db.execute(
            """
            INSERT INTO reminders
                (id, user_id, text, trigger_at, created_at, recurrence,
                 recurrence_end, ha_actions, state, last_fired_at, fire_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder.id,
                reminder.user_id,
                reminder.text,
                reminder.trigger_at,
                reminder.created_at,
                reminder.recurrence,
                reminder.recurrence_end,
                ha_json,
                reminder.state,
                reminder.last_fired_at,
                reminder.fire_count,
            ),
        )
        await self._db.commit()
        logger.debug("Created reminder %s for user %s", reminder.id, user_id)
        return reminder

    @staticmethod
    def _row_to_reminder(row: aiosqlite.Row) -> Reminder:
        """Convert a database row to a Reminder dataclass."""
        ha_actions = json.loads(row["ha_actions"]) if row["ha_actions"] else None
        return Reminder(
            id=row["id"],
            user_id=row["user_id"],
            text=row["text"],
            trigger_at=row["trigger_at"],
            created_at=row["created_at"],
            recurrence=row["recurrence"],
            recurrence_end=row["recurrence_end"],
            ha_actions=ha_actions,
            state=ReminderState(row["state"]),
            last_fired_at=row["last_fired_at"],
            fire_count=row["fire_count"],
        )

    async def get_by_id(self, reminder_id: str) -> Reminder | None:
        """Fetch a single reminder by ID."""
        async with self._db.execute(
            "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_reminder(row) if row else None

    async def get_active_for_user(self, user_id: str) -> list[Reminder]:
        """Return all active reminders for a user."""
        async with self._db.execute(
            "SELECT * FROM reminders WHERE user_id = ? AND state = 'active' ORDER BY trigger_at ASC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_reminder(r) for r in rows]

    async def get_next_pending(self) -> Reminder | None:
        """Return the next active reminder ordered by trigger_at ASC."""
        async with self._db.execute(
            "SELECT * FROM reminders WHERE state = 'active' ORDER BY trigger_at ASC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_reminder(row) if row else None

    async def get_due(self, now: float) -> list[Reminder]:
        """Return all active reminders with trigger_at <= now."""
        async with self._db.execute(
            "SELECT * FROM reminders WHERE state = 'active' AND trigger_at <= ? ORDER BY trigger_at ASC",
            (now,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_reminder(r) for r in rows]

    async def mark_fired(self, reminder_id: str) -> None:
        """Mark a reminder as fired, increment fire_count."""
        now = time.time()
        await self._db.execute(
            "UPDATE reminders SET state = ?, last_fired_at = ?, fire_count = fire_count + 1 WHERE id = ?",
            (ReminderState.FIRED, now, reminder_id),
        )
        await self._db.commit()
        logger.debug("Marked reminder %s as fired", reminder_id)

    async def cancel(self, reminder_id: str) -> None:
        """Cancel a reminder."""
        await self._db.execute(
            "UPDATE reminders SET state = ? WHERE id = ?",
            (ReminderState.CANCELLED, reminder_id),
        )
        await self._db.commit()
        logger.debug("Cancelled reminder %s", reminder_id)

    async def update_trigger(self, reminder_id: str, new_trigger: float) -> None:
        """Update trigger_at and reset state to active."""
        await self._db.execute(
            "UPDATE reminders SET trigger_at = ?, state = ? WHERE id = ?",
            (new_trigger, ReminderState.ACTIVE, reminder_id),
        )
        await self._db.commit()
        logger.debug("Updated trigger for reminder %s to %s", reminder_id, new_trigger)

    async def get_overdue_for_user(self, user_id: str, now: float) -> list[Reminder]:
        """Return active reminders whose trigger_at has already passed for a user.

        These are reminders that were due but could not be delivered
        (e.g. user was away). Useful for missed-reminder-on-arrival.

        Args:
            user_id: The user to query.
            now: Current timestamp.

        Returns:
            List of overdue Reminder objects ordered by trigger_at.
        """
        async with self._db.execute(
            "SELECT * FROM reminders WHERE user_id = ? AND state = 'active' "
            "AND trigger_at <= ? ORDER BY trigger_at ASC",
            (user_id, now),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_reminder(r) for r in rows]

    async def cleanup_old(self, days: int = 30) -> int:
        """Delete old fired/cancelled reminders without recurrence.

        Returns the number of deleted rows.
        """
        cutoff = time.time() - (days * 86400)
        async with self._db.execute(
            """
            DELETE FROM reminders
            WHERE state IN ('fired', 'cancelled')
              AND recurrence IS NULL
              AND trigger_at < ?
            """,
            (cutoff,),
        ) as cursor:
            deleted = cursor.rowcount
        await self._db.commit()
        logger.info("Cleaned up %d old reminders (older than %d days)", deleted, days)
        return deleted
