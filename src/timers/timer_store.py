"""SCHEMA-V2 timers and their overdue notices on the reminder SQLite owner."""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.core.sqlite_work import SQLiteWork
from src.reminders.reminder_store import NOTICE_ID_PREFIX, ReminderActionStatus, ReminderStore, TimerState

RETENTION_S = 48 * 3600
_NOTICE_PREFIX = NOTICE_ID_PREFIX + "timer:"


class TimerConflict(RuntimeError):
    """The timer's persisted revision no longer matches the caller's."""


@dataclass(frozen=True)
class TimerRecord:
    """The SCHEMA-V2 representation of a named timer."""

    id: str
    label: str
    actor: str
    delivery_room: str | None
    state: TimerState
    duration_s: float
    ends_at: float | None
    remaining_s: float | None
    created_at: float
    updated_at: float
    revision: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TimerRecord":
        """Project an owned SQLite row into a detached record."""
        values = dict(row)
        values["state"] = TimerState(values["state"])
        return cls(**values)


class TimerStore:
    """Use the initialized ReminderStore's transaction and connection lifecycle."""

    def __init__(self, reminders: ReminderStore) -> None:
        self._reminders = reminders

    @property
    def _work(self) -> SQLiteWork:
        work = self._reminders._work
        if work is None:
            raise RuntimeError("timer_store_unavailable")
        return work

    @staticmethod
    def _update(db: sqlite3.Connection, timer: TimerRecord) -> None:
        changed = db.execute(
            "UPDATE timers SET state = ?, ends_at = ?, remaining_s = ?, updated_at = ?, revision = revision + 1 "
            "WHERE id = ? AND revision = ? AND state IN ('running', 'paused')",
            (timer.state, timer.ends_at, timer.remaining_s, timer.updated_at, timer.id, timer.revision),
        ).rowcount
        if changed != 1:
            raise TimerConflict("timer_revision_conflict")

    async def save(self, timers: Sequence[TimerRecord]) -> None:
        """Insert new timers and compare-and-swap existing ones in a single commit.

        A new timer carries revision 0 from the caller and starts at revision 1
        in storage. An existing timer carries its expected revision.
        """
        def write(db: sqlite3.Connection) -> None:
            for timer in timers:
                if timer.revision:
                    self._update(db, timer)
                    continue
                db.execute(
                    "INSERT INTO timers (id, label, actor, delivery_room, state, duration_s, ends_at, "
                    "remaining_s, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (timer.id, timer.label, timer.actor, timer.delivery_room, timer.state, timer.duration_s,
                     timer.ends_at, timer.remaining_s, timer.created_at, timer.updated_at),
                )
        await self._work.write(write)

    async def get(self, timer_id: str) -> TimerRecord | None:
        """Read a timer by its stable id, including a terminal one."""
        def read(db: sqlite3.Connection) -> TimerRecord | None:
            row = db.execute("SELECT * FROM timers WHERE id = ?", (timer_id,)).fetchone()
            return TimerRecord.from_row(row) if row is not None else None
        return await self._work.read(read)

    async def recover(self, now: float) -> list[TimerRecord]:
        """Complete every overdue running timer with one persistent pending notice.

        The state change and the notice insert share a transaction, so neither
        can survive alone. The timer id identifies this one-shot episode.
        """
        def write(db: sqlite3.Connection) -> list[TimerRecord]:
            active = []
            for row in db.execute("SELECT * FROM timers WHERE state IN ('running', 'paused') "
                                  "ORDER BY created_at, id").fetchall():
                timer = TimerRecord.from_row(row)
                if timer.state is TimerState.PAUSED or timer.ends_at > now:
                    active.append(timer)
                    continue
                self._update(db, replace(timer, state=TimerState.COMPLETED, remaining_s=0, updated_at=now))
                # 'failed' is the ledger's retryable state (no output started).
                # Timer-specific bodies are owned by TimerStore's retention.
                body = json.dumps({"kind": "timer_expired", "timer_id": timer.id, "trigger_at": timer.ends_at,
                                   "room": timer.delivery_room}, separators=(",", ":"))
                db.execute("INSERT INTO reminder_home_actions (request_id, body, status) VALUES (?, ?, 'failed')",
                           (_NOTICE_PREFIX + timer.id, body))
            self._cleanup(db, now)
            return active
        return await self._work.write(write)

    async def pending_notices(self) -> list[TimerRecord]:
        """Return the overdue timers whose single notice is pending delivery."""
        return await self._work.read(lambda db: [
            TimerRecord.from_row(row) for row in db.execute(
                "SELECT t.* FROM reminder_home_actions AS n JOIN timers AS t ON n.request_id = ? || t.id "
                "WHERE n.status = 'failed' ORDER BY t.updated_at, t.id", (_NOTICE_PREFIX,))
        ])

    async def claim_notice(self, timer_id: str) -> bool:
        """Own delivery of an existing pending notice before submitting audio."""
        return await self._work.write(lambda db: db.execute(
            "UPDATE reminder_home_actions SET status = 'sending' WHERE request_id = ? AND status = 'failed'",
            (_NOTICE_PREFIX + timer_id,)).rowcount == 1)

    async def finish_notice(self, timer_id: str, status: ReminderActionStatus, reason: str | None) -> None:
        """Settle the claim through the reminder ledger's existing receipt API."""
        await self._reminders.finish_home_action(_NOTICE_PREFIX + timer_id, status, reason)

    @staticmethod
    def _cleanup(db: sqlite3.Connection, now: float) -> list[str]:
        rows = db.execute(
            "SELECT t.id FROM timers AS t LEFT JOIN reminder_home_actions AS n ON n.request_id = ? || t.id "
            "WHERE t.state IN ('completed', 'cancelled') AND t.updated_at < ? "
            "AND (n.status IS NULL OR n.status IN ('confirmed', 'uncertain'))",
            (_NOTICE_PREFIX, now - RETENTION_S),
        ).fetchall()
        ids = [row["id"] for row in rows]
        db.executemany("DELETE FROM timers WHERE id = ?", [(timer_id,) for timer_id in ids])
        db.executemany("DELETE FROM reminder_home_actions WHERE request_id = ?",
                       [(_NOTICE_PREFIX + timer_id,) for timer_id in ids])
        return ids

    async def cleanup(self, now: float) -> list[str]:
        """Prune old terminal timers and their settled notice receipts."""
        return await self._work.write(lambda db: self._cleanup(db, now))
