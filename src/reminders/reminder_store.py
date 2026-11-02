"""SQLite persistence layer for reminders."""
import asyncio
from collections.abc import Callable
from contextvars import Context
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

import aiosqlite

from src.core.owned_work import ClosePhase, WorkCloseTimeout
from src.core.sqlite_work import AuthorityGuard, SQLiteWork

logger = logging.getLogger(__name__)


class ReminderState(StrEnum):
    ACTIVE = "active"
    FIRED = "fired"
    CANCELLED = "cancelled"


class ReminderActionStatus(StrEnum):
    """Durable send evidence for one reminder occurrence and action."""

    SENDING = "sending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ReminderHomeActionReceipt:
    """Evidence retained independently of a reminder's delivery state."""

    request_id: str
    status: ReminderActionStatus
    reason: str | None


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

    def __init__(self, db_path: str, *, on_created: Callable[[], None] | None = None):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._on_created = on_created
        self._work: SQLiteWork | None = None
        self._closing: asyncio.Task | None = None
        self._lifecycle = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Prepare schema and interrupted-send recovery before admitting consumers."""
        if self._closing is not None and not self._closing.done():
            raise RuntimeError("reminder_store_closing")
        async with self._lifecycle:
            if self._db is not None:
                if self._initialized:
                    return
                raise RuntimeError("reminder_store_initialization_incomplete")
            if self._closing is not None and not self._closing.done():
                raise RuntimeError("reminder_store_closing")
            self._closing = None
            self._db = await aiosqlite.connect(self._db_path)
            if self._closing is not None:
                raise RuntimeError("reminder_store_closing")
            self._db.row_factory = aiosqlite.Row
            self._work = SQLiteWork(self._db)

            def schema(db):
                db.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                        trigger_at REAL NOT NULL, created_at REAL NOT NULL,
                        recurrence TEXT, recurrence_end REAL, ha_actions TEXT,
                        state TEXT NOT NULL DEFAULT 'active', last_fired_at REAL,
                        fire_count INTEGER NOT NULL DEFAULT 0
                    )
                """)
                db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_trigger "
                           "ON reminders (trigger_at) WHERE state = 'active'")
                db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders (user_id, state)")
                db.execute("""
                    CREATE TABLE IF NOT EXISTS reminder_home_actions (
                        request_id TEXT PRIMARY KEY, body TEXT NOT NULL,
                        status TEXT NOT NULL, reason TEXT
                    )
                """)
                db.execute("UPDATE reminder_home_actions SET status = 'uncertain', "
                           "reason = 'interrupted_send' WHERE status = 'sending'")
            await self._work.write(schema)
            self._initialized = True
            logger.info("ReminderStore initialized at %s", self._db_path)

    async def claim_home_action(self, request_id: str, action: dict, *,
                                guard: AuthorityGuard | None = None) -> bool:
        """Commit one immutable occurrence body and its claim as one native operation."""
        body = json.dumps(action, sort_keys=True, separators=(",", ":"), allow_nan=False)
        def operation(db):
            created = db.execute(
                "INSERT OR IGNORE INTO reminder_home_actions (request_id, body, status) VALUES (?, ?, 'sending')",
                (request_id, body)).rowcount == 1
            previous = db.execute("SELECT body FROM reminder_home_actions WHERE request_id = ?",
                                  (request_id,)).fetchone()
            if previous is None or previous["body"] != body:
                raise ValueError("reminder_action_body_conflict")
            return created
        return await self._work.write(operation, guard=guard)

    async def finish_home_action(self, request_id: str, status: str, reason: str | None, *,
                                 guard: AuthorityGuard | None = None) -> None:
        """Commit terminal evidence without making an uncertain send retryable."""
        status = ReminderActionStatus(status)
        if status is ReminderActionStatus.SENDING:
            raise ValueError("invalid_home_action_status")
        def operation(db):
            db.execute("UPDATE reminder_home_actions SET status = ?, reason = ? "
                "WHERE request_id = ? AND status = 'sending'", (status.value, reason, request_id))
        await self._work.write(operation, guard=guard)

    async def get_home_action_receipt(self, request_id: str, *,
                                      guard: AuthorityGuard | None = None) -> ReminderHomeActionReceipt | None:
        """Read detached execution evidence; absence cannot authorize replay."""
        def operation(db):
            row = db.execute("SELECT request_id, status, reason FROM reminder_home_actions WHERE request_id = ?",
                             (request_id,)).fetchone()
            return ReminderHomeActionReceipt(row["request_id"], ReminderActionStatus(row["status"]),
                row["reason"]) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def close(self) -> None:
        """Retain native drain and captured connection close through observer cancellation."""
        if self._closing is None:
            if self._work is not None:
                self._work.begin_close()

            async def finish():
                async with self._lifecycle:
                    work, database = self._work, self._db
                    if work is not None:
                        await work.begin_close()
                    if database is not None:
                        await database.close()
                    if self._db is database:
                        self._db = None
                        self._initialized = False
                    if self._work is work:
                        self._work = None

            self._closing = asyncio.create_task(finish(), context=Context())
            def observed(done):
                if not done.cancelled():
                    done.exception()
            self._closing.add_done_callback(observed, context=Context())
        closing = self._closing
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(closing)
        except TimeoutError as error:
            raise WorkCloseTimeout("reminder_store_close_pending", phase=ClosePhase.DRAIN,
                diagnostics=self._work.diagnostics() if self._work is not None else {}) from error

    async def create(
        self, user_id: str, text: str, trigger_at: float, recurrence: str | None = None,
        recurrence_end: float | None = None, ha_actions: list[dict] | None = None, *,
        guard: AuthorityGuard | None = None,
    ) -> Reminder:
        """Own the native insert and publish its wake after commit, even without its observer."""
        reminder = Reminder(id=uuid.uuid4().hex[:12], user_id=user_id, text=text,
            trigger_at=trigger_at, created_at=time.time(), recurrence=recurrence,
            recurrence_end=recurrence_end, ha_actions=ha_actions)
        ha_json = json.dumps(ha_actions) if ha_actions else None
        def operation(db):
            db.execute("""
                INSERT INTO reminders
                    (id, user_id, text, trigger_at, created_at, recurrence,
                     recurrence_end, ha_actions, state, last_fired_at, fire_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (reminder.id, reminder.user_id, reminder.text, reminder.trigger_at,
                  reminder.created_at, reminder.recurrence, reminder.recurrence_end, ha_json,
                  reminder.state.value, reminder.last_fired_at, reminder.fire_count))
            return reminder
        def publish(_reminder):
            if self._on_created is not None:
                try:
                    self._on_created()
                except Exception:
                    # The committed row remains active. This receipt describes
                    # stored software state, never future speech delivery.
                    logger.error("reminder_scheduler_wake_failed")
        return await self._work.write(operation, guard=guard, publish=publish)

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

    async def get_by_id(self, reminder_id: str, *, guard: AuthorityGuard | None = None) -> Reminder | None:
        """Read the complete reminder without a cursor crossing await."""
        def operation(db):
            row = db.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
            return self._row_to_reminder(row) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def get_active_for_user(self, user_id: str, *,
                                  guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Project active reminders in one retained native callback."""
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE user_id = ? AND state = 'active' "
                              "ORDER BY trigger_at ASC", (user_id,)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def get_next_pending(self, *, guard: AuthorityGuard | None = None) -> Reminder | None:
        """Project the next durable job without retaining a live cursor."""
        def operation(db):
            row = db.execute("SELECT * FROM reminders WHERE state = 'active' ORDER BY trigger_at ASC LIMIT 1").fetchone()
            return self._row_to_reminder(row) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def get_due(self, now: float, *, guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Project due durable jobs as a detached batch."""
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE state = 'active' AND trigger_at <= ? "
                              "ORDER BY trigger_at ASC", (now,)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def mark_fired(self, reminder_id: str, *, guard: AuthorityGuard | None = None) -> None:
        """Commit software delivery state and its fire count atomically."""
        now = time.time()
        def operation(db):
            db.execute("UPDATE reminders SET state = ?, last_fired_at = ?, fire_count = fire_count + 1 WHERE id = ?",
                       (ReminderState.FIRED.value, now, reminder_id))
        await self._work.write(operation, guard=guard)

    async def cancel(self, reminder_id: str, *, guard: AuthorityGuard | None = None) -> bool:
        """Commit cancellation only while its original authority remains current."""
        def operation(db):
            return db.execute("UPDATE reminders SET state = ? WHERE id = ?",
                              (ReminderState.CANCELLED.value, reminder_id)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    async def update_trigger(self, reminder_id: str, new_trigger: float, *,
                             guard: AuthorityGuard | None = None) -> None:
        """Commit the next durable recurrence without inheriting its creator's lease."""
        def operation(db):
            db.execute("UPDATE reminders SET trigger_at = ?, state = ? WHERE id = ?",
                       (new_trigger, ReminderState.ACTIVE.value, reminder_id))
        await self._work.write(operation, guard=guard)

    async def get_overdue_for_user(self, user_id: str, now: float, *,
                                   guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Project overdue active reminders in one native read."""
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE user_id = ? AND state = 'active' "
                              "AND trigger_at <= ? ORDER BY trigger_at ASC", (user_id, now)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def cleanup_old(self, days: int = 30, *, guard: AuthorityGuard | None = None) -> int:
        """Commit deletion of old terminal reminders without recurrence."""
        cutoff = time.time() - days * 86400
        def operation(db):
            return db.execute("DELETE FROM reminders WHERE state IN ('fired', 'cancelled') "
                "AND recurrence IS NULL AND trigger_at < ?", (cutoff,)).rowcount
        return await self._work.write(operation, guard=guard)
