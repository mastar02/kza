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
from src.memory.output_registration import (
    DestinationRegistration, before_store_open, store_open_returned, store_close_returned,
)

logger = logging.getLogger(__name__)


class ReminderScope(StrEnum):
    PRIVATE = "private"
    SHARED = "shared"


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


@dataclass(frozen=True)
class ReminderLedgerRow:
    """One raw ledger row with its immutable claim body (JSON text)."""

    request_id: str
    status: ReminderActionStatus
    reason: str | None
    body: str


# Voice-notice receipts share the action ledger; this half-open PK range holds them all.
NOTICE_ID_PREFIX = "notice:"
NOTICE_ID_RANGE_END = "notice;"
MAX_LEDGER_PROBE = 10000


# A reminder in scope: shared, or (outside shared_only) owned by ``owner`` when
# one is given. Parameters: (shared_only, owner, owner). ``owner=None`` without
# shared_only keeps the unscoped internal view (T035-U1).
_REMINDER_SCOPE = "(scope = 'shared' OR (? = 0 AND (? IS NULL OR user_id = ?)))"


def _ledger_row(row) -> ReminderLedgerRow:
    return ReminderLedgerRow(row["request_id"], ReminderActionStatus(row["status"]), row["reason"], row["body"])


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
    scope: ReminderScope = ReminderScope.PRIVATE


def reminder_schema(db) -> None:
    """Create or migrate the reminder tables: DDL only, no row changes (U3b-0e2 S1).

    ReminderStore.initialize runs it before its interrupted-send recovery; the
    adoption's schema-only child runs it alone on the adoption copy. The SQL
    text is kept byte-identical: sqlite_master stores it verbatim.
    """
    db.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                        trigger_at REAL NOT NULL, created_at REAL NOT NULL,
                        recurrence TEXT, recurrence_end REAL, ha_actions TEXT,
                        state TEXT NOT NULL DEFAULT 'active', last_fired_at REAL,
                        fire_count INTEGER NOT NULL DEFAULT 0,
                        scope TEXT NOT NULL DEFAULT 'private' CHECK (scope IN ('private', 'shared'))
                    )
                """)
    columns = {row[1] for row in db.execute("PRAGMA table_info(reminders)").fetchall()}
    if "scope" not in columns:
        db.execute("ALTER TABLE reminders ADD COLUMN scope TEXT NOT NULL DEFAULT 'private' "
                   "CHECK (scope IN ('private', 'shared'))")
    db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_trigger "
               "ON reminders (trigger_at) WHERE state = 'active'")
    db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders (user_id, state)")
    db.execute("""
                    CREATE TABLE IF NOT EXISTS reminder_home_actions (
                        request_id TEXT PRIMARY KEY, body TEXT NOT NULL,
                        status TEXT NOT NULL, reason TEXT
                    )
                """)


class ReminderStore:
    """Async SQLite store for reminder CRUD operations."""

    def __init__(self, db_path: str, *, on_created: Callable[[], None] | None = None,
                 registration: "DestinationRegistration | None" = None,
                 verify_only: bool = False) -> None:
        self._db_path = db_path
        # SC-4a skip list d-reminder-sending: a verification run leaves 'sending' as it is.
        self._verify_only = verify_only is True
        self._destination_registration = registration
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
            await before_store_open(self, "reminders", self._db_path, self._destination_registration)
            self._db = await aiosqlite.connect(self._db_path)
            if self._closing is not None:
                raise RuntimeError("reminder_store_closing")
            self._db.row_factory = aiosqlite.Row
            self._work = SQLiteWork(self._db)

            def schema(db):
                reminder_schema(db)
                if not self._verify_only:
                    db.execute("UPDATE reminder_home_actions SET status = 'uncertain', "
                               "reason = 'interrupted_send' WHERE status = 'sending'")
            await self._work.write(schema)
            await store_open_returned(self)
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

    async def probe_ledger(self, request_id_for: Callable[[int], str], *, guard: AuthorityGuard | None = None,
                           limit: int = MAX_LEDGER_PROBE) -> list[ReminderLedgerRow]:
        """Read the rows of ids 1, 2, 3, ... up to the first missing one, in one read.

        Args:
            request_id_for: Pure function from attempt number to request_id.
            guard: Optional authority guard of the caller.
            limit: Upper bound on rows; reaching it raises instead of scanning further.

        Returns:
            The consecutive existing rows, attempt 1 first.
        """
        def operation(db):
            rows = []
            for attempt in range(1, limit + 1):
                row = db.execute("SELECT request_id, body, status, reason FROM reminder_home_actions "
                                 "WHERE request_id = ?", (request_id_for(attempt),)).fetchone()
                if row is None:
                    return rows
                rows.append(_ledger_row(row))
            raise ValueError("reminder_ledger_probe_unbounded")
        return await self._work.read(operation, guard=guard)

    async def get_notice_rows(self, *, guard: AuthorityGuard | None = None) -> list[ReminderLedgerRow]:
        """Read every voice-notice row (a primary-key range scan), oldest id first."""
        def operation(db):
            return [_ledger_row(row) for row in db.execute(
                "SELECT request_id, body, status, reason FROM reminder_home_actions "
                "WHERE request_id >= ? AND request_id < ? ORDER BY request_id",
                (NOTICE_ID_PREFIX, NOTICE_ID_RANGE_END)).fetchall()]
        return await self._work.read(operation, guard=guard)

    async def purge_notice_rows(self, claimed_before: float, *, reminder_id: str | None = None,
                                trigger_at: float | None = None,
                                guard: AuthorityGuard | None = None) -> int:
        """Delete the notice rows of finished occurrences whose newest claim is before a cutoff.

        Retention works per occurrence (reminder id, trigger_at), so a kept
        history entry never loses its earlier attempts: all of an occurrence's
        rows go together, once every one of them was claimed before the cutoff.
        Selection uses status only ('sending' rows are never deleted, and an
        occurrence with one keeps all its rows) plus the immutable body; the
        diagnostic reason is never read. An occurrence is kept while an active
        reminder still has its id and trigger_at, because its attempts still
        count. Rows with an unreadable body are kept.

        Args:
            claimed_before: Epoch seconds; only occurrences whose every row was claimed earlier are deleted.
            reminder_id: Restrict to one occurrence (with trigger_at), for a web purge.
            trigger_at: The occurrence's scheduled time when reminder_id is given.
            guard: Optional authority guard of the caller.

        Returns:
            Number of deleted rows.
        """
        def operation(db):
            occurrences: dict[tuple[str, float], list] = {}
            for row in db.execute("SELECT request_id, body, status FROM reminder_home_actions "
                                  "WHERE request_id >= ? AND request_id < ?",
                                  (NOTICE_ID_PREFIX, NOTICE_ID_RANGE_END)).fetchall():
                try:
                    body = json.loads(row["body"])
                except ValueError:
                    continue
                if type(body) is not dict:
                    continue
                identity, trigger, claimed = body.get("reminder_id"), body.get("trigger_at"), body.get("claimed_at")
                if type(identity) is not str or type(trigger) not in (int, float):
                    continue
                if reminder_id is not None and (identity != reminder_id or trigger != trigger_at):
                    continue
                occurrences.setdefault((identity, trigger), []).append((row["request_id"], row["status"], claimed))
            doomed = []
            for (identity, trigger), rows in occurrences.items():
                if any(status == "sending" or type(claimed) not in (int, float) or not claimed < claimed_before
                       for _, status, claimed in rows):
                    continue
                if db.execute("SELECT 1 FROM reminders WHERE id = ? AND trigger_at = ? AND state = 'active'",
                              (identity, trigger)).fetchone() is not None:
                    continue
                doomed.extend(request_id for request_id, _, _ in rows)
            for request_id in doomed:
                db.execute("DELETE FROM reminder_home_actions WHERE request_id = ? AND status != 'sending'",
                           (request_id,))
            return len(doomed)
        return await self._work.write(operation, guard=guard)

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
                    if database is not None:
                        await store_close_returned(self)

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
        guard: AuthorityGuard | None = None, scope: ReminderScope = ReminderScope.PRIVATE,
    ) -> Reminder:
        """Own the native insert and publish its wake after commit, even without its observer."""
        reminder = Reminder(id=uuid.uuid4().hex[:12], user_id=user_id, text=text,
            trigger_at=trigger_at, created_at=time.time(), recurrence=recurrence,
            recurrence_end=recurrence_end, ha_actions=ha_actions, scope=ReminderScope(scope))
        ha_json = json.dumps(ha_actions) if ha_actions else None
        def operation(db):
            db.execute("""
                INSERT INTO reminders
                    (id, user_id, text, trigger_at, created_at, recurrence,
                     recurrence_end, ha_actions, state, last_fired_at, fire_count, scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (reminder.id, reminder.user_id, reminder.text, reminder.trigger_at,
                  reminder.created_at, reminder.recurrence, reminder.recurrence_end, ha_json,
                  reminder.state.value, reminder.last_fired_at, reminder.fire_count, reminder.scope.value))
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
            scope=ReminderScope(row["scope"]),
        )

    async def get_by_id(self, reminder_id: str, *, guard: AuthorityGuard | None = None,
                        shared_only: bool = False, owner: str | None = None) -> Reminder | None:
        """Resolve current scope before projecting any reminder material."""
        def operation(db):
            row = db.execute(f"SELECT * FROM reminders WHERE id = ? AND {_REMINDER_SCOPE}",
                             (reminder_id, shared_only, owner, owner)).fetchone()
            return self._row_to_reminder(row) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def get_active_for_user(self, user_id: str | None, *,
                                  guard: AuthorityGuard | None = None,
                                  shared_only: bool = False, include_shared: bool = False) -> list[Reminder]:
        """Select shared household jobs or a trusted user's private view before DTOs.

        ``include_shared`` adds every shared job to the user's own view (T035-U1).
        """
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE state = 'active' "
                "AND ((scope = 'shared' AND (? = 1 OR ? = 1)) OR (? = 0 AND user_id = ?)) "
                "ORDER BY trigger_at ASC", (shared_only, include_shared, shared_only, user_id)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def count_active(self, *, guard: AuthorityGuard | None = None,
                           shared_only: bool = False) -> int:
        """Count eligible active jobs within the same native read owner."""
        def operation(db):
            return db.execute("SELECT COUNT(*) FROM reminders WHERE state = 'active' "
                "AND (? = 0 OR scope = 'shared')", (shared_only,)).fetchone()[0]
        return await self._work.read(operation, guard=guard)

    async def get_next_pending(self, *, guard: AuthorityGuard | None = None,
                               shared_only: bool = False) -> Reminder | None:
        """Project the next eligible durable job without a live cursor."""
        def operation(db):
            row = db.execute("SELECT * FROM reminders WHERE state = 'active' "
                "AND (? = 0 OR scope = 'shared') ORDER BY trigger_at ASC LIMIT 1",
                (shared_only,)).fetchone()
            return self._row_to_reminder(row) if row is not None else None
        return await self._work.read(operation, guard=guard)

    async def get_due(self, now: float, *, guard: AuthorityGuard | None = None,
                      shared_only: bool = False) -> list[Reminder]:
        """Project only eligible due jobs as one detached batch."""
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE state = 'active' AND trigger_at <= ? "
                "AND (? = 0 OR scope = 'shared') ORDER BY trigger_at ASC", (now, shared_only)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def mark_fired(self, reminder_id: str, *, guard: AuthorityGuard | None = None,
                         shared_only: bool = False, expected_trigger_at: float | None = None) -> bool:
        """Commit delivery only for the selected active occurrence when specified."""
        now = time.time()
        def operation(db):
            return db.execute("UPDATE reminders SET state = ?, last_fired_at = ?, fire_count = fire_count + 1 "
                "WHERE id = ? AND (? = 0 OR scope = 'shared') "
                "AND (? IS NULL OR (trigger_at = ? AND state = 'active'))",
                (ReminderState.FIRED.value, now, reminder_id, shared_only,
                 expected_trigger_at, expected_trigger_at)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    async def cancel(self, reminder_id: str, *, guard: AuthorityGuard | None = None,
                     shared_only: bool = False, owner: str | None = None) -> bool:
        """Commit cancellation only under original authority and current scope.

        Only an active reminder transitions; a fired or already cancelled one
        returns False, so no caller reports a cancellation that did not happen.
        """
        def operation(db):
            return db.execute("UPDATE reminders SET state = ? WHERE id = ? AND state = 'active' "
                f"AND {_REMINDER_SCOPE}",
                (ReminderState.CANCELLED.value, reminder_id, shared_only, owner, owner)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    async def update_trigger(self, reminder_id: str, new_trigger: float, *,
                             guard: AuthorityGuard | None = None, shared_only: bool = False,
                             expected_trigger_at: float | None = None) -> bool:
        """Advance only the same active occurrence without overwriting a newer one."""
        def operation(db):
            return db.execute("UPDATE reminders SET trigger_at = ?, state = ? WHERE id = ? "
                "AND (? = 0 OR scope = 'shared') "
                "AND (? IS NULL OR (trigger_at = ? AND state = 'active'))",
                (new_trigger, ReminderState.ACTIVE.value, reminder_id, shared_only,
                 expected_trigger_at, expected_trigger_at)).rowcount > 0
        return await self._work.write(operation, guard=guard)

    async def get_overdue_for_user(self, user_id: str | None, now: float, *,
                                   guard: AuthorityGuard | None = None,
                                   shared_only: bool = False) -> list[Reminder]:
        """Filter shared or trusted user jobs before exposing overdue text."""
        def operation(db):
            rows = db.execute("SELECT * FROM reminders WHERE state = 'active' AND trigger_at <= ? "
                "AND ((? = 1 AND scope = 'shared') OR (? = 0 AND user_id = ?)) "
                "ORDER BY trigger_at ASC", (now, shared_only, shared_only, user_id)).fetchall()
            return [self._row_to_reminder(row) for row in rows]
        return await self._work.read(operation, guard=guard)

    async def cleanup_old(self, days: int = 30, *, guard: AuthorityGuard | None = None) -> int:
        """Commit deletion of old terminal reminders without recurrence."""
        cutoff = time.time() - days * 86400
        def operation(db):
            return db.execute("DELETE FROM reminders WHERE state IN ('fired', 'cancelled') "
                "AND recurrence IS NULL AND trigger_at < ?", (cutoff,)).rowcount
        return await self._work.write(operation, guard=guard)
