"""Transactional SQLite definitions and durable execution receipts."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import sqlite3
from typing import AsyncIterator
import uuid

import aiosqlite

from src.routines.models import (
    ActionStatus, ClaimResult, RoutineActionResult, RoutineConflict, RoutineNotFound,
    RoutineRun, RoutineSpec, RoutineValidationError, RunStatus, ScheduledRoutine,
    parse_routine_spec, routine_to_dict, spec_to_dict,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS routines (
    routine_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
    created_by TEXT NOT NULL, created_at TEXT NOT NULL, definition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routine_runs (
    run_id TEXT PRIMARY KEY, routine_id TEXT NOT NULL, occurrence_key TEXT NOT NULL,
    actor TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
    snapshot_json TEXT NOT NULL, scheduled_for TEXT,
    UNIQUE (routine_id, occurrence_key)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_running_run_per_routine
    ON routine_runs(routine_id) WHERE status = 'running';
CREATE TABLE IF NOT EXISTS routine_run_actions (
    run_id TEXT NOT NULL, action_index INTEGER NOT NULL, entity_id TEXT NOT NULL,
    status TEXT NOT NULL, reason TEXT, PRIMARY KEY (run_id, action_index),
    FOREIGN KEY (run_id) REFERENCES routine_runs(run_id)
);
"""


def _decode_routine(payload: dict) -> ScheduledRoutine:
    definition = {key: value for key, value in payload.items()
                  if key not in {"routine_id", "revision", "created_by", "created_at"}}
    return ScheduledRoutine(payload["routine_id"], payload["revision"], payload["created_by"],
                            datetime.fromisoformat(payload["created_at"]), parse_routine_spec(definition))


def _routine_from_row(row: aiosqlite.Row) -> ScheduledRoutine:
    return ScheduledRoutine(row["routine_id"], row["revision"], row["created_by"],
                            datetime.fromisoformat(row["created_at"]), parse_routine_spec(json.loads(row["definition_json"])))


class RoutineStore:
    """Own one SQLite connection, serializing complete local transactions.

    Args:
        db_path: SQLite file path, or an in-memory SQLite database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> str:
        """Return the database path for diagnostics without allowing mutation."""
        return self._db_path

    async def initialize(self) -> None:
        """Open storage and initialize schema and SQLite safety settings."""
        async with self._lock:
            if self._db is not None:
                return
            db = await aiosqlite.connect(self._db_path)
            try:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.execute("PRAGMA journal_mode = WAL")
                await db.executescript(_SCHEMA)
                await db.commit()
            except BaseException:
                await db.close()
                raise
            self._db = db

    async def close(self) -> None:
        """Close the owned connection after any active operation completes."""
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("RoutineStore must be initialized")
        return self._db

    @asynccontextmanager
    async def _transaction_unlocked(self) -> AsyncIterator[aiosqlite.Connection]:
        db = self._connection()
        try:
            await db.execute("BEGIN IMMEDIATE")
            yield db
            await db.commit()
        except BaseException:
            # Cancelling an aiosqlite await does not cancel its queued SQLite
            # operation. Rollback queues behind BEGIN/commit and must finish
            # before the caller releases the connection lock, even on re-cancel.
            rollback = asyncio.create_task(db.rollback())
            while not rollback.done():
                try:
                    await asyncio.shield(rollback)
                except asyncio.CancelledError:
                    pass
            rollback.result()
            raise

    async def _load_routine_unlocked(self, routine_id: str) -> ScheduledRoutine | None:
        async with self._connection().execute("SELECT * FROM routines WHERE routine_id = ?", (routine_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _routine_from_row(row)

    async def _load_run_unlocked(self, run_id: str) -> RoutineRun | None:
        db = self._connection()
        # One statement gives reads a coherent SQLite snapshot across connections.
        async with db.execute(
            "SELECT r.*, a.action_index, a.entity_id, a.status AS action_status, a.reason AS action_reason "
            "FROM routine_runs r LEFT JOIN routine_run_actions a ON a.run_id = r.run_id "
            "WHERE r.run_id = ? ORDER BY a.action_index", (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return None
        row = rows[0]
        return RoutineRun(
            row["run_id"], row["routine_id"], row["occurrence_key"], row["actor"], row["source"],
            RunStatus(row["status"]), _decode_routine(json.loads(row["snapshot_json"])),
            datetime.fromisoformat(row["scheduled_for"]) if row["scheduled_for"] else None,
            tuple(RoutineActionResult(item["action_index"], item["entity_id"], ActionStatus(item["action_status"]), item["action_reason"])
                  for item in rows if item["action_index"] is not None),
            row["reason"],
        )

    async def get(self, routine_id: str) -> ScheduledRoutine | None:
        """Load a definition, or return None when absent."""
        async with self._lock:
            return await self._load_routine_unlocked(routine_id)

    async def list_all(self) -> list[ScheduledRoutine]:
        """Return only definitions owned by this KZA store."""
        async with self._lock:
            async with self._connection().execute("SELECT * FROM routines ORDER BY rowid") as cursor:
                rows = await cursor.fetchall()
            return [_routine_from_row(row) for row in rows]

    async def _require_idle_unlocked(self, routine_id: str) -> None:
        async with self._connection().execute("SELECT 1 FROM routine_runs WHERE routine_id = ? AND status = 'running'", (routine_id,)) as cursor:
            if await cursor.fetchone() is not None:
                raise RoutineConflict("routine_busy")

    async def save(self, spec: RoutineSpec, *, actor: str, routine_id: str | None = None,
                   expected_revision: int | None = None) -> ScheduledRoutine:
        """Commit a new definition or an optimistic revision of an idle routine."""
        spec = parse_routine_spec(spec_to_dict(spec))
        async with self._lock:
            async with self._transaction_unlocked() as db:
                if routine_id is None:
                    if expected_revision is not None:
                        raise RoutineConflict("revision_conflict")
                    routine = ScheduledRoutine(uuid.uuid4().hex, 1, actor, datetime.now(timezone.utc), spec)
                else:
                    current = await self._load_routine_unlocked(routine_id)
                    if current is None:
                        raise RoutineNotFound(routine_id)
                    if expected_revision != current.revision:
                        raise RoutineConflict("revision_conflict")
                    await self._require_idle_unlocked(routine_id)
                    routine = ScheduledRoutine(routine_id, current.revision + 1, current.created_by, current.created_at, spec)
                await db.execute(
                    "INSERT INTO routines (routine_id, revision, created_by, created_at, definition_json) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(routine_id) DO UPDATE SET revision = excluded.revision, definition_json = excluded.definition_json",
                    (routine.routine_id, routine.revision, routine.created_by, routine.created_at.isoformat(), json.dumps(spec_to_dict(spec), ensure_ascii=False)),
                )
            return routine

    async def delete(self, routine_id: str, *, expected_revision: int) -> None:
        """Delete an idle definition while retaining every historical receipt."""
        async with self._lock:
            async with self._transaction_unlocked() as db:
                routine = await self._load_routine_unlocked(routine_id)
                if routine is None:
                    raise RoutineNotFound(routine_id)
                if routine.revision != expected_revision:
                    raise RoutineConflict("revision_conflict")
                await self._require_idle_unlocked(routine_id)
                await db.execute("DELETE FROM routines WHERE routine_id = ?", (routine_id,))

    async def claim_run(self, routine_id: str, occurrence_key: str, *, actor: str, source: str,
                        scheduled_for: datetime | None = None, expected_revision: int | None = None) -> ClaimResult:
        """Commit one idempotent receipt before allowing any external effects."""
        if scheduled_for is not None:
            if scheduled_for.tzinfo is None or scheduled_for.utcoffset() is None:
                raise RoutineValidationError("scheduled_for must have a timezone")
            scheduled_for = scheduled_for.astimezone(timezone.utc)
        async with self._lock:
            created = False
            try:
                async with self._transaction_unlocked() as db:
                    async with db.execute("SELECT run_id FROM routine_runs WHERE routine_id = ? AND occurrence_key = ?", (routine_id, occurrence_key)) as cursor:
                        existing = await cursor.fetchone()
                    if existing is not None:
                        run_id = existing["run_id"]
                    else:
                        routine = await self._load_routine_unlocked(routine_id)
                        if routine is None:
                            raise RoutineNotFound(routine_id)
                        if not routine.spec.enabled:
                            raise RoutineConflict("routine_disabled")
                        if expected_revision is not None and routine.revision != expected_revision:
                            raise RoutineConflict("revision_conflict")
                        async with db.execute("SELECT 1 FROM routine_runs WHERE routine_id = ? AND status = 'running'", (routine_id,)) as cursor:
                            busy = await cursor.fetchone() is not None
                        if busy and source == "manual":
                            raise RoutineConflict("routine_busy")
                        run_id = uuid.uuid4().hex
                        await db.execute(
                            "INSERT INTO routine_runs (run_id, routine_id, occurrence_key, actor, source, status, reason, snapshot_json, scheduled_for) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (run_id, routine_id, occurrence_key, actor, source, "skipped" if busy else "running", "routine_busy" if busy else None,
                             json.dumps(routine_to_dict(routine), ensure_ascii=False), scheduled_for.isoformat() if scheduled_for else None),
                        )
                        await db.executemany(
                            "INSERT INTO routine_run_actions (run_id, action_index, entity_id, status) VALUES (?, ?, ?, ?)",
                            [(run_id, index, action.entity_id, "not_executed" if busy else "pending") for index, action in enumerate(routine.spec.actions)],
                        )
                        created = True
            except sqlite3.IntegrityError as error:
                if str(error) == "UNIQUE constraint failed: routine_runs.routine_id":
                    raise RoutineConflict("routine_busy") from error
                raise
            return ClaimResult(await self._load_run_unlocked(run_id), created)

    async def mark_action_sending(self, run_id: str, index: int) -> None:
        """Durably transition exactly one pending action before its HA send."""
        async with self._lock:
            async with self._transaction_unlocked() as db:
                cursor = await db.execute(
                    "UPDATE routine_run_actions SET status = 'sending' WHERE run_id = ? AND action_index = ? AND status = 'pending' "
                    "AND EXISTS (SELECT 1 FROM routine_runs WHERE run_id = ? AND status = 'running')", (run_id, index, run_id),
                )
                if cursor.rowcount != 1:
                    raise RoutineConflict("occurrence_conflict")

    async def record_action(self, run_id: str, result: RoutineActionResult) -> None:
        """Commit a sent action outcome or mark a pending action unexecuted."""
        status = ActionStatus(result.status)
        if status not in {ActionStatus.CONFIRMED, ActionStatus.FAILED, ActionStatus.UNCERTAIN, ActionStatus.NOT_EXECUTED}:
            raise RoutineConflict("occurrence_conflict")
        previous = "pending" if status == ActionStatus.NOT_EXECUTED else "sending"
        async with self._lock:
            async with self._transaction_unlocked() as db:
                cursor = await db.execute(
                    "UPDATE routine_run_actions SET status = ?, reason = ? WHERE run_id = ? AND action_index = ? AND entity_id = ? AND status = ? "
                    "AND EXISTS (SELECT 1 FROM routine_runs WHERE run_id = ? AND status = 'running')",
                    (status.value, result.reason, run_id, result.index, result.entity_id, previous, run_id),
                )
                if cursor.rowcount != 1:
                    raise RoutineConflict("occurrence_conflict")

    async def finish_run(self, run_id: str, status: RunStatus) -> RoutineRun:
        """Commit a terminal receipt, refusing success without confirmations."""
        status = RunStatus(status)
        if status == RunStatus.RUNNING:
            raise RoutineConflict("occurrence_conflict")
        async with self._lock:
            async with self._transaction_unlocked() as db:
                run = await self._load_run_unlocked(run_id)
                if run is None:
                    raise RoutineNotFound(run_id)
                if run.status != RunStatus.RUNNING:
                    if run.status != status:
                        raise RoutineConflict("occurrence_conflict")
                else:
                    if status == RunStatus.SUCCEEDED and any(item.status != ActionStatus.CONFIRMED for item in run.actions):
                        raise RoutineConflict("occurrence_conflict")
                    if status == RunStatus.SKIPPED and any(item.status not in {ActionStatus.PENDING, ActionStatus.NOT_EXECUTED} for item in run.actions):
                        raise RoutineConflict("occurrence_conflict")
                    await db.execute("UPDATE routine_run_actions SET status = 'not_executed' WHERE run_id = ? AND status = 'pending'", (run_id,))
                    await db.execute("UPDATE routine_run_actions SET status = 'uncertain' WHERE run_id = ? AND status = 'sending'", (run_id,))
                    await db.execute("UPDATE routine_runs SET status = ? WHERE run_id = ?", (status.value, run_id))
            return await self._load_run_unlocked(run_id)

    async def recover_interrupted(self) -> int:
        """Interrupt unfinished receipts without retrying any action."""
        async with self._lock:
            async with self._transaction_unlocked() as db:
                await db.execute("UPDATE routine_run_actions SET status = 'uncertain', reason = 'interrupted' WHERE status = 'sending' AND run_id IN (SELECT run_id FROM routine_runs WHERE status = 'running')")
                await db.execute("UPDATE routine_run_actions SET status = 'not_executed', reason = 'interrupted' WHERE status = 'pending' AND run_id IN (SELECT run_id FROM routine_runs WHERE status = 'running')")
                cursor = await db.execute("UPDATE routine_runs SET status = 'interrupted', reason = 'interrupted' WHERE status = 'running'")
                return cursor.rowcount

    async def get_run(self, run_id: str) -> RoutineRun | None:
        """Load one complete receipt, or None when absent."""
        async with self._lock:
            return await self._load_run_unlocked(run_id)

    async def get_run_by_occurrence(self, routine_id: str, occurrence_key: str) -> RoutineRun | None:
        """Read a persisted occurrence without claiming or modifying any attempt."""
        async with self._lock:
            async with self._connection().execute(
                "SELECT run_id FROM routine_runs WHERE routine_id = ? AND occurrence_key = ?",
                (routine_id, occurrence_key),
            ) as cursor:
                row = await cursor.fetchone()
            return await self._load_run_unlocked(row["run_id"]) if row else None

    async def get_latest_run(self, routine_id: str) -> RoutineRun | None:
        """Return the most recently claimed receipt, retaining deleted history."""
        async with self._lock:
            async with self._connection().execute("SELECT run_id FROM routine_runs WHERE routine_id = ? ORDER BY rowid DESC LIMIT 1", (routine_id,)) as cursor:
                row = await cursor.fetchone()
            return await self._load_run_unlocked(row["run_id"]) if row else None
