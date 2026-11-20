"""Transactional SQLite definitions and durable execution receipts."""

from src.memory import output_registration_v2 as output_registration
import asyncio
from collections.abc import Callable
from contextvars import Context
from dataclasses import replace
from datetime import datetime, timezone
import json
import sqlite3
import uuid

import aiosqlite

from src.core.owned_work import ClosePhase, WorkCloseTimeout
from src.core.sqlite_work import AuthorityGuard, SQLiteWork

from src.routines.models import (
    ActionStatus, BrakeStatus, ClaimResult, RoutineActionResult, RoutineBrake, RoutineBrakeReceipt,
    RoutineConflict, RoutineNameConflict, RoutineNotFound, RoutineRun, RoutineSpec,
    RoutineValidationError, RunStatus, ScheduledRoutine, parse_routine_spec, routine_name_key,
    routine_to_dict, spec_to_dict,
)

HOUSEHOLD_VOICE = "household_voice"
_VOICE_BRAKE = "voice_brake"


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
        self._work: SQLiteWork | None = None
        self._initialized = False
        self._closing: asyncio.Task | None = None

    @property
    def db_path(self) -> str:
        """Return the database path for diagnostics without allowing mutation."""
        return self._db_path

    async def initialize(self) -> None:
        """Own schema setup before admitting consumers; repeated startup is idempotent."""
        if self._closing is not None and not self._closing.done():
            raise RuntimeError("routine_store_closing")
        async with self._lock:
            if self._db is not None:
                if self._initialized:
                    return
                raise RuntimeError("routine_store_initialization_incomplete")
            if self._closing is not None and not self._closing.done():
                raise RuntimeError("routine_store_closing")
            self._closing = None
            await output_registration.before_routine_open(self)
            self._db = await aiosqlite.connect(self._db_path)
            if self._closing is not None:
                raise RuntimeError("routine_store_closing")
            self._db.row_factory = aiosqlite.Row
            self._work = SQLiteWork(self._db)
            def configure(db):
                db.execute("PRAGMA foreign_keys = ON")
                db.execute("PRAGMA busy_timeout = 5000")
                db.execute("PRAGMA journal_mode = WAL").fetchall()
            await self._work.read(configure)
            def schema(db):
                # This private constant contains only individual DDL statements.
                # executescript would implicitly commit the owner's transaction.
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        db.execute(statement)
            await self._work.write(schema)
            try:
                await output_registration.async_ready(self)
            except BaseException:
                self._work.begin_close()
                raise
            self._initialized = True

    async def close(self) -> None:
        """Retain actual native drain and captured connection close."""
        if self._closing is None:
            if self._work is not None:
                self._work.begin_close()
            async def finish():
                async with self._lock:
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
                    await output_registration.owner_closed(self, connection=database)
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
            raise WorkCloseTimeout("routine_store_close_pending", phase=ClosePhase.DRAIN,
                diagnostics=self._work.diagnostics() if self._work is not None else {}) from error

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("RoutineStore must be initialized")
        return self._db

    def _owner(self) -> SQLiteWork:
        """Fail closed while initialization or shutdown owns the connection."""
        if self._work is None or not self._initialized:
            raise RuntimeError("RoutineStore must be initialized")
        return self._work

    async def drain_admitted(self) -> None:
        """Include committed claim publication before a scheduler snapshots workers."""
        if self._work is not None:
            await self._work.drain_admitted()

    def _load_routine_unlocked(self, db: sqlite3.Connection, routine_id: str) -> ScheduledRoutine | None:
        row = db.execute("SELECT * FROM routines WHERE routine_id = ?", (routine_id,)).fetchone()
        return _routine_from_row(row) if row is not None else None

    def _load_run_unlocked(self, db: sqlite3.Connection, run_id: str) -> RoutineRun | None:
        # One statement preserves a coherent snapshot across independent connections.
        rows = db.execute(
            "SELECT r.*, a.action_index, a.entity_id, a.status AS action_status, a.reason AS action_reason "
            "FROM routine_runs r LEFT JOIN routine_run_actions a ON a.run_id = r.run_id "
            "WHERE r.run_id = ? ORDER BY a.action_index", (run_id,)).fetchall()
        if not rows:
            return None
        row = rows[0]
        return RoutineRun(
            row["run_id"], row["routine_id"], row["occurrence_key"], row["actor"], row["source"],
            RunStatus(row["status"]), _decode_routine(json.loads(row["snapshot_json"])),
            datetime.fromisoformat(row["scheduled_for"]) if row["scheduled_for"] else None,
            tuple(RoutineActionResult(item["action_index"], item["entity_id"], ActionStatus(item["action_status"]),
                item["action_reason"]) for item in rows if item["action_index"] is not None),
            row["reason"])

    async def get(self, routine_id: str, *, guard: AuthorityGuard | None = None) -> ScheduledRoutine | None:
        """Project a definition in one retained native callback."""
        return await self._owner().read(lambda db: self._load_routine_unlocked(db, routine_id), guard=guard)

    async def list_all(self, *, guard: AuthorityGuard | None = None) -> list[ScheduledRoutine]:
        """Return detached definitions from this store."""
        def operation(db):
            return [_routine_from_row(row) for row in db.execute("SELECT * FROM routines ORDER BY rowid").fetchall()]
        return await self._owner().read(operation, guard=guard)

    @staticmethod
    def _resolve_name(db: sqlite3.Connection, name: str) -> ScheduledRoutine:
        key = routine_name_key(name)
        rows = db.execute("SELECT * FROM routines ORDER BY rowid").fetchall()
        matches = [row for row in rows
                   if routine_name_key(json.loads(row["definition_json"])["name"]) == key]
        if len(matches) > 1:
            raise RoutineNameConflict(tuple(json.loads(row["definition_json"])["name"] for row in matches))
        if not matches:
            raise RoutineNotFound("routine_name")
        return _routine_from_row(matches[0])

    async def resolve_name(self, name: str, *, guard: AuthorityGuard | None = None) -> ScheduledRoutine:
        """Find a unique complete saved name, including disabled definitions."""
        return await self._owner().read(lambda db: self._resolve_name(db, name), guard=guard)

    def _require_idle_unlocked(self, db: sqlite3.Connection, routine_id: str) -> None:
        if db.execute("SELECT 1 FROM routine_runs WHERE routine_id = ? AND status = 'running'",
                      (routine_id,)).fetchone() is not None:
            raise RoutineConflict("routine_busy")

    async def save(self, spec: RoutineSpec, *, actor: str, routine_id: str | None = None,
                   expected_revision: int | None = None, guard: AuthorityGuard | None = None) -> ScheduledRoutine:
        """Commit normalization, revision checks and an idle definition atomically."""
        spec = parse_routine_spec(spec_to_dict(spec))
        def operation(db):
            if routine_id is None:
                if expected_revision is not None:
                    raise RoutineConflict("revision_conflict")
                routine = ScheduledRoutine(uuid.uuid4().hex, 1, actor, datetime.now(timezone.utc), spec)
            else:
                current = self._load_routine_unlocked(db, routine_id)
                if current is None:
                    raise RoutineNotFound(routine_id)
                if expected_revision != current.revision:
                    raise RoutineConflict("revision_conflict")
                self._require_idle_unlocked(db, routine_id)
                routine = ScheduledRoutine(routine_id, current.revision + 1,
                                           current.created_by, current.created_at, spec)
            db.execute(
                "INSERT INTO routines (routine_id, revision, created_by, created_at, definition_json) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(routine_id) DO UPDATE SET revision = excluded.revision, definition_json = excluded.definition_json",
                (routine.routine_id, routine.revision, routine.created_by, routine.created_at.isoformat(),
                 json.dumps(spec_to_dict(spec), ensure_ascii=False)))
            return routine
        return await self._owner().write(operation, guard=guard)

    async def delete(self, routine_id: str, *, expected_revision: int,
                     guard: AuthorityGuard | None = None) -> None:
        """Delete one idle revision without removing its historical receipts."""
        def operation(db):
            routine = self._load_routine_unlocked(db, routine_id)
            if routine is None:
                raise RoutineNotFound(routine_id)
            if routine.revision != expected_revision:
                raise RoutineConflict("revision_conflict")
            self._require_idle_unlocked(db, routine_id)
            db.execute("DELETE FROM routines WHERE routine_id = ?", (routine_id,))
        await self._owner().write(operation, guard=guard)

    async def claim_run(self, routine_id: str, occurrence_key: str, *, actor: str, source: str,
                        scheduled_for: datetime | None = None, expected_revision: int | None = None,
                        expected_name: str | None = None, guard: AuthorityGuard | None = None,
                        publish: Callable[[ClaimResult], None] | None = None) -> ClaimResult:
        """Own one native claim and its publication before admitting external effects."""
        if scheduled_for is not None:
            if scheduled_for.tzinfo is None or scheduled_for.utcoffset() is None:
                raise RoutineValidationError("scheduled_for must have a timezone")
            scheduled_for = scheduled_for.astimezone(timezone.utc)
        def operation(db):
            if source == HOUSEHOLD_VOICE:
                existing = self._voice_occurrence(db, occurrence_key, routine_id)
            else:
                existing = db.execute("SELECT run_id FROM routine_runs WHERE routine_id = ? AND occurrence_key = ?",
                                      (routine_id, occurrence_key)).fetchone()
            created = existing is None
            if existing is not None:
                run_id = existing["run_id"]
            else:
                routine = (self._resolve_name(db, expected_name) if expected_name is not None
                           else self._load_routine_unlocked(db, routine_id))
                if routine is None:
                    raise RoutineNotFound(routine_id)
                if routine.routine_id != routine_id:
                    raise RoutineConflict("revision_conflict")
                if not routine.spec.enabled:
                    raise RoutineConflict("routine_disabled")
                if expected_revision is not None and routine.revision != expected_revision:
                    raise RoutineConflict("revision_conflict")
                busy = db.execute("SELECT 1 FROM routine_runs WHERE routine_id = ? AND status = 'running'",
                                  (routine_id,)).fetchone() is not None
                if busy and source != "scheduled":
                    raise RoutineConflict("routine_busy")
                run_id = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO routine_runs (run_id, routine_id, occurrence_key, actor, source, status, reason, snapshot_json, scheduled_for) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, routine_id, occurrence_key, actor, source, "skipped" if busy else "running",
                     "routine_busy" if busy else None, json.dumps(routine_to_dict(routine), ensure_ascii=False),
                     scheduled_for.isoformat() if scheduled_for else None))
                db.executemany(
                    "INSERT INTO routine_run_actions (run_id, action_index, entity_id, status) VALUES (?, ?, ?, ?)",
                    [(run_id, index, action.entity_id, "not_executed" if busy else "pending")
                     for index, action in enumerate(routine.spec.actions)])
            run = self._load_run_unlocked(db, run_id)
            if run is None:
                raise RoutineNotFound(run_id)
            return ClaimResult(run, created)
        try:
            return await self._owner().write(operation, guard=guard, publish=publish)
        except sqlite3.IntegrityError as error:
            if str(error) == "UNIQUE constraint failed: routine_runs.routine_id":
                raise RoutineConflict("routine_busy") from error
            raise

    @staticmethod
    def _voice_occurrence(db: sqlite3.Connection, key: str, routine_id: str) -> sqlite3.Row | None:
        row = db.execute(
            "SELECT run_id, routine_id, reason FROM routine_runs WHERE source = ? AND occurrence_key = ?",
            (HOUSEHOLD_VOICE, key),
        ).fetchone()
        if row is not None and row["routine_id"] != routine_id:
            raise RoutineConflict("voice_request_conflict")
        return row

    @staticmethod
    def _brake_namespace(routine_id: str) -> str:
        # A reserved id in the existing runs table keeps brake receipts out of
        # the execution history even for binaries without voice brake support.
        return "voice-brake:" + routine_id

    @staticmethod
    def _brake_receipt(row: sqlite3.Row) -> RoutineBrakeReceipt:
        operation, status = row["reason"].split(":", 1)
        return RoutineBrakeReceipt(row["run_id"], RoutineBrake(operation), BrakeStatus(status))

    async def brake(self, name: str, occurrence_key: str, operation: RoutineBrake, *,
                    guard: AuthorityGuard) -> RoutineBrakeReceipt:
        """Fence pending sends, disable if requested, and record one local receipt."""
        def commit(db):
            routine = self._resolve_name(db, name)
            routine_id = routine.routine_id
            namespace = self._brake_namespace(routine_id)
            existing = self._voice_occurrence(db, occurrence_key, namespace)
            if existing is not None:
                receipt = self._brake_receipt(existing)
                if receipt.operation is not operation:
                    raise RoutineConflict("voice_request_conflict")
                return receipt
            snapshot = routine
            stopped = db.execute(
                "UPDATE routine_runs SET reason = ? WHERE routine_id = ? AND status = 'running' "
                "AND EXISTS (SELECT 1 FROM routine_run_actions "
                "WHERE run_id = routine_runs.run_id AND status = 'pending')",
                (_VOICE_BRAKE, routine_id),
            ).rowcount
            status = BrakeStatus.STOP_REQUESTED if stopped else BrakeStatus.NOTHING_TO_STOP
            if operation is RoutineBrake.DISABLE:
                status = BrakeStatus.ALREADY_DISABLED
                if routine.spec.enabled:
                    status = BrakeStatus.DISABLED
                    snapshot = replace(routine, revision=routine.revision + 1, spec=replace(routine.spec, enabled=False))
                    db.execute("UPDATE routines SET revision = ?, definition_json = ? WHERE routine_id = ?",
                               (snapshot.revision, json.dumps(spec_to_dict(snapshot.spec), ensure_ascii=False), routine_id))
            receipt_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO routine_runs (run_id, routine_id, occurrence_key, actor, source, status, reason, snapshot_json) "
                "VALUES (?, ?, ?, 'household', ?, 'succeeded', ?, ?)",
                (receipt_id, namespace, occurrence_key, HOUSEHOLD_VOICE, f"{operation}:{status}",
                 json.dumps(routine_to_dict(snapshot), ensure_ascii=False)),
            )
            return RoutineBrakeReceipt(receipt_id, operation, status)
        return await self._owner().write(commit, guard=guard)

    async def get_latest_brake(self, routine_id: str) -> RoutineBrakeReceipt | None:
        """Read the last committed voice brake independently of the last run."""
        def operation(db):
            row = db.execute(
                "SELECT run_id, reason FROM routine_runs WHERE routine_id = ? AND source = ? ORDER BY rowid DESC LIMIT 1",
                (self._brake_namespace(routine_id), HOUSEHOLD_VOICE),
            ).fetchone()
            return self._brake_receipt(row) if row is not None else None
        return await self._owner().read(operation)

    async def mark_action_sending(self, run_id: str, index: int) -> bool:
        """Admit one send atomically against a voice brake; false means braked."""
        def operation(db):
            changed = db.execute(
                "UPDATE routine_run_actions SET status = 'sending' WHERE run_id = ? AND action_index = ? AND status = 'pending' "
                "AND EXISTS (SELECT 1 FROM routine_runs WHERE run_id = ? AND status = 'running' AND reason IS NOT ?)",
                (run_id, index, run_id, _VOICE_BRAKE)).rowcount
            if changed == 1:
                return True
            if db.execute("SELECT 1 FROM routine_runs WHERE run_id = ? AND status = 'running' AND reason = ?",
                          (run_id, _VOICE_BRAKE)).fetchone() is not None:
                return False
            raise RoutineConflict("occurrence_conflict")
        return await self._owner().write(operation)

    async def record_action(self, run_id: str, result: RoutineActionResult) -> None:
        """Commit an approved run's actual action evidence without its creator lease."""
        status = ActionStatus(result.status)
        if status not in {ActionStatus.CONFIRMED, ActionStatus.FAILED, ActionStatus.UNCERTAIN, ActionStatus.NOT_EXECUTED}:
            raise RoutineConflict("occurrence_conflict")
        previous = "pending" if status == ActionStatus.NOT_EXECUTED else "sending"
        def operation(db):
            changed = db.execute(
                "UPDATE routine_run_actions SET status = ?, reason = ? "
                "WHERE run_id = ? AND action_index = ? AND entity_id = ? AND status = ? "
                "AND EXISTS (SELECT 1 FROM routine_runs WHERE run_id = ? AND status = 'running')",
                (status.value, result.reason, run_id, result.index, result.entity_id, previous, run_id)).rowcount
            if changed != 1:
                raise RoutineConflict("occurrence_conflict")
        await self._owner().write(operation)

    async def finish_run(self, run_id: str, status: RunStatus) -> RoutineRun:
        """Commit terminal evidence and its detached receipt in the same transaction."""
        status = RunStatus(status)
        if status == RunStatus.RUNNING:
            raise RoutineConflict("occurrence_conflict")
        def operation(db):
            run = self._load_run_unlocked(db, run_id)
            if run is None:
                raise RoutineNotFound(run_id)
            final = (RunStatus.INTERRUPTED if run.reason == _VOICE_BRAKE
                     and status in {RunStatus.SUCCEEDED, RunStatus.SKIPPED} else status)
            if run.status != RunStatus.RUNNING:
                if run.status != final:
                    raise RoutineConflict("occurrence_conflict")
            else:
                if final == RunStatus.SUCCEEDED and any(item.status != ActionStatus.CONFIRMED for item in run.actions):
                    raise RoutineConflict("occurrence_conflict")
                if final == RunStatus.SKIPPED and any(item.status not in {ActionStatus.PENDING, ActionStatus.NOT_EXECUTED}
                                                      for item in run.actions):
                    raise RoutineConflict("occurrence_conflict")
                db.execute("UPDATE routine_run_actions SET status = 'not_executed' WHERE run_id = ? AND status = 'pending'", (run_id,))
                db.execute("UPDATE routine_run_actions SET status = 'uncertain' WHERE run_id = ? AND status = 'sending'", (run_id,))
                db.execute("UPDATE routine_runs SET status = ? WHERE run_id = ?", (final.value, run_id))
            return self._load_run_unlocked(db, run_id)
        return await self._owner().write(operation)

    async def recover_interrupted(self) -> int:
        """Commit interruption evidence without resending any action."""
        def operation(db):
            db.execute("UPDATE routine_run_actions SET status = 'uncertain', reason = 'interrupted' "
                "WHERE status = 'sending' AND run_id IN (SELECT run_id FROM routine_runs WHERE status = 'running')")
            db.execute("UPDATE routine_run_actions SET status = 'not_executed', reason = 'interrupted' "
                "WHERE status = 'pending' AND run_id IN (SELECT run_id FROM routine_runs WHERE status = 'running')")
            return db.execute("UPDATE routine_runs SET status = 'interrupted', reason = COALESCE(reason, 'interrupted') "
                              "WHERE status = 'running'").rowcount
        return await self._owner().write(operation)

    async def get_run(self, run_id: str, *, guard: AuthorityGuard | None = None) -> RoutineRun | None:
        """Project one complete detached receipt."""
        return await self._owner().read(lambda db: self._load_run_unlocked(db, run_id), guard=guard)

    async def get_run_by_occurrence(self, routine_id: str, occurrence_key: str, *,
                                    guard: AuthorityGuard | None = None) -> RoutineRun | None:
        """Read a persisted occurrence without claiming another attempt."""
        def operation(db):
            row = db.execute("SELECT run_id FROM routine_runs WHERE routine_id = ? AND occurrence_key = ?",
                             (routine_id, occurrence_key)).fetchone()
            return self._load_run_unlocked(db, row["run_id"]) if row is not None else None
        return await self._owner().read(operation, guard=guard)

    async def get_latest_run(self, routine_id: str, *, guard: AuthorityGuard | None = None) -> RoutineRun | None:
        """Project the newest historical receipt without retaining a cursor."""
        def operation(db):
            row = db.execute("SELECT run_id FROM routine_runs WHERE routine_id = ? ORDER BY rowid DESC LIMIT 1",
                             (routine_id,)).fetchone()
            return self._load_run_unlocked(db, row["run_id"]) if row is not None else None
        return await self._owner().read(operation, guard=guard)
