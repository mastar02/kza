"""Durable conversation erasure intent and versioned cleanup receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
import math
import re
from uuid import UUID, uuid4
from typing import NoReturn


class ErasureParticipant(StrEnum):
    PRODUCERS = "producers"
    CAPTURES = "captures"
    CONTEXTS = "contexts"
    SEMANTIC_INDEX = "semantic_index"
    EVENTS = "events"
    DIAGNOSTICS = "diagnostics"
    COLLECTORS = "collectors"
    STREAMS = "streams"
    SUGGESTIONS = "suggestions"
    VOICES = "voices"


class ErasureState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class ErasureError(StrEnum):
    UNREGISTERED = "participant_unregistered"
    UNAVAILABLE = "participant_unavailable"
    FAILED = "cleanup_failed"
    INVALID_RESULT = "invalid_cleanup_result"
    PENDING = "cleanup_pending"
    RESTARTED = "interrupted_by_restart"


ERASURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_erasures (
  id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL UNIQUE REFERENCES conversation_tombstones(conversation_id),
  requested_at REAL NOT NULL, requested_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS erasure_targets (
  id TEXT PRIMARY KEY, participant TEXT NOT NULL, version INTEGER NOT NULL, descriptor TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_erasure_target_participant_id ON erasure_targets(participant, id);
CREATE TABLE IF NOT EXISTS erasure_steps (
  operation_id TEXT NOT NULL REFERENCES conversation_erasures(id), participant TEXT NOT NULL,
  target_id TEXT NOT NULL, version INTEGER NOT NULL, state TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL DEFAULT 0,
  claim_token TEXT, cursor TEXT, pending_count INTEGER, last_error TEXT,
  PRIMARY KEY(operation_id, participant, target_id)
);
CREATE INDEX IF NOT EXISTS idx_erasure_due ON erasure_steps(state, next_attempt_at);
"""


@dataclass(frozen=True)
class ErasureStep:
    """A server-owned cleanup claim; storage locations never enter API receipts."""

    operation_id: str
    conversation_id: str
    participant: ErasureParticipant
    target_id: str
    version: int
    descriptor: dict
    attempts: int
    claim_token: str
    cursor: str | None


@dataclass(frozen=True)
class ErasureResult:
    """Content-free cleanup progress, independent of observation timeouts."""

    done: bool
    cursor: str | None = None
    pending_count: int | None = None

    def __post_init__(self) -> None:
        if type(self.done) is not bool:
            raise ValueError("invalid_erasure_result")
        if self.cursor is not None and (type(self.cursor) is not str or not re.fullmatch(r"[a-zA-Z0-9:_-]{1,128}", self.cursor)):
            raise ValueError("invalid_erasure_cursor")
        if self.pending_count is not None and (type(self.pending_count) is not int or self.pending_count < 0):
            raise ValueError("invalid_erasure_pending_count")
        if self.done and self.pending_count not in (None, 0):
            raise ValueError("erasure_complete_with_pending_work")


MAX_TARGET_SNAPSHOT_ENTRIES = 256
MAX_TARGET_DESCRIPTOR_BYTES = 8192
_TARGET_PARTICIPANTS = frozenset({ErasureParticipant.EVENTS, ErasureParticipant.SUGGESTIONS})


@dataclass(frozen=True, slots=True)
class TargetRegistryEntry:
    """A bounded original-row observation, never a destination capability."""

    id: str
    participant: ErasureParticipant
    version: int | None
    descriptor_json: str | None = field(repr=False)
    status: str
    opaque_digest: str | None = None

    @property
    def descriptor(self) -> dict | None:
        """Return a fresh value; callers cannot mutate the frozen observation."""
        return None if self.descriptor_json is None else json.loads(self.descriptor_json)


@dataclass(frozen=True, eq=False, slots=True)
class TargetRegistrySnapshot:
    """One bounded SQL snapshot with its actual in-memory ledger provenance."""

    ledger: object = field(repr=False)
    lifecycle_generation: int
    participants: tuple[ErasureParticipant, ...]
    entries: tuple[TargetRegistryEntry, ...]
    digest: str
    status: str

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("target_registry_snapshot_not_serializable")


def _snapshot_arguments(participants, limit):
    if (type(participants) is not tuple or len(participants) != len(_TARGET_PARTICIPANTS)
            or any(type(p) is not ErasureParticipant or p not in _TARGET_PARTICIPANTS for p in participants)
            or set(participants) != _TARGET_PARTICIPANTS
            or type(limit) is not int or not 1 <= limit <= MAX_TARGET_SNAPSHOT_ENTRIES):
        raise ValueError("invalid_target_snapshot_scope")
    return tuple(sorted(participants, key=lambda p: p.value))


def _bounded_target_entry(row, participant):
    # This projection contains only bounded prefixes and scalar lengths. An
    # opaque digest identifies that projection, not unread oversized bytes.
    projected = [value.hex() if isinstance(value, bytes) else value for value in row]
    raw = json.dumps(projected, separators=(",", ":"), allow_nan=False).encode()
    opaque = hashlib.sha256(raw).hexdigest()
    id_type, id_size, id_raw, participant_type, participant_size, participant_raw, version_type, version, version_raw, descriptor_type, descriptor_size, descriptor_raw = row
    oversized = (id_size is not None and id_size > 64
                 or participant_size is not None and participant_size > 32
                 or descriptor_size is not None and descriptor_size > MAX_TARGET_DESCRIPTOR_BYTES)
    try:
        if (id_type != "text" or id_size != 64 or type(id_raw) is not bytes
                or participant_type != "text" or participant_size is None or participant_size > 32
                or type(participant_raw) is not bytes or version_type != "integer"
                or type(version) is not int or version < 1 or descriptor_type != "text"
                or type(descriptor_raw) is not bytes or descriptor_size is None
                or descriptor_size > MAX_TARGET_DESCRIPTOR_BYTES):
            raise ValueError("invalid_registered_target_row")
        identity = id_raw.decode("ascii")
        if re.fullmatch("[0-9a-f]{64}", identity) is None or participant_raw.decode("ascii") != participant.value:
            raise ValueError("invalid_registered_target_identity")
        encoded = descriptor_raw.decode("utf-8")
        descriptor = json.loads(encoded)
        if type(descriptor) is not dict:
            raise ValueError("invalid_registered_target_descriptor")
        canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        # Match _register_target exactly, including its outer JSON encoding.
        expected = hashlib.sha256(json.dumps([participant.value, version, canonical]).encode()).hexdigest()
        if canonical != encoded or expected != identity:
            raise ValueError("registered_target_identity_mismatch")
        return TargetRegistryEntry(identity, participant, version, canonical,
                                   "registered" if version == 1 else "unsupported_version"), raw
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return TargetRegistryEntry("opaque:" + opaque, participant, None, None,
                                   "oversized" if oversized else "corrupt", opaque), raw


class ErasureJournal:
    """Use the leased ledger transaction for intent, targets, claims and receipts."""

    def __init__(self, ledger):
        self.ledger = ledger

    def target_snapshot(
        self,
        participants: tuple[ErasureParticipant, ...] = (
            ErasureParticipant.EVENTS, ErasureParticipant.SUGGESTIONS,
        ),
        *, limit: int = MAX_TARGET_SNAPSHOT_ENTRIES,
    ) -> TargetRegistrySnapshot:
        """Observe bounded target rows in one original authority transaction.

        A complete observation is not a dispatch grant. Final admission must
        join it to the actual owner/native invocation under the same ledger.
        Existing transaction callers use _target_snapshot_db, not this method.
        """
        participants = _snapshot_arguments(participants, limit)
        with self.ledger._lock:
            generation = self.ledger._lifecycle_generation
            if self.ledger._db is None or not self.ledger.authority_available:
                return self._unavailable_target_snapshot(participants, generation)
            if self.ledger._db.in_transaction:
                raise RuntimeError("target_snapshot_requires_single_transaction")
            try:
                with self.ledger._transaction() as db:
                    return self._target_snapshot_db(db, participants, limit=limit)
            except Exception:
                # The original transaction owns rollback/authority recovery.
                # Failure is explicit, never an apparently complete empty set.
                return self._unavailable_target_snapshot(participants, generation)

    def _unavailable_target_snapshot(self, participants, generation):
        raw = json.dumps(["unavailable", generation, [p.value for p in participants]],
                         separators=(",", ":")).encode()
        return TargetRegistrySnapshot(self.ledger, generation, participants, (),
                                      hashlib.sha256(raw).hexdigest(), "unavailable")

    def _target_snapshot_db(
        self, db, participants: tuple[ErasureParticipant, ...], *, limit: int = MAX_TARGET_SNAPSHOT_ENTRIES,
    ) -> TargetRegistrySnapshot:
        """Read within one existing original transaction, without nested BEGIN."""
        participants = _snapshot_arguments(participants, limit)
        if (db is not self.ledger._db or not db.in_transaction
                or not self.ledger._lock._is_owned()):
            raise ValueError("target_snapshot_requires_original_transaction")
        generation = self.ledger._lifecycle_generation
        digest = hashlib.sha256(json.dumps(
            [generation, [p.value for p in participants]], separators=(",", ":"),
        ).encode())
        entries = []
        status = "complete"
        for participant in participants:
            remaining = limit - len(entries)
            # CASE avoids materializing a malformed non-integer version;
            # all text/blob projections are bounded before Python sees them.
            rows = db.execute("""
                SELECT typeof(id), length(CAST(id AS BLOB)), substr(CAST(id AS BLOB),1,65),
                       typeof(participant), length(CAST(participant AS BLOB)), substr(CAST(participant AS BLOB),1,33),
                       typeof(version), CASE WHEN typeof(version)='integer' THEN version ELSE NULL END,
                       substr(CAST(version AS BLOB),1,33), typeof(descriptor),
                       length(CAST(descriptor AS BLOB)), substr(CAST(descriptor AS BLOB),1,8193)
                FROM erasure_targets WHERE participant=? ORDER BY id LIMIT ?
            """, (participant.value, remaining + 1))
            try:
                for row in rows:
                    entry, raw = _bounded_target_entry(row, participant)
                    digest.update(len(raw).to_bytes(8, "big"))
                    digest.update(raw)
                    if len(entries) == limit:
                        status = "overflow"
                        break
                    entries.append(entry)
                    if entry.status in ("corrupt", "oversized"):
                        status = "corrupt"
            finally:
                rows.close()
            if status == "overflow":
                break
        digest.update(status.encode("ascii"))
        return TargetRegistrySnapshot(self.ledger, generation, participants,
                                      tuple(entries), digest.hexdigest(), status)

    def ensure_operation(self, db, conversation_id: str, actor) -> str:
        """Record intent in the same transaction as its conversation tombstone."""
        row = db.execute("SELECT id FROM conversation_erasures WHERE conversation_id=?", (conversation_id,)).fetchone()
        operation_id = row[0] if row is not None else str(uuid4())
        if row is None:
            db.execute("INSERT INTO conversation_erasures VALUES (?,?,?,?)",
                       (operation_id, conversation_id, self.ledger.clock(), actor.subject))
        self._ensure_steps(db, operation_id)
        return operation_id

    def _ensure_steps(self, db, operation_id: str) -> None:
        for participant in ErasureParticipant:
            targets = list(db.execute("SELECT id, version FROM erasure_targets WHERE participant=?", (participant.value,)))
            if targets:
                db.execute("DELETE FROM erasure_steps WHERE operation_id=? AND participant=? AND target_id='unregistered'",
                           (operation_id, participant.value))
                db.executemany("INSERT OR IGNORE INTO erasure_steps(operation_id, participant, target_id, version, state) VALUES (?,?,?,?,?)",
                    [(operation_id, participant.value, row[0], row[1], ErasureState.PENDING.value) for row in targets])
            else:
                db.execute("INSERT OR IGNORE INTO erasure_steps(operation_id, participant, target_id, version, state, last_error) VALUES (?,?,?,?,?,?)",
                    (operation_id, participant.value, "unregistered", 0, ErasureState.PENDING.value, ErasureError.UNREGISTERED.value))

    def recover(self) -> int:
        """Recover interrupted claims only after the runtime obtains its writer lease."""
        with self.ledger._transaction() as db:
            # Older releases already wrote tombstones without a cleanup journal.
            # Preserve the original actor/time instead of inventing a new request.
            for row in list(db.execute("""SELECT * FROM conversation_tombstones t WHERE NOT EXISTS
                (SELECT 1 FROM conversation_erasures o WHERE o.conversation_id=t.conversation_id)""")):
                db.execute("INSERT INTO conversation_erasures VALUES (?,?,?,?)",
                           (str(uuid4()), row["conversation_id"], row["deleted_at"], row["deleted_by"]))
            for row in list(db.execute("SELECT id FROM conversation_erasures")):
                self._ensure_steps(db, row[0])
            # Restored participant data may exist behind an old cursor
            # or completed receipt. Only leased startup may reopen that proof.
            db.execute("""UPDATE erasure_steps SET state='pending',claim_token=NULL,cursor=NULL,
                next_attempt_at=0,last_error=? WHERE participant IN ('producers','captures','events','contexts','suggestions','collectors','diagnostics','streams','voices')""", (ErasureError.RESTARTED.value,))
            return db.execute("UPDATE erasure_steps SET state=?, claim_token=NULL, next_attempt_at=0, last_error=? WHERE state=?",
                (ErasureState.PENDING.value, ErasureError.RESTARTED.value, ErasureState.RUNNING.value)).rowcount

    def register_target(self, participant: ErasureParticipant, version: int, descriptor: dict) -> str:
        """Persist every historical storage location and reopen existing cleanup jobs."""
        if not isinstance(participant, ErasureParticipant) or type(version) is not int or version < 1 or type(descriptor) is not dict:
            raise ValueError('invalid_erasure_target')
        if len(json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) > 8192:
            raise ValueError('erasure_target_limit')
        with self.ledger._transaction() as db:
            return self._register_target(db, participant, version, descriptor)

    def _register_target(self, db, participant: ErasureParticipant, version: int, descriptor: dict) -> str:
        """Share an existing authority transaction with a new data/location commit."""
        if db is not self.ledger._db or not db.in_transaction:
            raise ValueError('erasure_target_requires_authority_transaction')
        if not isinstance(participant, ErasureParticipant) or type(version) is not int or version < 1 or type(descriptor) is not dict:
            raise ValueError("invalid_erasure_target")
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > 8192:
            raise ValueError("erasure_target_limit")
        key = hashlib.sha256(json.dumps([participant.value, version, encoded]).encode()).hexdigest()
        db.execute("INSERT OR IGNORE INTO erasure_targets VALUES (?,?,?,?)", (key, participant.value, version, encoded))
        if participant is ErasureParticipant.SEMANTIC_INDEX:
            self.ledger.index_deletions.register_target(db, key)
        for row in list(db.execute("SELECT id FROM conversation_erasures")):
            self._ensure_steps(db, row[0])
        return key

    def request(self, conversation_id: str, actor) -> dict:
        """Revoke sources and persist one idempotent operation per conversation."""
        conversation_id = str(UUID(conversation_id))
        self.ledger.delete_conversation(conversation_id, actor)
        with self.ledger._transaction() as db:
            row = db.execute("SELECT id FROM conversation_erasures WHERE conversation_id=?", (conversation_id,)).fetchone()
            return self._snapshot(db, row[0])

    def targets(self, participant: ErasureParticipant) -> list[dict]:
        """Read private registered locators for a cleanup driver's ownership checks."""
        if not isinstance(participant, ErasureParticipant):
            raise ValueError('invalid_erasure_participant')
        with self.ledger._transaction() as db:
            return [{'id': row['id'], 'version': row['version'], 'descriptor': json.loads(row['descriptor'])}
                    for row in db.execute('SELECT * FROM erasure_targets WHERE participant=? ORDER BY id', (participant.value,))]

    def snapshot(self, operation_id: str) -> dict | None:
        """Read status without exposing storage descriptors, claims or raw content."""
        operation_id = str(UUID(operation_id))
        with self.ledger._transaction() as db:
            return self._snapshot(db, operation_id)

    def _snapshot(self, db, operation_id: str) -> dict | None:
        operation = db.execute("SELECT * FROM conversation_erasures WHERE id=?", (operation_id,)).fetchone()
        if operation is None:
            return None
        rows = list(db.execute("SELECT participant, state, attempts, next_attempt_at, pending_count, last_error FROM erasure_steps WHERE operation_id=? ORDER BY participant, target_id", (operation_id,)))
        present = {row["participant"] for row in rows}
        complete = present == {p.value for p in ErasureParticipant} and all(row["state"] == ErasureState.COMPLETE for row in rows)
        blocked = any(row["last_error"] in {ErasureError.UNREGISTERED, ErasureError.UNAVAILABLE} for row in rows)
        return {"id": operation_id, "conversation_id": operation["conversation_id"],
                "requested_at": operation["requested_at"], "requested_by": operation["requested_by"],
                "revoked": True, "state": (ErasureState.COMPLETE if complete else ErasureState.BLOCKED if blocked else ErasureState.PENDING).value,
                "steps": [dict(row) for row in rows]}

    def due(self, limit: int = 32) -> list[dict]:
        """Read bounded due work; unknown adapters remain visible and unacknowledged."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("invalid_erasure_batch")
        with self.ledger._transaction() as db:
            return [dict(row) for row in db.execute("""SELECT s.operation_id, s.participant, s.target_id, s.version
                FROM erasure_steps s JOIN conversation_erasures o ON o.id=s.operation_id
                WHERE s.state=? AND s.next_attempt_at<=? AND s.target_id!='unregistered'
                ORDER BY s.next_attempt_at, o.requested_at, s.participant LIMIT ?""",
                (ErasureState.PENDING.value, self.ledger.clock(), limit))]

    def unavailable(self, row: dict) -> None:
        """Persist adapter absence without inventing successful cleanup."""
        with self.ledger._transaction() as db:
            db.execute("UPDATE erasure_steps SET last_error=?, next_attempt_at=? WHERE operation_id=? AND participant=? AND target_id=? AND state=?",
                (ErasureError.UNAVAILABLE.value, self.ledger.clock()+30, row["operation_id"], row["participant"], row["target_id"], ErasureState.PENDING.value))

    def claim(self, row: dict) -> ErasureStep | None:
        """Mint one claim token; a stale worker cannot acknowledge another attempt."""
        with self.ledger._transaction() as db:
            token = str(uuid4())
            changed = db.execute("""UPDATE erasure_steps SET state=?, claim_token=?, attempts=attempts+1, last_error=NULL
                WHERE operation_id=? AND participant=? AND target_id=? AND version=? AND state=? AND next_attempt_at<=?""",
                (ErasureState.RUNNING.value, token, row["operation_id"], row["participant"], row["target_id"], row["version"], ErasureState.PENDING.value, self.ledger.clock())).rowcount
            if not changed:
                return None
            result = db.execute("""SELECT s.*, o.conversation_id, t.descriptor FROM erasure_steps s
                JOIN conversation_erasures o ON o.id=s.operation_id JOIN erasure_targets t ON t.id=s.target_id
                WHERE s.operation_id=? AND s.participant=? AND s.target_id=?""",
                (row["operation_id"], row["participant"], row["target_id"])).fetchone()
            return ErasureStep(result["operation_id"], result["conversation_id"], ErasureParticipant(result["participant"]),
                result["target_id"], result["version"], json.loads(result["descriptor"]), result["attempts"], token, result["cursor"])

    def is_current_claim(self, step: ErasureStep) -> bool:
        """Check the exact live claim and its committed conversation tombstone.

        This read does not acknowledge or renew a claim. Runtime participants
        recheck it after awaited work; finish() remains the atomic receipt guard.
        """
        if (type(step) is not ErasureStep
                or type(step.participant) is not ErasureParticipant
                or type(step.version) is not int or step.version < 1
                or type(step.attempts) is not int or step.attempts < 1
                or type(step.descriptor) is not dict
                or any(type(value) is not str or not value for value in (
                    step.operation_id, step.conversation_id, step.target_id, step.claim_token))
                or (step.cursor is not None and type(step.cursor) is not str)):
            return False
        try:
            encoded = json.dumps(
                step.descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError):
            return False
        with self.ledger._transaction() as db:
            return db.execute(
                """SELECT 1 FROM erasure_steps s
                JOIN conversation_erasures o ON o.id=s.operation_id
                JOIN conversation_tombstones revoked
                  ON revoked.conversation_id=o.conversation_id
                JOIN erasure_targets t ON t.id=s.target_id
                  AND t.participant=s.participant AND t.version=s.version
                WHERE s.operation_id=? AND o.conversation_id=?
                  AND s.participant=? AND s.target_id=? AND s.version=?
                  AND s.state=? AND s.claim_token=? AND s.attempts=?
                  AND s.cursor IS ? AND t.descriptor=?""",
                (step.operation_id, step.conversation_id, step.participant.value,
                 step.target_id, step.version, ErasureState.RUNNING.value,
                 step.claim_token, step.attempts, step.cursor, encoded),
            ).fetchone() is not None

    def finish(self, step: ErasureStep, result: ErasureResult | None, error: ErasureError | None = None) -> bool:
        """Commit a content-free receipt only for the exact owned claim."""
        if result is not None and not isinstance(result, ErasureResult) or error is not None and not isinstance(error, ErasureError):
            raise ValueError("invalid_erasure_receipt")
        complete = result is not None and result.done and error is None
        delay = min(60, 2**min(step.attempts-1, 6))
        if result is not None and error is None and result.cursor != step.cursor:
            delay = .01
        next_attempt = 0 if complete else self.ledger.clock()+delay
        if not math.isfinite(next_attempt):
            raise ValueError("invalid_erasure_retry_time")
        with self.ledger._transaction() as db:
            return bool(db.execute("""UPDATE erasure_steps SET state=?, claim_token=NULL, cursor=?, pending_count=?,
                last_error=?, next_attempt_at=? WHERE operation_id=? AND participant=? AND target_id=? AND version=? AND state=? AND claim_token=?""",
                (ErasureState.COMPLETE.value if complete else ErasureState.PENDING.value,
                 result.cursor if result is not None else step.cursor, result.pending_count if result is not None else None,
                 error.value if error is not None else None if complete else ErasureError.PENDING.value, next_attempt,
                 step.operation_id, step.participant.value, step.target_id, step.version, ErasureState.RUNNING.value, step.claim_token)).rowcount)

    def deleted_facts(self, after: str | None, limit: int = 32) -> list[dict]:
        """Sweep only policy facts already erased; quarantine/legacy data is excluded."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("invalid_erasure_index_batch")
        with self.ledger._transaction() as db:
            return [dict(row) for row in db.execute("SELECT id, revision FROM facts WHERE state='deleted' AND id>? ORDER BY id LIMIT ?", (after or "", limit))]
