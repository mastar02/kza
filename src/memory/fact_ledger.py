"""Authoritative fact lifecycle and durable vector-index reconciliation.

Blocking SQLite primitives are called from owned worker operations. Chroma is
an index: its documents and metadata never authorize memory use or retention.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class FactState(StrEnum):
    AUTOMATIC = "automatic"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DELETED = "deleted"
    QUARANTINED = "quarantined"


class MemoryOrigin(StrEnum):
    AMBIENT = "ambient"
    VOICE = "voice"
    INTERACTION = "interaction"


@dataclass(frozen=True)
class FactDependency:
    """Content-free receipt for one semantic fact included in a prompt."""

    fact_id: str
    revision: int
    expires_at: float | None

    def __post_init__(self) -> None:
        if type(self.fact_id) is not str or not self.fact_id.startswith("fact_") or len(self.fact_id) != 69:
            raise ValueError("invalid_fact_dependency")
        if any(c not in "0123456789abcdef" for c in self.fact_id[5:]):
            raise ValueError("invalid_fact_dependency")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("invalid_fact_dependency_revision")
        if self.expires_at is not None and (type(self.expires_at) not in (int, float) or not math.isfinite(self.expires_at)):
            raise ValueError("invalid_fact_dependency_expiry")

    def to_dict(self) -> dict:
        """Serialize a receipt without copying semantic content."""
        return {"fact_id": self.fact_id, "revision": self.revision, "expires_at": self.expires_at}

    @classmethod
    def from_dict(cls, data: dict) -> FactDependency:
        """Validate persisted receipts; malformed provenance fails closed."""
        return cls(data["fact_id"], data["revision"], data["expires_at"])


@dataclass(frozen=True)
class MemorySource:
    """Actual capture provenance; a speaker label is never an authority."""

    capture_id: str
    conversation_id: str
    captured_at: float
    expires_at: float
    origin: MemoryOrigin
    fact_dependencies: tuple[FactDependency, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.capture_id, self.conversation_id):
            if type(value) is not str or str(UUID(value)) != value:
                raise ValueError("invalid_memory_source_id")
        if not isinstance(self.origin, MemoryOrigin):
            raise ValueError("invalid_memory_origin")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in (self.captured_at, self.expires_at)):
            raise ValueError("invalid_memory_source_time")
        if not self.captured_at < self.expires_at <= self.captured_at + 48 * 3600:
            raise ValueError("invalid_memory_source_expiry")
        if not isinstance(self.fact_dependencies, tuple) or any(not isinstance(d, FactDependency) for d in self.fact_dependencies):
            raise ValueError("invalid_memory_fact_dependencies")
        if len({d.fact_id for d in self.fact_dependencies}) != len(self.fact_dependencies):
            raise ValueError("duplicate_memory_fact_dependency")
        if any(d.expires_at is not None and self.expires_at > d.expires_at for d in self.fact_dependencies):
            raise ValueError("memory_source_exceeds_dependency_expiry")

    def to_dict(self) -> dict:
        """Serialize verified provenance without adding derived authority."""
        return {"capture_id": self.capture_id, "conversation_id": self.conversation_id,
                "captured_at": self.captured_at, "expires_at": self.expires_at, "origin": self.origin.value,
                **({"fact_dependencies": [d.to_dict() for d in self.fact_dependencies]} if self.fact_dependencies else {})}

    @classmethod
    def from_dict(cls, data: dict) -> MemorySource:
        """Validate persisted provenance; missing legacy fields are not invented."""
        return cls(data["capture_id"], data["conversation_id"], data["captured_at"],
                   data["expires_at"], MemoryOrigin(data["origin"]),
                   tuple(FactDependency.from_dict(d) for d in data.get("fact_dependencies", [])))


def sources_current(sources: tuple[MemorySource, ...], now: float) -> bool:
    """Check raw-source deadlines without touching disk or renewing retention."""
    return bool(sources) and all(isinstance(s, MemorySource) and s.captured_at <= now < s.expires_at for s in sources)


def sources_authorized(sources: tuple[MemorySource, ...], now: float, policy=None) -> bool:
    """Fail closed for fact receipts when their authoritative ledger is absent."""
    return sources_current(sources, now) and (
        policy.sources_available(sources) if policy is not None else not any(s.fact_dependencies for s in sources))


@dataclass(frozen=True)
class MemoryReviewer:
    """Server-minted capability for an authenticated dashboard review."""

    subject: str

    @classmethod
    def from_dashboard(cls, principal) -> "MemoryReviewer":
        """Mint review authority from the dashboard's actual session DTO."""
        from src.dashboard.auth import DashboardPrincipal
        if not isinstance(principal, DashboardPrincipal) or "memory:review" not in principal.permissions:
            raise PermissionError("memory_review_requires_dashboard_session")
        return cls(principal.subject)


@dataclass(frozen=True)
class StoredFact:
    """A ledger receipt, independent of vector-index availability."""

    id: str
    content: str
    category: str
    confidence: float
    state: FactState
    extracted_at: float
    expires_at: float | None
    revision: int
    sources: tuple[MemorySource, ...]
    confirmed_at: float | None = None
    confirmed_by: str | None = None

    def to_dict(self) -> dict:
        """Project facts for context/review, without vector-provider metadata."""
        return {
            "id": self.id, "content": self.content, "category": self.category,
            "confidence": self.confidence, "state": self.state.value,
            "created_at": self.extracted_at, "extracted_at": self.extracted_at,
            "expires_at": self.expires_at, "revision": self.revision, "confirmed_at": self.confirmed_at,
            "confirmed_by": self.confirmed_by,
            "sources": [s.to_dict() for s in self.sources],
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY, content TEXT, category TEXT NOT NULL,
  confidence REAL NOT NULL, state TEXT NOT NULL,
  extracted_at REAL NOT NULL, expires_at REAL,
  revision INTEGER NOT NULL, confirmed_at REAL, confirmed_by TEXT
);
CREATE TABLE IF NOT EXISTS fact_sources (
  fact_id TEXT NOT NULL REFERENCES facts(id), capture_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL, captured_at REAL NOT NULL,
  expires_at REAL NOT NULL, origin TEXT NOT NULL, dependencies TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY(fact_id, capture_id)
);
CREATE TABLE IF NOT EXISTS fact_dependencies (
  fact_id TEXT NOT NULL REFERENCES facts(id), source_fact_id TEXT NOT NULL REFERENCES facts(id),
  PRIMARY KEY(fact_id, source_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_fact_dependency_source ON fact_dependencies(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_conversation ON fact_sources(conversation_id);
CREATE INDEX IF NOT EXISTS idx_fact_expiry ON facts(expires_at);
CREATE TABLE IF NOT EXISTS conversation_tombstones (
  conversation_id TEXT PRIMARY KEY, deleted_at REAL NOT NULL, deleted_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_jobs (
  fact_id TEXT PRIMARY KEY REFERENCES facts(id), revision INTEGER NOT NULL,
  operation TEXT NOT NULL, queued_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);
"""


class FactLedger:
    """Own expiry, source links, review transitions and durable deletion intent."""

    def __init__(self, path: str = ":memory:", *, clock: Callable[[], float] | None = None,
                 automatic_days: float = 30.0):
        if type(automatic_days) not in (int, float) or not math.isfinite(automatic_days) or not 0 < automatic_days <= 30:
            raise ValueError("automatic_memory_retention_must_be_at_most_30_days")
        self.path, self.automatic_days = path, automatic_days
        self.clock = time.time if clock is None else clock
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self._authority_available = False
        self._deleted_conversations: frozenset[str] = frozenset()
        self._deleted_times: dict[str, float] = {}
        self._pending_deleted_times: dict[str, float] = {}
        self._fact_receipts: dict[str, FactDependency] = {}
        self._pending_fact_ids: frozenset[str] = frozenset()
        self._pending_conversations: frozenset[str] = frozenset()
        self._authority_version = 0

    def initialize(self) -> None:
        """Create the private ledger without importing legacy documents."""
        with self._lock:
            if self._db is not None:
                return
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
                os.close(descriptor)
                os.chmod(self.path, 0o600)
            self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
            if "dependencies" not in {r[1] for r in self._db.execute("PRAGMA table_info(fact_sources)")}:
                try:
                    self._db.execute("BEGIN IMMEDIATE")
                    self._db.execute("ALTER TABLE fact_sources ADD COLUMN dependencies TEXT NOT NULL DEFAULT '[]'")
                    # Previous automatic derivations did not record prompt
                    # parents. Preserve their content privately, but do not
                    # invent independence or let old index rows authorize use.
                    for row in list(self._db.execute("SELECT id, revision FROM facts WHERE state=?", (FactState.AUTOMATIC.value,))):
                        revision = row["revision"]+1
                        self._db.execute("UPDATE facts SET state=?, revision=? WHERE id=?",
                            (FactState.QUARANTINED.value, revision, row["id"]))
                        self._queue(self._db, row["id"], revision, "delete")
                    self._db.commit()
                except BaseException:
                    try:
                        self._db.rollback()
                    finally:
                        self._db.close()
                        self._db = None
                    raise
            self._fact_receipts = self._read_fact_receipts(self._db)
            self._deleted_times = dict(self._db.execute("SELECT conversation_id, deleted_at FROM conversation_tombstones"))
            self._deleted_conversations = frozenset(self._deleted_times)
            self._authority_available = True

    @property
    def authority_available(self) -> bool:
        """Whether the initialized writer still owns a live authority snapshot."""
        return self._authority_available

    def sources_available(self, sources: tuple[MemorySource, ...]) -> bool:
        """Read this sole writer's committed revocations without disk on-loop."""
        return self._authority_available and sources_current(sources, self.clock()) and all(
            s.conversation_id not in self._deleted_conversations and s.conversation_id not in self._pending_conversations
            and self.facts_available(s.fact_dependencies) for s in sources)

    def conversation_revoked(self, conversation_id: str) -> bool:
        """Read a committed revocation without disk or a SQLite lock."""
        return conversation_id in self._deleted_conversations or conversation_id in self._pending_conversations

    def conversation_deleted_at(self, conversation_id: str) -> float | None:
        """Read the capture-time fence without disk or source-scope renewal."""
        return self._pending_deleted_times.get(conversation_id, self._deleted_times.get(conversation_id))

    def facts_available(self, dependencies: tuple[FactDependency, ...]) -> bool:
        """Validate committed fact receipts without SQLite or locks on-loop."""
        now = self.clock()
        return self._authority_available and all(d.fact_id not in self._pending_fact_ids and self._fact_receipts.get(d.fact_id) == d
                   and (d.expires_at is None or now < d.expires_at) for d in dependencies)

    def _read_fact_receipts(self, db) -> dict[str, FactDependency]:
        receipts = {}
        for fact_id, revision, expiry in db.execute(
            "SELECT id, revision, expires_at FROM facts WHERE state IN (?,?)",
            (FactState.AUTOMATIC.value, FactState.CONFIRMED.value)):
            previous = self._fact_receipts.get(fact_id)
            receipts[fact_id] = previous if previous is not None and (previous.revision, previous.expires_at) == (revision, expiry) else FactDependency(fact_id, revision, expiry)
        return receipts

    @contextmanager
    def admit_source_write(self, sources: tuple[MemorySource, ...]):
        """Serialize a derived-file commit before a conversation tombstone.

        Call only from an owned disk worker. A future deletion sees either
        the committed file or a rejected write, never a write after its marker.
        """
        with self._transaction() as db:
            if not self._sources_live(db, sources):
                raise PermissionError("memory_source_expired_or_deleted")
            yield

    def close(self) -> None:
        """Release the connection after all owning worker operations finish."""
        with self._lock:
            self._authority_available = False
            if self._db is not None:
                self._db.close()
                self._db = None

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.initialize()
            self._db.execute("BEGIN IMMEDIATE")
            changes = self._authority_version
            recovering = bool(self._pending_fact_ids or self._pending_conversations)
            clear_pending = True
            try:
                yield self._db
                changed = self._authority_version != changes or recovering
                if changed:
                    receipts = self._read_fact_receipts(self._db)
                    deleted_times = dict(self._db.execute("SELECT conversation_id, deleted_at FROM conversation_tombstones"))
                    deleted = frozenset(deleted_times)
                    # Deny affected readers before commit releases the GIL. New
                    # positive authority is published only after durable commit.
                    self._pending_fact_ids = frozenset(key for key in self._fact_receipts.keys() | receipts.keys()
                        if self._fact_receipts.get(key) != receipts.get(key))
                    self._pending_conversations = deleted - self._deleted_conversations
                    self._pending_deleted_times = {key: deleted_times[key] for key in self._pending_conversations}
                self._db.commit()
                if changed:
                    self._fact_receipts = receipts
                    self._deleted_times = deleted_times
                    self._deleted_conversations = deleted
            except BaseException:
                clear_pending = False
                try:
                    self._db.rollback()
                    # A failing commit may have reached disk. Re-read actual
                    # authority after rollback; never restore an assumed state.
                    self._fact_receipts = self._read_fact_receipts(self._db)
                    self._deleted_times = dict(self._db.execute("SELECT conversation_id, deleted_at FROM conversation_tombstones"))
                    self._deleted_conversations = frozenset(self._deleted_times)
                    clear_pending = True
                finally:
                    if clear_pending:
                        self._pending_fact_ids = self._pending_conversations = frozenset()
                        self._pending_deleted_times = {}
                raise
            finally:
                if clear_pending:
                    self._pending_fact_ids = self._pending_conversations = frozenset()
                    self._pending_deleted_times = {}

    def _queue(self, db, fact_id: str, revision: int, operation: str) -> None:
        db.execute(
            "INSERT INTO index_jobs(fact_id, revision, operation, queued_at) VALUES (?,?,?,?) "
            "ON CONFLICT(fact_id) DO UPDATE SET revision=excluded.revision, operation=excluded.operation, "
            "queued_at=excluded.queued_at, attempts=0, last_error=NULL",
            (fact_id, revision, operation, self.clock()),
        )

    def _sources_live(self, db, sources: tuple[MemorySource, ...]) -> bool:
        now = self.clock()
        return bool(sources) and all(
            s.captured_at <= now < s.expires_at
            and all(self._fact_receipts.get(d.fact_id) == d and (d.expires_at is None or now < d.expires_at) for d in s.fact_dependencies)
            and db.execute("SELECT 1 FROM conversation_tombstones WHERE conversation_id=?", (s.conversation_id,)).fetchone() is None
            for s in sources
        )

    def store(self, content: str, category: str, confidence: float, sources: tuple[MemorySource, ...],
              *, proposed: bool = False) -> StoredFact:
        """Persist a sourced automatic fact or a short-lived review proposal."""
        if type(content) is not str or not content.strip() or len(content) > 12000:
            raise ValueError("invalid_memory_content")
        if category not in {"personal", "preference", "pattern", "fact"}:
            raise ValueError("invalid_memory_category")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid_memory_confidence")
        if not isinstance(sources, tuple) or not sources or any(not isinstance(s, MemorySource) for s in sources):
            raise ValueError("memory_sources_required")
        if len({s.capture_id for s in sources}) != len(sources):
            raise ValueError("duplicate_memory_source")
        dependencies = {d.fact_id: d for source in sources for d in source.fact_dependencies}
        if proposed and dependencies:
            raise PermissionError("memory_proposal_requires_direct_capture")
        with self._transaction() as db:
            if not self._sources_live(db, sources):
                raise PermissionError("memory_source_expired_or_deleted")
            now = self.clock()
            state = FactState.PROPOSED if proposed else FactState.AUTOMATIC
            expiry = min(s.expires_at for s in sources) if proposed else now + self.automatic_days * 86400
            expiry = min([expiry, *(d.expires_at for d in dependencies.values() if d.expires_at is not None)])
            identity = [content.strip(), category, sorted(s.capture_id for s in sources), proposed]
            if dependencies:
                identity.append([dependencies[key].to_dict() for key in sorted(dependencies)])
            key = json.dumps(identity, ensure_ascii=False)
            fact_id = "fact_" + hashlib.sha256(key.encode()).hexdigest()
            previous = db.execute("SELECT id FROM facts WHERE id=?", (fact_id,)).fetchone()
            if previous is not None:
                existing = self._fact(db, fact_id)
                if not self._eligible(db, existing, review=True):
                    raise PermissionError("memory_derivation_already_expired_or_deleted")
                return existing
            db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
                       (fact_id, content.strip(), category, confidence, state.value, now, expiry, 1))
            db.executemany("INSERT INTO fact_sources VALUES (?,?,?,?,?,?,?)", [
                (fact_id, s.capture_id, s.conversation_id, s.captured_at, s.expires_at, s.origin.value,
                 json.dumps([d.to_dict() for d in s.fact_dependencies])) for s in sources])
            db.executemany("INSERT INTO fact_dependencies VALUES (?,?)", [(fact_id, d.fact_id) for d in dependencies.values()])
            self._authority_version += 1
            if not proposed:
                self._queue(db, fact_id, 1, "upsert")
            return self._fact(db, fact_id)

    def _fact(self, db, fact_id: str) -> StoredFact | None:
        row = db.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        if row is None or row["state"] == FactState.DELETED:
            return None
        sources = tuple(MemorySource(s["capture_id"], s["conversation_id"], s["captured_at"],
                                     s["expires_at"], MemoryOrigin(s["origin"]),
                                     tuple(FactDependency.from_dict(d) for d in json.loads(s["dependencies"])))
                        for s in db.execute("SELECT * FROM fact_sources WHERE fact_id=? ORDER BY capture_id", (fact_id,)))
        return StoredFact(row["id"], row["content"], row["category"], row["confidence"], FactState(row["state"]),
                          row["extracted_at"], row["expires_at"], row["revision"], sources,
                          row["confirmed_at"], row["confirmed_by"])

    def _eligible(self, db, fact: StoredFact | None, *, review: bool = False) -> bool:
        if fact is None or fact.state is FactState.QUARANTINED or (fact.expires_at is not None and fact.expires_at <= self.clock()):
            return False
        if fact.state is FactState.PROPOSED and not review:
            return False
        return all(db.execute("SELECT 1 FROM conversation_tombstones WHERE conversation_id=?", (s.conversation_id,)).fetchone() is None
                   for s in fact.sources)

    def get(self, fact_id: str, *, review: bool = False) -> StoredFact | None:
        """Return only facts whose current ledger state permits use."""
        with self._transaction() as db:
            fact = self._fact(db, fact_id)
            return fact if self._eligible(db, fact, review=review) else None

    def list(self, *, category: str | None = None, proposals: bool = False, limit: int = 100) -> list[StoredFact]:
        """Read live facts without refreshing their retention clock."""
        if type(limit) is not int or not 0 < limit <= 1000:
            raise ValueError("invalid_memory_limit")
        with self._transaction() as db:
            rows = db.execute(
                "SELECT id FROM facts WHERE state IN (?,?) AND (expires_at IS NULL OR expires_at>?) "
                "AND (? IS NULL OR category=?) ORDER BY extracted_at DESC, id",
                ((FactState.PROPOSED if proposals else FactState.AUTOMATIC).value,
                 (FactState.PROPOSED if proposals else FactState.CONFIRMED).value,
                 self.clock(), category, category),
            )
            result = []
            for row in rows:
                fact = self._fact(db, row["id"])
                if self._eligible(db, fact, review=proposals):
                    result.append(fact)
                    if len(result) == limit:
                        break
            return result

    def _reviewer(self, actor: MemoryReviewer) -> None:
        if not isinstance(actor, MemoryReviewer) or not actor.subject:
            raise PermissionError("memory_review_requires_dashboard_session")

    def confirm(self, fact_id: str, actor: MemoryReviewer) -> StoredFact:
        """Confirm a live proposal with explicit authenticated provenance."""
        self._reviewer(actor)
        with self._transaction() as db:
            fact = self._fact(db, fact_id)
            if not self._eligible(db, fact, review=True):
                raise LookupError("memory_proposal_unavailable")
            if fact.state is FactState.CONFIRMED:
                return fact
            if fact.state is not FactState.PROPOSED:
                raise ValueError("memory_proposal_required")
            revision = fact.revision + 1
            db.execute("UPDATE facts SET state=?, expires_at=NULL, confirmed_at=?, confirmed_by=?, revision=? WHERE id=?",
                       (FactState.CONFIRMED.value, self.clock(), actor.subject, revision, fact_id))
            self._queue(db, fact_id, revision, "upsert")
            self._authority_version += 1
            return self._fact(db, fact_id)

    def _delete(self, db, fact_id: str) -> int:
        rows = list(db.execute("""WITH RECURSIVE affected(id) AS (
            SELECT ? UNION SELECT d.fact_id FROM fact_dependencies d JOIN affected a ON d.source_fact_id=a.id
            ) SELECT f.id, f.revision FROM facts f JOIN affected a ON f.id=a.id WHERE f.state!=?""",
            (fact_id, FactState.DELETED.value)))
        for row in rows:
            revision = row["revision"] + 1
            db.execute("UPDATE facts SET content=NULL, state=?, expires_at=NULL, confirmed_at=NULL, confirmed_by=NULL, revision=? WHERE id=?",
                       (FactState.DELETED.value, revision, row["id"]))
            db.execute("DELETE FROM fact_sources WHERE fact_id=?", (row["id"],))
            self._queue(db, row["id"], revision, "delete")
        if rows:
            self._authority_version += 1
        return len(rows)

    def delete_fact(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """Reject/delete a fact through the authenticated review boundary."""
        self._reviewer(actor)
        with self._transaction() as db:
            return bool(self._delete(db, fact_id))

    def reject_proposal(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """A stale rejection must not delete a proposal confirmed elsewhere."""
        self._reviewer(actor)
        with self._transaction() as db:
            fact = self._fact(db, fact_id)
            if fact is None:
                return False
            if fact.state is not FactState.PROPOSED:
                raise ValueError("memory_proposal_state_changed")
            return bool(self._delete(db, fact_id))

    def snapshot(self) -> dict:
        """Read one coherent review snapshot with a server expiry clock."""
        with self._transaction() as db:
            facts, proposals = [], []
            for row in db.execute("SELECT id FROM facts WHERE state!=? ORDER BY extracted_at DESC, id", (FactState.DELETED.value,)):
                fact = self._fact(db, row["id"])
                if not self._eligible(db, fact, review=True):
                    continue
                target = proposals if fact.state is FactState.PROPOSED else facts
                if len(target) < 100:
                    target.append(fact.to_dict())
            return {"facts": facts, "proposals": proposals, "observed_at": self.clock(),
                    "legacy_policy": "quarantined",
                    "pending_index": db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0]}

    def delete_conversation(self, conversation_id: str, actor: MemoryReviewer) -> int:
        """Tombstone a conversation before scrubbing all currently derived facts."""
        self._reviewer(actor)
        conversation_id = str(UUID(conversation_id))
        with self._lock:
            with self._transaction() as db:
                db.execute("INSERT OR IGNORE INTO conversation_tombstones VALUES (?,?,?)",
                           (conversation_id, self.clock(), actor.subject))
                self._authority_version += 1
                ids = [r[0] for r in db.execute("SELECT DISTINCT fact_id FROM fact_sources WHERE conversation_id=?", (conversation_id,))]
                deleted = sum(self._delete(db, fact_id) for fact_id in ids)
            self._deleted_conversations = self._deleted_conversations | {conversation_id}
            return deleted

    def expire(self) -> int:
        """Scrub expired contents and enqueue index cleanup in one transaction."""
        with self._transaction() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM facts WHERE expires_at<=? AND state!=?",
                                          (self.clock(), FactState.DELETED.value))]
            return sum(self._delete(db, fact_id) for fact_id in ids)

    def jobs(self, limit: int = 32) -> list[dict]:
        """Read a bounded reconciliation batch; revisions prevent stale ack."""
        with self._transaction() as db:
            return [dict(r) for r in db.execute("SELECT * FROM index_jobs ORDER BY attempts, queued_at, fact_id LIMIT ?", (limit,))]

    def eligible_ids(self, min_confidence: float = 0, *, category: str | None = None) -> list[str]:
        """Pre-filter vector candidates so stale entries cannot crowd out live facts."""
        with self._transaction() as db:
            return [r[0] for r in db.execute(
                "SELECT id FROM facts WHERE state IN (?,?) AND confidence>=? AND (expires_at IS NULL OR expires_at>?) "
                "AND (? IS NULL OR category=?)",
                (FactState.AUTOMATIC.value, FactState.CONFIRMED.value, min_confidence, self.clock(), category, category))]

    def sources_admitted(self, sources: tuple[MemorySource, ...]) -> bool:
        """Check a producer's sources again before retaining a raw interaction."""
        with self._transaction() as db:
            return self._sources_live(db, sources)

    def next_maintenance_delay(self) -> float:
        """Bound retry delay and wake at the earliest pending fact expiry."""
        with self._transaction() as db:
            expiry = db.execute("SELECT MIN(expires_at) FROM facts WHERE state!=?", (FactState.DELETED.value,)).fetchone()[0]
            pending = db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0]
            delay = 30.0 if pending else 3600.0
            return delay if expiry is None else min(delay, max(.01, expiry-self.clock()))

    def acknowledge(self, fact_id: str, revision: int, *, error: str | None = None) -> None:
        """Acknowledge only the revision actually written to the index."""
        with self._transaction() as db:
            if error is None:
                db.execute("DELETE FROM index_jobs WHERE fact_id=? AND revision=?", (fact_id, revision))
            else:
                db.execute("UPDATE index_jobs SET attempts=attempts+1, last_error=? WHERE fact_id=? AND revision=?",
                           (error, fact_id, revision))

    def status(self) -> dict:
        """Report lifecycle counts without exposing deleted text or clip paths."""
        with self._transaction() as db:
            return {"pending_index": db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0],
                    "tombstones": db.execute("SELECT COUNT(*) FROM conversation_tombstones").fetchone()[0],
                    "states": {row[0]: row[1] for row in db.execute("SELECT state, COUNT(*) FROM facts GROUP BY state")}}
