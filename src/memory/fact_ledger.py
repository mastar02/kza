"""Authoritative fact lifecycle and durable vector-index reconciliation.

Blocking SQLite primitives are called from owned worker operations. Chroma is
an index: its documents and metadata never authorize memory use or retention.
"""

from __future__ import annotations

from src.memory import output_registration_v2 as output_registration
import math
import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import UUID
from weakref import WeakMethod

from src.core.voice_person_index import VoicePersonIndex
from src.memory.erasure_journal import ERASURE_SCHEMA, ErasureJournal
from src.dashboard.live_receipts import LIVE_RECEIPT_SCHEMA
from src.memory.index_deletions import INDEX_DELETION_SCHEMA, IndexDeletionQueue
from src.memory.voice_schema import (
    VOICE_SCHEMA, initialize_voice_attribution_schema, initialize_voice_people_schema,
    initialize_voice_review_schema,
)

logger = logging.getLogger(__name__)


class FactState(StrEnum):
    AUTOMATIC = "automatic"
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DELETED = "deleted"
    QUARANTINED = "quarantined"


class FactOwnerMethod(StrEnum):
    """Closed provenance of an owner row, including the voice stage (U3b W12)."""

    OPERATOR_ASSIGNMENT = "operator_assignment"
    PERSON_SESSION = "person_session"
    VOICE_PROPOSAL = "voice_proposal"


class FactOwnerChange(StrEnum):
    """Closed last-change values of fact_owner_revisions, including the voice stage."""

    ASSIGNED = "assigned"
    REASSIGNED = "reassigned"
    OWNER_RETURNED = "owner_returned"
    OPERATOR_UNASSIGNED = "operator_unassigned"
    VOICE_PROPOSED = "voice_proposed"
    OWNER_CONFIRMED = "owner_confirmed"
    OPERATOR_CONFIRMED = "operator_confirmed"


class MemoryScope(StrEnum):
    """Which facts a reader sees (U3b W5).

    HOUSEHOLD: never assigned (no fact_owner_revisions row). PERSON: currently
    owned by one person. ALL: every eligible fact; returned and unassigned
    facts are visible only here.
    """

    HOUSEHOLD = "household"
    PERSON = "person"
    ALL = "all"


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
    _current: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dashboard(cls, principal: object, current: Callable[[], bool] | None = None) -> "MemoryReviewer":
        """Mint review authority from the dashboard's actual session DTO."""
        from src.dashboard.auth import DashboardPrincipal
        if not isinstance(principal, DashboardPrincipal) or "memory:review" not in principal.permissions:
            raise PermissionError("memory_review_requires_dashboard_session")
        return cls(principal.subject, current)

    def require_current(self) -> None:
        """A queued native review cannot outlive its original session."""
        if self._current is not None:
            try:
                allowed = self._current()
            except Exception:
                raise PermissionError("memory_review_session_unavailable") from None
            if allowed is not True:
                raise PermissionError("memory_review_denied")


@dataclass(frozen=True)
class MemoryAssigner:
    """Server-minted capability to assign or unassign fact owners.

    Requires the permission string ``memory:assign``. No DashboardPermission
    member grants it until U3b-2, so production cannot mint one in U3b-1a.
    """

    subject: str
    _current: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dashboard(cls, principal: object, current: Callable[[], bool] | None = None) -> "MemoryAssigner":
        """Mint assignment authority from the dashboard's actual session DTO."""
        from src.dashboard.auth import DashboardPrincipal
        if not isinstance(principal, DashboardPrincipal) or "memory:assign" not in principal.permissions:
            raise PermissionError("memory_assign_denied")
        return cls(principal.subject, current)

    def require_current(self) -> None:
        """A queued assignment cannot outlive its original session."""
        if self._current is not None:
            try:
                allowed = self._current()
            except Exception:
                raise PermissionError("memory_assign_denied") from None
            if allowed is not True:
                raise PermissionError("memory_assign_denied")


@dataclass(frozen=True)
class MemoryOwnerActor:
    """Server-minted capability of one person acting on facts they own.

    Requires ``memory:own:write`` and a household principal bound to a user
    id (the authenticated account -> person relation; never a voice label).
    It authorizes ``return_owned`` and ``delete_owned`` (U3b-2).
    """

    subject: str
    person_id: str
    _current: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dashboard(cls, principal: object, current: Callable[[], bool] | None = None) -> "MemoryOwnerActor":
        """Mint owner authority from a person-bound dashboard session."""
        if not _person_principal(principal, "memory:own:write"):
            raise PermissionError("memory_owner_actor_denied")
        return cls(principal.subject, principal.user_id, current)

    def require_current(self) -> None:
        """A queued owner action cannot outlive its original session."""
        if self._current is not None:
            try:
                allowed = self._current()
            except Exception:
                raise PermissionError("memory_owner_actor_denied") from None
            if allowed is not True:
                raise PermissionError("memory_owner_actor_denied")


@dataclass(frozen=True)
class MemoryOwnerReader:
    """Server-minted capability of one person reading the facts they own (U3b-2).

    Requires ``memory:own:read`` and a household principal bound to a user id.
    The owner filter is applied in SQL from ``person_id``; nothing else selects it.
    """

    subject: str
    person_id: str
    _current: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dashboard(cls, principal: object, current: Callable[[], bool] | None = None) -> "MemoryOwnerReader":
        """Mint read authority from a person-bound dashboard session."""
        if not _person_principal(principal, "memory:own:read"):
            raise PermissionError("memory_owner_actor_denied")
        return cls(principal.subject, principal.user_id, current)

    def require_current(self) -> None:
        """A queued owner read cannot outlive its original session."""
        if self._current is not None:
            try:
                allowed = self._current()
            except Exception:
                raise PermissionError("memory_owner_actor_denied") from None
            if allowed is not True:
                raise PermissionError("memory_owner_actor_denied")


def _person_principal(principal: object, permission: str) -> bool:
    """A household dashboard principal bound to a person and holding ``permission``."""
    from src.dashboard.auth import DashboardPrincipal
    from src.dashboard.authority import DashboardRole
    return (isinstance(principal, DashboardPrincipal) and principal.role is DashboardRole.HOME
            and permission in principal.permissions and _valid_person(principal.user_id)
            and type(principal.subject) is str and bool(principal.subject))


MEMORY_PAGE_LIMIT = 100


@dataclass(frozen=True)
class FactPageCursor:
    """Keyset position on (extracted_at DESC, id ASC) (U3b V13).

    Encoded as ``k1.<16 hex digits of the big-endian IEEE-754 double>.<fact id>``:
    exact, canonical and content-free. Any other shape is invalid.
    """

    extracted_at: float
    fact_id: str

    def encode(self) -> str:
        """Serialize the position for a ``next`` field."""
        return "k1." + struct.pack(">d", self.extracted_at).hex() + "." + self.fact_id

    @classmethod
    def decode(cls, value: object) -> "FactPageCursor":
        """Parse a client cursor; any deviation raises ValueError('memory_cursor_invalid')."""
        match = _CURSOR.fullmatch(value) if type(value) is str else None
        if match is None:
            raise ValueError("memory_cursor_invalid")
        extracted_at = struct.unpack(">d", bytes.fromhex(match.group(1)))[0]
        if not math.isfinite(extracted_at):
            raise ValueError("memory_cursor_invalid")
        return cls(extracted_at, match.group(2))


_CURSOR = re.compile(r"k1\.([0-9a-f]{16})\.(fact_[0-9a-f]{64})")


@dataclass(frozen=True)
class FactPage:
    """One keyset page of projected facts and the cursor of the next one."""

    items: tuple[dict, ...]
    next_cursor: FactPageCursor | None

    def next(self) -> str | None:
        """Encoded cursor, or None on the last page."""
        return None if self.next_cursor is None else self.next_cursor.encode()


class MemoryOwnerDependents(ValueError):
    """An AUTOMATIC target has dependents and the caller did not opt to delete them."""

    code = "memory_owner_has_dependents"

    def __init__(self, dependents: list[tuple[str, str]]):
        super().__init__(self.code)
        self.dependents = dependents


@dataclass(frozen=True)
class AssignmentReceipt:
    """Revisions after an owner change; content-free."""

    fact_revision: int
    owner_revision: int
    deleted_dependents: int = 0


@dataclass(frozen=True)
class OwnedCounts:
    """Owner counts over non-deleted facts (U3b V5); also a future purge's scope."""

    currently_owned: int
    returned: int
    ever_assigned_live: int
    owned_proposals: int

    def to_dict(self) -> dict:
        """Serialize counts only."""
        return {"currently_owned": self.currently_owned, "returned": self.returned,
                "ever_assigned_live": self.ever_assigned_live, "owned_proposals": self.owned_proposals}


@dataclass(frozen=True)
class FactOwner:
    """A validated fact_owners row."""

    fact_id: str
    person_id: str
    method: FactOwnerMethod
    revision: int
    assigned_at: float
    assigned_by: str
    evidence: str


@dataclass(frozen=True)
class FactOwnerRevision:
    """A validated fact_owner_revisions row (the single owner CAS source)."""

    fact_id: str
    revision: int
    last_change: FactOwnerChange
    confirmed_by_assignment: int
    changed_at: float


def _valid_person(value: object) -> bool:
    return (type(value) is str and 1 <= len(value) <= 128
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _owner_from_row(row) -> FactOwner:
    fact_id, person_id, method, revision, assigned_at, assigned_by, evidence = tuple(row)
    try:
        if not (type(fact_id) is str and fact_id and _valid_person(person_id) and type(method) is str
                and type(revision) is int and revision >= 1 and _finite(assigned_at)
                and type(assigned_by) is str and assigned_by and type(evidence) is str
                and type(json.loads(evidence)) is dict):
            raise ValueError("invalid")
        return FactOwner(fact_id, person_id, FactOwnerMethod(method), revision, assigned_at, assigned_by, evidence)
    except ValueError:
        raise RuntimeError("fact_owner_value_unknown") from None


def _revision_from_row(row) -> FactOwnerRevision:
    fact_id, revision, last_change, confirmed, changed_at = tuple(row)
    try:
        if not (type(fact_id) is str and fact_id and type(revision) is int and revision >= 1
                and type(last_change) is str and type(confirmed) is int and confirmed in (0, 1)
                and _finite(changed_at)):
            raise ValueError("invalid")
        return FactOwnerRevision(fact_id, revision, FactOwnerChange(last_change), confirmed, changed_at)
    except ValueError:
        raise RuntimeError("fact_owner_value_unknown") from None


_OWNER_SELECT = ("SELECT fact_id,person_id,method,revision,assigned_at,assigned_by,evidence "
                 "FROM fact_owners")
_REVISION_SELECT = ("SELECT fact_id,revision,last_change,confirmed_by_assignment,changed_at "
                    "FROM fact_owner_revisions")


def _scan_fact_owner_rows(db: sqlite3.Connection) -> None:
    """Validate every owner and revision row; unknown values fail closed (1a-F4)."""
    owners = [_owner_from_row(row) for row in db.execute(_OWNER_SELECT)]
    revisions = {record.fact_id: record for record in map(_revision_from_row, db.execute(_REVISION_SELECT))}
    for owner in owners:
        record = revisions.get(owner.fact_id)
        if record is None or record.revision != owner.revision:
            raise RuntimeError("fact_owner_revision_mismatch")


def _owner_people(db: sqlite3.Connection) -> frozenset[str]:
    """Every person id referenced by an owner row; an invalid id fails closed."""
    people = frozenset(row[0] for row in db.execute("SELECT DISTINCT person_id FROM fact_owners"))
    if not all(_valid_person(person) for person in people):
        raise RuntimeError("fact_owner_value_unknown")
    return people


def _retirement_proven(retired: Callable[[frozenset[str]], bool], identities: frozenset[str]) -> bool:
    """Only an exact True from the durable-retirement proof counts; errors never do."""
    try:
        return retired(identities) is True
    except Exception:
        return False


def _scope_filter(scope: object, person_id: object) -> tuple[str, tuple]:
    """Translate a required reader scope into a SQL predicate over facts.id."""
    if not isinstance(scope, str) or scope not in {item.value for item in MemoryScope}:
        raise ValueError("invalid_memory_scope")
    scope = MemoryScope(scope)
    if scope is MemoryScope.PERSON:
        if not _valid_person(person_id):
            raise ValueError("invalid_memory_scope")
        return " AND id IN (SELECT fact_id FROM fact_owners WHERE person_id=?)", (person_id,)
    if person_id is not None:
        raise ValueError("invalid_memory_scope")
    if scope is MemoryScope.HOUSEHOLD:
        return " AND id NOT IN (SELECT fact_id FROM fact_owner_revisions)", ()
    return "", ()


def _page_arguments(limit: object, *cursors: object) -> None:
    """Validate a keyset request before any read: 1..100 items, typed cursors."""
    if type(limit) is not int or not 1 <= limit <= MEMORY_PAGE_LIMIT:
        raise ValueError("memory_page_invalid")
    if any(cursor is not None and type(cursor) is not FactPageCursor for cursor in cursors):
        raise ValueError("memory_page_invalid")


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

# U3b owner schema: the single DDL source for fresh and migrated ledgers, the
# admission fixture and every rehearsal. method/last_change/confirmed_by_assignment
# carry no CHECK; the ledger validates them against the closed enums (W12).
FACT_OWNER_SCHEMA = """
CREATE TABLE fact_owners (fact_id TEXT NOT NULL PRIMARY KEY REFERENCES facts(id), person_id TEXT NOT NULL, method TEXT NOT NULL, revision INTEGER NOT NULL, assigned_at REAL NOT NULL, assigned_by TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '{}');
CREATE INDEX fact_owners_person ON fact_owners(person_id);
CREATE TABLE fact_owner_revisions (fact_id TEXT NOT NULL PRIMARY KEY REFERENCES facts(id), revision INTEGER NOT NULL, last_change TEXT NOT NULL, confirmed_by_assignment INTEGER NOT NULL, changed_at REAL NOT NULL);
CREATE INDEX fact_owner_revisions_change ON fact_owner_revisions(last_change, changed_at);
CREATE TABLE fact_owner_schema (version INTEGER PRIMARY KEY);
INSERT INTO fact_owner_schema VALUES (1);
"""
_FACT_OWNER_TABLES = ("fact_owners", "fact_owner_revisions", "fact_owner_schema")
# PRAGMA table_info as (name, type, notnull, dflt_value, pk).
_FACT_OWNER_COLUMNS = {
    "fact_owners": [
        ("fact_id", "TEXT", 1, None, 1), ("person_id", "TEXT", 1, None, 0), ("method", "TEXT", 1, None, 0),
        ("revision", "INTEGER", 1, None, 0), ("assigned_at", "REAL", 1, None, 0),
        ("assigned_by", "TEXT", 1, None, 0), ("evidence", "TEXT", 1, "'{}'", 0)],
    "fact_owner_revisions": [
        ("fact_id", "TEXT", 1, None, 1), ("revision", "INTEGER", 1, None, 0), ("last_change", "TEXT", 1, None, 0),
        ("confirmed_by_assignment", "INTEGER", 1, None, 0), ("changed_at", "REAL", 1, None, 0)],
    "fact_owner_schema": [("version", "INTEGER", 0, None, 1)],
}
# PRAGMA index_list as (name, unique, origin, partial) plus the indexed columns.
_FACT_OWNER_INDEXES = {
    "fact_owners": {("fact_owners_person", 0, "c", 0, ("person_id",)),
                    ("sqlite_autoindex_fact_owners_1", 1, "pk", 0, ("fact_id",))},
    "fact_owner_revisions": {("fact_owner_revisions_change", 0, "c", 0, ("last_change", "changed_at")),
                             ("sqlite_autoindex_fact_owner_revisions_1", 1, "pk", 0, ("fact_id",))},
    "fact_owner_schema": set(),
}
_FACT_OWNER_FOREIGN_KEYS = {
    "fact_owners": [(0, 0, "facts", "fact_id", "id", "NO ACTION", "NO ACTION", "NONE")],
    "fact_owner_revisions": [(0, 0, "facts", "fact_id", "id", "NO ACTION", "NO ACTION", "NONE")],
    "fact_owner_schema": [],
}


def fact_owner_schema_objects() -> list[dict]:
    """The sqlite_master rows (type, name, tbl_name, sql) that FACT_OWNER_SCHEMA creates.

    SQLite stores each CREATE statement's text verbatim, so these rows are the
    byte-exact owner DDL. The ledger initializer and the payload mixed-cut rule
    both compare against them; CHECK, COLLATE, STRICT, generated columns or any
    other text drift that PRAGMA introspection cannot see is refused.
    """
    rows = []
    for statement in (part.strip() for part in FACT_OWNER_SCHEMA.split(";")):
        words = statement.split()
        if words[:2] == ["CREATE", "TABLE"]:
            rows.append({"type": "table", "name": words[2], "tbl_name": words[2], "sql": statement})
            if words[2] != "fact_owner_schema":  # INTEGER PRIMARY KEY aliases the rowid: no autoindex.
                rows.append({"type": "index", "name": "sqlite_autoindex_" + words[2] + "_1",
                             "tbl_name": words[2], "sql": None})
        elif words[:2] == ["CREATE", "INDEX"]:
            rows.append({"type": "index", "name": words[2], "tbl_name": words[4].split("(")[0], "sql": statement})
    return rows


def _fact_owner_shape_valid(db: sqlite3.Connection) -> bool:
    objects = sorted((row[0], row[1], row[2], row[3]) for row in db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN (?,?,?) OR name IN (?,?,?)",
        _FACT_OWNER_TABLES * 2))
    expected = sorted((row["type"], row["name"], row["tbl_name"], row["sql"]) for row in fact_owner_schema_objects())
    if objects != expected:
        return False
    for table in _FACT_OWNER_TABLES:
        columns = [(row[1], row[2], row[3], row[4], row[5])
                   for row in db.execute("PRAGMA table_info(" + table + ")")]
        indexes = {(row[1], row[2], row[3], row[4],
                    tuple(info[2] for info in db.execute("PRAGMA index_info(" + row[1] + ")")))
                   for row in db.execute("PRAGMA index_list(" + table + ")")}
        foreign_keys = [tuple(row) for row in db.execute("PRAGMA foreign_key_list(" + table + ")")]
        if (columns != _FACT_OWNER_COLUMNS[table] or indexes != _FACT_OWNER_INDEXES[table]
                or foreign_keys != _FACT_OWNER_FOREIGN_KEYS[table]):
            return False
    versions = [tuple(row) for row in db.execute("SELECT version FROM fact_owner_schema LIMIT 2")]
    return len(versions) == 1 and type(versions[0][0]) is int and versions[0][0] == 1


def initialize_fact_owner_schema(db: sqlite3.Connection) -> int:
    """Create or verify the owner tables atomically; prune rows of deleted facts.

    A ledger without any owner object gets the exact FACT_OWNER_SCHEMA. Any
    partial or foreign shape (table, index or marker missing, extra or
    different) raises ``fact_owner_schema_unavailable`` with nothing written.
    Every owner and revision row is then validated (unknown enum value, bad
    type or revision mismatch fails closed) before rows whose fact is
    DELETED are removed. Returns the number of pruned rows.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        present = db.execute("SELECT 1 FROM sqlite_master WHERE tbl_name IN (?,?,?) OR name IN (?,?,?) LIMIT 1",
                             _FACT_OWNER_TABLES * 2).fetchone()
        if present is None:
            for statement in FACT_OWNER_SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement.strip())
        if not _fact_owner_shape_valid(db):
            raise RuntimeError("fact_owner_schema_unavailable")
        _scan_fact_owner_rows(db)
        pruned = 0
        for table in ("fact_owners", "fact_owner_revisions"):
            pruned += db.execute("DELETE FROM " + table + " WHERE fact_id IN (SELECT id FROM facts WHERE state=?)",
                                 (FactState.DELETED.value,)).rowcount
        db.commit()
        return pruned
    except BaseException:
        db.rollback()
        raise


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
        self._lifecycle_generation = 0
        self._deleted_conversations: frozenset[str] = frozenset()
        self._deleted_times: dict[str, float] = {}
        self._pending_deleted_times: dict[str, float] = {}
        self._fact_receipts: dict[str, FactDependency] = {}
        self._pending_fact_ids: frozenset[str] = frozenset()
        self._pending_conversations: frozenset[str] = frozenset()
        self._authority_version = 0
        self._voice_projection_revision = 0
        self._voice_person_proof = VoicePersonIndex()
        # Same content-free proof pattern for owner rows (U3b-1b): the account
        # catalog reads it without ledger locks before removing an account.
        self._fact_owner_proof = VoicePersonIndex()
        # Content-free health signal (U3b-1b F2): the boot prune of retired
        # owners failed and the privacy maintenance tick is retrying it.
        self.retired_owner_prune_pending = False
        self._voice_group_revision = 0
        self._revocation_observers: list[WeakMethod] = []
        self.erasures = ErasureJournal(self)
        self.index_deletions = IndexDeletionQueue(self)

    def add_revocation_observer(self, callback: Callable[[frozenset[str]], None]) -> None:
        """Register weak metadata publication before revocation commit releases the GIL."""
        with self._lock:
            self._revocation_observers = [ref for ref in self._revocation_observers if ref() is not None]
            self._revocation_observers.append(WeakMethod(callback))

    def _publish_revocation_admission(self):
        if not self._pending_conversations:
            return
        for reference in tuple(self._revocation_observers):
            callback = reference()
            if callback is not None:
                callback(self._pending_conversations)

    @property
    def lifecycle_generation(self) -> int:
        """Identify this opening without locks, IO or reusable object addresses."""
        return self._lifecycle_generation

    def initialize(self) -> None:
        """Create the private ledger without importing legacy documents."""
        with self._lock:
            if self._db is not None:
                output_registration.borrowed_ledger(self)
                return
            output_registration.before_ledger_open(self)
            self._authority_available = False
            self._voice_person_proof.invalidate()
            self._fact_owner_proof.invalidate()
            db = None
            try:
                if self.path != ":memory:":
                    Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
                    os.close(descriptor)
                    os.chmod(self.path, 0o600)
                output_registration.before_ledger_open(self)
                db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
                output_registration.ledger_connected(self, db)
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("PRAGMA secure_delete=ON")
                initialize_voice_people_schema(db)
                initialize_voice_review_schema(db)
                db.executescript(_SCHEMA + ERASURE_SCHEMA + INDEX_DELETION_SCHEMA + LIVE_RECEIPT_SCHEMA + VOICE_SCHEMA)
                if "dependencies" not in {r[1] for r in db.execute("PRAGMA table_info(fact_sources)")}:
                    try:
                        db.execute("BEGIN IMMEDIATE")
                        db.execute("ALTER TABLE fact_sources ADD COLUMN dependencies TEXT NOT NULL DEFAULT '[]'")
                        # Old derivations have no recorded prompt parents.
                        # Quarantine them without inventing independence.
                        for row in list(db.execute("SELECT id, revision FROM facts WHERE state=?", (FactState.AUTOMATIC.value,))):
                            revision = row["revision"]+1
                            db.execute("UPDATE facts SET state=?, revision=? WHERE id=?",
                                (FactState.QUARANTINED.value, revision, row["id"]))
                            self._queue(db, row["id"], revision, "delete")
                        db.commit()
                    except BaseException:
                        db.rollback()
                        raise
                initialize_fact_owner_schema(db)
                # U3b-4a-0: voice-attribution authority (validated, never pruned at open).
                initialize_voice_attribution_schema(db)
                owners = _owner_people(db)
                receipts = self._read_fact_receipts(db)
                deleted_times = dict(db.execute("SELECT conversation_id, deleted_at FROM conversation_tombstones"))
                self._fact_receipts = receipts
                self._deleted_times = deleted_times
                self._deleted_conversations = frozenset(deleted_times)
                self._db = db
                self._lifecycle_generation += 1
                self._authority_available = True
                # Publish the owner proof from the SQL just validated; an outer
                # owner transaction still pending keeps it withdrawn.
                self._fact_owner_proof.begin(withdraw=False)
                self._fact_owner_proof.finish(True, owners)
            except BaseException:
                # No failed initializer may publish a connection that lets a
                # retry skip validation, even if native close itself fails.
                self._db = None
                self._authority_available = False
                if db is not None:
                    try:
                        db.close()
                    except BaseException:
                        output_registration.ledger_close_failed(self, db)
                        raise
                    else:
                        output_registration.ledger_closed(self, db)
                raise

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

    def conversation_committed_revoked(self, conversation_id: str) -> bool:
        """Read only published deletion authority for irreversible cleanup."""
        return self._authority_available and conversation_id in self._deleted_conversations

    def committed_revoked_conversations(self) -> frozenset[str]:
        """Return the immutable committed snapshot for indexed metadata cleanup."""
        if not self._authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        return self._deleted_conversations

    def activity_sources_available(self, sources: tuple[MemorySource, ...]) -> bool:
        """Check metadata revocations, separately from raw capture lifetimes.

        Pair this with resolved event receipts and their metadata deadline. It
        cannot authorize raw text or turn a cached proposal into write authority.
        """
        return self._authority_available and all(
            not self.conversation_revoked(source.conversation_id)
            and self.facts_available(source.fact_dependencies) for source in sources
        )

    def sources_committed_unavailable(self, sources: tuple[MemorySource, ...]) -> bool:
        """Authorize irreversible cleanup from published commits, not pending fences.

        A native event worker calls this after acquiring its SQLite write lock.
        Pending fences deny readers before commit, but a failed transaction may
        roll them back and must not trigger irreversible derived-row removal.
        Published receipt dictionaries are replaced, never mutated in place.
        """
        if not self._authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        deleted, receipts, now = self._deleted_conversations, self._fact_receipts, self.clock()
        return any(s.conversation_id in deleted or any(receipts.get(d.fact_id) != d
                   or d.expires_at is not None and now >= d.expires_at for d in s.fact_dependencies) for s in sources)

    def conversation_deleted_at(self, conversation_id: str) -> float | None:
        """Read the capture-time fence without disk or source-scope renewal."""
        return self._pending_deleted_times.get(conversation_id, self._deleted_times.get(conversation_id))

    def facts_available(self, dependencies: tuple[FactDependency, ...]) -> bool:
        """Validate committed fact receipts without SQLite or locks on-loop."""
        now = self.clock()
        return self._authority_available and all(d.fact_id not in self._pending_fact_ids and self._fact_receipts.get(d.fact_id) == d
                   and (d.expires_at is None or now < d.expires_at) for d in dependencies)

    def dependencies_committed_unavailable(self, dependencies: tuple[FactDependency, ...]) -> bool:
        """Read retired dependency authority, excluding reversible pending fences."""
        if not self._authority_available:
            raise RuntimeError('privacy_authority_unavailable')
        receipts, now = self._fact_receipts, self.clock()
        return any(receipts.get(d.fact_id) != d or d.expires_at is not None and now >= d.expires_at
                   for d in dependencies)

    def _read_fact_receipts(self, db) -> dict[str, FactDependency]:
        receipts = {}
        for fact_id, revision, expiry in db.execute(
            "SELECT id, revision, expires_at FROM facts WHERE state IN (?,?)",
            (FactState.AUTOMATIC.value, FactState.CONFIRMED.value)):
            previous = self._fact_receipts.get(fact_id)
            receipts[fact_id] = previous if previous is not None and (previous.revision, previous.expires_at) == (revision, expiry) else FactDependency(fact_id, revision, expiry)
        return receipts

    @contextmanager
    def serialize_activity_write(self):
        """Keep revocations stable while an event store validates real receipts.

        This lock is not source authorization. The event store must resolve its
        actual metadata rows and deadlines inside this scope before committing
        a derivative. Raw text continues to require ``admit_source_write``.
        Acquire this before event/derivative database locks, from an owned worker.
        """
        with self._transaction():
            if not self._authority_available:
                raise RuntimeError("privacy_authority_unavailable")
            yield

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
            if self._authority_available or self._db is not None:
                self._lifecycle_generation += 1
            self._authority_available = False
            self._voice_person_proof.invalidate()
            self._fact_owner_proof.invalidate()
            if self._db is not None:
                try:
                    self._db.close()
                except BaseException:
                    output_registration.ledger_close_failed(self, self._db)
                    raise
                else:
                    output_registration.ledger_closed(self, self._db)
                self._db = None

    @contextmanager
    def _transaction(self, guard: Callable[[], None] | None = None):
        with self._lock:
            if guard is not None:
                guard()
            self.initialize()
            if guard is not None:
                guard()
            self._db.execute("BEGIN IMMEDIATE")
            changes = self._authority_version
            recovering = bool(self._pending_fact_ids or self._pending_conversations)
            clear_pending = True
            try:
                yield self._db
                if guard is not None:
                    guard()
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
                    self._publish_revocation_admission()
                if guard is not None:
                    guard()
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
        if operation == 'delete':
            self.index_deletions.enqueue(db, fact_id, revision)
        else:
            self.index_deletions.changed()
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
            else:
                self.index_deletions.changed()
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

    def list(self, *, scope: str, person_id: str | None = None, category: str | None = None,
             proposals: bool = False, limit: int = 100) -> list[StoredFact]:
        """Read live facts in one owner scope without refreshing their retention clock.

        ``scope`` is required for every caller (U3b-2 removed the operator
        route's 'all' default); each call site is pinned by a test.
        """
        if type(limit) is not int or not 0 < limit <= 1000:
            raise ValueError("invalid_memory_limit")
        predicate, parameters = _scope_filter(scope, person_id)
        with self._transaction() as db:
            if predicate:
                _scan_fact_owner_rows(db)
            rows = db.execute(
                "SELECT id FROM facts WHERE state IN (?,?) AND (expires_at IS NULL OR expires_at>?) "
                "AND (? IS NULL OR category=?)" + predicate + " ORDER BY extracted_at DESC, id",
                ((FactState.PROPOSED if proposals else FactState.AUTOMATIC).value,
                 (FactState.PROPOSED if proposals else FactState.CONFIRMED).value,
                 self.clock(), category, category, *parameters),
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
        actor.require_current()

    def confirm(self, fact_id: str, actor: MemoryReviewer) -> StoredFact:
        """Confirm a live proposal with explicit authenticated provenance."""
        self._reviewer(actor)
        with self._transaction(actor.require_current) as db:
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
            # A deleted fact keeps no owner or ownership history (U3b pruning).
            db.execute("DELETE FROM fact_owners WHERE fact_id=?", (row["id"],))
            db.execute("DELETE FROM fact_owner_revisions WHERE fact_id=?", (row["id"],))
            self._queue(db, row["id"], revision, "delete")
        if rows:
            self._authority_version += 1
        return len(rows)

    def delete_fact(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """Reject/delete a fact through the authenticated review boundary."""
        self._reviewer(actor)
        with self._transaction(actor.require_current) as db:
            return bool(self._delete(db, fact_id))

    def reject_proposal(self, fact_id: str, actor: MemoryReviewer) -> bool:
        """A stale rejection must not delete a proposal confirmed elsewhere."""
        self._reviewer(actor)
        with self._transaction(actor.require_current) as db:
            fact = self._fact(db, fact_id)
            if fact is None:
                return False
            if fact.state is not FactState.PROPOSED:
                raise ValueError("memory_proposal_state_changed")
            return bool(self._delete(db, fact_id))

    # --- U3b-1a: per-person ownership. No production caller until U3b-1b/U3b-2. ---

    def _assigner(self, actor: MemoryAssigner) -> None:
        if type(actor) is not MemoryAssigner or type(actor.subject) is not str or not actor.subject:
            raise PermissionError("memory_assign_denied")
        actor.require_current()

    def _owner_state(self, db, fact_id: str) -> tuple[FactOwner | None, FactOwnerRevision | None]:
        """Load and validate one fact's owner rows; fact_owners mirrors the revision row."""
        row = db.execute(_OWNER_SELECT + " WHERE fact_id=?", (fact_id,)).fetchone()
        owner = None if row is None else _owner_from_row(row)
        row = db.execute(_REVISION_SELECT + " WHERE fact_id=?", (fact_id,)).fetchone()
        record = None if row is None else _revision_from_row(row)
        if owner is not None and (record is None or record.revision != owner.revision):
            raise RuntimeError("fact_owner_revision_mismatch")
        return owner, record

    def _write_owner(self, db, fact_id: str, record: FactOwnerRevision | None, change: FactOwnerChange, *,
                     person_id: str | None = None, method: FactOwnerMethod | None = None,
                     assigned_by: str | None = None, confirmed: bool = False) -> int:
        """Apply one owner change: upsert or remove the owner row, then bump the revision row."""
        if type(change) is not FactOwnerChange or (person_id is not None and type(method) is not FactOwnerMethod):
            raise RuntimeError("fact_owner_value_unknown")
        revision = (0 if record is None else record.revision) + 1
        sticky = 1 if confirmed or (record is not None and record.confirmed_by_assignment == 1) else 0
        now = self.clock()
        if person_id is None:
            db.execute("DELETE FROM fact_owners WHERE fact_id=?", (fact_id,))
        else:
            db.execute(
                "INSERT INTO fact_owners(fact_id,person_id,method,revision,assigned_at,assigned_by,evidence) "
                "VALUES (?,?,?,?,?,?,'{}') ON CONFLICT(fact_id) DO UPDATE SET person_id=excluded.person_id, "
                "method=excluded.method, revision=excluded.revision, assigned_at=excluded.assigned_at, "
                "assigned_by=excluded.assigned_by, evidence=excluded.evidence",
                (fact_id, person_id, method.value, revision, now, assigned_by))
        db.execute(
            "INSERT INTO fact_owner_revisions(fact_id,revision,last_change,confirmed_by_assignment,changed_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(fact_id) DO UPDATE SET revision=excluded.revision, "
            "last_change=excluded.last_change, confirmed_by_assignment=excluded.confirmed_by_assignment, "
            "changed_at=excluded.changed_at",
            (fact_id, revision, change.value, sticky, now))
        self._authority_version += 1
        return revision

    def _dependents(self, db, fact_id: str) -> list[tuple[str, str]]:
        """Every live fact derived (transitively) from fact_id, with its state."""
        return [(row[0], row[1]) for row in db.execute("""WITH RECURSIVE affected(id) AS (
            SELECT ? UNION SELECT d.fact_id FROM fact_dependencies d JOIN affected a ON d.source_fact_id=a.id
            ) SELECT f.id, f.state FROM facts f JOIN affected a ON f.id=a.id WHERE f.state!=? AND f.id!=?
            ORDER BY f.id""", (fact_id, FactState.DELETED.value, fact_id))]

    def assign_owner(self, fact_id: str, person_id: str, expected_fact_revision: int, expected_owner_revision: int,
                     actor: MemoryAssigner, *, retired: Callable[[frozenset[str]], bool],
                     active: Callable[[str], bool], delete_dependents: bool = False) -> AssignmentReceipt:
        """Assign an owner; assigning confirms the fact and stops its expiry (U3b P3).

        ``retired(frozenset({person_id}))`` must return exactly False and
        ``active(person_id)`` exactly True, before BEGIN, after the body and
        before commit; anything else (including an exception) refuses. The
        dependents rule applies only to the AUTOMATIC -> CONFIRMED transition.
        """
        self._assigner(actor)
        if not callable(retired) or not callable(active):
            raise PermissionError("memory_owner_authority_missing")
        if not _valid_person(person_id):
            raise ValueError("memory_owner_invalid_person")
        if type(delete_dependents) is not bool:
            raise ValueError("memory_owner_dependents_not_applicable")
        if (type(expected_fact_revision) is not int or expected_fact_revision < 1
                or type(expected_owner_revision) is not int or expected_owner_revision < 0):
            raise ValueError("memory_owner_conflict")
        targets = frozenset({person_id})

        def guard() -> None:
            actor.require_current()
            try:
                is_retired = retired(targets)
                is_active = active(person_id)
            except Exception:
                raise PermissionError("memory_owner_authority_unavailable") from None
            if is_retired is True:
                raise PermissionError("memory_owner_retired")
            if is_retired is not False:
                raise PermissionError("memory_owner_authority_unavailable")
            if is_active is False:
                raise PermissionError("memory_owner_inactive")
            if is_active is not True:
                raise PermissionError("memory_owner_authority_unavailable")

        with self._owner_transaction(guard) as db:
            fact = self._fact(db, fact_id) if type(fact_id) is str else None
            if not self._eligible(db, fact, review=True):
                raise LookupError("memory_fact_unavailable")
            owner, record = self._owner_state(db, fact_id)
            if fact.revision != expected_fact_revision or (
                    0 if record is None else record.revision) != expected_owner_revision:
                raise ValueError("memory_owner_conflict")
            if any(source.fact_dependencies for source in fact.sources) or db.execute(
                    "SELECT 1 FROM fact_dependencies WHERE fact_id=? LIMIT 1", (fact_id,)).fetchone():
                raise ValueError("memory_owner_derived_fact")
            if delete_dependents and fact.state is not FactState.AUTOMATIC:
                raise ValueError("memory_owner_dependents_not_applicable")
            if owner is not None and owner.person_id == person_id:
                return AssignmentReceipt(fact.revision, record.revision)
            deleted = 0
            confirming = fact.state in (FactState.AUTOMATIC, FactState.PROPOSED)
            if fact.state is FactState.AUTOMATIC:
                dependents = self._dependents(db, fact_id)
                if dependents and not delete_dependents:
                    raise MemoryOwnerDependents(dependents)
                deleted = sum(self._delete(db, identity) for identity, _state in dependents)
            fact_revision = fact.revision
            if confirming:
                fact_revision += 1
                db.execute("UPDATE facts SET state=?, expires_at=NULL, confirmed_at=?, confirmed_by=?, revision=? "
                           "WHERE id=?", (FactState.CONFIRMED.value, self.clock(), actor.subject, fact_revision,
                                          fact_id))
                self._queue(db, fact_id, fact_revision, "upsert")
            change = FactOwnerChange.REASSIGNED if owner is not None else FactOwnerChange.ASSIGNED
            owner_revision = self._write_owner(db, fact_id, record, change, person_id=person_id,
                                               method=FactOwnerMethod.OPERATOR_ASSIGNMENT,
                                               assigned_by=actor.subject, confirmed=confirming)
            return AssignmentReceipt(fact_revision, owner_revision, deleted)

    def unassign_owner(self, fact_id: str, expected_owner_revision: int, actor: MemoryAssigner) -> AssignmentReceipt:
        """Remove the owner as operator; the fact stays CONFIRMED and is not 'Devueltos'."""
        self._assigner(actor)
        if type(expected_owner_revision) is not int or expected_owner_revision < 0:
            raise ValueError("memory_owner_conflict")
        with self._owner_transaction(actor.require_current, withdraw=False) as db:
            fact = self._fact(db, fact_id) if type(fact_id) is str else None
            if not self._eligible(db, fact, review=True):
                raise LookupError("memory_fact_unavailable")
            owner, record = self._owner_state(db, fact_id)
            current = 0 if record is None else record.revision
            if current != expected_owner_revision:
                raise ValueError("memory_owner_conflict")
            if owner is None:
                return AssignmentReceipt(fact.revision, current)
            return AssignmentReceipt(fact.revision, self._write_owner(
                db, fact_id, record, FactOwnerChange.OPERATOR_UNASSIGNED))

    def return_owned(self, fact_id: str, person_id: str, actor: MemoryOwnerActor) -> bool:
        """'No es mío': the owner hands the fact back to the operator's 'Devueltos' queue.

        The fact stays as it is (CONFIRMED, no expiry) and ownerless. A fact the
        actor's person does not own returns False, like a missing one, after
        exactly the same statements (U3b-2: no timing class between the two).
        """
        if (type(actor) is not MemoryOwnerActor or type(actor.subject) is not str or not actor.subject
                or not _valid_person(actor.person_id) or person_id != actor.person_id):
            raise PermissionError("memory_owner_actor_denied")
        actor.require_current()
        with self._owner_transaction(actor.require_current, withdraw=False) as db:
            fact = self._owned_fact(db, fact_id, person_id)
            if fact is None:
                return False
            _owner, record = self._owner_state(db, fact_id)
            self._write_owner(db, fact_id, record, FactOwnerChange.OWNER_RETURNED)
            return True

    def _owner_actor(self, actor: object, kind: type) -> None:
        if (type(actor) is not kind or type(actor.subject) is not str or not actor.subject
                or not _valid_person(actor.person_id)):
            raise PermissionError("memory_owner_actor_denied")
        actor.require_current()

    def _owned_fact(self, db, fact_id: object, person_id: str) -> StoredFact | None:
        """The person's own live fact, or None.

        The first statement decides ownership from SQL alone, with the same
        parameters shape for a missing, a foreign and an unassigned id; only
        an owned id reads anything else (no existence oracle, no timing class).
        """
        identity = fact_id if type(fact_id) is str else ""
        if db.execute("SELECT 1 FROM fact_owners WHERE fact_id=? AND person_id=?",
                      (identity, person_id)).fetchone() is None:
            return None
        fact = self._fact(db, identity)
        if not self._eligible(db, fact, review=True):
            return None
        owner, _record = self._owner_state(db, identity)
        return fact if owner is not None and owner.person_id == person_id else None

    def delete_owned(self, fact_id: str, actor: MemoryOwnerActor) -> bool:
        """Delete one of the actor's own facts (cascade as ``_delete``, D6).

        A missing, foreign, unassigned or ineligible id returns False with the
        ledger unchanged, indistinguishable from each other.
        """
        self._owner_actor(actor, MemoryOwnerActor)
        with self._owner_transaction(actor.require_current, withdraw=False) as db:
            if self._owned_fact(db, fact_id, actor.person_id) is None:
                return False
            return bool(self._delete(db, fact_id))

    def _page(self, db, predicate: str, parameters: tuple, states: tuple[FactState, ...],
              after: FactPageCursor | None, limit: int) -> tuple[list[StoredFact], FactPageCursor | None]:
        """Keyset page: SQL selects the scope and position, eligibility is re-checked per row (V13)."""
        sql = ("SELECT id FROM facts WHERE state IN (" + ",".join("?" for _ in states) + ") "
               "AND (expires_at IS NULL OR expires_at>?) AND " + predicate)
        values = (*(state.value for state in states), self.clock(), *parameters)
        if after is not None:
            sql += " AND (extracted_at<? OR (extracted_at=? AND id>?))"
            values += (after.extracted_at, after.extracted_at, after.fact_id)
        result: list[StoredFact] = []
        more = False
        for row in db.execute(sql + " ORDER BY extracted_at DESC, id", values):
            fact = self._fact(db, row["id"])
            if not self._eligible(db, fact, review=True):
                continue
            if len(result) == limit:
                more = True
                break
            result.append(fact)
        last = result[-1] if more else None
        return result, None if last is None else FactPageCursor(float(last.extracted_at), last.id)

    def _owner_projection(self, db, fact: StoredFact) -> dict:
        """Operator view of one fact: full content (P2_A) plus its owner state."""
        owner, record = self._owner_state(db, fact.id)
        return {**fact.to_dict(),
                "owner": None if owner is None else {"person_id": owner.person_id, "method": owner.method.value,
                                                     "revision": owner.revision, "assigned_at": owner.assigned_at},
                "owner_revision": 0 if record is None else record.revision,
                "owner_change": None if record is None else record.last_change.value}

    def review_page(self, *, facts_after: FactPageCursor | None = None,
                    proposals_after: FactPageCursor | None = None, limit: int = MEMORY_PAGE_LIMIT) -> dict:
        """Operator review read with keyset pagination on both lists (U3b-2).

        Without cursors and with the default limit it returns the same lists
        as ``snapshot`` (the U3 contract), plus owner fields and next cursors.
        """
        _page_arguments(limit, facts_after, proposals_after)
        with self._transaction() as db:
            _scan_fact_owner_rows(db)
            facts, facts_next = self._page(db, "1", (), (FactState.AUTOMATIC, FactState.CONFIRMED),
                                           facts_after, limit)
            proposals, proposals_next = self._page(db, "1", (), (FactState.PROPOSED,), proposals_after, limit)
            return {"facts": [self._owner_projection(db, fact) for fact in facts],
                    "proposals": [self._owner_projection(db, fact) for fact in proposals],
                    "observed_at": self.clock(), "legacy_policy": "quarantined",
                    "pending_index_cleanup": db.execute(
                        "SELECT COUNT(*) FROM policy_index_deletions WHERE state!='complete'").fetchone()[0],
                    "pending_index": db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0],
                    "limit": limit,
                    "facts_next": None if facts_next is None else facts_next.encode(),
                    "proposals_next": None if proposals_next is None else proposals_next.encode()}

    def returned_page(self, actor: MemoryAssigner, *, after: FactPageCursor | None = None,
                      limit: int = MEMORY_PAGE_LIMIT) -> FactPage:
        """The operator's 'Devueltos' queue: ownerless facts whose last change is owner_returned."""
        self._assigner(actor)
        _page_arguments(limit, after)
        with self._transaction(actor.require_current) as db:
            _scan_fact_owner_rows(db)
            facts, cursor = self._page(
                db, "id IN (SELECT fact_id FROM fact_owner_revisions WHERE last_change=?) "
                    "AND id NOT IN (SELECT fact_id FROM fact_owners)",
                (FactOwnerChange.OWNER_RETURNED.value,),
                (FactState.AUTOMATIC, FactState.CONFIRMED, FactState.PROPOSED), after, limit)
            return FactPage(tuple(self._owner_projection(db, fact) for fact in facts), cursor)

    def owned_page(self, actor: MemoryOwnerReader, *, after: FactPageCursor | None = None,
                   limit: int = MEMORY_PAGE_LIMIT) -> FactPage:
        """'Mis recuerdos': the actor's own facts, proposals included, newest first.

        The owner filter is part of the SQL before the page limit, so a page
        never depends on other people's facts. The projection carries no
        sources, confirming subject, confidence, revisions or household counts.
        """
        self._owner_actor(actor, MemoryOwnerReader)
        _page_arguments(limit, after)
        with self._transaction(actor.require_current) as db:
            _scan_fact_owner_rows(db)
            facts, cursor = self._page(
                db, "id IN (SELECT fact_id FROM fact_owners WHERE person_id=?)", (actor.person_id,),
                (FactState.AUTOMATIC, FactState.CONFIRMED, FactState.PROPOSED), after, limit)
            return FactPage(tuple(self._person_projection(db, fact, actor.person_id) for fact in facts), cursor)

    def _person_projection(self, db, fact: StoredFact, person_id: str) -> dict:
        owner, _record = self._owner_state(db, fact.id)
        dependents = [identity for identity, _state in self._dependents(db, fact.id)]
        owned = [identity for identity in dependents if db.execute(
            "SELECT 1 FROM fact_owners WHERE fact_id=? AND person_id=?", (identity, person_id)).fetchone()]
        return {"id": fact.id, "content": fact.content, "category": fact.category, "state": fact.state.value,
                "created_at": fact.extracted_at, "expires_at": fact.expires_at,
                "owner_method": owner.method.value, "owner_assigned_at": owner.assigned_at,
                "owned_dependents": owned, "has_hidden_dependents": len(owned) < len(dependents)}

    def owned_counts(self, actor: MemoryAssigner) -> OwnedCounts:
        """Count owner state over non-deleted facts; every owner row is validated first."""
        self._assigner(actor)
        with self._transaction(actor.require_current) as db:
            _scan_fact_owner_rows(db)

            def count(sql: str) -> int:
                return db.execute(sql, (FactState.DELETED.value,)).fetchone()[0]

            return OwnedCounts(
                currently_owned=count("SELECT COUNT(*) FROM fact_owners o JOIN facts f ON f.id=o.fact_id "
                                      "WHERE f.state!=?"),
                returned=count("SELECT COUNT(*) FROM fact_owner_revisions r JOIN facts f ON f.id=r.fact_id "
                               "WHERE f.state!=? AND r.last_change='owner_returned' "
                               "AND NOT EXISTS (SELECT 1 FROM fact_owners o WHERE o.fact_id=r.fact_id)"),
                ever_assigned_live=count("SELECT COUNT(*) FROM fact_owner_revisions r JOIN facts f "
                                         "ON f.id=r.fact_id WHERE f.state!=?"),
                owned_proposals=db.execute("SELECT COUNT(*) FROM fact_owners o JOIN facts f ON f.id=o.fact_id "
                                           "WHERE f.state=?", (FactState.PROPOSED.value,)).fetchone()[0])


    # --- retirement participant (U3b-1b) --------------------------------------------

    @contextmanager
    def _owner_transaction(self, guard: Callable[[], None] | None = None, *, withdraw: bool = True):
        """Run an owner writer and republish the owner proof from actual SQL.

        A writer that can add owner rows withdraws the proof first (as
        VoicePersonLinks does). After a refusal or an uncertain commit the
        proof is rebuilt from the connection once it is outside any
        transaction; if that read is impossible it stays unavailable.
        """
        self._fact_owner_proof.begin(withdraw=withdraw)
        with self._lock:
            known, people = False, None
            try:
                with self._transaction(guard) as db:
                    yield db
                    people = _owner_people(db)
                known = True
            finally:
                if not known and self._db is not None and self._authority_available \
                        and not self._db.in_transaction:
                    try:
                        people, known = _owner_people(self._db), True
                    except Exception:
                        people, known = None, False
                self._fact_owner_proof.finish(known, people)

    def owner_refs_cleared(self, identities: frozenset[str]) -> bool:
        """Nonblocking proof that no owner row references these accounts.

        Reads one immutable snapshot; never takes ledger or account locks, so
        the account catalog may call it while holding its own lock.
        """
        snapshot = self._fact_owner_proof.snapshot
        if type(identities) is not frozenset or len(identities) > 256:
            return False
        if not all(_valid_person(identity) for identity in identities):
            return False
        return self._authority_available and snapshot.available and not identities & snapshot.users

    def clear_user_facts(self, identities: frozenset[str], retired: Callable[[frozenset[str]], bool]) -> int:
        """Delete every fact owned by durably retired accounts (D7); never creates an owner.

        ``retired(identities)`` must return exactly True before BEGIN, after
        the body and before commit (the durable retirement, not one in
        progress). Each owned fact goes through the ordinary ``_delete``
        cascade: content scrubbed, dependents deleted, index delete jobs
        queued and owner/revision rows pruned. Idempotent: a second run finds
        no owner rows and deletes nothing. Returns the number of facts deleted.
        """
        if type(identities) is not frozenset or not 1 <= len(identities) <= 256:
            raise ValueError("memory_owner_invalid_targets")
        if not callable(retired):
            raise PermissionError("memory_owner_retirement_required")
        if not all(_valid_person(identity) for identity in identities):
            raise ValueError("memory_owner_invalid_person")

        def guard() -> None:
            if not _retirement_proven(retired, identities):
                raise PermissionError("memory_owner_retirement_required")

        guard()
        with self._owner_transaction(guard) as db:
            _scan_fact_owner_rows(db)
            targets = sorted(identities)
            owned = [row[0] for row in db.execute(
                "SELECT fact_id FROM fact_owners WHERE person_id IN (" + ",".join("?" for _ in targets)
                + ") ORDER BY fact_id", targets)]
            deleted = sum(self._delete(db, fact_id) for fact_id in owned)
        if owned:
            logger.info("[FactLedger] retired-account owner cleanup: %d owned, %d deleted", len(owned), deleted)
        return deleted

    def prune_retired_owners(self, retired: Callable[[frozenset[str]], bool], *,
                             probe: Callable[[frozenset[str]], bool | None] | None = None) -> int:
        """Re-run of the D7 cascade: delete facts still owned by any durably retired account.

        Covers a crash between the authority retirement and
        ``clear_user_facts`` and owner rows brought back by a restore. Each
        referenced person is checked alone with ``retired`` (the durable
        proof); only an exact True deletes (never guesses). ``probe`` is a
        tri-state check (DashboardAuthority.memory_owner_retired) that answers
        None when the authority cannot tell: for any person not proven
        retired whose probe answer is None (or raises), the proven people are
        still deleted and then ``PermissionError('memory_owner_authority_unavailable')``
        is raised, so callers keep the prune pending instead of reporting a
        silent success (review R1). The read also republishes the owner proof.
        Returns facts deleted.
        """
        if not callable(retired) or (probe is not None and not callable(probe)):
            raise PermissionError("memory_owner_retirement_required")
        with self._lock:
            with self._owner_transaction(withdraw=False) as db:
                people = _owner_people(db)
            targets, unknown = [], False
            for person in sorted(people):
                if _retirement_proven(retired, frozenset({person})):
                    targets.append(person)
                elif probe is not None:
                    try:
                        answer = probe(frozenset({person}))
                    except Exception:
                        answer = None
                    unknown = unknown or answer is None
            deleted = sum(self.clear_user_facts(frozenset(targets[start:start + 256]), retired)
                          for start in range(0, len(targets), 256))
            if unknown:
                raise PermissionError("memory_owner_authority_unavailable")
            return deleted

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
                    "pending_index_cleanup": db.execute("SELECT COUNT(*) FROM policy_index_deletions WHERE state!='complete'").fetchone()[0],
                    "pending_index": db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0]}

    def delete_conversation(self, conversation_id: str, actor: MemoryReviewer) -> int:
        """Tombstone a conversation before scrubbing all currently derived facts."""
        self._reviewer(actor)
        conversation_id = str(UUID(conversation_id))
        with self._lock:
            with self._transaction(actor.require_current) as db:
                db.execute("INSERT OR IGNORE INTO conversation_tombstones VALUES (?,?,?)",
                           (conversation_id, self.clock(), actor.subject))
                self._authority_version += 1
                self.erasures.ensure_operation(db, conversation_id, actor)
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

    def eligible_ids(self, min_confidence: float = 0, *, scope: str, person_id: str | None = None,
                     category: str | None = None) -> list[str]:
        """Pre-filter vector candidates in one owner scope (required, U3b W5)."""
        predicate, parameters = _scope_filter(scope, person_id)
        with self._transaction() as db:
            if predicate:
                _scan_fact_owner_rows(db)
            return [r[0] for r in db.execute(
                "SELECT id FROM facts WHERE state IN (?,?) AND confidence>=? AND (expires_at IS NULL OR expires_at>?) "
                "AND (? IS NULL OR category=?)" + predicate,
                (FactState.AUTOMATIC.value, FactState.CONFIRMED.value, min_confidence, self.clock(), category, category,
                 *parameters))]

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
                    "pending_index_cleanup": db.execute("SELECT COUNT(*) FROM policy_index_deletions WHERE state!='complete'").fetchone()[0],
                    "tombstones": db.execute("SELECT COUNT(*) FROM conversation_tombstones").fetchone()[0],
                    "states": {row[0]: row[1] for row in db.execute("SELECT state, COUNT(*) FROM facts GROUP BY state")}}
