"""Content-free, append-only deletion ledger (U3b-0d3, user decision N3 = A).

Each SQLite authority that accepts user-requested deletions keeps one
``deletion_ledger_v1`` table: the privacy ledger (conversation erasures, fact
deletions, retired-owner prunes and voice-profile deletions) and the dashboard
authority (account retirements and credential revocations). A row holds only
the kind, a salted digest of the subject id and a timestamp, chained by
``row_sha256 = sha256(prev_sha256 || canonical([seq, kind, subject_sha256, at]))``
from a genesis of 64 zeros (managed-launcher design v3, N3).

Rules (launcher v3 review D3, compaction v5 review E3):

* A row is appended exactly once, by the code path that creates the deletion
  request, inside that request's transaction. ``(kind, subject_sha256)`` is
  UNIQUE and the append is insert-if-absent: every recorded subject is deleted
  for good (tombstone, DELETED state, ``deleted_at``, retirement, revoked id),
  so a repeated request, an erasure recovery or replay, a retirement re-run
  and a derived copy never add a second row.
* Automatic expiry, retention sweeps, session logout or expiry, failed logins
  and text scrubbing are never recorded: volume grows with user requests only.
* A person's own voice-consent withdrawal (user decision 2026-11-12, review
  K8) is recorded with subject ``consent_withdrawal_subject(person, revision)``,
  the revision the withdrawal wrote, so each withdrawal is its own row and a
  replay of the same request stays a no-op. Operator reopens are not recorded.
* The table is never updated or deleted; nothing but :func:`append_deletion`
  writes it. Derivation, compaction and materialization copy its rows
  byte-identically from the latest image and never call the append API.
* The per-database salt lives in ``deletion_ledger_salt_v1`` (one row,
  ``id = 1``). It is created by the first append, never rewritten, and the
  admission binding pins its row, so a changed salt changes the authority.

The admission controller (``scripts/stage_runtime.py``) recomputes the same
chain with its own stdlib copy of :func:`row_sha256`; a unit test pins both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import hmac
import json
import math
import re
import secrets
import sqlite3

LEDGER_TABLE = "deletion_ledger_v1"
SALT_TABLE = "deletion_ledger_salt_v1"
GENESIS_SHA256 = "0" * 64

# Single DDL source for fresh and migrated databases, the admission
# registration (scripts/stage_runtime.py _TABLE_COLUMNS) and payload
# derivation's additive schema groups. No triggers: admission refuses any
# view or trigger in a protected database, so append-only is enforced by the
# API and detected by the chain.
DELETION_LEDGER_SCHEMA = """
CREATE TABLE deletion_ledger_v1 (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, subject_sha256 TEXT NOT NULL, at REAL NOT NULL, prev_sha256 TEXT NOT NULL, row_sha256 TEXT NOT NULL, UNIQUE(kind, subject_sha256));
CREATE TABLE deletion_ledger_salt_v1 (id INTEGER PRIMARY KEY CHECK (id = 1), salt TEXT NOT NULL)
"""
DELETION_LEDGER_TABLES = (LEDGER_TABLE, SALT_TABLE)
_HEX64 = re.compile("[0-9a-f]{64}")


class DeletionKind(StrEnum):
    """Exactly the user-requested deletions the ledger records (design v3 N3)."""

    CONVERSATION_ERASED = "conversation_erased"
    FACT_DELETED = "fact_deleted"
    FACT_PRUNED_RETIRED_OWNER = "fact_pruned_retired_owner"
    VOICE_PROFILE_DELETED = "voice_profile_deleted"
    USER_RETIRED = "user_retired"
    CREDENTIAL_REVOKED = "credential_revoked"
    VOICE_CONSENT_WITHDRAWN = "voice_consent_withdrawn"


def consent_withdrawal_subject(person_id: str, revision: int) -> str:
    """Subject of one consent withdrawal: the person and the consent revision it wrote.

    R-b (unit 0d) treats a restored consent row of revision r as resurrected
    when a withdrawal subject exists for the same person at any revision > r.
    """
    if type(person_id) is not str or not person_id or type(revision) is not int or revision < 1:
        raise DeletionLedgerError("deletion_ledger_subject_invalid")
    return person_id + "#" + str(revision)


@dataclass(frozen=True)
class LedgerChain:
    """The constant-size projection the admission binding carries."""

    count: int
    max_seq: int
    chain_head: str
    chain_valid: bool

    def to_dict(self) -> dict:
        """The exact binding projection shape."""
        return {"count": self.count, "max_seq": self.max_seq, "chain_head": self.chain_head,
                "chain_valid": self.chain_valid}


class DeletionLedgerError(RuntimeError):
    """A content-free refusal: unknown shape, broken chain or missing salt."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def row_sha256(prev_sha256: str, seq: int, kind: str, subject_sha256: str, at: float) -> str:
    """Chain one row onto ``prev_sha256`` (the genesis is 64 zeros)."""
    return hashlib.sha256(prev_sha256.encode() + _canonical([seq, kind, subject_sha256, at])).hexdigest()


def subject_digest(salt: str, subject: str) -> str:
    """Salted, content-free digest of one subject id (HMAC-SHA256 keyed by the salt)."""
    if type(salt) is not str or _HEX64.fullmatch(salt) is None:
        raise DeletionLedgerError("deletion_ledger_salt_invalid")
    if type(subject) is not str or not subject:
        raise DeletionLedgerError("deletion_ledger_subject_invalid")
    return hmac.new(bytes.fromhex(salt), subject.encode(), hashlib.sha256).hexdigest()


def _statements() -> list[str]:
    return [part.strip() for part in DELETION_LEDGER_SCHEMA.split(";") if part.strip()]


def _objects(db: sqlite3.Connection) -> list[tuple]:
    placeholders = ",".join("?" for _ in DELETION_LEDGER_TABLES)
    return sorted(tuple(row) for row in db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN (" + placeholders + ") "
        "OR name IN (" + placeholders + ")", DELETION_LEDGER_TABLES * 2))


def _pragmas(db: sqlite3.Connection) -> dict:
    shape = {}
    for table in DELETION_LEDGER_TABLES:
        shape[table] = (
            [tuple(row) for row in db.execute("PRAGMA table_info(" + table + ")")],
            sorted((row[1], row[2], row[3], row[4],
                    tuple(info[2] for info in db.execute("PRAGMA index_info(" + row[1] + ")")))
                   for row in db.execute("PRAGMA index_list(" + table + ")")),
            [tuple(row) for row in db.execute("PRAGMA foreign_key_list(" + table + ")")])
    return shape


def _reference() -> tuple[list[tuple], dict]:
    """The exact objects and PRAGMA shape the DDL produces, built in memory."""
    reference = sqlite3.connect(":memory:")
    try:
        for statement in _statements():
            reference.execute(statement)
        return _objects(reference), _pragmas(reference)
    finally:
        reference.close()


# Built once at import: initialisation opens no connection besides the caller's.
_REFERENCE = _reference()


def deletion_ledger_schema_objects() -> list[dict]:
    """The sqlite_master rows (type, name, tbl_name, sql) DELETION_LEDGER_SCHEMA creates."""
    objects, _shape = _REFERENCE
    return [{"type": kind, "name": name, "tbl_name": table, "sql": sql} for kind, name, table, sql in objects]


def _salt(db: sqlite3.Connection) -> str | None:
    rows = [tuple(row) for row in db.execute("SELECT id, salt FROM " + SALT_TABLE + " LIMIT 2")]
    if not rows:
        return None
    if len(rows) != 1 or rows[0][0] != 1 or type(rows[0][1]) is not str or _HEX64.fullmatch(rows[0][1]) is None:
        raise DeletionLedgerError("deletion_ledger_salt_invalid")
    return rows[0][1]


def verify_chain(db: sqlite3.Connection, *, upto: int | None = None) -> LedgerChain:
    """Recompute the chain in seq order; any break raises ``deletion_ledger_chain_broken``.

    Args:
        db: A connection whose schema holds ``deletion_ledger_v1``.
        upto: When given, stop after ``seq == upto`` (forward-evolution checks).

    Returns:
        The constant-size projection of the verified prefix.
    """
    kinds = {item.value for item in DeletionKind}
    head, expected = GENESIS_SHA256, 1
    query = "SELECT seq,kind,subject_sha256,at,prev_sha256,row_sha256 FROM " + LEDGER_TABLE
    parameters: tuple = ()
    if upto is not None:
        query += " WHERE seq<=?"
        parameters = (upto,)
    for seq, kind, subject, at, prev, digest in db.execute(query + " ORDER BY seq", parameters):
        if (type(seq) is not int or seq != expected or kind not in kinds
                or type(subject) is not str or _HEX64.fullmatch(subject) is None
                or type(at) is not float or not math.isfinite(at) or prev != head
                or digest != row_sha256(prev, seq, kind, subject, at)):
            raise DeletionLedgerError("deletion_ledger_chain_broken")
        head, expected = digest, expected + 1
    count = expected - 1
    return LedgerChain(count, count, head, True)


def ensure_deletion_ledger(db: sqlite3.Connection) -> LedgerChain:
    """Create or verify the ledger inside the caller's open transaction; never prunes.

    A database without either table gets the exact DELETION_LEDGER_SCHEMA. Any
    partial or foreign shape raises ``deletion_ledger_schema_unavailable``.
    Every row is then chained again (``deletion_ledger_chain_broken``) and a
    non-empty ledger must have its salt (``deletion_ledger_salt_missing``).
    No row is written here, so opening a copied state never changes the ledger.
    """
    if not db.in_transaction:
        raise DeletionLedgerError("deletion_ledger_transaction_required")
    if not _objects(db):
        for statement in _statements():
            db.execute(statement)
    objects, shape = _REFERENCE
    if _objects(db) != objects or _pragmas(db) != shape:
        raise DeletionLedgerError("deletion_ledger_schema_unavailable")
    chain = verify_chain(db)
    if chain.count and _salt(db) is None:
        raise DeletionLedgerError("deletion_ledger_salt_missing")
    return chain


def initialize_deletion_ledger(db: sqlite3.Connection) -> LedgerChain:
    """:func:`ensure_deletion_ledger` in its own IMMEDIATE transaction (autocommit connections)."""
    db.execute("BEGIN IMMEDIATE")
    try:
        chain = ensure_deletion_ledger(db)
        db.commit()
        return chain
    except BaseException:
        db.rollback()
        raise


def _write_once_salt(db: sqlite3.Connection) -> str:
    """Return the salt, creating it only while the ledger is still empty."""
    salt = _salt(db)
    if salt is not None:
        return salt
    if db.execute("SELECT 1 FROM " + LEDGER_TABLE + " LIMIT 1").fetchone() is not None:
        raise DeletionLedgerError("deletion_ledger_salt_missing")
    salt = secrets.token_hex(32)
    db.execute("INSERT INTO " + SALT_TABLE + " VALUES (1, ?)", (salt,))
    return salt


def append_deletion(db: sqlite3.Connection, kind: DeletionKind, subject: str, at: float) -> bool:
    """Append one row inside the deletion request's own open transaction (insert-if-absent).

    Args:
        db: The authority connection, with the request's transaction open.
        kind: One of the six recorded kinds.
        subject: The deleted subject's id; only its salted digest is stored.
        at: The request time (the authority's clock).

    Returns:
        True when a row was appended, False when ``(kind, subject)`` was already recorded.
    """
    if type(kind) is not DeletionKind:
        raise DeletionLedgerError("deletion_ledger_kind_invalid")
    if type(at) not in (int, float) or not math.isfinite(at):
        raise DeletionLedgerError("deletion_ledger_time_invalid")
    if not db.in_transaction:
        raise DeletionLedgerError("deletion_ledger_transaction_required")
    digest = subject_digest(_write_once_salt(db), subject)
    if db.execute("SELECT 1 FROM " + LEDGER_TABLE + " WHERE kind=? AND subject_sha256=?",
                  (kind.value, digest)).fetchone() is not None:
        return False
    last = db.execute("SELECT seq, row_sha256 FROM " + LEDGER_TABLE + " ORDER BY seq DESC LIMIT 1").fetchone()
    seq, prev = (1, GENESIS_SHA256) if last is None else (last[0] + 1, last[1])
    at = float(at)
    db.execute("INSERT INTO " + LEDGER_TABLE + " VALUES (?,?,?,?,?,?)",
               (seq, kind.value, digest, at, prev, row_sha256(prev, seq, kind.value, digest, at)))
    return True


def deletion_recorded(db: sqlite3.Connection, kind: DeletionKind, subject: str) -> bool:
    """Whether ``(kind, subject)`` is already in the ledger (no salt yet means nothing is)."""
    salt = _salt(db)
    if salt is None:
        return False
    return db.execute("SELECT 1 FROM " + LEDGER_TABLE + " WHERE kind=? AND subject_sha256=?",
                      (kind.value, subject_digest(salt, subject))).fetchone() is not None


def recorded_subjects(db: sqlite3.Connection, kinds: frozenset[DeletionKind]) -> frozenset[str]:
    """Salted subject digests of the given kinds (for the R-b resurrection check)."""
    if not kinds or any(type(kind) is not DeletionKind for kind in kinds):
        raise DeletionLedgerError("deletion_ledger_kind_invalid")
    values = sorted(kind.value for kind in kinds)
    return frozenset(row[0] for row in db.execute(
        "SELECT subject_sha256 FROM " + LEDGER_TABLE + " WHERE kind IN (" + ",".join("?" for _ in values) + ")",
        values))


def resurrected(db: sqlite3.Connection, kinds: frozenset[DeletionKind], candidates) -> int:
    """Count candidate ids whose salted digest matches a recorded subject (content-free).

    Ids are hashed in memory with the database's own salt and compared; only
    the count leaves this function. An empty ledger (no salt) matches nothing.
    """
    salt = _salt(db)
    if salt is None:
        return 0
    recorded = recorded_subjects(db, kinds)
    return sum(1 for candidate in set(candidates) if subject_digest(salt, candidate) in recorded)
