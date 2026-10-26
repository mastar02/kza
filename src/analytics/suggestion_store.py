"""Private policy proposals; legacy JSON is never read, replaced or attributed."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from src.analytics.event_evidence import EventDependency
from src.analytics.event_logger import EventLogger
from src.analytics.pattern_analyzer import Pattern
from src.analytics.pattern_projection import pattern_key, pattern_spec, project_pattern
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import MemorySource
from src.analytics.suggestion_storage import MAX_METADATA_BYTES, suggestion_descriptor

MAX_RECORDS = 1000


class SuggestionStatus(StrEnum):
    """A voice response accepts a proposal; implementation needs a real receipt."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SNOOZED = "snoozed"
    IMPLEMENTED = "implemented"  # Reserved for a future verified executor.


@dataclass(frozen=True)
class SuggestionRecord:
    """A fresh projection with original dates and actual metadata provenance."""

    id: str
    pattern: Pattern
    message: str
    created_at: float
    expires_at: float
    status: SuggestionStatus
    snooze_until: float | None
    sources: tuple[MemorySource, ...]


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


class SuggestionStore:
    """Commit only event-derived proposals while both authority guards are held.

    All methods perform blocking I/O and must run in the EventLogger's retained
    native scope. Messages may be generated outside the guards, but final commit
    resolves the exact events again. No in-memory cache can refill a revoked row.
    """

    def __init__(self, events: EventLogger, legacy_path: str | Path):
        self.events = events
        self.policy = events.source_policy
        if self.policy is None or not self.policy.authority_available:
            raise PermissionError("suggestions_require_privacy_authority")
        events.initialize()
        self.descriptor = suggestion_descriptor(legacy_path)
        self.path = Path(self.descriptor["path"])
        if (
            self.path == events.db_path
            or self.policy.path != ":memory:"
            and self.path == Path(self.policy.path).resolve()
        ):
            raise ValueError("suggestion_store_location_conflict")
        # Register before opening the actual target, including historical paths.
        self.policy.erasures.register_target(
            ErasureParticipant.SUGGESTIONS, 1, self.descriptor
        )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS suggestions_policy_v1 (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                pattern_key TEXT NOT NULL, created_at REAL NOT NULL, observed_at REAL NOT NULL,
                expires_at REAL NOT NULL, status TEXT NOT NULL, snooze_until REAL,
                pattern_json TEXT NOT NULL, evidence_json TEXT NOT NULL, sources_json TEXT NOT NULL,
                message TEXT NOT NULL
            )""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestion_expiry ON suggestions_policy_v1(expires_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestion_pattern ON suggestions_policy_v1(pattern_key)"
            )

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            yield db

    def prepare(self, pattern: Pattern) -> Pattern:
        """Regenerate descriptions/statistics before supplying any model prompt."""
        inputs = self.events.resolve_pattern_evidence(pattern.evidence)
        if inputs is None:
            raise PermissionError("suggestion_evidence_unavailable")
        return project_pattern(pattern_spec(pattern), inputs)

    def _materialize(self, db, row) -> SuggestionRecord | None:
        now = self.policy.clock()
        try:
            dates = (row["created_at"], row["observed_at"], row["expires_at"])
            if (
                not self.policy.authority_available
                or type(row["row_id"]) is not int
                or row["row_id"] <= 0
                or not all(_number(v) for v in dates)
                or not row["observed_at"]
                <= row["created_at"]
                <= now
                < row["expires_at"]
                or re.fullmatch(r"sug_[0-9a-f]{32}", row["id"]) is None
            ):
                return None
            status = SuggestionStatus(row["status"])
            if status is SuggestionStatus.IMPLEMENTED:
                return None
            snooze = row["snooze_until"]
            if snooze is not None and (
                not _number(snooze)
                or not row["created_at"] <= snooze <= row["expires_at"]
            ):
                return None
            if status is SuggestionStatus.SNOOZED and snooze is None:
                return None
            for key in ("pattern_json", "evidence_json", "sources_json"):
                if (
                    type(row[key]) is not str
                    or len(row[key].encode()) > MAX_METADATA_BYTES
                ):
                    return None
            encoded = json.loads(row["evidence_json"])
            if type(encoded) is not list or not 0 < len(encoded) <= 10_000:
                return None
            evidence = tuple(EventDependency.from_dict(d) for d in encoded)
            inputs = self.events.resolve_pattern_evidence(evidence)
            if (
                inputs is None
                or row["observed_at"] != inputs.observed_at
                or row["expires_at"]
                > min(
                    inputs.expires_at,
                    inputs.observed_at + self.policy.automatic_days * 86400,
                )
            ):
                return None
            # A stored lineage is cleanup metadata, never self-authorizing.
            if json.loads(row["sources_json"]) != [s.to_dict() for s in inputs.sources]:
                return None
            pattern = project_pattern(json.loads(row["pattern_json"]), inputs)
            if pattern_key(pattern) != row["pattern_key"]:
                return None
            # Free text is projected only after all dates and actual inputs pass.
            message = db.execute(
                "SELECT message FROM suggestions_policy_v1 WHERE row_id=?",
                (row["row_id"],),
            ).fetchone()[0]
            if type(message) is not str or not 0 < len(message) <= 4096:
                return None
            if self.policy.clock() >= row[
                "expires_at"
            ] or not self.policy.activity_sources_available(inputs.sources):
                return None
            return SuggestionRecord(
                row["id"],
                pattern,
                message,
                row["created_at"],
                row["expires_at"],
                status,
                snooze,
                inputs.sources,
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            # Unknown rows remain quarantined in place, outside automatic use.
            return None

    def records(
        self,
        *,
        pending: bool = False,
        suggestion_id: str | None = None,
        selection_key: str | None = None,
    ) -> list[SuggestionRecord]:
        """Resolve fresh eligible projections without hydrating legacy/free text."""
        result = []
        now = self.policy.clock()
        with self._connect() as db:
            db.execute("BEGIN")
            cursor = db.execute(
                """SELECT row_id,id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json FROM suggestions_policy_v1
                WHERE expires_at>? AND (? IS NULL OR id=?)
                AND (? IS NULL OR pattern_key=?)
                AND (?=0 OR status='pending' OR (status='snoozed' AND snooze_until<=?))
                ORDER BY created_at DESC,row_id DESC LIMIT ?""",
                (
                    now,
                    suggestion_id,
                    suggestion_id,
                    selection_key,
                    selection_key,
                    int(pending),
                    now,
                    MAX_RECORDS,
                ),
            )
            for row in cursor:
                record = self._materialize(db, row)
                if record is not None:
                    result.append(record)
        return result

    def create(
        self, pattern: Pattern, message: str, created_at: float
    ) -> SuggestionRecord | None:
        """Atomically admit a new proposal; late output cannot cross a marker."""
        if (
            type(message) is not str
            or not 0 < len(message) <= 4096
            or not _number(created_at)
        ):
            raise ValueError("invalid_suggestion_output")
        with self.events.admit_pattern_write(pattern.evidence) as inputs:
            normalized = project_pattern(pattern_spec(pattern), inputs)
            expiry = min(
                inputs.expires_at,
                inputs.observed_at + self.policy.automatic_days * 86400,
            )
            if not inputs.observed_at <= created_at <= self.policy.clock() < expiry:
                raise PermissionError("suggestion_expired")
            encoded = (
                _json(pattern_spec(normalized)),
                _json([d.to_dict() for d in normalized.evidence]),
                _json([source.to_dict() for source in inputs.sources]),
            )
            if any(len(value.encode()) > MAX_METADATA_BYTES for value in encoded):
                raise ValueError("suggestion_metadata_capacity")
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute(
                    "SELECT row_id,id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json FROM suggestions_policy_v1 WHERE pattern_key=? AND expires_at>?",
                    (pattern_key(normalized), self.policy.clock()),
                )
                if any(self._materialize(db, row) is not None for row in existing):
                    return None
                if (
                    db.execute(
                        "SELECT COUNT(*) FROM suggestions_policy_v1 WHERE expires_at>?",
                        (self.policy.clock(),),
                    ).fetchone()[0]
                    >= MAX_RECORDS
                ):
                    raise RuntimeError("suggestion_capacity")
                identifier = "sug_" + uuid4().hex
                db.execute(
                    """INSERT INTO suggestions_policy_v1
                    (id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json,message)
                    VALUES (?,?,?,?,?,'pending',NULL,?,?,?,?)""",
                    (
                        identifier,
                        pattern_key(normalized),
                        created_at,
                        inputs.observed_at,
                        expiry,
                        *encoded,
                        message,
                    ),
                )
                if self.policy.clock() >= expiry:
                    raise PermissionError("suggestion_expired")
            return SuggestionRecord(
                identifier,
                normalized,
                message,
                created_at,
                expiry,
                SuggestionStatus.PENDING,
                None,
                inputs.sources,
            )

    def respond(
        self, suggestion_id: str, *, accept: bool, snooze_hours: float = 0
    ) -> SuggestionRecord | None:
        """Update a live proposal under the same authority/event commit fences."""
        if type(accept) is not bool or not _number(snooze_hours) or snooze_hours < 0:
            raise ValueError("invalid_suggestion_response")
        records = self.records(suggestion_id=suggestion_id)
        if not records:
            return None
        selected = records[0]
        with (
            self.events.admit_pattern_write(selected.pattern.evidence),
            self._connect() as db,
        ):
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT row_id,id,pattern_key,created_at,observed_at,expires_at,status,snooze_until,pattern_json,evidence_json,sources_json FROM suggestions_policy_v1 WHERE id=?",
                (suggestion_id,),
            ).fetchone()
            current = self._materialize(db, row) if row is not None else None
            if current is None or current.status not in (
                SuggestionStatus.PENDING,
                SuggestionStatus.SNOOZED,
            ):
                return None
            status = (
                SuggestionStatus.SNOOZED
                if snooze_hours
                else SuggestionStatus.ACCEPTED
                if accept
                else SuggestionStatus.REJECTED
            )
            snooze = (
                min(self.policy.clock() + snooze_hours * 3600, current.expires_at)
                if snooze_hours
                else None
            )
            db.execute(
                "UPDATE suggestions_policy_v1 SET status=?,snooze_until=? WHERE id=?",
                (status.value, snooze, suggestion_id),
            )
            if self.policy.clock() >= current.expires_at:
                raise PermissionError("suggestion_expired")
            return SuggestionRecord(
                current.id,
                current.pattern,
                current.message,
                current.created_at,
                current.expires_at,
                status,
                snooze,
                current.sources,
            )
