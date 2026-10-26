"""Read-only retirement checks for registered event metadata, without a logger."""

from contextlib import ExitStack, closing
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
import re
import sqlite3

from src.analytics.event_evidence import EventDependency, project_pattern_event
from src.analytics.event_storage import event_sources, validate_event_descriptor
from src.memory.fact_ledger import FactLedger


class EvidenceState(StrEnum):
    AVAILABLE = "available"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceCheck:
    """Only status and original/effective deadlines leave a metadata lookup."""

    state: EvidenceState
    next_at: float | None = None
    root_unknown: bool = False


class EventEvidenceReader:
    """Bounded read-only source checks; no file creation, raw projection or models."""

    def __init__(self, ledger: FactLedger, targets: dict, retention_days: float):
        self.ledger, self.targets, self.retention_days = ledger, targets, retention_days
        self._stack = ExitStack()
        self._databases = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)

    def _database(self, target: str):
        if target in self._databases:
            return self._databases[target]
        self._databases[target] = None
        descriptor = self.targets.get(target)
        if descriptor is None:
            return None
        validate_event_descriptor(descriptor)
        path = Path(descriptor["path"])
        if path.is_symlink():
            return None
        try:
            path.stat()
        except FileNotFoundError:
            return None
        db = self._stack.enter_context(
            closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.05))
        )
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        if (
            db.execute(
                "SELECT type FROM sqlite_master WHERE name='events_policy_v1'"
            ).fetchone()
            is None
        ):
            return None
        columns = db.execute("PRAGMA table_info(events_policy_v1)").fetchall()
        required = {
            "id",
            "timestamp",
            "event_type",
            "entity_id",
            "action",
            "user_id",
            "sources_json",
            "expires_at",
            "text_expires_at",
            "user_name",
            "trigger_phrase",
            "context_json",
            "metrics_json",
        }
        if not required <= {c["name"] for c in columns} or not any(
            c["name"] == "id" and c["type"].upper() == "INTEGER" and c["pk"] == 1
            for c in columns
        ):
            return None
        self._databases[target] = db
        return db

    def check(self, dependency: EventDependency) -> EvidenceCheck:
        """Treat committed absence/change as retirement; unavailable roots are unknown."""
        if not self.ledger.authority_available:
            raise RuntimeError("suggestion_retention_authority_unavailable")
        try:
            db = self._database(dependency.target_id)
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ):
            self._databases[dependency.target_id] = None
            return EvidenceCheck(EvidenceState.UNKNOWN, root_unknown=True)
        try:
            if db is None:
                return EvidenceCheck(EvidenceState.UNKNOWN, root_unknown=True)
            row = db.execute(
                """SELECT id,timestamp,event_type,entity_id,action,user_id,sources_json,expires_at,text_expires_at,
                (user_name IS NOT NULL OR trigger_phrase IS NOT NULL OR context_json IS NULL OR context_json!=metrics_json) AS raw
                FROM events_policy_v1 WHERE id=?""",
                (dependency.event_id,),
            ).fetchone()
            if row is None:
                return EvidenceCheck(EvidenceState.RETIRED)
            if (
                not all(
                    type(row[key]) in (int, float) and math.isfinite(row[key])
                    for key in ("timestamp", "expires_at")
                )
                or row["expires_at"] <= row["timestamp"]
                or row["timestamp"] > self.ledger.clock()
                or type(row["sources_json"]) is not str
                or len(row["sources_json"].encode()) > 4 * 1024 * 1024
            ):
                return EvidenceCheck(EvidenceState.UNKNOWN)
            for key in ("event_type", "entity_id", "action", "user_id"):
                if key == "user_id" and row[key] is None:
                    continue
                if (
                    type(row[key]) is not str
                    or re.fullmatch(r"[\w.:-]{1,256}", row[key]) is None
                ):
                    return EvidenceCheck(EvidenceState.UNKNOWN)
            sources = event_sources(row["sources_json"], row["raw"])
            if sources and (
                type(row["text_expires_at"]) not in (int, float)
                or not math.isfinite(row["text_expires_at"])
            ):
                return EvidenceCheck(EvidenceState.UNKNOWN)
            actual = project_pattern_event(row, dependency.target_id).dependency
            if actual != dependency or self.ledger.sources_committed_unavailable(
                sources
            ):
                return EvidenceCheck(EvidenceState.RETIRED)
            deadline = min(
                actual.expires_at, row["timestamp"] + self.retention_days * 86400
            )
            return EvidenceCheck(
                EvidenceState.RETIRED
                if self.ledger.clock() >= deadline
                else EvidenceState.AVAILABLE,
                deadline,
            )
        except (OSError, sqlite3.Error):
            self._databases[dependency.target_id] = None
            return EvidenceCheck(EvidenceState.UNKNOWN, root_unknown=True)
        except (ValueError, TypeError, KeyError, AttributeError):
            # One quarantined row must not hide a later committed retirement
            # in the same supported source database.
            return EvidenceCheck(EvidenceState.UNKNOWN)
