"""Bounded proposal expiry with resumable metadata-only source checks."""

from dataclasses import dataclass
import hashlib
import json
import sqlite3

from src.analytics.event_evidence_reader import EventEvidenceReader, EvidenceState
from src.analytics.suggestion_storage import (
    BATCH_SIZE,
    existing_suggestion_database,
    suggestion_metadata,
)
from src.memory.fact_ledger import FactLedger

REFERENCE_BUDGET = 128


@dataclass(frozen=True)
class SweepPosition:
    """No private rows or source arrays survive between native batches."""

    after: int = 0
    offset: int = 0
    identity: str | None = None
    unknown: bool = False
    deadline: float | None = None


@dataclass(frozen=True)
class SuggestionSweep:
    """Progress and anchored deadlines are safe to retain after cancellation."""

    checked_at: float
    cursor: SweepPosition | None = None
    next_at: float | None = None
    deleted: int = 0
    pending: int = 0
    references: int = 0


def sweep_suggestions(
    ledger: FactLedger,
    descriptor: dict,
    position: SweepPosition | None,
    event_targets: dict,
    retention_days: float,
) -> SuggestionSweep:
    """Expire 32 proposal rows and resolve at most 128 event references per batch."""
    if not ledger.authority_available:
        raise RuntimeError("suggestion_retention_authority_unavailable")
    position = position or SweepPosition()
    now = ledger.clock()
    deleted = pending = references = scanned = 0
    deadlines = []
    with (
        existing_suggestion_database(descriptor) as db,
        EventEvidenceReader(ledger, event_targets, retention_days) as reader,
    ):
        if db is None:
            return SuggestionSweep(now)
        if position.after == 0:
            pending = db.execute(
                "SELECT COUNT(*) FROM suggestions_policy_v1 WHERE row_id<=0"
            ).fetchone()[0]
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT row_id,id,created_at,observed_at,expires_at,sources_json,evidence_json
            FROM suggestions_policy_v1 WHERE row_id>? OR (row_id=? AND ?>0) ORDER BY row_id LIMIT ?""",
            (
                position.after,
                position.after,
                int(position.identity is not None),
                BATCH_SIZE,
            ),
        )
        after = position.after
        for row in rows:
            scanned += 1
            after = row["row_id"]
            try:
                metadata = suggestion_metadata(row)
            except (ValueError, TypeError, KeyError, AttributeError):
                pending += 1
                continue
            deadline = min(
                metadata.expires_at,
                metadata.observed_at + ledger.automatic_days * 86400,
            )
            identity = hashlib.sha256(
                json.dumps(
                    [
                        row[key]
                        for key in (
                            "id",
                            "created_at",
                            "observed_at",
                            "sources_json",
                            "evidence_json",
                        )
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            continuing = (
                after == position.after
                and position.identity == identity
                and position.identity is not None
            )
            offset = position.offset if continuing else 0
            unknown = position.unknown if continuing else False
            if continuing and position.deadline is not None:
                deadline = min(deadline, position.deadline)
            retired = (
                ledger.sources_committed_unavailable(metadata.sources)
                or now >= deadline
            )
            while (
                not retired
                and offset < len(metadata.evidence)
                and references < REFERENCE_BUDGET
            ):
                check = reader.check(metadata.evidence[offset])
                references += 1
                offset += 1
                retired = check.state is EvidenceState.RETIRED
                unknown |= check.state is EvidenceState.UNKNOWN
                if check.next_at is not None:
                    deadline = min(deadline, check.next_at)
                if check.root_unknown:
                    offset = len(metadata.evidence)
                    break
            if retired or ledger.clock() >= deadline:
                db.execute("DELETE FROM suggestions_policy_v1 WHERE row_id=?", (after,))
                deleted += 1
                continue
            if deadline < metadata.expires_at:
                db.execute(
                    "UPDATE suggestions_policy_v1 SET expires_at=?,snooze_until=MIN(snooze_until,?) WHERE row_id=?",
                    (deadline, deadline, after),
                )
            deadlines.append(deadline)
            if offset < len(metadata.evidence):
                return SuggestionSweep(
                    now,
                    SweepPosition(after, offset, identity, unknown, deadline),
                    min(deadlines),
                    deleted,
                    pending,
                    references,
                )
            pending += int(unknown)
            if references >= REFERENCE_BUDGET:
                return SuggestionSweep(
                    now,
                    SweepPosition(after),
                    min(deadlines),
                    deleted,
                    pending,
                    references,
                )
        return SuggestionSweep(
            now,
            SweepPosition(after) if scanned == BATCH_SIZE else None,
            min(deadlines) if deadlines else None,
            deleted,
            pending,
            references,
        )
