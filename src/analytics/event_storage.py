"""Cleanup-only access to existing policy event tables and source metadata."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3

from src.memory.fact_ledger import FactLedger, MemorySource


def event_descriptor(path: str | Path) -> dict:
    """Resolve the physical writer location before opening any event database."""
    return {"path": str(Path(path).resolve()), "table": "events_policy_v1"}


def validate_event_descriptor(descriptor: dict) -> None:
    """Reject unknown protocols before touching a registered private path."""
    if (
        set(descriptor) != {"path", "table"}
        or descriptor["table"] != "events_policy_v1"
        or type(descriptor["path"]) is not str
        or not Path(descriptor["path"]).is_absolute()
    ):
        raise ValueError("invalid_event_target")


@contextmanager
def existing_event_database(descriptor: dict):
    """Yield an existing policy table transaction, without creation or migration."""
    validate_event_descriptor(descriptor)
    path = Path(descriptor["path"])
    try:
        path.stat()
    except FileNotFoundError:
        yield None
        return
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
        table = db.execute(
            "SELECT type FROM sqlite_master WHERE name='events_policy_v1'"
        ).fetchone()
        if table is None:
            yield None
            return
        if table[0] != "table":
            raise ValueError("unknown_event_schema")
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM events_policy_v1 WHERE id<=0 LIMIT 1").fetchone():
            raise ValueError("unknown_event_row_id")
        yield db


def event_sources(encoded: str, raw: bool) -> tuple[MemorySource, ...]:
    """Validate metadata without attributing legacy or unsourced raw content."""
    decoded = json.loads(encoded)
    if type(decoded) is not list or len(decoded) > 256:
        raise ValueError("unknown_event_sources")
    sources = tuple(MemorySource.from_dict(value) for value in decoded)
    if not sources and raw:
        raise ValueError("unattributed_event_text")
    return sources


@dataclass(frozen=True)
class EventSweep:
    """Only metadata progress and anchored deadlines survive native completion."""

    checked_at: float
    cursor: int | None = None
    next_at: float | None = None
    deleted: int = 0
    scrubbed: int = 0


def sweep_events(
    descriptor: dict, cursor: int, ledger: FactLedger, retention_days: float
) -> EventSweep:
    """Apply persisted TTL/source authority to at most 128 existing policy rows."""
    if not ledger.authority_available:
        raise RuntimeError("event_retention_authority_unavailable")
    now = ledger.clock()
    with existing_event_database(descriptor) as db:
        if db is None:
            return EventSweep(now)
        now = ledger.clock()
        rows = db.execute(
            """SELECT id,timestamp,expires_at,text_expires_at,sources_json,
            (trigger_phrase IS NOT NULL OR user_name IS NOT NULL OR context_json IS NULL OR context_json!=metrics_json)
            FROM events_policy_v1 WHERE id>? ORDER BY id LIMIT 128""",
            (cursor,),
        ).fetchall()
        deleted = scrubbed = 0
        deadlines = []
        for row_id, captured, metadata_expiry, raw_expiry, encoded, raw in rows:
            if (
                any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in (captured, metadata_expiry)
                )
                or metadata_expiry <= captured
            ):
                raise ValueError("unknown_event_deadline")
            sources = event_sources(encoded, raw)
            if sources and (
                type(raw_expiry) not in (int, float) or not math.isfinite(raw_expiry)
            ):
                raise ValueError("unknown_event_raw_deadline")
            expiry = min(metadata_expiry, captured + retention_days * 86400)
            text_expiry = (
                min(raw_expiry, *(s.expires_at for s in sources))
                if sources
                else raw_expiry
            )
            revoked = ledger.sources_committed_unavailable(sources)
            if expiry <= now or revoked:
                db.execute("DELETE FROM events_policy_v1 WHERE id=?", (row_id,))
                deleted += 1
                continue
            # Persist shorter limits now. A later configuration expansion or
            # restart cannot renew an already accepted deadline.
            if expiry != metadata_expiry or text_expiry != raw_expiry:
                db.execute(
                    "UPDATE events_policy_v1 SET expires_at=?,text_expires_at=? WHERE id=?",
                    (expiry, text_expiry, row_id),
                )
            if raw and text_expiry <= now:
                db.execute(
                    """UPDATE events_policy_v1 SET trigger_phrase=NULL,user_name=NULL,
                    context_json=metrics_json WHERE id=?""",
                    (row_id,),
                )
                scrubbed += 1
            elif raw:
                deadlines.append(text_expiry)
            deadlines.append(expiry)
            deadlines.extend(
                d.expires_at
                for s in sources
                for d in s.fact_dependencies
                if d.expires_at is not None
            )
        return EventSweep(
            now,
            rows[-1][0] if len(rows) == 128 else None,
            min(deadlines) if deadlines else None,
            deleted,
            scrubbed,
        )
