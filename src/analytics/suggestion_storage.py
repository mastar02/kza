"""Cleanup-only access to registered proposal metadata, never to message text."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sqlite3

from src.analytics.event_evidence import EventDependency
from src.memory.erasure_journal import ErasureResult
from src.memory.fact_ledger import FactLedger, MemorySource

MAX_METADATA_BYTES = 4 * 1024 * 1024
BATCH_SIZE = 32
RUNTIME_TARGET = {"kind": "runtime", "name": "automation_proposals"}


def suggestion_descriptor(legacy_path: str | Path) -> dict:
    """Pin the actual policy sidecar without reading or attributing legacy JSON."""
    legacy = Path(legacy_path).resolve()
    return {
        "path": str(legacy.with_name(legacy.name + ".policy-v1.sqlite3")),
        "table": "suggestions_policy_v1",
    }


def validate_suggestion_descriptor(descriptor: dict) -> None:
    """Reject unknown protocols before opening any private registered path."""
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"path", "table"}
        or descriptor["table"] != "suggestions_policy_v1"
        or type(descriptor["path"]) is not str
        or not Path(descriptor["path"]).is_absolute()
    ):
        raise ValueError("invalid_suggestion_target")


@contextmanager
def existing_suggestion_database(descriptor: dict):
    """Open only existing policy data, without creating or migrating any file."""
    validate_suggestion_descriptor(descriptor)
    path = Path(descriptor["path"])
    if path.is_symlink():
        raise ValueError("redirected_suggestion_target")
    try:
        path.stat()
    except FileNotFoundError:
        yield None
        return
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        table = db.execute(
            "SELECT type FROM sqlite_master WHERE name='suggestions_policy_v1'"
        ).fetchone()
        if table is None:
            yield None
            return
        if table[0] != "table":
            raise ValueError("unknown_suggestion_schema")
        columns = db.execute("PRAGMA table_info(suggestions_policy_v1)").fetchall()
        expected = {
            "row_id",
            "id",
            "pattern_key",
            "created_at",
            "observed_at",
            "expires_at",
            "status",
            "snooze_until",
            "pattern_json",
            "evidence_json",
            "sources_json",
            "message",
        }
        if {column[1] for column in columns} != expected or not any(
            column[1] == "row_id" and column[2].upper() == "INTEGER" and column[5] == 1
            for column in columns
        ):
            raise ValueError("unknown_suggestion_schema")
        yield db


@dataclass(frozen=True)
class SuggestionMetadata:
    """Content-free provenance copied by the guarded policy writer."""

    sources: tuple[MemorySource, ...]
    expires_at: float


def suggestion_metadata(row) -> SuggestionMetadata:
    """Validate policy dates and exact metadata protocols without loading text."""
    created, observed, expiry = (
        row[key] for key in ("created_at", "observed_at", "expires_at")
    )
    if (
        not all(
            type(v) in (int, float) and math.isfinite(v)
            for v in (created, observed, expiry)
        )
        or not observed <= created < expiry <= observed + 30 * 86400
        or type(row["id"]) is not str
        or re.fullmatch(r"sug_[0-9a-f]{32}", row["id"]) is None
    ):
        raise ValueError("unknown_suggestion_metadata")
    decoded = []
    for key in ("sources_json", "evidence_json"):
        encoded = row[key]
        if type(encoded) is not str or len(encoded.encode()) > MAX_METADATA_BYTES:
            raise ValueError("unknown_suggestion_metadata")
        value = json.loads(encoded)
        if type(value) is not list or len(value) > 10_000:
            raise ValueError("unknown_suggestion_metadata")
        decoded.append(value)
    sources = tuple(MemorySource.from_dict(value) for value in decoded[0])
    if [source.to_dict() for source in sources] != decoded[0] or len(
        set(sources)
    ) != len(sources):
        raise ValueError("unknown_suggestion_sources")
    evidence = tuple(EventDependency.from_dict(value) for value in decoded[1])
    if (
        not evidence
        or len({d.target_id for d in evidence}) != 1
        or len({d.event_id for d in evidence}) != len(evidence)
        or any(expiry > d.expires_at for d in evidence)
        or any(
            expiry > d.expires_at
            for s in sources
            for d in s.fact_dependencies
            if d.expires_at is not None
        )
    ):
        raise ValueError("unknown_suggestion_evidence")
    return SuggestionMetadata(sources, expiry)


def _cursor(encoded: str | None) -> tuple[int, int]:
    if encoded is None:
        return 0, 0
    if type(encoded) is not str or len(encoded) > 128:
        raise ValueError("invalid_suggestion_cursor")
    match = re.fullmatch(r"s1:([1-9][0-9]{0,18}):(0|[1-9][0-9]{0,18})", encoded)
    if match is None:
        raise ValueError("invalid_suggestion_cursor")
    after, unknown = map(int, match.groups())
    if after > 2**63 - 1 or unknown > 2**63 - 1:
        raise ValueError("invalid_suggestion_cursor")
    return after, unknown


def cleanup_suggestions(
    ledger: FactLedger, descriptor: dict, conversation_id: str, cursor: str | None
) -> ErasureResult:
    """Delete known revoked rows in bounded batches; unknowns cannot starve them."""
    if not ledger.authority_available or not ledger.conversation_committed_revoked(
        conversation_id
    ):
        raise RuntimeError("suggestion_erasure_authority_unavailable")
    after, unknown = _cursor(cursor)
    with existing_suggestion_database(descriptor) as db:
        if db is None:
            return ErasureResult(True, pending_count=0)
        if after == 0:
            # An unsupported row ID stays quarantined without preventing
            # progress over positive IDs written by the actual policy writer.
            unknown = db.execute(
                "SELECT COUNT(*) FROM suggestions_policy_v1 WHERE row_id<=0"
            ).fetchone()[0]
        last, scanned = after, 0
        rows = db.execute(
            """SELECT row_id,id,created_at,observed_at,expires_at,sources_json,evidence_json
            FROM suggestions_policy_v1 WHERE row_id>? ORDER BY row_id LIMIT ?""",
            (after, BATCH_SIZE),
        )
        for row in rows:
            # Stream metadata one row at a time, rather than holding 32 large
            # JSON source arrays simultaneously. Message/pattern text is absent.
            row = dict(
                zip(
                    (
                        "row_id",
                        "id",
                        "created_at",
                        "observed_at",
                        "expires_at",
                        "sources_json",
                        "evidence_json",
                    ),
                    row,
                )
            )
            last, scanned = row["row_id"], scanned + 1
            try:
                metadata = suggestion_metadata(row)
            except (ValueError, TypeError, KeyError, AttributeError):
                unknown += 1
                continue
            if any(
                s.conversation_id == conversation_id for s in metadata.sources
            ) or ledger.sources_committed_unavailable(metadata.sources):
                db.execute("DELETE FROM suggestions_policy_v1 WHERE row_id=?", (last,))
    if scanned == BATCH_SIZE:
        return ErasureResult(
            False, f"s1:{last}:{unknown}", pending_count=unknown or None
        )
    # Revisit unknown quarantine on a later retry. The coordinator must not
    # certify it as erased, but subsequent known rows have already progressed.
    return ErasureResult(unknown == 0, pending_count=unknown)
