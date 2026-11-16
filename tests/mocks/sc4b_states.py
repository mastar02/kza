"""SC-4b: deterministic states for W1 and the golden digest of capture_state_authority.

``golden_state`` writes a fixed state (every database through the exact
admission column contract, fixed rows, a valid deletion-ledger chain, an
accounts record, a configuration file and .env) whose projection exercises
every selection capture_state_authority makes: the protected tables, the
deletion-ledger chain, the filtered fact, voice-group and suggestion
selections, the deleted fact revisions and the mixed deadline/closure
projections. Its digest is pinned in test_sc4b_w.py from the code before the
selection split (R3-OPS-2).
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts import stage_runtime as stage

SALT = "5a" * 32
GENESIS = "0" * 64
NOW = 1_800_000_000.5
# Distinctive values the content-free canaries look for in reports and log events.
CANARY = "sc4b-canary-7f3e"

_PRIVACY = ("conversation_tombstones", "conversation_erasures", "facts", "fact_sources", "fact_dependencies",
            "fact_owners", "fact_owner_revisions", "fact_owner_schema", "voice_profiles",
            "voice_attribution_suspensions",
            "voice_attribution_consents", "deletion_ledger_v1", "deletion_ledger_salt_v1", "policy_index_deletions",
            "erasure_targets", "erasure_steps", "voice_groups", "voice_group_members", "voice_group_lineage",
            "voice_group_schema", "voice_fingerprints", "wake_clips_v1", "stt_corrections_v1",
            "conversation_turns_v1")
_AUTH = ("auth_meta", "credentials", "sessions", "retired_users", "credential_attempts")
_AMBIENT = ("utterances", "capture_conversations", "pending_audio_deletions", "audio_write_intents",
            "ambient_runtime_control")
_SUGGESTIONS = ("suggestions_policy_v1",)
DATABASES = {"data/privacy.db": _PRIVACY, "data/auth.db": _AUTH, "data/ambient.db": _AMBIENT,
             "data/suggestions.db": _SUGGESTIONS}


def columns(table: str) -> list[tuple[str, str]]:
    return [tuple(item.split(":")) for item in stage._TABLE_COLUMNS[table].split()]


def ddl(table: str) -> str:
    return f"CREATE TABLE {table} (" + ", ".join(f"{name} {kind}" for name, kind in columns(table)) + ")"


def _sources(expires: float) -> str:
    return json.dumps([{"expires_at": expires, "capture_id": "cap-1"}])


def _value(table: str, column: str, kind: str, index: int):
    special = {
        ("facts", "state"): ("confirmed", "automatic", "deleted")[index % 3],
        ("facts", "expires_at"): NOW + index,
        ("fact_sources", "fact_id"): f"facts-id-{index}",
        ("fact_dependencies", "fact_id"): f"facts-id-{index}",
        ("fact_owners", "fact_id"): f"facts-id-{index}",
        ("voice_group_members", "group_id"): "voice_groups-id-0",
        ("voice_group_members", "fingerprint_id"): f"voice_fingerprints-id-{index}",
        ("voice_group_lineage", "group_id"): "voice_groups-id-0",
        ("fact_sources", "expires_at"): NOW + 100 + index,
        ("fact_owner_schema", "version"): 1,
        ("voice_group_schema", "version"): 1,
        ("voice_groups", "state"): ("manual", "automatic")[index % 2],
        ("voice_groups", "reason"): ("manual-merge", "similarity")[index % 2],
        ("voice_fingerprints", "sources_json"): _sources(NOW + 50),
        ("voice_fingerprints", "expires_at"): NOW + 10 + index,
        ("wake_clips_v1", "sources_json"): _sources(NOW + 50),
        ("wake_clips_v1", "expires_at"): NOW + 10 + index,
        ("wake_clips_v1", "id"): f"clip-{index}",
        ("stt_corrections_v1", "id"): f"clip-{index}",
        ("conversation_turns_v1", "quality"): ("good", "unmarked")[index % 2],
        ("conversation_turns_v1", "id"): f"clip-{index + 1}",
        ("suggestions_policy_v1", "status"): ("accepted", "pending")[index % 2],
        ("suggestions_policy_v1", "sources_json"): _sources(NOW + 50),
        ("suggestions_policy_v1", "expires_at"): NOW + 10 + index,
        ("suggestions_policy_v1", "snooze_until"): None,
        ("utterances", "expires_at"): NOW + index,
        ("utterances", "t0"): NOW - 1000.0,
        ("utterances", "created_at"): NOW - 1000.0,
        ("utterances", "distilled"): index % 2,
        ("auth_meta", "key"): ("schema_version", "other")[index % 2],
        ("auth_meta", "value"): ("3", "x")[index % 2],
        ("ambient_runtime_control", "id"): 1,
        ("ambient_runtime_control", "paused"): 0,
    }
    if (table, column) in special:
        return special[(table, column)]
    if kind == "INTEGER":
        return index + 1
    if kind == "REAL":
        return NOW + index
    if kind == "BLOB":
        return bytes([index, 1, 2])
    return f"{table}-{column}-{index}"


def _rows(table: str) -> list[tuple]:
    count = {"ambient_runtime_control": 1, "fact_owner_schema": 1, "voice_group_schema": 1,
             "deletion_ledger_v1": 0, "deletion_ledger_salt_v1": 0}.get(table, 2)
    return [tuple(_value(table, name, kind, index) for name, kind in columns(table)) for index in range(count)]


def ledger_rows(entries: list[tuple[str, str, float]]) -> list[tuple]:
    rows, previous = [], GENESIS
    for seq, (kind, subject, at) in enumerate(entries, start=1):
        row = stage._ledger_row_sha256(previous, seq, kind, subject, at)
        rows.append((seq, kind, subject, at, previous, row))
        previous = row
    return rows


LEDGER = [("conversation_erased", "1a" * 32, NOW), ("fact_deleted", "2b" * 32, NOW + 1.0),
          ("credential_revoked", "3c" * 32, NOW + 2.0)]


def insert(db: sqlite3.Connection, table: str, row: tuple) -> None:
    db.execute(f"INSERT INTO {table} VALUES (" + ",".join("?" * len(row)) + ")", row)


def golden_state(state: Path) -> Path:
    """Write the fixed state under ``state`` (created); returns it."""
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    for relative, tables in DATABASES.items():
        with closing(sqlite3.connect(state / relative)) as db, db:
            for table in tables:
                db.execute(ddl(table))
                for row in _rows(table):
                    insert(db, table, row)
            if relative == "data/privacy.db":
                insert(db, "deletion_ledger_salt_v1", (1, SALT))
                for row in ledger_rows(LEDGER):
                    insert(db, "deletion_ledger_v1", row)
    (state / "data" / "home_accounts_v1.json").write_text(json.dumps({"schema": "kza-users-accounts-v1", "users": [
        {"id": "u1", "name": "Ana", "role": "home", "last_seen": 1.0},
        {"id": "u2", "name": "Beto", "role": "operator", "last_seen": 2.0}]}))
    (state / "config" / "policy.yaml").write_text("retention: 48\n")
    (state / ".env").write_text("KZA_EXAMPLE=1\n")
    return state


def execute(path: Path, *statements: tuple[str, tuple]) -> None:
    """Run statements on one database of a state in one transaction."""
    with closing(sqlite3.connect(path)) as db, db:
        for text, parameters in statements:
            db.execute(text, parameters)
