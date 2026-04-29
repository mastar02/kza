"""Audit policy: log every relevant event to a SQLite DB for analytics.

Plan #3 OpenClaw — use case 3.
Schema: events(timestamp REAL, kind TEXT, payload_json TEXT)
"""

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.hooks import after_event

logger = logging.getLogger(__name__)


# Path comes from settings.yaml hooks.audit_sqlite_path; fall back to default.
_DEFAULT_PATH = Path(os.environ.get("KZA_AUDIT_DB", "./data/audit.db"))
_DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _open_db(path: Path = _DEFAULT_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "timestamp REAL NOT NULL, "
        "kind TEXT NOT NULL, "
        "payload_json TEXT NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_events_kind_ts ON events(kind, timestamp)")
    conn.commit()
    return conn


_db: sqlite3.Connection | None = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = _open_db()
    return _db


def _payload_to_json(payload) -> str:
    """Serialize a frozen-dataclass payload to JSON.

    Handles nested dataclasses (e.g. HaActionDispatchedPayload contains a HaActionCall).
    """
    def _convert(obj):
        if is_dataclass(obj):
            return {k: _convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return json.dumps(_convert(payload), default=str)


def _insert_sync(kind: str, timestamp: float, payload_json: str) -> None:
    db = _get_db()
    try:
        db.execute(
            "INSERT INTO events (timestamp, kind, payload_json) VALUES (?, ?, ?)",
            (timestamp, kind, payload_json),
        )
        db.commit()
    except sqlite3.Error as e:
        logger.warning(f"[Policy:audit_sqlite] insert failed: {e}")


@after_event(
    "wake", "stt", "intent",
    "ha_action_dispatched", "ha_action_blocked",
    "llm_call", "tts",
)
async def log_to_sqlite(payload):
    """Async: serialize payload + insert into SQLite via thread pool."""
    kind = type(payload).__name__.removesuffix("Payload").lower()
    if kind == "haactiondispatched":
        kind = "ha_action_dispatched"
    elif kind == "haactionblocked":
        kind = "ha_action_blocked"
    elif kind == "llmcall":
        kind = "llm_call"

    payload_json = _payload_to_json(payload)
    timestamp = getattr(payload, "timestamp", 0.0)
    await asyncio.to_thread(_insert_sync, kind, timestamp, payload_json)
