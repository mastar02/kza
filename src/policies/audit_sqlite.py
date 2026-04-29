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

from src.hooks import (
    after_event,
    HaActionCall,  # noqa: F401  — used in tests / payloads
    HaActionDispatchedPayload, HaActionBlockedPayload,
    SttPayload, IntentPayload, WakePayload, LlmCallPayload, TtsPayload,
)

logger = logging.getLogger(__name__)


# Path comes from settings.yaml hooks.audit_sqlite_path; fall back to default.
_DEFAULT_PATH = Path(os.environ.get("KZA_AUDIT_DB", "./data/audit.db"))
# Note: parent dir creation deferred to _open_db (lazy) so importing this
# module is side-effect-free. Tests can override KZA_AUDIT_DB or monkeypatch
# _db before the first event fires.


# Explicit map from payload class → event_name string. Keep in sync with
# EVENT_NAMES in src.hooks.types. Reviewers asked for this to avoid the
# CamelCase-to-snake_case derivation that fails silently for unknown payloads.
_PAYLOAD_TO_KIND: dict[type, str] = {
    WakePayload: "wake",
    SttPayload: "stt",
    IntentPayload: "intent",
    HaActionDispatchedPayload: "ha_action_dispatched",
    HaActionBlockedPayload: "ha_action_blocked",
    LlmCallPayload: "llm_call",
    TtsPayload: "tts",
}


def _open_db(path: Path = _DEFAULT_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)  # moved here from module level
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
    """Async after-event handler: serialize payload to JSON and INSERT in
    a worker thread (asyncio.to_thread) so the event loop is never blocked
    by SQLite I/O. Errors are logged in _insert_sync, never propagated.

    Notes:
        - Async-only: if execute_after_event is invoked from a sync code
          path with no running loop, the runner skips this handler with
          a WARNING (and the audit row is silently dropped). This is
          acceptable because all real callsites in the pipeline are async.
    """
    kind = _PAYLOAD_TO_KIND.get(type(payload))
    if kind is None:
        logger.warning(
            f"[Policy:audit_sqlite] unknown payload type {type(payload).__name__}; "
            "add to _PAYLOAD_TO_KIND. Skipping audit row."
        )
        return

    payload_json = _payload_to_json(payload)
    timestamp = getattr(payload, "timestamp", 0.0)
    await asyncio.to_thread(_insert_sync, kind, timestamp, payload_json)
