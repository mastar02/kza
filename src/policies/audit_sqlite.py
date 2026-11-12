"""Audit policy: log every relevant event to a SQLite DB for analytics.

Plan #3 OpenClaw — use case 3.
Schema: events(timestamp REAL, kind TEXT, payload_json TEXT)

U3b-0e (design v3 2b, review N1): the audit has its own switch,
``hooks.audit_sqlite.enabled`` (default false), separate from
``hooks.enabled``, which also loads safety_alarm and tts_rewrite_es.
Importing this module registers nothing and opens nothing. When the switch
is off, :func:`register_audit_sqlite` registers no handler and opens no path.
The store path comes from ``hooks.audit_sqlite.db_path``. ``KZA_AUDIT_DB``
is ignored without an admission ticket and refused under one.
"""

import asyncio
from collections.abc import Mapping
import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.core.runtime_admission import TICKET_ENV
from src.hooks import (
    HaActionCall,  # noqa: F401  — used in tests / payloads
    HaActionDispatchedPayload, HaActionBlockedPayload,
    SttPayload, IntentPayload, WakePayload, LlmCallPayload, TtsPayload,
)
from src.hooks.registry import HookRegistry

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/audit.db"
AUDIT_ENV = "KZA_AUDIT_DB"
AUDIT_EVENTS = ("wake", "stt", "intent", "ha_action_dispatched", "ha_action_blocked", "llm_call", "tts")

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


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class AuditSqlitePolicy:
    """After-event handler that appends each payload to one SQLite store.

    The connection opens lazily at the first event and closes with
    :meth:`close` (registered on the application's shutdown stack).
    """

    def __init__(self, db_path: str | os.PathLike, connection: sqlite3.Connection | None = None):
        """
        Args:
            db_path: Store path, from ``hooks.audit_sqlite.db_path``.
            connection: Optional pre-opened connection (tests).
        """
        self.db_path = Path(db_path)
        self._db = connection
        self._lock = threading.Lock()
        self._closed = False

    def _get_db(self) -> sqlite3.Connection:
        if self._closed:
            raise sqlite3.ProgrammingError("audit_sqlite_closed")
        if self._db is None:
            self._db = _open_db(self.db_path)
        return self._db

    def _insert_sync(self, kind: str, timestamp: float, payload_json: str) -> None:
        with self._lock:
            try:
                db = self._get_db()
                db.execute(
                    "INSERT INTO events (timestamp, kind, payload_json) VALUES (?, ?, ?)",
                    (timestamp, kind, payload_json),
                )
                db.commit()
            except sqlite3.Error as e:
                logger.warning(f"[Policy:audit_sqlite] insert failed: {type(e).__name__}")

    async def log_to_sqlite(self, payload) -> None:
        """Serialize ``payload`` and INSERT it in a worker thread.

        Errors are logged in ``_insert_sync``, never propagated. Async-only:
        a sync caller without a running loop makes the runner skip it.
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
        await asyncio.to_thread(self._insert_sync, kind, timestamp, payload_json)

    def close(self) -> None:
        """Close the connection; later events are dropped with a warning."""
        with self._lock:
            self._closed = True
            if self._db is not None:
                self._db.close()
                self._db = None


def audit_enabled(hooks_cfg: Mapping | None) -> bool:
    """True only for ``hooks.audit_sqlite.enabled: true`` (a real boolean)."""
    section = (hooks_cfg or {}).get("audit_sqlite") if isinstance(hooks_cfg, Mapping) else None
    return isinstance(section, Mapping) and section.get("enabled") is True


def register_audit_sqlite(registry: HookRegistry, hooks_cfg: Mapping | None, *,
                          environ: Mapping[str, str] | None = None) -> AuditSqlitePolicy | None:
    """Register the audit handler iff ``hooks.audit_sqlite.enabled`` is true.

    Args:
        registry: The hook registry the pipeline dispatches through.
        hooks_cfg: The ``hooks`` section of the central configuration.
        environ: Environment (os.environ by default).

    Returns:
        The registered policy, or None when the audit is off (nothing is
        registered and no path is opened).

    Raises:
        RuntimeError: ``KZA_AUDIT_DB`` set under an admission ticket.
    """
    environ = os.environ if environ is None else environ
    if AUDIT_ENV in environ:
        if TICKET_ENV in environ:
            raise RuntimeError("runtime_admission_audit_env_refused")
        logger.warning("[Policy:audit_sqlite] %s is ignored; the path comes from "
                       "hooks.audit_sqlite.db_path", AUDIT_ENV)
    if isinstance(hooks_cfg, Mapping) and "audit_sqlite_path" in hooks_cfg:
        logger.warning("[Policy:audit_sqlite] hooks.audit_sqlite_path is ignored; "
                       "use hooks.audit_sqlite.db_path")
    if not audit_enabled(hooks_cfg):
        logger.info("[Policy:audit_sqlite] disabled (hooks.audit_sqlite.enabled is not true)")
        return None
    path = hooks_cfg["audit_sqlite"].get("db_path") or DEFAULT_DB_PATH
    if not isinstance(path, (str, os.PathLike)):
        raise RuntimeError("audit_sqlite_db_path_invalid")
    policy = AuditSqlitePolicy(path)
    for name in AUDIT_EVENTS:
        registry.register_after(name, policy.log_to_sqlite)
    return policy
