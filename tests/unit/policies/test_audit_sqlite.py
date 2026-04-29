"""Tests for audit_sqlite policy — uses in-memory SQLite."""

import asyncio
import json
import sqlite3
from unittest.mock import patch
import pytest

from src.hooks import (
    SttPayload, HaActionCall, HaActionDispatchedPayload,
)


@pytest.fixture
def in_memory_audit(monkeypatch):
    """Replace the module-level _db with an in-memory sqlite."""
    import src.policies.audit_sqlite as mod

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "timestamp REAL, kind TEXT, payload_json TEXT"
        ")"
    )
    conn.commit()
    monkeypatch.setattr(mod, "_db", conn)
    yield conn
    conn.close()


@pytest.mark.asyncio
async def test_logs_stt_event(in_memory_audit):
    from src.policies.audit_sqlite import log_to_sqlite

    await log_to_sqlite(SttPayload(
        timestamp=12345.0, text="hola", latency_ms=100.0,
        user_id="juan", zone_id="z1", success=True,
    ))

    rows = in_memory_audit.execute("SELECT timestamp, kind, payload_json FROM events").fetchall()
    assert len(rows) == 1
    ts, kind, payload_json = rows[0]
    assert ts == 12345.0
    assert kind == "stt"
    payload = json.loads(payload_json)
    assert payload["text"] == "hola"


@pytest.mark.asyncio
async def test_logs_ha_dispatched_with_nested_call(in_memory_audit):
    from src.policies.audit_sqlite import log_to_sqlite

    call = HaActionCall(
        entity_id="light.x", domain="light", service="turn_on",
        service_data={"brightness_pct": 50},
        user_id="juan", user_name="Juan", zone_id="z1", timestamp=12345.0,
    )
    await log_to_sqlite(HaActionDispatchedPayload(
        timestamp=12345.0, call=call, success=True, error=None,
    ))

    row = in_memory_audit.execute(
        "SELECT kind, payload_json FROM events"
    ).fetchone()
    kind, payload_json = row
    assert kind == "ha_action_dispatched"
    payload = json.loads(payload_json)
    assert payload["call"]["entity_id"] == "light.x"
    assert payload["call"]["service_data"]["brightness_pct"] == 50
