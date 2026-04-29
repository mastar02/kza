"""E2E: 4 policies active, full block + rewrite + audit + tts flow."""

import asyncio
import sqlite3
import importlib
from pathlib import Path
from unittest.mock import patch
import pytest

from src.hooks import (
    HookRegistry, BlockResult,
    HaActionCall, TtsCall,
    execute_before_chain, execute_after_event,
    HaActionDispatchedPayload, SttPayload,
)


@pytest.fixture
def registry_with_policies(tmp_path, monkeypatch):
    """Load all 4 real policies into a fresh registry."""
    from src.hooks.registry import _global_registry
    _global_registry.clear()

    # Redirect audit_sqlite to a temp DB
    audit_db = tmp_path / "audit.db"

    import src.policies.audit_sqlite as audit_mod
    conn = sqlite3.connect(audit_db, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "timestamp REAL, kind TEXT, payload_json TEXT)"
    )
    monkeypatch.setattr(audit_mod, "_db", conn)

    # Importing the policy modules registers handlers via decorators.
    # If they're already imported, reload to re-run decorators against the
    # cleared global registry.
    for name in ["src.policies.safety_alarm",
                 "src.policies.permissions",
                 "src.policies.tts_rewrite_es",
                 "src.policies.audit_sqlite"]:
        if name in importlib.sys.modules:
            importlib.reload(importlib.sys.modules[name])
        else:
            importlib.import_module(name)

    # After reload, audit_sqlite._db was reset to None — re-patch.
    monkeypatch.setattr(audit_mod, "_db", conn)

    yield _global_registry, conn, audit_db
    _global_registry.clear()
    conn.close()


@pytest.mark.asyncio
async def test_block_alarm_at_night(registry_with_policies):
    reg, conn, _ = registry_with_policies

    call = HaActionCall(
        entity_id="alarm_control_panel.casa", domain="alarm_control_panel",
        service="alarm_disarm", service_data={},
        user_id="juan", user_name="Juan", zone_id="z", timestamp=12345.0,
    )

    fake_now = type("dt", (), {"hour": 23})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = execute_before_chain(reg, "before_ha_action", call)

    assert isinstance(result, BlockResult)
    assert result.rule_name == "proteger_alarma_de_noche"


@pytest.mark.asyncio
async def test_block_child_climate(registry_with_policies):
    reg, conn, _ = registry_with_policies

    call = HaActionCall(
        entity_id="climate.living", domain="climate",
        service="set_temperature", service_data={"temperature": 24},
        user_id="niño1", user_name="Niño", zone_id="z", timestamp=12345.0,
    )

    result = execute_before_chain(reg, "before_ha_action", call)
    assert isinstance(result, BlockResult)
    assert result.rule_name == "chicos_sin_dominios_adultos"


@pytest.mark.asyncio
async def test_tts_rewrites_pesos(registry_with_policies):
    reg, _, _ = registry_with_policies

    tts_call = TtsCall(text="cuesta $500", voice=None, lang="es",
                       user_id=None, zone_id=None)
    result = execute_before_chain(reg, "before_tts_speak", tts_call)

    assert isinstance(result, TtsCall)
    assert result.text == "cuesta 500 pesos"


@pytest.mark.asyncio
async def test_audit_logs_after_event(registry_with_policies):
    reg, conn, _ = registry_with_policies

    payload = SttPayload(
        timestamp=99999.0, text="prendé la luz", latency_ms=100.0,
        user_id="juan", zone_id="z1", success=True,
    )
    execute_after_event(reg, "stt", payload)

    # Wait for the async task to complete
    await asyncio.sleep(0.05)

    rows = conn.execute("SELECT kind, timestamp FROM events WHERE kind = 'stt'").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == 99999.0
