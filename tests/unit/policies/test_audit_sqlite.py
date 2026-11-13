"""Tests for audit_sqlite policy and its own switch (U3b-0e N1)."""

import asyncio
from contextlib import AsyncExitStack
import importlib
import json
import sqlite3
import sys

import pytest

from src.core.runtime_admission import TICKET_ENV
from src.hooks import (
    SttPayload, HaActionCall, HaActionDispatchedPayload, TtsCall,
    execute_after_event, execute_before_chain,
)
from src.policies.audit_sqlite import (
    AUDIT_ENV, AUDIT_EVENTS, AuditSqlitePolicy, audit_enabled, register_audit_sqlite,
)


@pytest.fixture
def in_memory_audit():
    """A policy bound to an in-memory sqlite."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "timestamp REAL, kind TEXT, payload_json TEXT"
        ")"
    )
    conn.commit()
    policy = AuditSqlitePolicy(":memory:", connection=conn)
    yield policy, conn
    policy.close()


@pytest.mark.asyncio
async def test_logs_stt_event(in_memory_audit):
    policy, conn = in_memory_audit

    await policy.log_to_sqlite(SttPayload(
        timestamp=12345.0, text="hola", latency_ms=100.0,
        user_id="juan", zone_id="z1", success=True,
    ))

    rows = conn.execute("SELECT timestamp, kind, payload_json FROM events").fetchall()
    assert len(rows) == 1
    ts, kind, payload_json = rows[0]
    assert ts == 12345.0
    assert kind == "stt"
    payload = json.loads(payload_json)
    assert payload["text"] == "hola"


@pytest.mark.asyncio
async def test_logs_ha_dispatched_with_nested_call(in_memory_audit):
    policy, conn = in_memory_audit

    call = HaActionCall(
        entity_id="light.x", domain="light", service="turn_on",
        service_data={"brightness_pct": 50},
        user_id="juan", user_name="Juan", zone_id="z1", timestamp=12345.0,
    )
    await policy.log_to_sqlite(HaActionDispatchedPayload(
        timestamp=12345.0, call=call, success=True, error=None,
    ))

    row = conn.execute(
        "SELECT kind, payload_json FROM events"
    ).fetchone()
    kind, payload_json = row
    assert kind == "ha_action_dispatched"
    payload = json.loads(payload_json)
    assert payload["call"]["entity_id"] == "light.x"
    assert payload["call"]["service_data"]["brightness_pct"] == 50


# --- the switch (U3b-0e N1) ------------------------------------------------------------------------

def _stt():
    return SttPayload(timestamp=1.0, text="hola", latency_ms=1.0, user_id="u", zone_id="z", success=True)


@pytest.mark.parametrize("hooks_cfg, enabled", [
    ({}, False),
    ({"enabled": True}, False),
    ({"audit_sqlite": {"enabled": False}}, False),
    ({"audit_sqlite": {"enabled": "true"}}, False),
    ({"audit_sqlite": {"enabled": 1}}, False),
    ({"audit_sqlite": True}, False),
    ({"audit_sqlite": {"enabled": True}}, True),
])
def test_audit_enabled_only_for_a_real_true(hooks_cfg, enabled):
    assert audit_enabled(hooks_cfg) is enabled


def test_import_registers_nothing():
    from src.hooks.registry import _global_registry
    _global_registry.clear()
    importlib.reload(sys.modules["src.policies.audit_sqlite"])
    assert sum(_global_registry.get_stats()["after_handler_count"].values()) == 0


def test_off_registers_no_handler_and_opens_no_path(tmp_path, monkeypatch):
    from src.hooks.registry import HookRegistry
    monkeypatch.chdir(tmp_path)
    registry = HookRegistry()
    path = tmp_path / "audit.db"
    assert register_audit_sqlite(registry, {"enabled": True, "audit_sqlite": {
        "enabled": False, "db_path": str(path)}}, environ={}) is None
    assert sum(registry.get_stats()["after_handler_count"].values()) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_on_writes_only_to_the_configured_path(tmp_path, monkeypatch):
    from src.hooks.registry import HookRegistry
    monkeypatch.chdir(tmp_path)
    registry = HookRegistry()
    path = tmp_path / "chosen" / "audit.db"
    other = tmp_path / "env.db"
    policy = register_audit_sqlite(registry, {"audit_sqlite": {"enabled": True, "db_path": str(path)}},
                                   environ={AUDIT_ENV: str(other)})
    try:
        assert {name for name in AUDIT_EVENTS if registry.get_after_handlers(name)} == set(AUDIT_EVENTS)
        await policy.log_to_sqlite(_stt())
    finally:
        policy.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT kind FROM events").fetchall() == [("stt",)]
    assert not other.exists() and not (tmp_path / "data").exists()


def test_env_path_under_a_ticket_refuses(tmp_path):
    from src.hooks.registry import HookRegistry
    registry = HookRegistry()
    for hooks_cfg in ({}, {"audit_sqlite": {"enabled": False}}, {"audit_sqlite": {"enabled": True}}):
        with pytest.raises(RuntimeError, match="runtime_admission_audit_env_refused"):
            register_audit_sqlite(registry, hooks_cfg, environ={AUDIT_ENV: str(tmp_path / "a.db"), TICKET_ENV: "{}"})
    assert sum(registry.get_stats()["after_handler_count"].values()) == 0
    assert list(tmp_path.iterdir()) == []


def _fresh_policies(monkeypatch):
    from src.hooks.registry import _global_registry
    _global_registry.clear()
    for name in [name for name in sys.modules if name == "src.policies" or name.startswith("src.policies.")]:
        monkeypatch.delitem(sys.modules, name)
    return _global_registry


@pytest.mark.asyncio
async def test_hooks_on_audit_off_loads_safety_and_rewrite_without_audit(tmp_path, monkeypatch):
    """Acceptance: hooks.enabled true + audit off: no handler, no audit.db, policies load."""
    from src.main import load_policy_hooks
    registry = _fresh_policies(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(AUDIT_ENV, raising=False)
    async with AsyncExitStack() as stack:
        hooks = load_policy_hooks({"enabled": True, "audit_sqlite": {"enabled": False}}, stack)
        assert hooks is registry
        stats = registry.get_stats()
        assert stats["before_handler_count"]["before_ha_action"] >= 1  # safety_alarm (and the demo)
        assert stats["before_handler_count"]["before_tts_speak"] == 1  # tts_rewrite_es
        assert sum(stats["after_handler_count"].values()) == 0
        rewritten = execute_before_chain(registry, "before_tts_speak",
                                         TtsCall(text="cuesta $500", voice=None, lang="es", user_id=None, zone_id=None))
        assert rewritten.text == "cuesta 500 pesos"
        execute_after_event(registry, "stt", _stt())
        await asyncio.gather(*registry._after_tasks)
    assert not (tmp_path / "data").exists() and list(tmp_path.iterdir()) == []
    registry.clear()


@pytest.mark.asyncio
async def test_hooks_on_audit_on_writes_and_closes(tmp_path, monkeypatch):
    from src.main import load_policy_hooks
    registry = _fresh_policies(monkeypatch)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "state" / "audit.db"
    async with AsyncExitStack() as stack:
        hooks = load_policy_hooks({"enabled": True, "audit_sqlite": {"enabled": True, "db_path": str(path)}}, stack)
        assert sum(hooks.get_stats()["after_handler_count"].values()) == len(AUDIT_EVENTS)
        execute_after_event(registry, "stt", _stt())
        await asyncio.gather(*registry._after_tasks)
        (policy,) = {handler.__self__ for handler in registry.get_after_handlers("stt")}
        assert policy._db is not None
    assert policy._db is None
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT kind FROM events").fetchall() == [("stt",)]
    registry.clear()


@pytest.mark.asyncio
async def test_hooks_off_registers_nothing(tmp_path, monkeypatch):
    from src.main import load_policy_hooks
    registry = _fresh_policies(monkeypatch)
    monkeypatch.chdir(tmp_path)
    async with AsyncExitStack() as stack:
        assert load_policy_hooks({"enabled": False, "audit_sqlite": {"enabled": True}}, stack) is None
    assert sum(registry.get_stats()["after_handler_count"].values()) == 0
    assert list(tmp_path.iterdir()) == []


# --- review F5/F6 -----------------------------------------------------------------------------------

def test_an_enabled_audit_under_a_ticket_is_refused(tmp_path):
    """Review F6 (P2=A): the audit store has no erasure participant, so R1 refuses it."""
    from src.hooks.audit_switch import UNSUPPORTED_CODE, refuse_managed_audit
    from src.hooks.registry import HookRegistry
    registry = HookRegistry()
    enabled = {"audit_sqlite": {"enabled": True, "db_path": str(tmp_path / "a.db")}}
    with pytest.raises(RuntimeError, match=UNSUPPORTED_CODE):
        register_audit_sqlite(registry, enabled, environ={TICKET_ENV: "{}"})
    with pytest.raises(RuntimeError, match=UNSUPPORTED_CODE):
        refuse_managed_audit({"hooks": {"enabled": False, **enabled}}, environ={TICKET_ENV: "{}"})
    refuse_managed_audit({"hooks": enabled}, environ={})  # development: allowed
    refuse_managed_audit({"hooks": {"audit_sqlite": {"enabled": "true"}}}, environ={TICKET_ENV: "{}"})
    assert register_audit_sqlite(registry, {"audit_sqlite": {"enabled": False}}, environ={TICKET_ENV: "{}"}) is None
    assert sum(registry.get_stats()["after_handler_count"].values()) == 0
    assert list(tmp_path.iterdir()) == []


def test_the_serve_entry_refuses_an_enabled_audit_under_a_ticket(tmp_path, monkeypatch):
    from src.core.runtime_admission import verify_runtime_admission
    from src.hooks.audit_switch import UNSUPPORTED_CODE
    monkeypatch.setenv(TICKET_ENV, "{}")
    monkeypatch.delenv(AUDIT_ENV, raising=False)
    config = {"memory": {"privacy_db_path": str(tmp_path / "p.db")},
              "hooks": {"enabled": True, "audit_sqlite": {"enabled": True, "db_path": str(tmp_path / "a.db")}}}
    with pytest.raises(RuntimeError, match=UNSUPPORTED_CODE):
        verify_runtime_admission(config)
    assert list(tmp_path.iterdir()) == []


def _custom_policies(tmp_path, monkeypatch, name="kza_custom_policies"):
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "rewrite.py").write_text(
        "from src.hooks import before_tts_speak\n\n"
        "@before_tts_speak()\n"
        "def keep(call):\n"
        "    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, f"{name}.rewrite", raising=False)
    return name


@pytest.mark.asyncio
async def test_a_custom_policies_module_with_the_audit_off_never_imports_src_policies(tmp_path, monkeypatch):
    """Review F5: src.policies is imported for the audit only when the audit is on."""
    from src.main import load_policy_hooks
    registry = _fresh_policies(monkeypatch)
    module = _custom_policies(tmp_path, monkeypatch)
    async with AsyncExitStack() as stack:
        hooks = load_policy_hooks({"enabled": True, "policies_module": module,
                                   "audit_sqlite": {"enabled": False}}, stack)
        assert hooks is registry
        assert registry.get_stats()["before_handler_count"]["before_tts_speak"] == 1
    assert not any(name == "src.policies" or name.startswith("src.policies.") for name in sys.modules)
    registry.clear()


@pytest.mark.asyncio
async def test_a_broken_audit_import_is_one_failed_policy(tmp_path, monkeypatch, caplog):
    """Review F5: the audit import failing follows the per-policy rule (hooks stay up)."""
    from src.main import load_policy_hooks
    registry = _fresh_policies(monkeypatch)
    module = _custom_policies(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "src.policies.audit_sqlite", None)  # import raises ImportError
    async with AsyncExitStack() as stack:
        hooks = load_policy_hooks({"enabled": True, "policies_module": module,
                                   "audit_sqlite": {"enabled": True, "db_path": str(tmp_path / "a.db")}}, stack)
        assert hooks is registry
        assert sum(registry.get_stats()["after_handler_count"].values()) == 0
    assert "Policy 'audit_sqlite' failed to load" in caplog.text and "PARTIAL" in caplog.text
    assert not (tmp_path / "a.db").exists()
    registry.clear()
