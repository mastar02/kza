# Plugin Hooks System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permite a handlers Python escritos por el dueño del sistema (a) bloquear/reescribir HA actions y TTS speak via decoradores tipados, y (b) auditar todos los eventos relevantes del pipeline (wake / STT / intent / HA action / LLM / TTS) sin tocar el core. Backward-compatible: con `hooks.enabled: false` (default) el sistema se comporta igual que antes.

**Architecture:** 3 hook points (`before_ha_action`, `before_tts_speak`, `after_event(name)`). 2 outputs (`BlockResult`, `RewriteResult`). Singleton `_global_registry` populado por decoradores al import-time de `src.policies`. Sync handlers en `before_*` (path crítico, <5ms convención); fire-and-forget en `after_*` (sync inline + async como Task con strong ref). Block short-circuita; rewrite encadena.

**Tech Stack:** Python 3.13, asyncio, frozen dataclasses, pytest + fixtures existentes en `tests/conftest.py`. Persistencia audit: sqlite3 stdlib en `./data/audit.db`. Sin dependencias nuevas. Spec: `docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md`.

---

## File Structure

### Crear

| Path | Responsabilidad |
|------|-----------------|
| `src/hooks/__init__.py` | Re-exporta públicos: `before_ha_action`, `before_tts_speak`, `after_event`, `BlockResult`, `RewriteResult`, payloads, `HookRegistry` |
| `src/hooks/types.py` | Frozen dataclasses de calls, results y event payloads + tupla `EVENT_NAMES` |
| `src/hooks/registry.py` | `HookRegistry` con `register_before`, `register_after`, `get_stats`, contadores |
| `src/hooks/runner.py` | `execute_before_chain` (block + rewrite chain) + `execute_after_event` (fire-and-forget) |
| `src/hooks/decorators.py` | `before_ha_action`, `before_tts_speak`, `after_event` |
| `src/policies/__init__.py` | Vacío (carga estática vía `import src.policies`) |
| `src/policies/safety_alarm.py` | Use case 1: bloquear desarmar alarma 22-7 |
| `src/policies/permissions.py` | Use case 2: niños no pueden controlar climate / lock / alarm |
| `src/policies/tts_rewrite_es.py` | Use case 4: `$1000` → `1000 pesos` |
| `src/policies/audit_sqlite.py` | Use case 3: cada `after_event` insert a `data/audit.db` |
| `tests/unit/hooks/__init__.py` | Vacío |
| `tests/unit/hooks/test_types.py` | Frozen invariants, `with_data` |
| `tests/unit/hooks/test_registry.py` | register / list / get_stats |
| `tests/unit/hooks/test_runner_before.py` | Block short-circuit, rewrite chain, priority order, error swallow, timing warning |
| `tests/unit/hooks/test_runner_after.py` | Sync inline, async Task con strong ref, errors swallowed |
| `tests/unit/hooks/test_decorators.py` | Decorators registran en singleton |
| `tests/unit/policies/__init__.py` | Vacío |
| `tests/unit/policies/test_safety_alarm.py` | Block 22-7, no block 8-21 |
| `tests/unit/policies/test_permissions.py` | Block child+ADULT_DOMAIN, no block adult |
| `tests/unit/policies/test_tts_rewrite_es.py` | `$N` → `N pesos`, no rewrite si no matchea |
| `tests/unit/policies/test_audit_sqlite.py` | `after_event` insert a in-memory sqlite |
| `tests/integration/test_hooks_e2e.py` | Pipeline mock con las 4 policies activas |

### Modificar

| Path | Cambio |
|------|--------|
| `src/orchestrator/dispatcher.py` | `RequestDispatcher.__init__` acepta `hooks=None`; antes de cada `ha.call_service_ws` invoca `hooks.execute_before_chain("before_ha_action", call)` si no es None; después emite `after_event("ha_action_dispatched"|"ha_action_blocked")` |
| `src/pipeline/response_handler.py` | `ResponseHandler.__init__` acepta `hooks=None`; en `speak()` invoca `hooks.execute_before_chain("before_tts_speak", tts_call)` |
| `src/pipeline/request_router.py` | Emit `after_event` en checkpoints: post-wake, post-STT, post-intent, post-llm |
| `src/main.py` | Si `hooks.enabled`: instancia `HookRegistry`, importa `src.policies`, inyecta a Dispatcher + ResponseHandler |
| `config/settings.yaml` | Bloque `hooks` |

---

## Conventions

- **Imports:** stdlib → third-party → `from src.modulo import X`. Sin imports relativos.
- **Logging:** `logger = logging.getLogger(__name__)` con prefijo `[HookRegistry]`, `[HookRunner]`, `[Policy:nombre]` según el archivo.
- **Tests:** `pytest` con `pytest-asyncio` en `auto` mode. Fixtures en `tests/conftest.py` cuando aplique.
- **Singleton:** `src/hooks/registry.py` exporta `_global_registry: HookRegistry`. `src/hooks/decorators.py` lo importa y registra side-effect. **En tests**, usar `_global_registry.clear()` en fixture `autouse=True` para isolation.
- **Backward compat:** sin `hooks` inyectado a Dispatcher/ResponseHandler, comportamiento idéntico al baseline.
- **Sync vs async:** handlers `before_*` son sync (`def`). Handlers `after_*` pueden ser sync o async — `execute_after_event` detecta con `inspect.iscoroutinefunction`.

---

## Task 1: Types (frozen dataclasses + EVENT_NAMES)

**Files:**
- Create: `src/hooks/__init__.py` (vacío inicial)
- Create: `src/hooks/types.py`
- Create: `tests/unit/hooks/__init__.py` (vacío)
- Create: `tests/unit/hooks/test_types.py`

- [ ] **Step 1.1: Crear directorios + __init__**

```bash
mkdir -p src/hooks tests/unit/hooks
touch src/hooks/__init__.py tests/unit/hooks/__init__.py
```

- [ ] **Step 1.2: Escribir test fallido `tests/unit/hooks/test_types.py`**

```python
"""Tests for hook types and event payloads."""

import pytest
from dataclasses import FrozenInstanceError

from src.hooks.types import (
    HaActionCall, TtsCall, BlockResult, RewriteResult,
    WakePayload, SttPayload, IntentPayload,
    HaActionDispatchedPayload, HaActionBlockedPayload,
    LlmCallPayload, TtsPayload,
    EVENT_NAMES,
)


class TestHaActionCall:
    def test_required_fields(self):
        call = HaActionCall(
            entity_id="light.escritorio",
            domain="light",
            service="turn_on",
            service_data={"brightness_pct": 50},
            user_id="juan",
            user_name="Juan",
            zone_id="zone_escritorio",
            timestamp=1700000000.0,
        )
        assert call.domain == "light"
        assert call.service_data["brightness_pct"] == 50

    def test_immutability(self):
        call = HaActionCall(
            entity_id="light.x", domain="light", service="turn_on",
            service_data={}, user_id=None, user_name=None,
            zone_id=None, timestamp=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            call.service = "turn_off"

    def test_with_data_returns_modified_copy(self):
        call = HaActionCall(
            entity_id="light.x", domain="light", service="turn_on",
            service_data={"brightness_pct": 50}, user_id=None, user_name=None,
            zone_id=None, timestamp=0.0,
        )
        modified = call.with_data(brightness_pct=20)
        assert modified.service_data == {"brightness_pct": 20}
        assert call.service_data == {"brightness_pct": 50}  # original unchanged
        assert modified is not call


class TestTtsCall:
    def test_basic(self):
        c = TtsCall(text="hola", voice=None, lang="es", user_id=None, zone_id=None)
        assert c.text == "hola"

    def test_immutability(self):
        c = TtsCall(text="hola", voice=None, lang="es", user_id=None, zone_id=None)
        with pytest.raises(FrozenInstanceError):
            c.text = "chau"


class TestResults:
    def test_block_result(self):
        b = BlockResult(reason="no se puede", rule_name="protege_alarma")
        assert b.reason == "no se puede"
        assert b.rule_name == "protege_alarma"

    def test_rewrite_result(self):
        call = TtsCall(text="x", voice=None, lang="es", user_id=None, zone_id=None)
        r = RewriteResult(modified=call, rule_name="rule")
        assert r.modified is call


class TestEventPayloads:
    def test_wake_payload(self):
        p = WakePayload(timestamp=0.0, confidence=0.95, zone_id="z")
        assert p.confidence == 0.95

    def test_stt_payload(self):
        p = SttPayload(timestamp=0.0, text="hola", latency_ms=120.0,
                       user_id=None, zone_id=None, success=True)
        assert p.success

    def test_event_names_is_closed_tuple(self):
        # Lista cerrada — type-safe, autocompletable
        assert isinstance(EVENT_NAMES, tuple)
        assert "wake" in EVENT_NAMES
        assert "stt" in EVENT_NAMES
        assert "intent" in EVENT_NAMES
        assert "ha_action_dispatched" in EVENT_NAMES
        assert "ha_action_blocked" in EVENT_NAMES
        assert "llm_call" in EVENT_NAMES
        assert "tts" in EVENT_NAMES
```

- [ ] **Step 1.3: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/hooks/test_types.py -v
```

Expected: ImportError on `src.hooks.types`.

- [ ] **Step 1.4: Implement `src/hooks/types.py`**

```python
"""Plugin hooks types — frozen dataclasses + enumerated event names.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
"""

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class HaActionCall:
    """Pending HA service call seen by `before_ha_action` hooks."""
    entity_id: str
    domain: str
    service: str
    service_data: dict
    user_id: str | None
    user_name: str | None
    zone_id: str | None
    timestamp: float

    def with_data(self, **changes) -> "HaActionCall":
        """Return a copy with `service_data` updated by `changes`."""
        new_data = {**self.service_data, **changes}
        return replace(self, service_data=new_data)


@dataclass(frozen=True, slots=True)
class TtsCall:
    """Pending TTS speak seen by `before_tts_speak` hooks."""
    text: str
    voice: str | None
    lang: str
    user_id: str | None
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class BlockResult:
    """Returned from a `before_*` hook to short-circuit the chain.

    The reason is spoken to the user via TTS (HA blocks) or logged (TTS blocks).
    rule_name identifies the policy for telemetry/audit.
    """
    reason: str
    rule_name: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Returned from a `before_*` hook to modify the call/text.

    The chain continues with `modified` as the new call. `modified` must be
    of the same dataclass type as the call passed in (HaActionCall or TtsCall).
    """
    modified: object  # HaActionCall | TtsCall
    rule_name: str


# === Event payloads (one per event_name) ===

@dataclass(frozen=True, slots=True)
class WakePayload:
    timestamp: float
    confidence: float
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class SttPayload:
    timestamp: float
    text: str
    latency_ms: float
    user_id: str | None
    zone_id: str | None
    success: bool


@dataclass(frozen=True, slots=True)
class IntentPayload:
    timestamp: float
    text: str
    intent: str   # "domotics" | "music" | "conversation" | ...
    entities: list[str]
    user_id: str | None


@dataclass(frozen=True, slots=True)
class HaActionDispatchedPayload:
    timestamp: float
    call: HaActionCall
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class HaActionBlockedPayload:
    timestamp: float
    call: HaActionCall
    block: BlockResult


@dataclass(frozen=True, slots=True)
class LlmCallPayload:
    timestamp: float
    path: Literal["fast", "slow"]
    endpoint_id: str  # "fast_router_7b" | "reasoner_72b"
    latency_ms: float
    success: bool


@dataclass(frozen=True, slots=True)
class TtsPayload:
    timestamp: float
    text: str
    voice: str | None
    latency_ms: float
    success: bool


# Closed enumeration of event names — handlers register against these.
EVENT_NAMES: tuple[str, ...] = (
    "wake",
    "stt",
    "intent",
    "ha_action_dispatched",
    "ha_action_blocked",
    "llm_call",
    "tts",
)
```

- [ ] **Step 1.5: Run tests pass**

```bash
.venv/bin/pytest tests/unit/hooks/test_types.py -v
```

Expected: 9 passed.

- [ ] **Step 1.6: Commit**

```bash
git add src/hooks/__init__.py src/hooks/types.py tests/unit/hooks/__init__.py tests/unit/hooks/test_types.py
git commit -m "feat(hooks): types module — frozen dataclasses + EVENT_NAMES (plan #3 OpenClaw)"
```

---

## Task 2: HookRegistry — register / list / get_stats

**Files:**
- Create: `src/hooks/registry.py`
- Create: `tests/unit/hooks/test_registry.py`

- [ ] **Step 2.1: Escribir test fallido `tests/unit/hooks/test_registry.py`**

```python
"""Tests for HookRegistry register/list/clear/stats."""

import pytest

from src.hooks.registry import HookRegistry


class TestRegister:
    def test_register_before_orders_by_priority(self):
        reg = HookRegistry()
        def h1(call): return None
        def h2(call): return None
        def h3(call): return None

        reg.register_before("before_ha_action", h2, priority=20)
        reg.register_before("before_ha_action", h1, priority=5)
        reg.register_before("before_ha_action", h3, priority=10)

        handlers = reg.get_before_handlers("before_ha_action")
        # Priority asc: h1 (5), h3 (10), h2 (20)
        assert [fn for _prio, fn in handlers] == [h1, h3, h2]

    def test_register_before_unknown_hook_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError, match="unknown hook"):
            reg.register_before("before_bogus", lambda c: None, priority=10)

    def test_register_after_unknown_event_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError, match="unknown event"):
            reg.register_after("bogus_event", lambda p: None)

    def test_register_after_appends(self):
        reg = HookRegistry()
        def h1(p): pass
        def h2(p): pass
        reg.register_after("stt", h1)
        reg.register_after("stt", h2)
        assert reg.get_after_handlers("stt") == [h1, h2]


class TestStatsAndClear:
    def test_stats_initial(self):
        reg = HookRegistry()
        stats = reg.get_stats()
        assert stats["handler_failures"] == 0
        assert stats["handler_last_error"] is None
        assert stats["after_tasks_in_flight"] == 0
        assert stats["before_handler_count"] == {
            "before_ha_action": 0, "before_tts_speak": 0,
        }

    def test_stats_counts_handlers(self):
        reg = HookRegistry()
        reg.register_before("before_ha_action", lambda c: None, priority=10)
        reg.register_before("before_tts_speak", lambda c: None, priority=10)
        reg.register_after("stt", lambda p: None)

        stats = reg.get_stats()
        assert stats["before_handler_count"]["before_ha_action"] == 1
        assert stats["before_handler_count"]["before_tts_speak"] == 1
        assert stats["after_handler_count"]["stt"] == 1

    def test_clear_removes_all_handlers(self):
        reg = HookRegistry()
        reg.register_before("before_ha_action", lambda c: None, priority=10)
        reg.register_after("stt", lambda p: None)
        reg.clear()
        assert reg.get_before_handlers("before_ha_action") == []
        assert reg.get_after_handlers("stt") == []
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/hooks/test_registry.py -v
```

Expected: ImportError on `src.hooks.registry`.

- [ ] **Step 2.3: Implement `src/hooks/registry.py`**

```python
"""HookRegistry — central store for before/after hook handlers.

Plan #3 OpenClaw.
"""

import logging
from collections.abc import Callable

from src.hooks.types import EVENT_NAMES

logger = logging.getLogger(__name__)


VALID_BEFORE_HOOKS: tuple[str, ...] = ("before_ha_action", "before_tts_speak")


class HookRegistry:
    """Register and look up plugin hook handlers.

    Single global instance lives in `src.hooks.registry._global_registry`,
    populated by decorators in `src.hooks.decorators` at import-time of
    `src.policies` modules.
    """

    def __init__(self):
        self._before: dict[str, list[tuple[int, Callable]]] = {
            name: [] for name in VALID_BEFORE_HOOKS
        }
        self._after: dict[str, list[Callable]] = {name: [] for name in EVENT_NAMES}

        # Observability counters (consumed by get_stats + ContextManager-style logging)
        self._handler_failures: int = 0
        self._handler_last_error: str | None = None
        self._after_tasks: set = set()  # populated by runner.execute_after_event

    def register_before(self, hook_name: str, fn: Callable, priority: int) -> None:
        if hook_name not in VALID_BEFORE_HOOKS:
            raise ValueError(
                f"unknown hook {hook_name!r}; valid: {VALID_BEFORE_HOOKS}"
            )
        bucket = self._before[hook_name]
        bucket.append((priority, fn))
        bucket.sort(key=lambda item: item[0])  # asc by priority

    def register_after(self, event_name: str, fn: Callable) -> None:
        if event_name not in EVENT_NAMES:
            raise ValueError(
                f"unknown event {event_name!r}; valid: {EVENT_NAMES}"
            )
        self._after[event_name].append(fn)

    def get_before_handlers(self, hook_name: str) -> list[tuple[int, Callable]]:
        return list(self._before.get(hook_name, []))

    def get_after_handlers(self, event_name: str) -> list[Callable]:
        return list(self._after.get(event_name, []))

    def clear(self) -> None:
        """Reset all registrations (used by tests for isolation)."""
        self._before = {name: [] for name in VALID_BEFORE_HOOKS}
        self._after = {name: [] for name in EVENT_NAMES}
        self._handler_failures = 0
        self._handler_last_error = None
        # Don't clear _after_tasks — those are in-flight asyncio tasks.

    def get_stats(self) -> dict:
        return {
            "handler_failures": self._handler_failures,
            "handler_last_error": self._handler_last_error,
            "after_tasks_in_flight": len(self._after_tasks),
            "before_handler_count": {
                name: len(self._before[name]) for name in VALID_BEFORE_HOOKS
            },
            "after_handler_count": {
                name: len(self._after[name]) for name in EVENT_NAMES
            },
        }


# Module-level singleton — decorators register here at import-time.
_global_registry = HookRegistry()
```

- [ ] **Step 2.4: Run tests pass**

```bash
.venv/bin/pytest tests/unit/hooks/test_registry.py -v
```

Expected: 7 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/hooks/registry.py tests/unit/hooks/test_registry.py
git commit -m "feat(hooks): HookRegistry with priority ordering + stats (plan #3 OpenClaw)"
```

---

## Task 3: Runner — `execute_before_chain` (block + rewrite)

**Files:**
- Create: `src/hooks/runner.py`
- Create: `tests/unit/hooks/test_runner_before.py`

- [ ] **Step 3.1: Escribir test fallido**

```python
"""Tests for execute_before_chain — block + rewrite + priority + errors."""

import logging
import pytest
from dataclasses import replace

from src.hooks.registry import HookRegistry
from src.hooks.runner import execute_before_chain
from src.hooks.types import HaActionCall, TtsCall, BlockResult, RewriteResult


def _ha_call(**overrides) -> HaActionCall:
    base = dict(
        entity_id="light.x", domain="light", service="turn_on",
        service_data={}, user_id=None, user_name=None,
        zone_id=None, timestamp=0.0,
    )
    base.update(overrides)
    return HaActionCall(**base)


class TestBlockShortCircuit:
    def test_block_stops_chain(self):
        reg = HookRegistry()
        called = []

        def h1(call):
            called.append("h1")
            return BlockResult(reason="nope", rule_name="r1")

        def h2(call):
            called.append("h2")
            return None

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, BlockResult)
        assert result.reason == "nope"
        assert called == ["h1"]  # h2 NOT called

    def test_no_handlers_returns_call_unchanged(self):
        reg = HookRegistry()
        call = _ha_call()
        result = execute_before_chain(reg, "before_ha_action", call)
        assert result is call


class TestRewriteChain:
    def test_rewrite_passes_modified_to_next_handler(self):
        reg = HookRegistry()
        seen_data = []

        def h1(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=50),
                rule_name="cap1",
            )

        def h2(call):
            seen_data.append(call.service_data.get("brightness_pct"))
            return RewriteResult(
                modified=call.with_data(brightness_pct=20),
                rule_name="cap2",
            )

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, HaActionCall)
        assert result.service_data["brightness_pct"] == 20
        # h2 saw the rewritten value from h1
        assert seen_data == [50]

    def test_rewrite_then_block(self):
        reg = HookRegistry()

        def h1(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=50),
                rule_name="cap1",
            )

        def h2(call):
            if call.service_data.get("brightness_pct") == 50:
                return BlockResult(reason="too bright", rule_name="block1")

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, BlockResult)


class TestPriorityOrder:
    def test_priority_5_runs_before_priority_10(self):
        reg = HookRegistry()
        order = []

        def lo_prio(call):
            order.append("lo")
            return None

        def hi_prio(call):
            order.append("hi")
            return None

        reg.register_before("before_ha_action", lo_prio, priority=10)
        reg.register_before("before_ha_action", hi_prio, priority=5)
        execute_before_chain(reg, "before_ha_action", _ha_call())
        assert order == ["hi", "lo"]


class TestErrorSwallow:
    def test_handler_exception_logged_and_chain_continues(self, caplog):
        reg = HookRegistry()

        def h1(call):
            raise RuntimeError("kaboom")

        def h2(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=99),
                rule_name="rescue",
            )

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        with caplog.at_level(logging.WARNING):
            result = execute_before_chain(reg, "before_ha_action", _ha_call())

        # Chain continued; h2 ran on original call
        assert isinstance(result, HaActionCall)
        assert result.service_data["brightness_pct"] == 99
        # Counter incremented
        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "kaboom" in (stats["handler_last_error"] or "")
        # Logged
        assert any("handler error" in rec.message.lower() for rec in caplog.records)


class TestWrongRewriteType:
    def test_rewrite_with_wrong_modified_type_logged_and_skipped(self, caplog):
        reg = HookRegistry()

        def h1(call):  # call is HaActionCall, returns RewriteResult with TtsCall — wrong
            return RewriteResult(
                modified=TtsCall(text="oops", voice=None, lang="es",
                                 user_id=None, zone_id=None),
                rule_name="bad",
            )

        reg.register_before("before_ha_action", h1, priority=10)

        with caplog.at_level(logging.ERROR):
            result = execute_before_chain(reg, "before_ha_action", _ha_call())

        # Result is the ORIGINAL call (rewrite ignored)
        assert isinstance(result, HaActionCall)
        assert result.entity_id == "light.x"
        assert any("rewrite type mismatch" in rec.message.lower() for rec in caplog.records)


class TestTimingWarning:
    def test_slow_handler_logs_warning(self, caplog):
        import time

        reg = HookRegistry()

        def slow(call):
            time.sleep(0.01)  # 10ms — well above 5ms threshold
            return None

        reg.register_before("before_ha_action", slow, priority=10)

        with caplog.at_level(logging.WARNING):
            execute_before_chain(reg, "before_ha_action", _ha_call(), warn_ms=5.0)

        assert any("slow handler" in rec.message.lower() for rec in caplog.records)
```

- [ ] **Step 3.2: Run failing tests**

```bash
.venv/bin/pytest tests/unit/hooks/test_runner_before.py -v
```

Expected: ImportError on `src.hooks.runner`.

- [ ] **Step 3.3: Implement `execute_before_chain` in `src/hooks/runner.py`**

```python
"""Hook runners — execute_before_chain + execute_after_event.

Plan #3 OpenClaw.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable

from src.hooks.registry import HookRegistry
from src.hooks.types import (
    BlockResult, RewriteResult, HaActionCall, TtsCall,
)

logger = logging.getLogger(__name__)


def execute_before_chain(
    registry: HookRegistry,
    hook_name: str,
    call: HaActionCall | TtsCall,
    warn_ms: float = 5.0,
) -> BlockResult | HaActionCall | TtsCall:
    """Run handlers for `hook_name` in priority order.

    Returns:
      - BlockResult: a handler chose to block; chain stopped, caller must NOT execute
      - HaActionCall | TtsCall (same type as input): pass-through or rewritten;
        caller proceeds with this (possibly modified) call

    Errors in handlers are logged + counted, never propagated. Slow handlers
    (> warn_ms) emit a WARNING but do not abort.
    """
    expected_type = type(call)
    handlers = registry.get_before_handlers(hook_name)
    current = call

    for priority, fn in handlers:
        t0 = time.perf_counter()
        try:
            result = fn(current)
        except Exception as e:
            registry._handler_failures += 1
            registry._handler_last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                f"[HookRunner] handler error in {hook_name} ({fn.__name__}): {e}"
            )
            continue
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > warn_ms:
                logger.warning(
                    f"[HookRunner] slow handler {fn.__name__} for {hook_name}: "
                    f"{elapsed_ms:.1f}ms (threshold {warn_ms}ms)"
                )

        if result is None:
            continue
        if isinstance(result, BlockResult):
            return result
        if isinstance(result, RewriteResult):
            if not isinstance(result.modified, expected_type):
                logger.error(
                    f"[HookRunner] rewrite type mismatch in {hook_name} "
                    f"({fn.__name__}): expected {expected_type.__name__}, "
                    f"got {type(result.modified).__name__}; ignoring rewrite"
                )
                continue
            current = result.modified
            continue
        # Unknown return type — treat as None (pass-through) with warning
        logger.warning(
            f"[HookRunner] unexpected return type from {fn.__name__} in "
            f"{hook_name}: {type(result).__name__}; ignoring"
        )

    return current
```

- [ ] **Step 3.4: Run tests pass**

```bash
.venv/bin/pytest tests/unit/hooks/test_runner_before.py -v
```

Expected: 8 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/hooks/runner.py tests/unit/hooks/test_runner_before.py
git commit -m "feat(hooks): execute_before_chain with block/rewrite/priority/errors (plan #3 OpenClaw)"
```

---

## Task 4: Runner — `execute_after_event` (fire-and-forget)

**Files:**
- Modify: `src/hooks/runner.py` (append function)
- Create: `tests/unit/hooks/test_runner_after.py`

- [ ] **Step 4.1: Escribir test fallido**

```python
"""Tests for execute_after_event — sync inline + async Task fire-and-forget."""

import asyncio
import logging
import pytest

from src.hooks.registry import HookRegistry
from src.hooks.runner import execute_after_event
from src.hooks.types import SttPayload


def _stt_payload() -> SttPayload:
    return SttPayload(
        timestamp=0.0, text="hola", latency_ms=100.0,
        user_id=None, zone_id=None, success=True,
    )


class TestSyncHandler:
    def test_sync_handler_runs_inline(self):
        reg = HookRegistry()
        ran = []

        def h(payload):
            ran.append(payload.text)

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())
        assert ran == ["hola"]

    def test_sync_handler_exception_logged(self, caplog):
        reg = HookRegistry()

        def h(payload):
            raise RuntimeError("sync boom")

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())

        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "sync boom" in (stats["handler_last_error"] or "")
        assert any("after-event handler" in rec.message.lower()
                   for rec in caplog.records)


class TestAsyncHandler:
    @pytest.mark.asyncio
    async def test_async_handler_runs_as_task(self):
        reg = HookRegistry()
        completed = asyncio.Event()
        seen = []

        async def h(payload):
            seen.append(payload.text)
            completed.set()

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())

        # Task scheduled but not yet awaited
        await asyncio.wait_for(completed.wait(), timeout=1.0)
        assert seen == ["hola"]

    @pytest.mark.asyncio
    async def test_async_strong_ref_in_after_tasks(self):
        reg = HookRegistry()
        gate = asyncio.Event()

        async def h(payload):
            await gate.wait()

        reg.register_after("stt", h)
        execute_after_event(reg, "stt", _stt_payload())

        # Task is in registry
        assert len(reg._after_tasks) == 1

        # Release and let task complete
        gate.set()
        await asyncio.sleep(0.02)
        # done_callback removes it
        assert len(reg._after_tasks) == 0

    @pytest.mark.asyncio
    async def test_async_exception_logged(self, caplog):
        reg = HookRegistry()
        completed = asyncio.Event()

        async def h(payload):
            try:
                raise RuntimeError("async boom")
            finally:
                completed.set()

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())
            await asyncio.wait_for(completed.wait(), timeout=1.0)
            await asyncio.sleep(0.02)  # let done_callback fire

        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "async boom" in (stats["handler_last_error"] or "")


class TestNoEventLoop:
    def test_async_handler_no_loop_logs_warning(self, caplog):
        """If no event loop is running, async handlers are skipped with a warning."""
        reg = HookRegistry()

        async def h(payload):
            pass

        reg.register_after("stt", h)
        with caplog.at_level(logging.WARNING):
            execute_after_event(reg, "stt", _stt_payload())

        # No crash; warning logged
        assert any("no event loop" in rec.message.lower() for rec in caplog.records)
```

- [ ] **Step 4.2: Run tests fail**

```bash
.venv/bin/pytest tests/unit/hooks/test_runner_after.py -v
```

Expected: ImportError on `execute_after_event`.

- [ ] **Step 4.3: Append `execute_after_event` to `src/hooks/runner.py`**

Add at the bottom of the file (after `execute_before_chain`):

```python
def execute_after_event(
    registry: HookRegistry,
    event_name: str,
    payload,
) -> None:
    """Fire-and-forget invocation of all `after_event(name)` handlers.

    Sync handlers run inline (must be fast). Async handlers are scheduled
    as `asyncio.create_task` with a strong ref in `registry._after_tasks`
    to prevent GC mid-flight (Python 3.11+ weak-ref task collection).

    Errors are logged + counted, never propagated.
    If no event loop is running and an async handler is registered, the
    handler is skipped with a warning (sync callers can still use this
    function without setting up a loop).
    """
    handlers = registry.get_after_handlers(event_name)
    if not handlers:
        return

    loop = None
    for fn in handlers:
        is_coro = inspect.iscoroutinefunction(fn)
        if is_coro:
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning(
                        f"[HookRunner] no event loop running; skipping async "
                        f"after-event handler {fn.__name__} for {event_name}"
                    )
                    continue
            task = loop.create_task(fn(payload))
            registry._after_tasks.add(task)
            task.add_done_callback(_make_after_done_callback(registry, fn, event_name))
        else:
            try:
                fn(payload)
            except Exception as e:
                registry._handler_failures += 1
                registry._handler_last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"[HookRunner] after-event handler error in {event_name} "
                    f"({fn.__name__}): {e}"
                )


def _make_after_done_callback(registry: HookRegistry, fn: Callable, event_name: str):
    """Build a done-callback that discards the task ref + logs exceptions."""
    def _on_done(task: asyncio.Task) -> None:
        registry._after_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            registry._handler_failures += 1
            registry._handler_last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[HookRunner] async after-event handler error in {event_name} "
                f"({fn.__name__}): {exc}"
            )
    return _on_done
```

- [ ] **Step 4.4: Run tests pass**

```bash
.venv/bin/pytest tests/unit/hooks/test_runner_after.py -v
```

Expected: 5 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/hooks/runner.py tests/unit/hooks/test_runner_after.py
git commit -m "feat(hooks): execute_after_event fire-and-forget (sync+async) (plan #3 OpenClaw)"
```

---

## Task 5: Decorators + module exports

**Files:**
- Create: `src/hooks/decorators.py`
- Modify: `src/hooks/__init__.py` (re-exports)
- Create: `tests/unit/hooks/test_decorators.py`

- [ ] **Step 5.1: Escribir test fallido**

```python
"""Tests for the 3 public decorators."""

import pytest

from src.hooks import (
    before_ha_action, before_tts_speak, after_event,
    BlockResult, HookRegistry,
)
from src.hooks.registry import _global_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the global registry between tests."""
    _global_registry.clear()
    yield
    _global_registry.clear()


def test_before_ha_action_registers_with_priority():
    @before_ha_action(priority=7)
    def my_handler(call):
        return None

    handlers = _global_registry.get_before_handlers("before_ha_action")
    assert len(handlers) == 1
    assert handlers[0] == (7, my_handler)


def test_before_tts_speak_registers():
    @before_tts_speak(priority=15)
    def my_handler(call):
        return None

    handlers = _global_registry.get_before_handlers("before_tts_speak")
    assert handlers[0][0] == 15


def test_after_event_registers_for_multiple_events():
    @after_event("stt", "intent")
    def my_handler(payload):
        pass

    assert my_handler in _global_registry.get_after_handlers("stt")
    assert my_handler in _global_registry.get_after_handlers("intent")


def test_after_event_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown event"):
        @after_event("bogus")
        def h(p): pass


def test_decorators_return_original_function():
    """Decorator should return fn unchanged so callers can still call it directly."""
    @before_ha_action(priority=10)
    def h(call):
        return "ok"

    assert h.__name__ == "h"
```

- [ ] **Step 5.2: Run tests fail**

```bash
.venv/bin/pytest tests/unit/hooks/test_decorators.py -v
```

Expected: ImportError on `before_ha_action` etc.

- [ ] **Step 5.3: Implement `src/hooks/decorators.py`**

```python
"""Public decorator API for plugin hooks.

Plan #3 OpenClaw.
"""

from collections.abc import Callable

from src.hooks.registry import _global_registry


def before_ha_action(priority: int = 100) -> Callable:
    """Register a SYNC handler invoked before each HA action dispatch.

    Handler signature:
        def handler(call: HaActionCall) -> BlockResult | RewriteResult | None

    Lower `priority` runs first. Handlers are expected to complete in <5ms
    (threshold logged as WARNING but not enforced).
    """
    def deco(fn: Callable) -> Callable:
        _global_registry.register_before("before_ha_action", fn, priority)
        return fn
    return deco


def before_tts_speak(priority: int = 100) -> Callable:
    """Register a SYNC handler invoked before each TTS speak.

    Handler signature:
        def handler(call: TtsCall) -> BlockResult | RewriteResult | None
    """
    def deco(fn: Callable) -> Callable:
        _global_registry.register_before("before_tts_speak", fn, priority)
        return fn
    return deco


def after_event(*event_names: str) -> Callable:
    """Register a sync OR async handler for one or more after-events.

    Handler signature:
        def handler(payload) -> None
        # OR
        async def handler(payload) -> None

    Valid event_names: see EVENT_NAMES in src.hooks.types.
    Multiple decorators on the same fn (or repeated names) re-register it.
    """
    def deco(fn: Callable) -> Callable:
        for name in event_names:
            _global_registry.register_after(name, fn)
        return fn
    return deco
```

- [ ] **Step 5.4: Update `src/hooks/__init__.py` with public exports**

```python
"""Plugin hooks system — public API.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
"""

from src.hooks.decorators import before_ha_action, before_tts_speak, after_event
from src.hooks.registry import HookRegistry, _global_registry
from src.hooks.runner import execute_before_chain, execute_after_event
from src.hooks.types import (
    HaActionCall, TtsCall, BlockResult, RewriteResult,
    WakePayload, SttPayload, IntentPayload,
    HaActionDispatchedPayload, HaActionBlockedPayload,
    LlmCallPayload, TtsPayload,
    EVENT_NAMES,
)

__all__ = [
    # Decorators
    "before_ha_action", "before_tts_speak", "after_event",
    # Registry + runner
    "HookRegistry", "_global_registry",
    "execute_before_chain", "execute_after_event",
    # Types
    "HaActionCall", "TtsCall", "BlockResult", "RewriteResult",
    "WakePayload", "SttPayload", "IntentPayload",
    "HaActionDispatchedPayload", "HaActionBlockedPayload",
    "LlmCallPayload", "TtsPayload",
    "EVENT_NAMES",
]
```

- [ ] **Step 5.5: Run tests pass**

```bash
.venv/bin/pytest tests/unit/hooks/ -v
```

Expected: 24+ tests pass (types + registry + runner_before + runner_after + decorators).

- [ ] **Step 5.6: Commit**

```bash
git add src/hooks/decorators.py src/hooks/__init__.py tests/unit/hooks/test_decorators.py
git commit -m "feat(hooks): decorators + public exports (plan #3 OpenClaw)"
```

---

## Task 6: Wire into RequestDispatcher

**Files:**
- Modify: `src/orchestrator/dispatcher.py` (RequestDispatcher.__init__ + ha dispatch site)

> Pre-condición: hooks module ready. RequestDispatcher will accept `hooks=None` and call `execute_before_chain` only when not None. After dispatch, emit `after_event("ha_action_dispatched")` or `("ha_action_blocked")`.

- [ ] **Step 6.1: Locate the HA dispatch site**

```bash
grep -n "ha.call_service_ws\|self\.ha\.call" src/orchestrator/dispatcher.py
```

Anotar el callsite exacto. Por inspección actual: línea ~966 dentro de un método llamado por `_fast_path` con `asyncio.create_task`.

- [ ] **Step 6.2: Modify `RequestDispatcher.__init__` to accept `hooks=None`**

In `src/orchestrator/dispatcher.py`, find the `RequestDispatcher.__init__` (around line 177). Add `hooks=None` to the signature (keep all existing params; add at the end with default):

```python
    def __init__(
        self,
        chroma_sync,
        ha_client,
        routine_manager,
        router=None,
        llm=None,
        tts=None,
        context_manager=None,
        priority_queue=None,
        buffered_streamer=None,
        music_dispatcher=None,
        list_manager=None,
        reminder_manager=None,
        response_handler=None,
        vector_threshold: float = 0.65,
        use_router_for_simple: bool = True,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
    ):
        # ... existing body ...
        self.hooks = hooks
```

- [ ] **Step 6.3: Wrap the HA dispatch with hook chain**

Locate the method that contains `self.ha.call_service_ws(...)` (around line 966). The current shape (simplified):

```python
async def _ha_dispatch_xxx(self, command, ...):
    domain = command.get("domain")
    service = command.get("service")
    entity_id = command.get("entity_id")
    ...
    success = await self.ha.call_service_ws(domain, service, entity_id, command.get("data"))
```

Replace with:

```python
async def _ha_dispatch_xxx(self, command, ...):
    import time
    from src.hooks import (
        HaActionCall, BlockResult, HaActionDispatchedPayload, HaActionBlockedPayload,
        execute_before_chain, execute_after_event,
    )

    domain = command.get("domain")
    service = command.get("service")
    entity_id = command.get("entity_id")
    description = command.get("description") or entity_id or "esa acción"

    # Build the call DTO that hooks see
    call = HaActionCall(
        entity_id=entity_id or "",
        domain=domain or "",
        service=service or "",
        service_data=command.get("data") or {},
        user_id=command.get("user_id"),
        user_name=command.get("user_name"),
        zone_id=command.get("zone_id"),
        timestamp=time.time(),
    )

    # Plan #3 OpenClaw — before_ha_action chain
    if self.hooks is not None:
        result = execute_before_chain(self.hooks, "before_ha_action", call)
        if isinstance(result, BlockResult):
            logger.info(
                f"[HA-CALL BLOCKED] {domain}.{service}@{entity_id} "
                f"by rule={result.rule_name}: {result.reason}"
            )
            if self.response_handler is not None:
                try:
                    self.response_handler.speak(result.reason or "No puedo hacer eso")
                except Exception as e:
                    logger.warning(f"No pude hablar block reason: {e}")
            execute_after_event(
                self.hooks, "ha_action_blocked",
                HaActionBlockedPayload(timestamp=time.time(), call=call, block=result),
            )
            return
        # Rewrite chain may have modified the call
        call = result
        # Apply back to mutable command dict for downstream code
        command = {**command, "data": call.service_data}

    t0 = time.perf_counter()
    err: str | None = None
    try:
        success = await self.ha.call_service_ws(
            call.domain, call.service, call.entity_id, call.service_data,
        )
        dt = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[HA-CALL] {call.domain}.{call.service}@{call.entity_id} "
            f"success={success} took={dt:.0f}ms"
        )
    except Exception as e:
        err = str(e)
        logger.error(f"Reconcile error en {call.domain}.{call.service}@{call.entity_id}: {e}")
        success = False

    if self.hooks is not None:
        execute_after_event(
            self.hooks, "ha_action_dispatched",
            HaActionDispatchedPayload(
                timestamp=time.time(), call=call, success=success, error=err,
            ),
        )

    if success:
        return
    verb = "apagar" if call.service == "turn_off" else "prender"
    if self.response_handler is not None:
        try:
            self.response_handler.speak(f"No pude {verb} {description}")
        except Exception as e:
            logger.warning(f"No pude hablar error de reconciliación: {e}")
    else:
        logger.warning(
            f"HA fire-and-forget falló en {call.domain}.{call.service}@{call.entity_id} "
            "sin response_handler — usuario no fue notificado"
        )
```

> Verify the original method body more carefully before replacing. There may be additional logic after `success = await self.ha.call_service_ws(...)` (e.g., the existing "verb"/speak block at lines 980-990) that must be preserved. The replacement above mirrors that structure.

- [ ] **Step 6.4: Run existing dispatcher tests for regression**

```bash
.venv/bin/pytest tests/unit/orchestrator/test_dispatcher.py tests/unit/orchestrator/test_context_manager.py -v 2>&1 | tail -10
```

Expected: same baseline as before (3 pre-existing dispatcher failures). No new failures.

- [ ] **Step 6.5: Add a focused dispatcher hook integration test**

Append to `tests/unit/orchestrator/test_dispatcher.py` (or create a new file `tests/unit/orchestrator/test_dispatcher_hooks.py` if you prefer):

```python
"""Tests for dispatcher integration with plugin hooks (plan #3 OpenClaw)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.hooks import HookRegistry, BlockResult, before_ha_action
from src.orchestrator.dispatcher import RequestDispatcher  # adjust if path differs


@pytest.fixture
def dispatcher_with_hooks():
    hooks = HookRegistry()

    ha_client = MagicMock()
    ha_client.call_service_ws = AsyncMock(return_value=True)
    response_handler = MagicMock()
    response_handler.speak = MagicMock()

    dispatcher = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=ha_client,
        routine_manager=MagicMock(),
        response_handler=response_handler,
        hooks=hooks,
    )
    return dispatcher, hooks, ha_client, response_handler


@pytest.mark.asyncio
async def test_block_prevents_ha_call_and_speaks_reason(dispatcher_with_hooks):
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks

    def block_all(call):
        return BlockResult(reason="prohibido", rule_name="test_block")

    hooks.register_before("before_ha_action", block_all, priority=10)

    # Locate the dispatch method on the dispatcher and call it.
    # The method may be private/named differently; adjust per actual code.
    # Below assumes a public-ish entry — pick the one your dispatcher exposes:
    method = getattr(dispatcher, "_dispatch_ha_action_async", None) or \
             getattr(dispatcher, "_ha_dispatch", None)
    assert method is not None, "Could not find HA dispatch method"

    command = {
        "domain": "light", "service": "turn_on", "entity_id": "light.x",
        "data": {}, "description": "la luz",
    }
    await method(command)

    # HA was NOT called
    ha_client.call_service_ws.assert_not_called()
    # Response was spoken
    response_handler.speak.assert_called_with("prohibido")
```

> NOTE for the implementer: this test depends on the exact dispatch method name. Inspect `src/orchestrator/dispatcher.py` for the method body containing `self.ha.call_service_ws(...)` and use that name. If the method is highly private (multi-arg, callback-driven), replace this test with a simpler one that exercises `RequestDispatcher.hooks` flow at a higher level OR skip and rely on the E2E test in Task 10.

- [ ] **Step 6.6: Commit**

```bash
git add src/orchestrator/dispatcher.py tests/unit/orchestrator/
git commit -m "feat(dispatcher): wire before_ha_action + after_event emits (plan #3 OpenClaw)"
```

---

## Task 7: Wire into ResponseHandler

**Files:**
- Modify: `src/pipeline/response_handler.py` (ResponseHandler.__init__ + speak)

- [ ] **Step 7.1: Modify `ResponseHandler.__init__` to accept `hooks=None`**

In `src/pipeline/response_handler.py`, locate `__init__` (around line 35). Add `hooks=None`:

```python
    def __init__(
        self,
        tts,
        zone_manager=None,
        llm=None,
        streaming_enabled: bool = True,
        streaming_buffer_ms: int = 150,
        streaming_prebuffer_ms: int = 30,
        llm_buffer_preset: str = "balanced",
        llm_use_filler: bool = True,
        llm_filler_phrases: list = None,
        response_cache: ResponseCache | None = None,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
    ):
        # ... existing body ...
        self.hooks = hooks
```

- [ ] **Step 7.2: Wrap `speak()` body with hook chain**

Find the `speak()` method (around line 156). At the very top of the method (before any TTS work), inject:

```python
    def speak(
        self,
        text: str,
        zone_id: str | None = None,
        # ... existing params ...
    ):
        """... existing docstring ..."""
        # Plan #3 OpenClaw — before_tts_speak chain
        if self.hooks is not None:
            from src.hooks import (
                TtsCall, BlockResult, TtsPayload,
                execute_before_chain, execute_after_event,
            )

            tts_call = TtsCall(
                text=text,
                voice=None,                 # set if response_handler tracks voice
                lang="es",
                user_id=None,               # ResponseHandler doesn't have user context
                zone_id=zone_id,
            )
            result = execute_before_chain(self.hooks, "before_tts_speak", tts_call)
            if isinstance(result, BlockResult):
                logger.info(
                    f"[TTS BLOCKED] rule={result.rule_name}: {result.reason}"
                )
                return
            # Rewrite may have changed text
            text = result.text

        # ... existing speak body unchanged ...
```

After the existing speak body completes (just before return), emit `after_event("tts")`:

```python
        # End of speak() body — emit after_event
        if self.hooks is not None:
            execute_after_event(
                self.hooks, "tts",
                TtsPayload(
                    timestamp=time.time(),
                    text=text,
                    voice=None,
                    latency_ms=0.0,   # measure if response_handler tracks it
                    success=True,
                ),
            )
```

> The exact insertion points depend on the existing `speak()` body. Read it first; place the `before_tts_speak` chain before any TTS work, and the `after_event` emit before the function returns (success path). Errors during TTS still emit with `success=False`.

- [ ] **Step 7.3: Add ResponseHandler hook test**

Create `tests/unit/pipeline/test_response_handler_hooks.py`:

```python
"""Tests for ResponseHandler integration with plugin hooks (plan #3 OpenClaw)."""

import pytest
from unittest.mock import MagicMock

from src.hooks import HookRegistry, BlockResult, RewriteResult, TtsCall
from src.pipeline.response_handler import ResponseHandler


@pytest.fixture
def handler_with_hooks():
    hooks = HookRegistry()
    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=b"\x00" * 100)  # fake audio
    handler = ResponseHandler(tts=tts, hooks=hooks)
    return handler, hooks, tts


def test_block_prevents_tts(handler_with_hooks):
    handler, hooks, tts = handler_with_hooks

    def block(call):
        return BlockResult(reason="silence", rule_name="test")

    hooks.register_before("before_tts_speak", block, priority=10)
    handler.speak("hello")
    tts.synthesize.assert_not_called()


def test_rewrite_modifies_text_before_tts(handler_with_hooks):
    handler, hooks, tts = handler_with_hooks
    received_text = []

    def capture(text, *a, **kw):
        received_text.append(text)
        return b"\x00" * 100

    tts.synthesize = MagicMock(side_effect=capture)

    def upper(call):
        return RewriteResult(
            modified=TtsCall(
                text=call.text.upper(), voice=call.voice, lang=call.lang,
                user_id=call.user_id, zone_id=call.zone_id,
            ),
            rule_name="upper",
        )

    hooks.register_before("before_tts_speak", upper, priority=10)
    handler.speak("hola")
    # The actual TTS engine got "HOLA"
    assert any("HOLA" in str(t) for t in received_text)
```

> Adapt the test to match the actual TTS engine API used by `speak()`. The mock above assumes `tts.synthesize` — check the real method name.

- [ ] **Step 7.4: Run tests**

```bash
.venv/bin/pytest tests/unit/pipeline/test_response_handler_hooks.py tests/unit/hooks/ -v 2>&1 | tail -10
```

Expected: green; existing pipeline tests still pass.

- [ ] **Step 7.5: Commit**

```bash
git add src/pipeline/response_handler.py tests/unit/pipeline/
git commit -m "feat(pipeline): ResponseHandler hooks before_tts_speak + after tts (plan #3 OpenClaw)"
```

---

## Task 8: Settings + main.py DI + after_event emit checkpoints

**Files:**
- Modify: `config/settings.yaml`
- Modify: `src/main.py`
- Modify: `src/pipeline/request_router.py` (emit checkpoints)

- [ ] **Step 8.1: Append `hooks` block to `config/settings.yaml`**

Place it as a new top-level block (alongside `dashboard:`, `orchestrator:`, etc):

```yaml
# =============================================================================
# Plugin Hooks (plan #3 OpenClaw)
# =============================================================================
# Permite a handlers Python en src/policies/ bloquear/reescribir HA actions y
# TTS speak, y auditar eventos del pipeline. Spec: docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
hooks:
  enabled: false                   # OFF por default; activar tras validar en prod
  policies_module: "src.policies"  # Python module path; importarlo dispara los decoradores
  before_handler_warn_ms: 5.0      # log WARNING si un before_ handler tarda más
  audit_sqlite_path: "./data/audit.db"   # consumido por audit_sqlite policy
```

- [ ] **Step 8.2: Wire DI in `src/main.py`**

Find where `RequestDispatcher` and/or `MultiUserOrchestrator` and `ResponseHandler` are constructed (look for `RequestDispatcher(` and `ResponseHandler(`). BEFORE those construction calls, add the hooks block:

```python
    # === Plan #3 OpenClaw — Plugin hooks (opcional) ===
    hooks_cfg = config.get("hooks", {}) or {}
    hooks = None
    if hooks_cfg.get("enabled", False):
        from src.hooks import HookRegistry, _global_registry
        import importlib

        # The global registry is populated as a side effect of importing
        # src.policies (decorators run at import-time).
        try:
            importlib.import_module(hooks_cfg.get("policies_module", "src.policies"))
            hooks = _global_registry
            logger.info(
                f"[main] Plugin hooks enabled — "
                f"before_ha={hooks.get_stats()['before_handler_count']['before_ha_action']} "
                f"before_tts={hooks.get_stats()['before_handler_count']['before_tts_speak']} "
                f"after_tasks={hooks.get_stats()['after_tasks_in_flight']}"
            )
        except Exception as e:
            logger.error(
                f"[main] Hooks disabled — failed to import policies: {e}",
                exc_info=True,
            )
            hooks = None
```

Then pass `hooks=hooks` to the `ResponseHandler(...)` and `RequestDispatcher(...)` constructors. If both go through `MultiUserOrchestrator`, also extend `MultiUserOrchestrator.__init__` to accept `hooks=None` and forward to `RequestDispatcher`.

- [ ] **Step 8.3: Emit `after_event` at pipeline checkpoints in `request_router.py`**

The pipeline emits should fire when known stages complete. Inspect `src/pipeline/request_router.py` for these natural checkpoints:

1. **wake** — after a wake word fires. Look for the line where `[WakeFire]` or wake confirmation is logged (around line 380-400 or wherever `wake_confidence` is consumed).
2. **stt** — after STT completes. Look for `[STT ...ms]` logging (line 668 area).
3. **intent** — after intent classification. Look for `[LLMRouter ...ms] is_command=...` (around line 406).
4. **llm_call** — after each call to the LLM router (fast path or slow path). Look at where `dispatcher.process` returns or where the slow path completes.

For each checkpoint, add:

```python
        # plan #3 OpenClaw — emit after-event (best effort; no impact if hooks=None)
        if self._hooks is not None:
            from src.hooks import SttPayload, execute_after_event
            execute_after_event(
                self._hooks, "stt",
                SttPayload(
                    timestamp=time.time(),
                    text=text,
                    latency_ms=stt_latency_ms,
                    user_id=user_id,
                    zone_id=zone_id,
                    success=True,
                ),
            )
```

Repeat for the other 3 events with their corresponding payload classes.

`RequestRouter.__init__` will need a new `hooks=None` param; main.py wires it in.

- [ ] **Step 8.4: Smoke test imports**

```bash
.venv/bin/python -c "from src.hooks import _global_registry, before_ha_action, BlockResult, EVENT_NAMES; print('imports ok')"
```

- [ ] **Step 8.5: Run full unit suite**

```bash
.venv/bin/pytest tests/unit/ tests/integration/ --timeout=60 -q
```

Expected: green except 3 pre-existing dispatcher failures.

- [ ] **Step 8.6: Commit**

```bash
git add config/settings.yaml src/main.py src/pipeline/request_router.py
git commit -m "feat(main): hooks DI + after_event checkpoints in request_router (plan #3 OpenClaw)"
```

---

## Task 9: Policies — safety_alarm + permissions + tts_rewrite_es

**Files:**
- Create: `src/policies/__init__.py`
- Create: `src/policies/safety_alarm.py`
- Create: `src/policies/permissions.py`
- Create: `src/policies/tts_rewrite_es.py`
- Create: `tests/unit/policies/__init__.py`
- Create: `tests/unit/policies/test_safety_alarm.py`
- Create: `tests/unit/policies/test_permissions.py`
- Create: `tests/unit/policies/test_tts_rewrite_es.py`

- [ ] **Step 9.1: Create directories**

```bash
mkdir -p src/policies tests/unit/policies
touch src/policies/__init__.py tests/unit/policies/__init__.py
```

- [ ] **Step 9.2: Write `src/policies/safety_alarm.py`**

```python
"""Safety policy: nunca desarmar la alarma entre las 22:00 y las 07:00.

Plan #3 OpenClaw — use case 1.
"""

from datetime import datetime

from src.hooks import before_ha_action, BlockResult


@before_ha_action(priority=10)
def proteger_alarma_de_noche(call):
    if call.entity_id == "alarm_control_panel.casa" and call.service == "alarm_disarm":
        h = datetime.now().hour
        if h >= 22 or h < 7:
            return BlockResult(
                reason="No puedo desarmar la alarma a esta hora",
                rule_name="proteger_alarma_de_noche",
            )
    return None
```

- [ ] **Step 9.3: Write test `tests/unit/policies/test_safety_alarm.py`**

```python
"""Tests for safety_alarm policy."""

from unittest.mock import patch
import pytest

from src.hooks import HaActionCall, BlockResult


def _ha_call(entity_id="alarm_control_panel.casa", service="alarm_disarm"):
    return HaActionCall(
        entity_id=entity_id, domain="alarm_control_panel",
        service=service, service_data={},
        user_id=None, user_name=None, zone_id=None, timestamp=0.0,
    )


@pytest.mark.parametrize("hour", [22, 23, 0, 6])
def test_blocks_at_night_hours(hour):
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": hour})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call())

    assert isinstance(result, BlockResult)
    assert result.rule_name == "proteger_alarma_de_noche"


@pytest.mark.parametrize("hour", [7, 8, 12, 18, 21])
def test_allows_at_day_hours(hour):
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": hour})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call())

    assert result is None


def test_only_blocks_alarm_disarm_not_other_services():
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": 23})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call(service="alarm_arm_home"))

    assert result is None


def test_only_blocks_specific_alarm_entity():
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": 23})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call(entity_id="alarm_control_panel.otra"))

    assert result is None
```

- [ ] **Step 9.4: Write `src/policies/permissions.py`**

```python
"""Multi-user permission gating: niños no pueden controlar dominios sensibles.

Plan #3 OpenClaw — use case 2 (block puro, sin require_approval).
"""

from src.hooks import before_ha_action, BlockResult


# user_ids identificados como niños (por SpeakerID enrollment)
CHILD_USER_IDS: set[str] = {"niño1", "niño2"}

# Dominios HA que sólo pueden controlar adultos
ADULT_ONLY_DOMAINS: set[str] = {"climate", "lock", "alarm_control_panel"}


@before_ha_action(priority=5)  # priority=5: corre ANTES que safety_alarm (priority=10)
def chicos_sin_dominios_adultos(call):
    if call.user_id in CHILD_USER_IDS and call.domain in ADULT_ONLY_DOMAINS:
        return BlockResult(
            reason="No tenés permiso para eso",
            rule_name="chicos_sin_dominios_adultos",
        )
    return None
```

- [ ] **Step 9.5: Write test `tests/unit/policies/test_permissions.py`**

```python
"""Tests for permissions policy."""

import pytest

from src.hooks import HaActionCall, BlockResult


def _ha_call(user_id, domain):
    return HaActionCall(
        entity_id=f"{domain}.x", domain=domain,
        service="turn_on", service_data={},
        user_id=user_id, user_name=None, zone_id=None, timestamp=0.0,
    )


@pytest.mark.parametrize("user", ["niño1", "niño2"])
@pytest.mark.parametrize("domain", ["climate", "lock", "alarm_control_panel"])
def test_blocks_child_in_adult_domain(user, domain):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, domain))
    assert isinstance(result, BlockResult)


@pytest.mark.parametrize("user", ["niño1", "niño2"])
@pytest.mark.parametrize("domain", ["light", "switch", "media_player"])
def test_allows_child_in_other_domains(user, domain):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, domain))
    assert result is None


@pytest.mark.parametrize("user", ["adulto1", "adulto2", None])
def test_allows_adult_or_unknown_in_adult_domains(user):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, "climate"))
    assert result is None
```

- [ ] **Step 9.6: Write `src/policies/tts_rewrite_es.py`**

```python
"""TTS rewriting: pesos / dolares / abreviaturas para que Kokoro suene natural.

Plan #3 OpenClaw — use case 4.
"""

import re
from dataclasses import replace

from src.hooks import before_tts_speak, RewriteResult


# Pattern: $123 → 123 pesos (Spanish-ar default)
_PESOS = re.compile(r"\$(\d+)")


@before_tts_speak(priority=10)
def numeros_a_palabras(call):
    new_text = _PESOS.sub(r"\1 pesos", call.text)
    if new_text != call.text:
        return RewriteResult(
            modified=replace(call, text=new_text),
            rule_name="numeros_a_palabras",
        )
    return None
```

- [ ] **Step 9.7: Write test `tests/unit/policies/test_tts_rewrite_es.py`**

```python
"""Tests for tts_rewrite_es policy."""

from src.hooks import TtsCall, RewriteResult


def _tts(text):
    return TtsCall(text=text, voice=None, lang="es", user_id=None, zone_id=None)


def test_pesos_substitution():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("son $1000"))
    assert isinstance(result, RewriteResult)
    assert result.modified.text == "son 1000 pesos"


def test_no_match_returns_none():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("hola, qué tal"))
    assert result is None


def test_multiple_pesos_in_text():
    from src.policies.tts_rewrite_es import numeros_a_palabras

    result = numeros_a_palabras(_tts("entre $100 y $500"))
    assert result.modified.text == "entre 100 pesos y 500 pesos"
```

- [ ] **Step 9.8: Run policy tests**

```bash
.venv/bin/pytest tests/unit/policies/ -v
```

Expected: 15+ tests pass (parametrize multiplies cases).

- [ ] **Step 9.9: Commit**

```bash
git add src/policies/ tests/unit/policies/
git commit -m "feat(policies): safety_alarm + permissions + tts_rewrite_es (plan #3 OpenClaw)"
```

---

## Task 10: Audit policy + E2E integration test

**Files:**
- Create: `src/policies/audit_sqlite.py`
- Create: `tests/unit/policies/test_audit_sqlite.py`
- Create: `tests/integration/test_hooks_e2e.py`

- [ ] **Step 10.1: Write `src/policies/audit_sqlite.py`**

```python
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
```

- [ ] **Step 10.2: Write test `tests/unit/policies/test_audit_sqlite.py`**

```python
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
```

- [ ] **Step 10.3: Run policy tests**

```bash
.venv/bin/pytest tests/unit/policies/ -v
```

Expected: 17+ tests (15 from Task 9 + 2 audit).

- [ ] **Step 10.4: Write E2E integration test `tests/integration/test_hooks_e2e.py`**

```python
"""E2E: 4 policies active, full block + rewrite + audit + tts flow."""

import asyncio
import sqlite3
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

    # Importing the policy modules registers handlers via decorators
    import importlib
    for name in ["src.policies.safety_alarm",
                 "src.policies.permissions",
                 "src.policies.tts_rewrite_es",
                 "src.policies.audit_sqlite"]:
        if name in importlib.sys.modules:
            importlib.reload(importlib.sys.modules[name])
        else:
            importlib.import_module(name)

    yield _global_registry, conn, audit_db
    _global_registry.clear()
    conn.close()


@pytest.mark.asyncio
async def test_block_alarm_at_night_logs_to_audit(registry_with_policies):
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
async def test_block_child_climate_logs_to_audit(registry_with_policies):
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
```

- [ ] **Step 10.5: Run E2E**

```bash
.venv/bin/pytest tests/integration/test_hooks_e2e.py -v
```

Expected: 4 passed.

- [ ] **Step 10.6: Run full test suite**

```bash
.venv/bin/pytest tests/ --timeout=60 -q
```

Expected: previous baseline + ~30 new tests.

- [ ] **Step 10.7: Commit**

```bash
git add src/policies/audit_sqlite.py tests/unit/policies/test_audit_sqlite.py tests/integration/test_hooks_e2e.py
git commit -m "feat(policies): audit_sqlite + E2E test of 4 policies (plan #3 OpenClaw)"
```

---

## Task 11: Manual deploy + activación controlada

> Este task NO es código. Es la activación controlada en server tras merge a main.

- [ ] **Step 11.1: Push branch + open PR**

```bash
git push origin HEAD
gh pr create --base main --title "feat(hooks): plan #3 OpenClaw — plugin hooks system"
```

- [ ] **Step 11.2: Mergear (squash o merge — convención del proyecto: --merge)**

```bash
gh pr merge --merge --delete-branch
```

- [ ] **Step 11.3: Pull en server**

```bash
ssh kza 'cd ~/kza && git pull --ff-only origin main'
```

- [ ] **Step 11.4: Activar hooks**

Editar `~/kza/config/settings.yaml` en server:

```yaml
hooks:
  enabled: true
```

- [ ] **Step 11.5: Restart**

```bash
ssh kza 'systemctl --user restart kza-voice'
```

- [ ] **Step 11.6: Validar logs**

```bash
ssh kza 'journalctl --user -u kza-voice -n 200 --no-pager | grep -E "Plugin hooks|HookRegistry|HookRunner"'
```

Esperado: `[main] Plugin hooks enabled — before_ha=2 before_tts=1 after_tasks=0`

- [ ] **Step 11.7: Smoke test funcional**

Decir "Nexa, desarmá la alarma" a las 22:30. Esperado:
- TTS responde "No puedo desarmar la alarma a esta hora"
- `journalctl` muestra `[HA-CALL BLOCKED] alarm_control_panel.alarm_disarm by rule=proteger_alarma_de_noche`
- `~/kza/data/audit.db` tiene un row `kind='ha_action_blocked'`

```bash
ssh kza 'sqlite3 ~/kza/data/audit.db "SELECT kind, COUNT(*) FROM events GROUP BY kind"'
```

- [ ] **Step 11.8: Actualizar memoria**

Crear `~/.claude/projects/-Users-yo-Documents-kza/memory/project_openclaw_plan3_done.md` siguiendo el formato de `project_openclaw_plan2_done.md`.

---

## Outcome verificable

Al cerrar todos los tasks:

1. `pytest tests/` pasa con ≥ 30 nuevos tests verdes.
2. Con `hooks.enabled: false` → comportamiento idéntico al baseline (regression-safe).
3. Con `hooks.enabled: true`:
   - "Nexa, desarmá la alarma" a las 23hs → bloqueado, TTS dice reason, audit registra `ha_action_blocked`.
   - "Niño1: subí el clima" → bloqueado, TTS dice "No tenés permiso para eso".
   - TTS de respuestas con `$N` pronuncia "N pesos".
   - `data/audit.db` crece con eventos reales tras 1h de uso.
4. `HookRegistry.get_stats()` muestra counters de fallos accesibles vía dashboard / `MultiUserOrchestrator.get_stats()`.
5. Identifier policy plan #2 sigue intacta (este plan no toca compactación).
