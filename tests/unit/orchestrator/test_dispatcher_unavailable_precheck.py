"""An unavailable entity must fail loudly, not be silently swallowed by HA."""
import pytest

from src.home_assistant.ha_client import HomeAssistantClient


def test_is_entity_available_true_for_live_entity():
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {"light.grupo_living": {"state": "off"}}
    assert ha.is_entity_available("light.grupo_living") is True


@pytest.mark.parametrize("state", ["unavailable", "unknown"])
def test_is_entity_available_false_for_dead_entity(state):
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {"light.grupo_cuarto": {"state": state}}
    assert ha.is_entity_available("light.grupo_cuarto") is False


def test_is_entity_available_none_when_not_cached():
    """No data is not the same as unavailable — the caller must fail open."""
    ha = HomeAssistantClient.__new__(HomeAssistantClient)
    ha._state_cache = {}
    assert ha.is_entity_available("light.desconocida") is None


import asyncio
from unittest.mock import AsyncMock, MagicMock


def _dispatcher_with(ha_available, dispatcher_cls, precheck_enabled=True):
    d = dispatcher_cls.__new__(dispatcher_cls)
    d.ha = MagicMock()
    d.ha.is_entity_available = MagicMock(return_value=ha_available)
    d.ha.call_service_ws = AsyncMock(return_value=True)
    d.response_handler = MagicMock()
    d.hooks = None
    d._unavailable_precheck_enabled = precheck_enabled
    return d


def test_unavailable_entity_is_not_called_and_plays_earcon():
    from src.orchestrator.dispatcher import RequestDispatcher

    d = _dispatcher_with(False, RequestDispatcher)
    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.grupo_cuarto", "service_data": {},
        "description": "luz del cuarto", "zone_id": "cocina",
    }))
    d.ha.call_service_ws.assert_not_awaited()
    d.response_handler.play_earcon.assert_called_once()


def test_unknown_availability_fails_open_and_calls():
    """None means 'no data', not 'broken'. We must still try."""
    from src.orchestrator.dispatcher import RequestDispatcher

    d = _dispatcher_with(None, RequestDispatcher)
    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.nueva", "service_data": {},
        "description": "luz nueva", "zone_id": "cocina",
    }))
    d.ha.call_service_ws.assert_awaited_once()


# --- Review findings (post Task 2): audit trail + kill switch --------------
#
# Important 1: el precheck era la única salida de _fire_and_reconcile_ha que
# no emitía evento de hook, dejando el comando retenido invisible para
# src/policies/audit_sqlite.py. Fix: emitir ha_action_blocked con un
# BlockResult sintético (rule_name="entity_unavailable") antes del return.
#
# Important 2: el precheck podía retener un comando de domótica sin cota de
# frescura ni kill switch (riesgo: reinicio de Z2M/HA + WS de events caído
# hasta 5 min). Fix: config/settings.yaml:home_assistant.
# unavailable_precheck_enabled (default true), forwardeado a
# RequestDispatcher._unavailable_precheck_enabled.

def test_unavailable_entity_emits_ha_action_blocked_for_audit_trail():
    """El comando retenido debe quedar en el audit trail (Important 1)."""
    from src.hooks import HookRegistry, HaActionBlockedPayload
    from src.orchestrator.dispatcher import RequestDispatcher

    hooks = HookRegistry()
    captured: list = []
    hooks.register_after("ha_action_blocked", captured.append)

    d = _dispatcher_with(False, RequestDispatcher)
    d.hooks = hooks
    d._before_handler_warn_ms = 5.0

    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.grupo_cuarto", "service_data": {},
        "description": "luz del cuarto", "zone_id": "cocina",
    }))

    d.ha.call_service_ws.assert_not_awaited()
    assert len(captured) == 1
    payload = captured[0]
    assert isinstance(payload, HaActionBlockedPayload)
    assert payload.block.rule_name == "entity_unavailable"
    assert payload.call.entity_id == "light.grupo_cuarto"


def test_unavailable_precheck_disabled_still_calls_ha():
    """Con el kill switch apagado, se llama a HA aunque la entidad esté
    unavailable en cache (Important 2)."""
    from src.orchestrator.dispatcher import RequestDispatcher

    d = _dispatcher_with(False, RequestDispatcher, precheck_enabled=False)
    asyncio.run(d._fire_and_reconcile_ha({
        "domain": "light", "service": "turn_on",
        "entity_id": "light.grupo_cuarto", "service_data": {},
        "description": "luz del cuarto", "zone_id": "cocina",
    }))
    d.ha.call_service_ws.assert_awaited_once()


# --- Re-review: poder de detección sobre el cableado del flag --------------
#
# Mutation testing sobre el commit anterior encontró que los tests de arriba
# usan `dispatcher_cls.__new__(dispatcher_cls)` y setean
# `_unavailable_precheck_enabled` como atributo crudo — ejercitan el *uso*
# del flag pero nunca el *constructor real* ni el reenvío
# MultiUserOrchestrator → RequestDispatcher. Dos mutaciones sobrevivieron:
#
# Mutación A: invertir el default de RequestDispatcher.__init__
#   (True → False). 8/8 tests de este archivo seguían verdes.
# Mutación B: borrar `unavailable_precheck_enabled=unavailable_precheck_enabled,`
#   del RequestDispatcher(...) interno de MultiUserOrchestrator.__init__.
#   Suite completa: mismo resultado que el baseline (ningún test nuevo caía).
#
# Los dos tests siguientes construyen los objetos reales vía __init__ (no
# __new__) para cerrar esa brecha.

def test_dispatcher_default_precheck_is_enabled_via_real_init():
    """Construcción real de RequestDispatcher, sin pasar el flag: el default
    vivo debe ser True. Debe ponerse rojo ante la Mutación A (invertir el
    default en la firma de __init__)."""
    from unittest.mock import MagicMock as _MM

    from src.orchestrator.dispatcher import RequestDispatcher

    d = RequestDispatcher(
        chroma_sync=_MM(),
        ha_client=_MM(),
        routine_manager=_MM(),
    )
    assert d._unavailable_precheck_enabled is True


def test_orchestrator_forwards_precheck_flag_to_dispatcher_via_real_init():
    """Construcción real de MultiUserOrchestrator con el flag en False: debe
    llegar al RequestDispatcher interno. Debe ponerse rojo ante la
    Mutación B (borrar el forwarding del kwarg)."""
    from unittest.mock import MagicMock as _MM

    from src.orchestrator.dispatcher import MultiUserOrchestrator

    orchestrator = MultiUserOrchestrator(
        chroma_sync=_MM(),
        ha_client=_MM(),
        routine_manager=_MM(),
        unavailable_precheck_enabled=False,
    )
    assert orchestrator.dispatcher._unavailable_precheck_enabled is False
