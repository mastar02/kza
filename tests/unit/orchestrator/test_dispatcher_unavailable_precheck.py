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


def _dispatcher_with(ha_available, dispatcher_cls):
    d = dispatcher_cls.__new__(dispatcher_cls)
    d.ha = MagicMock()
    d.ha.is_entity_available = MagicMock(return_value=ha_available)
    d.ha.call_service_ws = AsyncMock(return_value=True)
    d.response_handler = MagicMock()
    d.hooks = None
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
