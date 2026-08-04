"""Weather routing, and the thermostat collision it must not cause."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import PathType, Priority, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.world.weather import NO_DATA, NO_FORECAST


@pytest.fixture
def dispatcher():
    d = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
    )
    # Truthy stand-in so the music classification branch runs (gated on
    # `self.music`) — same pattern as test_dispatcher_music_routing.py.
    # Needed for the FAST_MUSIC cases in test_existing_paths_do_not_regress.
    d.music = MagicMock()
    return d


@pytest.mark.parametrize("text", [
    "qué tiempo hace",
    "cómo está el clima",
    "qué temperatura hace",
    "qué temperatura hace afuera",
    "cuántos grados hay afuera",
    "llueve mañana",
    "va a llover mañana",
    "cómo está el clima mañana",
    "el pronóstico para mañana",
])
def test_weather_questions_route_to_fast_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_WEATHER


@pytest.mark.parametrize("text", [
    # The collision this task exists for: these are the THERMOSTAT.
    "poné la temperatura en 22",
    "subí la temperatura",
    "bajá la temperatura del living",
    "prendé la calefacción",
    "apagá el aire",
])
def test_thermostat_commands_do_not_become_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_DOMOTICS


@pytest.mark.parametrize("text", [
    # Coordinator review finding (2026-08-04): bare "el clima" in
    # WEATHER_KEYWORDS swallowed these AC/thermostat commands. Minimum set
    # from the review, plus other _NON_LIGHT_DOMAIN_NOUNS climate synonyms
    # (termostato, grados) exercised the same way, to cover the whole class
    # of collision rather than just the literal reported cases.
    "prendé el clima",
    "prende el clima",
    "apagá el clima",
    "apaga el clima",
    "activá el clima",
    "poné el clima en 22",
    "apagá el clima del living",
    "apagá el termostato",
    "prendé el termostato",
    "subí los grados",
    "bajá los grados del termostato",
])
def test_ac_commands_with_clima_noun_do_not_become_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_DOMOTICS


@pytest.mark.parametrize("text,expected", [
    # Finding 3 (re-review 2026-08-04): the Phase-1 guard ("verb anywhere +
    # climate noun anywhere") was too broad — it also swallowed genuine
    # weather questions that merely mention a domotics verb in another
    # clause. The six Critical/collision cases below are the same ones
    # from test_ac_commands_with_clima_noun_do_not_become_weather (verb
    # immediately adjacent to the climate noun) and MUST keep routing
    # fast_domotics; the three hybrid cases are the sharpest counter-
    # examples the reviewer found (verb and noun co-occur but not
    # adjacent, and/or the utterance is a question) and MUST route
    # fast_weather. Parametrized together so neither direction can
    # silently flip again.
    ("prendé el clima", PathType.FAST_DOMOTICS),
    ("poné el clima en 22", PathType.FAST_DOMOTICS),
    ("apagá el clima del living", PathType.FAST_DOMOTICS),
    ("apagá el termostato", PathType.FAST_DOMOTICS),
    ("prendé el termostato", PathType.FAST_DOMOTICS),
    ("subí los grados", PathType.FAST_DOMOTICS),
    ("está el clima muy caluroso, tengo que activar el aire?", PathType.FAST_WEATHER),
    ("está el clima bien, no hace falta prender nada", PathType.FAST_WEATHER),
    ("¿tengo que prender el clima o hace calor afuera?", PathType.FAST_WEATHER),
])
def test_domotics_climate_adjacency_guard_finding_3(dispatcher, text, expected):
    path, _ = dispatcher._classify_request(text.lower())
    assert path == expected


@pytest.mark.parametrize("text,expected", [
    ("poné música de Spinetta", PathType.FAST_MUSIC),
    ("subí el volumen", PathType.FAST_MUSIC),
    ("prendé la luz del living", PathType.FAST_DOMOTICS),
    ("encender la luz", PathType.FAST_DOMOTICS),
    ("agrega leche a la lista de compras", PathType.FAST_LIST),
    ("recuérdame sacar la basura", PathType.FAST_REMINDER),
    ("por qué el cielo es azul", PathType.SLOW_LLM),
])
def test_existing_paths_do_not_regress(dispatcher, text, expected):
    path, _ = dispatcher._classify_request(text)
    assert path == expected


def test_service_filter_still_wins(dispatcher):
    # An upstream high-confidence domotics classification always wins.
    path, _ = dispatcher._classify_request(
        "subí la temperatura", service_filter="turn_on"
    )
    assert path == PathType.FAST_DOMOTICS


# ---------------------------------------------------------------------------
# Finding 2 (coordinator review, 2026-08-04): _handle_weather had zero test
# coverage. Hand-tracing found the (payload or {}) / chained .get(..., {})
# guards degrade safely, but that needs to be proven, not assumed — these
# tests exercise both the "hoy" (get_entity_state_cached) and "mañana"
# (call_service_with_response) branches with None and malformed HA
# responses, asserting the honest fallback string and nothing that could be
# spoken as a literal "None" or raise.
# ---------------------------------------------------------------------------

@pytest.fixture
def dispatcher_with_async_ha():
    d = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=AsyncMock(),
        routine_manager=MagicMock(),
        router=None,
        llm=None,
        context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(),
    )
    d.music = MagicMock()
    return d


async def test_handle_weather_forecast_none_response_is_honest_fallback(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value=None)

    result = await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)

    assert result.success is True
    assert result.response == NO_FORECAST
    assert "None" not in result.response


@pytest.mark.parametrize("malformed_payload", [
    {},
    {"service_response": {}},
    {"service_response": {"weather.forecast_home": {}}},
    {"service_response": {"weather.forecast_home": {"forecast": None}}},
    {"unexpected_shape": True},
])
async def test_handle_weather_forecast_malformed_response_is_honest_fallback(
    dispatcher_with_async_ha, malformed_payload
):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value=malformed_payload)

    result = await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)

    assert result.success is True
    assert result.response == NO_FORECAST
    assert "None" not in result.response


def test_handle_weather_current_none_state_is_honest_fallback(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.get_entity_state_cached = MagicMock(return_value=None)

    import asyncio
    result = asyncio.run(d._handle_weather("qué tiempo hace", Priority.HIGH))

    assert result.success is True
    assert result.response == NO_DATA
    assert "None" not in result.response


@pytest.mark.parametrize("malformed_state", [
    {},
    {"state": "unavailable"},
    # No "state" key at all, attributes present but useless: condition
    # normalizes to "" (not a CONDITIONS hit) and there's no temperature/
    # humidity to fall back on either, so this still degrades to NO_DATA.
    {"attributes": {"friendly_name": "Clima"}},
])
def test_handle_weather_current_malformed_state_is_honest_fallback(
    dispatcher_with_async_ha, malformed_state
):
    d = dispatcher_with_async_ha
    d.ha.get_entity_state_cached = MagicMock(return_value=malformed_state)

    import asyncio
    result = asyncio.run(d._handle_weather("qué tiempo hace", Priority.HIGH))

    assert result.success is True
    assert result.response == NO_DATA
    assert "None" not in result.response


async def test_dispatch_wires_fast_weather_end_to_end_with_missing_data(dispatcher_with_async_ha):
    """Full dispatch(), not just _classify_request + _handle_weather in isolation."""
    d = dispatcher_with_async_ha
    d.ha.get_entity_state_cached = MagicMock(return_value=None)

    result = await d.dispatch(user_id="u1", text="qué tiempo hace", zone_id="zone_test")

    assert result.path == PathType.FAST_WEATHER
    assert result.success is True
    assert result.response == NO_DATA
