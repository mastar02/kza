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
    #
    # Los ÚNICOS casos de este parámetro que dependen realmente del guard
    # (review 2026-08-06): desactivándolo (`climate_command_adjacent = False`)
    # los 53 tests del archivo seguían verdes. Los demás casos no lo tocan —
    # los "fast_domotics" porque WEATHER_KEYWORDS no tiene ninguna frase que
    # los capture, y los "fast_weather" porque son preguntas (`is_question`
    # ya los neutraliza) o no matchean la regex de adyacencia.
    #
    # Estas dos SÍ: cláusula de justificación pospuesta que contiene una
    # frase LITERAL de WEATHER_KEYWORDS ("hace calor" / "hace frío"). Con el
    # guard activo el verbo adyacente al sustantivo climático manda y rutea
    # fast_domotics; sin guard, el keyword de clima gana y la acción pedida
    # NO se ejecuta. Si alguien apaga el guard, estas dos se ponen rojas.
    ("prendé el aire, hace calor", PathType.FAST_DOMOTICS),
    ("prendé la calefacción, hace frío", PathType.FAST_DOMOTICS),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Misroute conocido y DELIBERADAMENTE reabierto (commit a513108 + su "
        "revert ef651e2, 2026-08-04). El guard de adyacencia exige el verbo "
        "INMEDIATAMENTE antes del sustantivo climático; un adverbio o un "
        "cuantificador de por medio ('prendé YA el clima', 'subí UN GRADO el "
        "aire') lo rompe y gana el keyword de clima -> fast_weather. "
        "Mitigación vigente: hoy NO hay ninguna entidad climate indexada en "
        "ChromaDB, así que el fast_domotics 'correcto' tampoco ejecutaría "
        "nada; el costo real es una respuesta de clima en vez de un 'no "
        "encontré esa entidad'. Este xfail existe para que la limitación "
        "viva en la suite y no solo en un mensaje de commit: EN CUANTO se "
        "indexe una entidad climate, esto pasa a ser un bug con consecuencia "
        "y hay que arreglar el guard. strict=True -> si alguien lo arregla, "
        "el XPASS obliga a borrar este marcador."
    ),
)
@pytest.mark.parametrize("text", [
    "prendé ya el clima que hace calor",
    "apagá ahora el termostato, hace frío",
    "prendé de una vez el clima, hace calor",
    "subí un grado el aire porque hace frío",
])
def test_climate_commands_with_interposed_adverb_misroute_to_weather(dispatcher, text):
    """Comandos de AC con una palabra entre el verbo y el sustantivo.

    Deberían ser fast_domotics (son órdenes); hoy rutean fast_weather.
    """
    path, _ = dispatcher._classify_request(text.lower())
    assert path == PathType.FAST_DOMOTICS


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


# ---------------------------------------------------------------------------
# Review 2026-08-06, bloqueante 1: el caso que la batería de arriba NO cubría.
# `.get(clave, {})` solo usa el default cuando la CLAVE FALTA — no cuando la
# clave existe con valor None. HA puede responder 200 con
# {"service_response": null} (el servicio corrió y no tiene nada que
# devolver): ahí el `.get("service_response", {})` devuelve None y el `.get`
# encadenado siguiente levanta AttributeError, que atraviesa cinco capas sin
# que nadie la atrape -> TURNO DE VOZ MUDO.
#
# Mutación que estos tests deben atrapar: volver `or {}` a `.get(k, {})`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("null_valued_payload", [
    {"service_response": None},
    {"service_response": {"weather.forecast_home": None}},
    {"service_response": {"weather.forecast_home": {"forecast": None}}},
])
async def test_handle_weather_null_valued_keys_do_not_raise(
    dispatcher_with_async_ha, null_valued_payload
):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value=null_valued_payload)

    result = await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)

    assert result.response == NO_FORECAST
    assert "None" not in result.response


@pytest.mark.parametrize("forecast", [
    [None, None],
    [{}, "mañana soleado"],
    [{}, 42],
    [{}, {"condition": "rainy", "temperature": "unknown", "templow": "unknown"}],
])
async def test_handle_weather_non_dict_forecast_items_do_not_raise(
    dispatcher_with_async_ha, forecast
):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(
        return_value={"service_response": {"weather.forecast_home": {"forecast": forecast}}}
    )

    result = await d._handle_weather("va a llover mañana", Priority.HIGH)

    assert isinstance(result.response, str) and result.response
    assert "None" not in result.response


async def test_handle_weather_never_propagates_an_exception(dispatcher_with_async_ha):
    """Red de seguridad final: aunque el cliente de HA explote, el turno habla.

    Un turno mudo es peor que un 'no tengo el dato': el usuario no distingue
    'falló' de 'no me escuchó'. Debe ponerse rojo si se saca el try/except de
    `_handle_weather`.
    """
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(side_effect=RuntimeError("boom"))

    result = await d._handle_weather("va a llover mañana", Priority.HIGH)

    assert result.path == PathType.FAST_WEATHER
    assert result.response == NO_DATA
    assert result.success is False  # el fallo se reporta, no se disfraza


async def test_handle_weather_forecast_uses_an_explicit_per_request_timeout(
    dispatcher_with_async_ha
):
    """El POST del pronóstico no puede heredar `home_assistant.timeout`.

    Ese valor se toca por razones ajenas al clima; el techo de esta rama es
    propio (WEATHER_FORECAST_TIMEOUT_S). Rojo si se borra el kwarg.
    """
    from src.orchestrator.dispatcher import WEATHER_FORECAST_TIMEOUT_S

    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value=None)

    await d._handle_weather("va a llover mañana", Priority.HIGH)

    assert d.ha.call_service_with_response.await_args.kwargs["timeout"] == (
        WEATHER_FORECAST_TIMEOUT_S
    )


# ---------------------------------------------------------------------------
# Review 2026-08-06, bloqueante 3: `weather_entity` no llegaba desde la config.
# MultiUserOrchestrator es la ÚNICA construcción de producción del dispatcher
# y no reenviaba el kwarg -> en producción siempre ganaba el literal por
# defecto. Si la entidad de HA se llama distinto, el asistente contesta "No
# tengo el dato del clima" para siempre, indistinguible de un sensor caído.
#
# Mutaciones que estos tests deben atrapar:
#   A) borrar `weather_entity=weather_entity` del RequestDispatcher(...)
#      interno de MultiUserOrchestrator.__init__
#   B) hardcodear la entidad de vuelta en _handle_weather
# ---------------------------------------------------------------------------

def test_orchestrator_forwards_weather_entity_to_dispatcher_via_real_init():
    from src.orchestrator.dispatcher import MultiUserOrchestrator

    orchestrator = MultiUserOrchestrator(
        chroma_sync=MagicMock(),
        ha_client=MagicMock(),
        routine_manager=MagicMock(),
        weather_entity="weather.casa_de_prueba",
    )
    assert orchestrator.dispatcher.weather_entity == "weather.casa_de_prueba"


def test_dispatcher_default_weather_entity_matches_the_shipped_config():
    """El default vivo debe ser el mismo valor que trae config/settings.yaml.

    Un default que diverja del YAML es exactamente el fallo silencioso que
    este bloqueante describe.
    """
    import yaml

    from src.orchestrator.dispatcher import RequestDispatcher

    d = RequestDispatcher(
        chroma_sync=MagicMock(), ha_client=MagicMock(), routine_manager=MagicMock()
    )
    with open("config/settings.yaml") as fh:
        configured = yaml.safe_load(fh)["home_assistant"]["weather_entity"]

    assert d.weather_entity == configured


async def test_configured_weather_entity_is_the_one_queried(dispatcher_with_async_ha):
    """La entidad configurada llega hasta la llamada a HA, en las dos ramas."""
    d = dispatcher_with_async_ha
    d.weather_entity = "weather.casa_de_prueba"
    d.ha.get_entity_state_cached = MagicMock(return_value=None)
    d.ha.call_service_with_response = AsyncMock(return_value=None)

    await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert d.ha.get_entity_state_cached.call_args.args[0] == "weather.casa_de_prueba"

    await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)
    assert d.ha.call_service_with_response.await_args.args[2] == "weather.casa_de_prueba"


async def test_forecast_is_read_from_the_configured_entity_key(dispatcher_with_async_ha):
    """El payload de HA viene indexado por entity_id: leer la clave correcta.

    Rojo si `_handle_weather` vuelve a indexar por el literal por defecto.
    """
    d = dispatcher_with_async_ha
    d.weather_entity = "weather.casa_de_prueba"
    d.ha.call_service_with_response = AsyncMock(return_value={
        "service_response": {
            "weather.casa_de_prueba": {
                "forecast": [{}, {"condition": "rainy", "temperature": 13}]
            },
            "weather.forecast_home": {"forecast": [{}, {"condition": "sunny"}]},
        }
    })

    result = await d._handle_weather("va a llover mañana", Priority.HIGH)

    assert "lluvioso" in result.response.lower()
    assert "soleado" not in result.response.lower()
