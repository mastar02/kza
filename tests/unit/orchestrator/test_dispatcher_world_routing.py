"""Weather routing, and the thermostat collision it must not cause."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.observation import ObservationReason, ObservationState
from src.core.task_outcome import TaskStatus
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import PathType, Priority, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.world.weather import (DEFAULT_ENTITY, INDOOR_TEMPERATURE_UNSUPPORTED, NO_DATA, NO_FORECAST,
                               NO_FORECAST_HORIZON, ONE_DAY_AT_A_TIME, HorizonKind, WeatherHorizon,
                               parse_weather_horizon)
from src.home_assistant.ha_client import HAStatesSnapshot


def weather_snapshot(state, entity_id=DEFAULT_ENTITY):
    """A synthetic, typed read with explicit time; no unclocked cache fallback."""
    now = datetime.now(UTC)
    if state is None:
        values = ()
    elif type(state) is not dict or type(state.get('state')) is not str or type(state.get('attributes', {})) is not dict:
        return HAStatesSnapshot(False, None, 'ha_states_invalid', attempted_at=now)
    else:
        values = ({**state, 'entity_id': state.get('entity_id', entity_id)},)
    return HAStatesSnapshot(True, now, None, values, now+timedelta(seconds=6), now)


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


@pytest.mark.parametrize("text", [
    # Verificados contra main en la review del PR #14 (2026-08-06): todos
    # ruteaban FAST_WEATHER — el usuario escuchaba el pronóstico y la
    # acción pedida nunca se ejecutaba (fallo domótico silencioso). El
    # discriminante: la cláusula de clima es JUSTIFICACIÓN del comando,
    # no consulta. Regla: un fragmento de WEATHER_CLAUSE_FRAGMENTS solo
    # rutea a clima si no hay ningún verbo domótico en el texto (o si es
    # pregunta).
    "prendé la luz que hace frío",
    "apagá la luz, hace calor",
    "prendé el ventilador que hace calor",
    "prendé la estufa, hace frío",
    "cerrá la persiana que llueve",
    "cerrá las ventanas que está lloviendo",
    "prendé la luz del living porque va a llover",
    "encender la luz, hace frío",
    "apagá todo, hace calor",
])
def test_non_climate_commands_with_weather_clause_stay_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_DOMOTICS


def test_temperature_of_day_keywords_are_weather_keywords():
    # T037-U2b: the setting-verb guard only filters keywords that exist.
    assert RequestDispatcher._TEMPERATURE_OF_DAY_KEYWORDS <= set(RequestDispatcher.WEATHER_KEYWORDS)


def test_weather_clause_fragments_is_a_subset_of_weather_keywords():
    # El tier de fragmentos no puede inventar keywords: si alguien saca un
    # fragmento de WEATHER_KEYWORDS y olvida el frozenset, esto lo detecta.
    assert RequestDispatcher.WEATHER_CLAUSE_FRAGMENTS <= set(
        RequestDispatcher.WEATHER_KEYWORDS
    )


@pytest.mark.parametrize("text", [
    # Finding I1 (review PR #15): el veto de WEATHER_CLAUSE_FRAGMENTS asume
    # que "fragmento + verbo domótico sin '?'" es un comando con una
    # cláusula de justificación colgada. Esa asunción se rompe cuando el
    # STT no transcribió la puntuación de una pregunta/observación real —
    # estos tres son los ejemplos del finding, uno por marcador de
    # _NON_COMMAND_HINTS. La tercera es la clase más dura: negación como
    # cláusula de NECESIDAD (no de comando) — el mismo fallo que canceló
    # el ruteo de clima por modelo (NO-GO 2026-08-04). Antes del fix los
    # tres caían al loop de DOMOTICS_KEYWORDS y ejecutaban una acción
    # fantasma nunca pedida.
    "puedo abrir las ventanas o va a llover",
    "tengo que prender el clima o hace calor afuera",
    "no hace falta prender nada, hace calor",
])
def test_non_command_hints_exempt_the_clause_fragment_veto(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_WEATHER


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


@pytest.mark.parametrize("text", [
    "prendé ya el clima que hace calor",
    "apagá ahora el termostato, hace frío",
    "prendé de una vez el clima, hace calor",
    "subí un grado el aire porque hace frío",
])
def test_climate_commands_with_interposed_adverb_route_to_domotics(dispatcher, text):
    """Comandos de AC con una palabra entre el verbo y el sustantivo.

    Misroute conocido desde 2026-08-04 (commit a513108 + revert ef651e2):
    el guard de adyacencia no los cubría y ganaba el keyword de clima.
    Cerrado por el veto de WEATHER_CLAUSE_FRAGMENTS (review PR #14,
    2026-08-06): todos estos casos contienen un fragmento de cláusula +
    un verbo domótico, así que el fragmento ya no captura y el loop de
    DOMOTICS_KEYWORDS resuelve.
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
    d.ha.get_state_snapshot.return_value = weather_snapshot(None)
    return d


async def test_handle_weather_forecast_none_response_is_honest_fallback(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value=None)

    result = await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)

    assert result.success is True
    assert result.response == NO_FORECAST
    assert "None" not in result.response
    # I5(a), review PR #15: la otra mitad del contador (rama NO_FORECAST de
    # "mañana"/"pasado mañana") no tenía cobertura — solo la rama NO_DATA de
    # "hoy" la ejercitaba (test_no_data_increments_stat_and_warns).
    assert d._stats["weather_no_data"] == 1


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
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(None))

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
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(malformed_state))

    import asyncio
    result = asyncio.run(d._handle_weather("qué tiempo hace", Priority.HIGH))

    assert result.success is True
    assert result.response == NO_DATA
    assert "None" not in result.response


async def test_dispatch_wires_fast_weather_end_to_end_with_missing_data(dispatcher_with_async_ha):
    """Full dispatch(), not just _classify_request + _handle_weather in isolation."""
    d = dispatcher_with_async_ha
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(None))

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
    # I5(b), review PR #15: weather_no_data cuenta "habló la disculpa con
    # éxito" (success=True), no "cualquier respuesta genérica". La rama de
    # excepción es un fallo real (success=False) y NO debe sumar al mismo
    # contador que un weather_entity mal configurado — mezclarlos haría
    # invisible cuál de los dos problemas está pasando.
    assert d._stats["weather_no_data"] == 0


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
    """Extract the exact configured entity from a typed read, with no cache fallback."""
    d = dispatcher_with_async_ha
    d.weather_entity = "weather.casa_de_prueba"
    d.ha.get_entity_state_cached = MagicMock(side_effect=AssertionError("unclocked cache used"))
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(
        {"state": "sunny", "attributes": {"temperature": 23}}, d.weather_entity))
    d.ha.call_service_with_response = AsyncMock(return_value=None)
    result = await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert "23 grados" in result.response
    d.ha.get_state_snapshot.assert_awaited_once()
    d.ha.get_entity_state_cached.assert_not_called()
    await d._handle_weather("qué tiempo hace mañana", Priority.HIGH)
    assert d.ha.call_service_with_response.await_args.args[2] == "weather.casa_de_prueba"


# T037-U1: el reloj del turno es inyectado (home_capabilities.now), nunca
# datetime.now(); 02:30Z del 5 son las 23:30 del 4 en Montevideo, así que
# "mañana" es el 5 aunque en UTC ya sea el 5.
FROZEN = datetime(2026, 8, 5, 2, 30, tzinfo=UTC)


def _entry(iso, condition, high, low=None):
    entry = {"datetime": iso, "condition": condition, "temperature": high}
    if low is not None:
        entry["templow"] = low
    return entry


def _forecast_payload(forecast, entity="weather.forecast_home"):
    return {"service_response": {entity: {"forecast": forecast}}}


def _frozen(d, now=FROZEN):
    d.home_capabilities = SimpleNamespace(now=lambda: now)
    return d


async def test_forecast_is_read_from_the_configured_entity_key(dispatcher_with_async_ha):
    """El payload de HA viene indexado por entity_id: leer la clave correcta.

    Rojo si `_handle_weather` vuelve a indexar por el literal por defecto.
    """
    d = _frozen(dispatcher_with_async_ha)
    d.weather_entity = "weather.casa_de_prueba"
    d.ha.call_service_with_response = AsyncMock(return_value={
        "service_response": {
            "weather.casa_de_prueba": {
                "forecast": [_entry("2026-08-05T15:00:00+00:00", "rainy", 13)]
            },
            "weather.forecast_home": {"forecast": [_entry("2026-08-05T15:00:00+00:00", "sunny", 20)]},
        }
    })

    result = await d._handle_weather("va a llover mañana", Priority.HIGH)

    assert "lluvioso" in result.response.lower()
    assert "soleado" not in result.response.lower()


async def test_pasado_manana_requests_day_after_tomorrow(dispatcher_with_async_ha):
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload([
        _entry("2026-08-06T15:00:00+00:00", "cloudy", 15, 7),
        _entry("2026-08-04T15:00:00+00:00", "sunny", 20, 10),
        _entry("2026-08-05T15:00:00+00:00", "rainy", 18, 9),
    ]))
    result = await d._handle_weather("qué tiempo hace pasado mañana", Priority.HIGH)
    assert result.response == "Pasado mañana: nublado, entre 7 y 15 grados."
    assert result.outcome.status is TaskStatus.COMPLETED and result.outcome.task_completed


@pytest.mark.parametrize(("now", "expected"), [
    # 23:30 del 4 en Montevideo: mañana es el 5 (por UTC sería el 6).
    (FROZEN, "Mañana: lluvioso, 18 grados."),
    # 03:30Z del 5 = 00:30 del 5 local: mañana es el 6.
    (datetime(2026, 8, 5, 3, 30, tzinfo=UTC), "Mañana: nublado, 15 grados."),
])
async def test_forecast_day_uses_the_injected_clock_in_household_time(dispatcher_with_async_ha, now, expected):
    d = _frozen(dispatcher_with_async_ha, now)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload([
        _entry("2026-08-06T15:00:00+00:00", "cloudy", 15),
        _entry("2026-08-05T15:00:00+00:00", "rainy", 18),
        _entry("2026-08-04T15:00:00+00:00", "sunny", 20),
    ]))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == expected
    assert result.observation.observed_at == now and result.observation.attempted_at == now


async def test_aligned_forecast_observes_date_alignment_but_never_issue_time(dispatcher_with_async_ha):
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload([
        _entry("2026-08-04T15:00:00+00:00", "sunny", 20),
        _entry("2026-08-05T15:00:00+00:00", "rainy", 18),
    ]))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == "Mañana: lluvioso, 18 grados."
    assert result.observation.unobserved_fields == ("issued_at",)
    assert result.observation.state is ObservationState.PARTIAL
    assert result.outcome.status is TaskStatus.COMPLETED
    assert d._stats["weather_no_data"] == 0
    d.ha.call_service_with_response.assert_awaited_once()


# PQ-T037-1 (user answer "Nombra el día"): a missing or conflicting day names
# the requested day; an empty payload keeps the generic NO_FORECAST.
NO_FORECAST_TOMORROW = "No tengo el pronóstico para mañana."


@pytest.mark.parametrize(("forecast", "spoken"), [
    # Falta el día pedido (por posición, el segundo sería "mañana").
    ([_entry("2026-08-04T15:00:00+00:00", "sunny", 20), _entry("2026-08-06T15:00:00+00:00", "rainy", 18)],
     NO_FORECAST_TOMORROW),
    # Primer elemento de ayer; el segundo es hoy, no mañana.
    ([_entry("2026-08-03T15:00:00+00:00", "sunny", 20), _entry("2026-08-04T15:00:00+00:00", "rainy", 18)],
     NO_FORECAST_TOMORROW),
    # Duplicados del mismo día local que se dirían distinto.
    ([_entry("2026-08-05T12:00:00+00:00", "sunny", 20), _entry("2026-08-05T18:00:00+00:00", "rainy", 18)],
     NO_FORECAST_TOMORROW),
    # Misma condición, distinta temperatura: también es conflicto (R1, review 1).
    ([_entry("2026-08-05T15:00:00+00:00", "rainy", 13), _entry("2026-08-05T18:00:00+00:00", "rainy", 20)],
     NO_FORECAST_TOMORROW),
    ([], NO_FORECAST),
])
async def test_unaligned_forecast_is_honest_and_not_completed(dispatcher_with_async_ha, forecast, spoken):
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(forecast))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == spoken
    assert result.outcome.status is TaskStatus.FAILED and not result.outcome.task_completed
    assert result.observation.unobserved_fields == ("issued_at", "forecast_date_alignment")
    assert result.observation.observed_at == FROZEN
    assert d._stats["weather_no_data"] == 1
    d.ha.call_service_with_response.assert_awaited_once()


@pytest.mark.parametrize("bad_datetime", ["2026-08-05T15:00:00", "mañana", None, 1754406000])
async def test_unusable_forecast_dates_are_an_invalid_observation(dispatcher_with_async_ha, bad_datetime):
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload([
        _entry("2026-08-04T15:00:00+00:00", "sunny", 20),
        {"datetime": bad_datetime, "condition": "rainy", "temperature": 18},
    ]))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == NO_FORECAST
    assert result.outcome.status is TaskStatus.FAILED
    assert result.observation.state is ObservationState.UNAVAILABLE
    assert result.observation.reason is ObservationReason.INVALID
    assert result.observation.observed_at is None


async def test_no_data_increments_stat_and_warns(dispatcher_with_async_ha, caplog):
    d = dispatcher_with_async_ha
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(None))
    with caplog.at_level("WARNING"):
        result = await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert result.success is True          # el degradado honesto NO es fallo
    assert result.response == NO_DATA
    assert d._stats["weather_no_data"] == 1
    assert any("weather_entity" in r.message for r in caplog.records)


@pytest.mark.parametrize(("forecast", "alignment", "spoken"), [
    ([_entry("2026-08-04T15:00:00+00:00", "sunny", 20)], "missing_day", NO_FORECAST_TOMORROW),
    ([_entry("2026-08-05T15:00:00+00:00", "rainy", 13), _entry("2026-08-05T18:00:00+00:00", "rainy", 20)],
     "conflicting_day", NO_FORECAST_TOMORROW),
    ([_entry("2026-08-05T15:00:00", "rainy", 13)], "invalid_entries", NO_FORECAST),
])
async def test_unaligned_forecast_warning_names_the_alignment_not_the_entity(
    dispatcher_with_async_ha, caplog, forecast, alignment, spoken
):
    """R2, review 1: la entidad respondió; culpar a weather_entity despista."""
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(forecast))
    with caplog.at_level("WARNING"):
        result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == spoken
    warnings = [r.getMessage() for r in caplog.records if "answered honestly with no data" in r.getMessage()]
    assert len(warnings) == 1
    assert f"alignment={alignment}" in warnings[0]
    assert "datetime convention" in warnings[0]
    assert "weather_entity exists" not in warnings[0]


async def test_forecast_without_a_payload_still_points_at_the_entity(dispatcher_with_async_ha, caplog):
    d = _frozen(dispatcher_with_async_ha)
    d.ha.call_service_with_response = AsyncMock(return_value={"service_response": {}})
    with caplog.at_level("WARNING"):
        result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == NO_FORECAST
    warnings = [r.getMessage() for r in caplog.records if "answered honestly with no data" in r.getMessage()]
    assert len(warnings) == 1
    assert "alignment=None" in warnings[0] and "weather_entity exists" in warnings[0]


async def test_data_present_does_not_touch_the_no_data_stat(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot({
        "state": "sunny", "attributes": {"temperature": 22.0},
    }))
    result = await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert "22 grados" in result.response
    assert d._stats["weather_no_data"] == 0


async def test_no_data_warning_is_rate_limited_but_stat_is_not(
    dispatcher_with_async_ha, caplog
):
    """M15, review PR #15: fija la ventana del rate-limit y el init en -inf.

    Dos requests consecutivos sin dato deben sumar 2 al contador (cada
    request sin dato ES un evento real) pero loguear el warning una sola
    vez (segunda llamada dentro de _WEATHER_NODATA_WARN_INTERVAL_S). Si el
    init de `_last_weather_nodata_warn` dejara de ser `float("-inf")`, la
    PRIMERA llamada silenciosamente podría no loguear tampoco.
    """
    d = dispatcher_with_async_ha
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(None))

    with caplog.at_level("WARNING"):
        r1 = await d._handle_weather("qué tiempo hace", Priority.HIGH)
        r2 = await d._handle_weather("qué tiempo hace", Priority.HIGH)

    assert r1.success is True and r2.success is True
    assert d._stats["weather_no_data"] == 2
    warnings = [r for r in caplog.records if "answered honestly with no data" in r.message]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# T037-U2: weather horizon and intent fixes (user answers Q23, PQ-T037-1).
# FROZEN is Tuesday 4 August 2026, 23:30 in Montevideo.
# ---------------------------------------------------------------------------

WEEK = [
    _entry("2026-08-04T15:00:00+00:00", "sunny", 20, 10),     # martes (hoy)
    _entry("2026-08-05T15:00:00+00:00", "rainy", 18, 9),      # miércoles (mañana)
    _entry("2026-08-06T15:00:00+00:00", "cloudy", 15, 7),     # jueves (pasado mañana)
    _entry("2026-08-08T15:00:00+00:00", "pouring", 12, 6),    # sábado
]


def _no_current_read(d):
    d.ha.get_state_snapshot = AsyncMock(side_effect=AssertionError("current conditions read"))
    return d


@pytest.mark.parametrize(("text", "spoken"), [
    ("¿va a llover hoy?", "Hoy: soleado, entre 10 y 20 grados."),
    ("va a llover esta mañana", "Hoy: soleado, entre 10 y 20 grados."),
    ("va a llover por la mañana", "Hoy: soleado, entre 10 y 20 grados."),
    ("va a llover a la mañana", "Hoy: soleado, entre 10 y 20 grados."),
    ("va a llover", "Hoy: soleado, entre 10 y 20 grados."),
    ("cuál es el pronóstico", "Hoy: soleado, entre 10 y 20 grados."),
    ("va a llover mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("va a llover mañana por la mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("qué tiempo hace pasado mañana", "Pasado mañana: nublado, entre 7 y 15 grados."),
    ("¿va a llover el sábado?", "El sábado: lloviendo fuerte, entre 6 y 12 grados."),
    ("qué tiempo va a hacer el miércoles", "El miércoles: lluvioso, entre 9 y 18 grados."),
    ("va a llover el martes", "Hoy: soleado, entre 10 y 20 grados."),
])
async def test_forecast_horizons_select_the_day_by_household_date(dispatcher_with_async_ha, text, spoken):
    d = _no_current_read(_frozen(dispatcher_with_async_ha))
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(list(reversed(WEEK))))
    assert d._classify_request(text.lower())[0] is PathType.FAST_WEATHER
    result = await d._handle_weather(text, Priority.HIGH)
    assert result.response == spoken
    assert result.outcome.status is TaskStatus.COMPLETED
    assert result.observation.source.value == "ha-weather-forecast"
    d.ha.call_service_with_response.assert_awaited_once()


@pytest.mark.parametrize(("text", "spoken"), [
    # The week has no Monday or next Tuesday: name the day, never the current conditions.
    ("va a llover el lunes", "No tengo el pronóstico para el lunes."),
    ("va a llover el martes que viene", "No tengo el pronóstico para el martes que viene."),
    ("va a llover el viernes", "No tengo el pronóstico para el viernes."),
])
async def test_weekday_outside_the_forecast_is_named_and_never_current(dispatcher_with_async_ha, text, spoken):
    d = _no_current_read(_frozen(dispatcher_with_async_ha))
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(WEEK))
    result = await d._handle_weather(text, Priority.HIGH)
    assert result.response == spoken
    assert result.outcome.status is TaskStatus.FAILED and not result.outcome.task_completed
    assert result.observation.unobserved_fields == ("issued_at", "forecast_date_alignment")
    assert d._stats["weather_no_data"] == 1


@pytest.mark.parametrize(("text", "spoken"), [
    ("qué temperatura va a hacer mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("cuántos grados va a hacer el sábado", "El sábado: lloviendo fuerte, entre 6 y 12 grados."),
    ("cuál es la temperatura de mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("qué temperatura va a hacer", "Hoy: soleado, entre 10 y 20 grados."),
    # T037-U2b (FV-L2).
    ("qué temperatura hará mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("cuántos grados van a hacer mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("cuál va a ser la temperatura mañana", "Mañana: lluvioso, entre 9 y 18 grados."),
    ("qué temperatura habrá el sábado", "El sábado: lloviendo fuerte, entre 6 y 12 grados."),
])
async def test_future_temperature_questions_answer_the_forecast_day(dispatcher_with_async_ha, text, spoken):
    """Review T037-U2-RV2: routed to FAST_WEATHER and answered by date, never the reasoner."""
    d = _no_current_read(_frozen(dispatcher_with_async_ha))
    d.router, d.llm = MagicMock(), MagicMock()
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(WEEK))
    # Classified first: a regression to SLOW_LLM fails here instead of waiting on the mocked reasoner.
    assert d._classify_request(text.lower())[0] is PathType.FAST_WEATHER
    result = await d.dispatch(user_id="u1", text=text, zone_id="zone_test")
    assert result.path is PathType.FAST_WEATHER and result.intent == "weather"
    assert result.response == spoken
    assert result.outcome.status is TaskStatus.COMPLETED
    assert d.router.mock_calls == [] and d.llm.mock_calls == []


async def test_today_forecast_missing_is_named_not_replaced_by_current(dispatcher_with_async_ha):
    d = _no_current_read(_frozen(dispatcher_with_async_ha))
    d.ha.call_service_with_response = AsyncMock(return_value=_forecast_payload(WEEK[1:]))
    result = await d._handle_weather("¿va a llover hoy?", Priority.HIGH)
    assert result.response == "No tengo el pronóstico para hoy."
    assert result.outcome.status is TaskStatus.FAILED


@pytest.mark.parametrize("text", [
    "qué tiempo hace", "qué tiempo hace hoy", "qué tiempo hace esta mañana", "llueve por la mañana",
    "cómo está el tiempo",
])
async def test_present_tense_about_today_reads_current_conditions(dispatcher_with_async_ha, text):
    # Current conditions need no forecast date: the typed snapshot carries its own clock.
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(side_effect=AssertionError("forecast read"))
    d.ha.get_state_snapshot = AsyncMock(return_value=weather_snapshot(
        {"state": "cloudy", "attributes": {"temperature": 17}}))
    assert d._classify_request(text.lower())[0] is PathType.FAST_WEATHER
    result = await d._handle_weather(text, Priority.HIGH)
    assert result.response == "Hay 17 grados, nublado."
    d.ha.call_service_with_response.assert_not_awaited()


@pytest.mark.parametrize(("text", "spoken", "status"), [
    ("va a llover la semana que viene", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("qué tiempo va a hacer el fin de semana", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("el pronóstico para el 25", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("va a llover en tres días", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("va a llover hoy o mañana", ONE_DAY_AT_A_TIME, TaskStatus.NEEDS_INPUT),
    # Review T037-U2-RV1: a past day is refused, never the coming weekday or the current conditions.
    ("estuvo lloviendo el sábado pasado", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("cuál fue el pronóstico del sábado pasado", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("estuvo lloviendo el mes pasado", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("qué tiempo va a hacer el mes pasado", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("estuvo lloviendo la semana pasada", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    # Review T037-U2-RV3: a weekday with a day of the month, or an ordinal, is a date.
    ("va a llover el lunes 17", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("va a llover el 1ro", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    # Fix verification T037-U2-FV2.
    ("va a llover el día 25", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
    ("estuvo lloviendo el pasado sábado", NO_FORECAST_HORIZON, TaskStatus.UNSUPPORTED),
])
async def test_unsupported_horizons_are_refused_without_any_read(dispatcher_with_async_ha, text, spoken, status):
    d = _no_current_read(_frozen(dispatcher_with_async_ha))
    d.ha.call_service_with_response = AsyncMock(side_effect=AssertionError("forecast read"))
    result = await d.dispatch(user_id="u1", text=text, zone_id="zone_test")
    assert result.path is PathType.FAST_WEATHER and result.intent == "weather"
    assert result.response == spoken and not result.success
    assert result.outcome.status is status and not result.outcome.task_completed
    assert result.observation is None
    d.ha.call_service_with_response.assert_not_awaited()
    assert d._stats["weather_no_data"] == 0


@pytest.mark.parametrize("text", [
    "qué temperatura hay en el living",
    "¿qué temperatura hace en la cocina?",
    "cuántos grados hay en el escritorio",
    "qué temperatura tiene el living",
    "qué temperatura hay adentro",
    "cuál es la temperatura del cuarto",
])
def test_room_temperature_questions_leave_fast_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text.lower())
    assert path is PathType.FAST_HOUSEHOLD
    # An upstream turn_on/turn_off guess never turns the question into an action.
    path, _ = dispatcher._classify_request(text.lower(), service_filter="turn_on")
    assert path is PathType.FAST_HOUSEHOLD


async def test_room_temperature_question_is_refused_with_no_read_send_or_model(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.router, d.llm = MagicMock(), MagicMock()
    d.ha.get_state_snapshot = AsyncMock(side_effect=AssertionError("state read"))
    d.ha.call_service_with_response = AsyncMock(side_effect=AssertionError("forecast read"))
    d.ha.call_service = AsyncMock(side_effect=AssertionError("send"))
    result = await d.dispatch(user_id="u1", text="qué temperatura hay en el living", zone_id="zone_test")
    assert result.path is PathType.FAST_HOUSEHOLD and result.intent == "indoor_temperature"
    assert result.response == INDOOR_TEMPERATURE_UNSUPPORTED and not result.success
    assert result.outcome.status is TaskStatus.UNSUPPORTED
    d.ha.call_service.assert_not_awaited()
    assert d.router.mock_calls == [] and d.llm.mock_calls == []


def test_outside_of_a_room_is_the_outdoor_weather(dispatcher):
    # Review T037-U2-RV5: "afuera de la cocina" is outside, answered by the weather entity.
    assert dispatcher._classify_request("cuántos grados hay afuera de la cocina")[0] is PathType.FAST_WEATHER


def test_command_with_an_indoor_temperature_question_is_refused_with_no_effects(dispatcher):
    # Review T037-U2-RV5, accepted and documented: the question part wins over an
    # upstream turn_on/turn_off guess, so nothing is acted on; the refusal is honest.
    text = "subí el aire del living, qué temperatura hay en el living?"
    assert dispatcher._classify_request(text, service_filter="turn_on")[0] is PathType.FAST_HOUSEHOLD


def test_configured_room_names_count_for_indoor_temperature():
    from src.core.household_clock import RoomNamespace

    d = RequestDispatcher(chroma_sync=MagicMock(), ha_client=MagicMock(), routine_manager=MagicMock(),
                          room_namespace=RoomNamespace(("quincho",)))
    assert d._classify_request("qué temperatura hay en el quincho")[0] is PathType.FAST_HOUSEHOLD


@pytest.mark.parametrize("text", [
    "qué temperatura hace afuera",
    "cuántos grados hay afuera",
    "qué temperatura hace",
    "qué tiempo va a hacer mañana",
    "cómo va a estar el clima el sábado",
    "cómo va a estar el tiempo mañana",
    "cómo está el tiempo",
    "va a hacer frío mañana",
    "va a hacer calor el sábado",
    # Review T037-U2-RV2: future temperature questions, not the cloud reasoner.
    "qué temperatura va a hacer mañana",
    "cuántos grados va a hacer mañana",
    "cuántos grados va a hacer el sábado",
    "qué temperatura va a hacer el sábado",
    "cuál es la temperatura de mañana",
    "¿qué temperatura va a hacer pasado mañana?",
])
def test_weather_intent_fixes_route_to_fast_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path is PathType.FAST_WEATHER


@pytest.mark.parametrize("text", [
    "prendé la luz que va a hacer frío",
    "cerrá la persiana que va a hacer calor",
    "bajá la temperatura del living",
    "poné la temperatura del living en 22",
    # Review T037-U2-RV2: the new temperature keywords leave thermostat commands alone.
    "poné la temperatura en 22",
    "subí la temperatura a 23 grados",
    "bajá los grados del aire",
])
def test_new_weather_fragments_keep_commands_as_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path is PathType.FAST_DOMOTICS


@pytest.mark.parametrize("text", [
    # Review T037-U2-RV4: "tiempo" as a duration and "clima" as the HVAC are not weather.
    "cómo está el tiempo del temporizador",
    "¿cómo está el tiempo del temporizador?",
    "cuánto tiempo va a hacer falta",
    "¿a qué hora va a estar el clima prendido?",
    "va a estar el clima prendido toda la noche?",
])
def test_tiempo_and_clima_outside_a_weather_question_frame_are_not_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path is not PathType.FAST_WEATHER


@pytest.mark.parametrize("text", [
    "cómo está el tiempo de mañana",
    "cómo va a estar el tiempo del sábado",
    "cómo va a estar el tiempo del fin de semana",
    "como esta el tiempo hoy",
    "cómo está el tiempo de hoy",  # fix verification: pins "hoy" in the day allowlist
    "qué tiempo va a hacer",
    # Fix verification T037-U2-FV1: mixed accents, as STT or typing leaves them.
    "como está el tiempo",
    "cómo esta el tiempo",
    "¿como está el tiempo hoy?",
])
def test_tiempo_questions_about_a_day_stay_weather(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path is PathType.FAST_WEATHER


# ---------------------------------------------------------------------------
# T037-U2b: follow-up of the T037-U2 fix verification (FV-L1..L4). One table:
# the phrasings that change and the neighbours that must keep their path.
# The horizon column is checked for every FAST_WEATHER row; None elsewhere.
# FV-L1 rows (past tense with a plain weekday) are horizon-only: their path
# was already FAST_WEATHER through "lloviendo"/"pronóstico".
# ---------------------------------------------------------------------------

_SAT = WeatherHorizon(HorizonKind.FORECAST, weekday=5)


def _ahead(offset):
    return WeatherHorizon(HorizonKind.FORECAST, offset=offset)


@pytest.mark.parametrize(("text", "path", "horizon"), [
    # FV-L2: more future temperature phrasings reach FAST_WEATHER with the horizon rules.
    ("qué temperatura hará mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("cuántos grados van a hacer mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("qué temperatura va a haber mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("cuál va a ser la temperatura mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("cuál es la temperatura para mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("qué temperatura habrá el sábado", PathType.FAST_WEATHER, _SAT),
    ("¿qué temperatura habrá hoy?", PathType.FAST_WEATHER, _ahead(0)),
    ("cuántos grados hará pasado mañana", PathType.FAST_WEATHER, _ahead(2)),
    ("cuantos grados van a haber el domingo que viene", PathType.FAST_WEATHER,
     WeatherHorizon(HorizonKind.FORECAST, weekday=6, next_week=True)),
    # Neighbours: the indoor future question stays household state, never the outdoor entity.
    ("cuál va a ser la temperatura del living mañana", PathType.FAST_HOUSEHOLD, None),
    ("qué temperatura habrá en la cocina mañana", PathType.FAST_HOUSEHOLD, None),
    # Neighbours: existing future temperature questions keep their path and horizon.
    ("qué temperatura va a hacer mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("cuál es la temperatura de mañana", PathType.FAST_WEATHER, _ahead(1)),
    ("cuántos grados va a hacer el sábado", PathType.FAST_WEATHER, _SAT),
    ("fijate qué temperatura va a hacer mañana", PathType.FAST_WEATHER, _ahead(1)),
    # FV-L3: an imperative climate verb on "la temperatura de/para <día>" is a
    # command, never weather; it keeps the path it had before T037-U2.
    ("programá la temperatura de mañana en 20", PathType.SLOW_LLM, None),
    ("programa la temperatura de manana en 20", PathType.SLOW_LLM, None),
    ("ajustá la temperatura de mañana a 22", PathType.SLOW_LLM, None),
    ("dejá la temperatura de mañana en 20", PathType.SLOW_LLM, None),
    ("configurá la temperatura para mañana en 21", PathType.SLOW_LLM, None),
    ("¿podés programar la temperatura de mañana en 20?", PathType.SLOW_LLM, None),
    ("programame la temperatura de mañana en 19", PathType.SLOW_LLM, None),
    # Neighbours: thermostat verbs the domotics guard already owns.
    ("poné la temperatura de mañana en 20", PathType.FAST_DOMOTICS, None),
    ("poné la temperatura en 22", PathType.FAST_DOMOTICS, None),
    ("bajá la temperatura del living", PathType.FAST_DOMOTICS, None),
    # FV-L4: "el tiempo de afuera" and "el tiempo del día de hoy" are the weather again.
    ("cómo está el tiempo de afuera", PathType.FAST_WEATHER, WeatherHorizon(HorizonKind.CURRENT)),
    ("cómo está el tiempo del día de hoy", PathType.FAST_WEATHER, WeatherHorizon(HorizonKind.CURRENT)),
    ("¿cómo va a estar el tiempo del día de mañana?", PathType.FAST_WEATHER, _ahead(1)),
    # Neighbours: "tiempo" as a duration stays out of the weather path.
    ("cómo está el tiempo del temporizador", PathType.SLOW_LLM, None),
    ("cómo está el tiempo de cocción", PathType.SLOW_LLM, None),
    ("cuánto tiempo va a hacer falta", PathType.SLOW_LLM, None),
    # FV-L1: past tense with a plain weekday is refused, never the coming day.
    # Neighbours: future or imperfect forms keep the coming weekday; no day keeps today.
    ("va a llover el sábado", PathType.FAST_WEATHER, _SAT),
    ("cuál es el pronóstico del sábado", PathType.FAST_WEATHER, _SAT),
    ("cuál era el pronóstico del sábado", PathType.FAST_WEATHER, _SAT),
    ("estuvo lloviendo", PathType.FAST_WEATHER, WeatherHorizon(HorizonKind.CURRENT)),
    ("cuál fue el pronóstico del sábado que viene", PathType.FAST_WEATHER,
     WeatherHorizon(HorizonKind.FORECAST, weekday=5, next_week=True)),
    ("estuvo lloviendo el sábado pasado", PathType.FAST_WEATHER, WeatherHorizon(HorizonKind.UNSUPPORTED)),
])
def test_t037_u2b_follow_up_routing_table(dispatcher, text, path, horizon):
    assert dispatcher._classify_request(text.lower())[0] is path
    if path is PathType.FAST_WEATHER:
        assert parse_weather_horizon(text) == horizon
