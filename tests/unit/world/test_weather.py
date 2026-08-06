from src.world.weather import describe_current


def _state(condition="partlycloudy", temp=12.7, **attrs):
    return {
        "entity_id": "weather.forecast_home",
        "state": condition,
        "attributes": {"temperature": temp, "temperature_unit": "°C", **attrs},
    }


def test_describes_temperature_and_condition_in_spanish():
    # temp=12.7 rounds to 13 for speech (see test_rounds_temperature_for_speech).
    out = describe_current(_state())
    assert "13" in out
    assert "grados" in out.lower()
    assert "nublado" in out.lower()


def test_rounds_temperature_for_speech():
    # 12.7 spoken as "13 grados": decimals are noise out loud.
    assert "13 grados" in describe_current(_state(temp=12.7))
    assert "12 grados" in describe_current(_state(temp=12.4))


def test_unknown_condition_falls_back_without_inventing():
    out = describe_current(_state(condition="hurricane_of_frogs"))
    assert "13 grados" in out
    assert "rana" not in out.lower()


def test_unavailable_entity_says_so():
    out = describe_current({"state": "unavailable", "attributes": {}})
    assert "no" in out.lower()
    assert "grados" not in out.lower()


def test_missing_entity_says_so():
    # HA restarted, entity not synced yet: never invent a number.
    out = describe_current(None)
    assert "no" in out.lower()
    assert "grados" not in out.lower()


def test_missing_temperature_attribute_does_not_crash():
    out = describe_current({"state": "cloudy", "attributes": {}})
    assert "nublado" in out.lower()
    assert "None" not in out


def test_includes_humidity_when_present():
    out = describe_current(_state(humidity=81))
    assert "81" in out


from src.world.weather import describe_forecast


FORECAST = [
    {"datetime": "2026-08-04T00:00:00", "condition": "fog",
     "templow": 11.8, "temperature": 12.4},
    {"datetime": "2026-08-05T00:00:00", "condition": "rainy",
     "templow": 11.7, "temperature": 13.1},
    {"datetime": "2026-08-06T00:00:00", "condition": "cloudy",
     "templow": 10.8, "temperature": 12.9},
]


def test_forecast_today_uses_first_entry():
    out = describe_forecast(FORECAST, "hoy")
    assert "niebla" in out.lower()
    assert "12" in out


def test_forecast_tomorrow_uses_second_entry():
    out = describe_forecast(FORECAST, "mañana")
    assert "lluvioso" in out.lower()
    assert "13" in out


def test_forecast_reports_min_and_max():
    out = describe_forecast(FORECAST, "mañana")
    assert "12" in out  # templow 11.7 -> 12
    assert "13" in out  # temperature 13.1 -> 13


def test_empty_forecast_says_so():
    out = describe_forecast([], "mañana")
    assert "no" in out.lower()
    assert "grados" not in out.lower()


def test_forecast_without_requested_day_says_so():
    out = describe_forecast(FORECAST[:1], "mañana")
    assert "no" in out.lower()


def test_forecast_never_mentions_rain_probability():
    # precipitation_probability is absent from this HA integration's payload
    # (verified 2026-08-04). Answering by percentage would invent data.
    out = describe_forecast(FORECAST, "mañana")
    assert "por ciento de lluvia" not in out.lower()
    assert "probabilidad" not in out.lower()


# ---------------------------------------------------------------------------
# Review 2026-08-06, bloqueante 1: payloads malformados NO pueden levantar una
# excepción. Desde `_handle_weather` una excepción atraviesa `dispatch()`,
# `MultiUserOrchestrator.process()`, `request_router` y `voice_pipeline` sin
# que nadie la atrape -> el usuario pregunta y no escucha NADA. El degradado
# obligatorio es el mismo que el módulo ya usa para vacío/None: NO_FORECAST /
# NO_DATA, hablado.
#
# Mutación que estos tests deben atrapar: volver a
#   day = forecast[index]; day.get("condition")   (sin isinstance)
#   round(float(low))                             (sin _as_number)
#   if not state / attrs = state.get("attributes") or {}   (sin isinstance)
# ---------------------------------------------------------------------------

import pytest

from src.world.weather import NO_DATA, NO_FORECAST


@pytest.mark.parametrize("bad_forecast", [
    None,
    "no soy una lista",
    {"forecast": []},          # dict: len() funciona, indexar por int no
    [None, None],              # elementos no-dict en la posición pedida
    [{}, "mañana soleado"],    # string donde se espera un dict
    [{}, 42],
    [{}, ["condition", "sunny"]],
])
def test_malformed_forecast_degrades_instead_of_raising(bad_forecast):
    assert describe_forecast(bad_forecast, "mañana") == NO_FORECAST


@pytest.mark.parametrize("bad_temps", [
    {"condition": "rainy", "templow": "unknown", "temperature": "unknown"},
    {"condition": "rainy", "templow": None, "temperature": "n/a"},
    {"condition": "rainy", "templow": {}, "temperature": []},
])
def test_non_numeric_forecast_temps_are_omitted_not_raised(bad_temps):
    # La condición sí se conoce, así que la frase se dice igual — solo se cae
    # el tramo de grados. Lo inaceptable sería el ValueError de float().
    out = describe_forecast([{}, bad_temps], "mañana")
    assert "lluvioso" in out.lower()
    assert "grados" not in out.lower()


def test_forecast_with_only_bad_temps_and_unknown_condition_says_no_forecast():
    out = describe_forecast([{}, {"condition": "no-existe", "temperature": "x"}], "mañana")
    assert out == NO_FORECAST


@pytest.mark.parametrize("bad_state", [
    "no soy un dict",
    42,
    [],
    ("state", "sunny"),
])
def test_malformed_current_state_degrades_instead_of_raising(bad_state):
    assert describe_current(bad_state) == NO_DATA


@pytest.mark.parametrize("bad_attrs", ["no soy un dict", None, 42, ["temperature", 20]])
def test_malformed_attributes_still_speak_the_condition(bad_attrs):
    # La condición vive en `state`, no en `attributes`: si el bloque de
    # atributos viene roto se pierden grados y humedad, pero la frase sale.
    # Lo inaceptable sería el AttributeError de `.get` sobre un no-dict.
    assert describe_current({"state": "sunny", "attributes": bad_attrs}) == "Hay soleado."


def test_non_numeric_current_temperature_is_omitted_not_raised():
    out = describe_current({
        "state": "sunny",
        "attributes": {"temperature": "unknown", "humidity": "n/a"},
    })
    assert out == "Hay soleado."


def test_boolean_temperature_is_not_spoken_as_a_number():
    # float(True) == 1.0 -> "1 grados". Un bool nunca es una temperatura.
    out = describe_current({"state": "sunny", "attributes": {"temperature": True}})
    assert "1 grados" not in out
    assert out == "Hay soleado."


def test_describe_forecast_pasado_manana_uses_index_2():
    forecast = [
        {"condition": "sunny", "temperature": 20, "templow": 10},
        {"condition": "rainy", "temperature": 18, "templow": 9},
        {"condition": "cloudy", "temperature": 15, "templow": 7},
    ]
    out = describe_forecast(forecast, "pasado mañana")
    assert out.startswith("Pasado mañana:")
    assert "nublado" in out and "entre 7 y 15 grados" in out


def test_describe_forecast_pasado_manana_short_forecast_is_honest():
    # Solo 2 días de pronóstico: índice 2 no existe → NO_FORECAST, no IndexError.
    forecast = [{"condition": "sunny", "temperature": 20},
                {"condition": "rainy", "temperature": 18}]
    assert describe_forecast(forecast, "pasado mañana") == NO_FORECAST
