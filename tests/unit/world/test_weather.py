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


from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.core.household_clock import SUPPORTED_HOUSEHOLD_TIMEZONE
from src.world.weather import (NO_DATA, NO_FORECAST, ForecastAlignment, HorizonKind, WeatherHorizon,
                               describe_forecast, forecast_target_date, is_indoor_temperature_question,
                               parse_weather_horizon)

# T037-U1: el día del pronóstico se elige por FECHA local del hogar, nunca por
# posición. Hora de Montevideo = UTC-3 sin horario de verano.
ZONE = ZoneInfo("America/Montevideo")
TODAY = date(2026, 8, 4)
TOMORROW = date(2026, 8, 5)
AFTER = date(2026, 8, 6)


def _day(iso: str, condition: str, low: float, high: float) -> dict:
    return {"datetime": iso, "condition": condition, "templow": low, "temperature": high}


FOG = _day("2026-08-04T15:00:00+00:00", "fog", 11.8, 12.4)
RAIN = _day("2026-08-05T15:00:00+00:00", "rainy", 11.7, 13.1)
CLOUD = _day("2026-08-06T15:00:00+00:00", "cloudy", 10.8, 12.9)
YESTERDAY = _day("2026-08-03T15:00:00+00:00", "sunny", 5.0, 9.0)
FORECAST = [FOG, RAIN, CLOUD]


def _say(forecast, dia="mañana", target=TOMORROW):
    return describe_forecast(forecast, dia, target, ZONE)


def test_forecast_today_is_selected_by_date():
    answer = _say(FORECAST, "hoy", TODAY)
    assert answer.aligned and answer.alignment is ForecastAlignment.ALIGNED
    assert answer.text == "Hoy: con niebla, entre 12 y 12 grados."


def test_forecast_tomorrow_is_selected_by_date():
    answer = _say(FORECAST)
    assert answer.aligned
    assert answer.text == "Mañana: lluvioso, entre 12 y 13 grados."


def test_forecast_reports_min_and_max():
    out = _say(FORECAST).text
    assert "12" in out  # templow 11.7 -> 12
    assert "13" in out  # temperature 13.1 -> 13


@pytest.mark.parametrize("order", [
    [CLOUD, RAIN, FOG],
    [RAIN, FOG, CLOUD],
    [CLOUD, FOG, RAIN],
])
def test_reordered_forecast_still_speaks_the_requested_day(order):
    assert _say(order).text == "Mañana: lluvioso, entre 12 y 13 grados."
    assert _say(order, "hoy", TODAY).text.startswith("Hoy: con niebla")
    assert _say(order, "pasado mañana", AFTER).text.startswith("Pasado mañana: nublado")


def test_first_entry_yesterday_does_not_shift_the_days():
    # Por posición, "mañana" sería FOG (hoy) y "hoy" sería YESTERDAY.
    forecast = [YESTERDAY, FOG, RAIN]
    assert _say(forecast).text.startswith("Mañana: lluvioso")
    assert _say(forecast, "hoy", TODAY).text.startswith("Hoy: con niebla")


def test_empty_forecast_says_so():
    answer = _say([])
    assert answer.text == NO_FORECAST and answer.alignment is ForecastAlignment.NO_ENTRIES
    assert not answer.aligned


@pytest.mark.parametrize("forecast", [[FOG], [FOG, CLOUD], [YESTERDAY, FOG], [CLOUD]])
def test_forecast_without_requested_day_says_so(forecast):
    # PQ-T037-1 (user answer: "Nombra el día"): a missing day names it.
    answer = _say(forecast)
    assert answer.text == "No tengo el pronóstico para mañana."
    assert answer.alignment is ForecastAlignment.MISSING_DAY and not answer.aligned


def test_unparsed_weekday_string_has_no_target_and_no_answer():
    # A raw weekday string is not a horizon: weekdays come parsed (T037-U2).
    assert forecast_target_date("el sábado", datetime(2026, 8, 4, 15, tzinfo=UTC), ZONE) is None
    answer = describe_forecast(FORECAST, "el sábado", None, ZONE)
    assert answer.text == "No tengo el pronóstico para el sábado."
    assert answer.alignment is ForecastAlignment.MISSING_DAY


@pytest.mark.parametrize("bad_datetime", [
    "2026-08-05T15:00:00",            # naive: su día local sería una suposición
    "2026-08-05",                      # solo fecha: también naive
    "mañana",                          # no parseable
    "",
    "2026-13-45T00:00:00+00:00",       # fecha inválida
    "0001-01-01T00:00:00+14:00",       # fuera de rango al convertir
    None,
    1754406000,                        # epoch numérico, no ISO
    True,
])
def test_unusable_entry_datetime_makes_the_payload_invalid(bad_datetime):
    broken = {**RAIN, "datetime": bad_datetime}
    for forecast in ([FOG, broken, CLOUD], [broken], [FOG, CLOUD, broken]):
        answer = _say(forecast, "hoy", TODAY)
        assert answer.text == NO_FORECAST
        assert answer.alignment is ForecastAlignment.INVALID_ENTRIES and not answer.aligned


def test_entry_without_datetime_makes_the_payload_invalid():
    undated = {key: value for key, value in RAIN.items() if key != "datetime"}
    assert _say([FOG, undated]).alignment is ForecastAlignment.INVALID_ENTRIES


def test_conflicting_duplicates_for_the_day_are_not_spoken():
    other = _day("2026-08-05T18:00:00+00:00", "sunny", 14.0, 20.0)
    answer = _say([FOG, RAIN, other, CLOUD])
    assert answer.text == "No tengo el pronóstico para mañana."
    assert answer.alignment is ForecastAlignment.CONFLICTING_DAY and not answer.aligned
    # Las otras fechas siguen respondiendo: el conflicto es solo de ese día.
    assert _say([FOG, RAIN, other, CLOUD], "hoy", TODAY).aligned


def test_same_condition_with_different_temperatures_is_a_conflict():
    # R1 (review 1): la condición sola no alcanza; 13 y 20 grados para el
    # mismo día local se dirían distinto, así que no se elige ninguno.
    warmer = {**RAIN, "datetime": "2026-08-05T18:00:00+00:00", "temperature": 20}
    answer = _say([FOG, RAIN, warmer, CLOUD])
    assert answer.text == "No tengo el pronóstico para mañana."
    assert answer.alignment is ForecastAlignment.CONFLICTING_DAY and not answer.aligned
    colder = {**RAIN, "datetime": "2026-08-05T18:00:00+00:00", "templow": 4}
    assert _say([FOG, colder, RAIN]).alignment is ForecastAlignment.CONFLICTING_DAY


def test_identical_duplicates_for_the_day_agree():
    same = {**RAIN, "datetime": "2026-08-05T18:00:00-03:00"}
    answer = _say([FOG, RAIN, same])
    assert answer.aligned and answer.text.startswith("Mañana: lluvioso")


def test_utc_midnight_entry_belongs_to_the_previous_local_day():
    # 00:00Z del 5 son las 21:00 del 4 en Montevideo: es "hoy", no "mañana".
    utc_midnight = _day("2026-08-05T00:00:00+00:00", "sunny", 8.0, 16.0)
    assert _say([utc_midnight]).alignment is ForecastAlignment.MISSING_DAY
    assert _say([utc_midnight], "hoy", TODAY).text == "Hoy: soleado, entre 8 y 16 grados."


@pytest.mark.parametrize("iso", ["2026-08-05T03:00:00+00:00", "2026-08-05T00:00:00-03:00",
                                 "2026-08-05T02:59:59+00:00"])
def test_local_midnight_boundary(iso):
    entry = _day(iso, "sunny", 8.0, 16.0)
    expected = TOMORROW if iso != "2026-08-05T02:59:59+00:00" else TODAY
    other = TODAY if expected == TOMORROW else TOMORROW
    assert _say([entry], "x", expected).aligned
    assert _say([entry], "x", other).alignment is ForecastAlignment.MISSING_DAY


@pytest.mark.parametrize(("now", "hoy", "manana", "pasado"), [
    # 23:30 del 4 en Montevideo aunque en UTC ya sea el 5.
    (datetime(2026, 8, 5, 2, 30, tzinfo=UTC), TODAY, TOMORROW, AFTER),
    # Medianoche local exacta: ya es el 5.
    (datetime(2026, 8, 5, 3, 0, tzinfo=UTC), TOMORROW, AFTER, date(2026, 8, 7)),
    # 00:30 UTC del 4 son las 21:30 del 3.
    (datetime(2026, 8, 4, 0, 30, tzinfo=UTC), date(2026, 8, 3), TODAY, TOMORROW),
])
def test_target_date_is_the_household_local_date(now, hoy, manana, pasado):
    assert forecast_target_date("hoy", now, ZONE) == hoy
    assert forecast_target_date("mañana", now, ZONE) == manana
    assert forecast_target_date("Manana", now, ZONE) == manana
    assert forecast_target_date("pasado mañana", now, ZONE) == pasado
    assert forecast_target_date("pasado manana", now, ZONE) == pasado


def test_target_date_refuses_a_naive_clock():
    with pytest.raises(ValueError):
        forecast_target_date("mañana", datetime(2026, 8, 5, 2, 30), ZONE)


def test_household_timezone_matches_the_shipped_routines_timezone():
    import yaml

    # Relativo al archivo de test, no al cwd: sigue valiendo si un shard cambia de directorio.
    settings = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
    with settings.open() as fh:
        configured = yaml.safe_load(fh)["routines"]["timezone"]
    assert SUPPORTED_HOUSEHOLD_TIMEZONE == configured == ZONE.key


def test_forecast_never_mentions_rain_probability():
    # precipitation_probability is absent from this HA integration's payload
    # (verified 2026-08-04). Answering by percentage would invent data.
    out = _say([{**RAIN, "precipitation_probability": 80}]).text
    assert "por ciento de lluvia" not in out.lower()
    assert "probabilidad" not in out.lower()
    assert "80" not in out


# ---------------------------------------------------------------------------
# Review 2026-08-06, bloqueante 1: payloads malformados NO pueden levantar una
# excepción. Desde `_handle_weather` una excepción atraviesa `dispatch()`,
# `MultiUserOrchestrator.process()`, `request_router` y `voice_pipeline` sin
# que nadie la atrape -> el usuario pregunta y no escucha NADA. El degradado
# obligatorio es el mismo que el módulo ya usa para vacío/None: NO_FORECAST /
# NO_DATA, hablado.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_forecast", [
    None,
    "no soy una lista",
    {"forecast": []},
    [None, None],
    [FOG, "mañana soleado"],
    [FOG, 42],
    [FOG, ["condition", "sunny"]],
])
def test_malformed_forecast_degrades_instead_of_raising(bad_forecast):
    answer = _say(bad_forecast)
    assert answer.text == NO_FORECAST and not answer.aligned


@pytest.mark.parametrize("bad_temps", [
    {"templow": "unknown", "temperature": "unknown"},
    {"templow": None, "temperature": "n/a"},
    {"templow": {}, "temperature": []},
])
def test_non_numeric_forecast_temps_are_omitted_not_raised(bad_temps):
    # La condición sí se conoce, así que la frase se dice igual — solo se cae
    # el tramo de grados. Lo inaceptable sería el ValueError de float().
    answer = _say([FOG, {**RAIN, **bad_temps}])
    assert answer.aligned and answer.text == "Mañana: lluvioso."


def test_only_max_temperature_is_spoken_alone():
    assert _say([{**RAIN, "templow": None}]).text == "Mañana: lluvioso, 13 grados."


def test_forecast_with_only_bad_temps_and_unknown_condition_says_no_forecast():
    answer = _say([FOG, {**RAIN, "condition": "no-existe", "temperature": "x", "templow": None}])
    assert answer.text == NO_FORECAST
    # El día sí quedó alineado; lo que falta es contenido decible.
    assert answer.alignment is ForecastAlignment.ALIGNED


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


def test_describe_forecast_pasado_manana_is_selected_by_date():
    out = _say([CLOUD, FOG, RAIN], "pasado mañana", AFTER).text
    assert out.startswith("Pasado mañana:")
    assert "nublado" in out and "entre 11 y 13 grados" in out


def test_describe_forecast_pasado_manana_short_forecast_is_honest():
    # Solo 2 días de pronóstico: la fecha pedida no está y se nombra (PQ-T037-1).
    answer = _say([FOG, RAIN], "pasado mañana", AFTER)
    assert answer.text == "No tengo el pronóstico para pasado mañana."
    assert answer.alignment is ForecastAlignment.MISSING_DAY


# ---------------------------------------------------------------------------
# T037-U2: horizon of a weather question (user answer Q23) and indoor
# temperature questions. The words decide the day; the clock resolves it.
# ---------------------------------------------------------------------------

CURRENT = WeatherHorizon(HorizonKind.CURRENT)
UNSUPPORTED = WeatherHorizon(HorizonKind.UNSUPPORTED)
SEVERAL = WeatherHorizon(HorizonKind.SEVERAL_DAYS)


def _day_ahead(offset):
    return WeatherHorizon(HorizonKind.FORECAST, offset=offset)


@pytest.mark.parametrize(("text", "expected"), [
    # Present tense about today: current conditions ("¿qué tiempo hace?", Q23).
    ("qué tiempo hace", CURRENT),
    ("qué tiempo hace hoy", CURRENT),
    ("¿Qué tiempo hace esta mañana?", CURRENT),
    ("llueve por la mañana", CURRENT),
    ("hace frío a la mañana", CURRENT),
    ("cómo está el clima esta noche", CURRENT),
    # The morning is today, never tomorrow.
    ("¿va a llover esta mañana?", _day_ahead(0)),
    ("va a llover por la mañana", _day_ahead(0)),
    ("va a llover a la mañana", _day_ahead(0)),
    ("va a llover en la mañana", _day_ahead(0)),
    # Future about today: today's forecast entry.
    ("¿va a llover hoy?", _day_ahead(0)),
    ("va a llover", _day_ahead(0)),
    ("cuál es el pronóstico", _day_ahead(0)),
    ("qué tiempo va a hacer esta noche", _day_ahead(0)),
    ("qué temperatura habrá hoy", _day_ahead(0)),  # T037-U2b: "habrá" is future
    ("va a llover hoy a la tarde", _day_ahead(0)),
    # Named days.
    ("va a llover mañana", _day_ahead(1)),
    ("Mañana llueve?", _day_ahead(1)),
    ("va a llover mañana por la mañana", _day_ahead(1)),
    ("qué tiempo hace mañana a la mañana", _day_ahead(1)),
    ("el pronóstico de mañana", _day_ahead(1)),
    ("qué tiempo hace pasado mañana", _day_ahead(2)),
    ("va a llover pasado mañana a la mañana", _day_ahead(2)),
    # Weekdays: by date, never the current conditions.
    ("¿va a llover el sábado?", WeatherHorizon(HorizonKind.FORECAST, weekday=5)),
    ("qué tiempo va a hacer este lunes", WeatherHorizon(HorizonKind.FORECAST, weekday=0)),
    ("cómo está el clima el miércoles", WeatherHorizon(HorizonKind.FORECAST, weekday=2)),
    ("va a llover el sábado que viene", WeatherHorizon(HorizonKind.FORECAST, weekday=5, next_week=True)),
    ("va a llover el próximo sábado", WeatherHorizon(HorizonKind.FORECAST, weekday=5, next_week=True)),
    ("va a llover el sábado próximo", WeatherHorizon(HorizonKind.FORECAST, weekday=5, next_week=True)),
    # Unsupported: refused, never the current conditions.
    ("va a llover la semana que viene", UNSUPPORTED),
    ("qué tiempo va a hacer la próxima semana", UNSUPPORTED),
    ("el pronóstico para esta semana", UNSUPPORTED),
    ("va a llover el fin de semana", UNSUPPORTED),
    ("va a llover el sábado de la semana que viene", UNSUPPORTED),
    ("qué tiempo va a hacer el mes que viene", UNSUPPORTED),
    ("va a llover los próximos días", UNSUPPORTED),
    ("va a llover en tres días", UNSUPPORTED),
    ("va a llover el 25", UNSUPPORTED),
    ("va a llover el 3 de diciembre", UNSUPPORTED),
    # A weekday with a day of the month, or an ordinal, is a calendar date (review T037-U2-RV3).
    ("qué tiempo hace el lunes 10", UNSUPPORTED),
    ("va a llover el lunes 17", UNSUPPORTED),
    ("va a llover el sábado 8?", UNSUPPORTED),
    ("va a llover el 1ro", UNSUPPORTED),
    ("qué tiempo va a hacer el 1ero de diciembre", UNSUPPORTED),
    ("va a llover el 1er día del mes", UNSUPPORTED),
    # Fix verification T037-U2-FV2: a day number after "día", a month named in words.
    ("va a llover el día 25", UNSUPPORTED),
    ("va a llover el primero de diciembre", UNSUPPORTED),
    ("va a llover en diciembre", UNSUPPORTED),
    ("llovió ayer?", UNSUPPORTED),
    # Past weekdays, weeks, months and years (review T037-U2-RV1): never the coming day.
    ("estuvo lloviendo el sábado pasado", UNSUPPORTED),
    ("el martes pasado llovió?", UNSUPPORTED),
    ("cuál fue el pronóstico del sábado pasado", UNSUPPORTED),
    ("qué tiempo hizo el miércoles pasado", UNSUPPORTED),
    ("estuvo lloviendo el mes pasado", UNSUPPORTED),
    ("qué tiempo va a hacer el mes pasado", UNSUPPORTED),
    ("llovió mucho la semana pasada", UNSUPPORTED),
    ("hizo frío el año pasado", UNSUPPORTED),
    # Fix verification T037-U2-FV2: "pasado" before the weekday or period.
    ("estuvo lloviendo el pasado sábado", UNSUPPORTED),
    ("estuvo lloviendo la pasada semana", UNSUPPORTED),
    ("va a llover el pasado mes", UNSUPPORTED),
    # A future day after a past-looking word stays a forecast day.
    ("va a llover el lunes, pasado mañana", SEVERAL),
    # Two different days in one question.
    ("va a llover hoy o mañana", SEVERAL),
    ("va a llover mañana o el sábado", SEVERAL),
    ("va a llover mañana o pasado mañana", SEVERAL),
    # "pasado mañana" after a weekday is a second future day, not a past weekday.
    ("va a llover el sábado o pasado mañana", SEVERAL),
])
def test_weather_horizon_table(text, expected):
    assert parse_weather_horizon(text) == expected


# Tuesday 4 August 2026, 23:30 in Montevideo (02:30Z on the 5th).
TUESDAY_NIGHT = datetime(2026, 8, 5, 2, 30, tzinfo=UTC)


@pytest.mark.parametrize(("horizon", "target", "label"), [
    (_day_ahead(0), TODAY, "hoy"),
    (_day_ahead(1), TOMORROW, "mañana"),
    (_day_ahead(2), AFTER, "pasado mañana"),
    (WeatherHorizon(HorizonKind.FORECAST, weekday=2), TOMORROW, "el miércoles"),
    (WeatherHorizon(HorizonKind.FORECAST, weekday=5), date(2026, 8, 8), "el sábado"),
    (WeatherHorizon(HorizonKind.FORECAST, weekday=0), date(2026, 8, 10), "el lunes"),
    # The weekday of today, said bare: today (named "hoy" in the answer).
    (WeatherHorizon(HorizonKind.FORECAST, weekday=1), TODAY, "hoy"),
    # With "que viene"/"próximo": a week from today.
    (WeatherHorizon(HorizonKind.FORECAST, weekday=1, next_week=True), date(2026, 8, 11), "el martes que viene"),
    # "el próximo sábado" on a Tuesday is this Saturday.
    (WeatherHorizon(HorizonKind.FORECAST, weekday=5, next_week=True), date(2026, 8, 8), "el sábado"),
])
def test_forecast_horizon_resolves_by_household_date(horizon, target, label):
    resolved = forecast_target_date(horizon, TUESDAY_NIGHT, ZONE)
    assert resolved == target
    assert horizon.spoken_day((resolved - TODAY).days) == label


@pytest.mark.parametrize("horizon", [CURRENT, UNSUPPORTED, SEVERAL])
def test_non_forecast_horizons_have_no_target(horizon):
    assert forecast_target_date(horizon, TUESDAY_NIGHT, ZONE) is None


def test_weekday_forecast_names_the_day_when_found_and_when_missing():
    saturday = _day("2026-08-08T15:00:00+00:00", "rainy", 9.0, 14.0)
    assert _say([FOG, RAIN, saturday], "el sábado", date(2026, 8, 8)).text == (
        "El sábado: lluvioso, entre 9 y 14 grados.")
    missing = _say(FORECAST, "el sábado", date(2026, 8, 8))
    assert missing.text == "No tengo el pronóstico para el sábado."
    assert missing.alignment is ForecastAlignment.MISSING_DAY


ROOM_WORDS = ("living", "cocina", "escritorio", "sala", "cuarto")


@pytest.mark.parametrize("text", [
    "qué temperatura hay en el living",
    "¿Qué temperatura hace en la cocina?",
    "cuántos grados hay en el escritorio",
    "qué temperatura tiene el living",
    "cuál es la temperatura del living",
    "a cuánto está la temperatura en la sala",
    "qué temperatura hay en el dormitorio",
    "qué temperatura hay en la pieza",
    "qué temperatura hay adentro",
    "qué temperatura hay en casa",
    "cuál va a ser la temperatura del living mañana",  # T037-U2b: future form
])
def test_room_or_house_temperature_questions_are_indoor(text):
    assert is_indoor_temperature_question(text, ROOM_WORDS)


@pytest.mark.parametrize("text", [
    "qué temperatura hace",
    "qué temperatura hace afuera",
    "cuántos grados hay afuera",
    "qué temperatura hace afuera de la casa",
    "cuántos grados hay afuera de la cocina",  # review T037-U2-RV5: outside, not the room
    "qué temperatura hace fuera del living",
    "qué tiempo hace en el living",          # not a temperature question
    "bajá la temperatura del living",        # thermostat command, not a question
    "poné la temperatura del living en 22",
    "subí los grados del escritorio",
])
def test_outdoor_or_non_question_temperatures_are_not_indoor(text):
    assert not is_indoor_temperature_question(text, ROOM_WORDS)


def test_configured_rooms_count_as_indoor_places():
    assert not is_indoor_temperature_question("qué temperatura hay en el quincho")
    assert is_indoor_temperature_question("qué temperatura hay en el quincho", ("quincho",))
