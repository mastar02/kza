"""T030-U1: rioplatense timer request parser (durations, labels, selectors), including negatives."""

import pytest

from src.nlu.household_intents import HouseholdFamily, HouseholdOperation, classify_household_intent
from src.timers.voice_request import (
    MAX_TIMER_SECONDS,
    TimerRequest,
    TimerRequestIssue,
    TimerSelector,
    fold_label,
    format_duration,
    parse_timer_request,
    parse_timer_selector,
)

I = TimerRequestIssue

# Whole utterance -> (seconds, label). The complement comes from the real P3-F2 classifier.
CREATE = [
    ("poné un temporizador de 5 minutos", 300, None),
    ("poneme un tímer de media hora", 1800, None),
    ("temporizador de 5 minutos", 300, None),
    ("timer de 90 segundos", 90, None),
    ("activá un timer de 10 minutos", 600, None),
    ("creá un temporizador de veinte minutos", 1200, None),
    ("programá un timer de una hora", 3600, None),
    ("arrancá un timer de un minuto", 60, None),
    ("haceme un timer de un cuarto de hora", 900, None),
    ("poné un timer de cuarto de hora", 900, None),
    ("poné un timer de tres cuartos de hora", 2700, None),
    ("poné un timer de una hora y media", 5400, None),
    ("poné un timer de hora y media", 5400, None),
    ("poné un timer de dos horas y cuarto", 8100, None),
    ("poné un timer de 1 hora 30 minutos", 5400, None),
    ("poné un timer de una hora y 15 minutos", 4500, None),
    ("poné un timer de dos minutos y medio", 150, None),
    ("poné un timer de un minuto y medio", 90, None),
    ("poné un timer de cinco y medio minutos", 330, None),
    ("poné un timer de dos minutos y 30 segundos", 150, None),
    ("poné un timer de medio minuto", 30, None),
    ("poné un timer de cuarenta y cinco minutos", 2700, None),
    ("poné un timer de veinticinco minutos", 1500, None),
    ("poné un timer de veintiún minutos", 1260, None),
    ("poné un timer de diez segundos", 10, None),
    ("poné un timer de 30 seg", 30, None),
    ("poné un timer de 2 min", 120, None),
    ("poné un timer de 24 horas", MAX_TIMER_SECONDS, None),
    ("tirame un timer de 3 minutos", 180, None),
    ("metele un timer de ocho minutos", 480, None),
    ("poné el timer en 5 minutos", 300, None),
    ("poné un temporizador de 1 minuto para la pastilla", 60, "pastilla"),
    ("poné un timer para la pasta de 8 minutos", 480, "pasta"),
    ("poné un timer de 8 minutos para la pasta", 480, "pasta"),
    ("timer de la pasta de 8 minutos", 480, "pasta"),
    ("creá un temporizador llamado pasta de 8 minutos", 480, "pasta"),
    ("poné un timer de 5 minutos para sacar el pollo", 300, "sacar el pollo"),
    ("poné un timer de 5 minutos para la salsa por favor", 300, "salsa"),
    ("nexa poné un timer de 5 minutos", 300, None),
    ("poné una alarma de 10 minutos", 600, None),
    ("poneme una alarma de diez minutos", 600, None),
    ("alarma de 5 minutos", 300, None),
    ("poné una alarma en 10 minutos", 600, None),
]


@pytest.mark.parametrize(("text", "seconds", "label"), CREATE)
def test_creation_reads_duration_and_label(text, seconds, label):
    intent = classify_household_intent(text, wake_words=("nexa",))
    assert (intent.family, intent.operation) == (HouseholdFamily.TIMER, HouseholdOperation.CREATE)
    assert parse_timer_request(intent.complement) == TimerRequest(seconds, label)


NEGATIVE = [
    ("", I.MISSING_DURATION),
    ("para la pasta", I.MISSING_DURATION),
    ("de 5", I.UNPARSEABLE),
    ("de cinco", I.UNPARSEABLE),
    ("de cero minutos", I.UNPARSEABLE),
    ("de 0 minutos", I.UNPARSEABLE),
    ("de minutos", I.UNPARSEABLE),
    ("de 1:30", I.UNPARSEABLE),
    ("de 1 hora 30", I.UNPARSEABLE),
    ("de 5 minutos y 10 minutos", I.UNPARSEABLE),
    ("de 5 minutos y un rato", I.UNPARSEABLE),
    ("de 5 minutos para las 7", I.UNPARSEABLE),
    ("de unos minutos", I.VAGUE_DURATION),
    ("de algunos segundos", I.VAGUE_DURATION),
    ("de un par de minutos", I.VAGUE_DURATION),
    ("de un rato", I.VAGUE_DURATION),
    ("por un ratito", I.VAGUE_DURATION),
    ("para las 7", I.CLOCK_TIME),
    ("a las 8 y media", I.CLOCK_TIME),
    ("para el mediodía", I.CLOCK_TIME),
    ("de 25 horas", I.TOO_LONG),
    ("de 24 horas y 1 minuto", I.TOO_LONG),
    ("de cinco minutos en la cocina", I.ROOM_TARGET),
    ("de 5 minutos en el living", I.ROOM_TARGET),
    ("en el hall de 5 minutos", I.ROOM_TARGET),
]


@pytest.mark.parametrize(("complement", "issue"), NEGATIVE)
def test_never_guesses(complement, issue):
    parsed = parse_timer_request(complement)
    assert (parsed.duration_s, parsed.issue) == (None, issue)


def test_room_word_as_a_label_is_not_a_room_target():
    """Only 'en <room>' names a ring room; 'el timer de la cocina' is a label."""
    assert parse_timer_request("de la cocina de 5 minutos") == TimerRequest(300, "cocina")
    assert parse_timer_request("de un cuarto de hora") == TimerRequest(900, None)


def test_label_keeps_accents_and_compares_folded():
    parsed = parse_timer_request("de 5 minutos para el café")
    assert parsed == TimerRequest(300, "café")
    assert fold_label("Café") == fold_label("cafe") == "cafe"


SELECT = [
    ("cancelá el timer de la pasta", TimerSelector("pasta")),
    ("pausá el timer de la pasta", TimerSelector("pasta")),
    ("seguí el timer", TimerSelector()),
    ("sacá el timer de 5 minutos", TimerSelector(None, 300)),
    ("borrá el temporizador de la pasta de 8 minutos", TimerSelector("pasta", 480)),
    ("cuánto falta para el timer de la pasta", TimerSelector("pasta")),
    ("¿cuánto le falta al timer?", TimerSelector()),
    ("qué timers hay", TimerSelector()),
    ("cuántos timers hay corriendo", TimerSelector()),
    ("decime los timers", TimerSelector()),
    ("pará el timer que está sonando", TimerSelector()),
    ("cancelá todos los timers", TimerSelector()),
]


@pytest.mark.parametrize(("text", "selector"), SELECT)
def test_selector_names_a_timer_by_label_or_duration(text, selector):
    intent = classify_household_intent(text)
    assert intent.family is HouseholdFamily.TIMER
    assert parse_timer_selector(intent.complement) == selector
    assert parse_timer_selector(intent.complement).empty is (selector == TimerSelector())


@pytest.mark.parametrize(("seconds", "spoken"), [
    (1, "1 segundo"), (30, "30 segundos"), (60, "1 minuto"), (90, "1 minuto y 30 segundos"), (300, "5 minutos"),
    (3600, "1 hora"), (5400, "1 hora y 30 minutos"), (7200, "2 horas"), (3661, "1 hora, 1 minuto y 1 segundo"),
    (0.2, "1 segundo"), (59.01, "1 minuto"), (0, "0 segundos"), (-3, "0 segundos"),
])
def test_format_duration_rounds_up_to_whole_seconds(seconds, spoken):
    assert format_duration(seconds) == spoken
