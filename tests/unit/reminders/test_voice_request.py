"""Tests for the rioplatense voice reminder parser (T031-U1).

Every case runs against a frozen, aware household clock (America/Montevideo,
UTC-3, no DST). A successful parse must yield a trigger strictly after "now";
anything that would need a guess must come back with an ambiguity code and no
trigger (user answers Q7/Q8: ask, never assume 09:00).
"""
import ast
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.reminders.recurrence import next_trigger, parse_recurrence
from src.reminders.voice_request import (
    DeferredClause,
    ReminderAmbiguity as A,
    VoiceReminderRequest,
    VoiceRequestParser,
    extract_time_expression,
    parse_reminder_request,
)

MVD = ZoneInfo("America/Montevideo")
MONDAY_10 = datetime(2026, 11, 16, 10, 0, tzinfo=MVD)          # Monday
MONDAY_08 = datetime(2026, 11, 16, 8, 0, tzinfo=MVD)            # every clock below is still ahead
MONDAY_2359 = datetime(2026, 11, 16, 23, 59, tzinfo=MVD)
TUESDAY_0010 = datetime(2026, 11, 17, 0, 10, tzinfo=MVD)
DEC_31_22 = datetime(2026, 12, 31, 22, 0, tzinfo=MVD)           # Thursday
JAN_31_21 = datetime(2027, 1, 31, 21, 0, tzinfo=MVD)            # Sunday
ALIASES = ("cocina", "living", "living comedor", "baño")


def local(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=MVD).timestamp()


def parse(text, now=MONDAY_10, **kwargs):
    result = parse_reminder_request(text, now, **kwargs)
    assert result is not None, text
    return result


# ---------------------------------------------------------------------------
# Successful parses: trigger as UTC epoch of the local wall time.
# ---------------------------------------------------------------------------

SUCCESS = [
    # (utterance, now, expected trigger, recurrence, text)
    ("recordame sacar la basura a las 8 de la noche", MONDAY_10, local(2026, 11, 16, 20), None,
     "sacar la basura"),
    ("Recuérdame mañana a las ocho y media de la mañana llamar al médico", MONDAY_10,
     local(2026, 11, 17, 8, 30), None, "llamar al médico"),
    ("acordame de regar las plantas en 20 minutos", MONDAY_10, MONDAY_10.timestamp() + 1200, None,
     "regar las plantas"),
    ("haceme acordar que llame a mamá a las 3 de la tarde", MONDAY_10, local(2026, 11, 16, 15), None,
     "llame a mamá"),
    ("avisame en una hora y media que saque la ropa", MONDAY_10, MONDAY_10.timestamp() + 5400, None,
     "saque la ropa"),
    ("avisame en dos horas y cuarto que vuelva", MONDAY_10, MONDAY_10.timestamp() + 8100, None, "vuelva"),
    ("recordame dentro de dos horas y diez minutos mover el auto", MONDAY_10,
     MONDAY_10.timestamp() + 7800, None, "mover el auto"),
    ("recordame en media hora apagar el horno", MONDAY_10, MONDAY_10.timestamp() + 1800, None,
     "apagar el horno"),
    ("recordame en un cuarto de hora sacar la pizza", MONDAY_10, MONDAY_10.timestamp() + 900, None,
     "sacar la pizza"),
    ("recordame en treinta segundos mirar la olla", MONDAY_10, MONDAY_10.timestamp() + 30, None,
     "mirar la olla"),
    ("recordame en 1 hora y 5 minutos x", MONDAY_10, MONDAY_10.timestamp() + 3900, None, "x"),
    ("recordame pasado mañana a las 3 de la tarde ir al banco", MONDAY_10, local(2026, 11, 18, 15), None,
     "ir al banco"),
    ("recordame esta noche a las 11 cerrar la puerta", MONDAY_10, local(2026, 11, 16, 23), None,
     "cerrar la puerta"),
    ("recordame a la noche a las 9 regar", MONDAY_10, local(2026, 11, 16, 21), None, "regar"),
    ("recordame mañana a la mañana a las 7 pagar", MONDAY_10, local(2026, 11, 17, 7), None, "pagar"),
    ("recordame mañana por la tarde a las 4 pagar", MONDAY_10, local(2026, 11, 17, 16), None, "pagar"),
    ("recordame a las 8 menos cuarto de la noche x", MONDAY_10, local(2026, 11, 16, 19, 45), None, "x"),
    ("recordame a las 12 menos diez de la noche x", MONDAY_10, local(2026, 11, 16, 23, 50), None, "x"),
    ("recordame a la una menos cuarto de la tarde x", MONDAY_10, local(2026, 11, 16, 12, 45), None, "x"),
    ("recordame a las 8 y cuarto de la noche x", MONDAY_10, local(2026, 11, 16, 20, 15), None, "x"),
    ("recordame a las ocho y veinticinco de la noche x", MONDAY_10, local(2026, 11, 16, 20, 25), None, "x"),
    ("recordame a las ocho cuarenta de la noche x", MONDAY_10, local(2026, 11, 16, 20, 40), None, "x"),
    ("recordame a las 8 05 de la noche x", MONDAY_10, local(2026, 11, 16, 20, 5), None, "x"),
    ("recordame a las veintitrés y cuarenta y cinco x", MONDAY_10, local(2026, 11, 16, 23, 45), None, "x"),
    ("recordame a las 20:30 x", MONDAY_10, local(2026, 11, 16, 20, 30), None, "x"),
    ("recordame a las 8.15 de la noche x", MONDAY_10, local(2026, 11, 16, 20, 15), None, "x"),
    ("recordame a las 20 hs x", MONDAY_10, local(2026, 11, 16, 20), None, "x"),
    ("recordame a las 9 en punto de la noche x", MONDAY_10, local(2026, 11, 16, 21), None, "x"),
    ("recordame a las 3 de la tarde x", MONDAY_10, local(2026, 11, 16, 15), None, "x"),
    ("recordame a las 7 de la noche x", MONDAY_10, local(2026, 11, 16, 19), None, "x"),
    ("recordame a las 11 de la mañana x", MONDAY_10, local(2026, 11, 16, 11), None, "x"),
    ("recordame a las 11 de mañana x", MONDAY_10, local(2026, 11, 16, 11), None, "x"),
    ("recordame a las 9 de noche x", MONDAY_10, local(2026, 11, 16, 21), None, "x"),
    ("recordame a las 15 de la tarde x", MONDAY_10, local(2026, 11, 16, 15), None, "x"),
    ("recordame a las 22 de la noche x", MONDAY_10, local(2026, 11, 16, 22), None, "x"),
    ("recordame a la una del mediodía x", MONDAY_10, local(2026, 11, 16, 13), None, "x"),
    ("recordame a las 12 del mediodía x", MONDAY_10, local(2026, 11, 16, 12), None, "x"),
    ("recordame al mediodía almorzar", MONDAY_10, local(2026, 11, 16, 12), None, "almorzar"),
    ("recordame a eso de las 6 de la tarde x", MONDAY_10, local(2026, 11, 16, 18), None, "x"),
    ("recordame para las 6 de la tarde x", MONDAY_10, local(2026, 11, 16, 18), None, "x"),
    # 12 a.m. / p.m. and midnight.
    ("recordame a las 12 p.m. almorzar", MONDAY_10, local(2026, 11, 16, 12), None, "almorzar"),
    ("recordame a las 12 pm almorzar", MONDAY_10, local(2026, 11, 16, 12), None, "almorzar"),
    ("recordame a las 12 a.m. apagar", MONDAY_10, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame a las 11 a. m. x", MONDAY_10, local(2026, 11, 16, 11), None, "x"),
    ("recordame a las 7 p.m. x", MONDAY_10, local(2026, 11, 16, 19), None, "x"),
    ("recordame a las 12 de la noche apagar", MONDAY_10, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame a medianoche apagar", MONDAY_10, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame a la medianoche apagar", MONDAY_10, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame a las 24 apagar", MONDAY_10, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame a las 00:30 apagar", MONDAY_10, local(2026, 11, 17, 0, 30), None, "apagar"),
    # Small hours mean the coming night.
    ("recordame a las 3 de la madrugada x", MONDAY_10, local(2026, 11, 17, 3), None, "x"),
    ("recordame hoy a las 2 de la mañana x", MONDAY_10, local(2026, 11, 17, 2), None, "x"),
    ("recordame a las 3 de la madrugada x", TUESDAY_0010, local(2026, 11, 17, 3), None, "x"),
    ("recordame mañana a las 3 de la madrugada x", MONDAY_10, local(2026, 11, 17, 3), None, "x"),
    # 23:59, midnight crossing and month/year ends.
    ("recordame a las 12 de la noche apagar", MONDAY_2359, local(2026, 11, 17, 0), None, "apagar"),
    ("recordame en 2 minutos apagar", MONDAY_2359, MONDAY_2359.timestamp() + 120, None, "apagar"),
    ("recordame mañana a las 8 de la mañana x", MONDAY_2359, local(2026, 11, 17, 8), None, "x"),
    ("recordame a las 8 de la mañana x", TUESDAY_0010, local(2026, 11, 17, 8), None, "x"),
    ("recordame mañana a las 8 de la mañana x", DEC_31_22, local(2027, 1, 1, 8), None, "x"),
    ("recordame pasado mañana a las 9 de la noche x", DEC_31_22, local(2027, 1, 2, 21), None, "x"),
    ("recordame mañana a las 8 de la mañana x", JAN_31_21, local(2027, 2, 1, 8), None, "x"),
    ("recordame en 3 horas x", DEC_31_22, local(2027, 1, 1, 1), None, "x"),
    # Weekdays.
    ("recordame el miércoles a las 9 de la mañana pagar", MONDAY_10, local(2026, 11, 18, 9), None, "pagar"),
    ("recordame el domingo a las 9 de la mañana pagar", MONDAY_10, local(2026, 11, 22, 9), None, "pagar"),
    ("recordame el próximo lunes a las 9 de la mañana pagar", MONDAY_10, local(2026, 11, 23, 9), None,
     "pagar"),
    ("recordame el lunes que viene a las 9 de la mañana pagar", MONDAY_10, local(2026, 11, 23, 9), None,
     "pagar"),
    ("recordame el lunes próximo a las 9 de la mañana pagar", MONDAY_10, local(2026, 11, 23, 9), None,
     "pagar"),
    ("recordame este lunes a las 9 de la noche pagar", MONDAY_10, local(2026, 11, 16, 21), None, "pagar"),
    ("recordame el viernes a las 8 de la noche x", DEC_31_22, local(2027, 1, 1, 20), None, "x"),
    ("recordame mañana martes a las 8 de la noche x", MONDAY_10, local(2026, 11, 17, 20), None, "x"),
    ("recordame el martes a las 2 de la mañana x", MONDAY_10, local(2026, 11, 17, 2), None, "x"),
    # Recurrence: first occurrence strictly after now.
    ("recordame todos los días a las 22 tomar la pastilla", MONDAY_10, local(2026, 11, 16, 22), "daily",
     "tomar la pastilla"),
    ("recordame todos los días a las 9 de la mañana tomar la pastilla", MONDAY_10, local(2026, 11, 17, 9),
     "daily", "tomar la pastilla"),
    ("recordame cada día a las 9 de la noche x", MONDAY_10, local(2026, 11, 16, 21), "daily", "x"),
    ("recordame diariamente a las 9 de la noche x", MONDAY_10, local(2026, 11, 16, 21), "daily", "x"),
    ("recordame todas las noches a las 10 x", MONDAY_10, local(2026, 11, 16, 22), "daily", "x"),
    ("recordame todas las mañanas a las 7 x", MONDAY_10, local(2026, 11, 17, 7), "daily", "x"),
    ("recordame cada noche a las 10 x", MONDAY_10, local(2026, 11, 16, 22), "daily", "x"),
    ("recordame de lunes a viernes a las 7 de la mañana despertar a los chicos", MONDAY_10,
     local(2026, 11, 17, 7), "weekdays", "despertar a los chicos"),
    ("recordame de lunes a viernes a las 7 de la mañana x", datetime(2026, 11, 20, 9, 0, tzinfo=MVD),
     local(2026, 11, 23, 7), "weekdays", "x"),
    ("recordame los días de semana a las 7 de la noche x", datetime(2026, 11, 21, 9, 0, tzinfo=MVD),
     local(2026, 11, 23, 19), "weekdays", "x"),
    ("recordame todos los días hábiles a las 7 de la noche x", MONDAY_10, local(2026, 11, 16, 19),
     "weekdays", "x"),
    ("recordame entre semana a las 7 de la noche x", MONDAY_10, local(2026, 11, 16, 19), "weekdays", "x"),
    ("recordame todos los lunes a las 7 de la tarde sacar la basura", MONDAY_10, local(2026, 11, 16, 19),
     "weekly:1", "sacar la basura"),
    ("recordame todos los lunes a las 7 de la mañana sacar la basura", MONDAY_10, local(2026, 11, 23, 7),
     "weekly:1", "sacar la basura"),
    ("recordame los domingos a las 10 de la mañana x", MONDAY_10, local(2026, 11, 22, 10), "weekly:7", "x"),
    ("recordame cada jueves a las 10 de la noche x", MONDAY_10, local(2026, 11, 19, 22), "weekly:4", "x"),
    ("recordame todos los sábados a las 10 de la noche x", MONDAY_10, local(2026, 11, 21, 22), "weekly:6",
     "x"),
    # Content words that look like time words stay text.
    ("recordame llevar a las nenas a las 5 de la tarde", MONDAY_10, local(2026, 11, 16, 17), None,
     "llevar a las nenas"),
    ("recordame poner la ropa en el lavarropas a las 6 de la tarde", MONDAY_10, local(2026, 11, 16, 18), None,
     "poner la ropa en el lavarropas"),
    ("recordame que en 2 horas viene el técnico", MONDAY_10, MONDAY_10.timestamp() + 7200, None,
     "viene el técnico"),
]


@pytest.mark.parametrize("utterance,now,expected,recurrence,text", SUCCESS)
def test_success_table(utterance, now, expected, recurrence, text):
    result = parse(utterance, now)
    assert result.ambiguity is None, (result.ambiguity, result.detail)
    assert result.ok is True
    assert result.issues == ()
    assert result.trigger_at == pytest.approx(expected, abs=1e-6)
    assert result.trigger_at > now.timestamp()
    assert result.recurrence == recurrence
    assert result.text == text
    if recurrence is not None:
        parse_recurrence(result.recurrence)


def test_trigger_is_utc_epoch_of_local_wall_time():
    result = parse("recordame a las 8 de la noche x")
    # 20:00 in Montevideo (UTC-3) is 23:00 UTC.
    assert datetime.fromtimestamp(result.trigger_at, timezone.utc) == datetime(2026, 11, 16, 23, 0,
                                                                                  tzinfo=timezone.utc)


def test_clock_timezone_is_honoured():
    utc_now = MONDAY_10.astimezone(timezone.utc)
    madrid = MONDAY_10.astimezone(ZoneInfo("Europe/Madrid"))
    assert parse("recordame a las 8 de la noche x", utc_now).trigger_at == datetime(
        2026, 11, 16, 20, tzinfo=timezone.utc).timestamp()
    assert parse("recordame a las 8 de la noche x", madrid).trigger_at == datetime(
        2026, 11, 16, 20, tzinfo=ZoneInfo("Europe/Madrid")).timestamp()


# ---------------------------------------------------------------------------
# Ambiguity codes: no trigger, no recurrence, precise code.
# ---------------------------------------------------------------------------

AMBIGUOUS = [
    # MISSING_TIME (Q8: no default hour)
    ("recordame sacar la basura", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame mañana sacar la basura", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame mañana a la noche sacar la basura", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame el jueves pagar", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame todos los días tomar la pastilla", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame esta noche cerrar", MONDAY_10, A.MISSING_TIME, "no_time"),
    ("recordame en un rato llamar", MONDAY_10, A.MISSING_TIME, "vague_time"),
    ("recordame en unos minutos llamar", MONDAY_10, A.MISSING_TIME, "vague_time"),
    ("recordame en un par de horas llamar", MONDAY_10, A.MISSING_TIME, "vague_time"),
    ("recordame más tarde llamar", MONDAY_10, A.MISSING_TIME, "no_time"),
    # AMBIGUOUS_MERIDIEM
    ("recordame tomar la pastilla a las 8", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a la una pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 12 pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 8 y media pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 8 horas pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame mañana a las 8 pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame todos los días a las 8 pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame tomar la pastilla de la mañana a las 8", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 12 de la mañana pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 2 de la noche pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 10 de la tarde pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 8 de la madrugada pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a las 5 del mediodía pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame esta noche a las 8 de la mañana pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    ("recordame a la mañana a la noche a las 8 pagar", MONDAY_10, A.AMBIGUOUS_MERIDIEM, None),
    # PAST_TODAY: a clock time from 06:00 on that already passed today
    ("recordame a las 9 de la mañana pagar", MONDAY_10, A.PAST_TODAY, None),
    ("recordame hoy a las 9 de la mañana pagar", MONDAY_10, A.PAST_TODAY, None),
    ("recordame a las 10 de la mañana pagar", MONDAY_10, A.PAST_TODAY, None),
    ("recordame este lunes a las 9 de la mañana pagar", MONDAY_10, A.PAST_TODAY, None),
    ("recordame a las 8 de la mañana pagar", MONDAY_2359, A.PAST_TODAY, None),
    ("recordame a las 23:59 pagar", MONDAY_2359, A.PAST_TODAY, None),
    ("recordame esta noche a las 9 pagar", datetime(2026, 11, 16, 22, 0, tzinfo=MVD), A.PAST_TODAY, None),
    # AMBIGUOUS_DAY
    ("recordame el lunes a las 9 de la noche pagar", MONDAY_10, A.AMBIGUOUS_DAY, None),
    ("recordame lunes a las 9 de la noche pagar", MONDAY_10, A.AMBIGUOUS_DAY, None),
    ("recordame mañana a las 12 de la noche apagar", MONDAY_10, A.AMBIGUOUS_DAY, "midnight_with_day"),
    ("recordame el jueves a medianoche apagar", MONDAY_10, A.AMBIGUOUS_DAY, "midnight_with_day"),
    ("recordame mañana a las 8 de la mañana x", TUESDAY_0010, A.AMBIGUOUS_DAY, "tomorrow_before_dawn"),
    ("recordame pasado mañana a las 8 de la noche x", TUESDAY_0010, A.AMBIGUOUS_DAY, "tomorrow_before_dawn"),
    ("recordame hoy mañana a las 8 de la noche x", MONDAY_10, A.AMBIGUOUS_DAY, None),
    ("recordame mañana miércoles a las 8 de la noche x", MONDAY_10, A.AMBIGUOUS_DAY, None),
    ("recordame mañana todos los días a las 8 de la noche x", MONDAY_10, A.AMBIGUOUS_DAY, None),
    # MISSING_TEXT
    ("recordame a las 8 de la noche", MONDAY_10, A.MISSING_TEXT, None),
    ("recordame que mañana a las 8 de la noche", MONDAY_10, A.MISSING_TEXT, None),
    ("recordame que en 10 minutos por favor", MONDAY_10, A.MISSING_TEXT, None),
    # UNPARSEABLE (garbled or unsupported, never guessed)
    ("recordame a las 25 sacar", MONDAY_10, A.UNPARSEABLE, "invalid_clock_time"),
    ("recordame a las ochenta sacar", MONDAY_10, A.UNPARSEABLE, "invalid_clock_time"),
    ("recordame a las 8:75 sacar", MONDAY_10, A.UNPARSEABLE, "invalid_clock_time"),
    ("recordame a las 24 y media sacar", MONDAY_10, A.UNPARSEABLE, "invalid_clock_time"),
    ("recordame en blablá minutos sacar", MONDAY_10, A.UNPARSEABLE, "invalid_duration"),
    ("recordame en cero minutos sacar", MONDAY_10, A.UNPARSEABLE, "invalid_duration"),
    ("recordame a las 15 de la mañana x", MONDAY_10, A.UNPARSEABLE, "period_contradicts_hour"),
    ("recordame a las 23 de la tarde x", MONDAY_10, A.UNPARSEABLE, "period_contradicts_hour"),
    ("recordame a las 00:30 de la tarde x", MONDAY_10, A.UNPARSEABLE, "period_contradicts_hour"),
    ("recordame a las 8 de la noche y a las 9 de la noche x", MONDAY_10, A.UNPARSEABLE, "multiple_times"),
    ("recordame en 10 minutos y en 20 minutos x", MONDAY_10, A.UNPARSEABLE, "multiple_times"),
    ("recordame todos los días cada noche a las 9 x", MONDAY_10, A.UNPARSEABLE, "multiple_times"),
    ("recordame mañana en 10 minutos x", MONDAY_10, A.UNPARSEABLE, "conflicting_times"),
    ("recordame en 10 minutos a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "conflicting_times"),
    ("recordame los lunes y jueves a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_recurrence"),
    ("recordame todos los lunes y los viernes a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE,
     "unsupported_recurrence"),
    ("recordame el 15 a las 8 de la noche pagar la luz", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame el 3 de diciembre a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame en 3 días a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame la semana que viene a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame cada dos horas tomar agua", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame todos los meses a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
    ("recordame el fin de semana a las 8 de la noche x", MONDAY_10, A.UNPARSEABLE, "unsupported_date"),
]


# Words the grammar does not read are never dropped into the text and scheduled
# for today (review round 1, F1). The clock is Monday 08:00 so every hour named
# below is still ahead: a guess would come back ok, not PAST_TODAY by luck.
UNREAD_DATES = [
    ("recordame para navidad a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame el finde a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame a fin de mes a las 9 de la noche llamar", "unsupported_date"),
    ("recordame a principio de mes a las 9 de la noche llamar", "unsupported_date"),
    ("recordame la semana del 20 a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame día por medio a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame una vez por semana a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame un día sí y uno no a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame el día 15 a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame el día quince a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame el quince a las 10 de la mañana llamar", "unrecognised_date"),
    ("recordame el primero a las 10 de la mañana llamar", "unrecognised_date"),
    ("recordame el primero de mes a las 10 de la mañana llamar", "unrecognised_date"),
    ("recordame en una semana a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame dentro de un mes a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame en un día a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame el año que viene a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame el próximo año a las 10 de la mañana llamar", "unsupported_date"),
    ("recordame el feriado a las 10 de la mañana llamar", "unrecognised_date"),
    ("recordame mañana o pasado a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame el martes 17 a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame mañana 17 a las 9 de la noche llamar", "unrecognised_date"),
    ("recordame las medialunas de mañana a las 9 de la noche", "unrecognised_date"),
    ("recordame todos los días hábiles salvo feriados a las 9 de la noche x", "unrecognised_time"),
    ("recordame todos los días excepto el domingo a las 9 de la noche x", "unrecognised_time"),
    ("recordame a las 20 y pico sacar la basura", "unrecognised_time"),
    ("recordame a las 9 de la noche menos cuarto sacar", "unrecognised_time"),
    ("recordame a las 9 de la noche y cuarto sacar", "unrecognised_time"),
    ("recordame a las 9 de la noche y 10 sacar", "unrecognised_time"),
    ("recordame en 1 hora 30 sacar la ropa", "unrecognised_time"),
    ("recordame en 10 minutos y pico sacar la ropa", "unrecognised_time"),
]


@pytest.mark.parametrize("utterance,detail", UNREAD_DATES)
def test_unread_time_words_are_refused(utterance, detail):
    result = parse(utterance, MONDAY_08)
    assert result.ambiguity is A.UNPARSEABLE, (result.ambiguity, result.text)
    assert result.issues == (A.UNPARSEABLE,)
    assert result.trigger_at is None
    assert result.recurrence is None
    assert result.detail == detail


def test_unread_time_words_are_refused_in_a_deferred_clause():
    result = extract_time_expression("prendé la luz el finde a las 9 de la noche", MONDAY_08)
    assert (result.ambiguity, result.detail, result.trigger_at) == (A.UNPARSEABLE, "unrecognised_date", None)


def test_period_noun_and_singular_numbers_after_a_time_stay_content():
    morning = parse("recordame cambiar la alarma de la mañana a las 9 de la noche", MONDAY_08)
    assert (morning.ok, morning.text) == (True, "cambiar la alarma de la mañana")
    one = parse("recordame a las 9 de la noche una cosa importante", MONDAY_08)
    assert (one.ok, one.text) == (True, "una cosa importante")
    joined = parse("recordame mañana a las 8 de la noche y llamar a Ana", MONDAY_08)
    assert (joined.ok, joined.text) == (True, "llamar a Ana")


@pytest.mark.parametrize("utterance,now,code,detail", AMBIGUOUS)
def test_ambiguity_table(utterance, now, code, detail):
    result = parse(utterance, now)
    assert result.ambiguity is code
    assert result.issues[0] is code
    assert result.ok is False
    assert result.trigger_at is None
    assert result.recurrence is None
    assert result.detail == detail


def test_every_ambiguity_code_is_covered():
    assert {code for _, _, code, _ in AMBIGUOUS} == set(A)


def test_all_issues_reported_most_important_first():
    result = parse("recordame a las 8")
    assert result.issues == (A.MISSING_TEXT, A.AMBIGUOUS_MERIDIEM)
    assert result.ambiguity is A.MISSING_TEXT
    empty = parse("recordame")
    assert empty.issues == (A.MISSING_TEXT, A.MISSING_TIME)
    both = parse("recordame el lunes a las 8 pagar")
    assert both.issues == (A.AMBIGUOUS_DAY, A.AMBIGUOUS_MERIDIEM)


def test_seed_c02_rem_0010_is_ambiguous_meridiem():
    result = parse("recordame tomar la pastilla a las 8")
    assert (result.ambiguity, result.text, result.trigger_at) == (A.AMBIGUOUS_MERIDIEM, "tomar la pastilla", None)


# ---------------------------------------------------------------------------
# Verbs (voseo and tú), heads and non-reminders.
# ---------------------------------------------------------------------------

VERBS = [
    "recordame", "Recordáme", "recuérdame", "recuerdame", "recordanos", "recordale", "recordales",
    "recuérdale", "acordame", "acuérdame", "haceme acordar", "hacenos acordar", "avisame", "avísame",
    "avisanos", "avisale", "avisales", "no me dejes olvidar", "me podés recordar", "podrías recordarme",
    "podés avisarme", "me recordás", "me avisás",
]


@pytest.mark.parametrize("verb", VERBS)
def test_reminder_verbs(verb):
    result = parse(f"{verb} sacar la basura a las 8 de la noche")
    assert result.ok
    assert result.text == "sacar la basura"
    assert result.trigger_at == local(2026, 11, 16, 20)


@pytest.mark.parametrize("utterance", [
    "acordate que mañana viene Juan",          # memory proposal (U3b), not a reminder
    "recordá que me gusta el café",
    "apagá la luz en 10 minutos",
    "poné un temporizador de 5 minutos",
    "¿qué recordatorios tengo?",
    "cancelá el recordatorio de sacar la basura",
    "el recordatorio de las 8",
    "",
    "   ",
    "recordamelo",
])
def test_not_a_reminder_request(utterance):
    assert parse_reminder_request(utterance, MONDAY_10) is None


@pytest.mark.parametrize("prefix", ["", "che, ", "Che ", "por favor ", "¿", "Nexa, ", "nexa ", "che nexa, por favor "])
def test_leading_politeness_and_wake_words(prefix):
    result = parse(f"{prefix}recordame en 5 minutos mirar la olla", wake_words=("Nexa",))
    assert result.ok
    assert result.text == "mirar la olla"


def test_wake_word_needs_a_word_boundary():
    assert parse_reminder_request("nexarecordame en 5 minutos x", MONDAY_10, wake_words=("nexa",)) is None


def test_unconfigured_wake_word_is_not_stripped():
    assert parse_reminder_request("nexa recordame en 5 minutos x", MONDAY_10) is None


def test_complement_is_the_raw_span_after_the_verb():
    utterance = "Che, recordame  que MAÑANA a las 7 de la mañana apague la luz del living, por favor"
    result = parse(utterance)
    assert result.complement == "que MAÑANA a las 7 de la mañana apague la luz del living, por favor"
    assert utterance.endswith(result.complement)
    assert result.text == "apague la luz del living"
    assert result.time_phrase == "MAÑANA a las 7 de la mañana"


def test_complement_kept_when_ambiguous():
    result = parse("recordame apagar la luz a las 8")
    assert result.complement == "apagar la luz a las 8"
    assert result.text == "apagar la luz"


def test_text_keeps_accents_and_case():
    result = parse("recordame Llamar a Ñoño en 5 minutos")
    assert result.text == "Llamar a Ñoño"


def test_text_cleanup_drops_connectors_and_politeness():
    assert parse("recordame de que pague la luz en 5 minutos").text == "pague la luz"
    assert parse("acordame de comprar pan en 5 minutos, porfa").text == "comprar pan"
    assert parse("recordame, en 5 minutos, que llame a Ana.").text == "llame a Ana"
    assert parse("recordame mañana a las 8 de la noche y llamar a Ana").text == "llamar a Ana"


# ---------------------------------------------------------------------------
# Number words.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("words,minutes", [
    ("un", 1), ("uno", 1), ("dos", 2), ("tres", 3), ("cuatro", 4), ("cinco", 5), ("seis", 6),
    ("siete", 7), ("ocho", 8), ("nueve", 9), ("diez", 10), ("once", 11), ("doce", 12), ("trece", 13),
    ("catorce", 14), ("quince", 15), ("dieciséis", 16), ("diecisiete", 17), ("dieciocho", 18),
    ("diecinueve", 19), ("veinte", 20), ("veintiún", 21), ("veintiuno", 21), ("veintidós", 22),
    ("veintitrés", 23), ("veinticuatro", 24), ("veinticinco", 25), ("veintiséis", 26),
    ("veintisiete", 27), ("veintiocho", 28), ("veintinueve", 29), ("treinta", 30),
    ("treinta y uno", 31), ("cuarenta", 40), ("cuarenta y cinco", 45), ("cincuenta", 50),
    ("cincuenta y nueve", 59), ("sesenta", 60), ("setenta", 70), ("ochenta", 80), ("noventa", 90),
    ("noventa y nueve", 99), ("12", 12), ("120", 120),
])
def test_number_words_in_durations(words, minutes):
    result = parse(f"recordame en {words} minutos x")
    assert result.ok
    assert result.trigger_at == MONDAY_10.timestamp() + minutes * 60


@pytest.mark.parametrize("words,hour", [
    ("una", 13), ("dos", 14), ("tres", 15), ("cuatro", 16), ("cinco", 17), ("seis", 18), ("siete", 19),
    ("ocho", 20), ("nueve", 21),
])
def test_number_words_in_clock_hours(words, hour):
    article = "la" if words == "una" else "las"
    result = parse(f"recordame a {article} {words} de la tarde x")
    assert result.trigger_at == local(2026, 11, 16, hour)


@pytest.mark.parametrize("words,hour,minute", [
    ("diez y diez", 22, 10), ("once y veinte", 23, 20), ("nueve y cinco", 21, 5),
    ("diez y cuarto", 22, 15), ("diez y media", 22, 30), ("once menos cuarto", 22, 45),
    ("once menos veinte", 22, 40), ("diez y cincuenta y cinco", 22, 55), ("diez y 7 minutos", 22, 7),
])
def test_number_words_in_minutes(words, hour, minute):
    result = parse(f"recordame a las {words} de la noche x")
    assert result.trigger_at == local(2026, 11, 16, hour, minute)


def test_period_words_inside_the_content_are_not_a_day():
    result = parse("recordame cambiar la alarma de la mañana a las 9 de la noche")
    assert result.text == "cambiar la alarma de la mañana"
    assert result.trigger_at == local(2026, 11, 16, 21)


def test_tonight_counts_as_today_against_other_day_words():
    assert parse("recordame mañana esta noche a las 9 x").ambiguity is A.AMBIGUOUS_DAY
    assert parse("recordame todos los días esta noche a las 9 x").ambiguity is A.AMBIGUOUS_DAY


def test_time_phrase_lists_the_spoken_time_words():
    result = parse("recordame pasado mañana sacar la basura a las 9 de la noche")
    assert result.time_phrase == "pasado mañana a las 9 de la noche"


# ---------------------------------------------------------------------------
# Zone phrase (configured aliases only, no name inference).
# ---------------------------------------------------------------------------

def test_zone_phrase_after_the_verb():
    result = parse("avisame en la cocina a las 8 de la noche que saque la basura", zone_aliases=ALIASES)
    assert result.zone_phrase == "en la cocina"
    assert result.text == "saque la basura"
    assert result.ok


def test_zone_phrase_at_the_end():
    result = parse("recordame sacar la basura a las 8 de la noche en el Living, por favor", zone_aliases=ALIASES)
    assert result.zone_phrase == "en el Living"
    assert result.text == "sacar la basura"


def test_zone_phrase_multiword_alias_and_accents():
    assert parse("recordame en el living comedor x en 5 minutos",
                 zone_aliases=ALIASES).zone_phrase == "en el living comedor"
    assert parse("recordame x en 5 minutos en el baño", zone_aliases=ALIASES).zone_phrase == "en el baño"
    assert parse("recordame x en 5 minutos en el bano", zone_aliases=("Baño",)).zone_phrase == "en el bano"


def test_zone_phrase_needs_a_configured_alias():
    result = parse("recordame sacar la basura a las 8 de la noche en el garage", zone_aliases=ALIASES)
    assert result.zone_phrase is None
    assert result.text == "sacar la basura en el garage"
    assert parse("recordame x en 5 minutos en la cocina").zone_phrase is None


def test_zone_phrase_in_the_middle_is_content():
    result = parse("recordame dejar la leche en la cocina y sacar la basura en 5 minutos", zone_aliases=ALIASES)
    assert result.zone_phrase is None
    assert result.text == "dejar la leche en la cocina y sacar la basura"


def test_zone_phrase_needs_contiguous_words():
    result = parse("recordame en la, cocina x en 5 minutos", zone_aliases=ALIASES)
    assert result.zone_phrase == "en la, cocina"
    split = parse("recordame en la x cocina en 5 minutos", zone_aliases=ALIASES)
    assert split.zone_phrase is None
    # A time phrase between the article and the alias does not make a zone phrase.
    around_time = parse("recordame x en el a las 9 de la noche living", zone_aliases=ALIASES)
    assert around_time.zone_phrase is None
    assert around_time.text == "x en el living"
    assert around_time.trigger_at == local(2026, 11, 16, 21)


def test_zone_only_request_is_missing_text():
    result = parse("avisame en la cocina en 10 minutos", zone_aliases=ALIASES)
    assert result.zone_phrase == "en la cocina"
    assert result.ambiguity is A.MISSING_TEXT


# ---------------------------------------------------------------------------
# Second entry point: domotics clause without a reminder verb (R4).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clause,expected,rest", [
    ("apagá la luz en 10 minutos", MONDAY_10.timestamp() + 600, "apagá la luz"),
    ("apagá la luz del living dentro de media hora", MONDAY_10.timestamp() + 1800, "apagá la luz del living"),
    ("prendé la luz del living mañana a las 7 de la mañana", local(2026, 11, 17, 7), "prendé la luz del living"),
    ("en 10 minutos apagá la luz", MONDAY_10.timestamp() + 600, "apagá la luz"),
    ("Che, bajá la persiana a las 9 de la noche", local(2026, 11, 16, 21), "bajá la persiana"),
])
def test_deferred_clause(clause, expected, rest):
    result = extract_time_expression(clause, MONDAY_10)
    assert isinstance(result, DeferredClause)
    assert result.ok
    assert result.trigger_at == expected
    assert result.clause == rest
    assert result.recurrence is None


@pytest.mark.parametrize("clause,code", [
    ("apagá la luz", A.MISSING_TIME),
    ("apagá la luz más tarde", A.MISSING_TIME),
    ("apagá la luz a las 8", A.AMBIGUOUS_MERIDIEM),
    ("apagá la luz a las 9 de la mañana", A.PAST_TODAY),
    ("apagá la luz el lunes a las 9 de la noche", A.AMBIGUOUS_DAY),
    ("apagá la luz a las 31", A.UNPARSEABLE),
    ("en 10 minutos", A.MISSING_TEXT),
])
def test_deferred_clause_ambiguities(clause, code):
    result = extract_time_expression(clause, MONDAY_10)
    assert result.ambiguity is code
    assert result.trigger_at is None


def test_deferred_clause_keeps_leading_que_and_zone_words():
    result = extract_time_expression("que apague la luz en la cocina en 10 minutos", MONDAY_10)
    assert result.clause == "que apague la luz en la cocina"
    assert result.time_phrase == "en 10 minutos"


def test_deferred_clause_recurrence():
    result = extract_time_expression("prendé la luz del patio todos los días a las 8 de la noche", MONDAY_10)
    assert result.recurrence == "daily"
    assert result.trigger_at == local(2026, 11, 16, 20)
    assert result.clause == "prendé la luz del patio"


# ---------------------------------------------------------------------------
# Injected clock, purity, recurrence compatibility, invariants, speed.
# ---------------------------------------------------------------------------

def test_naive_clock_is_rejected():
    with pytest.raises(ValueError, match="aware_clock_required"):
        parse_reminder_request("recordame en 5 minutos x", datetime(2026, 11, 16, 10, 0))
    with pytest.raises(ValueError, match="aware_clock_required"):
        extract_time_expression("apagá la luz en 5 minutos", datetime(2026, 11, 16, 10, 0))
    with pytest.raises(ValueError, match="aware_clock_required"):
        VoiceRequestParser(lambda: datetime(2026, 11, 16, 10, 0)).parse_reminder("recordame en 5 minutos x")
    with pytest.raises(ValueError, match="aware_clock_required"):
        VoiceRequestParser(lambda: datetime(2026, 11, 16, 10, 0)).parse_deferred("apagá la luz en 5 minutos")


def test_parser_reads_the_injected_clock_on_every_call():
    times = iter([MONDAY_10, MONDAY_2359])
    parser = VoiceRequestParser(lambda: next(times), zone_aliases=ALIASES, wake_words=("nexa",))
    first = parser.parse_reminder("nexa, recordame en la cocina sacar la basura a las 12 de la noche")
    second = parser.parse_reminder("recordame en 5 minutos x")
    assert isinstance(first, VoiceReminderRequest)
    assert first.trigger_at == local(2026, 11, 17, 0)
    assert first.zone_phrase == "en la cocina"
    assert second.trigger_at == MONDAY_2359.timestamp() + 300


def test_parser_deferred_uses_the_clock():
    parser = VoiceRequestParser(lambda: MONDAY_2359)
    result = parser.parse_deferred("apagá la luz en 5 minutos")
    assert result.trigger_at == MONDAY_2359.timestamp() + 300


def test_parser_non_reminder_returns_none():
    assert VoiceRequestParser(lambda: MONDAY_10).parse_reminder("prendé la luz") is None


def test_module_is_pure_stdlib():
    path = Path(__file__).resolve().parents[3] / "src" / "reminders" / "voice_request.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= set(sys.stdlib_module_names), modules - set(sys.stdlib_module_names)
    assert not {"open", "asyncio", "sqlite3", "os", "socket", "subprocess"} & modules


@pytest.mark.parametrize("recurrence", sorted({r for *_, r, _ in SUCCESS if r}))
def test_recurrence_strings_feed_next_trigger(recurrence):
    parse_recurrence(recurrence)
    assert next_trigger(MONDAY_10.timestamp(), recurrence) > MONDAY_10.timestamp()


PROPERTY_PHRASES = [
    "recordame x a las {h} de la {p}", "recordame x mañana a las {h} de la {p}", "recordame x hoy a las {h} de la {p}",
    "recordame x a las {h} y media de la {p}", "recordame x a las {h} menos cuarto de la {p}",
    "recordame x el {d} a las {h} de la {p}", "recordame x el próximo {d} a las {h} de la {p}",
    "recordame x este {d} a las {h} de la {p}", "recordame x todos los {d} a las {h} de la {p}",
    "recordame x todos los días a las {h} de la {p}", "recordame x de lunes a viernes a las {h} de la {p}",
    "recordame x pasado mañana a las {h} de la {p}", "recordame x a las {h24}:{m:02d}", "recordame x en {h} minutos",
    "recordame x en {h} horas", "recordame x a las {h}", "recordame x a la medianoche", "recordame x al mediodía",
    "recordame x a las 12 a.m.", "recordame x a las 12 p.m.",
]


def _property_cases():
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    for step in range(0, 7 * 24 * 60, 97):           # a week of clocks, 97-minute steps
        now = datetime(2026, 11, 16, tzinfo=MVD) + timedelta(minutes=step)
        k = step // 97
        yield now, dict(h=1 + k % 12, p=("mañana", "tarde", "noche", "madrugada")[k % 4], d=days[k % 7],
                        h24=k % 24, m=(k * 7) % 60)


def test_property_success_implies_future_trigger():
    checked = successes = 0
    for now, values in _property_cases():
        for template in PROPERTY_PHRASES:
            result = parse(template.format(**values), now)
            checked += 1
            if result.ok:
                successes += 1
                assert result.trigger_at is not None and result.trigger_at > now.timestamp(), (template, values, now)
                assert result.trigger_at - now.timestamp() <= 15 * 86400
                if result.recurrence:
                    parse_recurrence(result.recurrence)
            else:
                assert result.trigger_at is None and result.recurrence is None
                assert result.issues and result.ambiguity is result.issues[0]
    assert checked > 2000
    assert successes > checked // 3


def test_microbenchmark_under_one_millisecond_per_parse():
    phrases = [utterance for utterance, *_ in SUCCESS] + [utterance for utterance, *_ in AMBIGUOUS]
    parser = VoiceRequestParser(lambda: MONDAY_10, zone_aliases=ALIASES, wake_words=("nexa",))
    for phrase in phrases:
        parser.parse_reminder(phrase)
    best = float("inf")
    for _ in range(5):
        started = time_module.perf_counter()
        for phrase in phrases:
            parser.parse_reminder(phrase)
        best = min(best, (time_module.perf_counter() - started) / len(phrases))
    assert best < 0.001, f"{best * 1000:.3f} ms per parse"
