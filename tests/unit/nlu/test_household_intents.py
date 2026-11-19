"""P3-F2: the deterministic household intent precedence layer (classifier layer)."""

import logging
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.nlu.command_grammar import llm_intent_evidenced, non_immediate_action_reason, parse_command
from src.nlu.household_intents import (
    HouseholdFamily,
    HouseholdIntentClassifier,
    HouseholdIntentSettings,
    HouseholdOperation,
    classify_household_intent,
    filler_span,
    starts_with_reminder_request,
    strip_fillers,
)
from src.reminders.voice_request import parse_reminder_request
from tests.mocks.c02_matrix import CLASSIFIER, C02Entry, effects

F, O = HouseholdFamily, HouseholdOperation


def _shape(intent):
    if intent is None:
        return None
    return intent.family, intent.operation, intent.rule


# Precedence table: utterance -> (family, operation, rule) or None (unclaimed).
PRECEDENCE = [
    # R3 reminder/deferral verbs win over every domotics, music, list, sync or enrollment word.
    ("recordame apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("recordáme apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("recuérdame tomar agua", (F.REMINDER, O.CREATE, "R3")),
    ("recuerdame poner música", (F.REMINDER, O.CREATE, "R3")),
    ("recordame agregar leche a la lista de compras", (F.REMINDER, O.CREATE, "R3")),
    ("recordame abrir la persiana mañana", (F.REMINDER, O.CREATE, "R3")),
    ("recordame actualizar la lista", (F.REMINDER, O.CREATE, "R3")),
    ("recordame registrar el gasto del super", (F.REMINDER, O.CREATE, "R3")),
    ("recordame cancelar el turno del médico", (F.REMINDER, O.CREATE, "R3")),
    ("recordame que mañana viene el plomero", (F.REMINDER, O.CREATE, "R3")),
    ("recordanos sacar la basura", (F.REMINDER, O.CREATE, "R3")),
    ("acordame de cerrar la persiana", (F.REMINDER, O.CREATE, "R3")),
    ("acuérdame de llamar a mamá", (F.REMINDER, O.CREATE, "R3")),
    ("haceme acordar de apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("hacenos acordar de regar", (F.REMINDER, O.CREATE, "R3")),
    ("avisame cuando termine el lavarropas", (F.REMINDER, O.CREATE, "R3")),
    ("avísame en diez minutos", (F.REMINDER, O.CREATE, "R3")),
    ("no me dejes olvidar sacar la basura", (F.REMINDER, O.CREATE, "R3")),
    ("me podés recordar apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("podés recordarme apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("Nexa, recordame prender la luz del living", (F.REMINDER, O.CREATE, "R3")),
    ("che por favor recordame cerrar la puerta", (F.REMINDER, O.CREATE, "R3")),
    ("recordale a Ana apagar la luz", (F.REMINDER, O.CREATE, "R3")),
    ("avisales a los chicos que la cena está lista", (F.REMINDER, O.CREATE, "R3")),
    ("recordame apagar la luz y cerrar la puerta", (F.REMINDER, O.CREATE, "R3")),
    ("recordame tomar la pastilla a las 8 y apagá la luz", (F.REMINDER, O.CLARIFY, "R3")),
    ("recordame apagar la luz y prendé la tele", (F.REMINDER, O.CLARIFY, "R3")),
    # U3b memory proposals are not reminders (the router runs them earlier).
    ("acordate que mañana a las 8 viene el plomero", None),
    # Review T031-U2-RV7: 'acordate de <algo>' asks KZA to remind; 'acordate (de) que' stays a memory request.
    ("acordate de sacar la basura a las 8 de la noche", (F.REMINDER, O.CREATE, "R3")),
    ("acuérdate de llamar a mamá", (F.REMINDER, O.CREATE, "R3")),
    ("acordate de que mañana viene el plomero", None),
    ("avisame cuando sean las 9 de la noche", (F.REMINDER, O.CREATE, "R3")),
    ("poné una alarma al mediodía", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma cuando sean las 7 de la tarde", (F.REMINDER, O.CREATE, "R3a")),
    ("recordá que el gas se paga el 10", None),
    # R1 object operations.
    ("cancelá el recordatorio de agregar leche a la lista", (F.REMINDER, O.CANCEL, "R1")),
    ("cancela el recordatorio de agua", (F.REMINDER, O.CANCEL, "R1")),
    ("borrá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("eliminá los recordatorios", (F.REMINDER, O.CANCEL, "R1")),
    ("olvidate del recordatorio de la basura", (F.REMINDER, O.CANCEL, "R1")),
    ("olvidate de la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("cancelá el timer de la pasta", (F.TIMER, O.CANCEL, "R1")),
    ("sacá el tímer", (F.TIMER, O.CANCEL, "R1")),
    ("anulá el temporizador", (F.TIMER, O.CANCEL, "R1")),
    ("pará el timer", (F.TIMER, O.STOP, "R1")),
    ("Para el timer", (F.TIMER, O.STOP, "R1")),
    ("detené el temporizador", (F.TIMER, O.STOP, "R1")),
    ("apagá el timer", (F.TIMER, O.STOP, "R1")),
    ("callá el temporizador", (F.TIMER, O.STOP, "R1")),
    ("apagá la rutina de la noche", (F.ROUTINE, O.STOP, "R1")),
    ("borrá todos los timers", (F.TIMER, O.CANCEL, "R1")),
    ("limpiá el timer", None),
    ("borrá el recordatorio de comprar pan y la lista de compras", (F.REMINDER, O.CANCEL, "R1")),
    ("pausá el timer de la pasta", (F.TIMER, O.PAUSE, "R1")),
    ("seguí el timer", (F.TIMER, O.RESUME, "R1")),
    ("reanudá el temporizador", (F.TIMER, O.RESUME, "R1")),
    ("desactivá el timer", (F.TIMER, O.DISABLE, "R1")),
    ("pará la rutina de la noche", (F.ROUTINE, O.STOP, "R1")),
    ("desactivá la rutina buenas noches", (F.ROUTINE, O.DISABLE, "R1")),
    ("borrá la rutina buenas noches", (F.ROUTINE, O.CANCEL, "R1")),
    ("borrá la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("eliminá la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("borra la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("vaciá la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("vacía la lista de compras", (F.LIST, O.CANCEL, "R1")),
    ("limpiá la lista del super", (F.LIST, O.CANCEL, "R1")),
    ("pará la lista", (F.LIST, O.STOP, "R1")),
    ("¿qué recordatorios hay?", (F.REMINDER, O.QUERY, "R1")),
    ("qué recordatorios tengo", (F.REMINDER, O.QUERY, "R1")),
    ("¿cuánto le falta al timer?", (F.TIMER, O.QUERY, "R1")),
    ("cuánto queda del temporizador", (F.TIMER, O.QUERY, "R1")),
    ("qué timers hay", (F.TIMER, O.QUERY, "R1")),
    ("qué hay en la lista de compras", (F.LIST, O.QUERY, "R1")),
    ("cuáles son las listas", (F.LIST, O.QUERY, "R1")),
    ("decime la lista de compras", (F.LIST, O.QUERY, "R1")),
    ("lee la lista de compras", (F.LIST, O.QUERY, "R1")),
    ("qué rutinas hay", (F.ROUTINE, O.QUERY, "R1")),
    # R2 creation and list items.
    ("poné un temporizador de 5 minutos", (F.TIMER, O.CREATE, "R2")),
    ("poneme un tímer de media hora", (F.TIMER, O.CREATE, "R2")),
    ("creá un timer de diez minutos para la pasta", (F.TIMER, O.CREATE, "R2")),
    ("programá un nuevo temporizador de una hora", (F.TIMER, O.CREATE, "R2")),
    ("activá un timer de 3 minutos", (F.TIMER, O.CREATE, "R2")),
    ("podés poner un timer de 3 minutos", (F.TIMER, O.CREATE, "R2")),
    ("temporizador de 5 minutos", (F.TIMER, O.CREATE, "R2")),
    # Rioplatense creation forms and queries (review F6).
    ("haceme un timer de 5 minutos", (F.TIMER, O.CREATE, "R2")),
    ("metele un timer de 10 minutos", (F.TIMER, O.CREATE, "R2")),
    ("tirame un timer de 3 minutos", (F.TIMER, O.CREATE, "R2")),
    ("temporizador cinco minutos", (F.TIMER, O.CREATE, "R2")),
    ("¿tengo recordatorios?", (F.REMINDER, O.QUERY, "R1")),
    ("¿hay timers?", (F.TIMER, O.QUERY, "R1")),
    ("¿hay algún temporizador?", (F.TIMER, O.QUERY, "R1")),
    ("hay que poner la mesa", None),
    ("haceme una lista de compras", None),
    ("temporizador", None),
    ("agregá pan y leche a la lista de compras", (F.LIST, O.ADD_ITEM, "R2")),
    ("agrega leche a la lista de compras", (F.LIST, O.ADD_ITEM, "R2")),
    ("anotá yerba en la lista", (F.LIST, O.ADD_ITEM, "R2")),
    ("poné tomates en la lista de compras", (F.LIST, O.ADD_ITEM, "R2")),
    ("quitá la leche de la lista", (F.LIST, O.REMOVE_ITEM, "R2")),
    ("sacá el pan de la lista de compras", (F.LIST, O.REMOVE_ITEM, "R2")),
    ("borrá la leche de la lista", (F.LIST, O.REMOVE_ITEM, "R2")),
    ("crea una lista compartida de viajes", (F.LIST, O.CREATE, "R2")),
    # R3a alarma (user answer Q4).
    ("poné una alarma de 10 minutos", (F.TIMER, O.CREATE, "R3a")),
    ("poneme una alarma en media hora", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma a las 7", (F.REMINDER, O.CREATE, "R3a")),
    ("programá la alarma para las siete", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma", (F.TIMER, O.CLARIFY, "R3a")),
    ("activá la alarma", None),
    ("desactivá la alarma", None),
    ("armá la alarma", None),
    # T030-U1 R3: the alarm a duration created is a timer, so its follow-ups reach the timers (R1).
    # The security alarm verbs above stay unclaimed (design R3a).
    ("desarmá la alarma", None),
    ("activá las alarmas", None),
    ("cancelá la alarma", (F.TIMER, O.CANCEL, "R1")),
    ("sacá la alarma de la pasta", (F.TIMER, O.CANCEL, "R1")),
    ("cancelá las alarmas", (F.TIMER, O.CANCEL, "R1")),
    ("apagá la alarma", (F.TIMER, O.STOP, "R1")),
    ("pará la alarma", (F.TIMER, O.STOP, "R1")),
    ("pausá la alarma", (F.TIMER, O.PAUSE, "R1")),
    ("reanudá la alarma", (F.TIMER, O.RESUME, "R1")),
    ("cancelá el despertador", (F.TIMER, O.CANCEL, "R1")),
    ("apagá el despertador", (F.TIMER, O.STOP, "R1")),
    ("desactivá el despertador", (F.TIMER, O.DISABLE, "R1")),
    ("qué alarmas hay", (F.TIMER, O.QUERY, "R1")),
    ("cuánto falta para la alarma", (F.TIMER, O.QUERY, "R1")),
    ("¿hay alguna alarma?", (F.TIMER, O.QUERY, "R1")),
    ("decime las alarmas", (F.TIMER, O.QUERY, "R1")),
    # User answer T030-U1-Q8 (A): a cancel or query of an alarm with a clock time reaches the reminders (it was
    # created as one, R3a); without a clock time, and for pause/resume, the alarm stays a timer. User answer
    # T031-U2-Q1: a stop or disable of an alarm with a clock time is its cancel (rows below).
    ("cancelá la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("borrá el despertador de las 6:30", (F.REMINDER, O.CANCEL, "R1")),
    ("sacá la alarma de la una", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma de las siete y media", (F.REMINDER, O.CANCEL, "R1")),
    ("anulá la alarma de las veintitrés", (F.REMINDER, O.CANCEL, "R1")),
    ("olvidate de la alarma de las 8", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma del mediodía", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma para las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá las alarmas de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("qué alarmas hay a las 7", (F.REMINDER, O.QUERY, "R1")),
    ("¿hay alguna alarma a las 7?", (F.REMINDER, O.QUERY, "R1")),
    ("decime la alarma de las 8 y media", (F.REMINDER, O.QUERY, "R1")),
    ("cuánto falta para la alarma de las 7", (F.REMINDER, O.QUERY, "R1")),
    ("cancelá la alarma de 7 minutos", (F.TIMER, O.CANCEL, "R1")),
    ("cancelá la alarma de 7", (F.TIMER, O.CANCEL, "R1")),  # no article: not a clock time
    ("cancelá la alarma de las 7 minutos", (F.TIMER, O.CANCEL, "R1")),
    ("cancelá la alarma de la pasta", (F.TIMER, O.CANCEL, "R1")),
    ("cancelá la alarma de la lasaña", (F.TIMER, O.CANCEL, "R1")),
    ("cancelá el timer de las 7", (F.TIMER, O.CANCEL, "R1")),
    ("pausá la alarma de las 7", (F.TIMER, O.PAUSE, "R1")),
    ("reanudá la alarma de las 7", (F.TIMER, O.RESUME, "R1")),
    # User answer T031-U2-Q1 (2026-11-18): apagá/pará (and the other stop verbs) an alarm with a clock time is
    # its cancel; with a duration, a name or no time it stays the timers' stop/disable. User answer BH-R1-Q1: the
    # security verbs (desactivá/activá/armá/desarmá la alarma) stay unclaimed with a clock time or a day too
    # (design R3a); "desactivá el despertador <hora>" is no security alarm and stays the cancel.
    ("apagá la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("pará la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("para la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("frená la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("silenciá la alarma de mañana", (F.REMINDER, O.CANCEL, "R1")),
    ("desactivá la alarma de las 7", None),
    ("desactivar la alarma de las 7 de la mañana", None),
    ("desactivá la alarma para mañana", None),
    ("desactivá la alarma de la casa a las 23", None),
    ("desactive las alarmas de las 7", None),
    ("desactivá el despertador de las 6:30", (F.REMINDER, O.CANCEL, "R1")),
    ("apagame el despertador de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("detené la alarma del mediodía", (F.REMINDER, O.CANCEL, "R1")),
    ("callá la alarma de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("apagá las alarmas de las 7", (F.REMINDER, O.CANCEL, "R1")),
    ("pará la alarma para las 8", (F.REMINDER, O.CANCEL, "R1")),
    ("apagá la alarma de 10 minutos", (F.TIMER, O.STOP, "R1")),
    ("pará la alarma de las 7 minutos", (F.TIMER, O.STOP, "R1")),
    ("apagá la alarma de 7", (F.TIMER, O.STOP, "R1")),  # no article: not a clock time
    ("apagá la alarma de la pasta", (F.TIMER, O.STOP, "R1")),
    ("desactivá el despertador de 10 minutos", (F.TIMER, O.DISABLE, "R1")),
    ("apagá el timer de las 7", (F.TIMER, O.STOP, "R1")),
    ("activá la alarma de las 7", None),
    ("armá la alarma de las 7", None),
    ("desarmá la alarma de las 7", None),
    ("desactivá las alarmas", None),
    ("cancelá el recordatorio de las 7", (F.REMINDER, O.CANCEL, "R1")),
    # Review T031-U2-RV2: an article before 'N horas' names the hour of the day, not a duration.
    ("poné una alarma a las 19 horas", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma para las 8 horas", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma a las 7 horas de la mañana", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma a las 19 hs", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma de las 19 horas", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma de 19 horas", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de 2 horas", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de las 7 minutos", (F.TIMER, O.CREATE, "R3a")),
    ("cancelá la alarma de las 10 horas", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma de las 10 hs", (F.REMINDER, O.CANCEL, "R1")),
    ("qué alarmas hay a las 19 horas", (F.REMINDER, O.QUERY, "R1")),
    ("cancelá la alarma de 2 horas", (F.TIMER, O.CANCEL, "R1")),
    # Review T031-U2-RV5: 'a las H y M minutos' is a clock time; 'de H hora(s) y M minutos' stays a duration.
    ("poné una alarma a las 20 y 5 minutos para la reunión", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma a las 7 y 15 minutos de la mañana", (F.REMINDER, O.CREATE, "R3a")),
    ("poné una alarma de 1 hora y 5 minutos", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de 20 y 5 minutos", (F.TIMER, O.CREATE, "R3a")),
    # Review T031-U2-RV3: a day alone names an alarm set for a clock time (a timer has no day); the
    # time question is answered from the timers and the alarm reminders, never the cloud model.
    ("cancelá la alarma de mañana", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma de hoy", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá la alarma de pasado mañana", (F.REMINDER, O.CANCEL, "R1")),
    ("borrá el despertador del martes", (F.REMINDER, O.CANCEL, "R1")),
    ("qué alarmas hay para mañana", (F.REMINDER, O.QUERY, "R1")),
    ("apagá la alarma de mañana", (F.REMINDER, O.CANCEL, "R1")),  # user answer T031-U2-Q1
    ("cancelá la alarma de la mañana", (F.TIMER, O.CANCEL, "R1")),
    ("a qué hora suena la alarma", (F.TIMER, O.QUERY, "R1")),
    ("¿a qué hora suena el despertador?", (F.TIMER, O.QUERY, "R1")),
    ("a qué hora suena la alarma de mañana", (F.REMINDER, O.QUERY, "R1")),
    ("a qué hora abre el banco", None),
    # R3a durations the timer parser reads (16-19, 21-29, tres cuartos, hora y media).
    ("poné una alarma de diecisiete minutos", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de veintidós minutos", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de veintiún minutos", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de tres cuartos de hora", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de hora y media", (F.TIMER, O.CREATE, "R3a")),
    ("poné una alarma de setenta minutos", (F.TIMER, O.CREATE, "R3a")),
    # R4 deferred home actions.
    ("apagá la luz en 10 minutos", (F.DEFERRED_ACTION, O.CREATE, "R4")),
    ("prendé el aire a las 8", (F.DEFERRED_ACTION, O.CREATE, "R4")),
    ("cerrá la persiana mañana", (F.DEFERRED_ACTION, O.CREATE, "R4")),
    # R6 routines, intercom and notifications.
    ("ejecutá la rutina buenas noches", (F.ROUTINE, O.EXECUTE, "R6")),
    ("poné la rutina de buenas noches", (F.ROUTINE, O.EXECUTE, "R6")),
    ("activá la rutina buenas noches", (F.ROUTINE, O.PROPOSE, "R6")),
    ("creá una rutina que cuando salga el sol prenda la luz", (F.ROUTINE, O.PROPOSE, "R6")),
    ("automatizá las luces del living", (F.ROUTINE, O.PROPOSE, "R6")),
    # P3F2-R1-Q4: a routine creation request claims the turn before any domotics producer.
    ("quiero crear una nueva rutina que apague las luces", (F.ROUTINE, O.PROPOSE, "R6")),
    ("armame una rutina que apague las luces", (F.ROUTINE, O.PROPOSE, "R6")),
    ("una rutina que apague las luces", (F.ROUTINE, O.PROPOSE, "R6")),
    ("anunciá que la comida está lista", (F.INTERCOM, O.EXECUTE, "R6")),
    ("avisá en la cocina que la comida está lista", (F.INTERCOM, O.EXECUTE, "R6")),
    ("decí en toda la casa que nos vamos", (F.INTERCOM, O.EXECUTE, "R6")),
    # R4 needs a domotics head verb: an announcement that mentions a deferred light stays intercom.
    ("anunciá que la luz se apaga en 10 minutos", (F.INTERCOM, O.EXECUTE, "R6")),
    # An infinitive is not an intercom request.
    ("avisar en la cocina que la comida está lista", None),
    ("anunciar en la cocina que la comida está lista", None),
    ("activá el no molestar", (F.NOTIFICATION, O.EXECUTE, "R6")),
    ("desactivá el no molestar", (F.NOTIFICATION, O.DISABLE, "R6")),
    ("solo urgentes esta noche", (F.NOTIFICATION, O.EXECUTE, "R6")),
    # R5 compounds.
    ("apagá la luz y pausá la música", (F.COMPOUND, O.EXECUTE, "R5")),
    ("cancelá el timer y el recordatorio", (F.COMPOUND, O.EXECUTE, "R5")),
    ("apagá la luz del escritorio y de la cocina", (F.COMPOUND, O.EXECUTE, "R5")),
    ("poné el aire a 22 y apagá la luz", (F.COMPOUND, O.EXECUTE, "R5")),
    ("poné un temporizador de 5 minutos y apagá la luz", (F.COMPOUND, O.EXECUTE, "R5")),
    ("prendé la luz y apagá la luz", (F.COMPOUND, O.CLARIFY, "R5")),
    ("apagá la luz en 10 minutos y pausá la música", (F.COMPOUND, O.CLARIFY, "R5")),
    ("apagá la luz y recordame sacar la basura", (F.COMPOUND, O.CLARIFY, "R5")),
    ("abrí la persiana y cerrá la persiana", (F.COMPOUND, O.CLARIFY, "R5")),
    ("prendé la luz y poné una alarma a las 7", (F.COMPOUND, O.CLARIFY, "R5")),
    # A household noun as the second target of a domotics/object verb is claimed (review F1),
    # in either word order.
    ("apagá la luz y el timer", (F.COMPOUND, O.EXECUTE, "R5")),
    ("apagá el timer y la luz", (F.COMPOUND, O.EXECUTE, "R5")),
    ("pará la música y el timer", (F.COMPOUND, O.EXECUTE, "R5")),
    ("prendé la luz y un timer de 5 minutos", (F.COMPOUND, O.CLARIFY, "R5")),
    ("apagá la tele y el temporizador", (F.COMPOUND, O.CLARIFY, "R5")),
    # Every 'y' of a target ellipsis splits (review F4): four rooms exceed compound_max_parts.
    ("encendé las luces del living y de la cocina y del baño y del cuarto", (F.COMPOUND, O.CLARIFY, "R5")),
    ("encendé las luces del living y de la cocina y del baño", (F.COMPOUND, O.EXECUTE, "R5")),
    # Unclaimed: domotics, music, generic cancel, questions, ambient talk.
    ("apagá la luz", None),
    ("prendé la luz del living", None),
    ("poné la luz cálida y suave", None),
    ("poné la luz al cuarenta y cinco por ciento", None),
    ("subí el volumen", None),
    ("pausá la música", None),
    ("poné música de la lista de reproducción", None),
    ("poné un tema en la lista de reproducción", None),
    ("poné la lista de compras", None),
    # A playlist is music, never the LIST family (review F2).
    ("pausá la lista de reproducción", None),
    ("pará la lista de reproducción", None),
    ("seguí la lista de reproducción", None),
    ("reanudá la lista de reproducción", None),
    ("cancelá la lista de reproducción", None),
    ("agregá esta canción a la lista de reproducción", None),
    ("qué hay en la lista de reproducción", None),
    ("decí algo gracioso", None),
    ("cancelá", None),
    ("pará", None),
    ("olvidate de eso", None),
    ("es un listado muy realista, cancelá", None),
    ("cuando juega Peñarol", None),
    ("cuándo es el cumpleaños de Ana", None),
    ("qué hora es", None),
    # A query head needs a query verb ('hay', 'tengo', 'quedan', ...).
    ("qué lindo timer", None),
    ("qué rutina más aburrida", None),
    ("para la lista de compras agregá leche", None),
    ("sincroniza los comandos", None),
    ("actualizá la lista de compras", None),
    ("apagá la luz que está prendida y cerrá la persiana", None),
    ("", None),
    ("nexa", None),
]


@pytest.mark.parametrize(("text", "expected"), PRECEDENCE, ids=[t or "<empty>" for t, _ in PRECEDENCE])
def test_precedence_table(text, expected):
    assert _shape(classify_household_intent(text)) == expected


@pytest.mark.parametrize(("text", "reason"), [
    ("recordale a Ana apagar la luz", "other_person"),
    ("recordame tomar la pastilla a las 8 y apagá la luz", "second_clause"),
    ("poné una alarma de 10 minutos", "duration"),
    ("poné una alarma a las 7", "clock_time"),
    ("poné una alarma", "no_time"),
    ("cancelá la alarma de las 7", "clock_alarm"),
    ("qué alarmas hay a las 7", "clock_alarm"),
    ("apagá la alarma de las 7", "clock_alarm"),
    ("pará el despertador de las 7", "clock_alarm"),
    ("desactivá el despertador de las 7", "clock_alarm"),
    ("prendé la luz y apagá la luz", "conflicting_parts"),
    ("abrí la persiana y cerrá la persiana", "conflicting_parts"),
    ("apagá la luz en 10 minutos y pausá la música", "deferred_part"),
    ("apagá la luz y recordame sacar la basura", "deferred_part"),
    ("prendé la luz y la tele", "unparsed_part"),
    ("prendé la luz y un timer de 5 minutos", "unparsed_part"),
    ("cerrá la persiana y prendé la luz con el timer", "unparsed_part"),  # an unclaimed part names a timer
    ("encendé las luces del living y de la cocina y del baño y del cuarto", "too_many_parts"),
    ("apagá la luz y pausá la música y cerrá la persiana y prendé el aire", "too_many_parts"),
])
def test_reason_codes(text, reason):
    assert classify_household_intent(text).reason == reason


@pytest.mark.parametrize("text", ["cancelá el recordatorio de las 7", "cancelá la alarma", "qué alarmas hay",
                                  "apagá la alarma", "pausá la alarma de las 7", "apagá la alarma de 10 minutos",
                                  "poné una alarma a las 7"])
def test_clock_alarm_reason_only_on_an_alarm_cancel_or_query(text):
    """T030-U1-Q8, T031-U2-Q1: the reason marks only a cancel/stop/disable/query of an alarm with a clock time."""
    assert classify_household_intent(text).reason != "clock_alarm"


def test_clock_alarm_complement_keeps_the_time_words():
    intent = classify_household_intent("nexa cancelá la alarma de las 7 de la mañana")
    assert (intent.family, intent.reason, intent.verb, intent.complement) == (
        F.REMINDER, "clock_alarm", "cancela", "de las 7 de la mañana")


@pytest.mark.parametrize(("text", "verb"), [
    ("nexa apagá la alarma de las 7 de la mañana", "apaga"), ("pará la alarma de las 7 de la mañana", "para"),
    ("desactivá el despertador de las 7 de la mañana", "desactiva"),
    ("bueno, detené la alarma de las 7 de la mañana", "detene"),
])
def test_a_stopped_or_disabled_clock_alarm_is_its_cancel(text, verb):
    """User answer T031-U2-Q1: the operation is CANCEL (the reminders take no stop); verb and time words kept."""
    intent = classify_household_intent(text)
    assert (intent.family, intent.operation, intent.rule, intent.reason, intent.verb, intent.complement) == (
        F.REMINDER, O.CANCEL, "R1", "clock_alarm", verb, "de las 7 de la mañana")


def test_governed_object_and_complement():
    """'cancelá el recordatorio de agregar leche a la lista' governs the reminder; the list is content."""
    intent = classify_household_intent("cancelá el recordatorio de agregar leche a la lista")
    assert (intent.family, intent.operation, intent.verb) == (F.REMINDER, O.CANCEL, "cancela")
    assert intent.complement == "de agregar leche a la lista"
    reminder = classify_household_intent("Nexa, recordame que mañana a las ocho viene el plomero")
    assert reminder.complement == "mañana a las ocho viene el plomero"
    assert classify_household_intent("haceme acordar de sacar la basura").complement == "sacar la basura"
    # A trailing 'por favor' is politeness, never part of the complement.
    assert classify_household_intent("poné un timer de 3 minutos por favor").complement == "de 3 minutos"
    item = classify_household_intent("agregá pan y leche a la lista de compras")
    assert (item.verb, item.complement) == ("agrega", "pan y leche a la lista de compras")


@pytest.mark.parametrize(("text", "security"), [
    ("apagá la luz del escritorio y abrí la cerradura", "abrí la cerradura"),
    ("abrí la puerta y prendé la luz del living", "abrí la puerta"),
    ("apagá la luz y cerrá la puerta con llave", "cerrá la puerta con llave"),
    ("apagá la luz y desactivá la alarma", "desactivá la alarma"),
    ("cerrá la persiana y abrí el portón", "abrí el portón"),
])
def test_security_order_is_a_parsed_part_its_own_part_refuses(text, security):
    """T037-U6 (seed C02-CMP-0004): a lock, door or alarm-panel order never makes the compound CLARIFY.

    It is an unclaimed part (no household family, never a domotics producer's: the part refuses it), and
    the other part keeps its own grammar parse.
    """
    from src.nlu.command_grammar import security_order
    intent = classify_household_intent(text)
    assert (intent.family, intent.operation, intent.reason) == (F.COMPOUND, O.EXECUTE, None)
    parts = {part.segment: part.intent for part in intent.parts}
    assert security in parts and parts[security] is None
    assert security_order(security) in ("lock", "alarm")
    assert [security_order(part.segment) for part in intent.parts].count(None) == 1


@pytest.mark.parametrize("text", [
    "apagá la luz y activá la alarma a las 11",
    "apagá la luz y abrí la puerta mañana a las 5",
    "prendé la luz del living y cerrá la puerta con llave en diez minutos",
    "activá la alarma más tarde y apagá la luz",
])
def test_security_part_with_a_time_makes_the_compound_clarify(text):
    """T037-U6-RV2: a deferred security part makes the whole turn CLARIFY (deferred_part), zero effects.

    The time may be meant for every part, so no part runs now (design compound_contract.not_admissible).
    """
    intent = classify_household_intent(text)
    assert (intent.family, intent.operation, intent.reason) == (F.COMPOUND, O.CLARIFY, "deferred_part")


def test_compound_parts_in_spoken_order():
    """R5 runs before R1/R2 (user answer Q21): each part is re-classified alone."""
    timer_and_light = classify_household_intent("poné un temporizador de 5 minutos y apagá la luz")
    assert [p.segment for p in timer_and_light.parts] == ["poné un temporizador de 5 minutos", "apagá la luz"]
    assert _shape(timer_and_light.parts[0].intent) == (F.TIMER, O.CREATE, "R2")
    assert timer_and_light.parts[1].intent is None
    objects = classify_household_intent("cancelá el timer y el recordatorio")
    assert [p.segment for p in objects.parts] == ["cancelá el timer", "cancelá el recordatorio"]
    assert [_shape(p.intent) for p in objects.parts] == [(F.TIMER, O.CANCEL, "R1"), (F.REMINDER, O.CANCEL, "R1")]
    rooms = classify_household_intent("apagá la luz del escritorio y de la cocina")
    assert [p.segment for p in rooms.parts] == ["apagá la luz del escritorio", "apagá la luz de la cocina"]
    rooms3 = classify_household_intent("encendé las luces del living y de la cocina y del baño")
    assert [p.segment for p in rooms3.parts] == ["encendé las luces del living", "encendé las luces de la cocina",
                                                 "encendé las luces del baño"]
    four = "encendé las luces del living y de la cocina y del baño y del cuarto"
    assert classify_household_intent(four, max_compound_parts=4).operation is O.EXECUTE
    assert len(classify_household_intent(four, max_compound_parts=4).parts) == 4
    three = classify_household_intent("apagá la luz y después cerrá la persiana y también pausá la música")
    assert [p.segment for p in three.parts] == ["apagá la luz", "cerrá la persiana", "pausá la música"]
    assert three.operation is O.EXECUTE


@pytest.mark.parametrize(("text", "segments"), [
    ("apagá la luz y el timer", ["apagá la luz", "apagá el timer"]),
    ("apagá el timer y la luz", ["apagá el timer", "apagá la luz"]),
    ("pará la música y el timer", ["pará la música", "pará el timer"]),
])
def test_household_noun_target_is_its_own_claimed_part(text, segments):
    """Review F1: the household part is never dropped behind a domotics or music part."""
    intent = classify_household_intent(text)
    assert [p.segment for p in intent.parts] == segments
    timer = [p.intent for p in intent.parts if p.intent is not None]
    assert [_shape(t) for t in timer] == [(F.TIMER, O.STOP, "R1")]


# P3F2-R1-Q4 (user answer 2026-11-17): a request to create a routine, in its rioplatense
# variants, is a routine proposal; its content ("que apague las luces") is never a domotics command.
ROUTINE_CREATION_REQUESTS = [
    "quiero crear una nueva rutina que apague las luces",
    "quiero crear una rutina que apague las luces",
    "quisiera crear una rutina que apague las luces del living",
    "quería crear una rutina que apague las luces",
    "creá una rutina para apagar las luces a la noche",
    "crea una rutina que apague las luces",
    "creame una automatización que apague las luces",
    "armame una rutina que apague las luces",
    "armá una rutina que prenda la luz del living",
    "haceme una rutina que apague todas las luces",
    "hacé una rutina que cierre las persianas",
    "programame una rutina que apague las luces",
    "generá una rutina que apague las luces",
    "agregá una rutina que apague las luces",
    "sumá una rutina que prenda la luz",
    "nueva rutina que apague las luces",
    "una rutina que apague las luces",
    "nueva automatización que apague las luces",
    "necesito una rutina que apague las luces",
    "me gustaría crear una rutina que apague las luces",
    "quiero que me armes una rutina que apague las luces",
    "quiero que crees una rutina que apague las luces",
    "te pido que crees una rutina que apague las luces",
    "podés crear una rutina que apague las luces",
    "me podés armar una rutina que apague las luces",
    "¿me armás una rutina que apague las luces?",
    "¿me harías una rutina nueva que apague las luces?",
    "vamos a crear una rutina que apague las luces",
    "quiero armar una rutina para apagar las luces",
    "quiero hacer una rutina que apague la luz",
    "quiero configurar una rutina que apague las luces",
    "quiero agregar una rutina que apague las luces",
    "quiero activar una nueva rutina que apague las luces",
    "quiero una rutina nueva que apague las luces",
    "nexa quiero crear una nueva rutina que apague las luces",
    "che armame una rutina que apague las luces a las once",
    "dale creá una rutina que apague las luces",
    "a ver, armame otra rutina que apague las luces",
    "quiero crear una rutina que apague las luces y cierre la persiana",
    "armame una rutina que apague la luz del living y la de la cocina",
    "haceme una rutina para apagar la luz del living y de la cocina",
    "nueva rutina para la noche",
    "nexa nueva automatización de luces",
    "creá una rutina que cuando salga el sol prenda la luz",
    "automatizá las luces del living",
    # A creation verb needs no 'una' ('creá la rutina ...'); 'nueva' may follow the noun.
    "creá la rutina de la mañana que apague las luces",
    "rutina nueva que apague las luces",
    # Q4-RV-2: hesitations and a leading 'y' kept by speech recognition, more creation verbs, 'regla'.
    "y creá una rutina que apague las luces",
    "eh creá una rutina que apague las luces",
    "mmm creá una rutina que apague las luces",
    "este, armame una rutina que apague las luces",
    "inventá una rutina que apague las luces",
    "diseñá una rutina que apague las luces",
    "escribí una rutina que apague las luces",
    "inventame la rutina de la noche que apague las luces",
    "quiero inventar una rutina que apague las luces",
    "creá una regla que apague las luces",
    "quiero crear reglas que apaguen las luces",
    # Q4-RV-4: 'otra' with a verb that can also run a routine, the plural noun, the 'bueno' filler.
    "hacé otra rutina que apague las luces",
    "armame rutinas que apaguen las luces",
    "bueno, creá una rutina que apague las luces",
    # Two determiners before the noun.
    "armame esta otra rutina que apague las luces",
]


@pytest.mark.parametrize("text", ROUTINE_CREATION_REQUESTS)
def test_routine_creation_request_is_a_routine_proposal(text):
    intent = classify_household_intent(text, wake_words=("nexa",))
    assert (intent.family, intent.operation, intent.rule, intent.reason) == (F.ROUTINE, O.PROPOSE, "R6",
                                                                             "new_routine")
    assert intent.parts == ()


@pytest.mark.parametrize("text", [
    "quiero crear una nueva rutina que apague las luces", "armame una rutina que apague las luces",
    "una rutina que apague las luces", "quiero que me armes una rutina que apague las luces",
    "creá una rutina para apagar las luces a la noche",
])
def test_routine_content_alone_would_be_a_grammar_command(text):
    """Why the claim matters: unclaimed, the grammar reads the routine's content as an immediate turn_off."""
    parsed = parse_command(text)
    assert (parsed.intent, parsed.domain, parsed.quality) == ("turn_off", "light", "full")
    assert non_immediate_action_reason(text) is None


@pytest.mark.parametrize(("text", "expected"), [
    ("activá la rutina buenas noches", (F.ROUTINE, O.PROPOSE, "R6", None)),
    ("programá la rutina de la noche", (F.ROUTINE, O.PROPOSE, "R6", None)),
    ("hacé la rutina de la noche", (F.ROUTINE, O.PROPOSE, "R6", None)),
    ("ejecutá la rutina buenas noches", (F.ROUTINE, O.EXECUTE, "R6", None)),
    ("poné la rutina de buenas noches", (F.ROUTINE, O.EXECUTE, "R6", None)),
    ("qué rutinas tengo", (F.ROUTINE, O.QUERY, "R1", None)),
    ("borrá la rutina buenas noches", (F.ROUTINE, O.CANCEL, "R1", None)),
    ("creá una rutina para la noche y apagá las luces", (F.COMPOUND, O.EXECUTE, "R5", None)),
    ("la rutina de la noche", None),
    ("quiero la rutina de la noche", None),
    ("rutina", None),
    ("quiero apagar las luces", None),
    ("quiero que apagues las luces", None),
    # The unlisted-verb slot never stands for a known command verb, and needs 'una'/'nueva'.
    ("quiero borrar una rutina", None),
    ("quiero ejecutar una nueva rutina", None),
    ("me encanta la rutina de la noche", None),
    ("salió una nueva rutina de ejercicios", None),
    ("no sé si crear rutina para la noche", None),
    ("pensaba en una nueva rutina para la noche", None),
    ("qué rutina más aburrida", None),
    ("haceme una lista", None),
    ("una canción", None),
    # Q4-RV-3: a conjugated command verb or a look verb is not the unlisted verb, and a trailing
    # 'nueva' after a verb that can run a routine (or the unlisted verb) is not a creation.
    ("activá la rutina nueva", (F.ROUTINE, O.PROPOSE, "R6", None)),
    ("me ponés una rutina relajante", None),
    ("quiero que me pongas una rutina relajante", None),
    ("me ponés la rutina nueva", None),
    ("quiero ver las rutinas nuevas", None),
    ("quiero ver una rutina nueva", None),
    ("me harías la rutina nueva", None),
    # Q4-RV-4: only a desire head ('quiero que ...') skips 'que'; 'a ver qué ...' is a question.
    ("a ver qué rutinas nuevas hay", None),
    # A known command verb with an enclitic ('ponerme') is not the unlisted verb either.
    ("quiero ponerme una rutina de ejercicios", None),
])
def test_routine_creation_leaves_other_phrases_alone(text, expected):
    intent = classify_household_intent(text)
    assert (None if intent is None else (intent.family, intent.operation, intent.rule, intent.reason)) == expected


def test_routine_creation_complement_and_verb():
    intent = classify_household_intent("Nexa, quiero crear una nueva rutina que apague las luces", wake_words=("nexa",))
    assert (intent.verb, intent.complement) == ("crear", "que apague las luces")
    assert (classify_household_intent("armame una rutina nueva para la noche").verb,
            classify_household_intent("armame una rutina nueva para la noche").complement) == ("arma", "para la noche")
    verbless = classify_household_intent("una automatización que cierre las persianas")
    assert (verbless.verb, verbless.complement) == ("", "que cierre las persianas")


def test_compound_limit_comes_from_settings():
    text = "apagá la luz y cerrá la persiana y pausá la música"
    assert classify_household_intent(text, max_compound_parts=3).operation is O.EXECUTE
    assert classify_household_intent(text, max_compound_parts=2).reason == "too_many_parts"
    classifier = HouseholdIntentClassifier(HouseholdIntentSettings(compound_max_parts=2))
    assert classifier.classify(text).reason == "too_many_parts"


@pytest.mark.parametrize("text", [
    "poné la luz cálida y suave",                          # attribute pair
    "poné la luz al cuarenta y cinco por ciento",          # number words
    "agregá pan y leche a la lista de compras",            # list item enumeration
    "recordame apagar la luz y cerrar la puerta",          # R3 head: complement is content
    "apagá la luz que está prendida y cerrá la persiana",  # relative clause is content
    "creá una rutina que apague la luz y cierre la persiana",
])
def test_compound_exemptions(text):
    intent = classify_household_intent(text)
    assert intent is None or intent.family is not F.COMPOUND


def test_normalization_strips_wake_words_politeness_and_accents():
    for text in ("nexa poné un timer de 3 minutos", "Nexa, por favor, poné un timer de 3 minutos",
                 "che poné un tímer de 3 minutos por favor", "PONÉ UN TIMER DE 3 MINUTOS",
                 "¡Poné un timer de 3 minutos!", "porfa poné un timer de 3 minutos porfa"):
        assert _shape(classify_household_intent(text)) == (F.TIMER, O.CREATE, "R2"), text
    # A configured wake word that the grammar does not know is stripped too.
    assert classify_household_intent("jarvis poné un timer de 3 minutos") is None
    configured = HouseholdIntentClassifier(HouseholdIntentSettings.from_config({}, wake_words=["Jarvis"]))
    assert _shape(configured.classify("jarvis poné un timer de 3 minutos")) == (F.TIMER, O.CREATE, "R2")


@pytest.mark.parametrize(("text", "expected"), [
    ("bueno, cancelá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("bue cancelá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("dale cancelá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("no, cancelá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("Nexa, no. Borrá el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("cancelá ya el recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("olvidate ya del recordatorio de tomar agua", (F.REMINDER, O.CANCEL, "R1")),
    ("dale, poné un timer de 3 minutos", (F.TIMER, O.CREATE, "R2")),
    # A "no" without a pause is a negation; "ya" before the verb is not a filler.
    ("no cancelar el recordatorio de tomar agua", None),
    ("no cancelá el recordatorio de tomar agua", None),
    ("no cancele el recordatorio de tomar agua", None),
    ("¿ya cancelaron el recordatorio de tomar agua?", None),
    ("olvidate ya mismo del recordatorio de tomar agua", None),
    ("olvidate ya de tomar agua", None),  # as "olvidate de tomar agua": no reminder noun
    ("¿cancelaste el recordatorio de tomar agua?", None),
    ("quiero un recordatorio para cancelar la suscripción", None),
    ("no me dejes olvidar sacar la basura", (F.REMINDER, O.CREATE, "R3")),
])
def test_fillers_a_paused_no_and_ya_after_a_cancel_verb(text, expected):
    """Review T032-U1-R3: real imperatives are claimed; questions and negations are not."""
    intent = classify_household_intent(text)
    assert _shape(intent) == expected
    if expected is not None and expected[1] is O.CANCEL:
        assert intent.complement == "de tomar agua"


@pytest.mark.parametrize("text", [
    "hola, cancelá el recordatorio de tomar agua", "porfa cancelá el recordatorio de tomar agua",
    "dale, cancelá el recordatorio de tomar agua", "cancelá el recordatorio de tomar agua, porfa, por favor",
    "cancelá el recordatorio de tomar agua por favor porfa", "cancelá el recordatorio de tomar agua porfa por favor",
    "por favor, cancelá el recordatorio de tomar agua",
])
def test_trailing_fillers_in_any_order(text):
    """Review BFR-1: a trailing 'porfa' or 'por favor', in any order, is not part of the complement."""
    intent = classify_household_intent(text)
    assert _shape(intent) == (F.REMINDER, O.CANCEL, "R1")
    assert intent.complement == "de tomar agua"


@pytest.mark.parametrize(("text", "complement"), [
    ("cancelá el recordatorio de decirle que dale", "de decirle que dale"),
    ("cancelá el recordatorio de decirle que, dale", "de decirle que dale"),
    ("cancelá el recordatorio de llamar a Dale", "de llamar a dale"),
    ("cancelá el recordatorio de tomar agua, dale", "de tomar agua dale"),
    ("cancelá el recordatorio de tomar agua dale", "de tomar agua dale"),
    ("cancelá el recordatorio de tomar agua, dale, porfa", "de tomar agua dale"),
    ("cancelá el recordatorio de tomar agua, por favor, dale", "de tomar agua por favor dale"),
])
def test_a_trailing_dale_stays_in_the_complement(text, complement):
    """Review BFR1-R1: a trailing 'dale' may end the reminder's name, so the classifier keeps it.

    Dropping it turned 'decirle que dale' into 'decirle que', which cancelled
    'decirle que no' directly instead of asking (T032-NUM-FV-1).
    """
    intent = classify_household_intent(text)
    assert _shape(intent) == (F.REMINDER, O.CANCEL, "R1")
    assert intent.complement == complement


@pytest.mark.parametrize(("text", "kept"), [
    ("Nexa, bueno, el segundo, porfa", ("el", "segundo")), ("bue, el segundo", ("el", "segundo")),
    ("hola, el segundo", ("el", "segundo")), ("porfa el segundo", ("el", "segundo")),
    ("no, el segundo", ("el", "segundo")), ("Nexa, no. El segundo", ("el", "segundo")),
    ("no el segundo", ("no", "el", "segundo")), ("el segundo, no", ("el", "segundo", "no")),
    ("che oye ey podés el segundo por favor", ("el", "segundo")), ("el segundo porfa por favor porfa", ("el", "segundo")),
    # Review BFR1-R1: a trailing 'dale' may be part of a name and is kept.
    ("el segundo, dale", ("el", "segundo", "dale")), ("decirle que dale", ("decirle", "que", "dale")),
    ("el segundo porfa por favor dale", ("el", "segundo", "porfa", "por", "favor", "dale")),
    ("dale", ()), ("", ()), ("el número dos", ("el", "numero", "dos")),
    # At most six lead words are dropped.
    ("dale dale dale dale dale dale el segundo", ("el", "segundo")),
    ("dale dale dale dale dale dale dale el segundo", ("dale", "el", "segundo")),
])
def test_strip_fillers_is_the_shared_filler_rule(text, kept):
    """Review BFR-1: the fillers the classifier ignores, as folded tokens."""
    assert strip_fillers(text) == kept


def test_strip_fillers_takes_configured_wake_words_and_extra_leads():
    assert strip_fillers("jarvis, el segundo") == ("jarvis", "el", "segundo")
    assert strip_fillers("Jarvis, el segundo", ["Jarvis"]) == ("el", "segundo")
    assert strip_fillers("sí, borrá el segundo", extra_leads=frozenset({"si", "borra"})) == ("el", "segundo")
    assert strip_fillers("sí, borrá el segundo") == ("si", "borra", "el", "segundo")
    # Review BFR1-R1: a caller with a closed vocabulary drops further trailing words, in any order.
    assert strip_fillers("el segundo, dale, porfa", extra_trails=frozenset({"dale"})) == ("el", "segundo")
    assert strip_fillers("el segundo porfa dale", extra_trails=frozenset({"dale"})) == ("el", "segundo")
    assert strip_fillers("el segundo, dale, porfa") == ("el", "segundo", "dale")


FILLER_SHAPES = [
    "Nexa, bueno, el segundo, porfa", "bue, el segundo", "hola, el segundo", "porfa el segundo", "no, el segundo",
    "Nexa, no. El segundo", "no el segundo", "el segundo, no", "che oye ey podés el segundo por favor",
    "el segundo porfa por favor porfa", "el segundo, dale", "decirle que dale", "dale", "", "el número dos",
    "dale dale dale dale dale dale el segundo", "dale dale dale dale dale dale dale el segundo",
    "bueno, recordame tomar agua a las 5, porfa", "no, recordame tomar agua", "no recordame tomar agua",
]


@pytest.mark.parametrize("text", FILLER_SHAPES)
def test_filler_span_keeps_exactly_the_words_strip_fillers_keeps(text):
    """Unit BFR1-Q1: the character range of the reminder request is the shared filler rule, not a second one."""
    from src.nlu.household_intents import _tokens
    start, end = filler_span(text, ("nexa",))
    assert _tokens(text[start:end]).folded == strip_fillers(text, ("nexa",))


@pytest.mark.parametrize(("text", "heard"), [
    ("bueno, recordame tomar agua a las 5, dale", "recordame tomar agua a las 5"),
    ("Nexa, no, recordame tomar agua a las 5. Dale.", "recordame tomar agua a las 5"),
    ("recordame tomar agua a las 5, dale, porfa", "recordame tomar agua a las 5"),
    ("recordame tomar agua a las 5, porfa, dale", "recordame tomar agua a las 5"),
    ("recordame tomar agua a las 5 por favor, dale", "recordame tomar agua a las 5"),
    ("recordame tomar agua a las 5; dale", "recordame tomar agua a las 5"),
    # Without a pause a trailing 'dale' may end the text (review BFR1-R1): kept.
    ("recordame tomar agua a las 5 dale", "recordame tomar agua a las 5 dale"),
    ("recordame a las 5 decirle que dale", "recordame a las 5 decirle que dale"),
    ("recordame a las 5 decirle que dale, porfa", "recordame a las 5 decirle que dale"),
    ("recordame decirle que dale, dale", "recordame decirle que dale"),
    ("no recordame tomar agua, dale", "no recordame tomar agua"),
    # A colon usually opens quoted speech: the 'dale' after it is the text (review BH-R2).
    ("recordame a las 5 decirle a Juan: dale", "recordame a las 5 decirle a Juan: dale"),
    ("recordame a las 5 decirle a Juan: dale, porfa", "recordame a las 5 decirle a Juan: dale"),
    ("recordame a las 5 decirle a Juan: dale, dale", "recordame a las 5 decirle a Juan: dale"),
    ("dale, dale", ""),
])
def test_filler_span_drops_a_trailing_dale_only_after_a_pause(text, heard):
    start, end = filler_span(text, ("nexa",), paused_trails=frozenset({"dale"}))
    assert text[start:end] == heard


def test_filler_span_without_paused_trails_keeps_a_paused_dale():
    text = "recordame tomar agua a las 5, dale"
    assert filler_span(text) == (0, len(text))


def test_filler_span_strips_nothing_when_lowercasing_moves_the_offsets():
    """'İ' lowercases to two characters: offsets into the lowered text would cut the wrong words."""
    text = "bueno, recordame İr al banco a las 5, dale"
    assert filler_span(text, paused_trails=frozenset({"dale"})) == (0, len(text))


@pytest.mark.parametrize("text", [
    "poné ya un timer de 3 minutos", "apagá ya la luz", "pausá ya la música",
])
def test_ya_after_a_verb_that_is_not_a_cancel_is_left_as_before(text):
    """Review T032-U1-RR-F3 (RR08): 'ya' is dropped only after a cancel verb.

    These are unclaimed at base 1541909a; stripping 'ya' after any verb
    would claim them and widen precedence.
    """
    assert classify_household_intent(text) is None


def test_unaccented_para_is_a_clause_head_only_when_spoken_as_a_verb():
    """'pará' heads a clause inside a turn; the preposition 'para' does not."""
    assert classify_household_intent("apagá la luz y pará la música").family is F.COMPOUND
    assert classify_household_intent("apagá la luz y para la noche") is None


def test_object_nouns_need_word_boundaries():
    for text in ("un listado muy realista, cancelá", "algo rutinario, pará", "los temporizadorcitos, cancelá"):
        assert classify_household_intent(text) is None, text


@pytest.mark.parametrize("head", [
    "recordame", "recordáme", "recuérdame", "recuerdame", "recordanos", "acordame", "acuérdame",
    "haceme acordar de", "hacenos acordar de", "avisame", "avísame", "avisanos", "no me dejes olvidar",
    "me podés recordar", "recordale a Ana", "avisales a todos",
])
def test_reminder_heads_agree_with_the_voice_reminder_parser(head):
    """One vocabulary: the classifier's R3 heads are the T031-U1 parser's reminder verbs."""
    text = f"{head} sacar la basura en 5 minutos"
    now = datetime(2026, 11, 16, 10, 0, tzinfo=ZoneInfo("America/Montevideo"))
    assert starts_with_reminder_request(text)
    assert parse_reminder_request(text, now) is not None
    assert classify_household_intent(text).family is F.REMINDER


@pytest.mark.parametrize("text", [
    "recordame apagar la luz", "nexa recordame apagar la luz", "haceme acordar de apagar la luz",
    "recordale a Ana apagar la luz", "recordame prender la luz", "recuérdame abrir la persiana",
])
def test_grammar_reminder_request_blocker(text):
    """Defence in depth: the grammar never turns a reminder's content into an immediate action."""
    assert non_immediate_action_reason(text) == "non_immediate:reminder_request"
    assert parse_command(text).quality == "none"
    assert not llm_intent_evidenced("turn_off", text) and not llm_intent_evidenced("turn_on", text)


@pytest.mark.parametrize("text", ["apagá la luz", "prendé la luz del living", "recordá que apagué la luz"])
def test_grammar_blocker_leaves_other_phrases_alone(text):
    assert non_immediate_action_reason(text) != "non_immediate:reminder_request"


def test_settings_from_config_and_validation():
    settings = HouseholdIntentSettings.from_config(None, wake_words=["Nexa", " Kaza "])
    assert (settings.unvetoed_max_words, settings.compound_max_parts, settings.compound_execute) == (6, 3, True)
    assert settings.wake_words == ("nexa", "kaza")
    custom = HouseholdIntentSettings.from_config({"unvetoed_max_words": 9, "compound_max_parts": 4})
    assert (custom.unvetoed_max_words, custom.compound_max_parts) == (9, 4)
    for bad in ({"unvetoed_max_words": 0}, {"unvetoed_max_words": "6"}, {"compound_max_parts": 1},
                {"compound_execute": "false"}, {"unvetoed_max_words": True}):
        with pytest.raises(ValueError):
            HouseholdIntentSettings.from_config(bad)
    # T037-U6 landed: execution is on by default and false stays available as the kill switch.
    assert HouseholdIntentSettings.from_config({"compound_execute": False}).compound_execute is False
    assert HouseholdIntentSettings.from_config({"compound_execute": True}).compound_execute is True
    with pytest.raises(ValueError):
        HouseholdIntentSettings.from_config(["not", "a", "mapping"])


def test_settings_file_declares_the_section():
    from pathlib import Path

    import yaml
    config = yaml.safe_load(Path("config/settings.yaml").read_text())
    settings = HouseholdIntentSettings.from_config(config["household_intents"])
    assert (settings.unvetoed_max_words, settings.compound_max_parts, settings.compound_execute) == (6, 3, True)


def test_classifier_never_logs(caplog):
    caplog.set_level(logging.DEBUG)
    for text, _ in PRECEDENCE:
        classify_household_intent(text)
    assert not [r for r in caplog.records if r.name == "src.nlu.household_intents"]


def test_summary_carries_no_user_text():
    summary = classify_household_intent("recordame tomar la pastilla").summary()
    assert summary == {"family": "reminder", "operation": "create", "rule": "R3", "reason": None, "parts": 0}


def test_microbenchmark_under_one_millisecond():
    """Design latency: S5/D1 are regex only (< 1 ms per call); measured, labelled simulation."""
    texts = [text for text, _ in PRECEDENCE if text]
    for text in texts:
        classify_household_intent(text)  # warm caches and lazy imports
    samples = []
    for _ in range(5):
        for text in texts:
            started = time.perf_counter()
            classify_household_intent(text)
            samples.append(time.perf_counter() - started)
    assert statistics.median(samples) < 0.001
    assert statistics.mean(samples) < 0.001


COMPOUNDS = ("apagá la luz del escritorio y de la cocina", "apagá la luz y pausá la música",
             "encendé las luces del living y de la cocina y del baño", "pará la música y el timer")


@pytest.mark.parametrize("text", COMPOUNDS)
def test_compound_per_call_under_one_millisecond(text):
    """Review F7: the corpus average hid the compound cost; each compound call is bounded on its own."""
    classify_household_intent(text)
    samples = []
    for _ in range(50):
        started = time.perf_counter()
        classify_household_intent(text)
        samples.append(time.perf_counter() - started)
    assert statistics.median(samples) < 0.001


def test_one_turn_is_classified_once_for_the_gate_router_and_dispatcher():
    """Review F7: the shared instance memoizes, so a turn's three classifications cost one."""
    for text in COMPOUNDS:
        classifier = HouseholdIntentClassifier()
        first = classifier.classify(text)
        assert classifier.classify(text) is first and classifier.classify(text) is first
        assert first == classify_household_intent(text)
    turns = []
    for _ in range(20):
        for text in COMPOUNDS:
            classifier = HouseholdIntentClassifier()
            started = time.perf_counter()
            for _caller in ("gate", "router", "dispatcher"):
                classifier.classify(text)
            turns.append(time.perf_counter() - started)
    assert statistics.median(turns) < 0.001


def test_memo_is_bounded_and_keeps_settings():
    classifier = HouseholdIntentClassifier(HouseholdIntentSettings(compound_max_parts=2))
    for n in range(20):
        classifier.classify(f"poné un timer de {n} minutos")
    assert classifier._memo.cache_info().currsize <= 8
    assert classifier.classify("apagá la luz y cerrá la persiana y pausá la música").reason == "too_many_parts"


# --------------------------------------------------------------------------
# C02 classifier-layer entries (design seed_entries; every counter is zero:
# the classifier is pure).
# --------------------------------------------------------------------------

def _cls(id_, text, family, classification, *, unit="P3-F2", effect_class="wrong_family", note=""):
    return C02Entry(id_, text, "classifier", family, effect_class, CLASSIFIER, None, CLASSIFIER, CLASSIFIER,
                    "none", effects(), classification=classification, unit=unit, note=note)


C02_CLASSIFIER = (
    _cls("C02-TIM-0002", "poneme un tímer de media hora", "timer",
         {"family": "timer", "operation": "create", "rule": "R2"}, unit="T030-U1"),
    _cls("C02-ALM-0001", "poné una alarma de 10 minutos", "timer",
         {"family": "timer", "operation": "create", "rule": "R3a"}, effect_class="unauthorized_effect"),
    _cls("C02-ALM-0002", "poné una alarma a las 7", "reminder",
         {"family": "reminder", "operation": "create", "rule": "R3a"}, effect_class="unauthorized_effect"),
    _cls("C02-ALM-0004", "poné una alarma", "timer",
         {"family": "timer", "operation": "clarify", "rule": "R3a"}),
    _cls("C02-ALM-0005", "cancelá la alarma", "timer",
         {"family": "timer", "operation": "cancel", "rule": "R1"}, unit="T030-U1",
         note="R3: pairs C02-ALM-0001; before T030-U1 R3 it was unclaimed and reached home control"),
    _cls("C02-ALM-0006", "apagá la alarma", "timer",
         {"family": "timer", "operation": "stop", "rule": "R1"}, unit="T030-U1", effect_class="unauthorized_effect",
         note="R3: pairs C02-ALM-0001; never a switch or alarm entity"),
    _cls("C02-ALM-0201", "cancelá la alarma de las 7", "reminder",
         {"family": "reminder", "operation": "cancel", "rule": "R1"}, unit="T031-U2", effect_class="wrong_family",
         note="user answer T030-U1-Q8 (A): the alarm with a clock time is a reminder (R3a), never a timer"),
    _cls("C02-ALM-0202", "qué alarmas hay a las 7", "reminder",
         {"family": "reminder", "operation": "query", "rule": "R1"}, unit="T031-U2",
         note="user answer T030-U1-Q8 (A)"),
    _cls("C02-ALM-0211", "poné una alarma a las 19 horas", "reminder",
         {"family": "reminder", "operation": "create", "rule": "R3a"}, unit="T031-U2", effect_class="wrong_family",
         note="review T031-U2-RV2: 'a las N horas' is a clock time (user answer Q4), never an N-hour timer"),
    _cls("C02-ALM-0212", "poné una alarma de 19 horas", "timer",
         {"family": "timer", "operation": "create", "rule": "R3a"}, unit="T031-U2", effect_class="wrong_family",
         note="review T031-U2-RV2: without the article 'N horas' stays a duration"),
    _cls("C02-ALM-0215", "apagá la alarma de las 7", "reminder",
         {"family": "reminder", "operation": "cancel", "rule": "R1"}, unit="T031-U2-Q1", effect_class="wrong_family",
         note="user answer T031-U2-Q1: turning off the alarm set for 7 is that reminder's cancel (numbered list "
              "first); it never stops a running timer in its place"),
    _cls("C02-ALM-0216", "pará el despertador de las 7", "reminder",
         {"family": "reminder", "operation": "cancel", "rule": "R1"}, unit="T031-U2-Q1", effect_class="wrong_family",
         note="user answer T031-U2-Q1"),
    _cls("C02-ALM-0217", "desactivá la alarma de las 7", "special", None, unit="BH-R1-Q1",
         effect_class="unauthorized_effect",
         note="user answer BH-R1-Q1: a security alarm verb is never claimed, with a clock time too (design R3a), "
              "like 'desactivá la alarma' alone (C02-ALM-0007)"),
    _cls("C02-ALM-0221", "desactivá el despertador de las 7", "reminder",
         {"family": "reminder", "operation": "cancel", "rule": "R1"}, unit="BH-R1-Q1", effect_class="wrong_family",
         note="user answer BH-R1-Q1 (B): 'despertador' is no security alarm: with a clock time it stays the "
              "reminder's cancel (numbered list first), never a timer disable"),
    _cls("C02-ALM-0218", "apagá la alarma de 10 minutos", "timer",
         {"family": "timer", "operation": "stop", "rule": "R1"}, unit="T031-U2-Q1", effect_class="wrong_family",
         note="T030-U1: an alarm with a duration stays a timer"),
    _cls("C02-ALM-0007", "desactivá la alarma", "special", None, unit="T030-U1", effect_class="unauthorized_effect",
         note="security alarm verb: never claimed (design R3a), unlike 'desactivá el despertador'"),
    _cls("C02-ALM-0104", "activá la alarma", "special", None, effect_class="unauthorized_effect",
         note="seed C02-ALM-0003 at classifier layer: never claimed; locks/alarms stay on the blocked path"),
    _cls("C02-CMP-0002", "poné la luz cálida y suave", "compound", None, unit="T037-U5"),
    _cls("C02-CMP-0003", "cancelá el timer y el recordatorio", "compound",
         {"family": "compound", "operation": "execute", "rule": "R5",
          "parts": [{"family": "timer", "operation": "cancel", "rule": "R1"},
                    {"family": "reminder", "operation": "cancel", "rule": "R1"}]}),
    _cls("C02-CMP-0007", "prendé la luz y apagá la luz", "compound",
         {"family": "compound", "operation": "clarify", "rule": "R5"}, unit="T037-U5"),
    _cls("C02-CMP-0008", "agregá pan y leche a la lista de compras", "compound",
         {"family": "list", "operation": "add_item", "rule": "R2"}, unit="T037-U5"),
    _cls("C02-CMP-0009", "poné un temporizador de 5 minutos y apagá la luz", "compound",
         {"family": "compound", "operation": "execute", "rule": "R5",
          "parts": [{"family": "timer", "operation": "create", "rule": "R2"},
                    {"family": None, "operation": None, "rule": None}]}, unit="T037-U5"),
    _cls("C02-CMP-0010", "apagá la luz en 10 minutos y pausá la música", "compound",
         {"family": "compound", "operation": "clarify", "rule": "R5"}, unit="T037-U5",
         effect_class="unauthorized_effect"),
    _cls("C02-MEM-0101", "acordate que mañana a las 8 viene el plomero", "memory", None,
         note="seed C02-REM-0012 at classifier layer: a U3b memory proposal, never claimed as a reminder"),
    _cls("C02-RTN-0105", "cuando juega Peñarol", "routine", None, unit="T033-U1",
         note="bare 'cuando' is not a routine"),
)


def _as_classification(intent):
    if intent is None:
        return None
    shape = {"family": intent.family.value, "operation": intent.operation.value, "rule": intent.rule}
    if intent.parts:
        shape["parts"] = [
            {"family": p.intent.family.value, "operation": p.intent.operation.value, "rule": p.intent.rule}
            if p.intent is not None else {"family": None, "operation": None, "rule": None} for p in intent.parts]
    return shape


@pytest.mark.parametrize("entry", C02_CLASSIFIER, ids=lambda e: e.id)
def test_c02_classifier_entries(entry):
    assert entry.effects == effects()
    assert (entry.task_status, entry.persistence, entry.path) == (CLASSIFIER, CLASSIFIER, None)
    assert _as_classification(classify_household_intent(entry.text)) == entry.classification
