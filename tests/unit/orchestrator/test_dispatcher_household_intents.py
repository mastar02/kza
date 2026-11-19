"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

import asyncio
import sqlite3
import time

import pytest

from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.orchestrator.dispatcher import MultiUserOrchestrator, PathType
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderScope
from tests.mocks.c02_matrix import C02Entry, effects
from tests.mocks.household_dispatch import CountedLeaves

ZERO = effects()
ROUTINE_CREATE = "Todavía no creo rutinas por voz. Podés crearlas desde la web."


def _d(id_, text, family, effect_class, outcome, path, status, persistence, spoken, *, counts=ZERO,
       conditions=None, unit="P3-F2", finding=None, seed=None, note="", actor="household_unknown_voice"):
    return C02Entry(id_, text, "dispatcher", family, effect_class, outcome, path, status, persistence, spoken,
                    counts, conditions or {}, actor=actor, unit=unit, finding=finding, seed=seed, note=note)


def _refused(id_, text, family, path="fast_reminder", *, sf=None, state=None, effect_class="unauthorized_effect",
             status="unsupported", spoken="refusal", **kw):
    if kw.get("note") == CREATE_ASKS:
        status, spoken = "needs_input", "clarification"
    conditions = {k: v for k, v in (("service_filter", sf), ("state", state)) if v}
    return _d(id_, text, family, effect_class, status, path, status, "not_written", spoken,
              conditions=conditions, **kw)


def _timer(id_, text, outcome, persistence, spoken, *, counts=ZERO, state=None, sf=None,
           effect_class="unauthorized_effect", seed=None, note="", unit="T030-U1"):
    """A T030 timer entry on the dispatcher layer (a real, started NamedTimerManager)."""
    conditions = {k: v for k, v in (("service_filter", sf), ("state", state)) if v}
    return _d(id_, text, "timer", effect_class, outcome, "fast_timer", outcome, persistence, spoken, counts=counts,
              conditions=conditions, unit=unit, seed=seed, note=note)


CREATED = effects(timer_delta=1)
CHANGED = effects(timer_transitions=1)

CREATE_ASKS = ("T031-U2 closes the P3-F2 interim UNSUPPORTED: creation exists, and a request it cannot schedule "
               "without a guess asks for the whole request again (NEEDS_INPUT, the seed's clarification); "
               "every counter as the seed")

# 'recordame/recuérdame' x {apagar la luz, poner música, agregar X a la lista, abrir la persiana}
# x service_filter {turn_on, turn_off}: zero HA sends, music calls or list mutations.
_REMINDER_GRID = [
    _refused(f"C02-REM-{101 + i:04d}", f"{verb} {complement}", "reminder", sf=sf, state="shared_list_compras",
             finding="CORE-02", note=CREATE_ASKS)
    for i, (verb, complement, sf) in enumerate(
        (verb, complement, sf)
        for verb in ("recordame", "recuérdame")
        for complement in ("apagar la luz", "poner música", "agregar leche a la lista de compras",
                           "abrir la persiana")
        for sf in ("turn_on", "turn_off"))
]

C02_DISPATCHER = tuple(_REMINDER_GRID) + (
    _refused("C02-REM-0002", "recordame poner música a las ocho", "reminder", seed="C02-REM-0002", note=CREATE_ASKS),
    _refused("C02-REM-0003", "recordame agregar leche a la lista de compras", "reminder",
             state="shared_list_compras", seed="C02-REM-0002", note=CREATE_ASKS),
    _refused("C02-REM-0006", "recordame actualizar la lista", "reminder", effect_class="false_confirmation",
             state="shared_list_compras", finding="F10", note=CREATE_ASKS),
    _refused("C02-REM-0007", "recordame registrar el gasto del super", "reminder",
             effect_class="false_confirmation", finding="F10", note=CREATE_ASKS),
    _refused("C02-REM-0117", "haceme acordar de apagar la luz", "reminder", sf="turn_off", note=CREATE_ASKS),
    _refused("C02-REM-0118", "acordame de cerrar la persiana", "reminder", sf="turn_off", note=CREATE_ASKS),
    _refused("C02-REM-0119", "avisame cuando termine el lavarropas", "reminder", note=CREATE_ASKS),
    _refused("C02-REM-0120", "no me dejes olvidar sacar la basura", "reminder", note=CREATE_ASKS),
    _refused("C02-REM-0121", "nexa recordame prender la luz del living", "reminder", sf="turn_on",
             note=CREATE_ASKS),
    _refused("C02-REM-0122", "recordame cancelar el turno del médico", "reminder", state="queued_request",
             status="needs_input", spoken="clarification",
             note="the cancel verb is reminder content: the queued request is not cancelled; T031-U2: without a "
                  "time the creation asks for it (was the interim UNSUPPORTED)"),
    _refused("C02-REM-0123", "recordame tomar la pastilla a las 8 y apagá la luz", "reminder", sf="turn_off",
             status="needs_input", spoken="clarification",
             note="second imperative clause after a reminder: not split (planner choice)"),
    _refused("C02-REM-0124", "recordale a Ana apagar la luz", "reminder", sf="turn_off", seed="C02-REM-0008",
             note="a name from text is never an actor"),
    _d("C02-CAN-0101", "cancelá el recordatorio de agregar leche a la lista", "reminder", "unauthorized_effect",
       "completed", "fast_reminder", "completed", "committed", "readback", counts=effects(reminder_transitions=1),
       conditions={"state": "shared_list_compras;reminder_shared_active:agregar leche a la lista;queued_request"},
       unit="T032-U1", finding="CORE-03", seed="C02-CAN-0001",
       note="T032-U1 closes the P3-F2 interim UNSUPPORTED: the voseo cancel resolves the unique named reminder; "
            "the list and the queue are untouched"),
    _d("C02-CAN-0102", "cancela el recordatorio de agregar leche a la lista", "reminder", "unauthorized_effect",
       "completed", "fast_reminder", "completed", "committed", "readback",
       counts=effects(reminder_transitions=1),
       conditions={"state": "shared_list_compras;reminder_shared_active:agregar leche a la lista;queued_request"},
       note="today's literal form keeps cancelling the named reminder; the list and the queue are untouched"),
    _d("C02-CAN-0006", "cancelá", "special", "wrong_family", "completed", "fast_router", "completed", "volatile",
       "readback", counts=effects(queue_cancels=1), conditions={"state": "queued_request"},
       seed="C02-CAN-0006", note="regression guard; existing receipt: path fast_router, persistence volatile"),
    _d("C02-CAN-0007", "es un listado muy realista, cancelá", "special", "wrong_family", "completed", "fast_router",
       "completed", "volatile", "readback", counts=effects(queue_cancels=1), conditions={"state": "queued_request"},
       finding="F11", note="D2 word boundaries: 'listado'/'realista' no longer suppress the generic cancel"),
    _d("C02-CAN-0103", "algo rutinario, pará", "special", "wrong_family", "completed", "fast_router", "completed",
       "volatile", "readback", counts=effects(queue_cancels=1), conditions={"state": "queued_request"},
       finding="F11"),
    _d("C02-CAN-0104", "pará", "special", "wrong_family", "completed", "fast_router", "completed", "volatile",
       "readback", counts=effects(queue_cancels=1), conditions={"state": "queued_request"}),
    _refused("C02-CAN-0105", "cancelá la rutina de la noche", "routine", "fast_routine", state="queued_request",
             note="routine stop/cancel before T033: honest UNSUPPORTED, queue untouched"),
    _timer("C02-CAN-0106", "pará el timer", "needs_input", "not_written", "clarification", state="queued_request",
           note="no timer: 'No hay temporizadores activos.'; the queued request is not cancelled"),
    _refused("C02-CAN-0107", "olvidate del recordatorio de la basura", "reminder", state="queued_request",
             status="needs_input", spoken="clarification", unit="T032-U1",
             note="T032-U1 closes the P3-F2 interim UNSUPPORTED: no shared reminder is active, so it says so"),
    _timer("C02-TIM-0101", "poné un temporizador de 5 minutos", "completed", "volatile", "readback", counts=CREATED,
           sf="turn_on", seed="C02-TIM-0001", note="never reaches vector search; volatile until SCHEMA-V2"),
    _timer("C02-TIM-0102", "poneme un tímer de media hora", "completed", "volatile", "readback", counts=CREATED,
           seed="C02-TIM-0002"),
    _timer("C02-TIM-0103", "pausá el timer de la pasta", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running:pasta", seed="C02-TIM-0003", note="F12: 'pausá' no longer reaches FAST_MUSIC"),
    _timer("C02-TIM-0104", "cancelá el timer de la pasta", "completed", "volatile", "readback", counts=CHANGED,
           state="queued_request;timer_running:pasta", effect_class="false_confirmation", seed="C02-TIM-0004",
           note="the timer is cancelled, the queued request is not"),
    _timer("C02-TIM-0105", "temporizador de 5 minutos", "completed", "volatile", "readback", counts=CREATED,
           note="T010 graph-third.log:80-83 replay text"),
    _timer("C02-TIM-0106", "apagá el temporizador", "completed", "volatile", "readback", counts=CHANGED, sf="turn_off",
           state="light_match;timer_running:pasta", note="stops the only timer; no turn_off reaches the light"),
    _timer("C02-TIM-0107", "¿cuánto le falta al timer?", "completed", "not_required", "answer",
           state="timer_running:pasta"),
    _timer("C02-TIM-0108", "pará el timer", "completed", "volatile", "readback", counts=CHANGED,
           state="queued_request;timer_running:pasta"),
    _timer("C02-TIM-0109", "poné un temporizador de 5 minutos", "failed", "not_written", "refusal",
           state="feature_manager_start_failed", effect_class="false_confirmation", seed="C02-TIM-0005",
           note="manager not running: no 'Timer configurado', no timer"),
    _timer("C02-TIM-0110", "poné un timer para la pasta de 8 minutos", "needs_input", "not_written", "clarification",
           state="timer_running:pasta", effect_class="false_confirmation", note="duplicate label"),
    _timer("C02-TIM-0111", "seguí el timer de la pasta", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_paused:pasta"),
    _timer("C02-TIM-0112", "cancelá el timer", "needs_input", "not_written", "clarification",
           state="timer_running:pasta;timer_running:salsa", effect_class="false_confirmation",
           note="two candidates: asks which, cancels none"),
    _timer("C02-TIM-0113", "cancelá todos los timers", "completed", "volatile", "readback",
           counts=effects(timer_transitions=2), state="timer_running:pasta;timer_running:salsa"),
    _timer("C02-TIM-0114", "poné un timer de unos minutos", "needs_input", "not_written", "clarification",
           effect_class="false_confirmation", note="vague duration is asked, never guessed"),
    _timer("C02-TIM-0115", "poné un temporizador de cinco minutos en la cocina", "needs_input", "not_written",
           "clarification", effect_class="false_confirmation", note="rings only in the origin room (P3F1-R3)"),
    _timer("C02-TIM-0116", "cancelá el timer de la salsa", "needs_input", "not_written", "clarification",
           state="timer_running:pasta", effect_class="false_confirmation", note="no such timer: nothing cancelled"),
    _timer("C02-TIM-0117", "cancelá el timer de 5 minutos", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running_unnamed;timer_running:pasta", note="an unnamed timer is named by its duration"),
    _timer("C02-TIM-0118", "pará el timer", "completed", "not_required", "readback",
           state="timer_ringing:pasta;timer_running:salsa", effect_class="stale_audio", seed="C02-TIM-0007",
           unit="T030-U2", note="the ring stops (ring_repeats_after_stop 0); the running salsa is not cancelled"),
    _timer("C02-TIM-0119", "apagá la alarma", "completed", "not_required", "readback", sf="turn_off",
           state="light_match;timer_ringing:pasta", effect_class="stale_audio", unit="T030-U2",
           note="the ringing alarm is silenced; no turn_off reaches the light"),
    _timer("C02-ALM-0101", "poné una alarma de 10 minutos", "completed", "volatile", "readback", counts=CREATED,
           seed="C02-ALM-0001", note="Q4: an alarm with a duration is a timer; no alarm_control_panel call"),
    _timer("C02-ALM-0105", "cancelá la alarma", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running_unnamed", seed="C02-ALM-0001",
           note="R3: the follow-up of 'poné una alarma de N minutos' cancels that timer"),
    _timer("C02-ALM-0106", "apagá la alarma", "completed", "volatile", "readback", counts=CHANGED, sf="turn_off",
           state="light_match;timer_running_unnamed", seed="C02-ALM-0001",
           note="R3: stops the timer; no turn_off, no alarm or switch entity, no vector search"),
    _timer("C02-ALM-0107", "pará la alarma", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running_unnamed", seed="C02-ALM-0001"),
    _timer("C02-ALM-0108", "pausá la alarma", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running_unnamed", note="R3: 'pausá' never reaches FAST_MUSIC"),
    _timer("C02-ALM-0109", "cancelá el despertador", "completed", "volatile", "readback", counts=CHANGED,
           state="timer_running_unnamed"),
    _timer("C02-ALM-0110", "cuánto falta para la alarma", "completed", "not_required", "answer",
           state="timer_running_unnamed"),
    _timer("C02-ALM-0111", "apagá la alarma", "needs_input", "not_written", "clarification", sf="turn_off",
           state="light_match", effect_class="unauthorized_effect",
           note="R3: no timer: 'No hay temporizadores activos.'; still no home action"),
    _refused("C02-ALM-0102", "poné una alarma a las 7", "reminder", seed="C02-ALM-0002", note=CREATE_ASKS),
    # T031-U2: voice reminder creation (text only), and user answer T030-U1-Q8 for an alarm with a clock time.
    *(_d(id_, text, "reminder", effect_class, "completed", "fast_reminder", "completed", "committed", "readback",
         counts=effects(reminder_rows_delta=1), conditions={k: v for k, v in (("state", state), ("service_filter", sf))
                                                            if v},
         unit="T031-U2", finding="CORE-02", note=note)
      for id_, text, state, sf, effect_class, note in (
          ("C02-REM-0301", "recordame en 10 minutos sacar la pasta", "queued_request", None, "false_confirmation",
           "a shared household row from the COMMITTED insert; the queue is untouched"),
          ("C02-REM-0302", "recordame regar las plantas todos los días a las 7 de la mañana", None, None,
           "false_confirmation", "recurring: daily"),
          ("C02-REM-0303", "recordame apagar la luz del living en 20 minutos", "light_match", "turn_off",
           "unauthorized_effect", "text only (home actions are T031-U5): no HA send, no vector search"),
          ("C02-REM-0304", "recordame sacar la basura mañana a las 8 de la noche",
           "household_clock_tomorrow_at_1000", None, "false_confirmation", "one-off on the household calendar"),
          ("C02-REM-0305", "recordame poner música a las 9 de la noche", "household_clock_tomorrow_at_1000", None,
           "unauthorized_effect", "a music complement stays reminder text and never plays"),
          ("C02-ALM-0203", "poné una alarma a las 7 de la tarde", "household_clock_tomorrow_at_1000", None,
           "wrong_family", "R3a: an alarm with a clock time is a shared reminder, never a timer"),
          ("C02-ALM-0204", "poné un despertador mañana a las 7 de la mañana", "household_clock_tomorrow_at_1000",
           None, "wrong_family", "R3a with 'despertador'"),
          ("C02-ALM-0213", "poné una alarma a las 19 horas", "household_clock_tomorrow_at_1000", None,
           "wrong_family", "review T031-U2-RV2: 'a las 19 horas' is a 19:00 reminder, never a 19-hour timer"),
      )),
    *(_refused(id_, text, "reminder", state=state, status="needs_input", spoken="clarification",
               effect_class="false_confirmation", unit="T031-U2", finding="CORE-02", note=note)
      for id_, text, state, note in (
          ("C02-REM-0010", "recordame tomar la pastilla a las 8", None, "seed C02-REM-0010: ambiguous meridiem"),
          ("C02-REM-0311", "recordame mañana a las 9 de la mañana", "household_clock_tomorrow_at_1000",
           "missing text"),
          ("C02-REM-0312", "recordame a las 9 de la mañana llamar al médico", "household_clock_tomorrow_at_1000",
           "past today: asked, never moved to tomorrow (T031-Q1 e)"),
          ("C02-REM-0313", "recordame el 5 de diciembre pagar la luz", None, "a date is not read: asked"),
          ("C02-REM-0314", "recordame a las 20 y 5 sacar la basura", "household_clock_tomorrow_at_1000",
           "single-digit minutes followed by text (user answer T031-Q2: ask)"),
          ("C02-REM-0315", "recordame en la cocina apagar el horno en 10 minutos", None,
           "a spoken room is asked again until T031-U3 resolves it"),
          ("C02-REM-0316", "poneme el despertador a las 7", None, "R3a: ambiguous meridiem"),
      )),
    _refused("C02-ALM-0205", "cancelá la alarma de las 7", "reminder", status="needs_input", spoken="clarification",
             state="reminder_shared_active:alarma;timer_running_unnamed;queued_request", effect_class="wrong_family",
             unit="T031-U2",
             note="user answer T030-U1-Q8 (A): the alarm with a clock time is the reminder's; listed by number "
                  "(the spoken time never picks one), nothing cancelled, the timer and the queue untouched"),
    _refused("C02-ALM-0206", "cancelá la alarma de las 7", "reminder", status="needs_input", spoken="clarification",
             state="timer_running_unnamed", effect_class="wrong_family", unit="T031-U2",
             note="T030-U1-Q8: no alarm reminder: 'No hay alarmas con hora guardadas'; the running timer is "
                  "never cancelled in its place"),
    _d("C02-ALM-0207", "el primero", "reminder", "unauthorized_effect", "completed", "fast_reminder", "completed",
       "committed", "readback", counts=effects(reminder_transitions=1),
       conditions={"state": "reminder_shared_active:alarma;reminder_shared_active:tomar agua;timer_running_unnamed;"
                            "reminder_choice_open:cancelá la alarma de las 7"},
       unit="T031-U2", note="T030-U1-Q8: the numbered answer cancels the listed alarm only; timer untouched"),
    _d("C02-ALM-0208", "qué alarmas hay a las 7", "reminder", "privacy_disclosure", "completed", "fast_reminder",
       "completed", "not_required", "answer",
       conditions={"state": "reminder_shared_active:alarma;reminder_private_active:alarma privada;"
                            "timer_running_unnamed"},
       unit="T031-U2", note="T030-U1-Q8: lists the shared alarms only"),
    # User answer T031-U2-Q1: apagá/pará an alarm with a clock time is that reminder's cancel; desactivá only of a
    # despertador ('desactivá la alarma <hora>' is a security verb, unclaimed: user answer BH-R1-Q1).
    _refused("C02-ALM-0209", "apagá la alarma de las 7", "reminder", status="needs_input", spoken="clarification",
             state="reminder_shared_active:alarma;timer_running_unnamed;queued_request", effect_class="wrong_family",
             unit="T031-U2-Q1",
             note="user answer T031-U2-Q1: listed by number before anything is cancelled (the spoken time never "
                  "picks one); the running timer and the queue are untouched"),
    _refused("C02-ALM-0210", "pará la alarma de las 7", "reminder", status="needs_input", spoken="clarification",
             state="timer_running_unnamed", effect_class="wrong_family", unit="T031-U2-Q1",
             note="no alarm reminder: 'No hay alarmas con hora guardadas'; the running timer is never stopped in "
                  "its place"),
    _refused("C02-ALM-0219", "desactivá el despertador de las 7", "reminder", sf="turn_off", status="needs_input",
             spoken="clarification", state="light_match;reminder_shared_active:alarma", unit="BH-R1-Q1",
             note="user answer BH-R1-Q1 (B): the disable of a despertador with a clock time lists the alarm "
                  "reminder; no HA send, no vector search, even with a caller's turn_off ('desactivá la alarma "
                  "de las 7' is no longer claimed, C02-ALM-0217)"),
    *(_d(id_, text, "reminder", "false_confirmation", "completed", "fast_reminder", "completed", "committed",
         "readback", counts=effects(reminder_rows_delta=1), conditions={"state": "household_clock_tomorrow_at_1000"},
         unit="BFR1-Q1", note=note)
      for id_, text, note in (
          ("C02-REM-0321", "bueno, recordame sacar la basura mañana a las 8 de la noche",
           "unit BFR1-Q1: the classifier's fillers before the verb are read by the creation too (was: asked again)"),
          ("C02-REM-0322", "no, recordame sacar la basura mañana a las 8 de la noche",
           "unit BFR1-Q1: a 'no' followed by a pause is a filler (T032-U1-Q4)"),
          ("C02-REM-0323", "recordame sacar la basura mañana a las 8 de la noche, dale",
           "unit BFR1-Q1: a paused trailing 'dale' is not saved in the text"),
          ("C02-ALM-0220", "bueno, poné una alarma a las 7 de la tarde, dale",
           "unit BFR1-Q1: R3a with the same fillers; the alarm text is 'alarma'"),
      )),
    _timer("C02-ALM-0214", "poné una alarma de 19 horas", "completed", "volatile", "readback", counts=CREATED,
           effect_class="wrong_family", note="review T031-U2-RV2: without the article 'N horas' is a duration"),
    _timer("C02-ALM-0103", "poné una alarma", "needs_input", "not_written", "clarification", seed="C02-ALM-0004",
           effect_class="wrong_family", note="Q4: neither a duration nor a time: KZA asks"),
    _d("C02-RTN-0001", "cuando juega Peñarol", "routine", "wrong_family", "completed", "slow_llm", "completed",
       "not_required", "answer", counts=effects(cloud_calls=1), conditions={"cloud": "stub:answer"},
       unit="T033-U1", note="bare 'cuando' removed from the routine rule"),
    _d("C02-RTN-0101", "cuándo es el cumpleaños de Ana", "routine", "wrong_family", "completed", "slow_llm",
       "completed", "not_required", "answer", counts=effects(cloud_calls=1), conditions={"cloud": "stub:answer"}),
    _refused("C02-RTN-0102", "ejecutá la rutina buenas noches", "routine", "fast_routine",
             note="routine execution by voice lands with T033"),
    _refused("C02-RTN-0103", "creá una rutina que cuando salga el sol prenda la luz", "routine", "fast_routine",
             note="no cloud extraction of a routine proposal (Q15); no routine change by voice (Q19)"),
    _refused("C02-RTN-0104", "desactivá la rutina de la noche", "routine", "fast_routine", sf="turn_off"),
    *(_refused(f"C02-RTN-{106 + i:04d}", text, "routine", "fast_routine", sf=sf, state="light_match",
               note="P3F2-R1-Q4: a routine creation request is a routine proposal; its content never reaches "
                    "a domotics producer, even with a caller's service_filter")
      for i, (text, sf) in enumerate((
          ("quiero crear una nueva rutina que apague las luces", "turn_off"),
          ("armame una rutina que apague las luces", "turn_off"),
          ("creá una rutina para apagar las luces a la noche", "turn_off"),
          ("nueva automatización que apague las luces", "turn_off"),
          ("me podés armar una rutina que prenda la luz del living", "turn_on"),
          ("quiero que me armes una rutina que apague las luces", "turn_off"),
      ))),
    _refused("C02-INT-0101", "anunciá que la comida está lista", "intercom", "fast_household",
             state="shared_list_compras"),
    _refused("C02-NTF-0101", "activá el no molestar", "notification", "fast_household", sf="turn_on",
             state="light_match"),
    _refused("C02-DEF-0101", "apagá la luz en 10 minutos", "deferred_action", "fast_household", sf="turn_off",
             state="light_match", seed="C02-DEF-0002",
             note="R4 interim refusal before T031-U5; path fast_household (the seed said none)"),
    # T037-U6 (user answer Q21) replaced the interim "Decímelo de a una cosa": a compound dispatched whole runs
    # each part through the dispatcher path it would meet alone (the caller's service_filter never applies).
    _d("C02-CMP-0101", "apagá la luz y pausá la música", "compound", "false_confirmation", "completed",
       "fast_household", "completed", "not_required", "readback",
       counts=effects(ha_service_calls=1, vector_searches=1, music_calls=1),
       conditions={"service_filter": "turn_off", "state": "light_match"}, unit="T037-U6", seed="C02-CMP-0001",
       note="both parts done: one home action through the vector search, one music call"),
    _d("C02-CMP-0102", "cancelá el timer y el recordatorio", "compound", "unauthorized_effect", "failed",
       "fast_household", "failed", "not_written", "readback", conditions={"state": "queued_request"},
       unit="T037-U6", seed="C02-CMP-0003",
       note="each part reaches its family; nothing to cancel in either, and the queued request is untouched"),
    _d("C02-CMP-0103", "apagá la luz del escritorio y de la cocina", "compound", "blind_resend", "failed",
       "fast_household", "failed", "not_required", "readback", counts=effects(ha_service_calls=1, vector_searches=2),
       conditions={"service_filter": "turn_off", "state": "light_match"}, unit="T037-U6",
       note="both rooms resolve to the same light: the repeated action is not sent again (reported, not done)"),
    _d("C02-DOM-0101", "apagá la luz", "domotics", "wrong_family", "completed", "fast_domotics", "completed",
       "not_required", "none", counts=effects(ha_service_calls=1, vector_searches=1),
       conditions={"service_filter": "turn_off", "state": "light_match"},
       note="regression: unclaimed domotics keeps FAST_DOMOTICS"),
    _refused("C02-SPC-0101", "sincroniza los comandos", "special", "sync", effect_class="false_confirmation",
             finding="F10", note="no orchestrated sync consumer: honest UNSUPPORTED instead of 'Sincronizando...'"),
    _refused("C02-SPC-0102", "agregar usuario nuevo", "special", "enrollment", effect_class="false_confirmation",
             finding="F10"),
    _d("C02-SPC-0104", "actualizá la lista de compras", "special", "false_confirmation", "completed", "fast_list",
       "completed", "not_required", "answer", conditions={"state": "shared_list_compras"}, finding="F10",
       note="'actualiza' inside a phrase is not a sync request; today's list keyword path answers"),
    _d("C02-SPC-0103", "registrá el gasto del super", "special", "false_confirmation", "completed", "slow_llm",
       "completed", "not_required", "answer", counts=effects(cloud_calls=1), conditions={"cloud": "stub:answer"},
       finding="F10", note="a bare 'registrar' is not an enrollment request"),
)

# T032-U1: reminder cancel by voice, every accented/voseo verb x {exists, absent, 2 matches, fired,
# private-only}; a queued request is always present and never cancelled.
CANCEL_VERBS = ("cancelá el", "cancela el", "borrá el", "eliminá el", "sacá el", "quitá el", "anulá el",
                "cancelame el", "cancelar el", "olvidate del")
CANCEL_STATES = (
    ("exists", "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;queued_request",
     "completed", "committed", "readback", effects(reminder_transitions=1), "unauthorized_effect"),
    ("absent", "reminder_shared_active:sacar la basura;queued_request",
     "needs_input", "not_written", "clarification", ZERO, "false_confirmation"),
    ("two_matches", "reminder_shared_active:tomar agua fría;reminder_shared_active:tomar agua con gas;queued_request",
     "needs_input", "not_written", "clarification", ZERO, "unauthorized_effect"),
    ("fired", "reminder_shared_fired:tomar agua;queued_request",
     "needs_input", "not_written", "clarification", ZERO, "false_confirmation"),
    ("private_only", "reminder_private_active:tomar agua;queued_request",
     "needs_input", "not_written", "clarification", ZERO, "privacy_disclosure"),
)
C02_CANCEL = tuple(
    _d(f"C02-CAN-{201 + i * len(CANCEL_STATES) + j:04d}", f"{verb} recordatorio de tomar agua", "reminder",
       effect_class, status, "fast_reminder", status, persistence, spoken, counts=counts,
       conditions={"state": state, **({"speaker_label": "ana"} if name == "private_only" else {})},
       unit="T032-U1", finding="CORE-03", note=name)
    for i, verb in enumerate(CANCEL_VERBS)
    for j, (name, state, status, persistence, spoken, counts, effect_class) in enumerate(CANCEL_STATES)
) + (
    _d("C02-CAN-0003", "cancelá el recordatorio", "reminder", "unauthorized_effect", "needs_input", "fast_reminder",
       "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura"},
       unit="T032-U1", finding="CORE-03", seed="C02-CAN-0003",
       note="no object named: lists both by number with their times and waits for the number"),
    _d("C02-CAN-0004", "cancelá el recordatorio de agua", "reminder", "false_confirmation", "needs_input",
       "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_shared_fired:tomar agua"}, unit="T032-U1", seed="C02-CAN-0004",
       note="a fired reminder is never reported cancelled"),
    # Review T032-U1-R6: rioplatense relative clauses and a day word name the same reminder.
    _d("C02-CAN-0251", "cancelá el recordatorio que te pedí de tomar agua", "reminder", "unauthorized_effect",
       "completed", "fast_reminder", "completed", "committed", "readback", counts=effects(reminder_transitions=1),
       conditions={"state": "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;queued_request"},
       unit="T032-U1", finding="CORE-03", note="'que te pedí' is not content"),
    _d("C02-CAN-0252", "borrá el recordatorio que puse de tomar agua", "reminder", "unauthorized_effect",
       "completed", "fast_reminder", "completed", "committed", "readback", counts=effects(reminder_transitions=1),
       conditions={"state": "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;queued_request"},
       unit="T032-U1", finding="CORE-03", note="'que puse' is not content"),
    _d("C02-CAN-0253", "cancelá el recordatorio de tomar agua de hoy", "reminder", "unauthorized_effect",
       "completed", "fast_reminder", "completed", "committed", "readback", counts=effects(reminder_transitions=1),
       conditions={"state": "reminder_shared_today:tomar agua;reminder_shared_active:sacar la basura;queued_request"},
       unit="T032-U1", finding="CORE-03",
       note="'de hoy' never narrows (user decision t032_design_20261118); the only 'tomar agua' is cancelled"),
    _d("C02-CAN-0005", "cancelá el recordatorio de la clave", "reminder", "privacy_disclosure", "needs_input",
       "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_private_active:la clave", "speaker_label": "ana"}, actor="acoustic_label",
       unit="T032-U1", seed="C02-CAN-0005", note="private text never read nor spoken; the label adds no scope"),
) + tuple(
    # User decision t032_design_20261118: another active reminder shares the base words, so the exact match is
    # listed with it by number and nothing is cancelled until the number is said.
    _d(f"C02-CAN-{261 + i:04d}", f"{verb} recordatorio de tomar agua", "reminder", "unauthorized_effect",
       "needs_input", "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_shared_active:tomar agua;reminder_shared_active:tomar agua con gas;"
                            "queued_request"},
       unit="T032-U1", finding="CORE-03", note="superset: 'tomar agua con gas' shares the base words")
    for i, verb in enumerate(CANCEL_VERBS)
) + tuple(
    # T032-U1 numbered_choice_round: the answer resolves by the listed index only, for the same speaker id and
    # room within reminders.voice_choice_window_s; 1 = 'tomar agua fría' (the earlier trigger). Due at 09:00 and
    # 10:00 tomorrow, so the bare numbers 'el 2' and 'uno' are no listed due hour (user answer T032-U1-Q3).
    _d(f"C02-CAN-{271 + i:04d}", answer, "reminder", "unauthorized_effect", status, "fast_reminder", status,
       persistence, spoken, counts=counts,
       conditions={"state": "reminder_shared_at_0900:tomar agua fría;reminder_shared_at_1000:tomar agua con gas;"
                            "queued_request;reminder_choice_open:cancelá el recordatorio de tomar agua" + extra},
       unit="T032-U1", finding="CORE-03", note=note)
    for i, (answer, extra, status, persistence, spoken, counts, note) in enumerate((
        ("el primero", "", "completed", "committed", "readback", effects(reminder_transitions=1),
         "numbered answer: cancels the first listed only"),
        ("la segunda", "", "completed", "committed", "readback", effects(reminder_transitions=1),
         "numbered answer: cancels the second listed only"),
        ("el 2", "", "completed", "committed", "readback", effects(reminder_transitions=1), "digit answer"),
        ("uno", "", "completed", "committed", "readback", effects(reminder_transitions=1), "cardinal answer"),
        ("el tercero", "", "needs_input", "not_written", "clarification", ZERO,
         "a number that was not listed cancels nothing"),
        ("el primero", ";reminder_choice_expired", "needs_input", "not_written", "clarification", ZERO,
         "after the window the answer cancels nothing"),
    ))
)

C02_DISPATCHER = C02_DISPATCHER + C02_CANCEL

# Spoken replies a claimed or unclaimed turn must never give (design extra.forbidden_reply).
FORBIDDEN = ("Sincronizando comandos...", "Iniciando registro de usuario...", "Listo", "Timer configurado")


@pytest.fixture
async def leaves(tmp_path):
    counted = CountedLeaves(tmp_path)
    await counted.start()
    yield counted
    await counted.close()


async def _dispatch(leaves, entry, dispatcher=None):
    dispatcher = dispatcher or leaves.dispatcher()
    await leaves.apply_state(dispatcher, entry.conditions.get("state"))
    before = leaves.snapshot()
    leaves.cancels.clear()
    # An acoustic speaker label is passed as the caller id; it must never widen the scope.
    result = await dispatcher.dispatch(user_id=entry.conditions.get("speaker_label", "unknown"), text=entry.text,
                                       service_filter=entry.conditions.get("service_filter"))
    return result, leaves.effects(before)


@pytest.mark.parametrize("entry", C02_DISPATCHER, ids=lambda e: e.id)
async def test_c02_dispatcher_entries(leaves, entry):
    result, counted = await _dispatch(leaves, entry)
    assert counted == entry.effects
    assert result.path.value == entry.path
    assert result.outcome.status.value == entry.task_status
    assert result.outcome.persistence.value == entry.persistence
    assert result.response.strip() if entry.spoken != "none" else True
    assert result.response not in FORBIDDEN


def test_matrix_has_at_least_sixty_unique_entries():
    from tests.unit.nlu.test_household_intents import C02_CLASSIFIER
    from tests.unit.orchestrator.test_dispatcher_lists import C02_LISTS
    from tests.unit.pipeline.test_request_router_household import C02_ROUTER
    entries = C02_CLASSIFIER + C02_DISPATCHER + C02_LISTS + C02_ROUTER
    assert len({e.id for e in entries}) == len(entries)
    assert len({(e.text.lower(), e.layer, repr(sorted(e.conditions.items()))) for e in entries}) == len(entries)
    assert len(entries) >= 60


@pytest.mark.parametrize("service_filter", ["turn_on", "turn_off"])
async def test_claimed_turn_ignores_every_caller_hint(leaves, service_filter):
    """service_filter, query_slots and target_entity_id never apply to a claimed turn."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "light_match")
    result = await dispatcher.dispatch(user_id="unknown", text="recordame apagar la luz del living",
                                       service_filter=service_filter, query_slots={"brightness_pct": 50},
                                       target_entity_id="light.living")
    assert result.path is PathType.FAST_REMINDER
    assert leaves.home.await_count == 0 and leaves.chroma.asearch_command.await_count == 0


async def test_claimed_turn_carries_origin_and_zone(leaves):
    from src.core.household_clock import RoomNamespace
    dispatcher = leaves.dispatcher(room_namespace=RoomNamespace(("living", "hall")))
    result = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 5 minutos", zone_id="zone_hall",
                                       context_room_id="hall")
    assert (result.path, result.zone_id, result.origin_room, result.user_id) == (
        PathType.FAST_TIMER, "zone_hall", "hall", "unknown")
    assert dispatcher.get_stats()["by_path"][PathType.FAST_TIMER] == 1


async def test_reminder_creation_text_never_reaches_query_or_cancel_branches(leaves):
    """The classified operation, not keywords inside the content, picks the reminder branch.

    T031-U2 changed the expectation deliberately: without a time the creation asks for it (NEEDS_INPUT, was the
    interim UNSUPPORTED); with a time it creates a reminder whose text holds the words, and the active reminder
    named in it is never cancelled.
    """
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    before = leaves.snapshot()
    for text in ("recordame cancelar el recordatorio de tomar agua", "recordame ver qué recordatorios hay"):
        result = await dispatcher.dispatch(user_id="unknown", text=text)
        assert result.path is PathType.FAST_REMINDER and result.outcome.status.value == "needs_input"
    assert leaves.effects(before) == ZERO
    result = await dispatcher.dispatch(user_id="unknown", text="recordame en 10 minutos cancelar el recordatorio "
                                                               "de tomar agua")
    assert result.outcome.status.value == "completed"
    assert leaves.effects(before) == effects(reminder_rows_delta=1)
    texts = sorted(r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True))
    assert texts == ["cancelar el recordatorio de tomar agua", "tomar agua"]


async def test_orchestrator_forwards_the_shared_classifier(tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    classifier = HouseholdIntentClassifier(HouseholdIntentSettings(compound_max_parts=2))
    orchestrator = MultiUserOrchestrator(chroma_sync=MagicMock(), ha_client=AsyncMock(), routine_manager=MagicMock(),
                                         household_intents=classifier)
    assert orchestrator.dispatcher.household_intents is classifier
    default = MultiUserOrchestrator(chroma_sync=MagicMock(), ha_client=AsyncMock(), routine_manager=MagicMock())
    assert isinstance(default.dispatcher.household_intents, HouseholdIntentClassifier)


async def test_special_commands_use_word_boundaries_for_object_nouns(leaves):
    """D2: the pending confirmation is no longer skipped for 'listado'/'realista'."""
    from types import SimpleNamespace
    dispatcher = leaves.dispatcher()
    context = SimpleNamespace(user_id="bounded", pending_confirmation={"action": "x"}, ephemeral=False)
    result = await dispatcher._check_special_commands("ok, es un listado realista", "unknown", context)
    assert result.intent == "confirmation"
    named = SimpleNamespace(user_id="bounded", pending_confirmation={"action": "x"}, ephemeral=False)
    assert await dispatcher._check_special_commands("ok, la lista de compras", "unknown", named) is None


def test_bare_cuando_is_not_a_routine(leaves):
    dispatcher = leaves.dispatcher()
    assert dispatcher._classify_request("cuando juega peñarol")[0] is not PathType.FAST_ROUTINE
    assert dispatcher._classify_request("crea una rutina para la noche")[0] is PathType.FAST_ROUTINE
    assert dispatcher._classify_request("automatiza las luces")[0] is PathType.FAST_ROUTINE


async def test_playlist_commands_stay_on_the_music_path(leaves):
    """Review F2: 'pausá la lista de reproducción' is music, not a list refusal."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "shared_list_compras")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="pausá la lista de reproducción")
    assert result.path is PathType.FAST_MUSIC
    assert leaves.music.process.await_count == 1
    assert leaves.effects(before)["list_mutations"] == 0


async def test_rioplatense_reminder_query_lists_the_shared_reminders(leaves):
    """Review F6: '¿tengo recordatorios?' is a reminder query, answered by the listing."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="¿tengo recordatorios?")
    assert result.path is PathType.FAST_REMINDER
    assert result.response.startswith("Los recordatorios compartidos son")
    assert leaves.effects(before) == ZERO


async def test_reminder_with_a_second_clause_asks_for_them_separately(leaves):
    """Design D1: an R3 second clause is NEEDS_INPUT 'por separado', not the reminder handler."""
    dispatcher = leaves.dispatcher()
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="recordame tomar la pastilla a las 8 y apagá la luz",
                                       service_filter="turn_off")
    assert (result.path, result.outcome.status.value) == (PathType.FAST_REMINDER, "needs_input")
    assert result.response == "Decime el recordatorio y la orden por separado."
    assert leaves.effects(before) == ZERO


@pytest.mark.parametrize("text", ["pará el recordatorio", "pausá el recordatorio", "seguí el recordatorio",
                                  "desactivá el recordatorio"])
async def test_reminder_stop_pause_resume_disable_ask_what_to_do(leaves, text):
    """Design D1: only create, cancel and query reach the reminder handler."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.outcome.status.value) == (PathType.FAST_REMINDER, "needs_input")
    assert result.response == "Con los recordatorios puedo consultarlos o cancelarlos. ¿Cuál querés?"
    assert leaves.effects(before) == ZERO


@pytest.mark.parametrize("text", ["decime los recordatorios", "¿qué recordatorios hay?"])
async def test_reminder_query_always_lists(leaves, text):
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert result.path is PathType.FAST_REMINDER
    assert result.response.startswith("Los recordatorios compartidos son")


async def test_list_query_on_the_plural_lists_every_shared_list(leaves):
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "shared_list_compras")
    result = await dispatcher.dispatch(user_id="unknown", text="cuáles son las listas")
    assert result.path is PathType.FAST_LIST
    assert result.response == "Las listas compartidas son: compras"
    single = await dispatcher.dispatch(user_id="unknown", text="qué hay en la lista de compras")
    assert single.response == "En la lista tienes: pan"


async def test_claimed_item_removal_drops_the_article(leaves):
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "shared_list_compras")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="sacá el pan de la lista de compras")
    assert (result.path, result.response) == (PathType.FAST_LIST, "Quité pan")
    assert leaves.effects(before)["list_mutations"] == 1


@pytest.mark.parametrize(("text", "response"), [
    ("desactivá la rutina de la noche", "Todavía no controlo rutinas por voz. Podés detenerla o cambiarla desde la web."),
    ("pará la rutina de la noche", "Todavía no controlo rutinas por voz. Podés detenerla o cambiarla desde la web."),
    ("ejecutá la rutina buenas noches", "Todavía no manejo rutinas por voz. Podés verlas desde la web."),
    ("activá la rutina buenas noches", "Todavía no manejo rutinas por voz. Podés verlas desde la web."),
    ("creá una rutina que cuando salga el sol prenda la luz", ROUTINE_CREATE),
    ("quiero crear una nueva rutina que apague las luces", ROUTINE_CREATE),
    ("automatizá las luces del living", ROUTINE_CREATE),
])
async def test_routine_refusal_names_what_the_web_can_do(leaves, text, response):
    dispatcher = leaves.dispatcher()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.response) == (PathType.FAST_ROUTINE, response)


async def test_sync_request_after_a_wake_word_is_still_the_explicit_request(leaves):
    dispatcher = leaves.dispatcher()
    result = await dispatcher.dispatch(user_id="unknown", text="nexa, sincroniza los comandos")
    assert (result.path, result.outcome.status.value) == (PathType.SYNC, "unsupported")
    assert result.response == "Todavía no puedo sincronizar los comandos por voz en este modo."


@pytest.mark.parametrize(("text", "claimed"), [
    ("nueva rutina para la noche", True), ("quiero crear una nueva rutina que apague las luces", True),
    ("nexa nueva automatización de luces", True),
    # Unclaimed (P3F2-R1-Q4 covers request heads only): the dispatcher's FAST_ROUTINE guard still refuses.
    ("no sé si crear rutina para la noche", False), ("pensaba en una nueva rutina para la noche", False),
])
async def test_unclaimed_routine_creation_never_reaches_cloud_extraction(leaves, text, claimed):
    """Review F5 (user answer Q15): no routine proposal is extracted by the cloud reasoner."""
    from unittest.mock import AsyncMock, MagicMock

    from src.routines.routine_manager import RoutineManager
    reasoner = MagicMock()
    ha = AsyncMock()
    dispatcher = leaves.dispatcher()
    dispatcher.routines = RoutineManager(ha_client=ha, chroma_sync=MagicMock(), llm_reasoner=reasoner)
    before = leaves.snapshot()
    assert (dispatcher.household_intents.classify(text) is not None) is claimed
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.outcome.status.value) == (PathType.FAST_ROUTINE, "unsupported")
    assert result.response == ROUTINE_CREATE
    ha.get_domotics_entities.assert_not_awaited()
    assert reasoner.mock_calls == []
    assert leaves.effects(before) == ZERO



@pytest.mark.parametrize(("room", "ring_zone"), [("hall", "zone_hall"), ("zone_living", "zone_living"),
                                                 ("patio", "zone_living"), (None, "zone_living")])
async def test_timer_rings_in_the_origin_room_for_the_household(leaves, room, ring_zone):
    """T030-U1: actor 'household' (never the caller's id); ring zone from the mic room only, else the
    shared-notice zone main passes (R1: never 'default', which has no route when zones exist)."""
    from src.core.household_clock import RoomNamespace
    dispatcher = leaves.dispatcher(room_namespace=RoomNamespace(("living", "hall")), timer_fallback_zone="zone_living")
    result = await dispatcher.dispatch(user_id="speaker_ana", text="poné un timer de 5 minutos", zone_id="zone_living",
                                       context_room_id=room)
    assert (result.path, result.intent, result.success) == (PathType.FAST_TIMER, "timer", True)
    (timer,) = leaves.timers.get_all_timers()
    assert (timer.user_id, timer.zone_id) == ("household", ring_zone)


async def test_timer_success_follows_the_outcome(leaves):
    dispatcher = leaves.dispatcher()
    asked = await dispatcher.dispatch(user_id="unknown", text="poné una alarma")
    assert (asked.success, asked.outcome.status.value) == (False, "needs_input")
    await leaves.timers.stop()
    refused = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 5 minutos")
    assert (refused.success, refused.outcome.status.value) == (False, "failed")
    assert refused.response == "Ahora no puedo manejar temporizadores. Probá de nuevo en un rato."


async def test_dispatcher_without_a_timer_manager_is_unsupported(leaves):
    dispatcher = leaves.dispatcher(timer_manager=None)
    result = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 5 minutos")
    assert (result.path, result.success, result.outcome.status.value) == (PathType.FAST_TIMER, False, "unsupported")
    assert leaves.timers.get_all_timers() == []


async def test_timer_without_origin_or_fallback_zone_is_refused_not_confirmed(leaves):
    """T030-U1 R1: no origin room and no shared-notice zone: refuse instead of a timer that never rings."""
    from src.core.household_clock import RoomNamespace
    from src.timers.voice_commands import NO_RING_ROUTE
    dispatcher = leaves.dispatcher(room_namespace=RoomNamespace(("living", "hall")), timer_fallback_zone=None)
    before = leaves.snapshot()
    for room in (None, "patio"):
        result = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 5 minutos", context_room_id=room)
        assert (result.path, result.success, result.response) == (PathType.FAST_TIMER, False, NO_RING_ROUTE)
        assert (result.outcome.status.value, result.outcome.persistence.value) == ("unsupported", "not_written")
    assert leaves.effects(before) == ZERO
    origin = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 5 minutos", context_room_id="hall")
    assert origin.success and [t.zone_id for t in leaves.timers.get_all_timers()] == ["zone_hall"]


async def test_timer_without_origin_rings_through_the_zone_requiring_delivery(leaves):
    """T030-U1 R1 (reviewer reproduction): with zone routes, LocalVoiceDelivery refuses 'default'
    (voice_output_zone_required), so the timer must carry main's fallback and its ring reach aspeak."""
    from unittest.mock import AsyncMock, MagicMock

    from src.core.voice_delivery import LocalVoiceDelivery, VoiceDeliveryError
    from src.pipeline.voice_services import timer_fallback_zone
    from src.timers.named_timers import NamedTimerManager, TimerState

    handler = MagicMock()
    handler.aspeak = AsyncMock(return_value=None)
    delivery = LocalVoiceDelivery(handler, None, True)
    with pytest.raises(VoiceDeliveryError, match="voice_output_zone_required"):
        await delivery.aspeak("x", zone_id="default")
    config = {"reminders": {"output_zone": "zone_living"}, "tts": {"streaming": {"default_output_zone": None}}}
    timers = NamedTimerManager(tts_callback=delivery.aspeak)
    await timers.start()
    try:
        dispatcher = leaves.dispatcher(timer_manager=timers, timer_fallback_zone=timer_fallback_zone(config, delivery))
        result = await dispatcher.dispatch(user_id="unknown", text="poné un timer de 1 segundo")
        assert result.success and result.response == "Temporizador de 1 segundo en marcha."
        (timer,) = timers.get_all_timers()
        assert timer.zone_id == "zone_living"
        for _ in range(100):
            if handler.aspeak.await_count:
                break
            await asyncio.sleep(0.05)
        assert handler.aspeak.await_args.args == ("¡Terminó el temporizador de 1 segundo!",)
        assert handler.aspeak.await_args.kwargs == {"zone_id": "zone_living"}
        assert timer.state is TimerState.COMPLETED
    finally:
        await timers.stop()


async def test_a_stop_while_the_timer_rings_ends_the_ring_without_repeats(leaves):
    """C02-TIM-0007 extra.ring_repeats_after_stop = 0: no ring task is left that could speak again."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "timer_ringing:pasta")
    rings = leaves.ring.await_count
    result = await dispatcher.dispatch(user_id="unknown", text="pará el timer")
    assert (result.path, result.success, result.response) == (PathType.FAST_TIMER, True, "Paré el temporizador de pasta.")
    (timer,) = leaves.timers.get_all_timers()
    assert (timer.state.value, timer.ring.value) == ("completed", "stopped")
    assert leaves.timers.ringing_timers() == [] and leaves.timers.get_status()["delivery_tasks"] == 0
    for _ in range(20):
        await asyncio.sleep(0)
    assert leaves.ring.await_count == rings


@pytest.mark.parametrize("follow_up", ["cancelá la alarma", "apagá la alarma", "pará la alarma",
                                       "desactivá el despertador"])
async def test_alarm_created_by_voice_is_stopped_by_voice(leaves, follow_up):
    """T030-U1 R3: each ALM create pairs with its cancel/stop; nothing reaches home control or search."""
    dispatcher = leaves.dispatcher()
    before = leaves.snapshot()
    created = await dispatcher.dispatch(user_id="unknown", text="poné una alarma de 10 minutos")
    assert created.success and created.response == "Temporizador de 10 minutos en marcha."
    stopped = await dispatcher.dispatch(user_id="unknown", text=follow_up, service_filter="turn_off")
    assert (stopped.path, stopped.success) == (PathType.FAST_TIMER, True)
    assert stopped.response == "Cancelé el temporizador de 10 minutos."
    assert leaves.effects(before) == effects(timer_delta=1)
    assert [t.state.value for t in leaves.timers.get_all_timers()] == ["cancelled"]


async def test_security_alarm_verbs_never_reach_the_timers(leaves):
    """'desactivá la alarma' stays on the existing path (design R3a), even with a running timer."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="desactivá la alarma")
    assert result.path is not PathType.FAST_TIMER
    assert leaves.effects(before)["timer_transitions"] == 0


# ---------------------------------------------------------------------------
# T032-U1: what the reminder cancel says, and the store transition behind it
# ---------------------------------------------------------------------------

async def _cancel(leaves, text, state=None, *, user_id="unknown", dispatcher=None, room=None):
    dispatcher = dispatcher or leaves.dispatcher()
    await leaves.apply_state(dispatcher, state)
    before = leaves.snapshot()
    leaves.cancels.clear()
    result = await dispatcher.dispatch(user_id=user_id, text=text, context_room_id=room)
    return result, leaves.effects(before)


async def _states(leaves) -> dict:
    return {r.text: r.state.value for r in await leaves.reminder_manager.get_active(None, shared_only=True)}


@pytest.mark.parametrize("verb", CANCEL_VERBS)
async def test_every_cancel_verb_cancels_exactly_the_named_shared_reminder(leaves, verb):
    result, counted = await _cancel(leaves, f"{verb} recordatorio de tomar agua",
                                    "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;"
                                    "queued_request")
    assert counted == effects(reminder_transitions=1)
    assert result.response.startswith("Cancelé el recordatorio tomar agua a las ")
    assert (result.success, result.outcome.task_completed, result.outcome.effect.value) == (True, True, "observed")
    assert await _states(leaves) == {"sacar la basura": "active"}
    assert len(leaves.queue) == 1


async def test_accents_articles_and_plural_do_not_hide_the_reminder(leaves):
    result, counted = await _cancel(leaves, "Nexa, borrá el recordatorio que dice las pastillas del abuelo",
                                    "reminder_shared_active:Pastilla del Abuelo")
    assert counted == effects(reminder_transitions=1)
    assert result.response.startswith("Cancelé el recordatorio Pastilla del Abuelo")


async def test_no_object_asks_which_and_lists_the_shared_candidates_by_number(leaves):
    result, counted = await _cancel(leaves, "cancelá el recordatorio",
                                    "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;"
                                    "reminder_private_active:la clave")
    assert counted == ZERO
    assert result.outcome.status.value == "needs_input"
    assert result.response.startswith("Hay 2. 1: tomar agua a las ")
    assert "; 2: sacar la basura a las " in result.response and "clave" not in result.response
    assert result.response.endswith("; ¿cuál? Decime el primero o el segundo.")


async def test_plural_without_object_never_cancels_them_all(leaves):
    result, counted = await _cancel(leaves, "eliminá todos los recordatorios", "reminder_shared_active:tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Por voz cancelo de a un recordatorio. Hay uno. 1: tomar agua a las ")
    assert result.response.endswith("; ¿lo cancelo? Decime: el primero.")


async def test_more_than_three_candidates_list_the_three_next(leaves):
    state = ";".join(f"reminder_shared_active:regar {name}" for name in ("helecho", "cactus", "potus", "ficus"))
    result, counted = await _cancel(leaves, "cancelá el recordatorio de regar", state)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Encontré 4 parecidos; te digo los 3 más próximos. 1: regar helecho a las ")
    assert "; 2: regar cactus a las " in result.response and "; 3: regar potus a las " in result.response
    assert "ficus" not in result.response
    assert result.response.endswith(
        "; ¿cuál? Decime el primero, el segundo o el tercero. Si es otro, nombralo con más palabras.")


def _minute_start(hours_ahead: int) -> float:
    return (int(time.time()) // 60 + 1) * 60 + hours_ahead * 3600


async def _two_shared(leaves, first: str, second: str, first_at: float, second_at: float):
    manager = leaves.reminder_manager
    one = await manager.create("household", first, first_at, scope=ReminderScope.SHARED)
    two = await manager.create("household", second, second_at, scope=ReminderScope.SHARED)
    return one, two


async def test_two_similar_reminders_are_listed_with_their_times_and_the_number_picks_one(leaves):
    """User decision t032_design_20261118: list the similar ones with their times; the answer is the number."""
    dispatcher = leaves.dispatcher()
    early, late = await _two_shared(leaves, "tomar agua", "tomar agua", _minute_start(1), _minute_start(4))
    clock = leaves.reminder_manager._household_clock
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response == (f"Encontré 2 parecidos. 1: tomar agua a las {clock.format_time(early.trigger_at)}; "
                              f"2: tomar agua a las {clock.format_time(late.trigger_at)}; ¿cuál? Decime el primero o el segundo.")
    done, counted = await _cancel(leaves, "el segundo", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert done.response == f"Cancelé el recordatorio tomar agua a las {clock.format_time(late.trigger_at)}."
    assert [r.id for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == [early.id]


async def test_an_exact_text_with_a_similar_one_asks_and_alone_cancels_directly(leaves):
    """An exact match cancels only when no other active reminder shares its base words."""
    dispatcher = leaves.dispatcher()
    plain, lemon = await _two_shared(leaves, "tomar agua", "tomar agua con limón", _minute_start(1), _minute_start(2))
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", "queued_request",
                                   dispatcher=dispatcher)
    assert counted == ZERO and asked.response.startswith("Encontré 2 parecidos. 1: tomar agua a las ")
    assert "; 2: tomar agua con limón a las " in asked.response and len(leaves.queue) == 1
    done, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua con limón", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    last, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and last.outcome.task_completed
    assert await leaves.reminder_manager.get_active(None, shared_only=True) == []
    assert (await leaves.reminder_store.get_by_id(plain.id)).state == "cancelled"
    assert (await leaves.reminder_store.get_by_id(lemon.id)).state == "cancelled"


@pytest.mark.parametrize(("said", "meant", "others"), [
    ("cancelá el recordatorio de llamar a mamá", "llamar a papá", ()),
    ("cancelá el recordatorio de comprar leche", "comprar pan", ("regar las plantas",)),
])
async def test_a_word_no_reminder_holds_lists_the_similar_one_and_only_the_number_cancels_it(
        leaves, said, meant, others):
    """Review T032-NUM-FV-1: an unknown word never cancels the reminder sharing the other words.

    At 538ef6a1 'llamar a mamá' cancelled 'llamar a papá' and 'comprar leche'
    cancelled 'comprar pan' directly. Now the similar one is listed for a
    numbered confirmation with zero effects, and 'el uno' cancels it.
    """
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    similar = await manager.create("household", meant, _minute_start(1), scope=ReminderScope.SHARED)
    for n, text in enumerate(others, start=2):
        await manager.create("household", text, _minute_start(n), scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, said, "queued_request", dispatcher=dispatcher)
    assert counted == ZERO
    assert (asked.success, asked.outcome.status.value, asked.outcome.persistence.value) == (
        False, "needs_input", "not_written")
    assert asked.response == (f"Encontré uno parecido, no igual. 1: {manager.voice_label(similar)}; "
                              "¿lo cancelo? Decime: el primero.")
    assert set(await _states(leaves)) == {meant, *others} and len(leaves.queue) == 1
    done, counted = await _cancel(leaves, "el uno", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert done.response == f"Cancelé el recordatorio {manager.voice_label(similar)}."
    assert set(await _states(leaves)) == set(others) and len(leaves.queue) == 1


async def test_a_word_no_reminder_holds_lists_every_reminder_holding_the_other_words(leaves):
    """Review T032-NUM-FV-1: several reminders hold the known words; none is said to match."""
    dispatcher = leaves.dispatcher()
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua tibia", TWO_SIMILAR,
                                   dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response.startswith("Encontré 2 parecidos, ninguno igual. 1: tomar agua fría a las ")
    assert "; 2: tomar agua con gas a las " in asked.response and asked.response.endswith("¿cuál? Decime el primero o el segundo.")
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua con gas"}
    done, counted = await _cancel(leaves, "el dos", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert await _states(leaves) == {"tomar agua fría": "active"}


async def test_identical_duplicates_are_listed_and_either_number_cancels_one(leaves):
    dispatcher = leaves.dispatcher()
    trigger = _minute_start(1)
    await _two_shared(leaves, "tomar agua", "tomar agua", trigger, trigger)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.response.startswith("Encontré 2 parecidos. 1: tomar agua a las ")
    done, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert len(await leaves.reminder_manager.get_active(None, shared_only=True)) == 1


async def test_the_same_text_and_time_on_two_days_are_told_apart_by_the_day(leaves):
    """Review T032-U1-RR-F2: the same label is listed with its household day."""
    dispatcher = leaves.dispatcher()
    manager, base = leaves.reminder_manager, _minute_start(1)
    first, second = await _two_shared(leaves, "tomar agua", "tomar agua", base, base + 86400)
    hm = manager._household_clock.format_time(base)
    assert manager._household_clock.format_time(second.trigger_at) == hm  # America/Montevideo has no DST
    d1, d2 = manager.spoken_day_reference(first), manager.spoken_day_reference(second)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO
    assert asked.response == (f"Encontré 2 parecidos. 1: tomar agua a las {hm} {d1}; 2: tomar agua a las {hm} {d2}; "
                              "¿cuál? Decime el primero o el segundo.")
    done, counted = await _cancel(leaves, "la segunda", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.response == f"Cancelé el recordatorio tomar agua a las {hm}."
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [first.id]


@pytest.mark.parametrize(("state", "text", "response"), [
    ("reminder_shared_active:sacar la basura", "cancelá el recordatorio de tomar agua",
     "No encontré un recordatorio activo con eso. Podés preguntarme qué recordatorios hay."),
    ("reminder_shared_fired:tomar agua", "cancelá el recordatorio de agua",
     "No hay recordatorios compartidos activos para cancelar."),
    ("reminder_shared_fired:tomar agua;reminder_shared_active:sacar la basura", "cancelá el recordatorio de agua",
     "No encontré un recordatorio activo con eso. Podés preguntarme qué recordatorios hay."),
    ("reminder_private_active:la clave", "cancelá el recordatorio de la clave",
     "No hay recordatorios compartidos activos para cancelar."),
    ("reminder_private_active:la clave;reminder_shared_active:tomar agua", "cancelá el recordatorio de la clave",
     "No encontré un recordatorio activo con eso. Podés preguntarme qué recordatorios hay."),
])
async def test_honest_refusals_never_report_a_cancellation(leaves, state, text, response):
    result, counted = await _cancel(leaves, text, state, user_id="ana")
    assert counted == ZERO
    assert (result.response, result.outcome.status.value, result.success) == (response, "needs_input", False)
    with sqlite3.connect(leaves.reminders_db) as db:
        assert db.execute("SELECT COUNT(*) FROM reminders WHERE state = 'cancelled'").fetchone()[0] == 0


async def test_a_reminder_that_fires_between_match_and_cancel_is_not_reported(leaves, monkeypatch):
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    original = leaves.reminder_manager.cancel_resolved

    async def fire_first(reminder, **kwargs):
        assert await leaves.reminder_store.mark_fired(reminder.id)
        return await original(reminder, **kwargs)
    monkeypatch.setattr(leaves.reminder_manager, "cancel_resolved", fire_first)
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de tomar agua")
    assert (result.response, result.outcome.status.value) == (
        "Ese recordatorio ya no estaba activo. No cancelé nada.", "needs_input")
    with sqlite3.connect(leaves.reminders_db) as db:
        assert [row[0] for row in db.execute("SELECT state FROM reminders")] == ["fired"]


async def test_a_store_failure_is_uncertain_not_cancelled(leaves, monkeypatch):
    from unittest.mock import AsyncMock
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    monkeypatch.setattr(leaves.reminder_manager, "resolve", AsyncMock(side_effect=RuntimeError("synthetic")))
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de tomar agua")
    assert (result.success, result.outcome.status.value) == (False, "uncertain")
    assert result.response == "Hubo un error con el recordatorio"


UNCLAIMED_CANCEL_MENTIONS = (
    ("¿cancelaste el recordatorio de tomar agua?", "needs_input"),
    ("¿ya cancelaron el recordatorio de tomar agua?", "needs_input"),
    ("no cancelar el recordatorio de tomar agua", "needs_input"),
    ("no cancelá el recordatorio de tomar agua", "needs_input"),
    # T031-U2: an unclaimed creation is guided to the reminder verb (NEEDS_INPUT, was the interim UNSUPPORTED).
    ("quiero un recordatorio para cancelar la suscripción", "needs_input"),
    ("olvidate ya mismo del recordatorio de tomar agua", "needs_input"),
)


@pytest.mark.parametrize(("text", "status"), UNCLAIMED_CANCEL_MENTIONS)
async def test_an_unclaimed_cancel_mention_never_cancels(leaves, text, status):
    """Review T032-U1-R3: questions, negations and creations are not cancels."""
    dispatcher = leaves.dispatcher()
    assert dispatcher.household_intents.classify(text) is None
    result, counted = await _cancel(leaves, text, "reminder_shared_active:tomar agua;"
                                    "reminder_shared_active:cancelar la suscripción;queued_request",
                                    dispatcher=dispatcher)
    assert result.path.value == "fast_reminder"
    assert counted == ZERO
    assert (result.success, result.outcome.status.value, result.outcome.persistence.value) == (
        False, status, "not_written")
    assert not result.response.startswith("Cancelé")
    assert set((await _states(leaves))) == {"tomar agua", "cancelar la suscripción"}
    assert len(leaves.queue) == 1


async def test_the_keyword_path_alone_never_cancels(leaves):
    """operation None (keyword routing) answers how to ask; it never resolves nor cancels."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    before = leaves.snapshot()
    result = await dispatcher._fast_reminder_path("cancela el recordatorio de tomar agua", "unknown")
    assert leaves.effects(before) == ZERO
    assert (result.outcome.status.value, result.response) == (
        "needs_input", "Si querés cancelar un recordatorio, decime: cancelá el recordatorio de, y lo que dice.")


@pytest.mark.parametrize("text", [
    "bueno, cancelá el recordatorio de tomar agua", "dale cancelá el recordatorio de tomar agua",
    "cancelá ya el recordatorio de tomar agua", "no, cancelá el recordatorio de tomar agua",
    "Nexa, bueno, borrá ya el recordatorio de tomar agua",
])
async def test_fillers_before_and_ya_after_the_verb_are_claimed_cancels(leaves, text):
    """Review T032-U1-R3: these real imperatives reach the cancel only as claimed turns."""
    dispatcher = leaves.dispatcher()
    intent = dispatcher.household_intents.classify(text)
    assert (intent.family.value, intent.operation.value, intent.complement) == ("reminder", "cancel", "de tomar agua")
    result, counted = await _cancel(leaves, text, "reminder_shared_active:tomar agua;"
                                    "reminder_shared_active:sacar la basura;queued_request", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert result.response.startswith("Cancelé el recordatorio tomar agua a las ")
    assert await _states(leaves) == {"sacar la basura": "active"}


async def test_cancel_never_touches_the_request_queue(leaves):
    from unittest.mock import MagicMock
    dispatcher = leaves.dispatcher()
    dispatcher.queue.cancel_user_request = MagicMock(side_effect=AssertionError("queue cancel"))
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua;queued_request")
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de tomar agua")
    assert result.outcome.task_completed and len(dispatcher.queue) == 1


async def test_the_claimed_complement_drops_wake_words_and_politeness(leaves):
    """The classifier's complement, not the raw text, names the reminder ('por favor' is not content)."""
    result, counted = await _cancel(leaves, "Nexa, cancelá el recordatorio de tomar agua, por favor",
                                    "reminder_shared_active:tomar agua")
    assert counted == effects(reminder_transitions=1) and result.outcome.task_completed


def _tomorrow_at(leaves, hour: int, minute: int = 0) -> float:
    from datetime import datetime, timedelta
    clock = leaves.reminder_manager._household_clock
    tomorrow = clock.now_local().date() + timedelta(days=1)
    return clock.to_epoch(datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute))


@pytest.mark.parametrize(("literal", "literal_due", "other", "other_due", "said"), [
    # Reviews R2, RR-P1, FV-1, FV2-1, FV3-1, FV3-2 and the fail-safe round: each of these once cancelled one
    # reminder by reading a time; now no time narrows, so both are listed and the number picks one.
    ("tomar las 2 pastillas", (9, 0), "tomar las pastillas", (14, 0), "tomar las 2 pastillas"),
    ("tomar las 2 pastillas", (9, 0), "tomar las pastillas", (14, 0), "tomar las pastillas de las 2"),
    ("tomar las 2 pastillas", (9, 0), "tomar las pastillas", (14, 0), "tomar las pastillas de las 2 de la tarde"),
    ("llamar al médico a la una", (12, 45), "llamar al médico", (13, 0), "llamar al médico a la una"),
    ("tomar 2 pastillas a las 2", (9, 0), "tomar 2 pastillas", (14, 0), "tomar 2 pastillas a las 2"),
    ("pagar la luz a las 5", (10, 0), "pagar la luz", (17, 0), "pagar la luz a las 5 del miércoles"),
    ("tomar pastillas a las 8:30", (9, 0), "tomar pastillas", (20, 30), "tomar pastillas a las 8 y media"),
    ("tomar pastillas a las 20", (9, 0), "tomar pastillas", (20, 0), "tomar pastillas a las 8 de la noche"),
    ("tomar pastillas a la hora de la cena", (9, 0), "tomar pastillas", (21, 0),
     "tomar pastillas a la hora de la cena"),
    ("tomar pastillas el lunes que viene", (9, 0), "tomar pastillas", (10, 0), "tomar pastillas el próximo lunes"),
])
async def test_a_spoken_time_never_picks_one_of_two_similar_reminders(leaves, literal, literal_due, other,
                                                                      other_due, said):
    """User decision t032_design_20261118: Nexa never guesses by the spoken time; the listed number decides."""
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    clock = manager._household_clock
    named = await manager.create("household", literal, _tomorrow_at(leaves, *literal_due), scope=ReminderScope.SHARED)
    timed = await manager.create("household", other, _tomorrow_at(leaves, *other_due), scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, f"cancelá el recordatorio de {said}", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response == (f"Encontré 2 parecidos. 1: {literal} a las {clock.format_time(named.trigger_at)}; "
                              f"2: {other} a las {clock.format_time(timed.trigger_at)}; ¿cuál? Decime el primero o el segundo.")
    assert {r.id for r in await manager.get_active(None, shared_only=True)} == {named.id, timed.id}
    # An ordinal: 'el uno' asks back when the other one is due at 1 (13:00, T032-U1-Q3; pinned below).
    done, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [timed.id]
    assert (await leaves.reminder_store.get_by_id(named.id)).state == "cancelled"


@pytest.mark.parametrize(("text", "response"), [
    ("cancelá el recordatorio de tomar agua y el de sacar la basura",
     "Por voz cancelo de a un recordatorio. Decime uno por vez, por ejemplo: cancelá el recordatorio de tomar agua."),
    ("borrá el recordatorio de tomar agua y la de sacar la basura",
     "Por voz cancelo de a un recordatorio. Decime uno por vez, por ejemplo: cancelá el recordatorio de tomar agua."),
    ("cancelá el recordatorio de tomar agua y el recordatorio de sacar la basura",
     "Por voz cancelo de a un recordatorio. Decime uno por vez, por ejemplo: cancelá el recordatorio de tomar agua."),
])
async def test_two_reminders_in_one_turn_are_asked_one_at_a_time(leaves, text, response):
    """Review T032-U1-R7: 'y el de ...' names a second reminder; nothing is cancelled."""
    result, counted = await _cancel(leaves, text, "reminder_shared_active:tomar agua;"
                                    "reminder_shared_active:sacar la basura;queued_request")
    assert counted == ZERO
    assert (result.response, result.outcome.status.value) == (response, "needs_input")
    assert set(await _states(leaves)) == {"tomar agua", "sacar la basura"}


async def test_the_only_reminder_with_a_clock_in_its_text_is_listed_and_named(leaves):
    """Review T032-U1-RR-F1: the only reminder is offered by number and its text cancels it directly."""
    dispatcher = leaves.dispatcher()
    manager, at = leaves.reminder_manager, _minute_start(1)
    await manager.create("household", "llamar a mamá a las 3", at, scope=ReminderScope.SHARED)
    hm = manager._household_clock.format_time(at)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response == f"Hay uno. 1: llamar a mamá a las 3 a las {hm}; ¿lo cancelo? Decime: el primero."
    done, counted = await _cancel(leaves, "cancelá el recordatorio de llamar a mamá a las 5", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert await manager.get_active(None, shared_only=True) == []


@pytest.mark.parametrize("said", ["llevar el auto al taller mañana", "llevar el auto al taller",
                                  "llevar el auto al taller el lunes"])
async def test_the_only_reminder_with_a_day_in_its_text_is_cancelled_by_its_words(leaves, said):
    await leaves.reminder_manager.create("household", "llevar el auto al taller mañana", _minute_start(1),
                                         scope=ReminderScope.SHARED)
    done, counted = await _cancel(leaves, f"cancelá el recordatorio de {said}")
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed


# ---------------------------------------------------------------------------
# T032-U1 numbered_choice_round: the answer goes back to the open choice
# ---------------------------------------------------------------------------

# Due at 09:00 and 10:00 tomorrow: no bare number 1-5 is a listed due hour (user answer T032-U1-Q3).
TWO_SIMILAR = "reminder_shared_at_0900:tomar agua fría;reminder_shared_at_1000:tomar agua con gas;queued_request"


async def _asked(leaves, dispatcher=None, *, user_id="unknown", room=None):
    dispatcher = dispatcher or leaves.dispatcher()
    await leaves.apply_state(dispatcher, TWO_SIMILAR)
    asked = await dispatcher.dispatch(user_id=user_id, text="cancelá el recordatorio de tomar agua",
                                      context_room_id=room)
    assert asked.response.startswith("Encontré 2 parecidos. 1: tomar agua fría a las ")
    return dispatcher


@pytest.mark.parametrize(("answer", "kept"), [
    ("el primero", "tomar agua con gas"), ("la primera", "tomar agua con gas"), ("el 1", "tomar agua con gas"),
    ("uno", "tomar agua con gas"), ("Nexa, el primero, por favor", "tomar agua con gas"),
    ("cancelá el primero", "tomar agua con gas"), ("el número uno", "tomar agua con gas"),
    ("el dos", "tomar agua fría"), ("la segunda", "tomar agua fría"), ("el 2", "tomar agua fría"),
    ("segundo", "tomar agua fría"), ("sí, el segundo", "tomar agua fría"),
])
async def test_the_numbered_answer_cancels_exactly_the_listed_reminder(leaves, answer, kept):
    dispatcher = await _asked(leaves)
    result, counted = await _cancel(leaves, answer, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert (result.path.value, result.outcome.task_completed, result.outcome.persistence.value) == (
        "fast_reminder", True, "committed")
    assert result.response.startswith("Cancelé el recordatorio tomar agua ")
    assert await _states(leaves) == {kept: "active"}
    assert len(leaves.queue) == 1
    # The choice is spent: the same answer again cancels nothing.
    again, counted = await _cancel(leaves, answer, dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0 and again.path.value != "fast_reminder"
    assert await _states(leaves) == {kept: "active"}


async def test_a_number_that_was_not_listed_cancels_nothing(leaves):
    dispatcher = await _asked(leaves)
    result, counted = await _cancel(leaves, "el tercero", dispatcher=dispatcher)
    assert counted == ZERO
    assert (result.response, result.outcome.status.value) == ("No nombré un 3. No cancelé nada.", "needs_input")
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua con gas"}


@pytest.mark.parametrize("other", ["el de las 2", "el de la tarde", "qué hora es", "el primero de la lista",
                                   "cancelá el recordatorio de regar"])
async def test_any_other_turn_drops_the_choice_and_cancels_nothing(leaves, other):
    dispatcher = await _asked(leaves)
    first, counted = await _cancel(leaves, other, dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0
    late, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0 and late.path.value != "fast_reminder"
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua con gas"}


async def test_an_answer_after_the_window_cancels_nothing(leaves):
    leaves.reminder_manager = ReminderManager(store=leaves.reminder_store, config={"voice_choice_window_s": 5})
    dispatcher = leaves.dispatcher()
    now = [1000.0]
    dispatcher._monotonic = lambda: now[0]
    await _asked(leaves, dispatcher)
    now[0] += 5.5
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response == "Pasó el tiempo para elegir. No cancelé nada; pedímelo de nuevo."
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua con gas"}


async def test_an_answer_inside_the_window_cancels(leaves):
    leaves.reminder_manager = ReminderManager(store=leaves.reminder_store, config={"voice_choice_window_s": 5})
    dispatcher = leaves.dispatcher()
    now = [1000.0]
    dispatcher._monotonic = lambda: now[0]
    await _asked(leaves, dispatcher)
    now[0] += 4.9
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and result.outcome.task_completed


async def test_the_window_comes_from_the_reminder_settings(leaves):
    leaves.reminder_manager = ReminderManager(store=leaves.reminder_store, config={"voice_choice_window_s": 7})
    dispatcher = leaves.dispatcher()
    dispatcher._monotonic = lambda: 100.0
    await _asked(leaves, dispatcher)
    assert dispatcher._reminder_choices[("unknown", None)].deadline == 107.0


@pytest.mark.parametrize(("speaker", "room"), [("ana", None), ("unknown", "hall")])
async def test_another_speaker_or_room_cannot_answer(leaves, speaker, room):
    dispatcher = await _asked(leaves, room=None)
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher, user_id=speaker,
                                    room=room)
    assert counted["reminder_transitions"] == 0 and result.path.value != "fast_reminder"
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua con gas"}
    assert dispatcher.awaits_reminder_choice("unknown", None, "el primero")
    done, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 1 and done.outcome.task_completed


async def test_a_listed_reminder_that_fired_before_the_answer_is_not_reported(leaves):
    dispatcher = await _asked(leaves)
    cold = next(r for r in await leaves.reminder_manager.get_active(None, shared_only=True)
                if r.text == "tomar agua fría")
    assert await leaves.reminder_store.mark_fired(cold.id)
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted == ZERO
    assert (result.response, result.outcome.status.value) == (
        "Ese recordatorio ya no estaba activo. No cancelé nada.", "needs_input")


async def test_a_number_without_an_open_choice_is_not_a_reminder_turn(leaves):
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, TWO_SIMILAR)
    assert not dispatcher.awaits_reminder_choice("unknown", None, "el primero")
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0 and result.path.value != "fast_reminder"


async def test_a_choice_store_failure_is_uncertain(leaves, monkeypatch):
    from unittest.mock import AsyncMock
    dispatcher = await _asked(leaves)
    monkeypatch.setattr(leaves.reminder_manager, "cancel_resolved", AsyncMock(side_effect=RuntimeError("synthetic")))
    result, counted = await _cancel(leaves, "el primero", dispatcher=dispatcher)
    assert (result.success, result.outcome.status.value) == (False, "uncertain")


@pytest.mark.parametrize(("text", "answer"), [
    ("el primero", ("ordinal", 1)), ("la primera", ("ordinal", 1)), ("el 1", ("number", 1)), ("uno", ("number", 1)),
    ("el uno", ("number", 1)), ("el dos", ("number", 2)), ("la segunda", ("ordinal", 2)), ("tres", ("number", 3)),
    ("el tercero", ("ordinal", 3)), ("Nexa, la segunda, por favor", ("ordinal", 2)), ("bueno, el 2", ("number", 2)),
    ("borrá el primero", ("ordinal", 1)), ("el número dos", ("ordinal", 2)), ("la opción 2", ("ordinal", 2)),
    ("el número primero", ("ordinal", 1)), ("la una", ("number", 1)),
    ("el de las 2", None), ("el primero de la lista", None), ("dos pastillas", None), ("el 0", None), ("", None),
    ("el", None), ("primero cancelá", None), ("el número", None),
])
def test_the_numbered_answer_grammar(text, answer):
    """User answer T032-U1-Q3: an ordinal or 'el número N' is never an hour; a bare number may be one."""
    from src.orchestrator.dispatcher import _reminder_choice_answer
    parsed = _reminder_choice_answer(text)
    assert (None if parsed is None else (parsed.form.value, parsed.value)) == answer
    assert parsed is None or parsed.period is None


@pytest.mark.parametrize(("text", "answer"), [
    ("el de las 2", (2, None)), ("de las 2", (2, None)), ("el de la una", (1, None)), ("la de las dos", (2, None)),
    ("el de las 2 de la tarde", (2, "tarde")), ("Nexa, el de las doce de la noche, por favor", (12, "noche")),
    ("cancelá el de las 9 de la mañana", (9, "manana")), ("el de las 3 de la madrugada", (3, "madrugada")),
    ("el de las 13", None), ("el de las 0", None), ("el de la 2", None), ("el de las una", None),
    ("el de las 2 y media", None), ("el de las 2 de la siesta", None), ("las 2", None),
])
def test_the_hour_answer_grammar_after_the_hour_question(text, answer):
    """'el de las N' answers only after Nexa asked '¿El segundo o el de las 2?' (T032-U1-Q3)."""
    from src.orchestrator.dispatcher import _reminder_choice_answer
    parsed = _reminder_choice_answer(text, hours=True)
    assert (None if parsed is None or parsed.form.value != "hour" else (parsed.value, parsed.period)) == answer
    if answer is not None:
        assert _reminder_choice_answer(text) is None


# Review BFR-1: the fillers around an answer are the household classifier's (strip_fillers).
BFR1_FILLED = ("no, {}", "bue, {}", "hola, {}", "porfa {}", "bueno, {}", "dale, {}", "Nexa, no. {}",
               "{}, por favor, porfa", "por favor, {}", "oye, {} porfa")


@pytest.mark.parametrize("filled", BFR1_FILLED)
def test_the_answer_and_the_classifier_ignore_the_same_fillers(filled):
    """'no, el segundo', 'el segundo, dale': the fillers the classifier drops before a request answer the choice."""
    from src.nlu.household_intents import classify_household_intent
    from src.orchestrator.dispatcher import _reminder_choice_answer
    parsed = _reminder_choice_answer(filled.format("el segundo"))
    assert (parsed.form.value, parsed.value) == ("ordinal", 2)
    hour = _reminder_choice_answer(filled.format("el de las 2"), hours=True)
    assert (hour.form.value, hour.value) == ("hour", 2)
    intent = classify_household_intent(filled.format("cancelá el recordatorio de tomar agua"))
    assert (intent.rule, intent.complement) == ("R1", "de tomar agua")


@pytest.mark.parametrize(("filled", "complement"), [
    ("{}, dale", "de tomar agua dale"), ("{} dale", "de tomar agua dale"),
    ("{}, por favor, dale", "de tomar agua por favor dale"), ("{}, dale, porfa", "de tomar agua dale"),
])
def test_a_trailing_dale_ends_an_answer_but_not_a_reminder_name(filled, complement):
    """Review BFR1-R1: 'el segundo, dale' answers; the classifier keeps a trailing 'dale' (it may be a name)."""
    from src.nlu.household_intents import classify_household_intent
    from src.orchestrator.dispatcher import _reminder_choice_answer
    parsed = _reminder_choice_answer(filled.format("el segundo"))
    assert (parsed.form.value, parsed.value) == ("ordinal", 2)
    hour = _reminder_choice_answer(filled.format("el de las 2"), hours=True)
    assert (hour.form.value, hour.value) == ("hour", 2)
    intent = classify_household_intent(filled.format("cancelá el recordatorio de tomar agua"))
    assert (intent.rule, intent.complement) == ("R1", complement)


@pytest.mark.parametrize("text", ["no el segundo", "el segundo, no", "no, no lo sé", "hola", "dale", "no",
                                  "che, el primero de la lista", "el segundo y el tercero"])
def test_a_filler_never_makes_another_turn_an_answer(text):
    """A 'no' without a pause stays a negation (user default T032-U1-Q4); fillers alone answer nothing."""
    from src.orchestrator.dispatcher import _reminder_choice_answer
    assert _reminder_choice_answer(text) is None
    assert _reminder_choice_answer(text, hours=True) is None


@pytest.mark.parametrize(("answer", "kept"), [
    ("no, el segundo", "tomar agua fría"), ("el segundo, dale", "tomar agua fría"),
    ("bue, el primero", "tomar agua con gas"), ("hola, la segunda", "tomar agua fría"),
    ("porfa el primero", "tomar agua con gas"),
])
async def test_a_filled_answer_cancels_the_listed_reminder(leaves, answer, kept):
    dispatcher = await _asked(leaves)
    assert dispatcher.awaits_reminder_choice("unknown", None, answer)
    result, counted = await _cancel(leaves, answer, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and result.outcome.task_completed
    assert await _states(leaves) == {kept: "active"}


@pytest.mark.parametrize(("follow_up", "cancelled"), [
    ("no, el de las 2", "tomar agua tibia"), ("no, el segundo", "tomar agua con gas"),
    ("el de las 2, dale", "tomar agua tibia"), ("bue, el de las 9", "tomar agua fría"),
])
async def test_a_filled_answer_to_the_hour_question_cancels_exactly_one(leaves, follow_up, cancelled):
    """Review BFR-1: 'no, el de las 2' after '¿El segundo o el de las 2?' answers by hour."""
    dispatcher, _ = await _listed_at(leaves)
    asked, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El segundo o el de las 2?"
    assert not dispatcher.awaits_reminder_choice("unknown", None, "no el de las 2")
    done, counted = await _cancel(leaves, follow_up, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith(f"Cancelé el recordatorio {cancelled} a las ")
    assert set(await _states(leaves)) == {text for text, _ in THREE_WATERS} - {cancelled}


async def test_a_configured_wake_word_before_the_answer_is_ignored(leaves):
    classifier = HouseholdIntentClassifier(HouseholdIntentSettings.from_config({}, wake_words=["Jarvis"]))
    dispatcher = await _asked(leaves, leaves.dispatcher(household_intents=classifier))
    assert dispatcher.awaits_reminder_choice("unknown", None, "Jarvis, el segundo")
    result, counted = await _cancel(leaves, "Jarvis, el segundo", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert await _states(leaves) == {"tomar agua fría": "active"}
    # Without the configured wake word the same turn is not an answer.
    from src.orchestrator.dispatcher import _reminder_choice_answer
    assert _reminder_choice_answer("Jarvis, el segundo") is None


@pytest.mark.parametrize(("said", "meant"), [
    ("cancelá el recordatorio de decirle que dale", "decirle que no"),
    ("cancelá el recordatorio de decirle que, dale", "decirle que no"),
    ("cancelá el recordatorio de llamar a Dale", "llamar a juan"),
    ("cancelá el recordatorio de pagarle a dale", "pagarle a juan"),
])
async def test_a_trailing_dale_in_the_name_never_cancels_another_reminder(leaves, said, meant):
    """Review BFR1-R1: 'decirle que dale' never cancels 'decirle que no' directly (T032-NUM-FV-1).

    At 03967c7d the trailing 'dale' was dropped as a filler, the reference
    became 'decirle que' and 'decirle que no' was cancelled with no question.
    Now it is listed for a numbered confirmation with zero effects, and a
    filled answer ('el primero, dale') cancels it.
    """
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    similar = await manager.create("household", meant, _minute_start(1), scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, said, "queued_request", dispatcher=dispatcher)
    assert counted == ZERO
    assert (asked.success, asked.outcome.status.value, asked.outcome.persistence.value) == (
        False, "needs_input", "not_written")
    assert asked.response == (f"Encontré uno parecido, no igual. 1: {manager.voice_label(similar)}; "
                              "¿lo cancelo? Decime: el primero.")
    assert set(await _states(leaves)) == {meant} and len(leaves.queue) == 1
    done, counted = await _cancel(leaves, "el primero, dale", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert done.response == f"Cancelé el recordatorio {manager.voice_label(similar)}."
    assert await _states(leaves) == {}


# ---------------------------------------------------------------------------
# User answer T032-U1-Q3 (t032_num_fv4_20261118): an ordinal always answers; a
# bare number only when no other listed reminder is due at that hour (12-hour
# reading, 14:00 -> 2); on a clash Nexa asks '¿El segundo o el de las 2?'.
# ---------------------------------------------------------------------------

THREE_WATERS = (("tomar agua fría", 9), ("tomar agua con gas", 10), ("tomar agua tibia", 14))


async def _listed_at(leaves, specs=THREE_WATERS, *, room=None):
    """Shared reminders due tomorrow at the given hours, listed by one cancel; returns dispatcher and reminders."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "queued_request")
    manager = leaves.reminder_manager
    dues = [due if isinstance(due, tuple) else (due,) for _, due in specs]  # hour or (hour, minute)
    created = [await manager.create("household", text, _tomorrow_at(leaves, *due), scope=ReminderScope.SHARED)
               for (text, _), due in zip(specs, dues)]
    asked = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de tomar agua",
                                      context_room_id=room)
    assert asked.response.startswith(f"Encontré {len(specs)} parecidos. 1: {specs[0][0]} a las "), asked.response
    return dispatcher, created


@pytest.mark.parametrize("bare", ["dos", "el 2", "el dos", "Nexa, el dos", "cancelá el 2", "el dos, por favor"])
async def test_a_bare_number_that_is_another_listed_due_hour_asks_and_cancels_nothing(leaves, bare):
    """'dos' with 'tomar agua tibia' due at 14:00 listed third: the second or the one at 2? Nothing is cancelled."""
    dispatcher, created = await _listed_at(leaves)
    asked, counted = await _cancel(leaves, bare, dispatcher=dispatcher)
    assert counted == ZERO
    assert (asked.path.value, asked.success, asked.outcome.status.value, asked.outcome.persistence.value) == (
        "fast_reminder", False, "needs_input", "not_written")
    assert asked.response == "¿El segundo o el de las 2?"
    assert set(await _states(leaves)) == {text for text, _ in THREE_WATERS} and len(leaves.queue) == 1
    assert dispatcher.awaits_reminder_choice("unknown", None, "el de las 2")
    assert dispatcher.awaits_reminder_choice("unknown", None, "el segundo")


@pytest.mark.parametrize(("follow_up", "cancelled"), [
    ("el segundo", "tomar agua con gas"), ("la segunda", "tomar agua con gas"),
    ("el número dos", "tomar agua con gas"), ("el de las 2", "tomar agua tibia"),
    ("el de las dos de la tarde", "tomar agua tibia"), ("Nexa, el de las 2, por favor", "tomar agua tibia"),
    ("el de las 9", "tomar agua fría"), ("el primero", "tomar agua fría"),
])
async def test_after_the_hour_question_an_ordinal_or_the_hour_cancels_exactly_one(leaves, follow_up, cancelled):
    dispatcher, created = await _listed_at(leaves)
    asked, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El segundo o el de las 2?"
    done, counted = await _cancel(leaves, follow_up, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert (done.path.value, done.outcome.task_completed, done.outcome.persistence.value) == (
        "fast_reminder", True, "committed")
    assert done.response.startswith(f"Cancelé el recordatorio {cancelled} a las ")
    assert set(await _states(leaves)) == {text for text, _ in THREE_WATERS} - {cancelled}
    assert not dispatcher.awaits_reminder_choice("unknown", None, "el primero")


@pytest.mark.parametrize(("follow_up", "response"), [
    ("dos", "¿El segundo o el de las 2?"),
    ("el de las 2 de la mañana", "No nombré uno de las 2. No cancelé nada."),
    ("el de las 5", "No nombré uno de las 5. No cancelé nada."),
])
async def test_after_the_hour_question_an_unclear_answer_cancels_nothing(leaves, follow_up, response):
    dispatcher, _ = await _listed_at(leaves)
    await _cancel(leaves, "el 2", dispatcher=dispatcher)
    again, counted = await _cancel(leaves, follow_up, dispatcher=dispatcher)
    assert counted == ZERO and (again.response, again.outcome.status.value) == (response, "needs_input")
    assert set(await _states(leaves)) == {text for text, _ in THREE_WATERS}


# T032-NUM-FV-5: "12 de la mañana" is ambiguous when a reminder is created, so after
# the hour question it must not narrow to 00:xx and cancel the reminder due at 00:30.
NOON_CLASH = (("tomar agua fría", (0, 30)), ("tomar agua con gas", 12), ("tomar agua tibia", 14))


@pytest.mark.parametrize(("follow_up", "response"), [
    ("el de las 12 de la mañana", "Hay más de uno de las 12: decime el primero o el segundo."),
    ("el de las doce de la mañana", "Hay más de uno de las 12: decime el primero o el segundo."),
    ("el de las 12 de la noche", "Hay más de uno de las 12: decime el primero o el segundo."),
    ("el de las 12", "Hay más de uno de las 12: decime el primero o el segundo."),
])
async def test_twelve_in_the_morning_after_the_hour_question_asks_by_ordinal(leaves, follow_up, response):
    """Listed 00:30, 12:00 and 14:00: '12 de la mañana' may mean either of the first two, so nothing is cancelled."""
    dispatcher, _ = await _listed_at(leaves, NOON_CLASH)
    asked, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El segundo o el de las 2?"
    again, counted = await _cancel(leaves, follow_up, dispatcher=dispatcher)
    assert counted == ZERO
    assert (again.response, again.outcome.status.value, again.outcome.persistence.value) == (
        response, "needs_input", "not_written")
    assert set(await _states(leaves)) == {text for text, _ in NOON_CLASH}
    assert dispatcher.awaits_reminder_choice("unknown", None, "el primero")
    done, counted = await _cancel(leaves, "el segundo", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua con gas a las ")
    assert set(await _states(leaves)) == {"tomar agua fría", "tomar agua tibia"}


@pytest.mark.parametrize(("follow_up", "cancelled"), [
    ("el de las 2 de la tarde", "tomar agua tibia"), ("el de las 12 de la tarde", "tomar agua con gas"),
    ("el de las 2", "tomar agua tibia"),
])
async def test_a_period_the_creation_table_holds_still_names_one_reminder(leaves, follow_up, cancelled):
    dispatcher, _ = await _listed_at(leaves, NOON_CLASH)
    await _cancel(leaves, "dos", dispatcher=dispatcher)
    done, counted = await _cancel(leaves, follow_up, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith(f"Cancelé el recordatorio {cancelled} a las ")
    assert set(await _states(leaves)) == {text for text, _ in NOON_CLASH} - {cancelled}


@pytest.mark.parametrize("period", ["manana", "madrugada", "tarde"])
@pytest.mark.parametrize("hour", range(1, 13))
def test_a_narrowing_period_reads_the_hour_as_reminder_creation_does(period, hour):
    """T032-NUM-FV-5: a period narrows the hour only where creating a reminder resolves it; else both readings."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.orchestrator.dispatcher import _choice_hours
    from src.reminders.voice_request import ReminderAmbiguity, parse_reminder_request
    zone = ZoneInfo("America/Montevideo")
    word = {"manana": "mañana"}.get(period, period)
    parsed = parse_reminder_request(f"recordame regar el jueves a las {hour} de la {word}",
                                    datetime(2026, 11, 16, 10, 0, tzinfo=zone))
    assert parsed is not None, hour
    if parsed.ok:
        expected = {datetime.fromtimestamp(parsed.trigger_at, zone).hour}
    elif parsed.detail == "midnight_with_day":  # the hour resolved to 00:00; only the day is asked
        expected = {0}
    else:
        assert parsed.issues == (ReminderAmbiguity.AMBIGUOUS_MERIDIEM,), parsed
        expected = {hour % 12, hour % 12 + 12}
    assert _choice_hours(hour, period) == expected


async def test_the_hour_question_keeps_the_choice_for_a_fresh_window(leaves):
    leaves.reminder_manager = ReminderManager(store=leaves.reminder_store, config={"voice_choice_window_s": 5})
    now = [1000.0]
    dispatcher, _ = await _listed_at(leaves)
    dispatcher._monotonic = lambda: now[0]
    # Reopen the list under the controlled clock.
    await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    now[0] += 4.0
    asked, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El segundo o el de las 2?"
    assert dispatcher._reminder_choices[("unknown", None)].deadline == 1009.0
    now[0] += 4.0
    done, counted = await _cancel(leaves, "el de las 2", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed


async def test_an_ordinal_always_answers_even_when_its_number_is_a_listed_due_hour(leaves):
    dispatcher, _ = await _listed_at(leaves)
    done, counted = await _cancel(leaves, "el segundo", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua con gas a las ")


@pytest.mark.parametrize("explicit", ["el número dos", "la opción 2", "número 2"])
async def test_el_numero_n_is_never_an_hour(leaves, explicit):
    """'el número dos' cannot mean a time of day, so it names the listed number like an ordinal."""
    dispatcher, _ = await _listed_at(leaves)
    done, counted = await _cancel(leaves, explicit, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua con gas a las ")


@pytest.mark.parametrize(("bare", "cancelled"), [("dos", "tomar agua tibia"), ("el 1", "tomar agua fría"),
                                                 ("tres", None)])
async def test_a_bare_number_whose_hour_is_its_own_listed_reminder_cancels_it(leaves, bare, cancelled):
    """Both readings name the same reminder: 'dos' when the second listed is the one due at 14:00."""
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 1), ("tomar agua tibia", 14)))
    done, counted = await _cancel(leaves, bare, dispatcher=dispatcher)
    if cancelled is None:
        assert counted == ZERO and done.response == "No nombré un 3. No cancelé nada."
        return
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith(f"Cancelé el recordatorio {cancelled} a las ")


async def test_a_bare_number_with_no_listed_due_hour_cancels_the_listed_number(leaves):
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 9), ("tomar agua tibia", 21)))
    done, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua tibia a las ")


async def test_a_bare_number_one_with_a_reminder_due_at_one_asks_with_la_una(leaves):
    """Review T032-NUM-FV-4: 'el uno' with a reminder due at 13:00 listed second; 1 reads 'de la una'."""
    dispatcher, created = await _listed_at(leaves, (("tomar agua fría", 9), ("tomar agua a la una", 13)))
    asked, counted = await _cancel(leaves, "el uno", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El primero o el de la una?"
    done, counted = await _cancel(leaves, "el de la una", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua a la una a las ")
    assert await _states(leaves) == {"tomar agua fría": "active"}


async def test_a_bare_number_not_listed_but_a_listed_due_hour_points_to_its_ordinal(leaves):
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 9), ("tomar agua tibia", 15)))
    asked, counted = await _cancel(leaves, "tres", dispatcher=dispatcher)
    assert counted == ZERO
    assert (asked.response, asked.outcome.status.value) == (
        "No nombré un 3. Si querés el de las 3, decime el segundo.", "needs_input")
    done, counted = await _cancel(leaves, "el de las 3", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua tibia a las ")


async def test_an_unlisted_ordinal_is_never_read_as_an_hour(leaves):
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 9), ("tomar agua tibia", 15)))
    result, counted = await _cancel(leaves, "el tercero", dispatcher=dispatcher)
    assert counted == ZERO and result.response == "No nombré un 3. No cancelé nada."
    assert not dispatcher.awaits_reminder_choice("unknown", None, "el de las 3")


async def test_two_listed_reminders_at_the_hour_ask_by_ordinal(leaves):
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 2), ("tomar agua con gas", 9),
                                              ("tomar agua tibia", 14)))
    asked, counted = await _cancel(leaves, "dos", dispatcher=dispatcher)
    assert counted == ZERO and asked.response == "¿El segundo o el de las 2?"
    again, counted = await _cancel(leaves, "el de las 2", dispatcher=dispatcher)
    assert counted == ZERO and again.response == "Hay más de uno de las 2: decime el primero o el tercero."
    done, counted = await _cancel(leaves, "el de las 2 de la tarde", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua tibia a las ")


async def test_two_listed_reminders_at_an_unlisted_number_ask_by_ordinal(leaves):
    dispatcher, _ = await _listed_at(leaves, (("tomar agua fría", 3), ("tomar agua tibia", 15)))
    asked, counted = await _cancel(leaves, "el 3", dispatcher=dispatcher)
    assert counted == ZERO
    assert asked.response == "No nombré un 3. Hay más de uno de las 3: decime el primero o el segundo."
    done, counted = await _cancel(leaves, "el segundo", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert done.response.startswith("Cancelé el recordatorio tomar agua tibia a las ")


async def test_the_hour_is_no_answer_before_the_hour_question(leaves):
    """The choice is by number (t032_design_20261118); 'el de las 2' answers only the question that offered it."""
    dispatcher, _ = await _listed_at(leaves)
    assert not dispatcher.awaits_reminder_choice("unknown", None, "el de las 2")
    result, counted = await _cancel(leaves, "el de las 2", dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0
    late, counted = await _cancel(leaves, "el segundo", dispatcher=dispatcher)
    assert counted["reminder_transitions"] == 0 and late.path.value != "fast_reminder"


async def test_the_hour_question_waits_on_the_same_speaker_and_room(leaves):
    dispatcher, _ = await _listed_at(leaves, room="living")
    asked, _ = await _cancel(leaves, "dos", dispatcher=dispatcher, room="living")
    assert asked.response == "¿El segundo o el de las 2?"
    assert not dispatcher.awaits_reminder_choice("unknown", "hall", "el de las 2")
    other, counted = await _cancel(leaves, "el de las 2", dispatcher=dispatcher, room="hall")
    assert counted["reminder_transitions"] == 0 and other.path.value != "fast_reminder"
    done, counted = await _cancel(leaves, "el de las 2", dispatcher=dispatcher, room="living")
    assert counted["reminder_transitions"] == 1 and done.outcome.task_completed


async def test_the_orchestrator_tells_the_router_when_a_household_answer_is_awaited(leaves):
    orchestrator = MultiUserOrchestrator(
        chroma_sync=leaves.chroma, ha_client=leaves.ha, routine_manager=leaves.routines, router=leaves.router,
        list_manager=leaves.list_manager, reminder_manager=leaves.reminder_manager)
    leaves.attach(orchestrator.dispatcher)
    await leaves.apply_state(orchestrator.dispatcher, TWO_SIMILAR)
    assert not orchestrator.awaits_reminder_choice("el primero", "living")
    await orchestrator.process(user_id=None, text="cancelá el recordatorio de tomar agua", context_room_id="living")
    assert orchestrator.awaits_reminder_choice("el primero", "living")
    assert not orchestrator.awaits_reminder_choice("el primero", "hall")
    assert not orchestrator.awaits_reminder_choice("qué hora es", "living")
    done = await orchestrator.process(user_id=None, text="el segundo", context_room_id="living")
    assert done.outcome.task_completed and await _states(leaves) == {"tomar agua fría": "active"}


async def test_olvidate_ya_del_recordatorio_cancels_exactly_that_reminder(leaves):
    """Review T032-U1-RR-F4: 'ya' between 'olvidate' and 'del' is a filler, not a creation request."""
    result, counted = await _cancel(leaves, "olvidate ya del recordatorio de tomar agua",
                                    "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura")
    assert counted == effects(reminder_transitions=1) and result.outcome.task_completed
    assert result.response.startswith("Cancelé el recordatorio tomar agua a las ")
    assert await _states(leaves) == {"sacar la basura": "active"}


# --------------------------------------------------------------------------
# T031-U2: voice reminder creation (text only) on the P3-F2 reminder family.
# --------------------------------------------------------------------------

def _frozen_leaves(leaves, dispatcher, local_now):
    """Freeze the household clock of ``dispatcher`` and of the reminder manager at ``local_now``."""
    from src.core.household_clock import HouseholdClock
    clock = HouseholdClock(time_source=lambda: local_now.timestamp())
    dispatcher.household_clock = clock
    leaves.reminder_manager._household_clock = clock
    return clock


def _monday_10():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(2026, 11, 16, 10, 0, tzinfo=ZoneInfo("America/Montevideo"))


def _rows(db):
    with sqlite3.connect(db) as conn:
        return [tuple(r) for r in conn.execute(
            "SELECT user_id, scope, text, trigger_at, recurrence, ha_actions, state FROM reminders")]


EXAMPLE = "recordame sacar la basura mañana a las 8 de la noche"
ALARM_EXAMPLE = "poné una alarma mañana a las 7 de la mañana"


@pytest.mark.parametrize(("text", "response"), [
    ("recordame mañana a las 9 de la mañana",
     f"¿Qué querés que te recuerde? Decímelo completo, por ejemplo: {EXAMPLE}."),
    ("recuérdame tomar agua", f"¿Para cuándo? Decímelo completo, con el día y la hora, por ejemplo: {EXAMPLE}."),
    ("recordame sacar turno en un rato",
     f"¿Para cuándo? Decímelo completo, con el día y la hora, por ejemplo: {EXAMPLE}."),
    ("recordame el lunes a las 9 de la mañana sacar turno",
     "No sé para qué día es. Decímelo completo nombrando el día, hoy, mañana o el martes, por ejemplo: "
     f"{EXAMPLE}."),
    ("recordame tomar la pastilla a las 8",
     f"¿Es de la mañana, de la tarde o de la noche? Decímelo completo con la hora, por ejemplo: {EXAMPLE}."),
    ("recordame a las 9 de la mañana llamar al médico",
     f"Esa hora de hoy ya pasó. Si es para mañana, decímelo completo con el día, por ejemplo: {EXAMPLE}."),
    ("recordame a las 25 llamar",
     f"No entendí bien la fecha o la hora. Decímelo completo con un día y una hora exacta, por ejemplo: {EXAMPLE}."),
    ("recordame el 5 de diciembre pagar la luz",
     "Por ahora programo recordatorios para hoy, mañana, pasado mañana, un día de la semana, todos los días o de "
     f"lunes a viernes; no por fecha ni por mes. Decímelo completo así, por ejemplo: {EXAMPLE}."),
    # Review T031-U2-RV5: a form that can be said aloud and that both R3a and the parser read as a clock time.
    ("recordame a las 20 y 5 sacar la basura",
     "No sé si ese número es parte de la hora o de lo que hay que recordar. Decímelo completo diciendo los "
     "minutos, por ejemplo: recordame a las 8 y 5 minutos de la noche sacar la basura."),
    ("poné una alarma a las 20 y 5 tomar la pastilla",
     "No sé si ese número es parte de la hora o de lo que hay que recordar. Decímelo completo diciendo los "
     "minutos, por ejemplo: poné una alarma a las 7 y 15 minutos de la mañana."),
    ("poné una alarma a las 7 y 15 minutos de la mañana",
     f"Esa hora de hoy ya pasó. Si es para mañana, decímelo completo con el día, por ejemplo: {ALARM_EXAMPLE}."),
    # Review T031-U2-RV7: a time with nothing to remember asks what.
    ("avisame cuando sean las 9 de la noche", f"¿Qué querés que te recuerde? Decímelo completo, por ejemplo: {EXAMPLE}."),
    ("recordame nada a las 9 de la noche", f"¿Qué querés que te recuerde? Decímelo completo, por ejemplo: {EXAMPLE}."),
    ("recordame a las 8 de la noche y a las 9 de la noche llamar",
     f"Escuché más de una hora. Decímelo completo con una sola, por ejemplo: {EXAMPLE}."),
    ("recordame en 10 minutos llamar a las 8 de la noche",
     f"Escuché más de una hora. Decímelo completo con una sola, por ejemplo: {EXAMPLE}."),
    ("recordame en la cocina apagar el horno en 10 minutos",
     "Por ahora no elijo en qué habitación suena un recordatorio. Decímelo de nuevo sin nombrar la habitación."),
    ("recordarme sacar la basura a las 8 de la noche",
     f"No entendí el recordatorio. Decímelo completo, por ejemplo: {EXAMPLE}."),
    ("poné una alarma a las 7",
     f"¿Es de la mañana, de la tarde o de la noche? Decímelo completo con la hora, por ejemplo: {ALARM_EXAMPLE}."),
    ("poné un despertador a las 9 de la mañana",
     f"Esa hora de hoy ya pasó. Si es para mañana, decímelo completo con el día, por ejemplo: {ALARM_EXAMPLE}."),
])
async def test_each_ambiguity_asks_for_the_whole_request_and_writes_nothing(leaves, text, response):
    """Single turn, always ask (user answers Q7/Q8): a precise question, NEEDS_INPUT, zero rows."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.success, result.response) == (PathType.FAST_REMINDER, False, response)
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("needs_input", "not_written")
    assert leaves.effects(before) == ZERO


@pytest.mark.parametrize("text", [
    "recordame a las 9 de la noche que mañana viene el plomero",
    "recordame a las 10 de la noche que el martes hay reunión",
    "recordame a las 10 de la noche que pasado mañana viene la abuela",
    "recordame a las 10 de la noche llevar mañana el auto al taller",
    "recordame mañana a las 8 de la noche que el informe es para el martes",
])
async def test_a_day_word_inside_the_reminder_text_asks_and_writes_nothing(leaves, text):
    """Review T031-U2-RV1 (user answer T032-U1-RR-P1, 'Pregunta cuál'): never a row for a guessed day."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.success, result.response) == (
        PathType.FAST_REMINDER, False,
        f"No sé para qué día es. Decímelo completo nombrando el día, hoy, mañana o el martes, por ejemplo: {EXAMPLE}.")
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("needs_input", "not_written")
    assert leaves.effects(before) == ZERO
    assert _rows(leaves.reminders_db) == []


async def test_every_ambiguity_code_has_its_question():
    from src.orchestrator.dispatcher import _REMINDER_QUESTIONS
    from src.reminders.voice_request import ReminderAmbiguity
    assert set(_REMINDER_QUESTIONS) == set(ReminderAmbiguity)


@pytest.mark.parametrize(("text", "row", "response"), [
    ("recordame sacar la basura mañana a las 8 de la noche", ("sacar la basura", (2026, 11, 17, 20, 0), None),
     "Anotado: sacar la basura, mañana a las 20:00."),
    ("Nexa, recuérdame pasado mañana a las 3 de la tarde ir al banco", ("ir al banco", (2026, 11, 18, 15, 0), None),
     "Anotado: ir al banco, pasado mañana a las 15:00."),
    ("acordame de pagar la luz para el jueves a las 9 de la mañana", ("pagar la luz", (2026, 11, 19, 9, 0), None),
     "Anotado: pagar la luz, el jueves a las 09:00."),
    ("recordame en 20 minutos sacar la pizza", ("sacar la pizza", (2026, 11, 16, 10, 20), None),
     "Anotado: sacar la pizza, hoy a las 10:20."),
    ("avisame a las 21 que llame a mamá", ("llame a mamá", (2026, 11, 16, 21, 0), None),
     "Anotado: llame a mamá, hoy a las 21:00."),
    ("recordame regar las plantas todos los días a las 7 de la mañana",
     ("regar las plantas", (2026, 11, 17, 7, 0), "daily"), "Anotado: regar las plantas, todos los días a las 07:00."),
    ("recordame sacar la basura de lunes a viernes a las 20", ("sacar la basura", (2026, 11, 16, 20, 0), "weekdays"),
     "Anotado: sacar la basura, de lunes a viernes a las 20:00."),
    ("recordame llamar a la abuela todos los sábados a las 10 de la mañana",
     ("llamar a la abuela", (2026, 11, 21, 10, 0), "weekly:6"),
     "Anotado: llamar a la abuela, todos los sábados a las 10:00."),
    ("poné una alarma a las 7 de la tarde", ("alarma", (2026, 11, 16, 19, 0), None),
     "Anotado: alarma, hoy a las 19:00."),
    ("poné un despertador mañana a las 6 y media de la mañana", ("despertador", (2026, 11, 17, 6, 30), None),
     "Anotado: despertador, mañana a las 06:30."),
    ("programá la alarma para las 19 para ir al médico", ("alarma para ir al médico", (2026, 11, 16, 19, 0), None),
     "Anotado: alarma para ir al médico, hoy a las 19:00."),
    ("recordame apagar la luz del living a las 11 de la noche",
     ("apagar la luz del living", (2026, 11, 16, 23, 0), None),
     "Anotado: apagar la luz del living, hoy a las 23:00. Es solo un aviso: eso no lo hago yo."),
    ("recordame poner música a las 9 de la noche", ("poner música", (2026, 11, 16, 21, 0), None),
     "Anotado: poner música, hoy a las 21:00. Es solo un aviso: eso no lo hago yo."),
    # Review T031-U2-RV4: locks, doors, gates and the alarm panel are never controllable; Q9d says so aloud.
    ("recordame cerrar la puerta con llave a las 11 de la noche",
     ("cerrar la puerta con llave", (2026, 11, 16, 23, 0), None),
     "Anotado: cerrar la puerta con llave, hoy a las 23:00. Es solo un aviso: eso no lo hago yo."),
    ("recordame activar la alarma a las 11 de la noche", ("activar la alarma", (2026, 11, 16, 23, 0), None),
     "Anotado: activar la alarma, hoy a las 23:00. Es solo un aviso: eso no lo hago yo."),
    ("recordame desactivar la alarma a las 7 de la tarde", ("desactivar la alarma", (2026, 11, 16, 19, 0), None),
     "Anotado: desactivar la alarma, hoy a las 19:00. Es solo un aviso: eso no lo hago yo."),
    ("recordame abrir el portón a las 8 de la noche", ("abrir el portón", (2026, 11, 16, 20, 0), None),
     "Anotado: abrir el portón, hoy a las 20:00. Es solo un aviso: eso no lo hago yo."),
    ("recordame pintar la puerta a las 8 de la noche", ("pintar la puerta", (2026, 11, 16, 20, 0), None),
     "Anotado: pintar la puerta, hoy a las 20:00."),
    # Review T031-U2-RV5: the minutes example works as said, for a reminder and for an alarm.
    ("recordame a las 8 y 5 minutos de la noche sacar la basura", ("sacar la basura", (2026, 11, 16, 20, 5), None),
     "Anotado: sacar la basura, hoy a las 20:05."),
    ("poné una alarma a las 20 y 5 minutos para la reunión",
     ("alarma para la reunión", (2026, 11, 16, 20, 5), None), "Anotado: alarma para la reunión, hoy a las 20:05."),
    # Review T031-U2-RV7: common rioplatense forms.
    ("acordate de sacar la basura a las 8 de la noche", ("sacar la basura", (2026, 11, 16, 20, 0), None),
     "Anotado: sacar la basura, hoy a las 20:00."),
    ("avisame cuando sean las 9 de la noche que hay que salir", ("hay que salir", (2026, 11, 16, 21, 0), None),
     "Anotado: hay que salir, hoy a las 21:00."),
    ("poné una alarma al mediodía", ("alarma", (2026, 11, 16, 12, 0), None), "Anotado: alarma, hoy a las 12:00."),
])
@pytest.mark.parametrize("speaker", ["unknown", "ana"])
async def test_voice_creation_commits_one_shared_household_row(leaves, text, row, response, speaker):
    """T031-U2: user_id 'household', scope shared, no home actions, whatever label spoke; success from the commit."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id=speaker, text=text, service_filter="turn_off")
    assert (result.path, result.success, result.response) == (PathType.FAST_REMINDER, True, response)
    outcome = result.outcome
    assert (outcome.status.value, outcome.persistence.value, outcome.effect.value) == (
        "completed", "committed", "observed")
    assert outcome.task_completed
    assert leaves.effects(before) == effects(reminder_rows_delta=1)
    expected = datetime(*row[1], tzinfo=ZoneInfo("America/Montevideo")).timestamp()
    assert _rows(leaves.reminders_db) == [("household", "shared", row[0], expected, row[2], None, "active")]


async def test_a_store_failure_is_uncertain_and_never_confirmed(leaves, monkeypatch):
    """Success only from the committed insert; an exception from the store is UNCERTAIN (the row may exist)."""
    from unittest.mock import AsyncMock
    dispatcher = leaves.dispatcher()
    monkeypatch.setattr(leaves.reminder_store, "create", AsyncMock(side_effect=sqlite3.OperationalError("locked")))
    result = await dispatcher.dispatch(user_id="unknown", text="recordame en 10 minutos sacar la pasta")
    assert (result.path, result.success) == (PathType.FAST_REMINDER, False)
    assert result.response == ("No pude confirmar si el recordatorio quedó guardado. Preguntame qué recordatorios "
                               "hay antes de pedirlo de nuevo.")
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("uncertain", "uncertain")
    assert not result.outcome.task_completed
    leaves.reminder_store.create.assert_awaited_once()


async def test_an_unreadable_household_clock_writes_nothing(leaves):
    from src.core.household_clock import HouseholdClock
    dispatcher = leaves.dispatcher()
    dispatcher.household_clock = HouseholdClock(time_source=lambda: float("nan"))
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="recordame en 10 minutos sacar la pasta")
    assert (result.path, result.response) == (PathType.FAST_REMINDER, "No pude leer el reloj de la casa. No guardé nada.")
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("failed", "not_written")
    assert leaves.effects(before) == ZERO


async def test_voice_creation_never_stores_home_actions(leaves, monkeypatch):
    """T031-U2 is text only: ReminderManager.create is called with ha_actions=None, the household and shared."""
    from unittest.mock import AsyncMock
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "light_match")
    create = AsyncMock(wraps=leaves.reminder_manager.create)
    monkeypatch.setattr(leaves.reminder_manager, "create", create)
    result = await dispatcher.dispatch(user_id="ana", text="recordame apagar la luz del living en 10 minutos",
                                       service_filter="turn_off", target_entity_id="light.living")
    assert result.outcome.status.value == "completed"
    args, kwargs = create.await_args
    assert args[:2] == ("household", "apagar la luz del living")
    assert kwargs["ha_actions"] is None and kwargs["scope"] is ReminderScope.SHARED
    assert leaves.home.await_count == 0 and leaves.chroma.asearch_command.await_count == 0


async def test_unclaimed_reminder_words_ask_how_to_create_one(leaves):
    """No reminder verb at the head: never read as a creation; the reply says how to ask for one."""
    dispatcher = leaves.dispatcher()
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="creá un recordatorio para mañana a las 8 de la noche")
    assert (result.path, result.success) == (PathType.FAST_REMINDER, False)
    assert result.response == f"Para crear un recordatorio decímelo así, por ejemplo: {EXAMPLE}."
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("needs_input", "not_written")
    assert leaves.effects(before) == ZERO


async def test_a_created_reminder_is_listed_and_cancelled_by_voice(leaves):
    """Create -> query -> cancel on the same shared row (T031-U2 with T032-U1)."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    created = await dispatcher.dispatch(user_id="unknown", text="recordame sacar la basura mañana a las 8 de la noche")
    assert created.success
    listed = await dispatcher.dispatch(user_id="unknown", text="¿qué recordatorios hay?")
    assert "sacar la basura a las 20:00" in listed.response
    before = leaves.snapshot()
    cancelled = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de sacar la basura")
    assert cancelled.response == "Cancelé el recordatorio sacar la basura a las 20:00."
    assert leaves.effects(before) == effects(reminder_transitions=1)


async def test_clock_alarm_cancel_lists_the_alarms_and_the_answer_cancels_only_that_one(leaves):
    """T030-U1-Q8 (A): 'cancelá la alarma de las 7' never cancels the timer nor guesses by the spoken time."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    for text in ("poné una alarma a las 7 de la tarde", "poné un despertador mañana a las 7 de la mañana",
                 "recordame sacar la basura a las 7 de la tarde"):
        assert (await dispatcher.dispatch(user_id="unknown", text=text)).success
    before = leaves.snapshot()
    asked = await dispatcher.dispatch(user_id="unknown", text="cancelá la alarma de las 7")
    assert asked.response == ("Hay 2. 1: alarma a las 19:00; 2: despertador a las 07:00; ¿cuál? Decime el primero "
                              "o el segundo.")
    assert asked.outcome.status.value == "needs_input" and leaves.effects(before) == ZERO
    answered = await dispatcher.dispatch(user_id="unknown", text="el segundo")
    assert answered.response == "Cancelé el recordatorio despertador a las 07:00."
    assert leaves.effects(before) == effects(reminder_transitions=1)
    remaining = sorted(r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True))
    assert remaining == ["alarma", "sacar la basura"]
    assert [t.state.value for t in leaves.timers.get_all_timers()] == ["running"]


async def test_clock_alarm_query_reads_the_shared_alarms_only(leaves):
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    assert (await dispatcher.dispatch(user_id="unknown", text="¿hay alguna alarma a las 7?")).response == (
        "No hay alarmas con hora guardadas.")
    await leaves.apply_state(dispatcher, "reminder_private_active:alarma privada")
    await dispatcher.dispatch(user_id="unknown", text="poné una alarma a las 7 de la tarde")
    await dispatcher.dispatch(user_id="unknown", text="poné un despertador para el jueves a las 7 de la mañana")
    await dispatcher.dispatch(user_id="unknown", text="recordame sacar la basura a las 8 de la noche")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="ana", text="qué alarmas hay a las 7")
    assert result.response == "Alarmas guardadas: alarma, hoy a las 19:00. despertador, el jueves a las 07:00."
    assert result.outcome.status.value == "completed" and leaves.effects(before) == ZERO
    assert "privada" not in result.response


@pytest.mark.parametrize(("text", "row"), [
    ("poné una alarma a las 19 horas", ("alarma", (2026, 11, 16, 19, 0))),
    ("poné una alarma a las 19 hs", ("alarma", (2026, 11, 16, 19, 0))),
    ("poné un despertador para las 21 horas", ("despertador", (2026, 11, 16, 21, 0))),
    ("poné una alarma a las 7 horas de la tarde", ("alarma", (2026, 11, 16, 19, 0))),
])
async def test_an_alarm_at_n_horas_is_a_reminder_at_that_hour_not_a_timer(leaves, text, row):
    """Review T031-U2-RV2 (user answer Q4): 'a las N horas' is the hour of the day; no timer starts."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.success) == (PathType.FAST_REMINDER, True)
    assert leaves.effects(before) == effects(reminder_rows_delta=1)
    assert leaves.timers.get_all_timers() == []
    expected = datetime(*row[1], tzinfo=ZoneInfo("America/Montevideo")).timestamp()
    assert _rows(leaves.reminders_db) == [("household", "shared", row[0], expected, None, None, "active")]


async def test_an_alarm_of_n_horas_without_the_article_stays_a_timer(leaves):
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="poné una alarma de 19 horas")
    assert result.path is PathType.FAST_TIMER
    assert leaves.effects(before) == effects(timer_delta=1)
    assert [t.duration_seconds for t in leaves.timers.get_all_timers()] == [19 * 3600]
    assert _rows(leaves.reminders_db) == []


ALARM_NOTE_CANCEL = (" Hay 1 alarma: mañana a las 07:00; para cancelarla decí: cancelá la alarma de las 7.")


@pytest.mark.parametrize(("text", "response"), [
    ("qué alarmas hay a las 7", "Alarmas guardadas: alarma, mañana a las 07:00."),
    ("cancelá la alarma de las 7", "Hay uno. 1: alarma a las 07:00; ¿lo cancelo? Decime: el primero."),
    ("cancelá la alarma", "No hay temporizadores activos." + ALARM_NOTE_CANCEL),
])
async def test_only_a_reminder_whose_text_starts_with_the_alarm_noun_is_an_alarm(leaves, text, response):
    """Review T031-U2-RV6: a reminder that only mentions an alarm is never listed or cancelled as one."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    due = datetime(2026, 11, 17, 7, 0, tzinfo=ZoneInfo("America/Montevideo")).timestamp()
    for saved in ("desactivar la alarma", "revisar el despertador", "alarma"):
        await leaves.reminder_manager.create("household", saved, due, scope=ReminderScope.SHARED)
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert result.response == response
    assert leaves.effects(before) == ZERO


@pytest.mark.parametrize(("text", "response"), [
    ("cancelá la alarma", "No hay temporizadores activos." + ALARM_NOTE_CANCEL),
    ("cancelá el despertador", "No hay temporizadores activos." + ALARM_NOTE_CANCEL),
    ("apagá la alarma", "No hay temporizadores activos." + ALARM_NOTE_CANCEL),
    ("cancelá la alarma de la pasta", "No hay temporizadores activos." + ALARM_NOTE_CANCEL),
    ("¿qué alarmas hay?", "No hay temporizadores activos. Hay 1 alarma: mañana a las 07:00."),
    ("¿hay alguna alarma?", "No hay temporizadores activos. Hay 1 alarma: mañana a las 07:00."),
    ("¿a qué hora suena la alarma?", "No hay temporizadores activos. Hay 1 alarma: mañana a las 07:00."),
])
async def test_a_timer_reply_on_an_alarm_tells_the_alarm_reminders_and_cancels_nothing(leaves, text, response):
    """Review T031-U2-RV3: after 'poné una alarma mañana a las 7', 'no timers' alone would hide the alarm."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    assert (await dispatcher.dispatch(user_id="unknown", text="poné una alarma mañana a las 7 de la mañana")).success
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.response) == (PathType.FAST_TIMER, response)
    assert leaves.effects(before) == ZERO
    assert [r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == ["alarma"]


async def test_an_alarm_query_with_timers_running_also_tells_the_alarm_reminders(leaves):
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await dispatcher.dispatch(user_id="unknown", text="poné una alarma mañana a las 7 de la mañana")
    await dispatcher.dispatch(user_id="unknown", text="poné un despertador para el jueves a las 6 y media de la mañana")
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    result = await dispatcher.dispatch(user_id="unknown", text="qué alarmas hay")
    assert result.path is PathType.FAST_TIMER
    assert result.response.endswith(" Hay 2 alarmas: mañana a las 07:00 y el jueves a las 06:30.")
    assert result.response.startswith("Al temporizador de ")


async def test_a_timer_reply_on_an_alarm_with_two_alarms_gives_one_example(leaves):
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await dispatcher.dispatch(user_id="unknown", text="poné una alarma a las 13 y 30")
    await dispatcher.dispatch(user_id="unknown", text="poné un despertador mañana a la 1 de la tarde")
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá la alarma")
    assert result.response == ("No hay temporizadores activos. Hay 2 alarmas: hoy a las 13:30 y mañana a las 13:00; "
                               "para cancelar una decí, por ejemplo: cancelá la alarma de las 13 y 30.")
    assert result.outcome.status.value == "needs_input"


async def test_a_timer_reply_on_a_timer_or_a_found_alarm_timer_adds_nothing(leaves):
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await dispatcher.dispatch(user_id="unknown", text="poné una alarma mañana a las 7 de la mañana")
    assert (await dispatcher.dispatch(user_id="unknown", text="cancelá el timer")).response == (
        "No hay temporizadores activos.")
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    cancelled = await dispatcher.dispatch(user_id="unknown", text="cancelá la alarma")
    assert "alarma:" not in cancelled.response and cancelled.response.startswith("Cancelé el temporizador")


async def test_unreadable_alarm_reminders_leave_the_timer_reply_as_is(leaves, monkeypatch):
    from unittest.mock import AsyncMock
    dispatcher = leaves.dispatcher()
    monkeypatch.setattr(leaves.reminder_manager, "get_active", AsyncMock(side_effect=sqlite3.OperationalError("x")))
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá la alarma")
    assert (result.path, result.response) == (PathType.FAST_TIMER, "No hay temporizadores activos.")


@pytest.mark.parametrize("text", ["cancelá la alarma de mañana", "borrá el despertador de mañana"])
async def test_a_day_only_alarm_cancel_lists_the_alarm_reminders_by_number(leaves, text):
    """Review T031-U2-RV3: 'la alarma de mañana' names an alarm set for a clock time: numbered, never cancelled."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await dispatcher.dispatch(user_id="unknown", text="poné una alarma mañana a las 7 de la mañana")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert result.path is PathType.FAST_REMINDER
    assert result.response == "Hay uno. 1: alarma a las 07:00; ¿lo cancelo? Decime: el primero."
    assert result.outcome.status.value == "needs_input" and leaves.effects(before) == ZERO


async def test_an_alarm_without_a_clock_time_still_reaches_the_timers(leaves):
    """T030-U1-Q8 (A): without a clock time the alarm stays a timer, even with an alarm reminder active."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:alarma;timer_running_unnamed")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá la alarma")
    assert result.path is PathType.FAST_TIMER
    assert leaves.effects(before) == effects(timer_transitions=1)


# ---------------------------------------------------------------------------
# User answer T031-U2-Q1 (batch_g_questions_20261118): "apagá/pará la alarma de
# las 7" is the cancel of the alarm set for that clock time, through the same
# numbered list as "cancelá la alarma de las 7" (T032-U1 rules). User answer
# BH-R1-Q1 (r1e_questions_20261118): "desactivá la alarma <hora>" is a security
# alarm verb and stays unclaimed (design R3a); "desactivá el despertador <hora>"
# stays the cancel.
# ---------------------------------------------------------------------------

CLOCK_ALARM_OFF = ["apagá la alarma de las 7", "pará la alarma de las 7", "desactivá el despertador de las 7",
                   "Nexa, apagame el despertador de las 7", "detené la alarma de las 7", "para la alarma de las 7",
                   "silenciá la alarma de las 7"]


@pytest.mark.parametrize("text", CLOCK_ALARM_OFF)
async def test_turning_off_a_clock_alarm_lists_the_alarms_and_the_answer_cancels_only_that_one(leaves, text):
    """Two alarms: numbered, nothing cancelled, the running timer untouched; the ordinal cancels exactly one."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    await leaves.apply_state(dispatcher, "timer_running_unnamed;queued_request")
    for create in ("poné una alarma a las 7 de la tarde", "poné un despertador mañana a las 7 de la mañana",
                   "recordame sacar la basura a las 7 de la tarde"):
        assert (await dispatcher.dispatch(user_id="unknown", text=create)).success
    before = leaves.snapshot()
    asked = await dispatcher.dispatch(user_id="unknown", text=text, service_filter="turn_off")
    assert (asked.path, asked.success) == (PathType.FAST_REMINDER, False)
    assert asked.response == ("Hay 2. 1: alarma a las 19:00; 2: despertador a las 07:00; ¿cuál? Decime el primero "
                              "o el segundo.")
    assert (asked.outcome.status.value, asked.outcome.persistence.value) == ("needs_input", "not_written")
    assert leaves.effects(before) == ZERO
    answered = await dispatcher.dispatch(user_id="unknown", text="el segundo")
    assert answered.response == "Cancelé el recordatorio despertador a las 07:00."
    assert leaves.effects(before) == effects(reminder_transitions=1)
    remaining = sorted(r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True))
    assert remaining == ["alarma", "sacar la basura"]
    assert [t.state.value for t in leaves.timers.get_all_timers()] == ["running"]
    assert len(leaves.queue) == 1


@pytest.mark.parametrize("text", ["apagá la alarma de las 7", "desactivá el despertador de las 7"])
async def test_turning_off_the_only_clock_alarm_still_asks_before_cancelling(leaves, text):
    """A spoken time never picks a reminder (t032_design_20261118): even one alarm is listed and confirmed."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    assert (await dispatcher.dispatch(user_id="unknown", text="poné una alarma mañana a las 7 de la mañana")).success
    before = leaves.snapshot()
    asked = await dispatcher.dispatch(user_id="unknown", text=text)
    assert asked.response == "Hay uno. 1: alarma a las 07:00; ¿lo cancelo? Decime: el primero."
    assert leaves.effects(before) == ZERO
    done = await dispatcher.dispatch(user_id="unknown", text="no, el primero, dale")
    assert done.response == "Cancelé el recordatorio alarma a las 07:00."
    assert leaves.effects(before) == effects(reminder_transitions=1)


async def test_turning_off_a_clock_alarm_is_never_narrowed_by_the_spoken_time(leaves):
    """Alarms at 07:00, 09:00 and 14:00: all listed for 'de las 7'; a bare 'dos' that is also the 14:00 hour asks."""
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    for noun, hour in (("alarma", 7), ("despertador", 9), ("alarma", 14)):
        await manager.create("household", noun, _tomorrow_at(leaves, hour), scope=ReminderScope.SHARED)
    before = leaves.snapshot()
    asked = await dispatcher.dispatch(user_id="unknown", text="apagá la alarma de las 7")
    assert asked.response == ("Hay 3. 1: alarma a las 07:00; 2: despertador a las 09:00; 3: alarma a las 14:00; "
                              "¿cuál? Decime el primero, el segundo o el tercero.")
    clash = await dispatcher.dispatch(user_id="unknown", text="dos")
    assert clash.response == "¿El segundo o el de las 2?"
    assert leaves.effects(before) == ZERO
    done = await dispatcher.dispatch(user_id="unknown", text="el de las 2")
    assert done.response == "Cancelé el recordatorio alarma a las 14:00."
    assert leaves.effects(before) == effects(reminder_transitions=1)
    assert sorted((r.text, manager.voice_label(r)) for r in await manager.get_active(None, shared_only=True)) == [
        ("alarma", "alarma a las 07:00"), ("despertador", "despertador a las 09:00")]


@pytest.mark.parametrize("text", ["apagá la alarma de las 7", "pará la alarma de las 7",
                                  "desactivá el despertador de las 7"])
async def test_turning_off_a_clock_alarm_without_one_never_stops_the_timer(leaves, text):
    """Before T031-U2-Q1 'apagá/pará la alarma de las 7' stopped the running timer instead."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "timer_running_unnamed")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.response) == (PathType.FAST_REMINDER, "No hay alarmas con hora guardadas. "
                                                                      "No cancelé nada.")
    assert result.outcome.status.value == "needs_input" and leaves.effects(before) == ZERO
    assert [t.state.value for t in leaves.timers.get_all_timers()] == ["running"]


@pytest.mark.parametrize(("text", "transitions"), [
    ("apagá la alarma", 1), ("pará la alarma", 1), ("desactivá el despertador", 1),
    ("apagá la alarma de 5 minutos", 1), ("pará el despertador de 5 minutos", 1),
])
async def test_an_alarm_without_a_clock_time_is_still_stopped_on_the_timers(leaves, text, transitions):
    """T030-U1: no clock time (or a duration) keeps the timer stop, even with an alarm reminder active."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:alarma;timer_running_unnamed")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.success) == (PathType.FAST_TIMER, True)
    assert leaves.effects(before) == effects(timer_transitions=transitions)
    assert [r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == ["alarma"]


@pytest.mark.parametrize("text", ["desactivá la alarma", "desactivá la alarma de las 7",
                                  "desactivar la alarma de las 7 de la mañana", "desactivá la alarma para mañana",
                                  "Nexa, desactivá la alarma de la casa a las 23"])
async def test_the_security_alarm_stays_unclaimed_with_or_without_a_clock_time(leaves, text):
    """'desactivá la alarma' never reaches the timers or the reminders (design R3a, user answer BH-R1-Q1)."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:alarma;timer_running_unnamed")
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert result.path not in (PathType.FAST_TIMER, PathType.FAST_REMINDER)
    assert not result.response.startswith(("Hay ", "No hay alarmas"))
    counted = leaves.effects(before)
    assert (counted["timer_transitions"], counted["reminder_transitions"]) == (0, 0)
    assert [r.text for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == ["alarma"]
    assert [t.state.value for t in leaves.timers.get_all_timers()] == ["running"]


# ---------------------------------------------------------------------------
# Unit BFR1-Q1 (user answer, assistant_defaults_recommended): voice reminder
# creation reads the request after the classifier's fillers (strip_fillers) and
# never saves a paused trailing "dale".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "row", "response"), [
    ("recordame regar las plantas a las 9 de la noche, dale", ("regar las plantas", (2026, 11, 16, 21, 0)),
     "Anotado: regar las plantas, hoy a las 21:00."),
    ("bueno, recordame regar las plantas a las 9 de la noche", ("regar las plantas", (2026, 11, 16, 21, 0)),
     "Anotado: regar las plantas, hoy a las 21:00."),
    ("no, recordame regar las plantas a las 9 de la noche", ("regar las plantas", (2026, 11, 16, 21, 0)),
     "Anotado: regar las plantas, hoy a las 21:00."),
    ("Nexa, bue, recordame sacar la basura a las 9 de la noche, dale, porfa", ("sacar la basura",
                                                                            (2026, 11, 16, 21, 0)),
     "Anotado: sacar la basura, hoy a las 21:00."),
    ("dale, recordame tomar agua a las 5 de la tarde. Dale.", ("tomar agua", (2026, 11, 16, 17, 0)),
     "Anotado: tomar agua, hoy a las 17:00."),
    ("hola, por favor, recordame tomar agua a las 5 de la tarde, porfa, dale", ("tomar agua", (2026, 11, 16, 17, 0)),
     "Anotado: tomar agua, hoy a las 17:00."),
    ("poné una alarma a las 7 de la tarde, dale", ("alarma", (2026, 11, 16, 19, 0)), "Anotado: alarma, hoy a las 19:00."),
    ("bueno, poné un despertador mañana a las 7 de la mañana", ("despertador", (2026, 11, 17, 7, 0)),
     "Anotado: despertador, mañana a las 07:00."),
    ("no, poné una alarma a las 7 de la tarde para ir al médico, dale",
     ("alarma para ir al médico", (2026, 11, 16, 19, 0)), "Anotado: alarma para ir al médico, hoy a las 19:00."),
    # Without a pause a trailing 'dale' may end the text (review BFR1-R1): it stays.
    ("recordame a las 9 de la noche decirle a Juan que dale", ("decirle a Juan que dale", (2026, 11, 16, 21, 0)),
     "Anotado: decirle a Juan que dale, hoy a las 21:00."),
    ("recordame a las 9 de la noche decirle a Juan que dale, dale", ("decirle a Juan que dale",
                                                                    (2026, 11, 16, 21, 0)),
     "Anotado: decirle a Juan que dale, hoy a las 21:00."),
    ("recordame regar las plantas a las 9 de la noche dale", ("regar las plantas dale", (2026, 11, 16, 21, 0)),
     "Anotado: regar las plantas dale, hoy a las 21:00."),
    # A colon usually opens quoted speech: the 'dale' after it is kept (review BH-R2).
    ("recordame a las 5 de la tarde decirle a Juan: dale", ("decirle a Juan: dale", (2026, 11, 16, 17, 0)),
     "Anotado: decirle a Juan: dale, hoy a las 17:00."),
])
async def test_voice_creation_ignores_the_shared_fillers(leaves, text, row, response):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.success, result.response) == (PathType.FAST_REMINDER, True, response)
    assert result.outcome.persistence.value == "committed"
    assert leaves.effects(before) == effects(reminder_rows_delta=1)
    expected = datetime(*row[1], tzinfo=ZoneInfo("America/Montevideo")).timestamp()
    assert _rows(leaves.reminders_db) == [("household", "shared", row[0], expected, None, None, "active")]


@pytest.mark.parametrize("text", ["no recordame regar las plantas a las 9 de la noche",
                                  "no me recuerdes regar las plantas a las 9 de la noche"])
async def test_a_leading_no_without_a_pause_is_a_negation_and_writes_nothing(leaves, text):
    """T032-U1-Q4: only 'no' followed by a pause is a filler; 'no recordame ...' never creates a reminder."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    assert dispatcher.household_intents.classify(text) is None
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert not result.response.startswith("Anotado")
    assert leaves.effects(before)["reminder_rows_delta"] == 0
    assert _rows(leaves.reminders_db) == []


async def test_a_filled_creation_question_asks_without_the_fillers(leaves):
    """A request that still needs input is asked as before: the fillers never become the reminder text."""
    dispatcher = leaves.dispatcher()
    _frozen_leaves(leaves, dispatcher, _monday_10())
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text="bueno, recordame a las 9 de la noche, dale")
    assert result.response == f"¿Qué querés que te recuerde? Decímelo completo, por ejemplo: {EXAMPLE}."
    assert (result.outcome.status.value, result.outcome.persistence.value) == ("needs_input", "not_written")
    assert leaves.effects(before) == ZERO


async def test_a_configured_wake_word_before_the_filled_request_is_ignored(leaves):
    """The creation strips the configured wake words with the other fillers, as the classifier that claimed it."""
    classifier = HouseholdIntentClassifier(HouseholdIntentSettings.from_config({}, wake_words=["Jarvis"]))
    dispatcher = leaves.dispatcher(household_intents=classifier)
    _frozen_leaves(leaves, dispatcher, _monday_10())
    result = await dispatcher.dispatch(user_id="unknown", text="Jarvis, bueno, recordame tomar agua a las 5 de la tarde")
    assert (result.success, result.response) == (True, "Anotado: tomar agua, hoy a las 17:00.")
    assert [row[2] for row in _rows(leaves.reminders_db)] == ["tomar agua"]
