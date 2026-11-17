"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

import asyncio
import sqlite3
import time

import pytest

from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.orchestrator.dispatcher import MultiUserOrchestrator, PathType
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
    conditions = {k: v for k, v in (("service_filter", sf), ("state", state)) if v}
    return _d(id_, text, family, effect_class, status, path, status, "not_written", spoken,
              conditions=conditions, **kw)


def _timer(id_, text, outcome, persistence, spoken, *, counts=ZERO, state=None, sf=None,
           effect_class="unauthorized_effect", seed=None, note=""):
    """A T030-U1 timer entry on the dispatcher layer (a real, started NamedTimerManager)."""
    conditions = {k: v for k, v in (("service_filter", sf), ("state", state)) if v}
    return _d(id_, text, "timer", effect_class, outcome, "fast_timer", outcome, persistence, spoken, counts=counts,
              conditions=conditions, unit="T030-U1", seed=seed, note=note)


CREATED = effects(timer_delta=1)
CHANGED = effects(timer_transitions=1)

INTERIM_CREATE = ("interim: reminder creation by voice lands with T031-U2; until then the existing honest "
                  "'Todavía no puedo crear ese recordatorio por voz' (UNSUPPORTED) instead of the seed's "
                  "needs_input clarification; every counter as the seed")

# 'recordame/recuérdame' x {apagar la luz, poner música, agregar X a la lista, abrir la persiana}
# x service_filter {turn_on, turn_off}: zero HA sends, music calls or list mutations.
_REMINDER_GRID = [
    _refused(f"C02-REM-{101 + i:04d}", f"{verb} {complement}", "reminder", sf=sf, state="shared_list_compras",
             finding="CORE-02", note=INTERIM_CREATE)
    for i, (verb, complement, sf) in enumerate(
        (verb, complement, sf)
        for verb in ("recordame", "recuérdame")
        for complement in ("apagar la luz", "poner música", "agregar leche a la lista de compras",
                           "abrir la persiana")
        for sf in ("turn_on", "turn_off"))
]

C02_DISPATCHER = tuple(_REMINDER_GRID) + (
    _refused("C02-REM-0002", "recordame poner música a las ocho", "reminder", seed="C02-REM-0002", note=INTERIM_CREATE),
    _refused("C02-REM-0003", "recordame agregar leche a la lista de compras", "reminder",
             state="shared_list_compras", seed="C02-REM-0002", note=INTERIM_CREATE),
    _refused("C02-REM-0006", "recordame actualizar la lista", "reminder", effect_class="false_confirmation",
             state="shared_list_compras", finding="F10", note=INTERIM_CREATE),
    _refused("C02-REM-0007", "recordame registrar el gasto del super", "reminder",
             effect_class="false_confirmation", finding="F10", note=INTERIM_CREATE),
    _refused("C02-REM-0117", "haceme acordar de apagar la luz", "reminder", sf="turn_off", note=INTERIM_CREATE),
    _refused("C02-REM-0118", "acordame de cerrar la persiana", "reminder", sf="turn_off", note=INTERIM_CREATE),
    _refused("C02-REM-0119", "avisame cuando termine el lavarropas", "reminder", note=INTERIM_CREATE),
    _refused("C02-REM-0120", "no me dejes olvidar sacar la basura", "reminder", note=INTERIM_CREATE),
    _refused("C02-REM-0121", "nexa recordame prender la luz del living", "reminder", sf="turn_on",
             note=INTERIM_CREATE),
    _refused("C02-REM-0122", "recordame cancelar el turno del médico", "reminder", state="queued_request",
             note="the cancel verb is reminder content: the queued request is not cancelled"),
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
    _refused("C02-ALM-0102", "poné una alarma a las 7", "reminder", seed="C02-ALM-0002", note=INTERIM_CREATE),
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
    _refused("C02-CMP-0101", "apagá la luz y pausá la música", "compound", "fast_household", sf="turn_off",
             state="light_match", status="needs_input", spoken="clarification",
             note="interim gate household_intents.compound_execute: false until T037-U6"),
    _refused("C02-CMP-0102", "cancelá el timer y el recordatorio", "compound", "fast_household",
             state="queued_request", status="needs_input", spoken="clarification"),
    _refused("C02-CMP-0103", "apagá la luz del escritorio y de la cocina", "compound", "fast_household",
             sf="turn_off", state="light_match", status="needs_input", spoken="clarification"),
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
       unit="T032-U1", finding="CORE-03", seed="C02-CAN-0003", note="no object named: asks which, lists both"),
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
       unit="T032-U1", finding="CORE-03", note="'de hoy' is the household date (T031 day grammar)"),
    _d("C02-CAN-0005", "cancelá el recordatorio de la clave", "reminder", "privacy_disclosure", "needs_input",
       "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_private_active:la clave", "speaker_label": "ana"}, actor="acoustic_label",
       unit="T032-U1", seed="C02-CAN-0005", note="private text never read nor spoken; the label adds no scope"),
) + tuple(
    # User answer T032-U1-Q1 = A: another active reminder holds every spoken word, so ask and cancel nothing.
    _d(f"C02-CAN-{261 + i:04d}", f"{verb} recordatorio de tomar agua", "reminder", "unauthorized_effect",
       "needs_input", "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_shared_active:tomar agua;reminder_shared_active:tomar agua con gas;"
                            "queued_request"},
       unit="T032-U1", finding="CORE-03", note="superset: 'tomar agua con gas' holds every spoken word")
    for i, verb in enumerate(CANCEL_VERBS)
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
    """The classified operation, not keywords inside the content, picks the reminder branch."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    before = leaves.snapshot()
    for text in ("recordame cancelar el recordatorio de tomar agua", "recordame ver qué recordatorios hay"):
        result = await dispatcher.dispatch(user_id="unknown", text=text)
        assert result.path is PathType.FAST_REMINDER and result.outcome.status.value == "unsupported"
    assert leaves.effects(before) == ZERO


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

async def _cancel(leaves, text, state=None, *, user_id="unknown", dispatcher=None):
    dispatcher = dispatcher or leaves.dispatcher()
    await leaves.apply_state(dispatcher, state)
    before = leaves.snapshot()
    leaves.cancels.clear()
    result = await dispatcher.dispatch(user_id=user_id, text=text)
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


async def test_no_object_asks_which_and_lists_the_shared_candidates(leaves):
    result, counted = await _cancel(leaves, "cancelá el recordatorio",
                                    "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura;"
                                    "reminder_private_active:la clave")
    assert counted == ZERO
    assert result.outcome.status.value == "needs_input"
    assert result.response.startswith("Hay 2: tomar agua a las ")
    assert " y sacar la basura a las " in result.response and "clave" not in result.response
    assert ". Para cancelar uno, decímelo entero, por ejemplo: cancelá el recordatorio de tomar agua de las " \
        in result.response


async def test_plural_without_object_never_cancels_them_all(leaves):
    result, counted = await _cancel(leaves, "eliminá todos los recordatorios", "reminder_shared_active:tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Por voz cancelo de a un recordatorio. Hay uno: tomar agua a las ")
    assert result.response.endswith(". Para cancelarlo, decime: cancelá el recordatorio de tomar agua.")


async def test_more_than_three_candidates_speak_three(leaves):
    state = ";".join(f"reminder_shared_active:regar {name}" for name in ("helecho", "cactus", "potus", "ficus"))
    result, counted = await _cancel(leaves, "cancelá el recordatorio de regar", state)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Encontré 4, por ejemplo: regar helecho a las ")
    assert ". Repetímelo con la hora, por ejemplo: cancelá el recordatorio de regar helecho de las " \
        in result.response
    assert result.response.count(" a las ") == 3


async def test_two_matches_ask_with_the_time_and_the_time_resolves_them(leaves):
    dispatcher = leaves.dispatcher()
    early = await leaves.reminder_manager.create("household", "tomar agua", time.time() + 3600,
                                                 scope=ReminderScope.SHARED)
    late = await leaves.reminder_manager.create("household", "tomar agua", time.time() + 4 * 3600,
                                                scope=ReminderScope.SHARED)
    clock = leaves.reminder_manager._household_clock
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = f"cancelá el recordatorio de tomar agua de {leaves.reminder_manager.spoken_time_reference(early)}"
    assert asked.response == (f"Encontré 2: tomar agua a las {clock.format_time(early.trigger_at)} y tomar agua "
                              f"a las {clock.format_time(late.trigger_at)}. Repetímelo con la hora, por ejemplo: "
                              f"{example}.")
    # Review T032-U1-R7: the proposed command, said as is, cancels exactly that reminder.
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == [late.id]
    again, counted = await _cancel(leaves, f"cancelá el recordatorio de tomar agua de las "
                                           f"{clock.format_time(late.trigger_at)}", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and again.outcome.task_completed


async def test_identical_duplicates_point_to_the_web(leaves):
    trigger = time.time() + 3600
    for _ in range(2):
        await leaves.reminder_manager.create("household", "tomar agua", trigger, scope=ReminderScope.SHARED)
    result, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Hay 2 recordatorios iguales: tomar agua a las ")
    assert result.response.endswith("Podés borrar el que sobra desde la web.")


# User answer T032-U1-Q1 = A: when another active reminder contains every spoken
# word, ask which; never cancel the wrong one.

def _minute_start(hours_ahead: int) -> float:
    return (int(time.time()) // 60 + 1) * 60 + hours_ahead * 3600


async def _two_shared(leaves, first: str, second: str, first_at: float, second_at: float):
    manager = leaves.reminder_manager
    one = await manager.create("household", first, first_at, scope=ReminderScope.SHARED)
    two = await manager.create("household", second, second_at, scope=ReminderScope.SHARED)
    return one, two


async def test_a_superset_at_the_same_time_asks_and_proposes_the_nameable_one(leaves):
    dispatcher = leaves.dispatcher()
    manager, base = leaves.reminder_manager, _minute_start(1)
    plain, lemon = await _two_shared(leaves, "tomar agua", "tomar agua con limón", base, base + 30)
    hm = manager._household_clock.format_time(base)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", "queued_request",
                                   dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input" and len(leaves.queue) == 1
    example = f"cancelá el recordatorio de tomar agua con limón de {manager.spoken_time_reference(lemon)}"
    assert asked.response == (f"Encontré 2: tomar agua a las {hm} y tomar agua con limón a las {hm}. "
                              f"Repetímelo con la hora, por ejemplo: {example}. Si es otro, borralo desde la web.")
    again, counted = await _cancel(leaves, f"cancelá el recordatorio de tomar agua de "
                                           f"{manager.spoken_time_reference(plain)}", dispatcher=dispatcher)
    assert counted == ZERO and again.outcome.status.value == "needs_input"
    assert await _states(leaves) == {"tomar agua": "active", "tomar agua con limón": "active"}
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [plain.id]
    last, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and last.outcome.task_completed
    assert await manager.get_active(None, shared_only=True) == []


async def test_a_superset_at_another_time_asks_and_the_time_names_the_exact_one(leaves):
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    plain, lemon = await _two_shared(leaves, "tomar agua", "tomar agua con limón", _minute_start(1),
                                     _minute_start(4))
    clock = manager._household_clock
    asked, counted = await _cancel(leaves, "borrá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = f"cancelá el recordatorio de tomar agua de {manager.spoken_time_reference(plain)}"
    assert asked.response == (f"Encontré 2: tomar agua a las {clock.format_time(plain.trigger_at)} y tomar agua "
                              f"con limón a las {clock.format_time(lemon.trigger_at)}. Repetímelo con la hora, "
                              f"por ejemplo: {example}.")
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [lemon.id]


async def test_no_object_with_a_superset_proposes_the_nameable_one(leaves):
    manager, base = leaves.reminder_manager, _minute_start(1)
    _, lemon = await _two_shared(leaves, "tomar agua", "tomar agua con limón", base, base + 30)
    hm = manager._household_clock.format_time(base)
    result, counted = await _cancel(leaves, "cancelá el recordatorio")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response == (f"Hay 2: tomar agua a las {hm} y tomar agua con limón a las {hm}. Para cancelar "
                               f"uno, decímelo entero, por ejemplo: cancelá el recordatorio de tomar agua con limón "
                               f"de {manager.spoken_time_reference(lemon)}. Si es otro, borralo desde la web.")


async def test_reminders_voice_cannot_tell_apart_point_to_the_web(leaves):
    dispatcher = leaves.dispatcher()
    manager, base = leaves.reminder_manager, _minute_start(1)
    await _two_shared(leaves, "tomar agua", "tomar el agua", base, base + 30)
    hm = manager._household_clock.format_time(base)
    choices = f"tomar agua a las {hm} y tomar el agua a las {hm}"
    web = "Por voz no los puedo distinguir; podés borrar el que querés desde la web."
    named, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and named.outcome.status.value == "needs_input"
    assert named.response == f"Encontré 2: {choices}. {web}"
    bare, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and bare.outcome.status.value == "needs_input"
    assert bare.response == f"Hay 2: {choices}. {web}"
    assert await _states(leaves) == {"tomar agua": "active", "tomar el agua": "active"}


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
    ("quiero un recordatorio para cancelar la suscripción", "unsupported"),
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


async def test_a_number_in_the_reminder_text_never_cancels_another_reminder(leaves):
    """Review T032-U1-R2: 'las 2 pastillas' is content, not 14:00."""
    from datetime import datetime, timedelta
    dispatcher = leaves.dispatcher()
    clock = leaves.reminder_manager._household_clock
    tomorrow = clock.now_local().date() + timedelta(days=1)

    def at(hour: int) -> float:
        return clock.to_epoch(datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour))
    literal = await leaves.reminder_manager.create("household", "tomar las 2 pastillas", at(9),
                                                   scope=ReminderScope.SHARED)
    other = await leaves.reminder_manager.create("household", "tomar las pastillas", at(14),
                                                 scope=ReminderScope.SHARED)
    result, counted = await _cancel(leaves, "cancelá el recordatorio de tomar las 2 pastillas", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1)
    assert result.response.startswith("Cancelé el recordatorio tomar las 2 pastillas a las 09:00")
    assert [r.id for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == [other.id]
    assert (await leaves.reminder_store.get_by_id(literal.id)).state == "cancelled"


async def test_a_clock_word_of_one_text_and_another_due_time_asks_which(leaves):
    """User decision T032-U1-RR-P1 (2026-11-17): before it, 'de las 2' cancelled the 14:00 reminder.

    "2" is also a word of "tomar las 2 pastillas", so KZA asks which,
    cancels nothing, and the proposed command cancels exactly one.
    """
    from datetime import datetime, timedelta
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    clock = manager._household_clock
    tomorrow = clock.now_local().date() + timedelta(days=1)

    def at(hour: int) -> float:
        return clock.to_epoch(datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour))
    literal = await manager.create("household", "tomar las 2 pastillas", at(9), scope=ReminderScope.SHARED)
    other = await manager.create("household", "tomar las pastillas", at(14), scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar las pastillas de las 2",
                                   dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = "cancelá el recordatorio de tomar las 2 pastillas de las 9 de la mañana"
    assert asked.response == ("Encontré 2: tomar las 2 pastillas a las 09:00 y tomar las pastillas a las 14:00. "
                              f"Repetímelo con la hora, por ejemplo: {example}.")
    assert {r.id for r in await manager.get_active(None, shared_only=True)} == {literal.id, other.id}
    done, counted = await _cancel(leaves, "cancelá el recordatorio de tomar las pastillas de las 2 de la tarde",
                                  dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [literal.id]
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert await manager.get_active(None, shared_only=True) == []


@pytest.mark.parametrize(("literal", "literal_due", "other", "other_due", "response"), [
    ("llamar al médico a la una", (12, 45), "llamar al médico", (13, 0),
     "Encontré 2: llamar al médico a la una a las 12:45 y llamar al médico a las 13:00. Repetímelo con la hora, "
     "por ejemplo: cancelá el recordatorio de llamar al médico de las 12:45 del mediodía."),
    ("tomar 2 pastillas a las 2", (9, 0), "tomar 2 pastillas", (14, 0),
     "Encontré 2: tomar 2 pastillas a las 2 a las 09:00 y tomar 2 pastillas a las 14:00. Repetímelo con la hora, "
     "por ejemplo: cancelá el recordatorio de tomar 2 pastillas de las 9 de la mañana."),
])
async def test_the_whole_text_with_its_clock_phrase_never_cancels_another_reminder(
        leaves, literal, literal_due, other, other_due, response):
    """Review T032-U1-FV-1: before it, these cancelled the reminder due at the text's clock word.

    "una" is a stopword and "2" is already a content word, so the literal
    reading was skipped; now KZA asks, cancels nothing, and the proposed
    command cancels exactly the named one.
    """
    from datetime import datetime, timedelta
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    clock = manager._household_clock
    tomorrow = clock.now_local().date() + timedelta(days=1)

    def at(hour: int, minute: int) -> float:
        return clock.to_epoch(datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute))
    named = await manager.create("household", literal, at(*literal_due), scope=ReminderScope.SHARED)
    timed = await manager.create("household", other, at(*other_due), scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, f"cancelá el recordatorio de {literal}", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response == response
    assert {r.id for r in await manager.get_active(None, shared_only=True)} == {named.id, timed.id}
    example = response.split("por ejemplo: ", 1)[1].removesuffix(".")
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
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


# Review T032-U1-RR-F1: a clock or day word inside the only reminder's text
# never makes KZA say it cannot tell reminders apart.

def _hour_word_not(hour: int) -> str:
    return next(str(h) for h in (3, 5, 7) if h not in (hour % 12, hour))


async def test_the_only_reminder_with_a_clock_in_its_text_is_named_and_cancelled(leaves):
    dispatcher = leaves.dispatcher()
    manager, at = leaves.reminder_manager, _minute_start(1)
    text = f"llamar a mamá a las {_hour_word_not(manager._household_clock.to_local(at).hour)}"
    await manager.create("household", text, at, scope=ReminderScope.SHARED)
    hm = manager._household_clock.format_time(at)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response == f"Hay uno: {text} a las {hm}. Para cancelarlo, decime: cancelá el recordatorio de {text}."
    done, counted = await _cancel(leaves, f"cancelá el recordatorio de {text}", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert await manager.get_active(None, shared_only=True) == []


@pytest.mark.parametrize("said", ["llevar el auto al taller mañana", "llevar el auto al taller"])
async def test_the_only_reminder_with_a_day_in_its_text_is_named_and_cancelled(leaves, said):
    dispatcher = leaves.dispatcher()
    manager, at = leaves.reminder_manager, _minute_start(1)
    await manager.create("household", "llevar el auto al taller mañana", at, scope=ReminderScope.SHARED)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    assert asked.response.endswith(". Para cancelarlo, decime: cancelá el recordatorio de llevar el auto al taller "
                                   "mañana.")
    done, counted = await _cancel(leaves, f"cancelá el recordatorio de {said}", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed


async def test_a_clock_in_the_text_is_left_out_of_a_proposed_command_with_the_time(leaves):
    """With several reminders the command carries the time, so the text's own clock is left out."""
    dispatcher = leaves.dispatcher()
    manager, at = leaves.reminder_manager, _minute_start(1)
    text = f"llamar a mamá a las {_hour_word_not(manager._household_clock.to_local(at).hour)}"
    call, water = await _two_shared(leaves, text, "tomar agua", at, at + 3 * 3600)
    clock = manager._household_clock
    asked, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = f"cancelá el recordatorio de llamar a mamá de {manager.spoken_time_reference(call)}"
    assert asked.response == (f"Hay 2: {text} a las {clock.format_time(at)} y tomar agua a las "
                              f"{clock.format_time(water.trigger_at)}. Para cancelar uno, decímelo entero, por "
                              f"ejemplo: {example}.")
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [water.id]


async def test_the_only_reminder_is_never_called_indistinct(leaves, monkeypatch):
    """Last resort when no command can be proposed: singular wording, not 'no los puedo distinguir'."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua")
    calls = []

    async def none_found(user_id, candidates, *, with_time, **kwargs):
        calls.append(with_time)
        return {}
    monkeypatch.setattr(leaves.reminder_manager, "voice_references", none_found)
    result, counted = await _cancel(leaves, "cancelá el recordatorio", dispatcher=dispatcher)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Hay uno: tomar agua a las ")
    assert result.response.endswith(". Por voz no lo puedo nombrar; podés borrarlo desde la web.")
    assert calls == [False, True]


# Review T032-U1-RR-F2: only the same text and the same trigger are duplicates;
# the same text and HH:MM on another day is named by its day.

async def test_the_same_text_and_time_on_two_days_are_not_duplicates(leaves):
    dispatcher = leaves.dispatcher()
    manager, base = leaves.reminder_manager, _minute_start(1)
    first, second = await _two_shared(leaves, "tomar agua", "tomar agua", base, base + 86400)
    hm, spoken = manager._household_clock.format_time(base), manager.spoken_time_reference(first)
    assert manager._household_clock.format_time(second.trigger_at) == hm  # America/Montevideo has no DST
    d1, d2 = manager.spoken_day_reference(first), manager.spoken_day_reference(second)
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = f"cancelá el recordatorio de tomar agua de {spoken} {d1}"
    assert asked.response == (f"Encontré 2: tomar agua a las {hm} {d1} y tomar agua a las {hm} {d2}. "
                              f"Repetímelo con la hora y el día, por ejemplo: {example}.")
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [second.id]


async def test_a_superset_on_another_day_is_named_by_the_day_not_sent_to_the_web(leaves):
    dispatcher = leaves.dispatcher()
    manager, base = leaves.reminder_manager, _minute_start(1)
    plain, lemon = await _two_shared(leaves, "tomar agua", "tomar agua con limón", base, base + 86400)
    hm = manager._household_clock.format_time(base)
    assert manager._household_clock.format_time(lemon.trigger_at) == hm  # America/Montevideo has no DST
    asked, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua", dispatcher=dispatcher)
    assert counted == ZERO and asked.outcome.status.value == "needs_input"
    example = (f"cancelá el recordatorio de tomar agua de {manager.spoken_time_reference(plain)} "
               f"{manager.spoken_day_reference(plain)}")
    assert asked.response == (f"Encontré 2: tomar agua a las {hm} y tomar agua con limón a las {hm}. "
                              f"Repetímelo con la hora y el día, por ejemplo: {example}.")
    done, counted = await _cancel(leaves, example, dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [lemon.id]


async def test_olvidate_ya_del_recordatorio_cancels_exactly_that_reminder(leaves):
    """Review T032-U1-RR-F4: 'ya' between 'olvidate' and 'del' is a filler, not a creation request."""
    result, counted = await _cancel(leaves, "olvidate ya del recordatorio de tomar agua",
                                    "reminder_shared_active:tomar agua;reminder_shared_active:sacar la basura")
    assert counted == effects(reminder_transitions=1) and result.outcome.task_completed
    assert result.response.startswith("Cancelé el recordatorio tomar agua a las ")
    assert await _states(leaves) == {"sacar la basura": "active"}
