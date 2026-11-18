"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

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
    conditions = {k: v for k, v in (("service_filter", sf), ("state", state)) if v}
    return _d(id_, text, family, effect_class, status, path, status, "not_written", spoken,
              conditions=conditions, **kw)


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
    _refused("C02-CAN-0106", "pará el timer", "timer", "fast_timer", state="queued_request"),
    _refused("C02-CAN-0107", "olvidate del recordatorio de la basura", "reminder", state="queued_request",
             status="needs_input", spoken="clarification", unit="T032-U1",
             note="T032-U1 closes the P3-F2 interim UNSUPPORTED: no shared reminder is active, so it says so"),
    _refused("C02-TIM-0101", "poné un temporizador de 5 minutos", "timer", "fast_timer", sf="turn_on",
             note="never reaches vector search; timers land with T030-U1"),
    _refused("C02-TIM-0102", "poneme un tímer de media hora", "timer", "fast_timer"),
    _refused("C02-TIM-0103", "pausá el timer de la pasta", "timer", "fast_timer", finding="F12",
             note="'pausá' no longer reaches FAST_MUSIC"),
    _refused("C02-TIM-0104", "cancelá el timer de la pasta", "timer", "fast_timer", state="queued_request"),
    _refused("C02-TIM-0105", "temporizador de 5 minutos", "timer", "fast_timer"),
    _refused("C02-TIM-0106", "apagá el temporizador", "timer", "fast_timer", sf="turn_off", state="light_match"),
    _refused("C02-TIM-0107", "¿cuánto le falta al timer?", "timer", "fast_timer"),
    _refused("C02-ALM-0101", "poné una alarma de 10 minutos", "timer", "fast_timer", seed="C02-ALM-0001"),
    _refused("C02-ALM-0102", "poné una alarma a las 7", "reminder", seed="C02-ALM-0002", note=INTERIM_CREATE),
    _refused("C02-ALM-0103", "poné una alarma", "timer", "fast_timer", seed="C02-ALM-0004",
             note="no handler yet for timers: refusal rather than the R3a clarification"),
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
    # room within reminders.voice_choice_window_s; 1 = 'tomar agua fría' (the earlier trigger).
    _d(f"C02-CAN-{271 + i:04d}", answer, "reminder", "unauthorized_effect", status, "fast_reminder", status,
       persistence, spoken, counts=counts,
       conditions={"state": "reminder_shared_active:tomar agua fría;reminder_shared_active:tomar agua con gas;"
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
    assert result.response.endswith("; ¿cuál? Decime el número.")


async def test_plural_without_object_never_cancels_them_all(leaves):
    result, counted = await _cancel(leaves, "eliminá todos los recordatorios", "reminder_shared_active:tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Por voz cancelo de a un recordatorio. Hay uno. 1: tomar agua a las ")
    assert result.response.endswith("; ¿lo cancelo? Decime: el uno.")


async def test_more_than_three_candidates_list_the_three_next(leaves):
    state = ";".join(f"reminder_shared_active:regar {name}" for name in ("helecho", "cactus", "potus", "ficus"))
    result, counted = await _cancel(leaves, "cancelá el recordatorio de regar", state)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Encontré 4 parecidos; te digo los 3 más próximos. 1: regar helecho a las ")
    assert "; 2: regar cactus a las " in result.response and "; 3: regar potus a las " in result.response
    assert "ficus" not in result.response
    assert result.response.endswith("; ¿cuál? Decime el número. Si es otro, nombralo con más palabras.")


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
                              f"2: tomar agua a las {clock.format_time(late.trigger_at)}; ¿cuál? Decime el número.")
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
                              "¿lo cancelo? Decime: el uno.")
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
    assert "; 2: tomar agua con gas a las " in asked.response and asked.response.endswith("¿cuál? Decime el número.")
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
                              "¿cuál? Decime el número.")
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
                              f"2: {other} a las {clock.format_time(timed.trigger_at)}; ¿cuál? Decime el número.")
    assert {r.id for r in await manager.get_active(None, shared_only=True)} == {named.id, timed.id}
    done, counted = await _cancel(leaves, "el uno", dispatcher=dispatcher)
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
    assert asked.response == f"Hay uno. 1: llamar a mamá a las 3 a las {hm}; ¿lo cancelo? Decime: el uno."
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

TWO_SIMILAR = "reminder_shared_active:tomar agua fría;reminder_shared_active:tomar agua con gas;queued_request"


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


@pytest.mark.parametrize(("text", "number"), [
    ("el primero", 1), ("la primera", 1), ("el 1", 1), ("uno", 1), ("el uno", 1), ("el dos", 2), ("la segunda", 2),
    ("tres", 3), ("el tercero", 3), ("Nexa, la segunda, por favor", 2), ("bueno, el 2", 2), ("borrá el primero", 1),
    ("el de las 2", None), ("el primero de la lista", None), ("dos pastillas", None), ("el 0", None), ("", None),
    ("el", None), ("primero cancelá", None),
])
def test_the_numbered_answer_grammar(text, number):
    from src.orchestrator.dispatcher import _reminder_choice_number
    assert _reminder_choice_number(text) == number


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
