"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

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
       unit="T032-U1", finding="CORE-03", seed="C02-CAN-0003", note="no object named: asks which, lists both"),
    _d("C02-CAN-0004", "cancelá el recordatorio de agua", "reminder", "false_confirmation", "needs_input",
       "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_shared_fired:tomar agua"}, unit="T032-U1", seed="C02-CAN-0004",
       note="a fired reminder is never reported cancelled"),
    _d("C02-CAN-0005", "cancelá el recordatorio de la clave", "reminder", "privacy_disclosure", "needs_input",
       "fast_reminder", "needs_input", "not_written", "clarification",
       conditions={"state": "reminder_private_active:la clave", "speaker_label": "ana"}, actor="acoustic_label",
       unit="T032-U1", seed="C02-CAN-0005", note="private text never read nor spoken; the label adds no scope"),
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
    assert result.response.startswith("Decime cuál querés cancelar. Hay 2: tomar agua a las ")
    assert " y sacar la basura a las " in result.response and "clave" not in result.response


async def test_plural_without_object_never_cancels_them_all(leaves):
    result, counted = await _cancel(leaves, "eliminá todos los recordatorios", "reminder_shared_active:tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Por voz cancelo de a un recordatorio. Decime cuál querés cancelar. Hay uno: "
                                      "tomar agua a las ")


async def test_more_than_three_candidates_speak_three(leaves):
    state = ";".join(f"reminder_shared_active:regar {name}" for name in ("helecho", "cactus", "potus", "ficus"))
    result, counted = await _cancel(leaves, "cancelá el recordatorio de regar", state)
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Encontré 4, por ejemplo: regar helecho a las ")
    assert result.response.endswith(". Decime cuál, con la hora.")
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
    assert asked.response == (f"Encontré 2: tomar agua a las {clock.format_time(early.trigger_at)} y tomar agua "
                              f"a las {clock.format_time(late.trigger_at)}. Decime cuál, con la hora.")
    done, counted = await _cancel(leaves, f"cancelá el recordatorio de tomar agua de las "
                                          f"{clock.format_time(late.trigger_at)}", dispatcher=dispatcher)
    assert counted == effects(reminder_transitions=1) and done.outcome.task_completed
    assert [r.id for r in await leaves.reminder_manager.get_active(None, shared_only=True)] == [early.id]


async def test_identical_duplicates_point_to_the_web(leaves):
    trigger = time.time() + 3600
    for _ in range(2):
        await leaves.reminder_manager.create("household", "tomar agua", trigger, scope=ReminderScope.SHARED)
    result, counted = await _cancel(leaves, "cancelá el recordatorio de tomar agua")
    assert counted == ZERO and result.outcome.status.value == "needs_input"
    assert result.response.startswith("Hay 2 recordatorios iguales: tomar agua a las ")
    assert result.response.endswith("Podés borrar el que sobra desde la web.")


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


async def test_unclaimed_keyword_path_uses_the_same_unique_resolution(leaves):
    """operation None (keyword routing) resolves like a claimed cancel, never by similarity."""
    dispatcher = leaves.dispatcher()
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua;reminder_shared_active:tomar mate")
    before = leaves.snapshot()
    ambiguous = await dispatcher._fast_reminder_path("cancela el recordatorio de tomar", "unknown")
    assert ambiguous.outcome.status.value == "needs_input" and leaves.effects(before) == ZERO
    misheard = await dispatcher._fast_reminder_path("cancela el recordatorio de tomar aguja", "unknown")
    assert misheard.outcome.status.value == "needs_input" and leaves.effects(before) == ZERO
    done = await dispatcher._fast_reminder_path("cancela el recordatorio de tomar agua", "unknown")
    assert done.outcome.task_completed and leaves.effects(before) == effects(reminder_transitions=1)


async def test_cancel_never_touches_the_request_queue(leaves):
    from unittest.mock import MagicMock
    dispatcher = leaves.dispatcher()
    dispatcher.queue.cancel_user_request = MagicMock(side_effect=AssertionError("queue cancel"))
    await leaves.apply_state(dispatcher, "reminder_shared_active:tomar agua;queued_request")
    result = await dispatcher.dispatch(user_id="unknown", text="cancelá el recordatorio de tomar agua")
    assert result.outcome.task_completed and len(dispatcher.queue) == 1
