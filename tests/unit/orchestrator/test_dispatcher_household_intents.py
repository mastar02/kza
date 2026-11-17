"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

import asyncio

import pytest

from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.orchestrator.dispatcher import MultiUserOrchestrator, PathType
from tests.mocks.c02_matrix import C02Entry, effects
from tests.mocks.household_dispatch import CountedLeaves

ZERO = effects()
ROUTINE_CREATE = "Todavía no creo rutinas por voz. Podés crearlas desde la web."


def _d(id_, text, family, effect_class, outcome, path, status, persistence, spoken, *, counts=ZERO,
       conditions=None, unit="P3-F2", finding=None, seed=None, note=""):
    return C02Entry(id_, text, "dispatcher", family, effect_class, outcome, path, status, persistence, spoken,
                    counts, conditions or {}, unit=unit, finding=finding, seed=seed, note=note)


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
    _refused("C02-CAN-0101", "cancelá el recordatorio de agregar leche a la lista", "reminder",
             state="shared_list_compras;reminder_shared_active:agregar leche a la lista;queued_request",
             seed="C02-CAN-0001",
             note="interim until T032-U1 (voseo cancel, unique match): honest UNSUPPORTED; the list, the "
                  "reminder and the queue are untouched"),
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
    _refused("C02-CAN-0107", "olvidate del recordatorio de la basura", "reminder", state="queued_request"),
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
    result = await dispatcher.dispatch(user_id="unknown", text=entry.text,
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
