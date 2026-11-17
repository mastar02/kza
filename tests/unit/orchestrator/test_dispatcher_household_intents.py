"""P3-F2 D1-D3: claimed household turns reach their family path with zero foreign effects."""

import pytest

from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.orchestrator.dispatcher import MultiUserOrchestrator, PathType
from tests.mocks.c02_matrix import C02Entry, effects
from tests.mocks.household_dispatch import CountedLeaves

ZERO = effects()


def _d(id_, text, family, effect_class, outcome, path, status, persistence, spoken, *, counts=ZERO,
       conditions=None, unit="P3-F2", finding=None, seed=None, note=""):
    return C02Entry(id_, text, "dispatcher", family, effect_class, outcome, path, status, persistence, spoken,
                    counts, conditions or {}, unit=unit, finding=finding, seed=seed, note=note)


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
    _refused("C02-CAN-0106", "pará el timer", "timer", "fast_timer", state="queued_request"),
    _refused("C02-CAN-0107", "olvidate del recordatorio de la basura", "reminder", state="queued_request"),
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
