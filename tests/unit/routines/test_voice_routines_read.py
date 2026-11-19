"""T033-U1: read-only voice routines over the persistent scheduler store.

Listing and consulting read routines.db through the RoutineScheduler, never
the legacy Chroma index; detection is lexical, so a turn that is not a routine
question never reaches the store (counted calls).
"""

import logging
from unittest.mock import MagicMock

import pytest

from src.core.task_outcome import TaskStatus
from src.orchestrator.dispatcher import PathType
from src.routines.models import ActionStatus, RoutineActionResult, RunStatus, parse_routine_spec
from src.routines.routine_manager import RoutineManager
from src.routines.routine_scheduler import RoutineScheduler
from src.routines.routine_store import RoutineStore
from src.routines.voice_routines import RoutineReadKind, RoutineReadQuery, VoiceRoutineService, name_key
from tests.mocks.c02_matrix import effects
from tests.mocks.household_dispatch import CountedLeaves

READS = ("list_all", "get", "get_run", "get_latest_run", "get_run_by_occurrence")
WRITES = ("save", "delete", "claim_run", "mark_action_sending", "record_action", "finish_run",
          "recover_interrupted")


def spec(name, *, enabled=True, schedule=None, entity="light.living"):
    return parse_routine_spec({"name": name, "description": "", "enabled": enabled,
                               "timezone": "America/Montevideo", "schedule": schedule, "conditions": [],
                               "actions": [{"entity_id": entity, "operation": "turn_on"}]})


class CountedStore:
    """A real RoutineStore whose reads and writes are counted by name."""

    def __init__(self, store: RoutineStore) -> None:
        self.store = store
        self.calls: list[str] = []
        for name in READS + WRITES:
            original = getattr(store, name)

            async def counted(*args, _name=name, _original=original, **kwargs):
                self.calls.append(_name)
                return await _original(*args, **kwargs)
            setattr(store, name, counted)

    def reads(self) -> int:
        return sum(1 for name in self.calls if name in READS)

    def writes(self) -> int:
        return sum(1 for name in self.calls if name in WRITES)


@pytest.fixture
async def routines(tmp_path):
    store = RoutineStore(str(tmp_path / "routines.db"))
    await store.initialize()
    scheduler = RoutineScheduler(MagicMock(), store=store)
    yield scheduler, store
    await store.close()


async def run_with(store, routine_id, status, key="manual-1"):
    """A receipt in ``status`` committed through the store's own transitions."""
    claim = await store.claim_run(routine_id, key, actor="household", source="manual")
    if status is RunStatus.RUNNING:
        return claim.run
    if status is RunStatus.SUCCEEDED:
        await store.mark_action_sending(claim.run.run_id, 0)
        await store.record_action(claim.run.run_id, RoutineActionResult(0, "light.living", ActionStatus.CONFIRMED))
    if status is RunStatus.UNCERTAIN:
        await store.mark_action_sending(claim.run.run_id, 0)
    return await store.finish_run(claim.run.run_id, status)


# ---------------------------------------------------------------- detection

LIST_TEXTS = [
    "qué rutinas hay", "¿Qué rutinas tenemos?", "Nexa, ¿cuáles son mis rutinas?", "cuántas rutinas hay",
    "decime las rutinas", "mostrame las automatizaciones", "quiero ver las rutinas", "a ver qué rutinas nuevas hay",
    "¿hay alguna rutina?", "¿cómo salieron las rutinas?", "¿hay una rutina que apague las luces?",
    "¿la rutina se ejecutó?", "¿qué rutina se ejecutó hoy?",
    # The legacy RoutineManager list phrases now read the scheduler store.
    *RoutineManager.LIST_PATTERNS,
]

CONSULT_TEXTS = [
    ("¿cómo salió la rutina de la mañana?", "manana"),
    ("¿Se ejecutó la rutina Buenas Noches?", "buenas noches"),
    ("¿a qué hora corre la rutina riego?", "riego"),
    ("¿qué hace la rutina buenas noches?", "buenas noches"),
    ("decime el estado de la rutina riego", "riego"),
    ("¿la rutina buenas noches anduvo bien hoy?", "buenas noches"),
    ("la rutina riego falló", "riego"),
    ("¿está activa la rutina riego?", "riego"),
    ("nexa, ¿cuándo corre la automatización riego?", "riego"),
]

NOT_ROUTINE_READS = [
    # CORE-04: a bare "cuando" is a question, never a routine.
    "cuando juega Peñarol", "cuándo es el cumpleaños de Ana", "¿cuándo es el partido?",
    # No routine noun at a word boundary.
    "algo rutinario, pará", "apagá la luz del living", "qué tiempo hace", "qué recordatorios tengo",
    "agregá leche a la lista de compras", "poné un timer de 5 minutos",
    # Execution, changes, stops and creation are other units (U2/U3), never a read.
    "ejecutá la rutina buenas noches", "corré la rutina riego", "poné la rutina de la mañana",
    "activá la rutina buenas noches", "desactivá la rutina de la noche", "pará la rutina de la noche",
    "cancelá la rutina de la noche", "borrá la rutina riego", "creá una rutina que apague las luces",
    "quiero crear una nueva rutina que apague las luces", "nueva rutina para la noche",
    "automatizá las luces del living", "cambiá la rutina riego", "renombrá la rutina riego",
    # A routine noun without a question.
    "rutina buenas noches", "la rutina riego",
]


@pytest.mark.parametrize("text", LIST_TEXTS)
def test_routine_list_questions_are_detected(text):
    assert VoiceRoutineService.detect(text) == RoutineReadQuery(RoutineReadKind.LIST)


@pytest.mark.parametrize(("text", "key"), CONSULT_TEXTS)
def test_routine_consult_questions_carry_the_heard_name(text, key):
    query = VoiceRoutineService.detect(text)
    assert query is not None and query.kind is RoutineReadKind.CONSULT
    assert key in query.names


@pytest.mark.parametrize("text", NOT_ROUTINE_READS)
def test_other_turns_are_not_routine_reads(text):
    assert VoiceRoutineService.detect(text) is None


@pytest.mark.parametrize(("stored", "heard"), [
    ("Rutina de la Mañana", "la rutina de la mañana"), ("Buenas noches", "buenas NOCHES"),
    ("Riego", "rutina riego"), ("Café", "cafe"),
])
def test_name_key_folds_accents_case_and_leading_articles(stored, heard):
    assert name_key(stored) == name_key(heard)


# ---------------------------------------------------------------- reads


async def test_list_reads_every_saved_routine_in_catalog_order(routines):
    scheduler, store = routines
    morning = await store.save(spec("Buenos días", schedule={"at": "07:00", "weekdays": [0, 1, 2, 3, 4]}),
                               actor="admin")
    night = await store.save(spec("Buenas noches", enabled=False), actor="admin")
    watering = await store.save(spec("Riego"), actor="admin")
    await run_with(store, watering.routine_id, RunStatus.FAILED)
    counted = CountedStore(store)
    result = await VoiceRoutineService(scheduler).answer(RoutineReadQuery(RoutineReadKind.LIST))
    assert result.response == "Hay 3 rutinas: Buenos días, Buenas noches (desactivada) y Riego."
    assert result.outcome.status is TaskStatus.COMPLETED and result.outcome.task_completed
    # The catalog is the scheduler's own projection: every saved routine (enabled or not) and its latest receipt.
    assert [(e.routine_id, e.name, e.enabled, e.revision, e.latest_run_status) for e in result.entries] == [
        (morning.routine_id, "Buenos días", True, 1, None), (night.routine_id, "Buenas noches", False, 1, None),
        (watering.routine_id, "Riego", True, 1, "failed")]
    assert counted.writes() == 0


async def test_an_empty_store_says_so(routines):
    scheduler, _ = routines
    result = await VoiceRoutineService(scheduler).answer(RoutineReadQuery(RoutineReadKind.LIST))
    assert result.response == "No hay rutinas guardadas. Podés crearlas desde la web."
    assert result.outcome.status is TaskStatus.COMPLETED and result.entries == ()


async def test_a_long_list_names_ten_and_counts_the_rest(routines):
    scheduler, store = routines
    for index in range(12):
        await store.save(spec(f"R{index}"), actor="admin")
    result = await VoiceRoutineService(scheduler).answer(RoutineReadQuery(RoutineReadKind.LIST))
    assert result.response.startswith("Hay 12 rutinas: R0, R1, ") and result.response.endswith("R9 y 2 más.")
    assert len(result.entries) == 12


@pytest.mark.parametrize(("status", "said"), [
    (None, "Todavía no se ejecutó."),
    (RunStatus.RUNNING, "Se está ejecutando ahora."),
    (RunStatus.SUCCEEDED, "La última ejecución terminó bien."),
    (RunStatus.FAILED, "La última ejecución falló."),
    (RunStatus.UNCERTAIN, "En la última ejecución no pude confirmar si se aplicaron todas sus acciones."),
    (RunStatus.SKIPPED, "La última vez no se ejecutó porque no se cumplían sus condiciones."),
    (RunStatus.INTERRUPTED, "La última ejecución se interrumpió."),
])
async def test_consult_says_the_state_and_the_last_run(routines, status, said):
    scheduler, store = routines
    routine = await store.save(spec("Rutina de la mañana", schedule={"at": "07:30", "weekdays": [0, 1, 2, 3, 4]}),
                               actor="admin")
    if status is not None:
        await run_with(store, routine.routine_id, status)
    query = VoiceRoutineService.detect("¿cómo salió la rutina de la mañana?")
    result = await VoiceRoutineService(scheduler).answer(query)
    assert result.response == ("La rutina Rutina de la mañana está activada y corre a las 7:30 de lunes a viernes. "
                               + said)
    assert result.outcome.status is TaskStatus.COMPLETED
    assert result.entries[0].latest_run_status == (status.value if status is not None else None)


@pytest.mark.parametrize(("routine_spec", "said"), [
    (dict(enabled=False), "La rutina Riego está desactivada."),
    (dict(), "La rutina Riego está activada y no tiene horario: se ejecuta solo cuando se pide."),
    (dict(schedule={"at": "01:05", "weekdays": [0, 1, 2, 3, 4, 5, 6]}),
     "La rutina Riego está activada y corre a la 1:05 todos los días."),
    (dict(schedule={"at": "09:00", "weekdays": [5, 6]}),
     "La rutina Riego está activada y corre a las 9:00 los fines de semana."),
    (dict(schedule={"at": "18:00", "weekdays": [0, 2, 4]}),
     "La rutina Riego está activada y corre a las 18:00 los lunes, miércoles y viernes."),
])
async def test_consult_describes_enabled_state_and_schedule(routines, routine_spec, said):
    scheduler, store = routines
    await store.save(spec("Riego", **routine_spec), actor="admin")
    result = await VoiceRoutineService(scheduler).answer(VoiceRoutineService.detect("¿cómo anda la rutina riego?"))
    assert result.response == said + " Todavía no se ejecutó."


async def test_unknown_and_ambiguous_names_ask_without_reading_receipts(routines):
    scheduler, store = routines
    await store.save(spec("Buenas noches"), actor="admin")
    await store.save(spec("BUENAS NOCHES"), actor="admin")
    service = VoiceRoutineService(scheduler)
    counted = CountedStore(store)
    unknown = await service.answer(VoiceRoutineService.detect("¿cómo salió la rutina riego?"))
    assert unknown.response == "No encontré una rutina con ese nombre. Podés preguntarme qué rutinas hay."
    assert unknown.outcome.status is TaskStatus.NEEDS_INPUT and not unknown.outcome.task_completed
    ambiguous = await service.answer(VoiceRoutineService.detect("¿cómo salió la rutina buenas noches?"))
    assert ambiguous.response == ("Hay 2 rutinas con ese nombre: Buenas noches y BUENAS NOCHES. "
                                  "¿Cuál querés consultar?")
    assert ambiguous.outcome.status is TaskStatus.NEEDS_INPUT
    assert counted.calls == ["list_all", "list_all"] and counted.writes() == 0


async def test_unbound_or_failing_store_answers_unavailable_without_names_in_logs(routines, caplog):
    scheduler, store = routines
    await store.save(spec("Secreto familiar"), actor="admin")
    service = VoiceRoutineService()
    unbound = await service.answer(RoutineReadQuery(RoutineReadKind.LIST))
    assert unbound.response == "No puedo consultar las rutinas ahora. Podés verlas desde la web."
    assert unbound.outcome.status is TaskStatus.FAILED
    service.bind_scheduler(scheduler)
    await store.close()
    with caplog.at_level(logging.DEBUG, logger="src.routines.voice_routines"):
        failed = await service.answer(VoiceRoutineService.detect("¿cómo salió la rutina secreto familiar?"))
    assert failed.outcome.status is TaskStatus.FAILED and failed.response == unbound.response
    assert "secreto" not in caplog.text.lower()


async def test_routine_names_are_never_logged_above_debug(routines, caplog):
    scheduler, store = routines
    await store.save(spec("Secreto familiar"), actor="admin")
    service = VoiceRoutineService(scheduler)
    with caplog.at_level(logging.INFO, logger="src.routines.voice_routines"):
        await service.answer(RoutineReadQuery(RoutineReadKind.LIST))
        await service.answer(VoiceRoutineService.detect("¿cómo salió la rutina secreto familiar?"))
    assert caplog.records and "secreto" not in caplog.text.lower()


async def test_scheduler_is_bound_once(routines):
    scheduler, _ = routines
    service = VoiceRoutineService()
    service.bind_scheduler(None)  # the routine subsystem failed to start: reads stay unavailable
    service.bind_scheduler(scheduler)
    service.bind_scheduler(scheduler)
    assert service.scheduler is scheduler
    with pytest.raises(ValueError, match="routine_scheduler_already_bound"):
        service.bind_scheduler(RoutineScheduler(MagicMock(), store=MagicMock()))
    assert service.scheduler is scheduler


# ---------------------------------------------------------------- dispatcher


@pytest.fixture
async def leaves(tmp_path):
    counted = CountedLeaves(tmp_path)
    await counted.start()
    yield counted
    await counted.close()


NON_ROUTINE_TURNS = [
    "cuando juega Peñarol", "cuándo es el cumpleaños de Ana", "algo rutinario, pará", "algo rutinario",
    "apagá la luz del living",
    "qué tiempo hace", "qué recordatorios tengo", "agregá leche a la lista de compras", "qué listas hay",
    "poné un timer de 5 minutos", "ejecutá la rutina buenas noches", "desactivá la rutina de la noche",
    "creá una rutina que apague las luces", "explicame la relatividad",
]


async def test_non_routine_turns_never_read_the_routine_store(leaves, routines):
    scheduler, store = routines
    await store.save(spec("Buenas noches"), actor="admin")
    counted = CountedStore(store)
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService(scheduler))
    for text in NON_ROUTINE_TURNS:
        result = await dispatcher.dispatch(user_id="unknown", text=text)
        assert result.outcome is not None, text
        if result.intent == "routine":
            # Honest refusals, or the legacy handler (a spy here) with no completion evidence.
            assert result.outcome.status in (TaskStatus.UNSUPPORTED, TaskStatus.UNCERTAIN), text
    assert counted.calls == []


@pytest.mark.parametrize(("success", "status"), [(True, TaskStatus.UNCERTAIN), (False, TaskStatus.FAILED)])
async def test_legacy_routine_results_carry_an_outcome_without_completion(leaves, success, status):
    leaves.routines.handle.return_value = {"handled": True, "success": success, "response": "legacy"}
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService())
    result = await dispatcher.dispatch(user_id="unknown", text="algo rutinario")
    assert (result.path, result.intent) == (PathType.FAST_ROUTINE, "routine")
    assert result.outcome.status is status and not result.outcome.task_completed


async def test_cuando_questions_no_longer_route_to_routines(leaves, routines):
    scheduler, _ = routines
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService(scheduler))
    for text in ("cuando juega peñarol", "cuándo es el cumpleaños de ana", "cuando llega el pedido"):
        assert dispatcher._classify_request(text)[0] is not PathType.FAST_ROUTINE
        leaves.cloud.reset_mock()
        before = leaves.snapshot()
        result = await dispatcher.dispatch(user_id="unknown", text=text)
        assert result.path is not PathType.FAST_ROUTINE
        assert leaves.effects(before) == effects(cloud_calls=1)


@pytest.mark.parametrize("text", ["qué rutinas hay", "¿cuáles son mis rutinas?", "decime las rutinas"])
async def test_claimed_and_unclaimed_list_questions_read_the_store_with_an_outcome(leaves, routines, text):
    scheduler, store = routines
    await store.save(spec("Buenas noches"), actor="admin")
    counted = CountedStore(store)
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService(scheduler))
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.intent) == (PathType.FAST_ROUTINE, "routine")
    assert result.response == "Hay una rutina: Buenas noches."
    assert result.outcome.status is TaskStatus.COMPLETED and result.success
    assert counted.writes() == 0 and counted.reads() == 2
    # Read-only: no HA send, no legacy routine handler, no vector search, no cloud.
    assert leaves.effects(before) == effects()
    leaves.chroma.list_routines.assert_not_called()


async def test_unclaimed_consult_reads_the_store_and_never_the_legacy_handler(leaves, routines):
    scheduler, store = routines
    routine = await store.save(spec("Riego"), actor="admin")
    await run_with(store, routine.routine_id, RunStatus.SKIPPED)
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService(scheduler))
    text = "¿cómo salió la rutina riego?"
    assert dispatcher.household_intents.classify(text) is None
    before = leaves.snapshot()
    result = await dispatcher.dispatch(user_id="unknown", text=text)
    assert (result.path, result.intent) == (PathType.FAST_ROUTINE, "routine")
    assert result.response == ("La rutina Riego está activada y no tiene horario: se ejecuta solo cuando se pide. "
                               "La última vez no se ejecutó porque no se cumplían sus condiciones.")
    assert result.outcome.status is TaskStatus.COMPLETED and "routine" in result.timings
    assert leaves.effects(before) == effects()
    assert leaves.routines.method_calls == []


async def test_unknown_routine_name_is_needs_input_not_completion(leaves, routines):
    scheduler, _ = routines
    dispatcher = leaves.dispatcher(voice_routines=VoiceRoutineService(scheduler))
    result = await dispatcher.dispatch(user_id="unknown", text="¿cómo salió la rutina riego?")
    assert result.path is PathType.FAST_ROUTINE and not result.success
    assert result.outcome.status is TaskStatus.NEEDS_INPUT


async def test_without_the_service_the_claimed_query_keeps_its_honest_refusal(leaves):
    result = await leaves.dispatcher().dispatch(user_id="unknown", text="qué rutinas hay")
    assert result.response == "Todavía no manejo rutinas por voz. Podés verlas desde la web."
    assert result.outcome.status is TaskStatus.UNSUPPORTED
