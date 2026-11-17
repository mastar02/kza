"""T030-U1: claimed timer commands against a real NamedTimerManager."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from src.core.task_outcome import PersistenceState, TaskStatus
from src.nlu.household_intents import HouseholdOperation, classify_household_intent
from src.timers.named_timers import NamedTimerManager, TimerState
from src.timers.voice_commands import (
    ALARM_CLARIFY,
    NO_ACTIVE,
    TIMERS_UNAVAILABLE,
    TIMERS_UNSUPPORTED,
    TimerVoiceCommands,
    describe,
)


class FrozenTimerManager(NamedTimerManager):
    """A started manager whose tick never advances: remaining times stay exact while a test reads them."""

    async def _update_loop(self):
        await asyncio.Event().wait()


@pytest.fixture
async def manager():
    timers = FrozenTimerManager(tts_callback=AsyncMock())
    await timers.start()
    yield timers
    await timers.stop()


async def say(commands: TimerVoiceCommands, text: str, zone_id: str = "zone_hall"):
    intent = classify_household_intent(text)
    return await commands.execute(intent.operation, intent.complement, text=text, zone_id=zone_id)


def states(manager: NamedTimerManager) -> dict:
    return {t.name: t.state for t in manager.get_all_timers()}


async def test_create_uses_household_actor_and_the_given_ring_zone(manager):
    reply = await say(TimerVoiceCommands(manager), "poné un temporizador de 5 minutos")
    (timer,) = manager.get_all_timers()
    assert (timer.user_id, timer.zone_id, timer.duration_seconds, timer.name) == ("household", "zone_hall", 300, "")
    assert timer.state is TimerState.RUNNING
    assert reply.response == "Temporizador de 5 minutos en marcha."
    assert (reply.outcome.status, reply.outcome.persistence) == (TaskStatus.COMPLETED, PersistenceState.VOLATILE)
    assert reply.outcome.effect.value == "observed" and reply.outcome.observation_scope == "software_state"


async def test_labelled_create_reads_back_label_and_duration(manager):
    reply = await say(TimerVoiceCommands(manager), "poné un timer de 8 minutos para la pasta", zone_id="default")
    assert reply.response == "Temporizador de pasta en marcha: 8 minutos."
    assert [(t.name, t.zone_id) for t in manager.get_all_timers()] == [("pasta", "default")]


@pytest.mark.parametrize("stopped_by", ["never_started", "stopped", "task_died"])
async def test_nothing_is_confirmed_unless_the_manager_runs(stopped_by):
    timers = NamedTimerManager(tts_callback=AsyncMock())
    if stopped_by == "stopped":
        await timers.start()
        await timers.stop()
    elif stopped_by == "task_died":
        await timers.start()
        timers._update_task.cancel()
        await asyncio.gather(timers._update_task, return_exceptions=True)
        assert timers._running and timers._update_task.done()
    commands = TimerVoiceCommands(timers)
    for text in ("poné un temporizador de 5 minutos", "cancelá el timer", "qué timers hay", "poné una alarma"):
        reply = await say(commands, text)
        assert (reply.response, reply.outcome.status) == (TIMERS_UNAVAILABLE, TaskStatus.FAILED)
        assert reply.outcome.persistence is PersistenceState.NOT_WRITTEN
    assert timers.get_all_timers() == []
    await timers.stop()


async def test_no_manager_is_unsupported():
    reply = await say(TimerVoiceCommands(None), "poné un temporizador de 5 minutos")
    assert (reply.response, reply.outcome.status) == (TIMERS_UNSUPPORTED, TaskStatus.UNSUPPORTED)


async def test_duplicate_label_needs_input_and_keeps_the_first(manager):
    commands = TimerVoiceCommands(manager)
    await say(commands, "poné un timer de 8 minutos para la pasta")
    reply = await say(commands, "poné un timer de 3 minutos para la Pasta")
    assert reply.outcome.status is TaskStatus.NEEDS_INPUT
    # The household classifier lowercases the heard words; the label still matches case-insensitively.
    assert reply.response == "Ya hay un temporizador de pasta. Cancelalo primero o usá otro nombre."
    assert [(t.name, t.duration_seconds) for t in manager.get_all_timers()] == [("pasta", 480)]
    manager.cancel_timer(manager.get_all_timers()[0].timer_id)
    assert (await say(commands, "poné un timer de 3 minutos para la pasta")).outcome.status is TaskStatus.COMPLETED


async def test_two_unnamed_timers_are_allowed(manager):
    commands = TimerVoiceCommands(manager)
    for _ in range(2):
        assert (await say(commands, "poné un timer de 5 minutos")).outcome.status is TaskStatus.COMPLETED
    assert len(manager.get_active_timers()) == 2


async def test_limit_of_active_timers(manager):
    manager.max_timers = 2
    commands = TimerVoiceCommands(manager)
    await say(commands, "poné un timer de 5 minutos")
    await say(commands, "poné un timer de 6 minutos")
    manager.pause_timer(manager.get_all_timers()[0].timer_id)
    reply = await say(commands, "poné un timer de 7 minutos")
    assert (reply.outcome.status, reply.response) == (
        TaskStatus.NEEDS_INPUT, "Ya hay 2 temporizadores activos. Cancelá alguno primero.")
    assert len(manager.get_all_timers()) == 2


@pytest.mark.parametrize(("text", "reply"), [
    ("poné un timer", "¿De cuánto tiempo es el temporizador?"),
    ("poné un timer de unos minutos", "¿De cuánto tiempo es el temporizador?"),
    ("poné un timer de 5", "No entendí la duración."),
    ("poné un timer para las 7", "Un temporizador va con una duración"),
    ("poné un timer de 25 horas", "Puedo poner temporizadores de hasta 24 horas."),
    ("poné un timer de 5 minutos en la cocina", "Por ahora el temporizador suena donde lo pedís."),
])
async def test_creation_issues_are_asked_with_zero_effects(manager, text, reply):
    result = await say(TimerVoiceCommands(manager), text)
    assert result.outcome.status is TaskStatus.NEEDS_INPUT and result.response.startswith(reply)
    assert result.outcome.persistence is PersistenceState.NOT_WRITTEN
    assert manager.get_all_timers() == []


async def test_alarm_without_time_asks(manager):
    reply = await say(TimerVoiceCommands(manager), "poné una alarma")
    assert (reply.response, reply.outcome.status) == (ALARM_CLARIFY, TaskStatus.NEEDS_INPUT)
    assert manager.get_all_timers() == []


@pytest.mark.parametrize(("text", "state"), [
    ("cancelá el timer de la pasta", TimerState.CANCELLED),
    ("sacá el timer de la pasta", TimerState.CANCELLED),
    ("borrá el temporizador de la pasta", TimerState.CANCELLED),
    ("pará el timer de la pasta", TimerState.CANCELLED),
    ("apagá el timer de la pasta", TimerState.CANCELLED),
    ("desactivá el timer de la pasta", TimerState.CANCELLED),
    ("pausá el timer de la pasta", TimerState.PAUSED),
])
async def test_one_named_timer_changes_and_the_other_does_not(manager, text, state):
    manager.create_timer("pasta", 600, user_id="household")
    manager.create_timer("salsa", 600, user_id="household")
    reply = await say(TimerVoiceCommands(manager), text)
    assert reply.outcome.status is TaskStatus.COMPLETED and reply.outcome.persistence is PersistenceState.VOLATILE
    assert states(manager) == {"pasta": state, "salsa": TimerState.RUNNING}
    assert "pasta" in reply.response


async def test_resume_a_paused_timer(manager):
    timer = manager.create_timer("pasta", 600)
    manager.pause_timer(timer.timer_id)
    reply = await say(TimerVoiceCommands(manager), "seguí el timer")
    assert timer.state is TimerState.RUNNING and reply.response.startswith("Sigue el temporizador de pasta; faltan ")
    again = await say(TimerVoiceCommands(manager), "reanudá el timer")
    assert (again.outcome.status, again.response) == (TaskStatus.NEEDS_INPUT, "No hay temporizadores en pausa.")


async def test_pause_needs_a_running_timer(manager):
    timer = manager.create_timer("pasta", 600)
    manager.pause_timer(timer.timer_id)
    reply = await say(TimerVoiceCommands(manager), "pausá el timer")
    assert (reply.outcome.status, reply.response) == (TaskStatus.NEEDS_INPUT, "No hay temporizadores corriendo.")


async def test_ambiguous_command_changes_nothing_and_asks(manager):
    manager.create_timer("pasta", 600)
    manager.create_timer("", 300)
    reply = await say(TimerVoiceCommands(manager), "cancelá el timer")
    assert reply.outcome.status is TaskStatus.NEEDS_INPUT
    assert reply.response == "Hay 2 temporizadores: de pasta y de 5 minutos. ¿Cuál cancelo? Decímelo completo."
    assert {t.state for t in manager.get_all_timers()} == {TimerState.RUNNING}


async def test_unknown_name_changes_nothing(manager):
    manager.create_timer("pasta", 600)
    reply = await say(TimerVoiceCommands(manager), "cancelá el timer de la salsa")
    assert (reply.outcome.status, reply.response) == (TaskStatus.NEEDS_INPUT, "No encontré un temporizador de salsa.")
    assert states(manager) == {"pasta": TimerState.RUNNING}
    by_duration = await say(TimerVoiceCommands(manager), "cancelá el timer de 5 minutos")
    assert by_duration.response == "No encontré un temporizador de 5 minutos."


async def test_whole_word_match_never_substring(manager):
    """'sal' must not pick 'salsa'; 'pasta' picks 'pasta de la abuela' only when exact is absent."""
    manager.create_timer("salsa", 600)
    assert (await say(TimerVoiceCommands(manager), "cancelá el timer de la sal")).outcome.status is TaskStatus.NEEDS_INPUT
    manager.create_timer("pasta de la abuela", 600)
    reply = await say(TimerVoiceCommands(manager), "pausá el timer de la pasta")
    assert reply.outcome.status is TaskStatus.COMPLETED
    assert states(manager) == {"salsa": TimerState.RUNNING, "pasta de la abuela": TimerState.PAUSED}


async def test_exact_label_wins_over_partial(manager):
    manager.create_timer("pasta de la abuela", 600)
    manager.create_timer("pasta", 600)
    await say(TimerVoiceCommands(manager), "cancelá el timer de la pasta")
    assert states(manager) == {"pasta de la abuela": TimerState.RUNNING, "pasta": TimerState.CANCELLED}


async def test_unnamed_timer_is_picked_by_duration(manager):
    manager.create_timer("", 300)
    manager.create_timer("", 600)
    reply = await say(TimerVoiceCommands(manager), "cancelá el timer de 10 minutos")
    assert reply.response == "Cancelé el temporizador de 10 minutos."
    assert sorted((t.duration_seconds, t.state) for t in manager.get_all_timers()) == [
        (300, TimerState.RUNNING), (600, TimerState.CANCELLED)]


async def test_cancel_all(manager):
    manager.create_timer("pasta", 600)
    manager.create_timer("salsa", 600)
    paused = manager.create_timer("arroz", 600)
    manager.pause_timer(paused.timer_id)
    reply = await say(TimerVoiceCommands(manager), "cancelá todos los timers")
    assert reply.response == "Cancelé 3 temporizadores." and reply.outcome.status is TaskStatus.COMPLETED
    assert {t.state for t in manager.get_all_timers()} == {TimerState.CANCELLED}
    empty = await say(TimerVoiceCommands(manager), "cancelá los timers")
    assert (empty.outcome.status, empty.response) == (TaskStatus.NEEDS_INPUT, NO_ACTIVE)


async def test_query_lists_active_timers_and_is_read_only(manager):
    commands = TimerVoiceCommands(manager)
    none = await say(commands, "qué timers hay")
    assert (none.response, none.outcome.status) == (NO_ACTIVE, TaskStatus.COMPLETED)
    assert none.outcome.persistence is PersistenceState.NOT_REQUIRED
    pasta = manager.create_timer("pasta", 600)
    one = await say(commands, "¿cuánto le falta al timer?")
    assert one.response == "Al temporizador de pasta, faltan 10 minutos."
    manager.create_timer("", 90)
    manager.pause_timer(pasta.timer_id)
    both = await say(commands, "qué timers hay")
    assert both.response == ("Hay 2 temporizadores: de pasta, faltan 10 minutos, en pausa; "
                             "de 1 minuto y 30 segundos, faltan 1 minuto y 30 segundos.")
    named = await say(commands, "cuánto falta para el timer de la pasta")
    assert named.response == "Al temporizador de pasta, faltan 10 minutos, en pausa."
    missing = await say(commands, "cuánto falta para el timer de la salsa")
    assert missing.outcome.status is TaskStatus.NEEDS_INPUT
    assert {t.state for t in manager.get_all_timers()} == {TimerState.RUNNING, TimerState.PAUSED}


async def test_completed_or_cancelled_timers_are_not_active(manager):
    done = manager.create_timer("pasta", 600)
    done.state = TimerState.COMPLETED
    assert (await say(TimerVoiceCommands(manager), "pará el timer")).response == NO_ACTIVE
    assert (await say(TimerVoiceCommands(manager), "qué timers hay")).response == NO_ACTIVE


async def test_labels_never_reach_info_logs(manager, caplog):
    commands = TimerVoiceCommands(manager)
    with caplog.at_level(logging.DEBUG):
        await say(commands, "poné un temporizador de 1 minuto para la pastilla")
        await say(commands, "poné un temporizador de 2 minutos para la pastilla")
        await say(commands, "pausá el timer de la pastilla")
        await say(commands, "qué timers hay")
        await say(commands, "cancelá el timer de la pastilla")
        timer = manager.get_all_timers()[0]
        quiet = NamedTimerManager()  # no voice callback: the ring's spacing sleeps are skipped
        await quiet._announce_warning(timer, 30)
        await quiet._announce_complete(timer)
    assert [r for r in caplog.records if r.levelno >= logging.INFO and "pastilla" in r.getMessage()] == []
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[Timer] created id=") for m in messages)
    assert any(m.startswith("[Timer] cancel_timer ids=") for m in messages)


async def test_describe_names_by_label_or_duration(manager):
    assert describe(manager.create_timer("pasta", 60)) == "de pasta"
    assert describe(manager.create_timer("", 90)) == "de 1 minuto y 30 segundos"


async def test_other_operations_need_input(manager):
    reply = await TimerVoiceCommands(manager).execute(HouseholdOperation.EXECUTE, "", text="x", zone_id="default")
    assert reply.outcome.status is TaskStatus.NEEDS_INPUT and manager.get_all_timers() == []
