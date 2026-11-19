"""T030-U1: claimed timer commands against a real NamedTimerManager."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from src.core.task_outcome import PersistenceState, TaskStatus
from src.nlu.household_intents import HouseholdOperation, classify_household_intent
from src.timers.named_timers import RING_FOLLOW_UP_S, NamedTimerManager, RingDelivery, TimerState
from src.timers.voice_commands import (
    ALARM_CLARIFY,
    NO_ACTIVE,
    NO_RING_ROUTE,
    TIMERS_UNAVAILABLE,
    TIMERS_UNSUPPORTED,
    TimerVoiceCommands,
    describe,
)


class FrozenClock:
    """Monotonic time that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FrozenTimerManager(NamedTimerManager):
    """A started manager whose clock and tick never advance on their own: remaining times stay exact."""

    def __init__(self, **kwargs) -> None:
        self.frozen = FrozenClock()
        super().__init__(clock=self.frozen, **kwargs)

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


@pytest.mark.parametrize(("text", "expected"), [
    ("poné un temporizador de 5 minutos", [60]),
    ("poné un temporizador de 3 minutos", [60]),
    ("poné un temporizador de 2 minutos", []),
    ("poné un temporizador de minuto y medio", []),
    ("poné un temporizador de 30 segundos", []),
])
async def test_voice_timer_warns_only_at_one_minute_and_never_when_short(manager, text, expected):
    """T030-U1-Q4: a single 'le queda 1 minuto' warning, none for timers of 2 minutes or less."""
    await say(TimerVoiceCommands(manager), text)
    (timer,) = manager.get_all_timers()
    assert timer.announce_at == expected


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


async def test_create_without_a_ring_zone_is_refused_and_other_commands_still_work(manager):
    """T030-U1 R1: no ring zone (no origin room, no shared-notice zone) never confirms a timer."""
    commands = TimerVoiceCommands(manager)
    reply = await say(commands, "poné un temporizador de 5 minutos", zone_id=None)
    assert (reply.response, reply.outcome.status) == (NO_RING_ROUTE, TaskStatus.UNSUPPORTED)
    assert reply.outcome.persistence is PersistenceState.NOT_WRITTEN and manager.get_all_timers() == []
    manager.create_timer("pasta", 600)
    assert (await say(commands, "cancelá el timer", zone_id=None)).outcome.status is TaskStatus.COMPLETED


@pytest.mark.parametrize(("config", "require_zone", "expected"), [
    ({"reminders": {"output_zone": "zone_living"}, "tts": {"streaming": {"default_output_zone": None}}}, True,
     "zone_living"),
    ({"reminders": {"output_zone": None}, "tts": {"streaming": {"default_output_zone": "zone_hall"}}}, True,
     "zone_hall"),
    ({"reminders": {"output_zone": None}, "tts": {"streaming": {"default_output_zone": None}}}, True, None),
    ({}, True, None),
    ({"reminders": {"output_zone": "zone_living"}}, False, "default"),
    ({}, False, "default"),
])
def test_timer_fallback_zone_is_a_deliverable_route_or_none(config, require_zone, expected):
    """main's fallback: the shared-notice zone when zones exist, the one local output without zones."""
    from unittest.mock import MagicMock

    from src.core.voice_delivery import LocalVoiceDelivery
    from src.pipeline.voice_services import timer_fallback_zone
    delivery = LocalVoiceDelivery(MagicMock(), config.get("tts", {}).get("streaming", {}).get("default_output_zone"),
                                  require_zone)
    assert timer_fallback_zone(config, delivery) == expected
    assert timer_fallback_zone(config, None) is None


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


async def test_courtesy_word_is_neither_a_label_nor_a_duplicate(manager):
    """T030-U1 R2: '..., gracias' twice gives two unnamed timers, read back by duration."""
    commands = TimerVoiceCommands(manager)
    for _ in range(2):
        reply = await say(commands, "poné un timer de 5 minutos, gracias")
        assert (reply.outcome.status, reply.response) == (TaskStatus.COMPLETED, "Temporizador de 5 minutos en marcha.")
    assert [(t.name, t.duration_seconds) for t in manager.get_all_timers()] == [("", 300), ("", 300)]


@pytest.mark.parametrize(("text", "state"), [
    ("cancelá el timer ya", TimerState.CANCELLED),
    ("pausá el timer un momento", TimerState.PAUSED),
    ("cancelá el timer que puse", TimerState.CANCELLED),
])
async def test_filler_selector_picks_the_only_timer(manager, text, state):
    """T030-U1 R2: a filler word is not a timer name; the only timer is the one meant."""
    manager.create_timer("pasta", 600)
    reply = await say(TimerVoiceCommands(manager), text)
    assert reply.outcome.status is TaskStatus.COMPLETED and "pasta" in reply.response
    assert states(manager) == {"pasta": state}


async def test_filler_after_a_label_still_picks_that_timer(manager):
    manager.create_timer("pasta", 600)
    manager.create_timer("salsa", 600)
    reply = await say(TimerVoiceCommands(manager), "pará el timer de la pasta nomás")
    assert reply.response == "Cancelé el temporizador de pasta."
    assert states(manager) == {"pasta": TimerState.CANCELLED, "salsa": TimerState.RUNNING}


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


@pytest.mark.parametrize("text", ["cancelá los timers", "cancelá los temporizadores", "cancelá las alarmas",
                                  "cancelá los despertadores"])
async def test_plural_noun_cancels_every_timer(manager, text):
    """T030-U1 R6 (reviewer mutant V11): the plural names every timer, without 'todos'."""
    manager.create_timer("pasta", 600)
    manager.create_timer("", 300)
    reply = await say(TimerVoiceCommands(manager), text)
    assert (reply.outcome.status, reply.response) == (TaskStatus.COMPLETED, "Cancelé 2 temporizadores.")
    assert {t.state for t in manager.get_all_timers()} == {TimerState.CANCELLED}


async def test_unnamed_timer_keeps_its_original_duration_as_name_after_time_passes(manager):
    """T030-U1 R6 (reviewer mutant V19): named by duration_seconds, never by what is left."""
    timer = manager.create_timer("", 300)
    manager.frozen.now += 180  # three minutes went by (T030-U2: the time left follows the deadline)
    paused = await say(TimerVoiceCommands(manager), "pausá el timer")
    assert paused.response == "Pausé el temporizador de 5 minutos; le quedan 2 minutos."
    resumed = await say(TimerVoiceCommands(manager), "seguí el timer de 5 minutos")
    assert resumed.response == "Sigue el temporizador de 5 minutos; faltan 2 minutos."
    query = await say(TimerVoiceCommands(manager), "¿cuánto le falta al timer?")
    assert query.response == "Al temporizador de 5 minutos, faltan 2 minutos."
    cancelled = await say(TimerVoiceCommands(manager), "cancelá el timer")
    assert cancelled.response == "Cancelé el temporizador de 5 minutos."
    assert timer.state is TimerState.CANCELLED


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


@pytest.mark.parametrize(("setup", "text"), [
    ((), "cancelá el timer"),
    ((), "qué timers hay"),
    (("poné un timer de pasta de 5 minutos",), "cancelá el timer de fideos"),
    (("poné un timer de pasta de 5 minutos",), "seguí el timer"),
    (("poné un timer de pasta de 5 minutos",), "cuánto falta para el timer de 10 minutos"),
])
async def test_no_matching_timer_is_flagged_for_the_caller(manager, setup, text):
    """Review T031-U2-RV3: the dispatcher adds the alarm reminders only when no timer matched."""
    commands = TimerVoiceCommands(manager)
    for create in setup:
        await say(commands, create)
    assert (await say(commands, text)).no_match


@pytest.mark.parametrize("text", ["cancelá el timer", "qué timers hay", "poné un timer de 5 minutos"])
async def test_a_matching_timer_or_a_creation_is_not_flagged(manager, text):
    commands = TimerVoiceCommands(manager)
    await say(commands, "poné un temporizador de 3 minutos")
    assert not (await say(commands, text)).no_match


# ---------------------------------------------------------------- T030-U2: the ring and the voice stop


async def _ring(manager: FrozenTimerManager, timer) -> None:
    """Make ``timer`` due now and let its ring start (it then waits in the gap between repetitions)."""
    timer.deadline = manager.frozen.now
    manager._tick()
    for _ in range(20):
        await asyncio.sleep(0)
    assert timer.state is TimerState.COMPLETED and manager.ringing_timers() == [timer]


@pytest.mark.parametrize("text", ["pará el timer", "apagá el timer", "cancelá el timer", "apagá la alarma",
                                  "desactivá el temporizador", "pará el timer de la pasta"])
async def test_a_stop_while_a_timer_rings_silences_it_and_touches_no_other_timer(manager, text):
    """C02-TIM-0007: the ring stops (no repeat), the running timer is not cancelled, nothing to store."""
    pasta = manager.create_timer("pasta", 300)
    salsa = manager.create_timer("salsa", 600)
    await _ring(manager, pasta)
    rings = manager.tts.await_count
    reply = await say(TimerVoiceCommands(manager), text)
    assert reply.response == "Paré el temporizador de pasta."
    assert (reply.outcome.status, reply.outcome.persistence) == (TaskStatus.COMPLETED, PersistenceState.NOT_REQUIRED)
    assert reply.outcome.observation_scope == "software_state" and not reply.no_match
    assert (pasta.ring, pasta.state, salsa.state) == (RingDelivery.STOPPED, TimerState.COMPLETED, TimerState.RUNNING)
    assert manager.ringing_timers() == [] and manager.get_status()["delivery_tasks"] == 0
    await asyncio.sleep(0)
    assert manager.tts.await_count == rings


async def test_a_stop_naming_another_timer_cancels_that_one_and_the_ring_goes_on(manager):
    pasta = manager.create_timer("pasta", 300)
    salsa = manager.create_timer("salsa", 600)
    await _ring(manager, pasta)
    reply = await say(TimerVoiceCommands(manager), "pará el timer de la salsa")
    assert reply.response == "Cancelé el temporizador de salsa."
    assert salsa.state is TimerState.CANCELLED and manager.ringing_timers() == [pasta]


async def test_cancel_all_while_one_rings_silences_it_and_cancels_the_rest(manager):
    pasta = manager.create_timer("pasta", 300)
    salsa = manager.create_timer("salsa", 600)
    await _ring(manager, pasta)
    reply = await say(TimerVoiceCommands(manager), "cancelá todos los timers")
    assert reply.response == "Cancelé el temporizador de salsa."
    assert (pasta.ring, salsa.state) == (RingDelivery.STOPPED, TimerState.CANCELLED)
    assert manager.ringing_timers() == []


@pytest.mark.parametrize(("ended_ago", "silenced"), [(10.0, True), (RING_FOLLOW_UP_S + 1, False)])
async def test_the_stop_that_follows_a_barge_in_names_that_ring(manager, ended_ago, silenced):
    """The barge-in already cut the ring; the command heard right after must not cancel another timer."""
    pasta = manager.create_timer("pasta", 300)
    salsa = manager.create_timer("salsa", 600)
    pasta.state, pasta.ring, pasta.ring_ended_at = TimerState.COMPLETED, RingDelivery.STOPPED, manager.frozen.now
    manager.frozen.now += ended_ago
    reply = await say(TimerVoiceCommands(manager), "pará el timer")
    if silenced:
        assert reply.response == "Paré el temporizador de pasta." and salsa.state is TimerState.RUNNING
        assert reply.outcome.status is TaskStatus.COMPLETED
    else:
        assert reply.response == "Cancelé el temporizador de salsa." and salsa.state is TimerState.CANCELLED
    assert pasta.ring is RingDelivery.STOPPED


@pytest.mark.parametrize("receipt", [RingDelivery.DELIVERED, RingDelivery.RING_FAILED, RingDelivery.UNDELIVERED])
@pytest.mark.parametrize("text", ["cancelá el timer", "pará el timer"])
async def test_a_ring_that_ended_on_its_own_opens_no_follow_up(manager, receipt, text):
    """Review T030-U2-RV2: only a ring the household cut (STOPPED) is named by the next stop; after a ring
    that played out or failed, the stop cancels the running timer and never claims 'Paré'."""
    pasta = manager.create_timer("pasta", 300)
    salsa = manager.create_timer("salsa", 600)
    pasta.state, pasta.ring, pasta.ring_ended_at = TimerState.COMPLETED, receipt, manager.frozen.now
    manager.frozen.now += 10.0
    reply = await say(TimerVoiceCommands(manager), text)
    assert reply.response == "Cancelé el temporizador de salsa." and salsa.state is TimerState.CANCELLED
    assert (reply.outcome.status, reply.outcome.persistence) == (TaskStatus.COMPLETED, PersistenceState.VOLATILE)
    assert pasta.ring is receipt


async def test_a_stop_after_a_ring_that_played_out_with_nothing_running_is_not_a_false_stop(manager):
    """Review T030-U2-RV2: nothing sounds and nothing runs, so nothing is claimed."""
    pasta = manager.create_timer("pasta", 300)
    pasta.state, pasta.ring, pasta.ring_ended_at = TimerState.COMPLETED, RingDelivery.DELIVERED, manager.frozen.now
    manager.frozen.now += 5.0
    reply = await say(TimerVoiceCommands(manager), "pará el timer")
    assert (reply.response, reply.outcome.status, reply.no_match) == (NO_ACTIVE, TaskStatus.NEEDS_INPUT, True)


async def test_a_ring_nobody_heard_is_said_once_in_the_next_query(manager):
    """A failed ring is never a silent completion (T030-U2)."""
    commands = TimerVoiceCommands(manager)
    pasta = manager.create_timer("pasta", 300)
    pasta.state, pasta.ring = TimerState.COMPLETED, RingDelivery.UNDELIVERED
    heard = manager.create_timer("arroz", 300)
    heard.state, heard.ring = TimerState.COMPLETED, RingDelivery.DELIVERED
    manager.create_timer("", 90)
    first = await say(commands, "qué timers hay")
    assert first.response == ("El temporizador de pasta terminó, pero no pude avisarlo. "
                              "Al temporizador de 1 minuto y 30 segundos, faltan 1 minuto y 30 segundos.")
    assert first.outcome.status is TaskStatus.COMPLETED
    second = await say(commands, "qué timers hay")
    assert second.response == "Al temporizador de 1 minuto y 30 segundos, faltan 1 minuto y 30 segundos."
    partial = manager.create_timer("té", 300)
    partial.state, partial.ring = TimerState.COMPLETED, RingDelivery.RING_FAILED
    manager.cancel_timer(manager.get_all_timers()[2].timer_id)
    only = await say(commands, "¿cuánto le falta al timer?")
    assert (only.response, only.no_match) == (f"El temporizador de té terminó, pero no pude avisarlo. {NO_ACTIVE}",
                                              False)
