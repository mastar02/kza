"""T030-U1: NamedTimerManager surface used by the orchestrated timer commands."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.timers.named_timers import NamedTimerManager, TimerState


async def test_running_requires_start_and_a_live_update_task():
    timers = NamedTimerManager()
    assert timers.running is False
    await timers.start()
    assert timers.running is True and timers.get_status()["running"] is True
    timers._update_task.cancel()
    await asyncio.gather(timers._update_task, return_exceptions=True)
    assert timers.running is False and timers.get_status()["running"] is False
    await timers.stop()
    assert timers.running is False


async def test_stop_makes_it_not_running():
    timers = NamedTimerManager()
    await timers.start()
    await timers.stop()
    assert timers.running is False


def test_status_says_timers_are_volatile():
    status = NamedTimerManager().get_status()
    assert status["persistence"] == "volatile"
    assert (status["running"], status["active"], status["paused"]) == (False, 0, 0)


def test_transitions_change_only_the_expected_states():
    timers = NamedTimerManager()
    timer = timers.create_timer("pasta", 60)
    assert timers.resume_timer(timer.timer_id) is False and timer.state is TimerState.RUNNING
    assert timers.pause_timer(timer.timer_id) is True and timer.state is TimerState.PAUSED
    assert timers.pause_timer(timer.timer_id) is False and timer.state is TimerState.PAUSED
    assert timers.resume_timer(timer.timer_id) is True and timer.state is TimerState.RUNNING
    assert timers.cancel_timer(timer.timer_id) is True and timer.state is TimerState.CANCELLED
    for change in (timers.cancel_timer, timers.pause_timer, timers.resume_timer):
        assert change(timer.timer_id) is False and timer.state is TimerState.CANCELLED
    paused = timers.create_timer("salsa", 60)
    timers.pause_timer(paused.timer_id)
    assert timers.cancel_timer(paused.timer_id) is True and paused.state is TimerState.CANCELLED
    done = timers.create_timer("arroz", 60)
    done.state = TimerState.COMPLETED
    assert timers.cancel_timer(done.timer_id) is False and done.state is TimerState.COMPLETED
    for change in (timers.cancel_timer, timers.pause_timer, timers.resume_timer):
        assert change("missing") is False


def test_unnamed_timer_never_matches_a_legacy_name_lookup():
    timers = NamedTimerManager()
    timers.create_timer("", 60)
    assert timers._find_timer_by_name("pasta") is None
    assert timers._find_timer_by_name("") is None
    named = timers.create_timer("pasta", 60)
    assert timers._find_timer_by_name("pasta") is named
    assert timers._find_timer_by_name("") is None


async def test_ring_and_warning_name_the_timer_by_label_or_duration(monkeypatch):
    spoken = AsyncMock()
    timers = NamedTimerManager(tts_callback=spoken)
    import src.timers.named_timers as module
    # Only this module's view of asyncio: the ring's 2 s spacing is skipped, the event loop is untouched.
    monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=AsyncMock()))
    await timers._announce_complete(timers.create_timer("", 300, zone_id="zone_hall"))
    await timers._announce_complete(timers.create_timer("pasta", 300))
    await timers._announce_warning(timers.create_timer("", 90), 30)
    await timers._announce_warning(SimpleNamespace(name="salsa", zone_id="default"), 60)
    await timers._announce_warning(timers.create_timer("", 300), 60)
    await timers._announce_warning(timers.create_timer("arroz", 7200), 3600)
    await timers._announce_warning(timers.create_timer("", 10), 5)
    await timers._announce_warning(timers.create_timer("té", 10), 1)
    # T030-U1 R4: the warning names the timer like the ring and agrees in number ("1 minuto", not "1 minutos").
    assert [call.args for call in spoken.await_args_list] == (
        [("¡Terminó el temporizador de 5 minutos!", "zone_hall")] * 3
        + [("¡Terminó el temporizador de pasta!", "default")] * 3
        + [("Al temporizador de 1 minuto y 30 segundos le quedan 30 segundos.", "default"),
           ("Al temporizador de salsa le queda 1 minuto.", "default"),
           ("Al temporizador de 5 minutos le queda 1 minuto.", "default"),
           ("Al temporizador de arroz le queda 1 hora.", "default"),
           ("Al temporizador de 10 segundos le quedan 5 segundos.", "default"),
           ("Al temporizador de té le queda 1 segundo.", "default")])


async def test_legacy_create_and_ring_logs_carry_no_label(caplog):
    timers = NamedTimerManager()
    with caplog.at_level(logging.DEBUG, logger="src.timers.named_timers"):
        result = timers.handle_voice_command("pon un timer pastilla 1 minuto")
        await timers._announce_warning(result["timer"], 30)
        await timers._announce_complete(result["timer"])
    assert result["timer"].name == "pastilla"
    assert [r for r in caplog.records if r.levelno >= logging.INFO and "pastilla" in r.getMessage()] == []
    assert len([r for r in caplog.records if r.levelno >= logging.INFO]) == 3
