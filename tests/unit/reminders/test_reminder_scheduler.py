"""Tests for ReminderScheduler — asyncio scheduling and delivery."""
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from unittest.mock import AsyncMock, MagicMock

from src.core.voice_delivery import (
    DeliveryState, GenerationState, ResponseResult, ResponseStage, VoiceDeliveryError,
)
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_scheduler import DEFAULT_LATE_ACTION_GRACE_S, ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.home_assistant.home_actions import HomeActionService
from tests.mocks.home_control import selected_actions

ROOT = Path(__file__).resolve().parents[3]
ACTION = {"domain": "light", "service": "turn_on", "entity_id": "light.patio"}
DAY = 86400


class FakeClock:
    """Deterministic wall clock injected into the scheduler."""

    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def failed(spoken_segments: int = 0, delivery: DeliveryState = DeliveryState.FAILED,
           stage: ResponseStage | None = None) -> ResponseResult:
    return ResponseResult("x", GenerationState.COMPLETE, delivery, spoken_segments=spoken_segments,
                          timeout_stage=stage)


def request_id_for(reminder, index: int = 0) -> str:
    identity = json.dumps([reminder.id, reminder.trigger_at, index], separators=(",", ":"), allow_nan=False)
    return "reminder:" + hashlib.sha256(identity.encode()).hexdigest()


async def receipts(store: ReminderStore) -> list[tuple]:
    return await store._work.read(lambda db: [tuple(row) for row in db.execute(
        "SELECT status, reason FROM reminder_home_actions ORDER BY rowid").fetchall()])


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest.fixture
def mock_deps():
    return {
        "tts": AsyncMock(),
        "presence_detector": MagicMock(),
        "ha_client": AsyncMock(),
    }

@pytest.fixture
def clock():
    return FakeClock(time.time())

@pytest.fixture
def build(store, mock_deps, clock):
    built = []

    def make(**overrides):
        config = {
            "max_retries": 3,
            "retry_interval_seconds": 1,
            "missed_reminder_on_arrival": True,
            "tts_prefix": "Oye, recuerda",
            "output_zone": "living",
            **overrides,
        }
        s = ReminderScheduler(
            store=store,
            tts=mock_deps["tts"],
            presence_detector=mock_deps["presence_detector"],
            ha_client=mock_deps["ha_client"],
            config=config,
            home_actions=selected_actions(mock_deps["ha_client"], ("light.patio",)),
            clock=clock,
        )
        built.append(s)
        return s

    return make

@pytest_asyncio.fixture
async def scheduler(build):
    s = build()
    yield s
    if s._running:
        await s.stop()

@pytest.mark.asyncio
async def test_fire_reminder_with_tts(scheduler, store, mock_deps, clock):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="cocina")
    r = await store.create(user_id="u1", text="basura", trigger_at=clock() - 10, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    mock_deps["tts"].aspeak.assert_awaited_once_with("Oye, recuerda basura", zone_id="living")
    mock_deps["presence_detector"].get_user_zone.assert_not_called()
    updated = await store.get_by_id(r.id)
    assert updated.state == "fired"

@pytest.mark.asyncio
async def test_fire_reminder_with_ha_actions(scheduler, store, mock_deps, clock):
    r = await store.create(user_id="u1", text="regar", trigger_at=clock() - 10, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    await scheduler._fire_reminder(r)
    await scheduler._fire_reminder(r)
    await scheduler.missed_reminder_on_arrival("u1", "living")
    mock_deps["ha_client"].call_service_once.assert_called_once_with("light", "turn_on", "light.patio", None)
    assert await receipts(store) == [("confirmed", None)]
    mock_deps["tts"].aspeak.assert_awaited_once()

@pytest.mark.asyncio
async def test_fire_recurring_reschedules(scheduler, store, mock_deps, clock):
    r = await store.create(user_id="u1", text="pastilla", trigger_at=clock() - 10, recurrence="daily", scope=ReminderScope.SHARED)
    await scheduler._fire_reminder(r)
    updated = await store.get_by_id(r.id)
    assert updated.state == "active"
    assert updated.trigger_at == next_trigger(r.trigger_at, "daily") > clock()

@pytest.mark.asyncio
async def test_no_output_zone_never_fires_or_acts_and_stops_after_max_retries(build, store, mock_deps, clock):
    """CORE-06: a person's presence zone is never used and nothing closes silently."""
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    scheduler = build(output_zone=None)
    r = await store.create(user_id="u1", text="x", trigger_at=clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    waits = []
    for attempt in range(1, 5):
        assert await scheduler._fire_reminder(r) is False
        assert scheduler._retry_counts == {r.id: attempt}
        hold = scheduler._holds[r.id]
        if attempt <= 3:
            waits.append(hold.retry_at - clock())
            # Still backing off: an early call is not an attempt.
            clock.now = hold.retry_at - .5
            assert await scheduler._fire_reminder(r) is False
            assert scheduler._retry_counts == {r.id: attempt}
            clock.now = hold.retry_at
        else:
            assert hold.retry_at is None
    assert waits == [1, 2, 4]
    clock.advance(10)
    for _ in range(3):
        assert await scheduler._fire_reminder(r) is False
        assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
    assert scheduler._retry_counts == {r.id: 4}
    mock_deps["tts"].aspeak.assert_not_awaited()
    mock_deps["presence_detector"].get_user_zone.assert_not_called()
    mock_deps["ha_client"].call_service_once.assert_not_called()
    current = await store.get_by_id(r.id)
    assert current.state is ReminderState.ACTIVE and current.trigger_at == r.trigger_at and current.fire_count == 0
    assert await receipts(store) == []

@pytest.mark.asyncio
async def test_restart_with_three_day_old_reminders_gives_one_notice_each_and_no_actions(tmp_path, mock_deps):
    path = str(tmp_path / "restart.db")
    first = ReminderStore(path)
    await first.initialize()
    now = time.time()
    one_shot = await first.create("u1", "sacar la basura", now - 3 * DAY, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    daily = await first.create("u2", "regar.", now - 3 * DAY - 600, recurrence="daily", ha_actions=[ACTION],
                               scope=ReminderScope.SHARED)
    await first.close()
    store = ReminderStore(path)
    await store.initialize()
    try:
        ha = mock_deps["ha_client"]
        scheduler = ReminderScheduler(store, mock_deps["tts"], None, ha, {"output_zone": "living"},
                                      home_actions=selected_actions(ha, ("light.patio",)))
        reads = AsyncMock(wraps=store.get_due)
        store.get_due = reads
        task = asyncio.create_task(scheduler.start())
        try:
            async with asyncio.timeout(3):
                while ((await store.get_by_id(one_shot.id)).state is not ReminderState.FIRED
                       or (await store.get_by_id(daily.id)).trigger_at < time.time()):
                    await asyncio.sleep(.01)
            await asyncio.sleep(.1)
            count = reads.await_count
            await asyncio.sleep(.1)
            assert reads.await_count == count
        finally:
            await scheduler.stop()
            await asyncio.gather(task, return_exceptions=True)
        texts = [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list]
        assert sorted(texts) == [
            "Oye, recuerda regar. Este aviso llega con atraso y no ejecuté sus acciones programadas.",
            "Oye, recuerda sacar la basura. Este aviso llega con atraso y no ejecuté sus acciones programadas.",
        ]
        ha.call_service_once.assert_not_called()
        assert await receipts(store) == [("failed", "late_not_executed")] * 2
        advanced = await store.get_by_id(daily.id)
        latest = daily.trigger_at
        while next_trigger(latest, "daily") <= time.time():
            latest = next_trigger(latest, "daily")
        assert advanced.state is ReminderState.ACTIVE and advanced.trigger_at == next_trigger(latest, "daily")
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_late_notice_without_actions_does_not_mention_actions(scheduler, store, mock_deps, clock):
    r = await store.create("u1", "llamar a la abuela", clock() - DEFAULT_LATE_ACTION_GRACE_S - 1, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    mock_deps["tts"].aspeak.assert_awaited_once_with(
        "Oye, recuerda llamar a la abuela. Este aviso llega con atraso.", zone_id="living")
    assert await receipts(store) == []

@pytest.mark.parametrize("age, executed", [(DEFAULT_LATE_ACTION_GRACE_S, True), (DEFAULT_LATE_ACTION_GRACE_S + .01, False)])
@pytest.mark.asyncio
async def test_grace_boundary_and_configured_value(build, store, mock_deps, clock, age, executed):
    scheduler = build()
    r = await store.create("u1", "regar", clock() - age, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    await scheduler._fire_reminder(r)
    assert mock_deps["ha_client"].call_service_once.await_count == int(executed)
    assert await receipts(store) == ([("confirmed", None)] if executed else [("failed", "late_not_executed")])
    other = build(late_action_grace_s=3600)  # fresh HA mock
    r2 = await store.create("u1", "regar", clock() - 3000, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    await other._fire_reminder(r2)
    mock_deps["ha_client"].call_service_once.assert_awaited_once()

@pytest.mark.asyncio
async def test_occurrence_that_becomes_late_while_speaking_does_not_execute(scheduler, store, mock_deps, clock):
    async def slow(text, **kwargs):
        clock.advance(DEFAULT_LATE_ACTION_GRACE_S)

    mock_deps["tts"].aspeak.side_effect = slow
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    assert "atraso" not in mock_deps["tts"].aspeak.await_args.args[0]
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == [("failed", "late_not_executed")]
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED

@pytest.mark.asyncio
async def test_sequence_that_becomes_late_between_actions_stops_executing(scheduler, store, mock_deps, clock):
    from tests.mocks.home_control import ACCEPTED

    async def slow_send(*args):
        clock.advance(DEFAULT_LATE_ACTION_GRACE_S)
        return ACCEPTED

    mock_deps["ha_client"].call_service_once.side_effect = slow_send
    r = await store.create("u1", "escena", clock() - 1, scope=ReminderScope.SHARED,
                           ha_actions=[ACTION, ACTION | {"service": "turn_off"}, ACTION | {"service": "toggle"}])
    assert await scheduler._fire_reminder(r) is True
    mock_deps["ha_client"].call_service_once.assert_awaited_once_with("light", "turn_on", "light.patio", None)
    # Every remaining action gets its durable barrier, not only the first one after lateness.
    assert await receipts(store) == [("confirmed", None)] + [("failed", "late_not_executed")] * 2

@pytest.mark.asyncio
async def test_recurring_catch_up_executes_only_the_latest_on_time_occurrence(scheduler, store, mock_deps, clock):
    start = clock() - 4 * DAY
    latest = start
    for _ in range(3):
        latest = next_trigger(latest, "daily")
    clock.now = latest + 30
    r = await store.create("u1", "persianas", start, recurrence="daily", ha_actions=[ACTION], scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    mock_deps["tts"].aspeak.assert_awaited_once_with("Oye, recuerda persianas", zone_id="living")
    mock_deps["ha_client"].call_service_once.assert_awaited_once()
    assert await receipts(store) == [("confirmed", None)]
    assert (await store.get_by_id(r.id)).trigger_at == next_trigger(latest, "daily")
    for stale in (r, await store.get_by_id(r.id)):
        assert await scheduler._fire_reminder(stale) is False
    assert mock_deps["tts"].aspeak.await_count == 1

@pytest.mark.asyncio
async def test_recurring_catch_up_respects_recurrence_end(scheduler, store, mock_deps, clock):
    start = clock() - 5 * DAY
    end = next_trigger(next_trigger(start, "daily"), "daily")
    r = await store.create("u1", "fin", start, recurrence="daily", recurrence_end=end + 1, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    assert mock_deps["tts"].aspeak.await_count == 1
    current = await store.get_by_id(r.id)
    assert current.state is ReminderState.FIRED and current.trigger_at == end

@pytest.mark.parametrize("result, attempts", [
    (failed(), 4),
    (failed(delivery=DeliveryState.BLOCKED), 4),
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.QUEUE), 4),
    (failed(delivery=DeliveryState.CANCELLED), 4),
    (VoiceDeliveryError("voice_output_zone_required"), 4),
    (failed(spoken_segments=1), 1),
    (failed(spoken_segments=2, delivery=DeliveryState.CANCELLED), 1),
    # Review F3: a delivery timeout reports no segments although audio may have played.
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.DELIVERY), 1),
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.FEEDBACK), 1),
    (failed(delivery=DeliveryState.TIMED_OUT), 1),
    (RuntimeError("playback device lost"), 1),
])
@pytest.mark.asyncio
async def test_failed_delivery_is_bounded_and_partial_delivery_is_never_repeated(scheduler, store, mock_deps, clock, result, attempts):
    if isinstance(result, Exception):
        mock_deps["tts"].aspeak.side_effect = result
    else:
        mock_deps["tts"].aspeak.return_value = result
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    for _ in range(10):
        assert await scheduler._fire_reminder(r) is False
        clock.advance(10)
    assert mock_deps["tts"].aspeak.await_count == attempts
    assert scheduler._holds[r.id].retry_at is None
    mock_deps["ha_client"].call_service_once.assert_not_called()
    current = await store.get_by_id(r.id)
    assert current.state is ReminderState.ACTIVE and current.fire_count == 0
    assert await receipts(store) == []

@pytest.mark.asyncio
async def test_late_barrier_is_durable_before_a_failed_announcement(build, store, mock_deps, clock):
    scheduler = build(output_zone=None)
    r = await store.create("u1", "regar", clock() - DAY, ha_actions=[ACTION, ACTION | {"service": "turn_off"}],
                           scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    assert await receipts(store) == [("failed", "late_not_executed")] * 2
    generous = build(late_action_grace_s=10 * DAY)
    assert await generous._fire_reminder(r) is True
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == [("failed", "late_not_executed")] * 2

@pytest.mark.parametrize("late", [False, True])
@pytest.mark.asyncio
async def test_uncertain_receipt_is_never_resent_or_overwritten(tmp_path, mock_deps, late):
    path = str(tmp_path / "uncertain.db")
    store = ReminderStore(path)
    await store.initialize()
    r = await store.create("u1", "regar", time.time() - (DAY if late else 1), ha_actions=[ACTION],
                           scope=ReminderScope.SHARED)
    try:
        # A send interrupted by a crash: claimed, never finished.
        assert await store.claim_home_action(request_id_for(r), ACTION)
    finally:
        await store.close()
    restarted = ReminderStore(path)
    await restarted.initialize()
    try:
        ha = mock_deps["ha_client"]
        scheduler = ReminderScheduler(restarted, mock_deps["tts"], None, ha, {"output_zone": "living"},
                                      home_actions=selected_actions(ha, ("light.patio",)))
        assert await scheduler._fire_reminder(await restarted.get_by_id(r.id)) is True
        ha.call_service_once.assert_not_called()
        assert await receipts(restarted) == [("uncertain", "interrupted_send")]
        assert (await restarted.get_home_action_receipt(request_id_for(r))).status == "uncertain"
        assert (await restarted.get_by_id(r.id)).state is ReminderState.FIRED
        mock_deps["tts"].aspeak.assert_awaited_once_with(
            "Oye, recuerda regar. Este aviso llega con atraso y no puedo confirmar si sus acciones "
            "programadas se ejecutaron." if late else "Oye, recuerda regar", zone_id="living")
    finally:
        await restarted.close()

LATE_PLAIN = "Oye, recuerda regar. Este aviso llega con atraso."
LATE_NOT_RUN = "Oye, recuerda regar. Este aviso llega con atraso y no ejecuté sus acciones programadas."
LATE_UNKNOWN = ("Oye, recuerda regar. Este aviso llega con atraso y no puedo confirmar si sus acciones "
                "programadas se ejecutaron.")

@pytest.mark.parametrize("existing, text, final", [
    ([("confirmed", None)], LATE_PLAIN, [("confirmed", None)]),
    ([("uncertain", "response_lost")], LATE_UNKNOWN, [("uncertain", "response_lost")]),
    ([("failed", "late_not_executed")], LATE_NOT_RUN, [("failed", "late_not_executed")]),
    ([("failed", "action_not_selected")], LATE_PLAIN, [("failed", "action_not_selected")]),
    ([("confirmed", None), None], LATE_PLAIN, [("confirmed", None), ("failed", "late_not_executed")]),
    ([("uncertain", "response_lost"), None], LATE_UNKNOWN,
     [("uncertain", "response_lost"), ("failed", "late_not_executed")]),
    ([("failed", "late_not_executed"), None], LATE_NOT_RUN, [("failed", "late_not_executed")] * 2),
    ([None, None], LATE_NOT_RUN, [("failed", "late_not_executed")] * 2),
])
@pytest.mark.asyncio
async def test_late_notice_states_only_what_the_receipts_prove(scheduler, store, mock_deps, clock, existing, text, final):
    """Review F1: the notice is built from the barrier's outcome, never from ha_actions alone."""
    actions = [ACTION, ACTION | {"service": "turn_off"}][:len(existing)]
    r = await store.create("u1", "regar", clock() - 600, ha_actions=actions, scope=ReminderScope.SHARED)
    for index, receipt in enumerate(existing):
        if receipt is not None:
            assert await store.claim_home_action(request_id_for(r, index), actions[index])
            await store.finish_home_action(request_id_for(r, index), *receipt)
    assert await scheduler._fire_reminder(r) is True
    mock_deps["tts"].aspeak.assert_awaited_once_with(text, zone_id="living")
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == final
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED

@pytest.mark.asyncio
async def test_confirmed_action_then_crash_then_late_restart_does_not_say_not_executed(tmp_path, mock_deps):
    """Review F1 reproduction: spoken on time, action confirmed, crash before advance, restart 10 min later."""
    path = str(tmp_path / "crash.db")
    clock = FakeClock(time.time())
    ha = mock_deps["ha_client"]
    store = ReminderStore(path)
    await store.initialize()
    try:
        r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
        scheduler = ReminderScheduler(store, mock_deps["tts"], None, ha, {"output_zone": "living"},
                                      home_actions=selected_actions(ha, ("light.patio",)), clock=clock)
        store.mark_fired = AsyncMock(side_effect=RuntimeError("crash before advance"))
        with pytest.raises(RuntimeError, match="crash before advance"):
            await scheduler._fire_reminder(r)
    finally:
        await store.close()
    clock.advance(600)
    restarted = ReminderStore(path)
    await restarted.initialize()
    try:
        scheduler = ReminderScheduler(restarted, mock_deps["tts"], None, ha, {"output_zone": "living"},
                                      home_actions=HomeActionService(ha, policy=scheduler.home_actions.policy),
                                      clock=clock)
        assert await scheduler._fire_reminder(await restarted.get_by_id(r.id)) is True
        assert [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list] == ["Oye, recuerda regar", LATE_PLAIN]
        ha.call_service_once.assert_awaited_once()
        assert await receipts(restarted) == [("confirmed", None)]
        assert (await restarted.get_by_id(r.id)).state is ReminderState.FIRED
    finally:
        await restarted.close()

@pytest.mark.asyncio
async def test_late_retry_after_failed_announcement_still_says_actions_were_not_executed(scheduler, store, mock_deps, clock):
    """The receipts written by an earlier barrier of the same occurrence prove non-execution."""
    mock_deps["tts"].aspeak.side_effect = [failed(), None]
    r = await store.create("u1", "regar", clock() - DAY, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    clock.advance(scheduler._holds[r.id].retry_at - clock())
    assert await scheduler._fire_reminder(r) is True
    assert [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list] == [LATE_NOT_RUN, LATE_NOT_RUN]
    assert await receipts(store) == [("failed", "late_not_executed")]
    assert scheduler._retry_counts == {}  # a confirmed retry clears the reported failure

@pytest.mark.asyncio
async def test_late_occurrence_cancelled_before_the_barrier_is_not_spoken(scheduler, store, mock_deps, clock):
    r = await store.create("u1", "regar", clock() - DAY, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    reads = 0
    original = store.get_by_id

    async def racing_cancel(identity, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:  # the barrier's re-read: a web cancel lands just before it
            assert await store.cancel(identity, shared_only=True)
        return await original(identity, **kwargs)

    store.get_by_id = racing_cancel
    assert await scheduler._fire_reminder(r) is False
    mock_deps["tts"].aspeak.assert_not_awaited()
    assert await receipts(store) == []
    store.get_by_id = original
    assert (await store.get_by_id(r.id)).state is ReminderState.CANCELLED

@pytest.mark.asyncio
async def test_arrival_racing_the_loop_speaks_once(store, mock_deps):
    entered, release = asyncio.Event(), asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await release.wait()

    mock_deps["tts"].aspeak.side_effect = speak
    presence = mock_deps["presence_detector"]
    ha = mock_deps["ha_client"]
    scheduler = ReminderScheduler(store, mock_deps["tts"], presence, ha, {"output_zone": "living"},
                                  home_actions=selected_actions(ha, ("light.patio",)))
    arrived = presence.on_user_arrived.call_args.args[0]
    r = await store.create("u1", "regar", time.time() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    task = asyncio.create_task(scheduler.start())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(3):
            arrived("u1", MagicMock(current_zone="cocina"))
        assert await asyncio.wait_for(scheduler.missed_reminder_on_arrival("u1", "cocina"), 2) == 0
        await asyncio.sleep(.05)
        release.set()
        async with asyncio.timeout(2):
            while (await store.get_by_id(r.id)).state is not ReminderState.FIRED or scheduler._arrival_tasks:
                await asyncio.sleep(.01)
    finally:
        release.set()
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
    mock_deps["tts"].aspeak.assert_awaited_once()
    assert mock_deps["tts"].aspeak.await_args.kwargs["zone_id"] == "living"
    ha.call_service_once.assert_awaited_once()

@pytest.mark.asyncio
async def test_loop_backs_off_after_failed_delivery_instead_of_hot_looping(store, mock_deps):
    mock_deps["tts"].aspeak.return_value = failed()
    ha = mock_deps["ha_client"]
    scheduler = ReminderScheduler(store, mock_deps["tts"], None, ha,
        {"output_zone": "living", "retry_interval_seconds": 60},
        home_actions=selected_actions(ha, ("light.patio",)))
    reads = AsyncMock(wraps=store.get_due)
    store.get_due = reads
    r = await store.create("u1", "regar", time.time() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    task = asyncio.create_task(scheduler.start())
    try:
        async with asyncio.timeout(2):
            while not mock_deps["tts"].aspeak.await_count:
                await asyncio.sleep(.01)
        await asyncio.sleep(.2)
        assert reads.await_count == 1
        for expected in range(2, 7):
            scheduler.notify_new_reminder()
            async with asyncio.timeout(2):
                while reads.await_count < expected:
                    await asyncio.sleep(.005)
        await asyncio.sleep(.1)
        assert reads.await_count == 6
        assert 59 < scheduler._holds[r.id].retry_at - time.time() <= 60
    finally:
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
    assert mock_deps["tts"].aspeak.await_count == 1
    ha.call_service_once.assert_not_called()
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE

@pytest.mark.asyncio
async def test_an_edited_occurrence_starts_its_own_backoff(scheduler, store, mock_deps, clock):
    """A parked occurrence's hold never counts or blocks a new occurrence of the same reminder."""
    mock_deps["tts"].aspeak.return_value = failed(spoken_segments=1)
    r = await store.create("u1", "regar", clock() - 5, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    assert scheduler._holds[r.id].retry_at is None
    assert await store.update_trigger(r.id, clock() - 1, shared_only=True)  # edited from the web
    edited = await store.get_by_id(r.id)
    mock_deps["tts"].aspeak.return_value = failed()
    assert await scheduler._fire_reminder(edited) is False
    hold = scheduler._holds[r.id]
    assert (hold.trigger_at, hold.failures, hold.retry_at) == (edited.trigger_at, 1, clock() + 1)
    assert await scheduler._fire_reminder(edited) is False  # still backing off
    assert mock_deps["tts"].aspeak.await_count == 2

@pytest.mark.parametrize("failing_step", ["home_action", "advance"])
@pytest.mark.asyncio
async def test_confirmed_speech_then_a_failing_step_is_never_spoken_again(scheduler, store, mock_deps, clock, failing_step):
    """Review F2 (1): once the announcement was confirmed, no later failure re-speaks it in this process."""
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    if failing_step == "home_action":
        store.claim_home_action = AsyncMock(side_effect=RuntimeError("database is locked"))
    else:
        store.mark_fired = AsyncMock(side_effect=RuntimeError("database is locked"))
    with pytest.raises(RuntimeError, match="database is locked"):
        await scheduler._fire_reminder(r)
    assert scheduler._holds[r.id].retry_at is None
    for _ in range(4):
        clock.advance(10)  # retries would still be on time (grace 120 s)
        assert await scheduler._fire_reminder(r) is False
        assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
        await scheduler._deliver_due(clock())
        assert await scheduler._next_timeout() is None
    mock_deps["tts"].aspeak.assert_awaited_once_with("Oye, recuerda regar", zone_id="living")
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE

@pytest.mark.asyncio
async def test_failed_arrival_delivery_is_retried_by_the_loop_without_an_external_wake(store, mock_deps):
    """Review F2 (2): the loop skipped the occurrence while an arrival task delivered it."""
    release, texts = asyncio.Event(), []

    async def speak(text, **kwargs):
        texts.append(text)
        if len(texts) == 1:
            await release.wait()
            return failed()
        return None

    mock_deps["tts"].aspeak.side_effect = speak
    presence, ha = mock_deps["presence_detector"], mock_deps["ha_client"]
    scheduler = ReminderScheduler(store, mock_deps["tts"], presence, ha,
                                  {"output_zone": "living", "retry_interval_seconds": .2},
                                  home_actions=selected_actions(ha, ("light.patio",)))
    arrived = presence.on_user_arrived.call_args.args[0]
    reads = AsyncMock(wraps=store.get_due)
    store.get_due = reads
    task = asyncio.create_task(scheduler.start())
    try:
        await asyncio.sleep(.05)  # nothing pending: the loop waits for a wake
        r = await store.create("u1", "regar", time.time() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
        arrived("u1", MagicMock(current_zone="cocina"))
        async with asyncio.timeout(2):
            while not texts:
                await asyncio.sleep(.01)
        scheduler.notify_new_reminder()
        async with asyncio.timeout(2):
            while not reads.await_count:
                await asyncio.sleep(.01)
        await asyncio.sleep(.05)
        assert len(texts) == 1  # the loop skipped the in-flight occurrence
        release.set()
        async with asyncio.timeout(3):
            while (await store.get_by_id(r.id)).state is not ReminderState.FIRED:
                await asyncio.sleep(.01)
    finally:
        release.set()
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
    assert texts == ["Oye, recuerda regar"] * 2
    ha.call_service_once.assert_awaited_once()

@pytest.mark.parametrize("failing, outcomes, expected", [
    ("get_next_pending", [RuntimeError("db")] * 9, [1, 2, 4, 8, 16, 32, 60, 60, 60]),
    ("get_due", [RuntimeError("db")] * 9, [1, 2, 4, 8, 16, 32, 60, 60, 60]),
    ("get_next_pending", [RuntimeError("db")] * 3 + ["future"] + [RuntimeError("db")] * 2, [1, 2, 4, 1, 2]),
])
@pytest.mark.asyncio
async def test_loop_errors_back_off_exponentially_up_to_sixty_seconds(store, mock_deps, monkeypatch, failing,
                                                                      outcomes, expected):
    """Review F2 (3): a failing store never makes a 1 s hot loop; a good pass resets the delay."""
    scheduler = ReminderScheduler(store, mock_deps["tts"], None, mock_deps["ha_client"], {"output_zone": "living"},
                                  home_actions=selected_actions(mock_deps["ha_client"], ("light.patio",)))
    if failing == "get_due":
        await store.create("u1", "regar", time.time() - 1, scope=ReminderScope.SHARED)
    future = MagicMock(trigger_at=0)

    async def answer(*args, **kwargs):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        future.trigger_at = time.time() + .01
        return future

    calls, delays, real_sleep = [], [], asyncio.sleep

    async def recorded_sleep(delay, result=None):
        if delay == 0:  # the store's own cooperative yields
            return await real_sleep(0, result)
        delays.append(delay)
        if len(delays) == len(expected):
            scheduler._running = False
        await real_sleep(0)

    setattr(store, failing, AsyncMock(side_effect=answer))
    monkeypatch.setattr(asyncio, "sleep", recorded_sleep)
    await asyncio.wait_for(scheduler.start(), 2)
    assert delays == expected
    mock_deps["tts"].aspeak.assert_not_awaited()

@pytest.mark.parametrize("race", ["cancel", "edit"])
@pytest.mark.asyncio
async def test_catch_up_that_loses_its_trigger_update_is_not_spoken(scheduler, store, mock_deps, clock, race):
    """Review F2 (4): a web cancel or edit that lands before the collapse wins; nothing is announced."""
    start = clock() - 4 * DAY
    latest = start
    for _ in range(3):
        latest = next_trigger(latest, "daily")
    clock.now = latest + 30
    r = await store.create("u1", "persianas", start, recurrence="daily", ha_actions=[ACTION], scope=ReminderScope.SHARED)
    original = store.update_trigger
    edited = clock() + DAY

    async def concurrent(identity, new_trigger, **kwargs):
        if race == "cancel":
            assert await store.cancel(identity, shared_only=True)
        else:
            assert await original(identity, edited, shared_only=True)
        return await original(identity, new_trigger, **kwargs)

    store.update_trigger = concurrent
    assert await scheduler._fire_reminder(r) is False
    mock_deps["tts"].aspeak.assert_not_awaited()
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == []
    current = await store.get_by_id(r.id)
    if race == "cancel":
        assert current.state is ReminderState.CANCELLED and current.trigger_at == start
    else:
        assert current.state is ReminderState.ACTIVE and current.trigger_at == edited

@pytest.mark.asyncio
async def test_one_failing_reminder_does_not_stop_the_rest_of_the_due_batch(scheduler, store, mock_deps, clock):
    """Review F2 (5): per-reminder isolation inside one loop pass."""
    broken = await store.create("u1", "primero", clock() - 20, scope=ReminderScope.SHARED)
    healthy = await store.create("u1", "segundo", clock() - 10, scope=ReminderScope.SHARED)
    original = store.get_by_id

    async def flaky(identity, **kwargs):
        if identity == broken.id:
            raise RuntimeError("row unreadable")
        return await original(identity, **kwargs)

    store.get_by_id = flaky
    await scheduler._deliver_due(clock())
    mock_deps["tts"].aspeak.assert_awaited_once_with("Oye, recuerda segundo", zone_id="living")
    store.get_by_id = original
    assert (await store.get_by_id(healthy.id)).state is ReminderState.FIRED
    assert (await store.get_by_id(broken.id)).state is ReminderState.ACTIVE
    assert scheduler._holds[broken.id].failures == 1 and scheduler._holds[broken.id].retry_at == clock() + 1

@pytest.mark.parametrize("key, value", [
    ("late_action_grace_s", -1), ("late_action_grace_s", math.nan), ("late_action_grace_s", "120"),
    ("late_action_grace_s", True), ("late_action_grace_s", None), ("retry_interval_seconds", 0),
    ("max_retries", -1), ("max_retries", 1.5),
])
def test_invalid_delivery_policy_configuration_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        ReminderScheduler(MagicMock(), AsyncMock(), None, AsyncMock(), {key: value}, home_actions=MagicMock())

NO_ZONE_WARNING = "No reminder output zone configured"
ALWAYS_LATE_WARNING = "every retry after a failed first announcement is late"

def scheduler_warnings(caplog, config: dict) -> list[str]:
    caplog.clear()
    with caplog.at_level("WARNING", logger="src.reminders.reminder_scheduler"):
        ReminderScheduler(MagicMock(), AsyncMock(), None, AsyncMock(), config, home_actions=MagicMock())
    return [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]

@pytest.mark.parametrize("config, expected", [
    ({"output_zone": None}, [NO_ZONE_WARNING]),
    ({}, [NO_ZONE_WARNING]),
    ({"output_zone": "living", "retry_interval_seconds": 300}, [ALWAYS_LATE_WARNING]),
    ({"output_zone": "living", "retry_interval_seconds": 120}, [ALWAYS_LATE_WARNING]),
    ({"output_zone": "living", "retry_interval_seconds": 119.5}, []),
    ({"output_zone": "living", "retry_interval_seconds": 300, "late_action_grace_s": 900}, []),
])
def test_construction_warns_about_unannounceable_or_always_late_policy(caplog, config, expected):
    """Review F4/F5: the consequences of the delivery settings are logged at startup, not discovered later."""
    messages = scheduler_warnings(caplog, config)
    assert len(messages) == len(expected)
    for message, fragment in zip(messages, expected):
        assert fragment in message

def test_committed_settings_announce_shared_reminders_in_the_living_zone_without_startup_warnings(caplog):
    """User decision 2026-11-16: until SCHEMA-V2 adds delivery_room, shared reminders are announced in the living.

    The committed first retry is shorter than the 120 s grace, so neither F4 nor F5 warns.
    """
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    reminders = dict(config["reminders"])
    assert reminders["output_zone"] == "zone_living"
    assert scheduler_warnings(caplog, reminders) == []

def test_committed_first_retry_falls_inside_the_late_action_grace():
    """Review F5: one failed first announcement must not by itself forfeit an on-time occurrence's actions."""
    reminders = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))["reminders"]
    assert 0 < reminders["retry_interval_seconds"] < reminders["late_action_grace_s"]
    assert reminders["max_retries"] >= 1

def test_settings_declare_the_recommended_late_action_grace():
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    assert config["reminders"]["late_action_grace_s"] == DEFAULT_LATE_ACTION_GRACE_S == 120

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler, store):
    await store.create(user_id="u1", text="test", trigger_at=time.time() + 10000)
    task = asyncio.create_task(scheduler.start())
    await asyncio.sleep(0.1)
    assert scheduler._running is True
    await scheduler.stop()
    assert scheduler._running is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
