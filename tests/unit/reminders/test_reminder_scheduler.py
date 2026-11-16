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

from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_scheduler import DEFAULT_LATE_ACTION_GRACE_S, ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
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


def failed(spoken_segments: int = 0, delivery: DeliveryState = DeliveryState.FAILED) -> ResponseResult:
    return ResponseResult("x", GenerationState.COMPLETE, delivery, spoken_segments=spoken_segments)


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
    r = await store.create("u1", "escena", clock() - 1, ha_actions=[ACTION, ACTION | {"service": "turn_off"}],
                           scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    mock_deps["ha_client"].call_service_once.assert_awaited_once_with("light", "turn_on", "light.patio", None)
    assert await receipts(store) == [("confirmed", None), ("failed", "late_not_executed")]

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
    (failed(delivery=DeliveryState.TIMED_OUT), 4),
    (failed(spoken_segments=1), 1),
    (failed(spoken_segments=2, delivery=DeliveryState.CANCELLED), 1),
])
@pytest.mark.asyncio
async def test_failed_delivery_is_bounded_and_partial_delivery_is_never_repeated(scheduler, store, mock_deps, clock, result, attempts):
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
    finally:
        await restarted.close()

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
        assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
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

@pytest.mark.parametrize("key, value", [
    ("late_action_grace_s", -1), ("late_action_grace_s", math.nan), ("late_action_grace_s", "120"),
    ("late_action_grace_s", True), ("late_action_grace_s", None), ("retry_interval_seconds", 0),
    ("max_retries", -1), ("max_retries", 1.5),
])
def test_invalid_delivery_policy_configuration_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        ReminderScheduler(MagicMock(), AsyncMock(), None, AsyncMock(), {key: value}, home_actions=MagicMock())

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
