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
from src.reminders.reminder_scheduler import (
    DEFAULT_LATE_ACTION_GRACE_S, DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS, DEFAULT_NOTICE_FALLBACK_ROUND_S,
    NoticeDismissal, ReminderScheduler, notice_request_id,
)
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.home_assistant.home_actions import HomeActionService
from tests.mocks.home_control import selected_actions
from tests.mocks.responses import delivered, voice_output

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
           stage: ResponseStage | None = None, started: bool = False) -> ResponseResult:
    return ResponseResult("x", GenerationState.COMPLETE, delivery, spoken_segments=spoken_segments,
                          timeout_stage=stage, output_started=started)


def request_id_for(reminder, index: int = 0) -> str:
    identity = json.dumps([reminder.id, reminder.trigger_at, index], separators=(",", ":"), allow_nan=False)
    return "reminder:" + hashlib.sha256(identity.encode()).hexdigest()


async def receipts(store: ReminderStore) -> list[tuple]:
    """Home-action receipts only ('reminder:' ids); notice attempts are read by notices()."""
    return await store._work.read(lambda db: [tuple(row) for row in db.execute(
        "SELECT status, reason FROM reminder_home_actions WHERE request_id LIKE 'reminder:%' "
        "ORDER BY rowid").fetchall()])


async def notices(store: ReminderStore) -> list[tuple]:
    """Voice-notice attempts as (status, reason, attempt), in claim order."""
    rows = await store._work.read(lambda db: [tuple(row) for row in db.execute(
        "SELECT status, reason, body FROM reminder_home_actions WHERE request_id LIKE 'notice:%' "
        "ORDER BY rowid").fetchall()])
    return [(status, reason, json.loads(body)["attempt"]) for status, reason, body in rows]


async def ledger(store: ReminderStore) -> list[tuple]:
    return await store._work.read(lambda db: [tuple(row) for row in db.execute(
        "SELECT request_id, body, status, reason FROM reminder_home_actions ORDER BY rowid").fetchall()])


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest.fixture
def mock_deps():
    return {
        "tts": voice_output(),
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
            "notice_max_attempts": 3,
            "notice_retry_min_gap_s": 1,
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
    assert await notices(store) == [("confirmed", None, 1)]

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

@pytest.mark.parametrize("missing", ["zone", "voice_output"])
@pytest.mark.asyncio
async def test_no_output_zone_never_claims_fires_or_acts_and_never_spins(build, store, mock_deps, clock, missing):
    """CORE-06: a person's presence zone is never used; no room or no output means no claim (C02-NTC-0005/0006)."""
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    if missing == "zone":
        scheduler = build(output_zone=None)
    else:
        mock_deps["tts"] = None
        scheduler = ReminderScheduler(store, None, mock_deps["presence_detector"], mock_deps["ha_client"],
                                      {"output_zone": "living"}, clock=clock,
                                      home_actions=selected_actions(mock_deps["ha_client"], ("light.patio",)))
    r = await store.create(user_id="u1", text="x", trigger_at=clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    for _ in range(4):
        assert await scheduler._fire_reminder(r) is False
        await scheduler._deliver_due(clock())
        assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
        assert await scheduler._next_timeout() is None  # nothing claimable: wait for a wake, no hot loop
        clock.advance(10)
    assert scheduler._holds == {}
    if missing == "zone":
        mock_deps["tts"].aspeak.assert_not_awaited()
    mock_deps["presence_detector"].get_user_zone.assert_not_called()
    mock_deps["ha_client"].call_service_once.assert_not_called()
    current = await store.get_by_id(r.id)
    assert current.state is ReminderState.ACTIVE and current.trigger_at == r.trigger_at and current.fire_count == 0
    assert await ledger(store) == []
    pending = await scheduler.notice_entries()
    assert [(e.reminder_id, e.status, e.attempts) for e in pending] == [(r.id, "pending", 0)]

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
        # Recovery: both occurrences are claimed first and announced in one summary for the room.
        texts = [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list]
        assert texts == [
            "Tienes 2 avisos pendientes. "
            "Oye, recuerda regar. Este aviso llega con atraso y no ejecuté sus acciones programadas. "
            "Oye, recuerda sacar la basura. Este aviso llega con atraso y no ejecuté sus acciones programadas.",
        ]
        ha.call_service_once.assert_not_called()
        assert await receipts(store) == [("failed", "late_not_executed")] * 2
        assert await notices(store) == [("confirmed", None, 1)] * 2
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
        return delivered(text)

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

@pytest.mark.parametrize("result, expected", [
    # complete -> confirmed: the only outcome that runs home actions
    (delivered(), [("confirmed", None, 1)]),
    # post-queue timeout: uncertain whatever spoken_segments says (review round 2 F1, C02-NTC-0009)
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.DELIVERY), [("uncertain", "delivery_partial", 1)]),
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.FEEDBACK), [("uncertain", "delivery_partial", 1)]),
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.GENERATION), [("uncertain", "delivery_partial", 1)]),
    (failed(delivery=DeliveryState.TIMED_OUT), [("uncertain", "delivery_partial", 1)]),
    # cancelled after output may have started (F2, C02-NTC-0010): even within the first sentence
    (failed(spoken_segments=2, delivery=DeliveryState.CANCELLED), [("uncertain", "delivery_cancelled", 1)]),
    (failed(delivery=DeliveryState.CANCELLED, started=True), [("uncertain", "delivery_cancelled", 1)]),
    # other non-success with output that may have started
    (failed(spoken_segments=1), [("uncertain", "delivery_partial", 1)]),
    (failed(started=True), [("uncertain", "delivery_partial", 1)]),
    (failed(delivery=DeliveryState.BLOCKED, started=True), [("uncertain", "delivery_partial", 1)]),
    # no output at all: failed, the only retryable outcome (3 attempts per round: gaps 1 s, 2 s)
    (failed(delivery=DeliveryState.CANCELLED), [("failed", "delivery_not_started:cancelled", k) for k in (1, 2, 3)]),
    (failed(), [("failed", "delivery_not_started:failed", k) for k in (1, 2, 3)]),
    (failed(delivery=DeliveryState.BLOCKED), [("failed", "delivery_not_started:blocked", k) for k in (1, 2, 3)]),
    (failed(delivery=DeliveryState.EMPTY), [("failed", "delivery_not_started:empty", k) for k in (1, 2, 3)]),
    (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.QUEUE),
     [("failed", "delivery_not_started:timed_out", k) for k in (1, 2, 3)]),
    # exception or no ResponseResult: unverified
    (VoiceDeliveryError("voice_output_zone_required"), [("uncertain", "delivery_unverified", 1)]),
    (RuntimeError("playback device lost"), [("uncertain", "delivery_unverified", 1)]),
    (None, [("uncertain", "delivery_unverified", 1)]),
    (True, [("uncertain", "delivery_unverified", 1)]),
    (ResponseResult("x", GenerationState.COMPLETE, DeliveryState.COMPLETE, error="late_error"),
     [("uncertain", "delivery_unverified", 1)]),
])
@pytest.mark.asyncio
async def test_delivery_mapping_row_by_row(scheduler, store, mock_deps, clock, result, expected):
    """notice_semantics.delivery_mapping, evaluated in order; possibly heard is never spoken again."""
    if isinstance(result, Exception):
        mock_deps["tts"].aspeak.side_effect = result
    else:
        mock_deps["tts"].aspeak.return_value = result
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    outcomes = []
    for _ in range(10):
        outcomes.append(await scheduler._fire_reminder(r))
        clock.advance(10)
    assert await notices(store) == expected
    assert mock_deps["tts"].aspeak.await_count == len(expected)
    final = expected[-1][0]
    current = await store.get_by_id(r.id)
    if final == "failed":
        # Running out of attempts is not terminal: still pending, no actions, no further try this round;
        # the next try is the fallback round 15 min after the last failed attempt (user answer Q31).
        hold, last_claim = scheduler._holds[r.id], json.loads((await ledger(store))[-1][1])["claimed_at"]
        assert (hold.failures, hold.fallback_rounds, hold.retry_at) == (3, 1, last_claim + 900)
        assert current.state is ReminderState.ACTIVE and current.fire_count == 0
        assert [e.status for e in await scheduler.notice_entries()] == ["pending"]
    else:
        assert current.state is ReminderState.FIRED and r.id not in scheduler._holds
        assert await scheduler.notice_entries() and (await scheduler.notice_entries())[0].status == final
    assert outcomes[0] is (final == "confirmed") and not any(outcomes[1:])
    if final == "confirmed":
        mock_deps["ha_client"].call_service_once.assert_awaited_once()
        assert await receipts(store) == [("confirmed", None)]
    else:
        # Never after a failed or uncertain notice: the action id is not even claimed.
        mock_deps["ha_client"].call_service_once.assert_not_called()
        assert await receipts(store) == []
    # Ledger values stay inside the pinned vocabulary (scripts/stage_runtime.py:252).
    assert {row[2] for row in await ledger(store)} <= {"sending", "confirmed", "failed", "uncertain"}

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
async def test_confirmed_action_then_crash_then_late_restart_advances_without_speaking(tmp_path, mock_deps):
    """Spoken on time, action confirmed, crash before advance, restart 10 min later (C02-NTC-0008).

    T034-U1 announced it again as a late notice; the confirmed notice receipt now
    completes the occurrence silently and the action is never resent.
    """
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
        assert await scheduler._fire_reminder(await restarted.get_by_id(r.id)) is False
        assert [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list] == ["Oye, recuerda regar"]
        ha.call_service_once.assert_awaited_once()
        assert await receipts(restarted) == [("confirmed", None)]
        assert await notices(restarted) == [("confirmed", None, 1)]
        assert (await restarted.get_by_id(r.id)).state is ReminderState.FIRED
    finally:
        await restarted.close()

@pytest.mark.asyncio
async def test_late_retry_after_failed_announcement_still_says_actions_were_not_executed(scheduler, store, mock_deps, clock):
    """The receipts written by an earlier barrier of the same occurrence prove non-execution."""
    mock_deps["tts"].aspeak.side_effect = [failed(), delivered()]
    r = await store.create("u1", "regar", clock() - DAY, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    clock.advance(scheduler._holds[r.id].retry_at - clock())
    assert await scheduler._fire_reminder(r) is True
    assert [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list] == [LATE_NOT_RUN, LATE_NOT_RUN]
    assert await receipts(store) == [("failed", "late_not_executed")]
    assert await notices(store) == [("failed", "delivery_not_started:failed", 1), ("confirmed", None, 2)]
    assert scheduler._holds == {}  # a confirmed retry clears the in-process backoff

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
    assert await ledger(store) == []
    store.get_by_id = original
    assert (await store.get_by_id(r.id)).state is ReminderState.CANCELLED

@pytest.mark.asyncio
async def test_arrival_racing_the_loop_speaks_once(store, mock_deps):
    """C02-NTC-0004: arrivals only wake the loop, which is the only announcer."""
    entered, release = asyncio.Event(), asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await release.wait()
        return delivered(text)

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
            while (await store.get_by_id(r.id)).state is not ReminderState.FIRED:
                await asyncio.sleep(.01)
    finally:
        release.set()
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
    mock_deps["tts"].aspeak.assert_awaited_once()
    assert mock_deps["tts"].aspeak.await_args.kwargs["zone_id"] == "living"
    ha.call_service_once.assert_awaited_once()
    assert await notices(store) == [("confirmed", None, 1)]

@pytest.mark.asyncio
async def test_loop_backs_off_after_failed_delivery_instead_of_hot_looping(store, mock_deps):
    mock_deps["tts"].aspeak.return_value = failed()
    ha = mock_deps["ha_client"]
    scheduler = ReminderScheduler(store, mock_deps["tts"], None, ha,
        {"output_zone": "living", "notice_retry_min_gap_s": 60},
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
    """An exhausted occurrence's hold never counts or blocks a new occurrence of the same reminder."""
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 5, scope=ReminderScope.SHARED)
    for _ in range(3):
        assert await scheduler._fire_reminder(r) is False
        clock.advance(5)
    exhausted = scheduler._holds[r.id]
    assert (exhausted.failures, exhausted.retry_at) == (3, clock() - 5 + 900)  # waiting for its fallback round
    assert await store.update_trigger(r.id, clock() - 1, shared_only=True)  # edited from the web
    edited = await store.get_by_id(r.id)
    assert await scheduler._fire_reminder(edited) is False
    hold = scheduler._holds[r.id]
    assert (hold.trigger_at, hold.failures, hold.retry_at) == (edited.trigger_at, 1, clock() + 1)
    assert await scheduler._fire_reminder(edited) is False  # still backing off
    assert mock_deps["tts"].aspeak.await_count == 4
    assert [attempt for *_, attempt in await notices(store)] == [1, 2, 3, 1]  # a new occurrence counts from 1

@pytest.mark.parametrize("failing_step", ["home_action", "advance"])
@pytest.mark.asyncio
async def test_confirmed_speech_then_a_failing_step_is_never_spoken_again(scheduler, store, mock_deps, clock, failing_step):
    """Review F2 (1): once the announcement was confirmed, no later failure re-speaks it in this process.

    Review round 1 R2: the terminal receipt only gets its silent advance retried (backoff 1 s, 2 s,
    then on any later wake), without a restart and without resending the home action.
    """
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    mark_fired = store.mark_fired
    if failing_step == "home_action":
        claim = store.claim_home_action

        async def locked(request_id, body, **kwargs):
            if request_id.startswith("reminder:"):
                raise RuntimeError("database is locked")
            return await claim(request_id, body, **kwargs)

        store.claim_home_action = locked
    else:
        store.mark_fired = AsyncMock(side_effect=RuntimeError("database is locked"))
    with pytest.raises(RuntimeError, match="database is locked"):
        await scheduler._fire_reminder(r)
    assert scheduler._holds[r.id].retry_at == clock() + 1
    assert await scheduler._next_timeout() == 1  # the silent retry wakes the loop
    assert await scheduler._fire_reminder(r) is False  # inside the gap: nothing at all
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE
    if failing_step == "advance":
        for gap, next_gap in ((1, 2), (2, None)):
            clock.advance(gap)
            assert await scheduler._fire_reminder(r) is False  # the retried advance fails again
            assert scheduler._holds[r.id].retry_at == (None if next_gap is None else clock() + next_gap)
            assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
        assert scheduler._holds[r.id].retry_at is None  # bounded: no timed retry any more, never spins
        assert await scheduler._next_timeout() is None
        assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE
        store.mark_fired = mark_fired
        clock.advance(10)
        await scheduler._deliver_due(clock())  # any later wake completes it
    else:
        clock.advance(1)
        await scheduler._deliver_due(clock())
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED
    assert r.id not in scheduler._holds
    for _ in range(3):
        clock.advance(10)
        assert await scheduler._fire_reminder(r) is False
        await scheduler._deliver_due(clock())
        assert await scheduler._next_timeout() is None
    mock_deps["tts"].aspeak.assert_awaited_once_with("Oye, recuerda regar", zone_id="living")
    assert await notices(store) == [("confirmed", None, 1)]
    # The action ran once before the advance failed, or never when its claim failed; never resent.
    assert mock_deps["ha_client"].call_service_once.await_count == int(failing_step == "advance")
    assert await receipts(store) == ([("confirmed", None)] if failing_step == "advance" else [])

@pytest.mark.asyncio
async def test_arrival_only_wakes_the_loop_which_retries_a_notice_that_never_started(store, mock_deps):
    """single_flight: an arrival never announces; the loop retries a not-started notice after the gap."""
    texts = []

    async def speak(text, **kwargs):
        texts.append(text)
        return failed() if len(texts) == 1 else delivered(text)

    mock_deps["tts"].aspeak.side_effect = speak
    presence, ha = mock_deps["presence_detector"], mock_deps["ha_client"]
    scheduler = ReminderScheduler(store, mock_deps["tts"], presence, ha,
                                  {"output_zone": "living", "notice_retry_min_gap_s": .2},
                                  home_actions=selected_actions(ha, ("light.patio",)))
    arrived = presence.on_user_arrived.call_args.args[0]
    task = asyncio.create_task(scheduler.start())
    try:
        await asyncio.sleep(.05)  # nothing pending: the loop waits for a wake
        r = await store.create("u1", "regar", time.time() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
        assert await scheduler.missed_reminder_on_arrival("u1", "cocina") == 0
        assert texts == []  # the arrival itself announced nothing
        arrived("u1", MagicMock(current_zone="cocina"))
        async with asyncio.timeout(3):
            while (await store.get_by_id(r.id)).state is not ReminderState.FIRED:
                await asyncio.sleep(.01)
    finally:
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
    assert texts == ["Oye, recuerda regar"] * 2
    assert all(call.kwargs["zone_id"] == "living" for call in mock_deps["tts"].aspeak.await_args_list)
    ha.call_service_once.assert_awaited_once()
    assert await notices(store) == [("failed", "delivery_not_started:failed", 1), ("confirmed", None, 2)]

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
    ("late_action_grace_s", True), ("late_action_grace_s", None), ("notice_retry_min_gap_s", 0),
    ("notice_retry_min_gap_s", math.inf), ("notice_max_attempts", 0), ("notice_max_attempts", 1.5),
    ("notice_max_attempts", True), ("notice_retention_hours", -1), ("notice_retention_hours", "48"),
    ("notice_fallback_round_s", 0), ("notice_fallback_round_s", math.inf), ("notice_fallback_round_s", "900"),
    ("notice_fallback_max_rounds", -1), ("notice_fallback_max_rounds", 1.5), ("notice_fallback_max_rounds", True),
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
    ({"output_zone": "living", "notice_retry_min_gap_s": 300}, [ALWAYS_LATE_WARNING]),
    ({"output_zone": "living", "notice_retry_min_gap_s": 120}, [ALWAYS_LATE_WARNING]),
    ({"output_zone": "living", "notice_retry_min_gap_s": 119.5}, []),
    ({"output_zone": "living", "notice_retry_min_gap_s": 300, "late_action_grace_s": 900}, []),
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
    assert 0 < reminders["notice_retry_min_gap_s"] < reminders["late_action_grace_s"]
    assert reminders["notice_max_attempts"] >= 2
    assert "max_retries" not in reminders and "retry_interval_seconds" not in reminders


def test_settings_declare_the_notice_policy_of_the_design():
    """P3-F0 notice_semantics: 3 attempts per round, 30 s first gap, 48 h retention (user answer Q3)."""
    reminders = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))["reminders"]
    assert (reminders["notice_max_attempts"], reminders["notice_retry_min_gap_s"],
            reminders["notice_retention_hours"]) == (3, 30, 48)
    # User answer Q31: a fallback round every 15 min, at most 4, when no output-free signal comes.
    assert (reminders["notice_fallback_round_s"], reminders["notice_fallback_max_rounds"]) == (
        DEFAULT_NOTICE_FALLBACK_ROUND_S, DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS) == (900, 4)

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


# ---------------------------------------------------------------------------
# T034-U2: durable notice receipts (P3-F0/notices/v1)
# ---------------------------------------------------------------------------

HOUR = 3600


async def reopen(path: str) -> ReminderStore:
    store = ReminderStore(path)
    await store.initialize()
    return store


def scheduler_on(store, tts, ha, clock, **config) -> ReminderScheduler:
    return ReminderScheduler(store, tts, None, ha, {"output_zone": "living", "notice_retry_min_gap_s": 1, **config},
                             home_actions=selected_actions(ha, ("light.patio",)), clock=clock)


@pytest.mark.asyncio
async def test_notice_request_id_and_body_follow_the_contract(scheduler, store, mock_deps, clock):
    """request_id = 'notice:'+sha256([id, trigger_at, attempt]); canonical body without the reminder text."""
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "texto privado del aviso", clock() - 1, scope=ReminderScope.SHARED)
    claimed = []
    for _ in range(2):
        claimed.append(clock())
        await scheduler._fire_reminder(r)
        clock.advance(5)
    rows = await ledger(store)
    assert len(rows) == 2
    for attempt, ((request_id, body, status, reason), at) in enumerate(zip(rows, claimed), start=1):
        identity = json.dumps([r.id, r.trigger_at, attempt], separators=(",", ":"), allow_nan=False)
        assert request_id == "notice:" + hashlib.sha256(identity.encode()).hexdigest()
        assert request_id == notice_request_id(r.id, r.trigger_at, attempt)
        expected = {"kind": "voice_notice", "reminder_id": r.id, "trigger_at": r.trigger_at, "attempt": attempt,
                    "claimed_at": at, "room": "living"}
        assert body == json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
        assert "texto privado" not in body
        assert (status, reason) == ("failed", "delivery_not_started:failed")


@pytest.mark.asyncio
async def test_restart_while_announcing_is_uncertain_and_never_spoken(tmp_path, mock_deps, clock):
    """C02-NTC-0003: a crash while announcing leaves 'sending'; recovery makes it uncertain, never re-spoken."""
    path = str(tmp_path / "crash.db")
    first, ha = await reopen(path), mock_deps["ha_client"]
    entered = asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await asyncio.Event().wait()  # the process dies while the notice plays

    tts = AsyncMock()
    tts.aspeak.side_effect = speak
    try:
        r = await first.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
        task = asyncio.create_task(scheduler_on(first, tts, ha, clock)._fire_reminder(r))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert await notices(first) == [("sending", None, 1)]
    finally:
        await first.close()
    for elapsed in (0, DEFAULT_LATE_ACTION_GRACE_S + 1):  # restart within the grace and after it
        clock.advance(elapsed)
        restarted = await reopen(path)
        try:
            after = voice_output()
            scheduler = scheduler_on(restarted, after, ha, clock)
            assert await scheduler._fire_reminder(r) is False
            await scheduler._deliver_due(clock())
            after.aspeak.assert_not_awaited()
            assert await notices(restarted) == [("uncertain", "interrupted_send", 1)]
            assert (await restarted.get_by_id(r.id)).state is ReminderState.FIRED
        finally:
            await restarted.close()
    tts.aspeak.assert_awaited_once()
    ha.call_service_once.assert_not_called()
    restarted = await reopen(path)
    try:
        assert await receipts(restarted) == []  # the action id was never claimed
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_two_recoveries_produce_one_announcement(store, mock_deps, clock):
    """Two passes (e.g. startup and an output-free round) over the same occurrence: one notice."""
    ha = mock_deps["ha_client"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await release.wait()
        return delivered(text)

    first_tts, second_tts = AsyncMock(), voice_output()
    first_tts.aspeak.side_effect = speak
    first, second = scheduler_on(store, first_tts, ha, clock), scheduler_on(store, second_tts, ha, clock)
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    task = asyncio.create_task(first._fire_reminder(r))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # Another owner sees attempt 1 'sending': it must not claim attempt 2 or speak.
        assert await second._fire_reminder(r) is False
        await second._deliver_due(clock())
        assert await second._next_timeout() is None
    finally:
        release.set()
    assert await asyncio.wait_for(task, 2) is True
    second.notify_output_available()
    assert await second._fire_reminder(r) is False
    await second._deliver_due(clock())
    first_tts.aspeak.assert_awaited_once()
    second_tts.aspeak.assert_not_awaited()
    ha.call_service_once.assert_awaited_once()
    assert await notices(store) == [("confirmed", None, 1)]


@pytest.mark.asyncio
async def test_a_body_conflict_means_another_claimant_and_nothing_is_spoken(scheduler, store, mock_deps, clock):
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    # Another process claimed attempt 1 with its own claimed_at just before this one.
    other = {"kind": "voice_notice", "reminder_id": r.id, "trigger_at": r.trigger_at, "attempt": 1,
             "claimed_at": clock() - .5, "room": "living"}
    original = store.claim_home_action

    async def racing(request_id, body, **kwargs):
        if request_id.startswith("notice:"):
            await original(request_id, other)
        return await original(request_id, body, **kwargs)

    store.claim_home_action = racing
    assert await scheduler._fire_reminder(r) is False
    mock_deps["tts"].aspeak.assert_not_awaited()
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert scheduler._holds[r.id].retry_at is None
    assert await notices(store) == [("sending", None, 1)]


@pytest.mark.parametrize("result, status, reason", [
    (delivered(), "confirmed", None),
    (failed(), "failed", "delivery_not_started:failed"),
    (failed(delivery=DeliveryState.CANCELLED, started=True), "uncertain", "delivery_cancelled"),
])
@pytest.mark.asyncio
async def test_recovery_claims_every_occurrence_then_speaks_one_summary(scheduler, store, mock_deps, clock,
                                                                        result, status, reason):
    """summary_per_room: claim all first, one announcement, every claim finished from the same result."""
    claims_before_speech = []

    async def speak(text, **kwargs):
        claims_before_speech.extend(await notices(store))
        return result

    mock_deps["tts"].aspeak.side_effect = speak
    created = [await store.create("u1", text, clock() - age, scope=ReminderScope.SHARED, ha_actions=[ACTION])
               for text, age in (("uno", 30), ("dos", 20), ("tres", 10))]
    await scheduler._deliver_due(clock())
    assert claims_before_speech == [("sending", None, 1)] * 3
    mock_deps["tts"].aspeak.assert_awaited_once()
    assert mock_deps["tts"].aspeak.await_args.args[0] == (
        "Tienes 3 avisos pendientes. Oye, recuerda uno. Oye, recuerda dos. Oye, recuerda tres.")
    assert await notices(store) == [(status, reason, 1)] * 3
    states = {(await store.get_by_id(r.id)).state for r in created}
    assert states == ({ReminderState.ACTIVE} if status == "failed" else {ReminderState.FIRED})
    assert mock_deps["ha_client"].call_service_once.await_count == (3 if status == "confirmed" else 0)


@pytest.mark.asyncio
async def test_crash_during_a_summary_leaves_every_occurrence_uncertain(tmp_path, mock_deps, clock):
    path = str(tmp_path / "summary.db")
    first, ha = await reopen(path), mock_deps["ha_client"]
    entered = asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    tts = AsyncMock()
    tts.aspeak.side_effect = speak
    try:
        for text in ("uno", "dos"):
            await first.create("u1", text, clock() - 5, scope=ReminderScope.SHARED)
        task = asyncio.create_task(scheduler_on(first, tts, ha, clock)._deliver_due(clock()))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    finally:
        await first.close()
    restarted = await reopen(path)
    try:
        after = voice_output()
        await scheduler_on(restarted, after, ha, clock)._deliver_due(clock())
        after.aspeak.assert_not_awaited()
        assert await notices(restarted) == [("uncertain", "interrupted_send", 1)] * 2
        assert await restarted.get_due(clock(), shared_only=True) == []
    finally:
        await restarted.close()


@pytest.mark.parametrize("retry_after, executed", [(1, True), (DEFAULT_LATE_ACTION_GRACE_S + 1, False)])
@pytest.mark.asyncio
async def test_a_notice_cancelled_before_output_stays_pending_and_actions_wait_for_a_confirmed_retry(
    scheduler, store, mock_deps, clock, retry_after, executed,
):
    """C02-NTC-0011 (unit level): failed attempt runs no action; the confirmed retry runs it only on time."""
    mock_deps["tts"].aspeak.side_effect = [failed(delivery=DeliveryState.CANCELLED), delivered()]
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == []
    [entry] = await scheduler.notice_entries()
    assert (entry.status, entry.attempts, entry.failed_attempts, entry.pending) == ("pending", 1, 1, True)
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE
    clock.advance(retry_after)
    assert await scheduler._fire_reminder(r) is True
    assert await notices(store) == [("failed", "delivery_not_started:cancelled", 1), ("confirmed", None, 2)]
    assert mock_deps["ha_client"].call_service_once.await_count == int(executed)
    assert await receipts(store) == [("confirmed", None) if executed else ("failed", "late_not_executed")]
    assert len(await ledger(store)) == 3  # ledger_claims: two notice attempts and one action receipt


POSSIBLY_HEARD = {
    "delivery_timeout": (failed(delivery=DeliveryState.TIMED_OUT, stage=ResponseStage.DELIVERY), "delivery_partial"),
    "living_barge_in_first_sentence": (failed(delivery=DeliveryState.CANCELLED, started=True), "delivery_cancelled"),
    "cancelled_after_a_sentence": (failed(spoken_segments=1, delivery=DeliveryState.CANCELLED), "delivery_cancelled"),
    "partial": (failed(spoken_segments=1), "delivery_partial"),
    "playback_failed_after_first_chunk": (failed(started=True), "delivery_partial"),
}


@pytest.mark.parametrize("advance_lost", [False, True])
@pytest.mark.parametrize("restart_after", [5, DEFAULT_LATE_ACTION_GRACE_S + 5])
@pytest.mark.parametrize("kind", sorted(POSSIBLY_HEARD))
@pytest.mark.asyncio
async def test_a_possibly_heard_notice_is_never_repeated_by_voice_after_restart(tmp_path, mock_deps, clock,
                                                                                kind, restart_after, advance_lost):
    """Closes T034-U1 interim limitation U1-L1 (user answer Q3); C02-NTC-0009/0010 for one action reminder.

    advance_lost: the process dies after the receipt but before the occurrence advanced;
    the terminal receipt alone keeps the restarted scheduler silent.
    """
    path = str(tmp_path / "heard.db")
    result, reason = POSSIBLY_HEARD[kind]
    ha = mock_deps["ha_client"]
    first, tts = await reopen(path), voice_output()
    tts.aspeak.return_value = result
    try:
        r = await first.create("u1", "regar", clock() - 1, ha_actions=[ACTION], recurrence="daily",
                               scope=ReminderScope.SHARED)
        scheduler = scheduler_on(first, tts, ha, clock)
        if advance_lost:
            first.update_trigger = AsyncMock(side_effect=RuntimeError("process killed"))
            with pytest.raises(RuntimeError, match="process killed"):
                await scheduler._fire_reminder(r)
        else:
            assert await scheduler._fire_reminder(r) is False
        clock.advance(2)
        assert await scheduler._fire_reminder(r) is False  # nothing more in-process
        await scheduler._deliver_due(clock())
    finally:
        await first.close()
    clock.advance(restart_after)
    restarted = await reopen(path)
    try:
        after = voice_output()
        scheduler = scheduler_on(restarted, after, ha, clock)
        # Review round 1 R5: before any pass completes a lost advance, the web lists the
        # occurrence once, as its terminal history entry, and never also as pending.
        [entry] = await scheduler.notice_entries()
        assert (entry.status, entry.reason, entry.pending) == ("uncertain", reason, False)
        await scheduler._deliver_due(clock())
        assert await scheduler._fire_reminder(r) is False
        after.aspeak.assert_not_awaited()
        tts.aspeak.assert_awaited_once()
        ha.call_service_once.assert_not_called()
        assert await notices(restarted) == [("uncertain", reason, 1)]
        assert await receipts(restarted) == []  # action_claims 0
        current = await restarted.get_by_id(r.id)
        assert current.state is ReminderState.ACTIVE and current.trigger_at == next_trigger(r.trigger_at, "daily")
        [entry] = await scheduler.notice_entries()
        assert (entry.status, entry.reason, entry.pending) == ("uncertain", reason, False)
    finally:
        await restarted.close()


@pytest.mark.parametrize("recurrence", [None, "daily"])
@pytest.mark.asyncio
async def test_dismiss_finishes_a_pending_notice_confirmed_and_advances_without_actions(
    scheduler, store, mock_deps, clock, recurrence,
):
    """web_projection: dismiss claims the next attempt, finishes it 'confirmed'/'dismissed', runs no action."""
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 10, ha_actions=[ACTION], recurrence=recurrence,
                           scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False
    clock.advance(1)
    assert await scheduler.dismiss_notice(r.id, r.trigger_at) is NoticeDismissal.DISMISSED
    assert await notices(store) == [("failed", "delivery_not_started:failed", 1), ("confirmed", "dismissed", 2)]
    rows = await ledger(store)
    assert json.loads(rows[1][1])["room"] is None
    current = await store.get_by_id(r.id)
    if recurrence:
        assert current.state is ReminderState.ACTIVE and current.trigger_at == next_trigger(r.trigger_at, "daily")
    else:
        assert current.state is ReminderState.FIRED
    assert r.id not in scheduler._holds
    for _ in range(3):
        clock.advance(5)
        assert await scheduler._fire_reminder(r) is False
    mock_deps["tts"].aspeak.assert_awaited_once()
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert await receipts(store) == []
    entries = await scheduler.notice_entries()
    assert [(e.status, e.reason, e.pending) for e in entries] == [("confirmed", "dismissed", False)]
    # Retention: purged 48 h after its claimed_at, not before.
    clock.advance(48 * HOUR - 16)
    await scheduler._cleanup_notices()
    assert len(await notices(store)) == 2
    clock.advance(2)
    await scheduler._cleanup_notices()
    assert await notices(store) == []
    assert await scheduler.dismiss_notice(r.id, r.trigger_at) is NoticeDismissal.NOT_FOUND


@pytest.mark.asyncio
async def test_dismiss_while_announcing_is_busy_and_a_finished_row_is_purged_early(scheduler, store, mock_deps, clock):
    entered, release = asyncio.Event(), asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await release.wait()
        return failed(spoken_segments=1)

    mock_deps["tts"].aspeak.side_effect = speak
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    task = asyncio.create_task(scheduler._fire_reminder(r))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert await scheduler.dismiss_notice(r.id, r.trigger_at) is NoticeDismissal.BUSY
        other = scheduler_on(store, voice_output(), mock_deps["ha_client"], clock)
        assert await other.dismiss_notice(r.id, r.trigger_at) is NoticeDismissal.BUSY  # attempt 1 is 'sending'
    finally:
        release.set()
    assert await task is False
    assert await notices(store) == [("uncertain", "delivery_partial", 1)]
    assert await scheduler.dismiss_notice(r.id, r.trigger_at) is NoticeDismissal.PURGED
    assert await notices(store) == []
    assert await scheduler.dismiss_notice("missing", 1.) is NoticeDismissal.NOT_FOUND


@pytest.mark.asyncio
async def test_retention_keeps_pending_and_sending_rows_and_never_touches_action_receipts(
    scheduler, store, mock_deps, clock,
):
    """retention: status-only selection, 48 h from claimed_at, only for occurrences no longer pending."""
    mock_deps["tts"].aspeak.side_effect = [delivered(), failed()]
    done = await store.create("u1", "hecho", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    pending = await store.create("u1", "pendiente", clock() - 1, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(done) is True
    assert await scheduler._fire_reminder(pending) is False
    cancelled = await store.create("u1", "cancelado", clock() - 1, scope=ReminderScope.SHARED)
    stuck = notice_request_id(cancelled.id, cancelled.trigger_at, 1)
    assert await store.claim_home_action(stuck, {"kind": "voice_notice", "reminder_id": cancelled.id,
                                                 "trigger_at": cancelled.trigger_at, "attempt": 1,
                                                 "claimed_at": clock(), "room": "living"})
    assert await store.cancel(cancelled.id, shared_only=True)
    clock.advance(48 * HOUR + 1)
    await scheduler._cleanup_notices()
    # Confirmed row of the fired occurrence is gone; the pending occurrence keeps its failed attempt;
    # a 'sending' row is never deleted; the action receipt is outside this retention.
    assert await notices(store) == [("failed", "delivery_not_started:failed", 1), ("sending", None, 1)]
    assert await receipts(store) == [("confirmed", None)]


@pytest.mark.asyncio
async def test_retention_keeps_every_attempt_of_an_occurrence_until_its_newest_claim_expires(
    scheduler, store, mock_deps, clock,
):
    """Review round 1 R6: rows of one occurrence are purged together, so history never has attempt gaps."""
    mock_deps["tts"].aspeak.side_effect = [failed(), delivered()]
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is False  # attempt 1 failed, claimed at T
    clock.advance(3 * HOUR)
    assert await scheduler._fire_reminder(r) is True  # attempt 2 confirmed, claimed at T + 3 h
    clock.advance(47 * HOUR)  # attempt 1 is 50 h old, attempt 2 is 47 h old
    await scheduler._cleanup_notices()
    assert await notices(store) == [("failed", "delivery_not_started:failed", 1), ("confirmed", None, 2)]
    [entry] = await scheduler.notice_entries()
    assert (entry.status, entry.attempts, entry.failed_attempts, entry.pending) == ("confirmed", 2, 1, False)
    clock.advance(HOUR + 1)
    await scheduler._cleanup_notices()
    assert await notices(store) == []
    assert await scheduler.notice_entries() == []


@pytest.mark.asyncio
async def test_history_shows_the_text_only_while_the_reminder_is_shared(scheduler, store, mock_deps, clock):
    """Review round 1 R8 (X05): the history entry reads the text in shared scope only."""
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    assert await scheduler._fire_reminder(r) is True
    [entry] = await scheduler.notice_entries()
    assert entry.text == "regar"
    # Scope is fixed at creation today; force a private row to pin the projection.
    await store._work.write(lambda db: db.execute("UPDATE reminders SET scope = 'private' WHERE id = ?",
                                                  (r.id,)).rowcount)
    [entry] = await scheduler.notice_entries()
    assert (entry.reminder_id, entry.status, entry.text) == (r.id, "confirmed", None)


@pytest.mark.asyncio
async def test_retention_runs_at_startup_and_every_six_hours(store, mock_deps, clock, monkeypatch):
    scheduler = scheduler_on(store, mock_deps["tts"], mock_deps["ha_client"], clock)
    calls = []
    purge = store.purge_notice_rows

    async def recorded(cutoff, **kwargs):
        calls.append((clock(), cutoff))
        return await purge(cutoff, **kwargs)

    store.purge_notice_rows = recorded
    task = asyncio.create_task(scheduler.start())
    try:
        async with asyncio.timeout(2):
            while not calls:
                await asyncio.sleep(.01)
        assert calls == [(clock(), clock() - 48 * HOUR)]
        clock.advance(6 * HOUR)
        scheduler.notify_new_reminder()
        async with asyncio.timeout(2):
            while len(calls) < 2:
                await asyncio.sleep(.01)
        clock.advance(HOUR)
        scheduler.notify_new_reminder()
        await asyncio.sleep(.1)
        assert len(calls) == 2
    finally:
        await scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_exhausted_round_is_not_terminal_and_a_new_round_continues_the_attempt_numbers(
    scheduler, store, mock_deps, clock,
):
    """attempts_and_backoff: 3 attempts, gaps 1 s then 2 s, no spin; a free output starts a new round (Q31)."""
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    await scheduler._deliver_due(clock())
    assert await scheduler._next_timeout() == 1
    clock.advance(.5)
    await scheduler._deliver_due(clock())  # inside the gap: no attempt
    assert mock_deps["tts"].aspeak.await_count == 1
    clock.advance(.5)
    await scheduler._deliver_due(clock())
    assert await scheduler._next_timeout() == 2
    clock.advance(2)
    await scheduler._deliver_due(clock())
    assert await scheduler._next_timeout() == 900  # exhausted: only the fallback round wakes it, never spin
    clock.advance(DEFAULT_LATE_ACTION_GRACE_S)
    await scheduler._deliver_due(clock())
    assert mock_deps["tts"].aspeak.await_count == 3
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE
    mock_deps["tts"].aspeak.return_value = delivered()
    scheduler.notify_output_available()
    await scheduler._deliver_due(clock())
    assert [(status, attempt) for status, _, attempt in await notices(store)] == [
        ("failed", 1), ("failed", 2), ("failed", 3), ("confirmed", 4)]
    # Late by now: the confirmed notice runs no action.
    mock_deps["ha_client"].call_service_once.assert_not_called()
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED


@pytest.mark.asyncio
async def test_dismiss_with_a_retired_session_writes_nothing(scheduler, store, clock):
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    with pytest.raises(PermissionError):
        await scheduler.dismiss_notice(r.id, r.trigger_at, guard=lambda: False)
    assert await ledger(store) == []
    assert (await store.get_by_id(r.id)).state is ReminderState.ACTIVE
    assert r.id not in scheduler._in_flight


# ---------------------------------------------------------------------------
# T034-U2 follow-up: user answers Q31 (new rounds) and Q32 (missed occurrences after a crash)
# ---------------------------------------------------------------------------

async def fail_one_round(scheduler, clock) -> None:
    """Three not-started attempts with the test gaps (1 s, 2 s)."""
    await scheduler._deliver_due(clock())
    clock.advance(1)
    await scheduler._deliver_due(clock())
    clock.advance(2)
    await scheduler._deliver_due(clock())


@pytest.mark.asyncio
async def test_an_exhausted_round_gets_at_most_four_fallback_rounds_fifteen_minutes_apart(
    scheduler, store, mock_deps, clock,
):
    """Q31: without an output-free signal, a fallback round 900 s after each exhausted round, at most 4."""
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 1, ha_actions=[ACTION], scope=ReminderScope.SHARED)
    for round_number in range(1 + DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS):
        await fail_one_round(scheduler, clock)
        attempts = 3 * (round_number + 1)
        assert mock_deps["tts"].aspeak.await_count == attempts
        hold = scheduler._holds[r.id]
        if round_number < DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS:
            assert (hold.fallback_rounds, hold.retry_at) == (round_number + 1, clock() + 900)
            assert await scheduler._next_timeout() == 900
            clock.advance(899)
            await scheduler._deliver_due(clock())  # still inside the fallback wait: nothing
            assert mock_deps["tts"].aspeak.await_count == attempts
            clock.advance(1)
    # Bounded: after the fourth fallback round only a free output, a restart or a dismiss retries it.
    assert scheduler._holds[r.id].retry_at is None
    assert await scheduler._next_timeout() is None
    clock.advance(10 * HOUR)
    await scheduler._deliver_due(clock())
    assert mock_deps["tts"].aspeak.await_count == 15
    assert await notices(store) == [("failed", "delivery_not_started:failed", k) for k in range(1, 16)]
    # Every fallback round is late: its actions were barred before it spoke, never executed.
    texts = [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list]
    assert texts[:3] == ["Oye, recuerda regar"] * 3 and texts[3:] == [LATE_NOT_RUN] * 12
    assert await receipts(store) == [("failed", "late_not_executed")]
    mock_deps["ha_client"].call_service_once.assert_not_called()
    # The output-free signal still opens a round after the fallback rounds ran out.
    mock_deps["tts"].aspeak.return_value = delivered()
    scheduler.notify_output_available()
    await scheduler._deliver_due(clock())
    assert (await notices(store))[-1] == ("confirmed", None, 16)
    assert (await store.get_by_id(r.id)).state is ReminderState.FIRED


@pytest.mark.asyncio
async def test_an_output_free_round_neither_resets_the_fallback_bound_nor_wakes_without_a_held_notice(
    build, store, mock_deps, clock,
):
    scheduler = build(notice_fallback_max_rounds=1)
    scheduler.notify_output_available()
    assert not scheduler._wake_event.is_set()  # nothing held: every finished response stays cheap
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 1, scope=ReminderScope.SHARED)
    await fail_one_round(scheduler, clock)
    assert (scheduler._holds[r.id].fallback_rounds, scheduler._holds[r.id].retry_at) == (1, clock() + 900)
    scheduler._wake_event.clear()
    scheduler.notify_output_available()
    assert scheduler._wake_event.is_set()
    hold = scheduler._holds[r.id]
    assert (hold.failures, hold.retry_at) == (0, clock())  # a fresh round, claimable now
    await fail_one_round(scheduler, clock)  # the new round has its own three attempts and gaps
    assert mock_deps["tts"].aspeak.await_count == 6
    hold = scheduler._holds[r.id]
    assert (hold.failures, hold.fallback_rounds, hold.retry_at) == (3, 1, None)  # the single fallback was used
    assert await scheduler._next_timeout() is None


@pytest.mark.asyncio
async def test_a_held_daily_reminder_still_announces_its_next_occurrence(build, store, mock_deps, clock):
    """Q31: an occurrence held after its rounds never makes the reminder lose the next day's occurrence."""
    scheduler = build(notice_fallback_max_rounds=0)
    mock_deps["tts"].aspeak.return_value = failed()
    r = await store.create("u1", "regar", clock() - 1, recurrence="daily", ha_actions=[ACTION],
                           scope=ReminderScope.SHARED)
    following = next_trigger(r.trigger_at, "daily")
    await fail_one_round(scheduler, clock)
    assert scheduler._holds[r.id].retry_at is None
    assert await scheduler._next_timeout() == following - clock()  # the next occurrence wakes the loop
    clock.now = following + 10
    assert await scheduler._next_timeout() == 0
    mock_deps["tts"].aspeak.return_value = delivered()
    await scheduler._deliver_due(clock())
    # Announced on time like any catch-up: one notice, its action runs after the confirmed notice.
    assert [call.args[0] for call in mock_deps["tts"].aspeak.await_args_list] == ["Oye, recuerda regar"] * 4
    assert await notices(store) == [("failed", "delivery_not_started:failed", k) for k in (1, 2, 3)] + [
        ("confirmed", None, 1)]
    mock_deps["ha_client"].call_service_once.assert_awaited_once()
    assert await receipts(store) == [("confirmed", None)]
    current = await store.get_by_id(r.id)
    assert current.state is ReminderState.ACTIVE and current.trigger_at == next_trigger(following, "daily")


@pytest.mark.asyncio
async def test_a_held_occurrence_gets_one_round_per_newly_due_occurrence_and_never_spins(
    scheduler, store, mock_deps, clock,
):
    """The successor round is opened once per due occurrence: a round that cannot claim waits for the next."""
    r = await store.create("u1", "regar", clock() - 1, recurrence="daily", scope=ReminderScope.SHARED)
    # Another owner claimed attempt 1 and is still announcing it: this scheduler parks the occurrence.
    assert await store.claim_home_action(notice_request_id(r.id, r.trigger_at, 1), {
        "kind": "voice_notice", "reminder_id": r.id, "trigger_at": r.trigger_at, "attempt": 1,
        "claimed_at": clock(), "room": "living"})
    await scheduler._deliver_due(clock())
    assert scheduler._holds[r.id].retry_at is None
    first, second = next_trigger(r.trigger_at, "daily"), next_trigger(next_trigger(r.trigger_at, "daily"), "daily")
    clock.now = first + 5
    assert await scheduler._next_timeout() == 0
    await scheduler._deliver_due(clock())  # the round is opened, but attempt 1 is still 'sending'
    assert scheduler._holds[r.id].successor_round == first
    assert await scheduler._next_timeout() == second - clock()  # not 0: no spin
    for _ in range(3):
        await scheduler._deliver_due(clock())
    mock_deps["tts"].aspeak.assert_not_awaited()
    assert (await store.get_by_id(r.id)).trigger_at == r.trigger_at
    assert await notices(store) == [("sending", None, 1)]


@pytest.mark.parametrize("restart_after_latest", [HOUR, 30])
@pytest.mark.asyncio
async def test_a_crash_while_announcing_then_a_multi_day_outage_groups_the_missed_occurrences(
    tmp_path, mock_deps, clock, restart_after_latest,
):
    """Q32: the missed occurrences get one notice, as without the crash; the cut notice is never repeated.

    One hour after the latest missed occurrence the notice is late: barrier first, no home actions.
    Thirty seconds after it the latest occurrence is still on time, exactly like the catch-up collapse.
    """
    path = str(tmp_path / "outage.db")
    first, ha = await reopen(path), mock_deps["ha_client"]
    entered = asyncio.Event()

    async def speak(text, **kwargs):
        entered.set()
        await asyncio.Event().wait()  # the process dies while the notice plays

    tts = AsyncMock()
    tts.aspeak.side_effect = speak
    try:
        r = await first.create("u1", "regar", clock() - 1, ha_actions=[ACTION], recurrence="daily",
                               scope=ReminderScope.SHARED)
        task = asyncio.create_task(scheduler_on(first, tts, ha, clock)._fire_reminder(r))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    finally:
        await first.close()
    latest = r.trigger_at
    for _ in range(3):
        latest = next_trigger(latest, "daily")
    clock.now = latest + restart_after_latest
    restarted = await reopen(path)
    try:
        after = voice_output()
        scheduler = scheduler_on(restarted, after, ha, clock)
        await scheduler._deliver_due(clock())  # completes the cut notice silently and groups the missed days
        after.aspeak.assert_not_awaited()
        assert (await restarted.get_by_id(r.id)).trigger_at == latest
        assert await scheduler._next_timeout() == 0
        for _ in range(3):
            await scheduler._deliver_due(clock())
        late = restart_after_latest > DEFAULT_LATE_ACTION_GRACE_S
        after.aspeak.assert_awaited_once_with(LATE_NOT_RUN if late else "Oye, recuerda regar", zone_id="living")
        assert await notices(restarted) == [("uncertain", "interrupted_send", 1), ("confirmed", None, 1)]
        if late:
            ha.call_service_once.assert_not_called()
            assert await receipts(restarted) == [("failed", "late_not_executed")]
        else:
            ha.call_service_once.assert_awaited_once()
            assert await receipts(restarted) == [("confirmed", None)]
        current = await restarted.get_by_id(r.id)
        assert current.state is ReminderState.ACTIVE and current.trigger_at == next_trigger(latest, "daily")
    finally:
        await restarted.close()
    tts.aspeak.assert_awaited_once()
