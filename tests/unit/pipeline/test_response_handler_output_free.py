"""Output-free signal of ResponseHandler and the reminder rounds it opens (T034-U2, user answer Q31)."""

import asyncio
import json
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())

from src.core.voice_delivery import SIGNALS_OUTPUT_FREE, LocalVoiceDelivery, ResponseBudgets
from src.pipeline.response_handler import ResponseHandler
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.tts import piper_tts
from tests.mocks.audio_output import AudioBackend
from tests.mocks.home_control import selected_actions

pytestmark = pytest.mark.asyncio


class Output:
    """Zone playback: 'zone_cocina' can be held busy; 'zone_living' can refuse before any audio."""

    def __init__(self):
        self.busy = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.living_plays = True
        self.zones: list[str] = []

    def play(self, zone_id, source, **kwargs):
        self.zones.append(zone_id)
        if zone_id == "zone_cocina" and not self.release.is_set():
            self.busy.set()
            self.release.wait(5)
        if zone_id == "zone_living" and not self.living_plays:
            return False  # nothing consumed: no output started
        return bool(list(source))


@pytest.fixture
def output():
    return Output()


@pytest.fixture
def handler(monkeypatch, output):
    monkeypatch.setattr(piper_tts, "sd", AudioBackend())
    tts = MagicMock()
    tts.synthesize.return_value = (np.ones(1600, dtype=np.float32), 0.1)
    tts.sample_rate = 16000
    del tts.synthesize_stream
    manager = MagicMock()
    manager.play_stream_to_zone.side_effect = output.play
    return ResponseHandler(tts=tts, zone_manager=manager, streaming_enabled=False,
                           response_budgets=ResponseBudgets(queue_wait_s=.05))


async def until(predicate, timeout: float = 3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.005)


async def test_a_finished_response_signals_the_free_output_once(handler):
    calls = []
    handler.add_output_free_listener(lambda: calls.append(threading.get_ident()))
    result = await handler.aspeak("hola", zone_id="zone_cocina")
    assert result.success
    await until(lambda: calls)
    await asyncio.sleep(.05)
    assert len(calls) == 1


async def test_a_background_notice_never_signals_the_free_output(handler):
    calls = []
    handler.add_output_free_listener(lambda: calls.append(1))
    token = SIGNALS_OUTPUT_FREE.set(False)
    try:
        assert (await handler.aspeak("aviso", zone_id="zone_living")).success
    finally:
        SIGNALS_OUTPUT_FREE.reset(token)
    await until(lambda: handler.get_delivery_status()["workers"] == 0)
    await asyncio.sleep(.05)
    assert calls == []
    assert (await handler.aspeak("hola", zone_id="zone_cocina")).success  # the context var was restored
    await until(lambda: calls == [1])


async def test_the_output_is_free_only_when_no_other_response_waits_for_it(handler, output):
    calls = []
    handler.add_output_free_listener(lambda: calls.append(1))
    output.release.clear()
    first = asyncio.create_task(handler.aspeak("uno", zone_id="zone_cocina", timeout_s=5))
    await until(output.busy.is_set)
    handler.budgets = ResponseBudgets(queue_wait_s=5)
    second = asyncio.create_task(handler.aspeak("dos", zone_id="zone_living"))
    await until(lambda: handler.get_delivery_status()["pending"] == 1)
    output.release.set()
    assert (await first).success and (await second).success
    await until(lambda: calls)
    await asyncio.sleep(.05)
    assert calls == [1]  # not when 'uno' handed the output to the waiting 'dos'


async def test_a_failing_listener_never_breaks_delivery_or_the_other_listeners(handler):
    calls = []

    def broken():
        raise RuntimeError("listener bug")

    handler.add_output_free_listener(broken)
    handler.add_output_free_listener(lambda: calls.append(1))
    assert (await handler.aspeak("hola", zone_id="zone_cocina")).success
    await until(lambda: calls == [1])
    assert (await handler.aspeak("otra", zone_id="zone_cocina")).success
    await until(lambda: calls == [1, 1])


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "free.db"))
    await s.initialize()
    yield s
    await s.close()


async def notices(store) -> list[tuple]:
    rows = await store._work.read(lambda db: [tuple(row) for row in db.execute(
        "SELECT status, reason, body FROM reminder_home_actions WHERE request_id LIKE 'notice:%' "
        "ORDER BY rowid").fetchall()])
    return [(status, reason, json.loads(body)["reminder_id"], json.loads(body)["attempt"])
            for status, reason, body in rows]


def reminder_scheduler(handler, store) -> ReminderScheduler:
    """The production wiring of build_voice_services: one attempt per round and no fallback round,
    so only the output-free signal can retry."""
    ha = AsyncMock()
    scheduler = ReminderScheduler(
        store, LocalVoiceDelivery(handler, "zone_living", True), None, ha,
        {"output_zone": "zone_living", "notice_max_attempts": 1, "notice_fallback_max_rounds": 0},
        home_actions=selected_actions(ha, ("light.patio",)))
    handler.add_output_free_listener(scheduler.notify_output_available)
    return scheduler


async def test_a_notice_that_found_the_output_busy_is_announced_when_the_response_ends(handler, output, store):
    """Q31: the living output busy past the queue budget -> not started; its end opens the new round."""
    scheduler = reminder_scheduler(handler, store)
    output.release.clear()
    other = asyncio.create_task(handler.aspeak("respuesta larga", zone_id="zone_cocina", timeout_s=10))
    await until(output.busy.is_set)
    r = await store.create("u1", "regar", time.time() - 1, scope=ReminderScope.SHARED)
    loop = asyncio.create_task(scheduler.start())
    try:
        await until(lambda: r.id in scheduler._holds)
        assert await notices(store) == [("failed", "delivery_not_started:timed_out", r.id, 1)]
        await asyncio.sleep(.2)
        assert len(await notices(store)) == 1  # exhausted, and no fallback round: waits for the signal
        output.release.set()
        assert (await other).success
        async with asyncio.timeout(3):
            while (await store.get_by_id(r.id)).state is not ReminderState.FIRED:
                await asyncio.sleep(.01)
    finally:
        output.release.set()
        await scheduler.stop()
        await asyncio.gather(loop, other, return_exceptions=True)
    assert await notices(store) == [("failed", "delivery_not_started:timed_out", r.id, 1),
                                    ("confirmed", None, r.id, 2)]
    assert output.zones == ["zone_cocina", "zone_living"]


async def test_the_end_of_a_notice_never_opens_another_round(handler, output, store):
    """No ping-pong: one notice's end is not other activity, so held notices stay held until a real response."""
    scheduler = reminder_scheduler(handler, store)
    output.living_plays = False
    first = await store.create("u1", "regar", time.time() - 1, scope=ReminderScope.SHARED)
    loop = asyncio.create_task(scheduler.start())
    try:
        await until(lambda: first.id in scheduler._holds)
        second = await store.create("u1", "ventilar", time.time() - 1, scope=ReminderScope.SHARED)
        scheduler.notify_new_reminder()
        await until(lambda: second.id in scheduler._holds)
        await asyncio.sleep(.3)
        assert [(item, attempt) for *_, item, attempt in await notices(store)] == [(first.id, 1), (second.id, 1)]
        assert (await handler.aspeak("hola", zone_id="zone_cocina")).success
        async with asyncio.timeout(3):
            while len(await notices(store)) < 4:
                await asyncio.sleep(.01)
        await asyncio.sleep(.3)
    finally:
        await scheduler.stop()
        await asyncio.gather(loop, return_exceptions=True)
    rows = await notices(store)
    assert all(status == "failed" for status, *_ in rows)
    assert sorted((item, attempt) for *_, item, attempt in rows) == sorted(
        [(first.id, 1), (first.id, 2), (second.id, 1), (second.id, 2)])
    assert output.zones.count("zone_living") == 3  # first, second, then one summary for both
