"""Output-free signal of ResponseHandler (T034-U2, user answer Q31)."""

import asyncio
import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())

from src.core.voice_delivery import SIGNALS_OUTPUT_FREE, ResponseBudgets
from src.pipeline.response_handler import ResponseHandler
from src.tts import piper_tts
from tests.mocks.audio_output import AudioBackend

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
