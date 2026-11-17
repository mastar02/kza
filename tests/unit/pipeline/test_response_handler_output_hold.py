"""Output hold of ResponseHandler while PortAudio is reset (T025-U2-RV2-2)."""

import asyncio
import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())

from src.core.voice_delivery import ResponseBudgets
from src.pipeline.response_handler import ResponseHandler
from src.tts import piper_tts
from tests.mocks.audio_output import AudioBackend

pytestmark = pytest.mark.asyncio


class Output:
    """Zone playback that can keep 'zone_cocina' busy until released."""

    def __init__(self):
        self.busy = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.zones: list[str] = []

    def play(self, zone_id, source, **kwargs):
        self.zones.append(zone_id)
        if zone_id == "zone_cocina" and not self.release.is_set():
            self.busy.set()
            self.release.wait(5)
        return bool(list(source))


def make_handler(monkeypatch, output: Output, queue_wait_s: float = 2.0) -> ResponseHandler:
    """ResponseHandler with a fake TTS and zone output, no native audio."""
    monkeypatch.setattr(piper_tts, "sd", AudioBackend())
    tts = MagicMock()
    tts.synthesize.return_value = (np.ones(1600, dtype=np.float32), 0.1)
    tts.sample_rate = 16000
    del tts.synthesize_stream
    manager = MagicMock()
    manager.play_stream_to_zone.side_effect = output.play
    return ResponseHandler(tts=tts, zone_manager=manager, streaming_enabled=False,
                           response_budgets=ResponseBudgets(queue_wait_s=queue_wait_s))


async def test_a_notice_queued_during_the_hold_waits_and_is_delivered(monkeypatch):
    output = Output()
    handler = make_handler(monkeypatch, output)
    token = handler.try_hold_output()
    assert token is not None
    notice = asyncio.create_task(handler.aspeak("Recordatorio: sacar la basura", zone_id="zone_living"))
    await asyncio.sleep(.1)
    assert not notice.done()
    assert output.zones == [], "ninguna salida arranca durante el reset"
    # Liberar desde otro hilo, como lo hace el hilo del catálogo.
    await asyncio.to_thread(handler.release_output_hold, token)
    result = await notice
    assert result.success
    assert output.zones == ["zone_living"]


async def test_no_hold_while_a_response_owns_the_output(monkeypatch):
    output = Output()
    output.release.clear()
    handler = make_handler(monkeypatch, output)
    speaking = asyncio.create_task(handler.aspeak("hola", zone_id="zone_cocina"))
    await asyncio.to_thread(output.busy.wait, 2)
    assert handler.try_hold_output() is None
    output.release.set()
    assert (await speaking).success
    token = handler.try_hold_output()
    assert token is not None
    assert handler.try_hold_output() is None, "un solo hold a la vez"
    handler.release_output_hold(token)


async def test_release_is_idempotent_and_ignores_a_stale_token(monkeypatch):
    handler = make_handler(monkeypatch, Output())
    first = handler.try_hold_output()
    handler.release_output_hold(first)
    second = handler.try_hold_output()
    handler.release_output_hold(first)
    handler.release_output_hold(None)
    assert handler.try_hold_output() is None, "un token viejo no libera el hold actual"
    handler.release_output_hold(second)
    handler.release_output_hold(second)
    assert (await handler.aspeak("hola", zone_id="zone_living")).success
