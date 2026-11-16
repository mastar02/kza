"""
Tests for ResponseHandler room context zone routing.

Verifies that ResponseHandler.speak() and speak_with_llm_stream() correctly
resolve zones from RoomContext when no explicit zone_id is provided.
"""

import sys
from unittest.mock import MagicMock
from types import SimpleNamespace
import numpy as np
from src.llm.generation_stream import GenerationStream
from tests.mocks.audio_output import AudioBackend
from src.tts import piper_tts

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest

from src.pipeline.response_handler import ResponseHandler
from src.rooms.room_context import RoomContext, ContextSource


@pytest.fixture
def handler(monkeypatch):
    """ResponseHandler with mocked TTS and ZoneManager."""
    monkeypatch.setattr(piper_tts, "sd", AudioBackend())
    tts = MagicMock()
    tts.synthesize.return_value = (np.ones(1600, dtype=np.float32), 0.1)
    tts.sample_rate = 16000
    # Remove synthesize_stream so it uses non-streaming path
    del tts.synthesize_stream
    manager = MagicMock()
    manager.play_stream_to_zone.side_effect = lambda zone_id, source, **kwargs: bool(list(source))
    return ResponseHandler(
        tts=tts,
        zone_manager=manager,
        streaming_enabled=False,
    )


@pytest.fixture
def cocina_context():
    """RoomContext for cocina."""
    return RoomContext(
        room_id="cocina",
        room_name="Cocina",
        display_name="la cocina",
        source=ContextSource.MICROPHONE,
        confidence=0.7,
        timestamp=0,
        entities={},
    )


class TestSpeakWithRoomContext:
    """Tests for speak() method with room_context parameter."""

    def test_speak_with_room_context_routes_to_zone(self, handler, cocina_context):
        """When room_context is provided, audio routes to zone_<room_id>."""
        handler.speak("Listo", room_context=cocina_context)

        handler.zone_manager.play_stream_to_zone.assert_called_once()
        call_kwargs = handler.zone_manager.play_stream_to_zone.call_args
        assert call_kwargs[0][0] == "zone_cocina"

    def test_speak_without_room_context_uses_default(self, handler):
        """Without room_context or zone_id, falls back to direct TTS."""
        handler._active_zone_id = None
        handler.speak("Hola")

        # No zone routing — should fall through to _speak_direct
        handler.zone_manager.play_stream_to_zone.assert_not_called()
        handler.tts.synthesize.assert_called_once_with("Hola")

    def test_speak_explicit_zone_overrides_room_context(self, handler, cocina_context):
        """Explicit zone_id takes precedence over room_context."""
        handler.speak("Hola", zone_id="zone_living", room_context=cocina_context)

        call_kwargs = handler.zone_manager.play_stream_to_zone.call_args
        assert call_kwargs[0][0] == "zone_living"

    def test_speak_room_context_with_active_zone_prefers_context(self, handler, cocina_context):
        """room_context should set zone_id, which takes precedence over _active_zone_id."""
        handler._active_zone_id = "zone_escritorio"
        handler.speak("Listo", room_context=cocina_context)

        call_kwargs = handler.zone_manager.play_stream_to_zone.call_args
        assert call_kwargs[0][0] == "zone_cocina"

    def test_speak_empty_text_does_nothing(self, handler, cocina_context):
        """Empty text should return immediately without zone routing."""
        handler.speak("", room_context=cocina_context)

        handler.zone_manager.play_stream_to_zone.assert_not_called()
        handler.tts.synthesize.assert_not_called()


class TestSpeakWithLlmStreamRoomContext:
    """Tests for speak_with_llm_stream() method with room_context parameter."""

    def test_llm_stream_with_room_context_sets_zone(self, handler, cocina_context):
        """speak_with_llm_stream resolves zone from room_context."""
        llm = SimpleNamespace(generate_stream=lambda *a, **kw: iter([{"token": "La luz esta encendida"}]))
        handler.llm = llm
        handler._generation = GenerationStream(llm)

        handler.speak_with_llm_stream(
            "Que esta prendido?",
            room_context=cocina_context,
        )

        # The fallback path calls set_active_zone then speak
        assert handler.zone_manager.play_stream_to_zone.call_args[0][0] == "zone_cocina"

    def test_llm_stream_explicit_zone_overrides_room_context(self, handler, cocina_context):
        """Explicit zone_id in speak_with_llm_stream overrides room_context."""
        llm = SimpleNamespace(generate_stream=lambda *a, **kw: iter([{"token": "Respuesta"}]))
        handler.llm = llm
        handler._generation = GenerationStream(llm)

        handler.speak_with_llm_stream(
            "Pregunta",
            zone_id="zone_living",
            room_context=cocina_context,
        )

        assert handler.zone_manager.play_stream_to_zone.call_args[0][0] == "zone_living"


# ---------------------------------------------------------------------------
# T034-U2: ResponseResult.output_started (notice delivery mapping input)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from src.core.voice_delivery import DeliveryState, ResponseBudgets, ResponseStage  # noqa: E402
from tests.mocks.audio_output import GatedZones  # noqa: E402


def gated_handler(monkeypatch, zones, **kwargs):
    monkeypatch.setattr(piper_tts, "sd", AudioBackend())
    tts = MagicMock()
    tts.synthesize.return_value = (np.ones(1600, dtype=np.float32), 0.1)
    tts.sample_rate = 16000
    del tts.synthesize_stream
    return ResponseHandler(tts=tts, zone_manager=zones, streaming_enabled=False, **kwargs)


@pytest.mark.asyncio
async def test_output_started_is_true_once_a_pcm_chunk_reached_the_player(monkeypatch):
    zones = GatedZones()
    handler = gated_handler(monkeypatch, zones)
    try:
        result = await handler.aspeak("Oye, recuerda regar", zone_id="zone_living")
    finally:
        await handler.aclose()
    assert result.success and result.spoken_segments == 1 and result.output_started is True
    assert zones.pulled == 1


@pytest.mark.asyncio
async def test_living_barge_in_during_the_first_sentence_reports_output_started_with_zero_segments(monkeypatch):
    """C02-NTC-0010 input: session.spoken counts finished segments only, so the cut sentence reports 0."""
    zones = GatedZones(block='after_first_chunk')
    handler = gated_handler(monkeypatch, zones)
    try:
        task = asyncio.create_task(handler.aspeak("Oye, recuerda regar", zone_id="zone_living"))
        assert await asyncio.to_thread(zones.waiting.wait, 2)
        assert await handler.cancel(zone_id="zone_living")  # as multi_room_audio_loop._trigger_barge_in issues it
        result = await asyncio.wait_for(task, 2)
    finally:
        zones.release.set()
        await handler.aclose()
    assert result.delivery is DeliveryState.CANCELLED
    assert result.spoken_segments == 0 and result.output_started is True


@pytest.mark.asyncio
async def test_cancel_before_the_first_chunk_reports_no_output(monkeypatch):
    zones = GatedZones(block='before_pull')
    handler = gated_handler(monkeypatch, zones)
    try:
        task = asyncio.create_task(handler.aspeak("Oye, recuerda regar", zone_id="zone_living"))
        assert await asyncio.to_thread(zones.waiting.wait, 2)
        assert await handler.cancel(zone_id="zone_living")
        result = await asyncio.wait_for(task, 2)
    finally:
        zones.release.set()
        await handler.aclose()
    assert result.delivery is DeliveryState.CANCELLED
    assert (result.spoken_segments, result.output_started, zones.pulled) == (0, False, 0)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_cancel_while_the_first_chunk_is_still_being_synthesized_reports_no_output(monkeypatch, stream):
    """Review round 1 R3: output starts when a PCM chunk reaches the player, not when synthesis starts.

    The zone pulls at once, so the player waits inside the first synthesis step
    (synthesize_stream) or the handler waits inside synthesize; a barge-in there played nothing,
    and the scheduler keeps the notice retryable.
    """
    import threading

    entered, release = threading.Event(), threading.Event()

    def blocked_synthesis():
        entered.set()
        release.wait(5)
        return np.ones(1600, dtype=np.float32)

    def synthesize_stream(text):
        yield blocked_synthesis()

    zones = GatedZones()
    handler = gated_handler(monkeypatch, zones)
    if stream:
        handler.tts.synthesize_stream = synthesize_stream
    else:
        handler.tts.synthesize.side_effect = lambda text: (blocked_synthesis(), 0.1)
    try:
        task = asyncio.create_task(handler.aspeak("Oye, recuerda regar", zone_id="zone_living", stream=stream))
        assert await asyncio.to_thread(entered.wait, 2)
        assert await handler.cancel(zone_id="zone_living")
        release.set()
        result = await asyncio.wait_for(task, 2)
    finally:
        release.set()
        await handler.aclose()
    assert result.delivery is DeliveryState.CANCELLED
    assert (result.spoken_segments, result.output_started, zones.pulled) == (0, False, 0)
    if stream:
        assert zones.plays == ["zone_living"]  # the player itself waited inside the first synthesis step
    from src.reminders.reminder_scheduler import notice_outcome
    from src.reminders.reminder_store import ReminderActionStatus
    assert notice_outcome(result) == (ReminderActionStatus.FAILED, "delivery_not_started:cancelled")


@pytest.mark.asyncio
async def test_queue_stage_cancel_result_has_output_started_false(monkeypatch):
    """C02-NTC-0011 input: a notice still waiting for the output is cancelled by the living barge-in."""
    zones = GatedZones(block='after_first_chunk')
    handler = gated_handler(monkeypatch, zones)
    try:
        busy = asyncio.create_task(handler.aspeak("Otra respuesta", zone_id="zone_hall"))
        assert await asyncio.to_thread(zones.waiting.wait, 2)
        queued = asyncio.create_task(handler.aspeak("Oye, recuerda regar", zone_id="zone_living"))
        while handler.get_delivery_status()['pending'] == 0:
            await asyncio.sleep(.005)
        assert await handler.cancel(zone_id="zone_living")
        result = await asyncio.wait_for(queued, 2)
        zones.release.set()
        assert (await asyncio.wait_for(busy, 2)).success
    finally:
        zones.release.set()
        await handler.aclose()
    assert result.delivery is DeliveryState.CANCELLED
    assert result.spoken_segments == 0 and result.output_started is False
    assert zones.pulled == 1  # only the other response played


@pytest.mark.parametrize("block, started", [("after_first_chunk", True), ("before_pull", False)])
@pytest.mark.asyncio
async def test_timed_out_result_carries_output_started(monkeypatch, block, started):
    zones = GatedZones(block=block)
    handler = gated_handler(monkeypatch, zones, response_budgets=ResponseBudgets(delivery_s=.2))
    try:
        result = await asyncio.wait_for(handler.aspeak("Oye, recuerda regar", zone_id="zone_living"), 5)
    finally:
        zones.release.set()
        await handler.aclose()
    assert result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is ResponseStage.DELIVERY
    assert result.spoken_segments == 0 and result.output_started is started
