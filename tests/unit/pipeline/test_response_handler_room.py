"""
Tests for ResponseHandler room context zone routing.

Verifies that ResponseHandler.speak() and speak_with_llm_stream() correctly
resolve zones from RoomContext when no explicit zone_id is provided.
"""

import sys
from unittest.mock import MagicMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest

from src.pipeline.response_handler import ResponseHandler
from src.rooms.room_context import RoomContext, ContextSource


@pytest.fixture
def handler():
    """ResponseHandler with mocked TTS and ZoneManager."""
    tts = MagicMock()
    tts.synthesize.return_value = (b"audio", 16000)
    tts.sample_rate = 16000
    # Remove synthesize_stream so it uses non-streaming path
    del tts.synthesize_stream
    return ResponseHandler(
        tts=tts,
        zone_manager=MagicMock(),
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

        handler.zone_manager.play_to_zone.assert_called_once()
        call_kwargs = handler.zone_manager.play_to_zone.call_args
        assert call_kwargs[1]["zone_id"] == "zone_cocina"

    def test_speak_without_room_context_uses_default(self, handler):
        """Without room_context or zone_id, falls back to direct TTS."""
        handler._active_zone_id = None
        handler.speak("Hola")

        # No zone routing — should fall through to _speak_direct
        handler.zone_manager.play_to_zone.assert_not_called()
        handler.tts.speak.assert_called_once_with("Hola")

    def test_speak_explicit_zone_overrides_room_context(self, handler, cocina_context):
        """Explicit zone_id takes precedence over room_context."""
        handler.speak("Hola", zone_id="zone_living", room_context=cocina_context)

        call_kwargs = handler.zone_manager.play_to_zone.call_args
        assert call_kwargs[1]["zone_id"] == "zone_living"

    def test_speak_room_context_with_active_zone_prefers_context(self, handler, cocina_context):
        """room_context should set zone_id, which takes precedence over _active_zone_id."""
        handler._active_zone_id = "zone_escritorio"
        handler.speak("Listo", room_context=cocina_context)

        call_kwargs = handler.zone_manager.play_to_zone.call_args
        assert call_kwargs[1]["zone_id"] == "zone_cocina"

    def test_speak_empty_text_does_nothing(self, handler, cocina_context):
        """Empty text should return immediately without zone routing."""
        handler.speak("", room_context=cocina_context)

        handler.zone_manager.play_to_zone.assert_not_called()
        handler.tts.synthesize.assert_not_called()


class TestSpeakWithLlmStreamRoomContext:
    """Tests for speak_with_llm_stream() method with room_context parameter."""

    def test_llm_stream_with_room_context_sets_zone(self, handler, cocina_context):
        """speak_with_llm_stream resolves zone from room_context."""
        llm = MagicMock()
        llm.generate.return_value = "La luz esta encendida"
        handler.llm = llm

        handler.speak_with_llm_stream(
            "Que esta prendido?",
            room_context=cocina_context,
        )

        # The fallback path calls set_active_zone then speak
        handler.zone_manager.set_active_zone.assert_called_with("zone_cocina")

    def test_llm_stream_explicit_zone_overrides_room_context(self, handler, cocina_context):
        """Explicit zone_id in speak_with_llm_stream overrides room_context."""
        llm = MagicMock()
        llm.generate.return_value = "Respuesta"
        handler.llm = llm

        handler.speak_with_llm_stream(
            "Pregunta",
            zone_id="zone_living",
            room_context=cocina_context,
        )

        handler.zone_manager.set_active_zone.assert_called_with("zone_living")
