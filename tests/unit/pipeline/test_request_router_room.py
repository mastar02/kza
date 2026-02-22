"""
Tests for RequestRouter room context integration.

Tests ensure that:
1. CommandEvent is unpacked correctly (audio + room_id extracted)
2. RoomContextManager.resolve_room() is called with correct args
3. Room context is attached to result dict
4. room_context is passed to response_handler.speak()
5. Backward compatibility: raw np.ndarray still works without room_context_manager
6. Both orchestrated and legacy paths handle room context
"""

import sys
from unittest.mock import MagicMock, AsyncMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
import numpy as np

from src.pipeline.request_router import RequestRouter
from src.pipeline.command_event import CommandEvent
from src.rooms.room_context import ContextSource


# ============================================================
# Helpers
# ============================================================

def _make_cmd_result(text="enciende la luz", user=None, emotion=None, timings=None):
    """Create a mock CommandProcessor result."""
    return {
        "text": text,
        "user": user,
        "emotion": emotion,
        "timings": timings or {"stt": 50.0}
    }


def _make_mock_user(user_id="user1", name="Test User"):
    """Create a mock user object."""
    user = MagicMock()
    user.user_id = user_id
    user.name = name
    user.permission_level = MagicMock()
    user.permission_level.name = "ADULT"
    return user


def _make_room_context(room_id="cocina", room_name="Cocina", confidence=0.9):
    """Create a mock RoomContext."""
    ctx = MagicMock()
    ctx.room_id = room_id
    ctx.room_name = room_name
    ctx.confidence = confidence
    ctx.source = ContextSource.MICROPHONE
    return ctx


def _make_router(room_context_manager=None, orchestrator=None, orchestrator_enabled=False):
    """Create a RequestRouter with mocked dependencies."""
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=_make_cmd_result())

    response_handler = MagicMock()
    response_handler.speak = MagicMock()
    response_handler.set_active_zone = MagicMock()

    audio_manager = MagicMock()
    audio_manager.detect_source_zone = MagicMock(return_value=None)

    # Legacy path needs routines
    routines = MagicMock()
    routines.handle = AsyncMock(return_value={"handled": False})

    chroma = MagicMock()
    chroma.search_command = MagicMock(return_value=None)

    router = RequestRouter(
        command_processor=command_processor,
        response_handler=response_handler,
        audio_manager=audio_manager,
        room_context_manager=room_context_manager,
        orchestrator=orchestrator,
        orchestrator_enabled=orchestrator_enabled,
        chroma_sync=chroma,
        routine_manager=routines,
    )
    return router


# ============================================================
# Tests: CommandEvent extraction
# ============================================================

class TestProcessCommandExtraction:
    """Test that process_command correctly handles CommandEvent vs np.ndarray."""

    @pytest.mark.asyncio
    async def test_process_command_extracts_room_from_event(self):
        """CommandEvent should be unpacked into audio + room_id."""
        router = _make_router()
        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        await router.process_command(event)

        # Verify command_processor received the raw audio (not the event)
        call_args = router.command_processor.process_command.call_args
        received_audio = call_args[0][0]
        assert isinstance(received_audio, np.ndarray)
        assert np.array_equal(received_audio, audio)

    @pytest.mark.asyncio
    async def test_process_command_backward_compat_ndarray(self):
        """Raw np.ndarray should still work without room_context_manager."""
        router = _make_router()
        audio = np.zeros(16000, dtype=np.float32)

        result = await router.process_command(audio)

        # Should complete without errors
        assert "text" in result
        router.command_processor.process_command.assert_awaited_once()


# ============================================================
# Tests: Room context resolution
# ============================================================

class TestRoomContextResolution:
    """Test room context resolution in both paths."""

    @pytest.mark.asyncio
    async def test_resolves_room_context_legacy(self):
        """Room context manager should be called with room_id from CommandEvent."""
        room_ctx = _make_room_context()
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        router = _make_router(room_context_manager=room_mgr)
        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        result = await router.process_command(event)

        room_mgr.resolve_room.assert_called_once_with(
            mic_zone_id="cocina",
            user_id=None,  # No user detected (mock returns no user)
        )

    @pytest.mark.asyncio
    async def test_resolves_room_context_with_user(self):
        """Room resolution should include user_id when speaker is identified."""
        room_ctx = _make_room_context()
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        mock_user = _make_mock_user(user_id="mastar")
        cmd_result = _make_cmd_result(user=mock_user)

        router = _make_router(room_context_manager=room_mgr)
        router.command_processor.process_command = AsyncMock(return_value=cmd_result)

        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="escritorio")

        await router.process_command(event)

        room_mgr.resolve_room.assert_called_once_with(
            mic_zone_id="escritorio",
            user_id="mastar",
        )

    @pytest.mark.asyncio
    async def test_no_room_resolution_without_room_id(self):
        """When raw ndarray is passed (no room_id), resolve_room should not be called."""
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock()

        router = _make_router(room_context_manager=room_mgr)
        audio = np.zeros(16000, dtype=np.float32)

        await router.process_command(audio)

        room_mgr.resolve_room.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_room_resolution_without_manager(self):
        """When no room_context_manager is set, it should work fine."""
        router = _make_router(room_context_manager=None)
        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        result = await router.process_command(event)

        # Should work, just no room in result
        assert "room" not in result


# ============================================================
# Tests: Room context in result
# ============================================================

class TestRoomContextInResult:
    """Test that room context data is attached to the result dict."""

    @pytest.mark.asyncio
    async def test_room_context_in_result(self):
        """Result dict should contain room info when context is resolved."""
        room_ctx = _make_room_context(
            room_id="cocina",
            room_name="Cocina",
            confidence=0.9,
        )
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        router = _make_router(room_context_manager=room_mgr)
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")

        result = await router.process_command(event)

        assert "room" in result
        assert result["room"]["id"] == "cocina"
        assert result["room"]["name"] == "Cocina"
        assert result["room"]["confidence"] == 0.9
        assert result["room"]["source"] == "microphone"

    @pytest.mark.asyncio
    async def test_no_room_in_result_when_no_context(self):
        """Result dict should NOT contain room key when no context resolved."""
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=None)

        router = _make_router(room_context_manager=room_mgr)
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="unknown_room")

        result = await router.process_command(event)

        assert "room" not in result


# ============================================================
# Tests: Room context passed to speak()
# ============================================================

class TestRoomContextInSpeak:
    """Test that room_context is forwarded to response_handler.speak()."""

    @pytest.mark.asyncio
    async def test_speak_receives_room_context_legacy_conversation(self):
        """In legacy conversation path, speak() should receive room_context."""
        room_ctx = _make_room_context()
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        # Set up router to use fast router path (conversation, no deep)
        fast_router = MagicMock()
        fast_router.classify_and_respond = MagicMock(return_value=(False, "Respuesta rapida"))

        router = _make_router(room_context_manager=room_mgr)
        router.router = fast_router
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")

        await router.process_command(event)

        # Check speak was called with room_context
        speak_calls = router.response_handler.speak.call_args_list
        assert len(speak_calls) > 0
        last_call = speak_calls[-1]
        assert last_call.kwargs.get("room_context") == room_ctx

    @pytest.mark.asyncio
    async def test_speak_receives_room_context_orchestrated(self):
        """In orchestrated path, speak() should receive room_context."""
        room_ctx = _make_room_context()
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        # Mock orchestrator
        dispatch_result = MagicMock()
        dispatch_result.intent = "domotics"
        dispatch_result.response = "Listo"
        dispatch_result.success = True
        dispatch_result.action = None
        dispatch_result.path = MagicMock()
        dispatch_result.path.value = "fast"
        dispatch_result.was_queued = False
        dispatch_result.timings = {}
        # Ensure path is not SLOW_LLM so speak() is called
        from src.orchestrator import PathType
        dispatch_result.path = PathType.FAST_DOMOTICS

        orchestrator = MagicMock()
        orchestrator.process = AsyncMock(return_value=dispatch_result)

        router = _make_router(
            room_context_manager=room_mgr,
            orchestrator=orchestrator,
            orchestrator_enabled=True,
        )
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="living")

        await router.process_command(event)

        # Check speak was called with room_context
        speak_calls = router.response_handler.speak.call_args_list
        assert len(speak_calls) > 0
        last_call = speak_calls[-1]
        assert last_call.kwargs.get("room_context") == room_ctx


# ============================================================
# Tests: Zone routing from room context
# ============================================================

class TestZoneRoutingFromRoom:
    """Test that zone_id is derived from room context."""

    @pytest.mark.asyncio
    async def test_zone_id_from_room_context_orchestrated(self):
        """In orchestrated path, zone_id should be zone_{room_id} when context is resolved."""
        room_ctx = _make_room_context(room_id="escritorio")
        room_mgr = MagicMock()
        room_mgr.resolve_room = MagicMock(return_value=room_ctx)

        dispatch_result = MagicMock()
        dispatch_result.intent = "domotics"
        dispatch_result.response = "Listo"
        dispatch_result.success = True
        dispatch_result.action = None
        dispatch_result.was_queued = False
        dispatch_result.timings = {}
        from src.orchestrator import PathType
        dispatch_result.path = PathType.FAST_DOMOTICS

        orchestrator = MagicMock()
        orchestrator.process = AsyncMock(return_value=dispatch_result)

        router = _make_router(
            room_context_manager=room_mgr,
            orchestrator=orchestrator,
            orchestrator_enabled=True,
        )
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="escritorio")

        await router.process_command(event)

        # set_active_zone should be called with zone derived from room context
        router.response_handler.set_active_zone.assert_called_with("zone_escritorio")

    @pytest.mark.asyncio
    async def test_zone_fallback_without_room_context_orchestrated(self):
        """Without room context in orchestrated path, should fall back to detect_source_zone."""
        dispatch_result = MagicMock()
        dispatch_result.intent = "domotics"
        dispatch_result.response = "Listo"
        dispatch_result.success = True
        dispatch_result.action = None
        dispatch_result.was_queued = False
        dispatch_result.timings = {}
        from src.orchestrator import PathType
        dispatch_result.path = PathType.FAST_DOMOTICS

        orchestrator = MagicMock()
        orchestrator.process = AsyncMock(return_value=dispatch_result)

        router = _make_router(orchestrator=orchestrator, orchestrator_enabled=True)
        router.audio_manager.detect_source_zone = MagicMock(return_value="zone_auto")
        audio = np.zeros(16000, dtype=np.float32)

        # Use raw ndarray (no room_id) in orchestrated path
        await router.process_command(audio)

        # Should have called detect_source_zone since no room context
        router.audio_manager.detect_source_zone.assert_called_once()
