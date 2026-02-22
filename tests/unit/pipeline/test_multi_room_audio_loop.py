"""
Tests for MultiRoomAudioLoop — parallel audio capture from multiple rooms.

Tests ensure that:
1. RoomStream holds per-room state correctly
2. MultiRoomAudioLoop initializes with room streams
3. Deduplication keeps strongest RMS within window
4. Deduplication allows independent commands after window
5. on_command registers callback
6. _dispatch_command calls the registered callback with CommandEvent
"""

import sys
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import numpy as np
import pytest

from src.pipeline.multi_room_audio_loop import (
    MultiRoomAudioLoop,
    RoomStream,
    CHUNK_SIZE,
)
from src.pipeline.command_event import CommandEvent


# ============================================================
# Helpers
# ============================================================

def _make_wake_detector():
    """Create a mock WakeWordDetector."""
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(return_value=None)
    m.get_active_models = MagicMock(return_value=["hey_jarvis"])
    return m


def _make_echo_suppressor():
    """Create a mock EchoSuppressor."""
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=True)
    m.config = MagicMock()
    m.config.post_speech_buffer_ms = 400
    return m


def _make_follow_up():
    """Create a mock FollowUpMode."""
    m = MagicMock()
    m.is_active = False
    m.follow_up_window = 8.0
    m.start_conversation = MagicMock()
    return m


def _make_room_stream(room_id: str, device_index: int = 0) -> RoomStream:
    """Create a RoomStream with mock dependencies."""
    return RoomStream(
        room_id=room_id,
        device_index=device_index,
        wake_detector=_make_wake_detector(),
        echo_suppressor=_make_echo_suppressor(),
    )


def _make_multi_room_loop(rooms=None, **kwargs) -> MultiRoomAudioLoop:
    """Create a MultiRoomAudioLoop with mock dependencies."""
    if rooms is None:
        rooms = {
            "cocina": _make_room_stream("cocina", device_index=2),
            "living": _make_room_stream("living", device_index=3),
        }
    return MultiRoomAudioLoop(
        room_streams=rooms,
        follow_up=_make_follow_up(),
        **kwargs,
    )


# ============================================================
# Tests
# ============================================================

class TestRoomStream:
    """Test RoomStream dataclass."""

    def test_room_stream_creation(self):
        """RoomStream holds per-room state."""
        wake = _make_wake_detector()
        echo = _make_echo_suppressor()

        rs = RoomStream(
            room_id="cocina",
            device_index=2,
            wake_detector=wake,
            echo_suppressor=echo,
        )

        assert rs.room_id == "cocina"
        assert rs.device_index == 2
        assert rs.wake_detector is wake
        assert rs.echo_suppressor is echo
        assert rs.listening is False
        assert rs.audio_buffer == []
        assert rs.command_start_time == 0.0

    def test_room_stream_mutable_state(self):
        """RoomStream state can be modified during capture."""
        rs = _make_room_stream("living", device_index=5)

        rs.listening = True
        rs.command_start_time = time.time()
        rs.audio_buffer = [0.1, 0.2, 0.3]

        assert rs.listening is True
        assert rs.command_start_time > 0
        assert len(rs.audio_buffer) == 3


class TestMultiRoomAudioLoopInit:
    """Test MultiRoomAudioLoop initialization."""

    def test_multi_room_audio_loop_init(self):
        """MultiRoomAudioLoop initializes with room streams."""
        rooms = {
            "cocina": _make_room_stream("cocina", 2),
            "living": _make_room_stream("living", 3),
        }
        follow_up = _make_follow_up()

        loop = MultiRoomAudioLoop(
            room_streams=rooms,
            follow_up=follow_up,
            sample_rate=16000,
            dedup_window_ms=200,
        )

        assert loop.room_streams is rooms
        assert loop.follow_up is follow_up
        assert loop.sample_rate == 16000
        assert loop.dedup_window_ms == 200
        assert loop._running is False
        assert loop._on_command_callback is None
        assert loop._on_post_command_callback is None
        assert len(loop.room_streams) == 2

    def test_multi_room_audio_loop_defaults(self):
        """MultiRoomAudioLoop uses sensible defaults."""
        loop = _make_multi_room_loop()

        assert loop.sample_rate == 16000
        assert loop.command_duration == 2.0
        assert loop.silence_threshold == 0.015
        assert loop.silence_duration_ms == 300
        assert loop.min_speech_ms == 300
        assert loop.dedup_window_ms == 200


class TestDeduplication:
    """Test wake word deduplication between rooms."""

    def test_dedup_same_wakeword_within_window(self):
        """Within 200ms: keep strongest RMS, discard weaker."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # First room detects wake word
        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        # Second room detects within window but weaker RMS — echo
        result2 = loop._should_accept_wakeword("living", rms=0.02, timestamp=now + 0.05)
        assert result2 is False

        # Verify cocina is still the accepted room
        assert loop._last_wakeword_room == "cocina"

    def test_dedup_stronger_replaces_weaker(self):
        """Within window: stronger RMS from second room replaces first."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # Weaker room detects first
        result1 = loop._should_accept_wakeword("cocina", rms=0.02, timestamp=now)
        assert result1 is True

        # Stronger room detects within window — replaces
        result2 = loop._should_accept_wakeword("living", rms=0.08, timestamp=now + 0.05)
        assert result2 is True
        assert loop._last_wakeword_room == "living"
        assert loop._last_wakeword_rms == 0.08

    def test_dedup_allows_after_window(self):
        """After 200ms: both accepted as independent commands."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # First room
        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        # Second room well after window
        result2 = loop._should_accept_wakeword("living", rms=0.03, timestamp=now + 0.5)
        assert result2 is True

        # Both are independent, living is now the latest
        assert loop._last_wakeword_room == "living"

    def test_dedup_same_room_within_window(self):
        """Same room within window is always accepted."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        result2 = loop._should_accept_wakeword("cocina", rms=0.04, timestamp=now + 0.05)
        assert result2 is True


class TestCallbacks:
    """Test callback registration."""

    def test_on_command_callback(self):
        """on_command registers callback."""
        loop = _make_multi_room_loop()

        async def my_callback(event: CommandEvent) -> dict:
            return {"text": "test"}

        loop.on_command(my_callback)
        assert loop._on_command_callback is my_callback

    def test_on_post_command_callback(self):
        """on_post_command registers callback."""
        loop = _make_multi_room_loop()

        async def my_post_callback(result: dict, event: CommandEvent) -> None:
            pass

        loop.on_post_command(my_post_callback)
        assert loop._on_post_command_callback is my_post_callback

    def test_both_callbacks(self):
        """Both callbacks can be registered independently."""
        loop = _make_multi_room_loop()

        async def cmd_cb(event):
            return {}

        async def post_cb(result, event):
            pass

        loop.on_command(cmd_cb)
        loop.on_post_command(post_cb)

        assert loop._on_command_callback is cmd_cb
        assert loop._on_post_command_callback is post_cb


class TestDispatchCommand:
    """Test _dispatch_command async method."""

    @pytest.mark.asyncio
    async def test_dispatch_command_calls_callback(self):
        """_dispatch_command calls the registered callback with CommandEvent."""
        loop = _make_multi_room_loop()

        received_events = []

        async def mock_callback(event: CommandEvent) -> dict:
            received_events.append(event)
            return {"text": "luz encendida", "success": True}

        loop.on_command(mock_callback)

        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina", mic_device_index=2)

        await loop._dispatch_command(event)

        assert len(received_events) == 1
        assert received_events[0].room_id == "cocina"
        assert received_events[0].mic_device_index == 2

    @pytest.mark.asyncio
    async def test_dispatch_command_calls_post_callback(self):
        """_dispatch_command calls both on_command and on_post_command."""
        loop = _make_multi_room_loop()

        post_results = []

        async def mock_cmd(event: CommandEvent) -> dict:
            return {"text": "ok", "success": True}

        async def mock_post(result: dict, event: CommandEvent) -> None:
            post_results.append((result, event.room_id))

        loop.on_command(mock_cmd)
        loop.on_post_command(mock_post)

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="living", mic_device_index=3)

        await loop._dispatch_command(event)

        assert len(post_results) == 1
        assert post_results[0][0] == {"text": "ok", "success": True}
        assert post_results[0][1] == "living"

    @pytest.mark.asyncio
    async def test_dispatch_command_no_callback(self, caplog):
        """_dispatch_command warns if no callback registered."""
        import logging

        loop = _make_multi_room_loop()

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        with caplog.at_level(logging.WARNING):
            await loop._dispatch_command(event)

        assert "No on_command callback registered" in caplog.text

    @pytest.mark.asyncio
    async def test_dispatch_command_handles_exception(self, caplog):
        """_dispatch_command catches and logs callback exceptions."""
        import logging

        loop = _make_multi_room_loop()

        async def failing_callback(event):
            raise RuntimeError("STT failed")

        loop.on_command(failing_callback)

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        with caplog.at_level(logging.ERROR):
            await loop._dispatch_command(event)

        assert "Command dispatch failed for cocina" in caplog.text


class TestStart:
    """Test start() method."""

    @pytest.mark.asyncio
    async def test_start_loads_wake_word_all_rooms(self):
        """start() calls wake_detector.load() for every room."""
        loop = _make_multi_room_loop()

        await loop.start()

        for room_id, rs in loop.room_streams.items():
            rs.wake_detector.load.assert_called_once()


class TestStop:
    """Test stop() method."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() sets _running to False."""
        loop = _make_multi_room_loop()
        loop._running = True

        await loop.stop()

        assert loop._running is False
