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
    _resolve_capture_channels,
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
            dedup_window_ms=500,
        )

        assert loop.room_streams is rooms
        assert loop.follow_up is follow_up
        assert loop.sample_rate == 16000
        assert loop.dedup_window_ms == 500
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
        assert loop.dedup_window_ms == 500


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


class TestMinWakeRmsGate:
    """Pre-gate de RMS post-wake (2026-06-02): rechaza activaciones de muy baja
    energía (near-silence) antes de transcribir. Default 0.0 = desactivado (no
    regresión); se calibra en repro porque el AGC ×64 infla el piso de ruido."""

    def test_rms_below_min_wake_rms_rejected(self):
        loop = _make_multi_room_loop(min_wake_rms=0.02)
        assert loop._should_accept_wakeword("cocina", rms=0.005, timestamp=1.0) is False

    def test_rms_at_or_above_min_wake_rms_accepted(self):
        loop = _make_multi_room_loop(min_wake_rms=0.02)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=2.0) is True

    def test_min_wake_rms_zero_disables_gate(self):
        # Default 0.0 → gate off → comportamiento idéntico al baseline (dedup).
        loop = _make_multi_room_loop(min_wake_rms=0.0)
        assert loop._should_accept_wakeword("cocina", rms=0.0001, timestamp=3.0) is True


def _detector_seq(detect_returns):
    """Wake detector mock SIN inline audio (simula el path openwakeword)."""
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(side_effect=list(detect_returns))
    m.get_active_models = MagicMock(return_value=[])
    # openwakeword no tiene estos métodos (son del WhisperWake) -> None para que
    # getattr(...) los saltee y se ejecute el path acústico.
    m.peek_pending_text = None
    m.pop_pending_command_audio = None
    m.pop_pending_text = None
    return m


class TestWakePreroll:
    """Pre-roll (2026-06-02): al disparar el wake, sembrar el buffer con el audio
    previo para no perder el comando dicho durante la latencia de openwakeword
    ('Nexa apagá la luz' -> 'apagá' se decía mientras el detector aún procesaba).
    """

    def _loop_with_room(self, detector, **kwargs):
        rs = RoomStream(
            room_id="escritorio", device_index=0,
            wake_detector=detector, echo_suppressor=_make_echo_suppressor(),
        )
        loop = _make_multi_room_loop(rooms={"escritorio": rs}, **kwargs)
        return loop, rs

    def test_preroll_seeds_command_buffer_on_wake(self):
        det = _detector_seq([None, None, None, ("nexa", 0.8)])
        loop, rs = self._loop_with_room(det, wake_preroll_s=0.24)  # ~3 chunks @ 80ms
        cb = loop._make_audio_callback(rs)
        for i in range(3):
            cb(np.full((CHUNK_SIZE, 2), 0.01 * (i + 1), dtype=np.float32), CHUNK_SIZE, None, None)
            assert rs.listening is False
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True
        # el buffer arranca con el pre-roll (≥3 chunks) en vez de vacío
        assert len(rs.audio_buffer) >= 3 * CHUNK_SIZE

    def test_preroll_off_keeps_empty_buffer(self):
        det = _detector_seq([None, None, ("nexa", 0.8)])
        loop, rs = self._loop_with_room(det, wake_preroll_s=0.0)  # default = off
        cb = loop._make_audio_callback(rs)
        for _ in range(2):
            cb(np.full((CHUNK_SIZE, 2), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True
        assert len(rs.audio_buffer) == 0  # sin pre-roll = comportamiento actual


class _FakeXvf:
    """XvfController falso: peak_since devuelve un valor fijo (o None)."""
    def __init__(self, peak):
        self._peak = peak
        self.started = False
    def start(self):
        self.started = True
        return True
    def stop(self):
        pass
    def peak_since(self, since_ts):
        return self._peak


class TestSpenergyGate:
    """Pre-gate SPENERGY (2026-06-02): no transcribir si el pico de SPENERGY
    durante la captura < umbral (secador/silencio → alucinación de Whisper).
    Fail-open: sin controller o sin datos → procesa."""

    def _rs(self):
        rs = _make_room_stream("escritorio")
        rs.command_start_time = 100.0
        return rs

    def test_no_controller_passes(self):
        loop = _make_multi_room_loop()  # xvf_controller None por defecto
        assert loop._passes_spenergy_gate(self._rs()) is True

    def test_peak_none_fail_open_passes(self):
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(None), spenergy_threshold=100.0)
        assert loop._passes_spenergy_gate(self._rs()) is True

    def test_low_peak_blocks(self):
        # secador/silencio = 0 < 100 → descarta
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(0.0), spenergy_threshold=100.0)
        assert loop._passes_spenergy_gate(self._rs()) is False

    def test_voice_peak_passes(self):
        # voz medida ~335k ≥ 100 → procesa
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(335000.0), spenergy_threshold=100.0)
        assert loop._passes_spenergy_gate(self._rs()) is True


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


class TestResolveCapturChannels:
    """Test _resolve_capture_channels pure function."""

    @pytest.mark.parametrize("reported,expected", [
        (0, 1),
        (1, 1),
        (2, 2),
        (6, 6),
    ])
    def test_resolve_capture_channels(self, reported, expected):
        assert _resolve_capture_channels(reported) == expected
