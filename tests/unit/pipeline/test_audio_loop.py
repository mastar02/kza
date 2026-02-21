"""
Tests for AudioLoop - audio capture loop extracted from VoicePipeline.

Tests ensure that:
1. Initialization works with all deps and with minimal deps
2. Callback registration works for on_command and on_post_command
3. start() loads wake word and initializes ambient detector
4. stop() sets _running=False and stops ambient detector
5. _handle_task_error logs exceptions correctly
6. _handle_task_error ignores cancelled tasks
"""

import sys
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest

from src.pipeline.audio_loop import AudioLoop


# ============================================================
# Helpers
# ============================================================

def _make_audio_manager():
    """Create a mock AudioManager."""
    m = MagicMock()
    m.wake_word_model = "hey_jarvis"
    m.load_wake_word = MagicMock()
    m.detect_wake_word = MagicMock(return_value=False)
    m.capture_command_with_vad = MagicMock(return_value=(False, 0, None, False))
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
    m.on_user_speech = MagicMock()
    m.on_kza_response = MagicMock()
    m.on_kza_finished_speaking = MagicMock()
    return m


def _make_ambient_detector():
    """Create a mock AudioEventDetector."""
    m = MagicMock()
    m.initialize = AsyncMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.analyze_chunk = AsyncMock(return_value=[])
    return m


def _make_audio_loop(ambient_detector=None):
    """Create an AudioLoop with mock dependencies."""
    return AudioLoop(
        audio_manager=_make_audio_manager(),
        echo_suppressor=_make_echo_suppressor(),
        follow_up=_make_follow_up(),
        sample_rate=16000,
        ambient_detector=ambient_detector,
    )


# ============================================================
# Tests
# ============================================================

class TestAudioLoopInit:
    """Test AudioLoop initialization."""

    def test_audio_loop_init(self):
        """Verify all attributes are set correctly."""
        am = _make_audio_manager()
        es = _make_echo_suppressor()
        fu = _make_follow_up()
        ad = _make_ambient_detector()

        loop = AudioLoop(
            audio_manager=am,
            echo_suppressor=es,
            follow_up=fu,
            sample_rate=22050,
            ambient_detector=ad,
        )

        assert loop.audio_manager is am
        assert loop.echo_suppressor is es
        assert loop.follow_up is fu
        assert loop.sample_rate == 22050
        assert loop.ambient_detector is ad
        assert loop._running is False
        assert loop._on_command_callback is None
        assert loop._on_post_command_callback is None

    def test_audio_loop_init_minimal(self):
        """Verify init works without ambient detector."""
        loop = _make_audio_loop()

        assert loop.ambient_detector is None
        assert loop._running is False

    def test_audio_loop_default_sample_rate(self):
        """Verify default sample rate is 16000."""
        loop = AudioLoop(
            audio_manager=_make_audio_manager(),
            echo_suppressor=_make_echo_suppressor(),
            follow_up=_make_follow_up(),
        )
        assert loop.sample_rate == 16000


class TestAudioLoopCallbacks:
    """Test callback registration."""

    def test_audio_loop_registers_on_command(self):
        """on_command stores the callback."""
        loop = _make_audio_loop()

        async def my_callback(audio_data):
            return {"text": "test"}

        loop.on_command(my_callback)
        assert loop._on_command_callback is my_callback

    def test_audio_loop_registers_on_post_command(self):
        """on_post_command stores the callback."""
        loop = _make_audio_loop()

        async def my_post_callback(result, audio_data):
            pass

        loop.on_post_command(my_post_callback)
        assert loop._on_post_command_callback is my_post_callback

    def test_audio_loop_registers_both_callbacks(self):
        """Both callbacks can be registered independently."""
        loop = _make_audio_loop()

        async def cmd_cb(audio_data):
            return {}

        async def post_cb(result, audio_data):
            pass

        loop.on_command(cmd_cb)
        loop.on_post_command(post_cb)

        assert loop._on_command_callback is cmd_cb
        assert loop._on_post_command_callback is post_cb


class TestAudioLoopStart:
    """Test start() method."""

    @pytest.mark.asyncio
    async def test_audio_loop_start_loads_wake_word(self):
        """start() calls audio_manager.load_wake_word."""
        loop = _make_audio_loop()
        await loop.start()
        loop.audio_manager.load_wake_word.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_loop_start_initializes_ambient(self):
        """start() calls ambient_detector.initialize and start."""
        ad = _make_ambient_detector()
        loop = _make_audio_loop(ambient_detector=ad)

        await loop.start()

        ad.initialize.assert_awaited_once()
        ad.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audio_loop_start_no_ambient(self):
        """start() works fine without ambient detector."""
        loop = _make_audio_loop()
        await loop.start()
        # No error, wake word still loaded
        loop.audio_manager.load_wake_word.assert_called_once()


class TestAudioLoopStop:
    """Test stop() method."""

    @pytest.mark.asyncio
    async def test_audio_loop_stop(self):
        """stop() sets _running=False and stops ambient detector."""
        ad = _make_ambient_detector()
        loop = _make_audio_loop(ambient_detector=ad)
        loop._running = True

        await loop.stop()

        assert loop._running is False
        ad.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audio_loop_stop_no_ambient(self):
        """stop() works without ambient detector."""
        loop = _make_audio_loop()
        loop._running = True

        await loop.stop()

        assert loop._running is False

    @pytest.mark.asyncio
    async def test_audio_loop_stop_idempotent(self):
        """stop() can be called even if not running."""
        loop = _make_audio_loop()
        assert loop._running is False

        await loop.stop()

        assert loop._running is False


class TestHandleTaskError:
    """Test _handle_task_error static method."""

    def test_handle_task_error_logs(self, caplog):
        """Verify error callback logs exceptions from failed tasks."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            AudioLoop._handle_task_error(task)

        assert "Background audio task failed" in caplog.text
        assert "boom" in caplog.text

    def test_handle_task_error_ignores_cancelled(self, caplog):
        """No log for cancelled tasks."""
        task = MagicMock()
        task.cancelled.return_value = True

        with caplog.at_level(logging.ERROR):
            AudioLoop._handle_task_error(task)

        assert "Background audio task failed" not in caplog.text

    def test_handle_task_error_ignores_successful(self, caplog):
        """No log for successful tasks (no exception)."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None

        with caplog.at_level(logging.ERROR):
            AudioLoop._handle_task_error(task)

        assert "Background audio task failed" not in caplog.text
