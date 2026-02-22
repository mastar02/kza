"""
Tests for VoicePipeline CommandEvent support.

Verifies that process_command accepts both CommandEvent (with room metadata)
and raw np.ndarray (backward compatibility).
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

from src.pipeline.command_event import CommandEvent
from src.pipeline.voice_pipeline import VoicePipeline


@pytest.fixture
def pipeline():
    """Create a VoicePipeline with mocked dependencies."""
    return VoicePipeline(
        audio_loop=MagicMock(),
        command_processor=MagicMock(),
        request_router=MagicMock(
            process_command=AsyncMock(return_value={"text": "ok", "success": True})
        ),
        response_handler=MagicMock(),
        feature_manager=MagicMock(),
    )


@pytest.mark.asyncio
async def test_process_command_accepts_command_event(pipeline):
    """process_command must accept CommandEvent and pass it to the router."""
    event = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="cocina",
        mic_device_index=3,
    )
    result = await pipeline.process_command(event)
    assert result["success"] is True
    pipeline.request_router.process_command.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_process_command_accepts_raw_audio(pipeline):
    """process_command must still accept raw np.ndarray for backward compat."""
    audio = np.zeros(16000, dtype=np.float32)
    result = await pipeline.process_command(audio)
    assert result["success"] is True
    pipeline.request_router.process_command.assert_called_once()


@pytest.mark.asyncio
async def test_process_command_no_router():
    """process_command returns error when no request_router is configured."""
    pipeline = VoicePipeline(
        audio_loop=MagicMock(),
        command_processor=MagicMock(),
        request_router=None,
        response_handler=MagicMock(),
        feature_manager=MagicMock(),
    )
    audio = np.zeros(16000, dtype=np.float32)
    result = await pipeline.process_command(audio)
    assert result["success"] is False
    assert "error" in result
