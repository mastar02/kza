"""Tests for CommandEvent dataclass."""
import sys
from unittest.mock import MagicMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import numpy as np
import time

from src.pipeline.command_event import CommandEvent


def test_command_event_creation():
    """CommandEvent carries audio data with room metadata."""
    audio = np.zeros(16000, dtype=np.float32)
    now = time.time()
    event = CommandEvent(
        audio=audio,
        room_id="cocina",
        mic_device_index=3,
        timestamp=now,
    )
    assert event.room_id == "cocina"
    assert event.mic_device_index == 3
    assert len(event.audio) == 16000
    assert event.timestamp == now


def test_command_event_defaults():
    """CommandEvent should work with minimal args."""
    audio = np.zeros(8000, dtype=np.float32)
    event = CommandEvent(audio=audio, room_id="living")
    assert event.room_id == "living"
    assert event.mic_device_index is None
    assert event.timestamp > 0
