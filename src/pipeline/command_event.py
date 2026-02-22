"""
Command Event — carries captured audio with room metadata.

Emitted by MultiRoomAudioLoop when a room's microphone completes
command capture after wake word detection.
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CommandEvent:
    """Audio command captured from a specific room's microphone."""
    audio: np.ndarray
    room_id: str
    mic_device_index: int | None = None
    timestamp: float = field(default_factory=time.time)
