"""
Command Event — carries captured audio with room metadata.

Emitted by MultiRoomAudioLoop when a room's microphone completes
command capture after wake word detection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.nlu.command_grammar import PartialCommand


@dataclass
class CommandEvent:
    """Audio command captured from a specific room's microphone."""
    audio: np.ndarray
    room_id: str
    mic_device_index: int | None = None
    timestamp: float = field(default_factory=time.time)
    # Early dispatch: si el parser streaming extrajo intent+entity antes del
    # silencio final, viene aquí. El RequestRouter puede usarlo para saltear
    # el STT (ya está el texto) o cruzarlo con el STT canónico como fallback.
    partial_command: "PartialCommand | None" = None
    early_dispatch: bool = False
