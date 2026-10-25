"""
Command Event — carries captured audio with room metadata.

Emitted by MultiRoomAudioLoop when a room's microphone completes
command capture after wake word detection.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.nlu.command_grammar import PartialCommand


class TranscriptSource(StrEnum):
    """Producer of a reusable transcription of one audio snapshot."""

    WAKE = "wake"
    PARTIAL = "partial"
    AMBIENT = "ambient"


@dataclass(frozen=True)
class CapturedTranscript:
    """Bind text to the exact audio it covers, without retaining another clip."""

    text: str
    source: TranscriptSource
    samples: int
    audio_digest: str

    @classmethod
    def from_audio(cls, text: str, source: TranscriptSource, audio: np.ndarray) -> CapturedTranscript:
        """Bind a producer's text to its mono float32 snapshot."""
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("A transcript must cover mono audio")
        return cls(text, source, len(samples), hashlib.sha256(samples.tobytes()).hexdigest())

    def covers(self, audio: np.ndarray) -> bool:
        """Return whether the final audio is exactly the transcribed snapshot."""
        samples = np.asarray(audio, dtype=np.float32)
        return (
            samples.ndim == 1 and len(samples) == self.samples
            and hashlib.sha256(samples.tobytes()).hexdigest() == self.audio_digest
        )


@dataclass
class CommandEvent:
    """Audio command captured from a specific room's microphone."""

    audio: np.ndarray
    room_id: str
    mic_device_index: int | None = None
    timestamp: float = field(default_factory=time.time)
    # Parser hints are diagnostic; only a matching transcript snapshot can
    # replace final STT, after the producer has completed capture.
    partial_command: "PartialCommand | None" = None
    early_dispatch: bool = False
    # Wake text alone does not prove coverage of later captured audio.
    wake_text: str | None = None
    transcript: CapturedTranscript | None = None
    # None supports existing complete-audio callers. Early events require an
    # explicit True; False means a timeout or another incomplete capture.
    capture_complete: bool | None = None
    # AmbientGuard (spec 2026-06-05): True cuando la room estaba en STRICT al
    # despachar. El RequestRouter lo usa para NO otorgar el bonus de
    # wake_acoustically_confirmed al grammar fast-path — con TV de fondo el
    # wake espurio invalida la premisa "wake ⇒ usuario".
    ambient_strict: bool = False
    # Score del wake (openwakeword) que abrió esta captura. Lo usa el earcon
    # gate del router para decidir "humano plausible" (wake fuerte). Default
    # 1.0 para llamadas que no vienen del wake loop (tests, raw ndarray).
    wake_score: float = 1.0
    # True si el disparo vino del wake textual (ambient), no de OWW/Porcupine.
    # El router NO toma grammar fast-path: el texto ya contiene "nexa" (así
    # matcheó) y un verbo+luz alucinado ejecutaba luces fantasma.
    textual_wake: bool = False
