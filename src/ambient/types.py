"""DTOs del ambient path (spec 2026-06-06-transcripcion-continua-multipista).

AmbientUtterance es la unidad de salida del pipeline: un tramo de voz
transcripto y etiquetado. RawSegment es la unidad intermedia que emite el
segmentador (audio multicanal crudo, aún sin transcribir).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Valores cerrados de `source`. 'self' = capturado mientras nuestro TTS
# hablaba (no se destila a memoria — el asistente no se cita a sí mismo).
SOURCE_VALUES = {"live", "tv", "self", "unknown"}


# eq=False: el __eq__ generado compararía np.ndarray con == (devuelve array,
# bool() ambiguo → ValueError). Identidad de objeto alcanza para este DTO.
@dataclass(eq=False)
class RawSegment:
    """Segmento de voz detectado por el VAD, audio multicanal sin procesar."""

    t0: float                 # epoch s — inicio (incluye pre-roll)
    t1: float                 # epoch s — fin
    audio: np.ndarray         # shape (n_samples, n_channels) float32
    during_tts: bool = False  # algún chunk llegó con el TTS propio activo
    # Promedio de la prob de Silero sobre los chunks del segmento (sin
    # pre-roll, con cola de silencio). Señal de calidad anti-alucinación:
    # el no_speech_prob del turbo está degenerado (~1e-10 siempre, verificado
    # 2026-06-07); este mean SÍ discrimina voz near-field vs TV/silencio.
    vad_prob: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


@dataclass
class AmbientUtterance:
    """Utterance transcripta y etiquetada — fila de data/ambient.db."""

    room_id: str
    t0: float
    t1: float
    text: str = ""
    speaker: str = "unknown"
    speaker_confidence: float = 0.0
    azimuth: float | None = None        # rad relativo (DoA propio); None = no disponible
    azimuth_stability: float = 0.0      # 0-1, dispersión circular entre sub-ventanas
    source: str = "unknown"             # ver SOURCE_VALUES
    confidence: float | None = None     # avg_logprob del STT
    no_speech_prob: float | None = None  # ⚠️ degenerado con turbo (~1e-10 siempre)
    vad_prob: float | None = None       # mean Silero del segmento (señal real)
    during_tts: bool = False
    distilled: bool = False
