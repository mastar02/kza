"""UtteranceSegmenter — Silero VAD chunked sobre mic crudo → RawSegment.

El XVF3800 post-DSP rompe a Silero (prob~0, visto 2026-06-04 en whisper_fast);
por eso el VAD del ambient path corre sobre la COLUMNA CRUDA (vad_col, default
2 = mic 0 del firmware 6ch). Con firmware 2ch la columna no existe y se degrada
a col 0 (peor señal para VAD, pero funcional).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Callable

import numpy as np

from src.ambient.types import RawSegment

logger = logging.getLogger(__name__)


def make_silero_predictor() -> Callable[[np.ndarray], float]:
    """Factory del predictor real (Silero VAD vía torch.hub, patrón
    whisper_wake.py). Devuelve prob máxima entre las ventanas de 512 samples
    del chunk. Carga el modelo UNA vez al construirse.
    """
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)

    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True,
    )
    model.eval()

    def predict(chunk_mono: np.ndarray) -> float:
        if chunk_mono.dtype != np.float32:
            chunk_mono = chunk_mono.astype(np.float32)
        probs = []
        for i in range(0, len(chunk_mono) - 511, 512):
            win = torch.from_numpy(chunk_mono[i : i + 512])
            with torch.no_grad():
                probs.append(float(model(win, 16000).item()))
        if not probs:
            # Chunk < 512 samples: sin ventana completa no hay predicción —
            # se trata como silencio. No ocurre con CHUNK_SIZE=1280; avisar
            # si un caller futuro rompe ese contrato.
            logger.warning(
                f"Silero predict: chunk de {len(chunk_mono)} samples (<512) "
                f"— tratado como silencio"
            )
            return 0.0
        return max(probs)

    # Silero es stateful (GRU): la doc oficial recomienda reset en límites
    # de utterance. El segmenter lo invoca en _close() vía getattr (opcional
    # — los predictores fake de los tests no lo exponen).
    predict.reset = model.reset_states
    return predict


class UtteranceSegmenter:
    """Máquina de estados: silencio → voz → cola de silencio → RawSegment."""

    def __init__(
        self,
        vad_predict: Callable[[np.ndarray], float],
        sample_rate: int = 16000,
        vad_col: int = 2,
        speech_threshold: float = 0.5,
        close_silence_ms: int = 700,
        preroll_ms: int = 500,
        max_segment_s: float = 30.0,
        min_speech_ms: int = 300,
    ):
        self._vad_predict = vad_predict
        self.sample_rate = sample_rate
        self.vad_col = vad_col
        self.speech_threshold = speech_threshold
        self.close_silence_ms = close_silence_ms
        self.max_segment_s = max_segment_s
        self.min_speech_ms = min_speech_ms

        chunk_ms = 80.0  # CHUNK_SIZE=1280 @ 16kHz — mismo del pipeline
        self._preroll: deque[tuple[float, np.ndarray, bool]] = deque(
            maxlen=max(0, int(round(preroll_ms / chunk_ms)))
        )
        self._in_speech = False
        self._buf: list[tuple[float, np.ndarray, bool]] = []
        self._probs: list[float] = []  # prob Silero por chunk del buffer
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._col_warned = False

    def _vad_column(self, chunk: np.ndarray) -> np.ndarray:
        col = self.vad_col
        if chunk.ndim == 1:
            return chunk
        if chunk.shape[1] <= col:
            if not self._col_warned:
                self._col_warned = True
                logger.warning(
                    f"vad_col={col} no existe (device de {chunk.shape[1]}ch) "
                    f"— fallback a col 0 (sin mic crudo, VAD degradado)"
                )
            col = 0
        return chunk[:, col]

    def feed(
        self, ts: float, chunk: np.ndarray, tts_active: bool = False
    ) -> list[RawSegment]:
        """Procesar un chunk; devolver los segmentos cerrados (0 o 1)."""
        chunk_ms = (chunk.shape[0] / self.sample_rate) * 1000
        prob = self._vad_predict(self._vad_column(chunk))
        is_speech = prob >= self.speech_threshold
        closed: list[RawSegment] = []

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._buf = list(self._preroll)
                self._buf.append((ts, chunk, tts_active))
                # vad_prob promedia los chunks evaluados in_speech; el
                # pre-roll (silencio previo) queda fuera a propósito.
                self._probs = [prob]
                self._preroll.clear()
                self._silence_ms = 0.0
                self._speech_ms = chunk_ms
            else:
                if self._preroll.maxlen:
                    self._preroll.append((ts, chunk, tts_active))
            return closed

        # in_speech
        self._buf.append((ts, chunk, tts_active))
        self._probs.append(prob)
        if is_speech:
            self._silence_ms = 0.0
            self._speech_ms += chunk_ms
        else:
            self._silence_ms += chunk_ms

        dur_s = sum(c.shape[0] for _, c, _ in self._buf) / self.sample_rate
        if self._silence_ms >= self.close_silence_ms or dur_s >= self.max_segment_s:
            seg = self._close()
            if seg is not None:
                closed.append(seg)
        return closed

    def _close(self) -> RawSegment | None:
        buf, self._buf = self._buf, []
        probs, self._probs = self._probs, []
        self._in_speech = False
        self._silence_ms = 0.0
        speech_ms, self._speech_ms = self._speech_ms, 0.0
        # Silero stateful: reset de la GRU en el límite de utterance (doc
        # oficial). Opcional vía getattr — los predictores fake no lo exponen.
        reset_fn = getattr(self._vad_predict, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception as e:
                logger.debug(f"Silero reset_states no-op: {e}")
        if not buf or speech_ms < self.min_speech_ms:
            return None  # blip: menos voz que min_speech_ms → descartar
        t0 = buf[0][0]
        last_ts, last_chunk, _ = buf[-1]
        t1 = last_ts + last_chunk.shape[0] / self.sample_rate
        audio = np.concatenate([c for _, c, _ in buf], axis=0)
        during_tts = any(t for _, _, t in buf)
        vad_prob = float(sum(probs) / len(probs)) if probs else 0.0
        return RawSegment(
            t0=t0, t1=t1, audio=audio, during_tts=during_tts, vad_prob=vad_prob
        )
