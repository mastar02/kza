"""
StreamingWhisperWakeDetector — wake word word-synchronous.

Alternativa a WhisperWakeDetector que NO espera end-of-utterance. En lugar de
acumular hasta silencio + transcribir, mantiene un ring buffer de 2s y re-transcribe
cada `interval_ms` (default 200ms) cuando hay VAD activo. La palabra se detecta
~100-300ms después de pronunciada (vs ~800ms end-of-utterance del no-streaming).

Trade-off:
  - GPU:0 (Whisper) sostenido más alto (~1 inferencia activa mientras hay voz).
  - Mucho mejor UX. Fase 2 del roadmap.

Combinar con speaker filter (Fase 1) reduce GPU porque filter corta antes de Whisper.

API compatible con WhisperWakeDetector: detect / predict / get_active_models /
pop_pending_command_audio.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from src.wakeword.whisper_wake import _normalize

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_MS = 80


class StreamingWhisperWakeDetector:
    """
    Wake word detector con Whisper en modo streaming (word-synchronous trigger).

    Args:
        whisper_stt: FastWhisperSTT cargado.
        wake_words: lista de wake words.
        interval_ms: cada cuánto re-transcribe el buffer si VAD activo.
        window_s: tamaño del ring buffer (seg). 2.0 es suficiente para "nexa prendé".
        dedup_ms: no dispara dos triggers dentro de este window.
        vad_threshold / min_rms: gates previos al Whisper call.
        language: idioma Whisper.
        speaker_identifier / speaker_embedding / speaker_threshold / speaker_min_audio_s:
            si pasan, aplica speaker filter sobre el buffer acumulado antes de Whisper.
    """

    def __init__(
        self,
        whisper_stt,
        wake_words: list[str],
        interval_ms: int = 200,
        window_s: float = 2.0,
        dedup_ms: int = 2000,
        vad_threshold: float = 0.7,
        min_rms: float = 0.025,
        language: str = "es",
        speaker_identifier=None,
        speaker_embedding: Optional[np.ndarray] = None,
        speaker_threshold: float = 0.65,
        speaker_min_audio_s: float = 0.8,
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
    ):
        self.whisper = whisper_stt
        self.wake_words_norm = [_normalize(w) for w in wake_words]
        self.interval_ms = interval_ms
        self.window_s = window_s
        self.window_samples = int(window_s * SAMPLE_RATE)
        self.dedup_ms = dedup_ms
        self.vad_threshold = vad_threshold
        self.min_rms = min_rms
        self.language = language

        self.speaker_identifier = speaker_identifier
        self.speaker_embedding = speaker_embedding
        self.speaker_threshold = speaker_threshold
        self.speaker_min_audio_s = speaker_min_audio_s
        self._speaker_filter_active = (
            speaker_identifier is not None and speaker_embedding is not None
        )
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._vad = None
        self._torch = None
        self._loaded = False

        self._last_analyze_t = 0.0
        self._last_trigger_t = 0.0
        self._recent_vad_activity_t = 0.0  # último chunk con VAD speech
        self._pending_command_audio: Optional[np.ndarray] = None

    def load(self):
        if self._loaded:
            return
        logger.info("StreamingWhisperWakeDetector: cargando silero-vad...")
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad = model
            self._torch = torch
        except Exception as e:
            logger.error(f"No pude cargar silero-vad: {e}. Fallback por RMS.")
            self._vad = None
        self._loaded = True
        filter_str = (
            f" +speaker_filter(threshold={self.speaker_threshold})"
            if self._speaker_filter_active else ""
        )
        logger.info(
            f"StreamingWhisperWakeDetector listo. Wake words: {self.wake_words_norm} "
            f"(window={self.window_s}s, interval={self.interval_ms}ms{filter_str})"
        )

    def get_active_models(self) -> list[str]:
        return [f"streaming_whisper:{w}" for w in self.wake_words_norm]

    def _is_speech(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        if rms < self.min_rms:
            return False
        if self._vad is not None:
            try:
                tensor = self._torch.from_numpy(chunk.astype(np.float32))
                with self._torch.no_grad():
                    prob = float(self._vad(tensor, SAMPLE_RATE).item())
                return prob >= self.vad_threshold
            except Exception:
                pass
        return rms > self.min_rms

    def _push_chunk(self, chunk: np.ndarray) -> None:
        """Push chunk al ring buffer; mantiene tamaño máximo window_samples."""
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)
        while self._buffer_samples > self.window_samples and self._buffer:
            oldest = self._buffer.popleft()
            self._buffer_samples -= len(oldest)

    def _concat_buffer(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._buffer)).astype(np.float32)

    def _speaker_match(self, audio: np.ndarray) -> tuple[bool, float]:
        if not self._speaker_filter_active:
            return True, 1.0
        dur_s = len(audio) / SAMPLE_RATE
        if dur_s < self.speaker_min_audio_s:
            return True, 0.0
        try:
            emb = self.speaker_identifier.get_embedding(audio)
            sim = self.speaker_identifier.compute_similarity(
                emb, self.speaker_embedding,
            )
        except Exception as e:
            logger.warning(f"Speaker match falló ({e}); dejando pasar.")
            return True, 0.0
        return sim >= self.speaker_threshold, sim

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {w: 0.0 for w in self.wake_words_norm}
        match = self._process_chunk(audio_chunk)
        if match is not None:
            result[match] = 1.0
        return result

    def detect(self, audio_chunk: np.ndarray) -> Optional[tuple[str, float]]:
        match = self._process_chunk(audio_chunk)
        if match is not None:
            return (match, 1.0)
        return None

    def _process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        if not self._loaded:
            self.load()

        chunk = audio_chunk
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        max_abs = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        if max_abs > 1.5:
            chunk = chunk / 32768.0

        self._push_chunk(chunk)

        now = time.time()
        if self._is_speech(chunk):
            self._recent_vad_activity_t = now

        vad_recent = (now - self._recent_vad_activity_t) < 0.5  # ventana 500ms
        interval_elapsed = (now - self._last_analyze_t) * 1000 >= self.interval_ms

        if not (vad_recent and interval_elapsed):
            return None

        self._last_analyze_t = now

        # Dedup: no re-triggerear rápido
        if (now - self._last_trigger_t) * 1000 < self.dedup_ms:
            return None

        audio = self._concat_buffer()
        if len(audio) < int(0.4 * SAMPLE_RATE):  # <400ms no vale la pena
            return None

        if self._speaker_filter_active:
            passed, sim = self._speaker_match(audio)
            if not passed:
                logger.debug(
                    f"Streaming speaker filter REJECT (sim={sim:.3f}) — skip Whisper"
                )
                return None

        match = self._transcribe_and_match(audio, now)
        if match is not None:
            self._last_trigger_t = now
        return match

    def _transcribe_and_match(
        self, audio: np.ndarray, now_t: float,
    ) -> Optional[str]:
        t0 = time.time()
        try:
            model = getattr(self.whisper, "_model", None) or self.whisper
            segments, _ = model.transcribe(
                audio, language=self.language,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt,
                word_timestamps=True, vad_filter=False,
                condition_on_previous_text=False,
            )
            segs = list(segments)
        except Exception as e:
            logger.error(f"StreamingWhisperWake transcribe error: {e}")
            return None

        stt_ms = (time.time() - t0) * 1000
        full_text = " ".join(s.text for s in segs).strip()
        if not full_text:
            return None
        norm_full = _normalize(full_text)
        logger.debug(
            f"StreamingWake [{len(audio)/SAMPLE_RATE*1000:.0f}ms→{stt_ms:.0f}ms]: "
            f"{norm_full!r}"
        )

        audio_dur = len(audio) / SAMPLE_RATE
        recent_window_start = max(0.0, audio_dur - 0.8)

        for seg in segs:
            words = getattr(seg, "words", None) or []
            for w in words:
                word_norm = _normalize(w.word)
                for wake in self.wake_words_norm:
                    if wake in word_norm and w.start >= recent_window_start:
                        logger.info(
                            f"🔥 Wake word '{wake}' detectado en: {full_text!r} "
                            f"@ t={w.start:.2f}s (stt={stt_ms:.0f}ms)"
                        )
                        wake_end_sample = int(w.end * SAMPLE_RATE)
                        if wake_end_sample < len(audio):
                            self._pending_command_audio = audio[wake_end_sample:].copy()
                        else:
                            self._pending_command_audio = audio.copy()
                        return wake
        return None

    def pop_pending_command_audio(self) -> Optional[np.ndarray]:
        audio = self._pending_command_audio
        self._pending_command_audio = None
        return audio
