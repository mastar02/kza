"""
StreamingSTT — captura audio continua con VAD dinámico.

Flujo:
  1. Abre stream del mic (sounddevice) en callback continuo.
  2. Chunks de ~30ms pasan por silero-vad → speech/silence.
  3. Al detectar inicio de speech empieza a acumular en ring buffer.
  4. Cuando hay 800ms continuos de silencio post-speech → transcribir buffer completo con faster-whisper-v3-turbo.
  5. Callback on_transcript(text, latency_ms) entregado al consumer.

Ventajas vs press-to-talk:
- No hay ventana fija: graba lo que el usuario dice, ni más ni menos.
- Latencia percibida end-of-utterance → texto final ≈ 150-250ms con turbo.
- Ignora silencio automáticamente, menos trabajo para Whisper.

Siguiente iteración (fuera de MVP): transcripción parcial durante speech,
streaming tokens, cross-fade con TTS.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_MS = 32  # silero-vad acepta 256 samples @ 16kHz = 16ms, o 512 = 32ms
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512


class VADGate:
    """silero-vad wrapper con umbral y cooldowns configurables."""

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        import torch
        # silero-vad vía torch.hub (se cachea local la primera vez)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model.to(device)
        self._device = device
        self._threshold = threshold
        self._torch = torch

    def is_speech(self, chunk_float32: np.ndarray) -> float:
        """Probabilidad de speech (0-1) para un chunk."""
        tensor = self._torch.from_numpy(chunk_float32).to(self._device)
        with self._torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()
        return prob

    @property
    def threshold(self) -> float:
        return self._threshold


class StreamingSTT:
    """Captura continua + VAD + Whisper transcribe on utterance end."""

    def __init__(
        self,
        whisper_model,  # faster_whisper.WhisperModel ya cargado
        mic_device: int = 9,
        silence_ms_end: int = 800,
        max_utterance_s: float = 12.0,
        min_utterance_ms: int = 200,
        pre_roll_ms: int = 200,
        vad_threshold: float = 0.5,
        language: str = "es",
    ):
        self.whisper = whisper_model
        self.mic_device = mic_device
        self.silence_ms_end = silence_ms_end
        self.max_utterance_s = max_utterance_s
        self.min_utterance_ms = min_utterance_ms
        self.pre_roll_ms = pre_roll_ms
        self.vad_threshold = vad_threshold
        self.language = language

        self._vad: VADGate | None = None
        self._stream = None
        self._running = False
        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._worker: threading.Thread | None = None
        self._on_transcript: Callable[[str, float], None] | None = None

    def start(self, on_transcript: Callable[[str, float], None]):
        """Arrancar captura + procesamiento. on_transcript invocado al cerrar cada utterance."""
        import sounddevice as sd

        if self._running:
            raise RuntimeError("StreamingSTT ya está corriendo")
        if self._vad is None:
            logger.info("Cargando silero-vad...")
            self._vad = VADGate(threshold=self.vad_threshold)

        self._on_transcript = on_transcript
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"sounddevice status: {status}")
            # ReSpeaker v2.0 devuelve 6 canales; ch0 = processed/beamformed
            mono = indata[:, 0].astype(np.float32).copy()
            try:
                self._chunk_queue.put_nowait(mono)
            except queue.Full:
                logger.warning("chunk_queue lleno — descartando")

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            channels=6,
            device=self.mic_device,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        logger.info(f"StreamingSTT iniciado (mic={self.mic_device}, silence_end={self.silence_ms_end}ms)")

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        logger.info("StreamingSTT detenido")

    def _process_loop(self):
        """Consume chunks, ejecuta VAD, detecta inicio/fin de utterance, transcribe."""
        pre_roll_chunks = max(1, self.pre_roll_ms // CHUNK_MS)
        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)

        in_speech = False
        utterance_chunks: list[np.ndarray] = []
        silence_run_ms = 0
        utterance_start_t = 0.0

        while self._running:
            try:
                chunk = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                prob = self._vad.is_speech(chunk)
            except Exception as e:
                logger.error(f"VAD error: {e}")
                continue

            is_speech = prob >= self._vad.threshold

            if not in_speech:
                pre_roll.append(chunk)
                if is_speech:
                    # Start utterance — include pre-roll
                    in_speech = True
                    utterance_start_t = time.time()
                    utterance_chunks = list(pre_roll)
                    utterance_chunks.append(chunk)
                    silence_run_ms = 0
                    logger.debug("VAD: utterance start")
            else:
                utterance_chunks.append(chunk)
                if is_speech:
                    silence_run_ms = 0
                else:
                    silence_run_ms += CHUNK_MS

                utt_dur_s = time.time() - utterance_start_t

                if silence_run_ms >= self.silence_ms_end or utt_dur_s >= self.max_utterance_s:
                    # End of utterance — transcribe
                    audio = np.concatenate(utterance_chunks)
                    if len(audio) * 1000 / SAMPLE_RATE < self.min_utterance_ms:
                        logger.debug(f"VAD: utterance demasiado corta ({len(audio)/SAMPLE_RATE*1000:.0f}ms), descartada")
                    else:
                        self._transcribe_and_emit(audio)

                    in_speech = False
                    utterance_chunks = []
                    silence_run_ms = 0
                    pre_roll.clear()

    def _transcribe_and_emit(self, audio: np.ndarray):
        t0 = time.time()
        try:
            segments, _info = self.whisper.transcribe(
                audio, language=self.language, beam_size=1, vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return
        elapsed_ms = (time.time() - t0) * 1000
        if text and self._on_transcript:
            self._on_transcript(text, elapsed_ms)
