"""ParakeetSTT — NVIDIA Parakeet-TDT-0.6B-v3 vía onnx-asr, motor del ambient path.

Reemplaza al Whisper turbo en el AMBIENT path (swap 2026-06-07, benchmark A/B
con audio real XVF3800: 0/5 alucinaciones sobre no-voz vs 5/5 del turbo,
mejor calidad sobre voz, RTF ~0.03 en CPU). Razones de arquitectura:
- Transducer frame-synchronous: no "free-runea" texto como el decoder
  autoregresivo de Whisper → sin "Gracias por ver el video." sobre silencio
  ni repetition loops (entrenado con 36kh de no-voz/target-vacío,
  arxiv 2509.14128).
- Corre en CPU (onnxruntime): 0 VRAM — libera ~1.5GB de cuda:0 que usaba
  el Whisper ambiental.

El COMMAND path no se toca: sigue en faster-whisper turbo GPU (~150ms,
presupuesto <300ms). Doc: docs/research/2026-06-07_SOTA_ASR_ESPANOL_INVESTIGACION.md.

Duck-type de FastWhisperSTT para AmbientSTT: expone
``transcribe_with_confidence(audio) -> STTResult``. Parakeet no produce
no_speech_prob/avg_logprob/compression_ratio → None ('sin penalizar');
la señal de calidad del ambient path es ``vad_prob`` (Silero), no el STT.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


class ParakeetSTT:
    """ASR Parakeet-TDT vía onnx-asr (CPU), carga lazy del modelo."""

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        language: str = "es",
    ):
        """
        Args:
            model_name: Modelo del hub de onnx-asr (descarga/caché en
                ~/.cache/huggingface la primera vez).
            language: Idioma forzado para recognize() (Parakeet v3 es
                multilingüe con autodetección; fijarlo evita drift).
        """
        self.model_name = model_name
        self.language = language
        self._model = None

    def load(self) -> None:
        """Cargar el modelo onnx (idempotente)."""
        if self._model is not None:
            return
        import onnx_asr

        logger.info(f"Cargando Parakeet (onnx, CPU): {self.model_name}")
        start = time.time()
        self._model = onnx_asr.load_model(self.model_name)
        logger.info(f"Parakeet cargado en {time.time() - start:.1f}s")

    def transcribe_with_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribir audio mono → STTResult (señales Whisper en None).

        Args:
            audio: Array mono float32 [-1, 1] o int16/int32 (se normaliza).
            sample_rate: Solo 16000 (contrato del pipeline).

        Returns:
            STTResult con text y elapsed_ms; no_speech_prob/avg_logprob/
            compression_ratio en None (Parakeet no los expone).
        """
        if self._model is None:
            self.load()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.perf_counter()
        text = self._model.recognize(audio, language=self.language) or ""
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Parakeet STT ({elapsed_ms:.0f}ms): {text[:50]}...")

        return STTResult(text=text.strip(), elapsed_ms=elapsed_ms)
