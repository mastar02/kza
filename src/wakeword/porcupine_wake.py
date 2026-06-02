"""Detector wake-word basado en Picovoice Porcupine.

Wake-word dedicado siempre-on para "Nexa" (modelo entrenado en la consola de
Picovoice, idioma español). Reemplaza el stopgap "Whisper-como-wake-word" que
alucinaba sobre silencio. Corre en CPU, no consume GPU.

Interfaz compatible con `multi_room_audio_loop`: expone `detect(chunk)` que
devuelve `(wake_word, score) | None`, igual que `WhisperWakeDetector` y
`WakeWordDetector`. No provee texto pretranscripto (el comando se transcribe
downstream con `FastWhisperSTT` sobre el clip grabado).

Porcupine procesa frames de EXACTAMENTE `frame_length` (512) muestras int16 a
16 kHz. El audio loop entrega chunks float32 [-1,1] de tamaño distinto (1280),
por eso acumulamos en un buffer interno y procesamos de a `frame_length`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PorcupineWakeDetector:
    """Wake-word detector que envuelve Picovoice Porcupine."""

    def __init__(
        self,
        access_key: str,
        keyword_path: str,
        model_path: Optional[str] = None,
        sensitivity: float = 0.5,
        wake_word: str = "nexa",
        refractory_s: float = 1.5,
    ):
        """Configurar el detector.

        Args:
            access_key: AccessKey de Picovoice Console (vía env var).
            keyword_path: Ruta al `.ppn` custom (modelo "Nexa" en español).
            model_path: Ruta al modelo de idioma (`porcupine_params_es.pv`);
                None usa el inglés bundled.
            sensitivity: 0-1, mayor = más sensible (más FP, menos misses).
            wake_word: Etiqueta a devolver en la detección.
            refractory_s: Segundos mínimos entre detecciones (anti-doble-trigger).
        """
        self._access_key = access_key
        self._keyword_path = keyword_path
        self._model_path = model_path
        self._sensitivity = float(sensitivity)
        self._wake_word = wake_word
        self._refractory_s = refractory_s
        self._handle = None
        self._frame_length = 512
        self._buf = np.empty(0, dtype=np.int16)
        self._last_detect = 0.0
        self._loaded = False

    def load(self) -> None:
        """Crear el handle de Porcupine (lazy o explícito)."""
        if self._loaded:
            return
        if not self._access_key:
            raise ValueError(
                "PorcupineWakeDetector: access_key vacío. Definir "
                "PORCUPINE_ACCESS_KEY en /home/kza/secrets/.env."
            )
        if not Path(self._keyword_path).exists():
            raise FileNotFoundError(
                f"PorcupineWakeDetector: keyword .ppn no encontrado: {self._keyword_path}"
            )
        import pvporcupine

        kwargs = {
            "access_key": self._access_key,
            "keyword_paths": [self._keyword_path],
            "sensitivities": [self._sensitivity],
        }
        if self._model_path:
            kwargs["model_path"] = self._model_path
        self._handle = pvporcupine.create(**kwargs)
        self._frame_length = self._handle.frame_length
        sr = self._handle.sample_rate
        if sr != 16000:
            logger.warning(
                f"PorcupineWakeDetector: sample_rate del handle={sr} (esperado 16000)"
            )
        self._loaded = True
        logger.info(
            f"Porcupine cargado: wake='{self._wake_word}' "
            f"keyword={Path(self._keyword_path).name} "
            f"model={Path(self._model_path).name if self._model_path else 'en(default)'} "
            f"frame_length={self._frame_length} sensitivity={self._sensitivity}"
        )

    def _to_int16(self, chunk: np.ndarray) -> np.ndarray:
        """Convertir el chunk a int16 (Porcupine espera PCM int16)."""
        if chunk.dtype == np.int16:
            return chunk
        c = chunk.astype(np.float32)
        # sounddevice/faster-whisper entregan float32 normalizado [-1,1].
        if c.size and float(np.max(np.abs(c))) <= 1.5:
            c = np.clip(c, -1.0, 1.0) * 32767.0
        return c.astype(np.int16)

    def detect(self, audio_chunk: np.ndarray) -> Optional[tuple[str, float]]:
        """Procesar un chunk y devolver (wake_word, score) si dispara.

        Acumula el audio entrante en un buffer y procesa de a `frame_length`
        muestras (lo que requiere Porcupine). Porcupine no expone un score
        continuo (solo índice del keyword), así que devolvemos score=1.0.
        """
        if not self._loaded:
            self.load()

        frame = self._to_int16(np.asarray(audio_chunk).reshape(-1))
        self._buf = np.concatenate([self._buf, frame]) if self._buf.size else frame

        detected = False
        while self._buf.size >= self._frame_length:
            f = self._buf[: self._frame_length]
            self._buf = self._buf[self._frame_length:]
            if self._handle.process(f) >= 0:
                detected = True

        if detected:
            now = time.time()
            if now - self._last_detect >= self._refractory_s:
                self._last_detect = now
                logger.info(f"Porcupine wake detectado: '{self._wake_word}'")
                return (self._wake_word, 1.0)
        return None

    def reset(self) -> None:
        """Limpiar el buffer de alineación de frames."""
        self._buf = np.empty(0, dtype=np.int16)

    def delete(self) -> None:
        """Liberar recursos nativos de Porcupine."""
        if self._handle is not None:
            try:
                self._handle.delete()
            finally:
                self._handle = None
                self._loaded = False
