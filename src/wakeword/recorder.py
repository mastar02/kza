"""
Wake Word Recorder
Graba muestras de audio para entrenar un wake word personalizado.
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from src.audio.native_capture_authority import WakewordTrainingAuthorityClient
from src.wakeword.live_capture import LiveWakewordCapture

logger = logging.getLogger(__name__)


class WakeWordRecorder:
    """
    Grabador de muestras para entrenar wake word personalizado.

    Uso:
        recorder = WakeWordRecorder("mi_wake_word")
        recorder.record_positive_samples(n=50)  # Grabaciones diciendo la palabra
        recorder.record_negative_samples(n=50)  # Conversación normal
    """

    def __init__(
        self,
        wake_word_name: str,
        output_dir: str = "./data/wakeword_training",
        sample_rate: int = 16000,
        sample_duration: float = 2.0,
        *,
        authority_factory: Callable[[], WakewordTrainingAuthorityClient] | None = None,
    ) -> None:
        self.wake_word_name = wake_word_name.lower().replace(" ", "_")
        self.output_dir = Path(output_dir) / self.wake_word_name
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self._authority_factory = authority_factory
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)

    def record_positive_samples(
        self,
        n: int = 50,
        delay_between: float = 1.5,
        callback: Callable | None = None
    ) -> int:
        """
        Grabar muestras positivas (diciendo el wake word).

        Args:
            n: Número de muestras a grabar
            delay_between: Segundos entre grabaciones
            callback: Función a llamar después de cada grabación (progress, total)

        Returns:
            Número de muestras grabadas exitosamente
        """
        return self._record_samples(
            output_dir=self.positive_dir,
            n=n,
            prefix="positive",
            prompt=f"Di '{self.wake_word_name.replace('_', ' ')}'",
            delay_between=delay_between,
            callback=callback
        )

    def record_negative_samples(
        self,
        n: int = 50,
        delay_between: float = 1.5,
        callback: Callable | None = None
    ) -> int:
        """
        Grabar muestras negativas (conversación normal SIN el wake word).

        Args:
            n: Número de muestras a grabar
            delay_between: Segundos entre grabaciones
            callback: Función a llamar después de cada grabación

        Returns:
            Número de muestras grabadas exitosamente
        """
        return self._record_samples(
            output_dir=self.negative_dir,
            n=n,
            prefix="negative",
            prompt="Di cualquier frase SIN la palabra de activación",
            delay_between=delay_between,
            callback=callback
        )

    def _record_samples(
        self,
        output_dir: Path,
        n: int,
        prefix: str,
        prompt: str,
        delay_between: float,
        callback: Callable | None,
    ) -> int:
        """Record samples under one explicit live authority per invocation."""
        if self._authority_factory is None:
            raise RuntimeError("wakeword_capture_authority_required")
        recorded = 0
        existing = len(list(output_dir.glob("*.wav")))
        with LiveWakewordCapture(self._authority_factory()) as live:
            import sounddevice as sd

            logger.info("Recording %s %s samples (existing: %s)", n, prefix, existing)
            for i in range(n):
                audio = None
                try:
                    live.publish(print, f"\n[{i+1}/{n}] {prompt}")
                    live.publish(print, "Grabando en: ", end="", flush=True)
                    for countdown in (3, 2, 1):
                        live.publish(print, f"{countdown}...", end="", flush=True)
                        live.pause(0.5)
                    live.publish(print, " 🎤 ¡HABLA!")
                    samples = int(self.sample_rate * self.sample_duration)
                    audio = np.empty((samples, 1), dtype=np.float32)
                    stream = live.open_stream(lambda: sd.InputStream(
                        samplerate=self.sample_rate, channels=1, dtype="float32",
                        blocksize=1280,
                    ))
                    offset = 0
                    while offset < samples:
                        live.require_current()
                        frames = min(1280, samples - offset)
                        block, _overflowed = stream.read(frames)
                        live.require_current()
                        if block.shape != (frames, 1):
                            raise RuntimeError("wakeword_capture_invalid_frames")
                        audio[offset:offset + frames] = block
                        offset += frames
                    live.close_stream()
                    live.require_current()
                    audio_flat = audio.flatten()
                    try:
                        rms = np.sqrt(np.mean(audio_flat**2))
                        if rms < 0.01:
                            live.publish(print, "⚠️ Audio muy bajo, intenta de nuevo")
                            continue
                        filename = f"{prefix}_{existing + recorded + 1:04d}.wav"
                        live.publish(self._save_wav, output_dir / filename, audio_flat)
                        live.publish(print, f"✅ Guardado: {filename} (RMS: {rms:.3f})")
                        recorded += 1
                        if callback:
                            live.publish(callback, recorded, n)
                    finally:
                        audio_flat.fill(0)
                    if i < n - 1:
                        live.pause(delay_between)
                except KeyboardInterrupt:
                    logger.info("Wakeword recording interrupted")
                    raise
                except Exception:
                    logger.exception("Wakeword recording failed")
                    raise
                finally:
                    if audio is not None:
                        audio.fill(0)
            live.require_current()
            logger.info("Recording completed: %s/%s samples", recorded, n)
        return recorded

    def _save_wav(self, filepath: Path, audio: np.ndarray):
        """Guardar audio como WAV"""
        # Convertir a int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def get_stats(self) -> dict:
        """Obtener estadísticas de muestras grabadas"""
        positive_count = len(list(self.positive_dir.glob("*.wav")))
        negative_count = len(list(self.negative_dir.glob("*.wav")))

        return {
            "wake_word": self.wake_word_name,
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "total_samples": positive_count + negative_count,
            "ready_for_training": positive_count >= 30 and negative_count >= 30,
            "positive_dir": str(self.positive_dir),
            "negative_dir": str(self.negative_dir)
        }

    def add_sample_from_audio(
        self,
        audio: np.ndarray,
        is_positive: bool
    ) -> str:
        """
        Agregar una muestra desde audio ya grabado.
        Útil para agregar muestras durante el uso normal.
        """
        output_dir = self.positive_dir if is_positive else self.negative_dir
        prefix = "positive" if is_positive else "negative"
        existing = len(list(output_dir.glob("*.wav")))

        filename = f"{prefix}_{existing + 1:04d}.wav"
        filepath = output_dir / filename

        # Asegurar formato correcto
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.flatten()

        self._save_wav(filepath, audio)
        return str(filepath)
