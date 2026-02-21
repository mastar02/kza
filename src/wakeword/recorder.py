"""
Wake Word Recorder
Graba muestras de audio para entrenar un wake word personalizado.
"""

import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

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
        sample_duration: float = 2.0,  # segundos por muestra
    ):
        self.wake_word_name = wake_word_name.lower().replace(" ", "_")
        self.output_dir = Path(output_dir) / self.wake_word_name
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration

        # Crear directorios
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        self.positive_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)

    def record_positive_samples(
        self,
        n: int = 50,
        delay_between: float = 1.5,
        callback: Optional[callable] = None
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
        callback: Optional[callable] = None
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
        callback: Optional[callable]
    ) -> int:
        """Grabar n muestras de audio"""
        import sounddevice as sd

        recorded = 0
        existing = len(list(output_dir.glob("*.wav")))

        logger.info(f"Grabando {n} muestras {prefix} (existentes: {existing})")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Duración por muestra: {self.sample_duration}s")

        for i in range(n):
            try:
                # Countdown
                print(f"\n[{i+1}/{n}] {prompt}")
                print("Grabando en: ", end="", flush=True)
                for countdown in [3, 2, 1]:
                    print(f"{countdown}...", end="", flush=True)
                    time.sleep(0.5)
                print(" 🎤 ¡HABLA!")

                # Grabar
                samples = int(self.sample_rate * self.sample_duration)
                audio = sd.rec(
                    samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()

                # Verificar que hay audio (no silencio)
                audio_flat = audio.flatten()
                rms = np.sqrt(np.mean(audio_flat**2))

                if rms < 0.01:
                    print("⚠️ Audio muy bajo, intenta de nuevo")
                    continue

                # Guardar
                filename = f"{prefix}_{existing + recorded + 1:04d}.wav"
                filepath = output_dir / filename
                self._save_wav(filepath, audio_flat)

                print(f"✅ Guardado: {filename} (RMS: {rms:.3f})")
                recorded += 1

                if callback:
                    callback(recorded, n)

                # Pausa entre grabaciones
                if i < n - 1:
                    time.sleep(delay_between)

            except KeyboardInterrupt:
                logger.info("Grabación interrumpida por usuario")
                break
            except Exception as e:
                logger.error(f"Error grabando: {e}")

        logger.info(f"Grabación completada: {recorded}/{n} muestras")
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
