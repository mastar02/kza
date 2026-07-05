"""
WakeClipWriter — persiste el audio de cada wake aceptado para entrenamiento.

Cada disparo del wake word guarda su clip (el preroll que sembró el buffer,
~1s) como WAV 16kHz mono int16 en un directorio con rotación. El objetivo es
construir el dataset de re-entrenamiento de nexa.onnx con datos REALES:
- Falsos wakes (TV: "gracias por ver el video") → hard negatives con la
  distribución exacta que engaña al modelo actual.
- Wakes de comandos aceptados → positivos far-field en condición real.

Etiquetado posterior: cruzar el timestamp del nombre de archivo con la
transcripción del journal (CommandProcessor Text=...).

Restricción de diseño: submit() se llama desde el AUDIO CALLBACK THREAD de
sounddevice → jamás bloquea ni lanza (cola bounded + worker thread propio;
si la cola está llena, el clip se descarta en silencio). Lección 2026-06-12:
cualquier bloqueo en ese thread congela la captura de todo el pipeline.
"""

import logging
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class WakeClipWriter:
    """Escribe clips de wake a disco desde un worker thread, sin bloquear."""

    def __init__(
        self,
        directory: str | Path,
        sample_rate: int = 16000,
        max_files: int = 2000,
        max_rejected_files: int = 4000,
        queue_size: int = 8,
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._max_files = max_files
        # Los rechazados (TV STRICT/COOLDOWN + comandos far-field que el guard
        # mató) son mucho más voluminosos → tope propio en rejected/, rota
        # independiente para no purgar los aceptados (raros y valiosos).
        self._max_rejected_files = max_rejected_files
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = True
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="wake-clip-writer"
        )
        self._worker.start()

    def submit(self, room_id: str, score: float, audio, accepted: bool = True) -> bool:
        """Encolar un clip para escritura. Nunca bloquea ni lanza.

        Args:
            room_id: Habitación que disparó el wake.
            score: Score del detector (va al nombre del archivo).
            audio: Samples float32 en [-1, 1] (ndarray o lista).
            accepted: True = wake aceptado (dir raíz, comportamiento previo);
                False = wake rechazado por el guard → subcarpeta rejected/
                (dataset de hard-negatives + positivos far-field que STRICT mató).

        Returns:
            True si se encoló; False si la cola estaba llena (clip descartado).
        """
        try:
            clip = np.asarray(audio, dtype=np.float32)
            self._queue.put_nowait(
                (room_id, float(score), clip, time.time(), bool(accepted))
            )
            return True
        except queue.Full:
            return False
        except Exception as e:  # jamás romper el audio callback
            logger.warning(f"[WakeClipWriter] submit failed: {e}")
            return False

    def stop(self, timeout: float = 2.0) -> None:
        """Drenar la cola y terminar el worker. Idempotente."""
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=timeout)

    # ---- worker ----

    def _run(self) -> None:
        while self._running or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write(*item)
            except Exception as e:
                logger.warning(f"[WakeClipWriter] write failed: {e}")

    def _write(
        self, room_id: str, score: float, clip: np.ndarray, ts: float,
        accepted: bool = True,
    ) -> None:
        target = self._dir if accepted else self._dir / "rejected"
        target.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))
        millis = int((ts % 1) * 1000)
        name = f"{stamp}-{millis:03d}_{room_id}_{score:.2f}.wav"
        pcm = (np.clip(clip, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(target / name), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self._sample_rate)
            f.writeframes(pcm.tobytes())
        self._rotate(target, self._max_files if accepted else self._max_rejected_files)

    def _rotate(self, target: Path, cap: int) -> None:
        files = sorted(target.glob("*.wav"))
        excess = len(files) - cap
        for path in files[:max(0, excess)]:
            try:
                path.unlink()
            except OSError:
                pass
