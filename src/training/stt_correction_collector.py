"""
STT Correction Collector Module
Recolecta correcciones de STT para fine-tune semanal de Whisper.

Cuando el usuario corrige una transcripcion, guarda el par
(audio, transcripcion_corregida) para entrenamiento posterior.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class STTCorrection:
    """Par de audio + transcripcion corregida para fine-tune."""
    audio_path: str
    original_text: str
    corrected_text: str
    timestamp: str
    user_id: Optional[str] = None
    confidence: float = 0.0
    used: bool = False


class STTCorrectionCollector:
    """Recolecta correcciones de STT para fine-tune de Whisper."""

    def __init__(self, data_dir: str, max_audio_duration: float = 30.0):
        """
        Args:
            data_dir: Directorio para guardar correcciones
            max_audio_duration: Duracion maxima de audio en segundos
        """
        self.data_dir = Path(data_dir)
        self.max_audio_duration = max_audio_duration
        self._counter = 0

        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"STTCorrectionCollector initialized: {self.data_dir}")

    def add_correction(
        self,
        audio: np.ndarray,
        original_text: str,
        corrected_text: str,
        sample_rate: int = 16000,
        user_id: Optional[str] = None,
        confidence: float = 0.0,
    ) -> Optional[str]:
        """
        Guardar correccion: audio WAV + metadata JSON.

        Args:
            audio: Audio como numpy array
            original_text: Transcripcion original de Whisper
            corrected_text: Texto corregido por el usuario
            sample_rate: Sample rate del audio
            user_id: ID del usuario que corrigio
            confidence: Confidence de Whisper original

        Returns:
            ID de la correccion o None si invalida
        """
        # Validate audio duration
        duration = len(audio) / sample_rate
        if duration > self.max_audio_duration:
            logger.warning(
                f"Audio too long ({duration:.1f}s > {self.max_audio_duration}s), skipping"
            )
            return None

        if not corrected_text.strip():
            return None

        # Generate unique ID
        self._counter += 1
        correction_id = f"corr_{int(time.time() * 1000)}_{self._counter}"
        correction_dir = self.data_dir / correction_id
        correction_dir.mkdir(parents=True, exist_ok=True)

        # Save audio
        audio_path = correction_dir / "audio.wav"
        self._save_wav(audio, sample_rate, audio_path)

        # Save metadata
        correction = STTCorrection(
            audio_path=str(audio_path),
            original_text=original_text,
            corrected_text=corrected_text,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            confidence=confidence,
        )

        meta_path = correction_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(correction), f, ensure_ascii=False, indent=2)

        logger.info(
            f"STT correction saved: '{original_text}' -> '{corrected_text}' ({correction_id})"
        )
        return correction_id

    def _save_wav(self, audio: np.ndarray, sample_rate: int, path: Path):
        """Guardar audio como WAV."""
        import soundfile as sf

        sf.write(str(path), audio, sample_rate)

    def get_corrections_count(self) -> int:
        """Cuantas correcciones hay pendientes (no usadas)."""
        count = 0
        for meta_file in self.data_dir.glob("*/metadata.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                if not meta.get("used", False):
                    count += 1
            except Exception:
                continue
        return count

    def get_training_pairs(self) -> list[STTCorrection]:
        """Lista de correcciones para training (no usadas)."""
        pairs = []
        for meta_file in sorted(self.data_dir.glob("*/metadata.json")):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                if meta.get("used", False):
                    continue

                # Verify audio exists
                if not Path(meta["audio_path"]).exists():
                    logger.warning(f"Audio missing for {meta_file.parent.name}")
                    continue

                pairs.append(STTCorrection(**meta))
            except Exception as e:
                logger.warning(f"Error reading correction {meta_file}: {e}")
                continue
        return pairs

    def mark_used(self, corrections: list[STTCorrection]):
        """Marcar correcciones como usadas post-training."""
        for correction in corrections:
            meta_path = Path(correction.audio_path).parent / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["used"] = True
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Error marking correction as used: {e}")

    def cleanup_old(self, days: int = 90):
        """Limpiar correcciones antiguas."""
        import shutil

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for meta_file in self.data_dir.glob("*/metadata.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                ts = datetime.fromisoformat(meta["timestamp"])
                if ts < cutoff:
                    shutil.rmtree(meta_file.parent)
                    removed += 1
            except Exception:
                continue

        if removed:
            logger.info(f"Cleaned up {removed} old STT corrections (>{days} days)")
