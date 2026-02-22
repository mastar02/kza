"""
Tests para STTCorrectionCollector - Recoleccion de correcciones de STT
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.training.stt_correction_collector import (
    STTCorrectionCollector,
    STTCorrection,
)


@pytest.fixture
def collector(tmp_path):
    """Collector con directorio temporal, soundfile mockeado"""
    c = STTCorrectionCollector(
        data_dir=str(tmp_path / "stt_corrections"),
        max_audio_duration=30.0,
    )
    # Mock _save_wav to avoid soundfile dependency in tests
    def _mock_save_wav(audio, sample_rate, path):
        import struct
        # Write minimal WAV header + raw data
        n_samples = len(audio)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sI4s", b"RIFF", 36 + n_samples * 2, b"WAVE"))
            f.write(struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate,
                                sample_rate * 2, 2, 16))
            f.write(struct.pack("<4sI", b"data", n_samples * 2))
            f.write((audio * 32767).astype(np.int16).tobytes())

    c._save_wav = _mock_save_wav
    return c


@pytest.fixture
def sample_audio():
    """Audio de prueba (1 segundo a 16kHz)"""
    return np.random.randn(16000).astype(np.float32)


class TestAddCorrection:
    def test_add_correction(self, collector, sample_audio):
        """Guarda audio WAV + metadata JSON"""
        correction_id = collector.add_correction(
            audio=sample_audio,
            original_text="prendo la luz",
            corrected_text="prende la luz",
            sample_rate=16000,
            user_id="mastar",
            confidence=0.85,
        )

        assert correction_id is not None
        assert correction_id.startswith("corr_")

        # Verify files created
        correction_dir = Path(collector.data_dir) / correction_id
        assert (correction_dir / "audio.wav").exists()
        assert (correction_dir / "metadata.json").exists()

        # Verify metadata content
        with open(correction_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["original_text"] == "prendo la luz"
        assert meta["corrected_text"] == "prende la luz"
        assert meta["user_id"] == "mastar"
        assert meta["confidence"] == 0.85
        assert meta["used"] is False

    def test_add_correction_too_long(self, collector):
        """Audio demasiado largo se rechaza"""
        long_audio = np.random.randn(16000 * 60).astype(np.float32)  # 60s

        result = collector.add_correction(
            audio=long_audio,
            original_text="original",
            corrected_text="corrected",
        )

        assert result is None

    def test_add_correction_empty_text(self, collector, sample_audio):
        """Texto vacio se rechaza"""
        result = collector.add_correction(
            audio=sample_audio,
            original_text="original",
            corrected_text="   ",
        )

        assert result is None


class TestGetCorrectionsCount:
    def test_count_empty(self, collector):
        """Sin correcciones retorna 0"""
        assert collector.get_corrections_count() == 0

    def test_count_correct(self, collector, sample_audio):
        """Cuenta correcta de correcciones pendientes"""
        for i in range(3):
            collector.add_correction(
                audio=sample_audio,
                original_text=f"original {i}",
                corrected_text=f"corrected {i}",
            )

        assert collector.get_corrections_count() == 3


class TestGetTrainingPairs:
    def test_get_training_pairs(self, collector, sample_audio):
        """Retorna pares validos para training"""
        for i in range(3):
            collector.add_correction(
                audio=sample_audio,
                original_text=f"original {i}",
                corrected_text=f"corrected {i}",
                user_id="mastar",
            )

        pairs = collector.get_training_pairs()
        assert len(pairs) == 3
        assert all(isinstance(p, STTCorrection) for p in pairs)
        assert pairs[0].corrected_text == "corrected 0"


class TestMarkUsed:
    def test_mark_used(self, collector, sample_audio):
        """Marca correcciones como usadas post-training"""
        for i in range(3):
            collector.add_correction(
                audio=sample_audio,
                original_text=f"original {i}",
                corrected_text=f"corrected {i}",
            )

        # Get pairs and mark as used
        pairs = collector.get_training_pairs()
        assert len(pairs) == 3

        collector.mark_used(pairs[:2])

        # Only 1 unused remaining
        assert collector.get_corrections_count() == 1

        # get_training_pairs also reflects this
        remaining = collector.get_training_pairs()
        assert len(remaining) == 1
        assert remaining[0].corrected_text == "corrected 2"
