"""
Tests para Dual TTS (Kokoro + Qwen3) y factory create_tts()
"""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock sounddevice before importing piper_tts (not available in CI)
sys.modules.setdefault("sounddevice", MagicMock())

from src.tts.piper_tts import (
    KokoroTTS,
    Qwen3TTS,
    DualTTS,
    create_tts,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_kokoro():
    """KokoroTTS con modelo mockeado (API Kokoro 0.7+: generator de Results)."""
    kokoro = KokoroTTS(model="test-kokoro", device="cpu", lang_code="e")
    # Kokoro 0.7+ __call__ es generator que yield Result con .audio (Tensor).
    # Mockeamos un objeto con .audio = ndarray (el código también maneja ndarray
    # directo porque solo llama .cpu().numpy() si tiene el método .cpu).
    audio = np.random.randn(12000).astype(np.float32)
    fake_result = MagicMock()
    fake_result.audio = audio  # ndarray directo; no pasa por .cpu().numpy()
    kokoro._model = MagicMock(return_value=iter([fake_result]))
    return kokoro


@pytest.fixture
def mock_qwen3():
    """Qwen3TTS con modelo mockeado"""
    qwen3 = Qwen3TTS(model="test-qwen3", device="cpu")
    qwen3._model = MagicMock()
    qwen3._processor = MagicMock()
    # Return 1s of audio at 24kHz
    audio = np.random.randn(24000).astype(np.float32)
    qwen3._processor.return_value = MagicMock()
    qwen3._processor.decode.return_value = audio
    qwen3._model.generate.return_value = [MagicMock()]
    return qwen3


@pytest.fixture
def dual_tts(mock_kokoro, mock_qwen3):
    """DualTTS con ambos motores mockeados"""
    dual = DualTTS(quality_threshold=50)
    dual.kokoro = mock_kokoro
    dual.qwen3 = mock_qwen3
    return dual


# =========================================================================
# Tests de DualTTS
# =========================================================================

class TestDualTTS:
    def test_routes_short_to_kokoro(self, dual_tts):
        """Textos cortos (<=50 chars) van a Kokoro"""
        text = "Luz encendida"  # 13 chars
        audio, elapsed, engine = dual_tts.synthesize(text)

        assert engine == "kokoro"
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_routes_long_to_qwen3(self, dual_tts):
        """Textos largos (>50 chars) van a Qwen3"""
        text = "La temperatura actual en la cocina es de 22 grados centígrados y el aire está encendido"
        assert len(text) > 50

        audio, elapsed, engine = dual_tts.synthesize(text)

        assert engine == "qwen3"
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_force_quality(self, dual_tts):
        """force_quality=True siempre usa Qwen3"""
        text = "Listo"  # Short but forced
        audio, elapsed, engine = dual_tts.synthesize(text, force_quality=True)

        assert engine == "qwen3"

    def test_threshold_boundary(self, dual_tts):
        """Exactamente en el threshold usa Kokoro"""
        text = "x" * 50  # Exactly 50 chars
        _, _, engine = dual_tts.synthesize(text)
        assert engine == "kokoro"

        text = "x" * 51  # 51 chars
        _, _, engine = dual_tts.synthesize(text)
        assert engine == "qwen3"

    def test_stream_routes_correctly(self, dual_tts):
        """synthesize_stream delega al motor correcto"""
        short_text = "Hola"
        chunks = list(dual_tts.synthesize_stream(short_text))
        assert len(chunks) > 0
        assert all(isinstance(c, np.ndarray) for c in chunks)


# =========================================================================
# Tests de Qwen3TTS
# =========================================================================

class TestQwen3TTS:
    def test_synthesize(self, mock_qwen3):
        """Qwen3 genera audio"""
        audio, elapsed = mock_qwen3.synthesize("Hola, ¿cómo estás?")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert elapsed > 0


# =========================================================================
# Tests de Factory
# =========================================================================

class TestCreateTTS:
    def test_create_dual(self):
        """Factory crea DualTTS con engine='dual'"""
        config = {
            "engine": "dual",
            "kokoro": {"model": "hexgrad/Kokoro-82M", "device": "cpu"},
            "qwen3": {"model": "Qwen/Qwen3-TTS-0.6B", "device": "cpu"},
            "quality_threshold": 50,
        }

        tts = create_tts(config)
        assert isinstance(tts, DualTTS)
        assert tts.quality_threshold == 50

    def test_create_piper(self):
        """Factory sigue creando PiperTTS"""
        from src.tts.piper_tts import PiperTTS

        config = {"engine": "piper", "piper": {"model": "test.onnx"}}
        tts = create_tts(config)
        assert isinstance(tts, PiperTTS)

    def test_create_unknown_raises(self):
        """Engine desconocido lanza error"""
        with pytest.raises(ValueError, match="Engine TTS desconocido"):
            create_tts({"engine": "unknown"})
