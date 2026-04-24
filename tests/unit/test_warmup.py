"""Tests para la función de warmup de modelos en src/main.py.

Inyecta stubs en sys.modules antes de importar src.main para evitar
cargar dependencias pesadas (sounddevice, torch, chromadb, etc.) en
entornos de test locales sin GPU.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


_HEAVY_MODULES = (
    "sounddevice",
    "dotenv",
    "chromadb",
    "chromadb.config",
    "sentence_transformers",
    "faster_whisper",
    "speechbrain",
    "speechbrain.pretrained",
    "transformers",
    "torch",
    "kokoro",
    "openwakeword",
    "resemblyzer",
    "llama_cpp",
)


@pytest.fixture(scope="module")
def warmup_fn():
    """Cargar src.main._warmup_models con stubs en sys.modules."""
    for name in _HEAVY_MODULES:
        sys.modules.setdefault(name, MagicMock())

    # dotenv.load_dotenv debe existir al importar
    if "dotenv" in sys.modules:
        sys.modules["dotenv"].load_dotenv = MagicMock()

    try:
        from src.main import _warmup_models
    except Exception as exc:
        pytest.skip(f"No se pudo importar _warmup_models: {exc}")
    return _warmup_models


def test_warmup_calls_each_model(warmup_fn):
    """Cada modelo con método esperado recibe una llamada de warmup."""
    stt = MagicMock()
    stt.transcribe = MagicMock(return_value=("", 0.0))

    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=(np.zeros(100), 0.0, "kokoro"))

    speaker_identifier = MagicMock()
    speaker_identifier.get_embedding = MagicMock(return_value=np.zeros(192))

    emotion_detector = MagicMock()
    emotion_detector.detect = MagicMock(return_value=MagicMock())

    chroma = MagicMock()
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.zeros((1, 1024)))
    chroma._embedder = embedder

    asyncio.run(warmup_fn(stt, tts, speaker_identifier, emotion_detector, chroma))

    stt.transcribe.assert_called_once()
    tts.synthesize.assert_called_once_with("hola")
    speaker_identifier.get_embedding.assert_called_once()
    emotion_detector.detect.assert_called_once()
    embedder.encode.assert_called_once_with(["warmup"])


def test_warmup_skips_none_optional_models(warmup_fn):
    """Modelos opcionales en None no deben causar errores."""
    stt = MagicMock()
    stt.transcribe = MagicMock(return_value=("", 0.0))

    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=(np.zeros(100), 0.0, "kokoro"))

    asyncio.run(warmup_fn(stt, tts, None, None, None))

    stt.transcribe.assert_called_once()
    tts.synthesize.assert_called_once()


def test_warmup_continues_on_model_failure(warmup_fn):
    """Si un modelo falla, los demás igual se ejecutan."""
    stt = MagicMock()
    stt.transcribe = MagicMock(side_effect=RuntimeError("stt boom"))

    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=(np.zeros(100), 0.0, "kokoro"))

    speaker_identifier = MagicMock()
    speaker_identifier.get_embedding = MagicMock(side_effect=RuntimeError("spk boom"))

    emotion_detector = MagicMock()
    emotion_detector.detect = MagicMock(return_value=MagicMock())

    chroma = MagicMock()
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.zeros((1, 1024)))
    chroma._embedder = embedder

    # No debe raisear
    asyncio.run(warmup_fn(stt, tts, speaker_identifier, emotion_detector, chroma))

    stt.transcribe.assert_called_once()
    tts.synthesize.assert_called_once()
    speaker_identifier.get_embedding.assert_called_once()
    emotion_detector.detect.assert_called_once()
    embedder.encode.assert_called_once()


def test_warmup_handles_async_tts(warmup_fn):
    """TTS que retorna coroutine debe ser await-eado."""
    stt = MagicMock()
    stt.transcribe = MagicMock(return_value=("", 0.0))

    async def async_synthesize(text):
        return (np.zeros(100), 0.0)

    tts = MagicMock()
    tts.synthesize = async_synthesize

    asyncio.run(warmup_fn(stt, tts, None, None, None))
    stt.transcribe.assert_called_once()


def test_warmup_skips_chroma_without_embedder(warmup_fn):
    """ChromaSync sin _embedder inicializado no debe causar error."""
    stt = MagicMock()
    stt.transcribe = MagicMock(return_value=("", 0.0))

    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=(np.zeros(100), 0.0, "kokoro"))

    chroma = MagicMock()
    chroma._embedder = None

    asyncio.run(warmup_fn(stt, tts, None, None, chroma))

    stt.transcribe.assert_called_once()
    tts.synthesize.assert_called_once()
