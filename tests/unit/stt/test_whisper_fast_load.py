"""Tests: idempotencia de FastWhisperSTT.load() (fix OOM doble-load 2026-05-29).

Regresión: load() se invoca sobre la MISMA instancia compartida desde el
warmup (vía transcribe() lazy) y desde CommandProcessor.__init__ (explícito).
Sin guard, la 2da llamada construía un segundo WhisperModel mientras el primero
seguía vivo → pico 2× VRAM → CUDA out of memory al reiniciar kza-voice.service.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.stt.whisper_fast import FastWhisperSTT


@pytest.fixture
def fake_whisper_model(monkeypatch):
    """Stub de faster_whisper.WhisperModel para no cargar nada real."""
    fake_module = MagicMock()
    ctor = MagicMock(return_value=MagicMock(name="WhisperModelInstance"))
    fake_module.WhisperModel = ctor
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return ctor


def _stt():
    return FastWhisperSTT(
        model="./models/whisper-v3-turbo",
        device="cuda:1",
        compute_type="float16",
        language="es",
    )


def test_load_constructs_model_once(fake_whisper_model):
    stt = _stt()
    assert stt._model is None
    stt.load()
    assert fake_whisper_model.call_count == 1
    assert stt._model is not None


def test_double_load_does_not_construct_second_model(fake_whisper_model):
    # El corazón del fix: 2da load() es no-op, NO crea un 2do WhisperModel
    # (lo que duplicaba la VRAM y causaba el OOM en cuda:1).
    stt = _stt()
    stt.load()
    first_model = stt._model
    stt.load()
    assert fake_whisper_model.call_count == 1, (
        "load() debe ser idempotente: la 2da llamada no debe construir "
        "un segundo WhisperModel"
    )
    assert stt._model is first_model, "no debe reemplazar el modelo ya cargado"


def test_transcribe_then_explicit_load_is_single_construction(fake_whisper_model):
    # Reproduce el patrón real: warmup transcribe (lazy load) + luego
    # CommandProcessor llama stt.load() explícito. Debe haber 1 sola construcción.
    import numpy as np

    fake_segment = MagicMock()
    fake_segment.text = "hola"
    loaded_instance = MagicMock()
    loaded_instance.transcribe.return_value = (iter([fake_segment]), MagicMock())
    fake_whisper_model.return_value = loaded_instance

    stt = _stt()
    # primer uso: transcribe dispara el lazy load()
    with patch.object(FastWhisperSTT, "_prepare_audio", return_value=np.zeros(16000)):
        stt.transcribe(np.zeros(16000, dtype="float32"))
    assert fake_whisper_model.call_count == 1
    # luego CommandProcessor hace load() explícito → no-op por el guard
    stt.load()
    assert fake_whisper_model.call_count == 1
