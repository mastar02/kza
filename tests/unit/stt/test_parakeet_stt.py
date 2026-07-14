"""Tests: ParakeetSTT — motor ASR onnx-asr (src/stt/parakeet_stt.py).

Benchmark A/B con audio real XVF3800 (doc 2026-06-07_SOTA_ASR_ESPANOL_
INVESTIGACION.md): Parakeet-TDT-0.6B-v3 0/5 alucinaciones sobre no-voz vs
5/5 del turbo, mejor calidad en voz, RTF ~0.03 en CPU (0 VRAM). Movido de
src/ambient/ a src/stt/ para que el command path (fast path) también pueda
seleccionarlo vía create_stt.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.stt.whisper_fast import STTResult


@pytest.fixture
def fake_onnx_asr(monkeypatch):
    """Stub de onnx_asr: registra cargas y devuelve un recognizer fake."""
    record = {"loads": [], "calls": []}
    recognizer = MagicMock()

    def fake_recognize(audio, language=None):
        record["calls"].append((audio, language))
        return "hola che"

    recognizer.recognize.side_effect = fake_recognize

    fake_module = types.SimpleNamespace(
        load_model=lambda name, **kw: (record["loads"].append(name), recognizer)[1]
    )
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)
    return record


def _stt(**kw):
    from src.stt.parakeet_stt import ParakeetSTT

    return ParakeetSTT(**kw)


AUDIO = np.zeros(16000, dtype=np.float32)


def test_transcribe_returns_sttresult_with_text(fake_onnx_asr):
    stt = _stt()
    res = stt.transcribe_with_confidence(AUDIO)
    assert isinstance(res, STTResult)
    assert res.text == "hola che"
    assert res.elapsed_ms >= 0
    # Parakeet no expone señales estilo Whisper: None = 'sin penalizar'
    assert res.no_speech_prob is None
    assert res.avg_logprob is None
    assert res.compression_ratio is None


def test_lazy_load_once(fake_onnx_asr):
    stt = _stt(model_name="nemo-parakeet-tdt-0.6b-v3")
    stt.transcribe_with_confidence(AUDIO)
    stt.transcribe_with_confidence(AUDIO)
    assert fake_onnx_asr["loads"] == ["nemo-parakeet-tdt-0.6b-v3"]


def test_language_forwarded(fake_onnx_asr):
    stt = _stt(language="es")
    stt.transcribe_with_confidence(AUDIO)
    assert fake_onnx_asr["calls"][0][1] == "es"


def test_none_text_becomes_empty(fake_onnx_asr):
    # onnx-asr puede devolver None sobre no-voz — el contrato del
    # transcriber espera str (text.strip() decide descartar).
    stt = _stt()
    stt.load()
    stt._model.recognize.side_effect = None
    stt._model.recognize.return_value = None
    res = stt.transcribe_with_confidence(AUDIO)
    assert res.text == ""


def test_transcribe_compat_tuple(fake_onnx_asr):
    # El early-parse (multi_room_audio_loop) usa la firma compat (text, ms).
    stt = _stt()
    text, ms = stt.transcribe(AUDIO)
    assert text == "hola che"
    assert isinstance(ms, float) and ms >= 0


def test_int16_audio_normalized(fake_onnx_asr):
    stt = _stt()
    stt.transcribe_with_confidence((np.ones(16000) * 16384).astype(np.int16))
    audio_sent = fake_onnx_asr["calls"][0][0]
    assert audio_sent.dtype == np.float32
    assert float(np.abs(audio_sent).max()) <= 1.0


def test_builder_selects_engine(fake_onnx_asr):
    from src.ambient.transcriber import _build_ambient_stt_engine

    pk = _build_ambient_stt_engine(
        {"engine": "parakeet"}, {"language": "es"}
    )
    from src.stt.parakeet_stt import ParakeetSTT

    assert isinstance(pk, ParakeetSTT)

    wh = _build_ambient_stt_engine({}, {"language": "es"})
    from src.stt.whisper_fast import FastWhisperSTT

    assert isinstance(wh, FastWhisperSTT)  # default: whisper (compat)
