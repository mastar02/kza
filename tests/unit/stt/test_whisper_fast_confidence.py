"""Tests: surfacing de confianza del STT (no_speech_prob/avg_logprob)."""
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.stt.whisper_fast import FastWhisperSTT, STTResult


def _seg(text, no_speech_prob, avg_logprob, compression_ratio=1.0):
    s = MagicMock()
    s.text = text
    s.no_speech_prob = no_speech_prob
    s.avg_logprob = avg_logprob
    s.compression_ratio = compression_ratio
    return s


def _stt_with_segments(segments):
    stt = FastWhisperSTT(model="x", device="cpu")
    model = MagicMock()
    model.transcribe.return_value = (iter(segments), MagicMock())
    stt._model = model
    return stt


def test_with_confidence_aggregates_segments():
    stt = _stt_with_segments([_seg("hola ", 0.1, -0.3), _seg("mundo", 0.3, -0.5)])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert isinstance(r, STTResult)
    assert r.text == "hola mundo"
    assert r.no_speech_prob == pytest.approx(0.2)      # media
    assert r.avg_logprob == pytest.approx(-0.4)        # media


def test_with_confidence_empty_segments_returns_none():
    stt = _stt_with_segments([])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert r.text == ""
    assert r.no_speech_prob is None
    assert r.avg_logprob is None


def test_with_confidence_surfaces_max_compression_ratio():
    # MAX y no media: un solo segmento basura repetitiva (cr alto) debe poder
    # disparar el guard anti-alucinación; la media lo diluiría.
    stt = _stt_with_segments([
        _seg("hola ", 0.1, -0.3, compression_ratio=1.2),
        _seg("mundo", 0.3, -0.5, compression_ratio=3.4),
    ])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert r.compression_ratio == pytest.approx(3.4)


def test_with_confidence_empty_segments_compression_none():
    stt = _stt_with_segments([])
    r = stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
    assert r.compression_ratio is None


def test_plain_transcribe_still_returns_2_tuple():
    stt = _stt_with_segments([_seg("hola", 0.1, -0.3)])
    out = stt.transcribe(np.zeros(16000, dtype="float32"))
    assert isinstance(out, tuple) and len(out) == 2
    text, ms = out
    assert ms >= 0
    assert text == "hola"


class TestVadFilterConfigurable:
    """Fix 2026-06-04: vad_filter de faster-whisper configurable. El Silero
    interno lee prob~0 sobre el audio del XVF3800 (el chip YA hace VAD/NS/
    beamforming por hardware) y borraba capturas ENTERAS → Text='' con voz
    real (confirmado en prod: 'VAD filter removed 02.000 of 02.000')."""

    def _transcribe_kwargs(self, **stt_kwargs):
        stt = FastWhisperSTT(model="x", device="cpu", **stt_kwargs)
        model = MagicMock()
        model.transcribe.return_value = (iter([]), MagicMock())
        stt._model = model
        stt.transcribe_with_confidence(np.zeros(16000, dtype="float32"))
        return model.transcribe.call_args.kwargs

    def test_vad_filter_disabled_propagates(self):
        kwargs = self._transcribe_kwargs(vad_filter=False)
        assert kwargs["vad_filter"] is False

    def test_vad_filter_default_true_preserved(self):
        kwargs = self._transcribe_kwargs()
        assert kwargs["vad_filter"] is True
