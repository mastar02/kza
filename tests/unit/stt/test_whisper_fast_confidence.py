"""Tests: surfacing de confianza del STT (no_speech_prob/avg_logprob)."""
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.stt.whisper_fast import FastWhisperSTT, STTResult


def _seg(text, no_speech_prob, avg_logprob):
    s = MagicMock()
    s.text = text
    s.no_speech_prob = no_speech_prob
    s.avg_logprob = avg_logprob
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


def test_plain_transcribe_still_returns_2_tuple():
    stt = _stt_with_segments([_seg("hola", 0.1, -0.3)])
    out = stt.transcribe(np.zeros(16000, dtype="float32"))
    assert isinstance(out, tuple) and len(out) == 2
    text, ms = out
    assert ms >= 0
    assert text == "hola"
