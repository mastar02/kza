# tests/unit/stt/test_whisper_fast_fallback.py
import numpy as np
from src.stt.whisper_fast import FastWhisperSTT, STTResult


class _FakeSegment:
    def __init__(self, text, no_speech_prob=1e-10, avg_logprob=-0.3, compression_ratio=1.1):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


class _FakeModel:
    """Captura los kwargs de transcribe() para verificar el fallback."""
    def __init__(self):
        self.last_kwargs = None

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        return iter([_FakeSegment("prendé la luz")]), object()


def _make(**over):
    stt = FastWhisperSTT(
        model="x", device="cpu",
        temperature_fallback=over.pop("temperature_fallback", True),
        fallback_temperatures=over.pop("fallback_temperatures", [0.0, 0.2, 0.4]),
        compression_ratio_threshold=over.pop("compression_ratio_threshold", 2.0),
        log_prob_threshold=over.pop("log_prob_threshold", -3.0),
    )
    stt._model = _FakeModel()
    return stt


def test_fallback_passes_temperature_list_and_compression_threshold():
    stt = _make()
    stt._transcribe_impl(np.zeros(1600, dtype=np.float32))
    kw = stt._model.last_kwargs
    assert kw["temperature"] == [0.0, 0.2, 0.4]
    assert kw["compression_ratio_threshold"] == 2.0
    assert kw["log_prob_threshold"] == -3.0


def test_fallback_disabled_uses_scalar_zero():
    stt = _make(temperature_fallback=False)
    stt._transcribe_impl(np.zeros(1600, dtype=np.float32))
    kw = stt._model.last_kwargs
    assert kw["temperature"] == 0
