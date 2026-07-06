"""Tests: AmbientSTT — wrapper async sobre FastWhisperSTT para el ambient path."""
import asyncio
from unittest.mock import MagicMock

import numpy as np

from src.ambient.ambient_stt import AmbientSTT
from src.stt.whisper_fast import STTResult


def _stt_mock(text: str = "hola mundo") -> MagicMock:
    m = MagicMock()
    m.transcribe_with_confidence.return_value = STTResult(
        text=text, elapsed_ms=42.0, no_speech_prob=0.1,
        avg_logprob=-0.3, compression_ratio=1.1,
    )
    return m


def test_transcribes_asr_column():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    audio = np.zeros((16000, 6), dtype=np.float32)
    audio[:, 1] = 0.5  # la columna ASR es reconocible

    result = asyncio.run(astt.transcribe(audio))

    assert result.text == "hola mundo"
    passed = inner.transcribe_with_confidence.call_args[0][0]
    assert passed.ndim == 1
    assert passed[0] == np.float32(0.5)


def test_falls_back_to_col0_when_asr_col_missing():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    mono_2d = np.full((16000, 1), 0.25, dtype=np.float32)

    asyncio.run(astt.transcribe(mono_2d))

    passed = inner.transcribe_with_confidence.call_args[0][0]
    assert passed[0] == np.float32(0.25)


def test_accepts_1d_audio():
    inner = _stt_mock()
    astt = AmbientSTT(stt=inner, asr_col=1)
    asyncio.run(astt.transcribe(np.zeros(16000, dtype=np.float32)))
    assert inner.transcribe_with_confidence.called


def test_asr_mono_public_method_extracts_asr_column():
    """Método público (Fix review final): expone la misma extracción que usa
    `transcribe` internamente, para consumidores externos (wake textual)."""
    astt = AmbientSTT(stt=_stt_mock(), asr_col=1)
    audio = np.zeros((16000, 6), dtype=np.float32)
    audio[:, 1] = 0.7

    mono = astt.asr_mono(audio)

    assert mono.ndim == 1
    assert mono[0] == np.float32(0.7)


def test_asr_mono_is_noop_passthrough_for_1d_input():
    astt = AmbientSTT(stt=_stt_mock(), asr_col=1)
    audio = np.zeros(16000, dtype=np.float32)

    mono = astt.asr_mono(audio)

    assert mono is audio
