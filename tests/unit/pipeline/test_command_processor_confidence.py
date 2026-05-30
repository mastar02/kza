"""Tests: ProcessedCommand carga la confianza del STT."""
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult


class _FakeSTT:
    def transcribe(self, audio, sr=16000):
        return "prendé la luz", 5.0
    def transcribe_with_confidence(self, audio, sr=16000):
        return STTResult("prendé la luz", 5.0, no_speech_prob=0.15, avg_logprob=-0.4)


@pytest.mark.asyncio
async def test_process_command_populates_confidence_sequential():
    cp = CommandProcessor(stt=_FakeSTT(), speaker_identifier=None, user_manager=None)
    result = await cp.process_command(np.zeros(16000, dtype="float32"), use_parallel=False)
    assert result.stt_confidence is not None
    assert result.stt_confidence.no_speech_prob == pytest.approx(0.15)
    assert result.stt_confidence.avg_logprob == pytest.approx(-0.4)


@pytest.mark.asyncio
async def test_pretranscribed_leaves_confidence_none():
    cp = CommandProcessor(stt=_FakeSTT(), speaker_identifier=None, user_manager=None)
    result = await cp.process_command(
        np.zeros(16000, dtype="float32"), pretranscribed_text="prendé la luz",
    )
    assert result.stt_confidence is None


@pytest.mark.asyncio
async def test_process_command_populates_confidence_parallel():
    """El path paralelo también debe propagar stt_confidence al ProcessedCommand."""
    cp = CommandProcessor(
        stt=_FakeSTT(),
        speaker_identifier=None,
        user_manager=None,
        emotion_detector=None,
    )
    # use_parallel=True pero sin speaker_identifier ni emotion_detector → cae al
    # else-branch (sequential) cuando ambos son None; pasamos un speaker_id fake
    # para forzar el gather path.
    from unittest.mock import MagicMock

    fake_speaker_id = MagicMock()
    fake_speaker_id.identify.return_value = MagicMock(is_known=False, confidence=0.0, user_id=None)
    fake_user_manager = MagicMock()
    fake_user_manager.get_all_embeddings.return_value = {}

    cp2 = CommandProcessor(
        stt=_FakeSTT(),
        speaker_identifier=fake_speaker_id,
        user_manager=fake_user_manager,
        emotion_detector=None,
    )
    result = await cp2.process_command(
        np.zeros(16000, dtype="float32"), use_parallel=True
    )
    assert result.stt_confidence is not None
    assert result.stt_confidence.no_speech_prob == pytest.approx(0.15)
