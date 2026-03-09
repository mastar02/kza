"""Tests for ProcessedCommand dataclass and CommandProcessor return type."""
import sys
from unittest.mock import MagicMock, AsyncMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import pytest
import numpy as np

from src.pipeline.command_processor import ProcessedCommand, CommandProcessor


class TestProcessedCommand:
    def test_defaults(self):
        cmd = ProcessedCommand(text="hola")
        assert cmd.text == "hola"
        assert cmd.user is None
        assert cmd.emotion is None
        assert cmd.speaker_confidence == 0.0
        assert cmd.timings == {}
        assert cmd.success is False

    def test_with_user(self):
        mock_user = MagicMock()
        mock_user.user_id = "u1"
        mock_user.name = "Ana"
        cmd = ProcessedCommand(text="prende luz", user=mock_user, speaker_confidence=0.92, success=True)
        assert cmd.user.user_id == "u1"
        assert cmd.user.name == "Ana"
        assert cmd.speaker_confidence == 0.92

    def test_with_emotion(self):
        mock_emotion = MagicMock()
        mock_emotion.emotion = "happy"
        cmd = ProcessedCommand(text="que lindo dia", emotion=mock_emotion)
        assert cmd.emotion.emotion == "happy"


class TestCommandProcessorReturnsProcessedCommand:
    @pytest.mark.asyncio
    async def test_returns_processed_command_type(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("hola mundo", 50.0))
        cp = CommandProcessor(stt=stt)
        audio = np.zeros(16000, dtype=np.float32)

        result = await cp.process_command(audio, use_parallel=False)
        assert isinstance(result, ProcessedCommand)
        assert result.text == "hola mundo"
        assert result.success is True
        assert result.user is None

    @pytest.mark.asyncio
    async def test_with_identified_speaker(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("prende la luz", 50.0))

        mock_user = MagicMock()
        mock_user.user_id = "u1"
        mock_user.name = "Ana"

        mock_match = MagicMock()
        mock_match.is_known = True
        mock_match.user_id = "u1"
        mock_match.confidence = 0.91

        speaker_id = MagicMock()
        speaker_id.identify = MagicMock(return_value=mock_match)

        user_manager = MagicMock()
        user_manager.get_all_embeddings = MagicMock(return_value={"u1": np.zeros(192)})
        user_manager.get_user = MagicMock(return_value=mock_user)
        user_manager.update_last_seen = MagicMock()

        cp = CommandProcessor(
            stt=stt,
            speaker_identifier=speaker_id,
            user_manager=user_manager,
        )

        audio = np.zeros(16000, dtype=np.float32)
        result = await cp.process_command(audio, use_parallel=False)

        assert isinstance(result, ProcessedCommand)
        assert result.user is mock_user
        assert result.speaker_confidence == 0.91

    @pytest.mark.asyncio
    async def test_empty_text_is_not_success(self):
        stt = MagicMock()
        stt.transcribe = MagicMock(return_value=("   ", 50.0))
        cp = CommandProcessor(stt=stt)
        audio = np.zeros(16000, dtype=np.float32)

        result = await cp.process_command(audio, use_parallel=False)
        assert result.success is False
