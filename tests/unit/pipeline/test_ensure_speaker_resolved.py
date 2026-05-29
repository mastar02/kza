import asyncio
import time

import numpy as np
import pytest

from src.pipeline.command_processor import CommandProcessor


class _FastSTT:
    def transcribe(self, audio, sample_rate):
        return "contame un chiste", 10.0


class _SlowSpeakerID:
    def identify(self, audio, embeddings):
        time.sleep(0.06)
        return type("M", (), {"is_known": True, "user_id": "u1", "confidence": 0.9})()


class _UserMgr:
    _version = 0
    def get_all_embeddings(self): return {"u1": np.zeros(192, dtype=np.float32)}
    def get_user(self, uid): return type("U", (), {"name": "Gabriel"})()
    def update_last_seen(self, uid): pass


@pytest.mark.asyncio
async def test_ensure_speaker_resolved_waits_for_deferred_task():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=_SlowSpeakerID(),
                          user_manager=_UserMgr(), emotion_detector=None)
    result = await cp.process_command(np.zeros(16000, dtype=np.float32), await_speaker_id=False)
    assert result.user is None  # difirió
    user = await cp.ensure_speaker_resolved(timeout_s=1.0)  # slow path espera
    assert user is not None and user.name == "Gabriel"


@pytest.mark.asyncio
async def test_ensure_speaker_resolved_returns_none_when_no_tasks():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=None,
                          user_manager=None, emotion_detector=None)
    user = await cp.ensure_speaker_resolved(timeout_s=0.1)
    assert user is None
