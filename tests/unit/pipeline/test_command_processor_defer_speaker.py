"""Tests for deferred speaker ID on the fast path (await_speaker_id=False)."""

import asyncio
import time

import numpy as np
import pytest

from src.pipeline.command_processor import CommandProcessor


class _FastSTT:
    def transcribe(self, audio, sample_rate):
        return "prendé la luz", 10.0


class _SlowSpeakerID:
    def identify(self, audio, embeddings):
        time.sleep(0.06)
        match = type("M", (), {"is_known": True, "user_id": "u1", "confidence": 0.9})()
        return match


class _UserMgr:
    _version = 0

    def get_all_embeddings(self):
        return {"u1": np.zeros(192, dtype=np.float32)}

    def get_user(self, uid):
        return type("U", (), {"name": "Gabriel"})()

    def update_last_seen(self, uid):
        pass


@pytest.mark.asyncio
async def test_defer_speaker_id_returns_before_speaker_resolves():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=_SlowSpeakerID(),
                          user_manager=_UserMgr(), emotion_detector=None)
    t0 = time.perf_counter()
    result = await cp.process_command(np.zeros(16000, dtype=np.float32), await_speaker_id=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result.text == "prendé la luz"
    assert result.user is None
    assert elapsed_ms < 50, f"retornó en {elapsed_ms:.0f}ms — bloqueó por el speaker (60ms)"
    await asyncio.sleep(0.1)
    assert cp.get_current_user() is not None
    assert cp.get_current_user().name == "Gabriel"


@pytest.mark.asyncio
async def test_await_speaker_id_true_keeps_blocking_behavior():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=_SlowSpeakerID(),
                          user_manager=_UserMgr(), emotion_detector=None)
    result = await cp.process_command(np.zeros(16000, dtype=np.float32), await_speaker_id=True)
    assert result.user is not None
    assert result.user.name == "Gabriel"


@pytest.mark.asyncio
async def test_defer_does_not_clobber_prior_user():
    cp = CommandProcessor(stt=_FastSTT(), speaker_identifier=_SlowSpeakerID(),
                          user_manager=_UserMgr(), emotion_detector=None)
    prior = type("U", (), {"name": "PriorUser"})()
    cp._current_user = prior
    result = await cp.process_command(np.zeros(16000, dtype=np.float32), await_speaker_id=False)
    assert result.user is None
    # El speaker todavía no resolvió (60ms): no debe haberse pisado a None.
    assert cp.get_current_user() is prior
