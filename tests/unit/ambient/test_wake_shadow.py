"""Tests: señal shadow anti-TV en _should_accept_wakeword (solo log, no bloquea)."""
import logging
import time
from unittest.mock import MagicMock

import numpy as np

from src.conversation.follow_up_mode import FollowUpMode
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


def _loop():
    detector = MagicMock()
    echo = MagicMock()
    rs = RoomStream(room_id="escritorio", device_index=4,
                    wake_detector=detector, echo_suppressor=echo)
    return MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )


def test_shadow_logs_but_accepts_when_tv_active(caplog):
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.return_value = True
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)

    with caplog.at_level(logging.INFO):
        accepted = loop._should_accept_wakeword(
            "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
        )
    assert accepted is True  # shadow: NUNCA bloquea
    assert any("[Ambient-shadow]" in r.message for r in caplog.records)


def test_no_log_when_tv_not_active(caplog):
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.return_value = False
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)

    with caplog.at_level(logging.INFO):
        accepted = loop._should_accept_wakeword(
            "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
        )
    assert accepted is True
    assert not any("[Ambient-shadow]" in r.message for r in caplog.records)


def test_transcriber_error_fails_open():
    loop = _loop()
    transcriber = MagicMock()
    transcriber.tv_active_recent.side_effect = RuntimeError("boom")
    loop.attach_ambient(tap=MagicMock(), transcriber=transcriber)
    accepted = loop._should_accept_wakeword(
        "escritorio", rms=0.05, timestamp=time.time(), wake_score=0.55
    )
    assert accepted is True  # fail-open, jamás afecta el wake
