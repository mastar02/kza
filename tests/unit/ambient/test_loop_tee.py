"""Tests: tee del audio callback de MultiRoomAudioLoop al MultiChannelTap."""
from unittest.mock import MagicMock

import numpy as np

from src.ambient.tap import MultiChannelTap
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


def _loop_with_tap(tts_speaking: bool = False):
    detector = MagicMock()
    detector.detect.return_value = None
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, "")
    rs = RoomStream(
        room_id="escritorio", device_index=4,
        wake_detector=detector, echo_suppressor=echo,
    )
    from src.conversation.follow_up_mode import FollowUpMode
    loop = MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )
    rh = MagicMock()
    rh.is_speaking = tts_speaking
    loop.attach_response_handler(rh)
    tap = MultiChannelTap(maxlen_chunks=10)
    tap.register_room("escritorio")
    loop.attach_ambient(tap=tap, transcriber=None)
    return loop, rs, tap


def test_callback_tees_full_multichannel_chunk():
    loop, rs, tap = _loop_with_tap()
    cb = loop._make_audio_callback(rs)
    indata = np.random.default_rng(1).normal(0, 0.1, size=(1280, 6)).astype(np.float32)
    cb(indata, 1280, None, None)
    items = tap.drain("escritorio")
    assert len(items) == 1
    _, chunk, tts = items[0]
    assert chunk.shape == (1280, 6)  # multicanal completo, no solo capture_channel
    assert tts is False


def test_tee_marks_tts_active_and_still_tees_during_tts():
    # Durante TTS el flujo normal de wake hace return temprano — el tap
    # igual debe recibir el chunk, marcado tts_active=True.
    loop, rs, tap = _loop_with_tap(tts_speaking=True)
    loop.barge_in_enabled = True
    cb = loop._make_audio_callback(rs)
    indata = np.zeros((1280, 6), dtype=np.float32)
    cb(indata, 1280, None, None)
    items = tap.drain("escritorio")
    assert len(items) == 1
    assert items[0][2] is True


def test_no_tap_attached_keeps_callback_working():
    detector = MagicMock()
    detector.detect.return_value = None
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, "")
    rs = RoomStream(room_id="escritorio", device_index=4,
                    wake_detector=detector, echo_suppressor=echo)
    from src.conversation.follow_up_mode import FollowUpMode
    loop = MultiRoomAudioLoop(
        room_streams={"escritorio": rs},
        follow_up=FollowUpMode(follow_up_window=4.0),
    )
    cb = loop._make_audio_callback(rs)
    cb(np.zeros((1280, 2), dtype=np.float32), 1280, None, None)  # no lanza
