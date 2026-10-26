"""Real capture callback and barge-in scheduling against owned streamed voice."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.voice_delivery import DeliveryState
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph


@pytest.mark.asyncio
async def test_capture_continues_thinking_then_barges_into_actual_stream_and_accepts_new_audio(monkeypatch, tmp_path):
    release = asyncio.Event()
    text = ' '.join(f'Oración número {i} suficientemente larga.' for i in range(100))
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch, parts=[release, event(text), event(finish='stop')])
    backend.callback_pause_s = .01
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, 'ok')
    echo.is_human_voice.return_value = True
    wake = MagicMock()
    wake.detect.return_value = None
    follow_up = MagicMock()
    follow_up.is_active = False
    room = RoomStream('origin', 0, wake_detector=wake, echo_suppressor=echo)
    capture = MultiRoomAudioLoop({'origin': room}, follow_up, response_handler=owner,
        barge_in_enabled=True, barge_in_min_duration_ms=100, barge_in_rms_threshold=.02,
        audio_health_path=str(tmp_path/'health.json'))
    capture._loop = asyncio.get_running_loop()
    callback = capture._make_audio_callback(room)
    chunk = np.full((320, 1), .5, dtype=np.float32)
    status = SimpleNamespace(output_underflow=False)

    def feed(count):
        for _ in range(count):
            callback(chunk, 320, None, status)

    async def until(predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    speaking = asyncio.create_task(owner.arespond('synthetic', zone_id='zone_origin'))
    try:
        await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
        assert owner.is_responding and not owner.is_speaking
        await asyncio.to_thread(feed, 1)
        wake.detect.assert_called_once()
        release.set()
        await until(lambda: backend.outputs and backend.outputs[0].delivered.is_set())
        assert owner.is_speaking
        await asyncio.to_thread(feed, 5)
        assert (await asyncio.wait_for(speaking, 1)).delivery is DeliveryState.CANCELLED
        await until(lambda: not owner.is_responding and room.listening)
        assert endpoint.body.closed and endpoint.client.is_closed()
        assert backend.outputs[0].closed and backend.outputs[0].aborted.is_set()
        follow_up.start_conversation.assert_called_once()
        await asyncio.to_thread(feed, 1)
        assert len(room.audio_buffer) == 320
        assert capture._microphone_captures['origin'].observation.frames_received == 7*320
        cancelled_frames = sum(len(chunk) for chunk in backend.outputs[0].frames)
        backend.callback_pause_s = .0002
        assert (await owner.aspeak('Respuesta nueva.', zone_id='zone_origin')).success
        assert sum(len(chunk) for chunk in backend.outputs[0].frames) == cancelled_frames
        new_pcm = np.concatenate(backend.outputs[-1].frames)[:, 0]
        np.testing.assert_array_equal(new_pcm, engine.audio('Respuesta nueva.'))
    finally:
        release.set()
        await owner.aclose()
        await capture.stop()
