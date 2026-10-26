"""Speculative prefix -> completed capture -> real gate and router, without HA."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tests.mocks.responses import response_handler_mock
from src.nlu.command_gate import CommandAcceptanceGate
from src.pipeline.command_event import CapturedTranscript, TranscriptSource
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.pipeline.request_router import RequestRouter


async def _wait_for(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("orchestrated", [False, True])
@pytest.mark.parametrize("suffix", ["mañana", "no la apagues", "cuando llegue"])
async def test_appended_clause_is_not_lost_after_a_ready_prefix(suffix, orchestrated):
    prefix = "Apagá la luz"
    final = f"{prefix}, {suffix}"
    audio = np.full(1280, 0.1, dtype=np.float32)
    echo = MagicMock()
    echo.is_safe_to_listen = True
    echo.should_process_audio.return_value = (True, "ok")
    stream = RoomStream("test", 0, MagicMock(), echo)
    stream.listening = True
    stream.command_start_time = time.time() - 0.5
    stream.audio_buffer = list(audio)
    stream.wake_text = prefix
    stream.transcript = CapturedTranscript.from_audio(prefix, TranscriptSource.WAKE, audio)
    stt = MagicMock()
    stt.transcribe.side_effect = lambda captured, **_: (prefix if len(captured) == len(audio) else final, 1)
    loop = MultiRoomAudioLoop(
        {"test": stream}, MagicMock(), stt=stt, command_duration=5,
        early_dispatch_enabled=True, early_dispatch_interval_ms=5,
        early_dispatch_min_audio_s=0.01,
    )
    processor = MagicMock()

    async def process(captured, *, pretranscribed_text=None, **_):
        return ProcessedCommand(text=pretranscribed_text or final, success=True)

    processor.process_command = AsyncMock(side_effect=process)
    orchestrator = MagicMock()
    orchestrator.process = AsyncMock()
    ha = MagicMock()
    ha.call_service_ws = AsyncMock()
    ha.call_service = AsyncMock()
    router = RequestRouter(
        command_processor=processor, response_handler=response_handler_mock(), audio_manager=MagicMock(),
        ha_client=ha, orchestrator=orchestrator, orchestrator_enabled=orchestrated,
        command_gate=CommandAcceptanceGate(), wake_acoustically_confirmed=True,
    )
    results = []
    completed = asyncio.Event()

    async def dispatch(event):
        results.append(await router.process_command(event))
        completed.set()

    loop.on_command(dispatch)
    sd = MagicMock()
    sd.query_devices.return_value = {"max_input_channels": 2}
    with patch("src.pipeline.multi_room_audio_loop.sd", sd):
        task = asyncio.create_task(loop.run())
        try:
            await asyncio.wait_for(_wait_for(lambda: stream.early_command is not None), 2)
            await asyncio.sleep(0.08)  # Include at least one main-loop poll.
            assert results == []
            processor.process_command.assert_not_awaited()
            # Add speech and then its endpoint through the production callback.
            callback = loop._make_audio_callback(stream)
            callback(audio[:, None], len(audio), None, None)
            silence = np.zeros((4800, 1), dtype=np.float32)
            callback(silence, len(silence), None, None)
            await asyncio.wait_for(completed.wait(), 2)
            assert results[0]["text"] == final
            assert results[0]["intent"].startswith("gate_rejected:non_immediate:")
            orchestrator.process.assert_not_awaited()
            ha.call_service_ws.assert_not_awaited()
            ha.call_service.assert_not_awaited()
        finally:
            await loop.stop()
            await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_pending_speculative_result_cannot_attach_to_a_new_capture():
    entered = asyncio.Event()
    release = asyncio.Event()
    stream = RoomStream("test", 0, MagicMock(), MagicMock())
    stream.listening = True
    stream.audio_buffer = [0.1] * 1280
    loop = MultiRoomAudioLoop(
        {"test": stream}, MagicMock(), stt=MagicMock(),
        early_dispatch_interval_ms=1, early_dispatch_min_audio_s=0.01,
    )
    # Control the thread boundary, without blocking a worker or the event loop.
    async def transcribe(*_, **__):
        entered.set()
        await release.wait()
        return "Apagá la luz", 1

    loop._running = True
    with patch("src.pipeline.multi_room_audio_loop.asyncio.to_thread", transcribe):
        task = asyncio.create_task(loop._early_parse_worker(stream))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            loop._reset_listening(stream)
            stream.listening = True
            stream.audio_buffer = [0.2] * 1280
            release.set()
            await asyncio.wait_for(task, 2)
            assert stream.early_command is None
            assert stream.transcript is None
        finally:
            loop._running = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
