"""
Integration test: concurrent commands from multiple rooms.
Verifies commands from different rooms process in parallel,
and echo deduplication works correctly.
"""
import sys
from unittest.mock import MagicMock

# Mock hardware dependencies BEFORE any imports
for mod in ["sounddevice", "soundfile", "pyaudio", "torch", "torch.cuda"]:
    sys.modules.setdefault(mod, MagicMock())

import asyncio
import numpy as np
import pytest
import time
from unittest.mock import AsyncMock

from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


@pytest.mark.asyncio
async def test_concurrent_commands_from_different_rooms():
    """Two commands from different rooms should process in parallel."""
    results = []

    async def mock_process(event: CommandEvent) -> dict:
        await asyncio.sleep(0.1)  # Simulate 100ms processing
        results.append(event.room_id)
        return {"text": f"processed_{event.room_id}", "success": True}

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(is_active=False),
    )
    loop.on_command(mock_process)

    event1 = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="cocina",
        mic_device_index=3,
    )
    event2 = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="escritorio",
        mic_device_index=4,
    )

    start = time.time()
    await asyncio.gather(
        loop._dispatch_command(event1),
        loop._dispatch_command(event2),
    )
    total = time.time() - start

    assert len(results) == 2
    assert "cocina" in results
    assert "escritorio" in results
    # Parallel: ~100ms, not ~200ms
    assert total < 0.18, f"Commands took {total:.3f}s — should be parallel"


@pytest.mark.asyncio
async def test_dedup_prevents_echo_but_allows_concurrent():
    """Echo dedup blocks echoes but allows genuine concurrent commands."""
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
        dedup_window_ms=200,
    )

    now = time.time()

    # Genuine concurrent (>200ms apart)
    assert loop._should_accept_wakeword("cocina", 0.5, now) is True
    assert loop._should_accept_wakeword("escritorio", 0.5, now + 0.3) is True

    # Echo (within 200ms, weaker)
    now2 = time.time()
    assert loop._should_accept_wakeword("living", 0.8, now2) is True
    assert loop._should_accept_wakeword("hall", 0.3, now2 + 0.05) is False


@pytest.mark.asyncio
async def test_post_command_callback_called_for_each():
    """Post-command callback should fire for each concurrent command."""
    post_results = []

    async def mock_process(event: CommandEvent) -> dict:
        return {"room": event.room_id}

    async def mock_post(result: dict, event: CommandEvent):
        post_results.append(event.room_id)

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
    )
    loop.on_command(mock_process)
    loop.on_post_command(mock_post)

    events = [
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="cocina"),
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="living"),
        CommandEvent(audio=np.zeros(8000, dtype=np.float32), room_id="bano"),
    ]

    await asyncio.gather(*[loop._dispatch_command(e) for e in events])

    assert len(post_results) == 3
    assert set(post_results) == {"cocina", "living", "bano"}
