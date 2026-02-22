# Multi-Interface Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire together XVF3800 microphones, BLE dongles, and room context so the voice pipeline processes commands per-room with concurrent multi-sala support.

**Architecture:** Replace single-stream `AudioLoop` with `MultiRoomAudioLoop` (5 parallel `sd.InputStream`). Each command carries a `room_id` through the pipeline. `RoomContextManager` fuses mic + BLE to resolve entities per-room. TTS routes to the correct MA1260 zone.

**Tech Stack:** Python 3.10+, sounddevice, asyncio, openwakeword, numpy

**Design doc:** `docs/plans/2026-02-21-multi-interface-integration-design.md`

---

### Task 1: Unify Config — Merge `zones` into `rooms`

**Files:**
- Modify: `config/settings.yaml:527-591` (zones section) and `config/settings.yaml:725-828` (rooms section)

**Step 1: Add MA1260 fields to each room in settings.yaml**

Add `ma1260_zone`, `output_mode`, `default_volume`, `noise_floor` to each room config. Remove the separate `zones.zone_list` section.

```yaml
rooms:
  enabled: true
  cross_validation: true
  fallback_room: "living"
  dedup_window_ms: 200

  # MA1260 amplifier connection (moved from zones)
  ma1260:
    connection_type: "serial"
    serial_port: "/dev/ttyUSB0"
    baudrate: 9600
    audio_output_device: null
    default_source: 1

  # Wake word config for all rooms
  wake_word:
    model: "hey_jarvis"
    threshold: 0.5

  # Detection config
  detection:
    priority_mode: "loudest"
    detection_window_ms: 500
    vad_threshold: 0.02

  living:
    name: "Living"
    display_name: "el living"
    # Input
    mic_device_name: "XVF3800"
    mic_device_index: 1
    bt_adapter: "hci0"
    # Output (MA1260)
    ma1260_zone: 1
    output_mode: "stereo"
    default_volume: 60
    noise_floor: 0.01
    # HA entities (unchanged)
    default_light: "light.living"
    default_climate: "climate.living_ac"
    default_cover: "cover.living_persiana"
    default_media_player: "media_player.living_tv"
    motion_sensor: "binary_sensor.motion_living"
    temperature_sensor: "sensor.temperature_living"
    tts_speaker: "media_player.living_speaker"
    aliases: ["living", "sala", "salón", "el living"]

  hall:
    name: "Hall"
    display_name: "el hall"
    mic_device_name: "XVF3800"
    mic_device_index: 2
    bt_adapter: "hci1"
    ma1260_zone: 2
    output_mode: "mono"
    default_volume: 50
    noise_floor: 0.01
    default_light: "light.hall"
    motion_sensor: "binary_sensor.motion_hall"
    tts_speaker: "media_player.hall_speaker"
    aliases: ["hall", "pasillo", "entrada", "el hall"]

  cocina:
    name: "Cocina"
    display_name: "la cocina"
    mic_device_name: "XVF3800"
    mic_device_index: 3
    bt_adapter: "hci2"
    ma1260_zone: 3
    output_mode: "mono"
    default_volume: 55
    noise_floor: 0.015
    default_light: "light.cocina"
    default_fan: "fan.cocina_extractor"
    motion_sensor: "binary_sensor.motion_cocina"
    temperature_sensor: "sensor.temperature_cocina"
    humidity_sensor: "sensor.humidity_cocina"
    tts_speaker: "media_player.cocina_speaker"
    aliases: ["cocina", "la cocina", "kitchen"]

  escritorio:
    name: "Escritorio"
    display_name: "el escritorio"
    mic_device_name: "XVF3800"
    mic_device_index: 4
    bt_adapter: "hci3"
    ma1260_zone: 4
    output_mode: "mono"
    default_volume: 50
    noise_floor: 0.01
    default_light: "light.escritorio"
    default_climate: "climate.escritorio_ac"
    default_media_player: "media_player.escritorio_monitor"
    motion_sensor: "binary_sensor.motion_escritorio"
    temperature_sensor: "sensor.temperature_escritorio"
    tts_speaker: "media_player.escritorio_speaker"
    aliases: ["escritorio", "oficina", "estudio", "el escritorio"]

  bano:
    name: "Baño"
    display_name: "el baño"
    mic_device_name: "XVF3800"
    mic_device_index: 5
    bt_adapter: "hci4"
    ma1260_zone: 5
    output_mode: "stereo"
    default_volume: 50
    noise_floor: 0.02
    default_light: "light.bano"
    default_fan: "fan.bano_extractor"
    motion_sensor: "binary_sensor.motion_bano"
    humidity_sensor: "sensor.humidity_bano"
    tts_speaker: "media_player.bano_speaker"
    aliases: ["baño", "el baño", "bathroom"]
```

**Step 2: Remove the old `zones.zone_list` section**

Delete lines 548-591 from settings.yaml (the `zone_list` entries). Keep the `zones` key only with `enabled: false` as a deprecated marker, or remove entirely.

**Step 3: Commit**

```bash
git add config/settings.yaml
git commit -m "config: unify zones into rooms with MA1260 fields"
```

---

### Task 2: Add `RoomConfig.ma1260_zone` and `output_mode` to room_context.py

**Files:**
- Modify: `src/rooms/room_context.py:37-65` (RoomConfig dataclass)
- Test: `tests/unit/rooms/test_room_context.py`

**Step 1: Write the failing test**

```python
# In tests/unit/rooms/test_room_context.py — add this test
def test_room_config_has_ma1260_fields():
    """RoomConfig should include MA1260 zone and output mode."""
    config = RoomConfig(
        room_id="living",
        name="Living",
        display_name="el living",
        ma1260_zone=1,
        output_mode="stereo",
        default_volume=60,
        noise_floor=0.01,
    )
    assert config.ma1260_zone == 1
    assert config.output_mode == "stereo"
    assert config.default_volume == 60
    assert config.noise_floor == 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/rooms/test_room_context.py::test_room_config_has_ma1260_fields -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'ma1260_zone'`

**Step 3: Add fields to RoomConfig**

In `src/rooms/room_context.py`, add to the `RoomConfig` dataclass after `bt_adapter`:

```python
    # MA1260 output
    ma1260_zone: Optional[int] = None          # MA1260 zone number (1-6)
    output_mode: str = "mono"                  # "stereo" or "mono"
    default_volume: int = 50                   # Default volume (0-100)
    noise_floor: float = 0.01                  # Noise floor for VAD
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/rooms/test_room_context.py::test_room_config_has_ma1260_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rooms/room_context.py tests/unit/rooms/test_room_context.py
git commit -m "feat: add MA1260 zone fields to RoomConfig"
```

---

### Task 3: Create `CommandEvent` dataclass

**Files:**
- Create: `src/pipeline/command_event.py`
- Test: `tests/unit/pipeline/test_command_event.py`

**Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_command_event.py
import numpy as np
import time
from src.pipeline.command_event import CommandEvent


def test_command_event_creation():
    """CommandEvent carries audio data with room metadata."""
    audio = np.zeros(16000, dtype=np.float32)
    now = time.time()
    event = CommandEvent(
        audio=audio,
        room_id="cocina",
        mic_device_index=3,
        timestamp=now,
    )
    assert event.room_id == "cocina"
    assert event.mic_device_index == 3
    assert len(event.audio) == 16000
    assert event.timestamp == now


def test_command_event_defaults():
    """CommandEvent should work with minimal args."""
    audio = np.zeros(8000, dtype=np.float32)
    event = CommandEvent(audio=audio, room_id="living")
    assert event.room_id == "living"
    assert event.mic_device_index is None
    assert event.timestamp > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/pipeline/test_command_event.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.command_event'`

**Step 3: Create CommandEvent**

```python
# src/pipeline/command_event.py
"""
Command Event — carries captured audio with room metadata.

Emitted by MultiRoomAudioLoop when a room's microphone completes
command capture after wake word detection.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CommandEvent:
    """Audio command captured from a specific room's microphone."""
    audio: np.ndarray
    room_id: str
    mic_device_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/pipeline/test_command_event.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/command_event.py tests/unit/pipeline/test_command_event.py
git commit -m "feat: add CommandEvent dataclass for room-aware audio"
```

---

### Task 4: Create `MultiRoomAudioLoop`

**Files:**
- Create: `src/pipeline/multi_room_audio_loop.py`
- Create: `tests/unit/pipeline/test_multi_room_audio_loop.py`

**Step 1: Write the failing tests**

```python
# tests/unit/pipeline/test_multi_room_audio_loop.py
import asyncio
import numpy as np
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import (
    MultiRoomAudioLoop,
    RoomStream,
)


@pytest.fixture
def mock_wake_detector():
    detector = MagicMock()
    detector.detect.return_value = None
    detector.load.return_value = None
    detector.get_active_models.return_value = ["hey_jarvis"]
    return detector


@pytest.fixture
def mock_echo_suppressor():
    suppressor = MagicMock()
    suppressor.is_safe_to_listen = True
    suppressor.should_process_audio.return_value = (True, "ok")
    return suppressor


@pytest.fixture
def room_streams(mock_wake_detector, mock_echo_suppressor):
    return {
        "cocina": RoomStream(
            room_id="cocina",
            device_index=3,
            wake_detector=mock_wake_detector,
            echo_suppressor=mock_echo_suppressor,
        ),
        "living": RoomStream(
            room_id="living",
            device_index=1,
            wake_detector=MagicMock(
                detect=MagicMock(return_value=None),
                load=MagicMock(),
                get_active_models=MagicMock(return_value=["hey_jarvis"]),
            ),
            echo_suppressor=MagicMock(
                is_safe_to_listen=True,
                should_process_audio=MagicMock(return_value=(True, "ok")),
            ),
        ),
    }


def test_room_stream_creation():
    """RoomStream holds per-room audio state."""
    stream = RoomStream(
        room_id="cocina",
        device_index=3,
        wake_detector=MagicMock(),
        echo_suppressor=MagicMock(),
    )
    assert stream.room_id == "cocina"
    assert stream.device_index == 3
    assert stream.listening is False
    assert stream.audio_buffer == []
    assert stream.command_start_time == 0.0


def test_multi_room_audio_loop_init(room_streams):
    """MultiRoomAudioLoop initializes with room streams."""
    follow_up = MagicMock()
    loop = MultiRoomAudioLoop(
        room_streams=room_streams,
        follow_up=follow_up,
    )
    assert len(loop.room_streams) == 2
    assert "cocina" in loop.room_streams
    assert "living" in loop.room_streams


def test_dedup_same_wakeword_within_window():
    """Wake words from different rooms within dedup window should keep strongest."""
    follow_up = MagicMock()
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=follow_up,
        dedup_window_ms=200,
    )

    now = time.time()
    # First detection
    assert loop._should_accept_wakeword("cocina", rms=0.5, timestamp=now) is True
    # Second detection within 200ms from different room — weaker
    assert loop._should_accept_wakeword("living", rms=0.3, timestamp=now + 0.1) is False
    # Second detection within 200ms from different room — stronger
    assert loop._should_accept_wakeword("living", rms=0.8, timestamp=now + 0.1) is True


def test_dedup_allows_after_window():
    """Wake words after dedup window are independent commands."""
    follow_up = MagicMock()
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=follow_up,
        dedup_window_ms=200,
    )

    now = time.time()
    assert loop._should_accept_wakeword("cocina", rms=0.5, timestamp=now) is True
    # After 300ms — independent command
    assert loop._should_accept_wakeword("living", rms=0.3, timestamp=now + 0.3) is True


def test_on_command_callback():
    """on_command registers callback for CommandEvent dispatch."""
    follow_up = MagicMock()
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=follow_up,
    )

    callback = AsyncMock()
    loop.on_command(callback)
    assert loop._on_command_callback is callback
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_multi_room_audio_loop.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create MultiRoomAudioLoop**

```python
# src/pipeline/multi_room_audio_loop.py
"""
Multi-Room Audio Loop
Opens one sounddevice InputStream per XVF3800 microphone.
Each room's stream independently detects wake words and captures commands.
Concurrent commands from different rooms are processed in parallel.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Awaitable, Tuple

import numpy as np
import sounddevice as sd

from src.pipeline.command_event import CommandEvent
from src.wakeword.detector import WakeWordDetector
from src.audio.echo_suppressor import EchoSuppressor
from src.conversation.follow_up_mode import FollowUpMode

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1280


@dataclass
class RoomStream:
    """Per-room audio capture state."""
    room_id: str
    device_index: int
    wake_detector: WakeWordDetector
    echo_suppressor: EchoSuppressor
    listening: bool = False
    audio_buffer: list = field(default_factory=list)
    command_start_time: float = 0.0


class MultiRoomAudioLoop:
    """
    Parallel audio capture from multiple XVF3800 microphones.

    Opens one sd.InputStream per room. Each stream runs wake word
    detection independently. When a stream captures a complete command,
    it dispatches a CommandEvent via asyncio.create_task (non-blocking),
    allowing concurrent processing of commands from different rooms.
    """

    def __init__(
        self,
        room_streams: Dict[str, RoomStream],
        follow_up: FollowUpMode,
        sample_rate: int = 16000,
        command_duration: float = 2.0,
        silence_threshold: float = 0.015,
        silence_duration_ms: int = 300,
        min_speech_ms: int = 300,
        dedup_window_ms: int = 200,
    ):
        self.room_streams = room_streams
        self.follow_up = follow_up
        self.sample_rate = sample_rate
        self.command_duration = command_duration
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms
        self.dedup_window_ms = dedup_window_ms

        self._running = False
        self._on_command_callback: Optional[
            Callable[[CommandEvent], Awaitable[dict]]
        ] = None
        self._on_post_command_callback: Optional[
            Callable[[dict, CommandEvent], Awaitable[None]]
        ] = None

        # Deduplication state
        self._last_wakeword_time: float = 0.0
        self._last_wakeword_room: str = ""
        self._last_wakeword_rms: float = 0.0

    def on_command(self, callback: Callable[[CommandEvent], Awaitable[dict]]):
        """Register callback for when command audio is captured."""
        self._on_command_callback = callback

    def on_post_command(
        self, callback: Callable[[dict, CommandEvent], Awaitable[None]]
    ):
        """Register callback for post-processing after command result."""
        self._on_post_command_callback = callback

    def _should_accept_wakeword(
        self, room_id: str, rms: float, timestamp: float
    ) -> bool:
        """
        Deduplicate wake words between rooms.

        If two rooms detect wake word within dedup_window_ms, keep
        the one with higher RMS (closer to the speaker). If outside
        the window, both are independent commands.
        """
        elapsed_ms = (timestamp - self._last_wakeword_time) * 1000

        if elapsed_ms < self.dedup_window_ms and self._last_wakeword_room:
            if self._last_wakeword_room == room_id:
                return True
            # Different room within window — echo?
            if rms > self._last_wakeword_rms:
                # This room is stronger — accept and replace
                logger.info(
                    f"Dedup: {room_id} (rms={rms:.3f}) replaces "
                    f"{self._last_wakeword_room} (rms={self._last_wakeword_rms:.3f})"
                )
                self._last_wakeword_time = timestamp
                self._last_wakeword_room = room_id
                self._last_wakeword_rms = rms
                return True
            else:
                # Weaker — likely echo
                logger.debug(
                    f"Dedup: ignoring {room_id} (rms={rms:.3f}), "
                    f"echo of {self._last_wakeword_room}"
                )
                return False

        # Outside window — independent command
        self._last_wakeword_time = timestamp
        self._last_wakeword_room = room_id
        self._last_wakeword_rms = rms
        return True

    async def start(self):
        """Initialize wake word detectors for all rooms."""
        for room_id, rs in self.room_streams.items():
            rs.wake_detector.load()
            logger.info(
                f"Room {room_id}: wake word loaded "
                f"(device={rs.device_index}, models={rs.wake_detector.get_active_models()})"
            )

    async def run(self):
        """
        Main loop — opens N InputStreams and polls for completed commands.

        Each room's audio callback runs on sounddevice's C thread.
        Command dispatch happens on the asyncio event loop via create_task.
        """
        self._running = True
        streams = []

        for room_id, rs in self.room_streams.items():
            callback = self._make_audio_callback(rs)
            try:
                stream = sd.InputStream(
                    device=rs.device_index,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=CHUNK_SIZE,
                    callback=callback,
                )
                stream.start()
                streams.append(stream)
                logger.info(f"Room {room_id}: audio stream started (device={rs.device_index})")
            except sd.PortAudioError as e:
                logger.error(f"Room {room_id}: failed to open device {rs.device_index}: {e}")

        logger.info(f"MultiRoomAudioLoop ready ({len(streams)}/{len(self.room_streams)} streams)")

        try:
            while self._running:
                await asyncio.sleep(0.05)

                for room_id, rs in self.room_streams.items():
                    if not rs.listening:
                        continue

                    is_complete, audio_data = self._check_vad_completion(rs)
                    if is_complete and audio_data is not None:
                        event = CommandEvent(
                            audio=audio_data,
                            room_id=room_id,
                            mic_device_index=rs.device_index,
                        )
                        # Non-blocking dispatch — allows concurrent commands
                        asyncio.create_task(self._dispatch_command(event))
                        rs.listening = False
                        rs.audio_buffer = []
        finally:
            for stream in streams:
                stream.stop()
                stream.close()

    async def stop(self):
        """Stop all audio streams."""
        self._running = False

    def _make_audio_callback(self, rs: RoomStream):
        """Create a sounddevice callback closure for one room."""

        def audio_callback(indata, frames, time_info, status):
            audio_chunk = indata[:, 0].copy()

            # Echo suppression per room
            if not rs.echo_suppressor.is_safe_to_listen:
                return
            should_process, reason = rs.echo_suppressor.should_process_audio(audio_chunk)
            if not should_process:
                return

            if not rs.listening:
                detection = rs.wake_detector.detect(audio_chunk)
                if detection:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if self._should_accept_wakeword(rs.room_id, rms, time.time()):
                        rs.listening = True
                        rs.command_start_time = time.time()
                        rs.audio_buffer = []
                        self.follow_up.start_conversation()
                        logger.info(f"Wake word in {rs.room_id} ({detection[0]}: {detection[1]:.2f})")

                elif self.follow_up.is_active:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if rms > 0.02 and rs.echo_suppressor.is_human_voice(audio_chunk):
                        rs.listening = True
                        rs.command_start_time = time.time()
                        rs.audio_buffer = []
            else:
                rs.audio_buffer.extend(audio_chunk)

        return audio_callback

    def _check_vad_completion(self, rs: RoomStream) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if a room's command capture is complete (VAD or timeout)."""
        elapsed = time.time() - rs.command_start_time
        elapsed_ms = elapsed * 1000

        if elapsed_ms < self.min_speech_ms:
            return False, None

        if not rs.audio_buffer:
            return False, None

        # Check for silence (VAD early exit)
        samples_per_ms = self.sample_rate // 1000
        silence_samples = int(self.silence_duration_ms * samples_per_ms)
        recent = rs.audio_buffer[-silence_samples:] if len(rs.audio_buffer) > silence_samples else rs.audio_buffer

        if recent:
            recent_array = np.array(recent, dtype=np.float32)
            rms = float(np.sqrt(np.mean(recent_array ** 2)))
            if rms < self.silence_threshold:
                audio_data = np.array(rs.audio_buffer, dtype=np.float32)
                logger.debug(f"VAD early exit in {rs.room_id}: {elapsed_ms:.0f}ms")
                return True, audio_data

        # Timeout
        if elapsed >= self.command_duration:
            audio_data = np.array(rs.audio_buffer, dtype=np.float32)
            return True, audio_data

        return False, None

    async def _dispatch_command(self, event: CommandEvent):
        """Dispatch a captured command via registered callback."""
        try:
            if self._on_command_callback:
                result = await self._on_command_callback(event)
            else:
                logger.warning("No on_command callback registered")
                result = {}

            if self._on_post_command_callback:
                await self._on_post_command_callback(result, event)
        except Exception as e:
            logger.error(f"Command dispatch failed for {event.room_id}: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_multi_room_audio_loop.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_multi_room_audio_loop.py
git commit -m "feat: add MultiRoomAudioLoop with parallel streams and dedup"
```

---

### Task 5: Modify `VoicePipeline` to accept `CommandEvent`

**Files:**
- Modify: `src/pipeline/voice_pipeline.py:37-47` (constructor) and `src/pipeline/voice_pipeline.py:128-142` (process_command)

**Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_voice_pipeline_room.py
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.pipeline.command_event import CommandEvent
from src.pipeline.voice_pipeline import VoicePipeline


@pytest.fixture
def pipeline():
    return VoicePipeline(
        audio_loop=MagicMock(),
        command_processor=MagicMock(),
        request_router=MagicMock(
            process_command=AsyncMock(return_value={"text": "ok", "success": True})
        ),
        response_handler=MagicMock(),
        feature_manager=MagicMock(),
    )


@pytest.mark.asyncio
async def test_process_command_accepts_command_event(pipeline):
    """VoicePipeline.process_command should accept CommandEvent."""
    event = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="cocina",
        mic_device_index=3,
    )
    result = await pipeline.process_command(event)
    assert result["success"] is True
    # Verify request_router received the event
    pipeline.request_router.process_command.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_process_command_backward_compat_audio(pipeline):
    """VoicePipeline.process_command should still accept raw audio."""
    audio = np.zeros(16000, dtype=np.float32)
    result = await pipeline.process_command(audio)
    assert result["success"] is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_voice_pipeline_room.py -v`
Expected: FAIL

**Step 3: Modify VoicePipeline.process_command**

In `src/pipeline/voice_pipeline.py`, update `process_command` to accept both `CommandEvent` and raw `np.ndarray` (backward compat):

```python
    async def process_command(self, audio_or_event) -> dict:
        """
        Process a complete audio command.

        Args:
            audio_or_event: CommandEvent with room metadata, or raw np.ndarray.

        Returns:
            Dict with text, intent, action, response, success, latency_ms, user.
        """
        if self.request_router:
            return await self.request_router.process_command(audio_or_event)
        return {"text": "", "success": False, "error": "No request router configured"}
```

Also add import at top:

```python
from src.pipeline.command_event import CommandEvent
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_voice_pipeline_room.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/voice_pipeline.py tests/unit/pipeline/test_voice_pipeline_room.py
git commit -m "feat: VoicePipeline accepts CommandEvent with room metadata"
```

---

### Task 6: Modify `RequestRouter` to use `RoomContext`

**Files:**
- Modify: `src/pipeline/request_router.py:70-95` (constructor), `src/pipeline/request_router.py:164-181` (process_command), `src/pipeline/request_router.py:183-275` (_process_command_orchestrated)

**Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_request_router_room.py
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.pipeline.command_event import CommandEvent
from src.rooms.room_context import RoomContext, ContextSource


@pytest.fixture
def router():
    from src.pipeline.request_router import RequestRouter
    return RequestRouter(
        command_processor=MagicMock(
            process_command=AsyncMock(return_value={
                "text": "apagá la luz",
                "timings": {"stt": 50},
                "user": None,
                "emotion": None,
            })
        ),
        response_handler=MagicMock(speak=MagicMock()),
        audio_manager=MagicMock(detect_source_zone=MagicMock(return_value=None)),
        room_context_manager=MagicMock(
            resolve_room=MagicMock(return_value=RoomContext(
                room_id="cocina",
                room_name="Cocina",
                display_name="la cocina",
                source=ContextSource.MICROPHONE,
                confidence=0.7,
                timestamp=0,
                entities={"light": "light.cocina"},
            ))
        ),
        orchestrator_enabled=False,
        chroma_sync=MagicMock(search_command=MagicMock(return_value=None)),
        ha_client=MagicMock(),
    )


@pytest.mark.asyncio
async def test_process_command_resolves_room_from_event(router):
    """RequestRouter should resolve room context from CommandEvent."""
    event = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="cocina",
        mic_device_index=3,
    )
    result = await router.process_command(event)
    router.room_context_manager.resolve_room.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_request_router_room.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'room_context_manager'`

**Step 3: Modify RequestRouter**

Add `room_context_manager` parameter to `__init__`:

```python
    def __init__(
        self,
        command_processor,
        response_handler,
        audio_manager,
        room_context_manager=None,  # NEW
        orchestrator=None,
        # ... rest unchanged
    ):
        # ... existing assignments ...
        self.room_context_manager = room_context_manager  # NEW
```

Modify `process_command` to extract room_id from CommandEvent:

```python
    async def process_command(self, audio_or_event) -> dict:
        """
        Process audio command. Accepts CommandEvent or raw np.ndarray.
        """
        from src.pipeline.command_event import CommandEvent

        if isinstance(audio_or_event, CommandEvent):
            audio = audio_or_event.audio
            room_id = audio_or_event.room_id
        else:
            audio = audio_or_event
            room_id = None

        if self.orchestrator_enabled and self._orchestrator:
            return await self._process_command_orchestrated(audio, room_id=room_id)
        else:
            return await self._process_command_legacy(audio, room_id=room_id)
```

In `_process_command_orchestrated`, after getting `user_id` (line ~217), add room context resolution:

```python
        # Resolve room context
        room_context = None
        if self.room_context_manager and room_id:
            room_context = self.room_context_manager.resolve_room(
                mic_zone_id=room_id,
                user_id=user_id,
            )

        # Use room_context zone instead of audio_manager detection
        if room_context:
            zone_id = f"zone_{room_context.room_id}"
            self.response_handler.set_active_zone(zone_id)
        else:
            zone_id = self.audio_manager.detect_source_zone(audio)
            if zone_id:
                self.response_handler.set_active_zone(zone_id)
```

Pass `room_context` to orchestrator and response_handler calls.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_request_router_room.py -v`
Expected: PASS

**Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/unit/pipeline/ -v`
Expected: All existing tests PASS (room_context_manager defaults to None)

**Step 6: Commit**

```bash
git add src/pipeline/request_router.py tests/unit/pipeline/test_request_router_room.py
git commit -m "feat: RequestRouter resolves room context from CommandEvent"
```

---

### Task 7: Modify `ResponseHandler` to route TTS by `RoomContext`

**Files:**
- Modify: `src/pipeline/response_handler.py:80-113` (speak method)

**Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_response_handler_room.py
import pytest
from unittest.mock import MagicMock

from src.pipeline.response_handler import ResponseHandler
from src.rooms.room_context import RoomContext, ContextSource


@pytest.fixture
def handler():
    return ResponseHandler(
        tts=MagicMock(synthesize=MagicMock(return_value=(b"audio", 16000)), sample_rate=16000),
        zone_manager=MagicMock(),
    )


def test_speak_with_room_context_routes_to_zone(handler):
    """speak() with RoomContext should route to correct MA1260 zone."""
    ctx = RoomContext(
        room_id="cocina",
        room_name="Cocina",
        display_name="la cocina",
        source=ContextSource.MICROPHONE,
        confidence=0.7,
        timestamp=0,
        entities={},
    )
    ctx.ma1260_zone = 3  # Set the zone directly for now

    handler.speak("Listo, apagué la luz", room_context=ctx)

    # Should have called play_to_zone with zone_cocina
    handler.zone_manager.play_to_zone.assert_called()


def test_speak_without_room_context_uses_active_zone(handler):
    """speak() without RoomContext should fall back to active zone."""
    handler.speak("Hola")
    # Should use default behavior (no zone routing)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/pipeline/test_response_handler_room.py -v`
Expected: FAIL with `TypeError: speak() got an unexpected keyword argument 'room_context'`

**Step 3: Add room_context parameter to speak()**

In `src/pipeline/response_handler.py`, modify the `speak` method signature:

```python
    def speak(
        self,
        text: str,
        zone_id: str = None,
        stream: bool = None,
        emotion_adjustment: dict = None,
        room_context=None,  # NEW
    ):
```

Add at the top of the method body, before determining target_zone:

```python
        # Resolve zone from room context if available
        if room_context and hasattr(room_context, 'ma1260_zone') and room_context.ma1260_zone:
            target_zone = f"zone_{room_context.room_id}"
        else:
            target_zone = zone_id or self._active_zone_id
```

Replace the existing `target_zone = zone_id or self._active_zone_id` line.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/pipeline/test_response_handler_room.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/response_handler.py tests/unit/pipeline/test_response_handler_room.py
git commit -m "feat: ResponseHandler routes TTS by RoomContext zone"
```

---

### Task 8: Wire everything in `main.py`

**Files:**
- Modify: `src/main.py`

**Step 1: Add imports for new components**

At the top of `src/main.py`, add:

```python
from src.rooms.room_context import RoomContextManager, RoomConfig
from src.presence.presence_detector import PresenceDetector
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.pipeline.command_event import CommandEvent
```

**Step 2: Build RoomContextManager from rooms config**

After the zone_manager creation block (~line 277), add:

```python
    # ----------------------------------------------------------------
    # Room Context (mic + BT per room)
    # ----------------------------------------------------------------
    rooms_config = config.get("rooms", {})
    room_context_manager = None
    presence_detector = None
    multi_room_loop = None

    if rooms_config.get("enabled", False):
        # Presence Detector (BLE)
        presence_config = config.get("presence", {})
        if presence_config.get("enabled", False):
            presence_detector = PresenceDetector(
                away_timeout=presence_config.get("away_timeout", 300),
                just_arrived_duration=presence_config.get("just_arrived_duration", 300),
            )

        # Room Context Manager
        room_context_manager = RoomContextManager(
            presence_detector=presence_detector,
            ha_client=ha_client,
            cross_validation=rooms_config.get("cross_validation", True),
            fallback_room=rooms_config.get("fallback_room", "living"),
        )

        # Register rooms and build room streams
        room_streams = {}
        room_wake_config = rooms_config.get("wake_word", wake_config)
        room_detection = rooms_config.get("detection", {})

        reserved_keys = {
            "enabled", "cross_validation", "fallback_room",
            "dedup_window_ms", "ma1260", "wake_word", "detection",
        }

        for room_id, room_cfg in rooms_config.items():
            if room_id in reserved_keys or not isinstance(room_cfg, dict):
                continue

            # Build RoomConfig
            rc = RoomConfig(
                room_id=room_id,
                name=room_cfg.get("name", room_id),
                display_name=room_cfg.get("display_name", room_id),
                mic_device_index=room_cfg.get("mic_device_index"),
                mic_device_name=room_cfg.get("mic_device_name"),
                bt_adapter=room_cfg.get("bt_adapter"),
                ma1260_zone=room_cfg.get("ma1260_zone"),
                output_mode=room_cfg.get("output_mode", "mono"),
                default_volume=room_cfg.get("default_volume", 50),
                noise_floor=room_cfg.get("noise_floor", 0.01),
                default_light=room_cfg.get("default_light"),
                default_climate=room_cfg.get("default_climate"),
                default_cover=room_cfg.get("default_cover"),
                default_media_player=room_cfg.get("default_media_player"),
                default_fan=room_cfg.get("default_fan"),
                motion_sensor=room_cfg.get("motion_sensor"),
                temperature_sensor=room_cfg.get("temperature_sensor"),
                humidity_sensor=room_cfg.get("humidity_sensor"),
                aliases=room_cfg.get("aliases", []),
                tts_speaker=room_cfg.get("tts_speaker"),
            )
            room_context_manager.add_room(rc)

            # Build RoomStream for MultiRoomAudioLoop
            if rc.mic_device_index is not None:
                wake_detector = WakeWordDetector(
                    models=[m.strip() for m in room_wake_config.get("model", "hey_jarvis").split(",")],
                    threshold=room_wake_config.get("threshold", 0.5),
                    refractory_period=2.0,
                )
                room_echo = EchoSuppressor(sample_rate=16000)
                room_streams[room_id] = RoomStream(
                    room_id=room_id,
                    device_index=rc.mic_device_index,
                    wake_detector=wake_detector,
                    echo_suppressor=room_echo,
                )

            # Register BLE zone
            if presence_detector and rc.bt_adapter:
                presence_detector.add_zone(
                    zone_id=room_id,
                    name=rc.name,
                    ble_adapter=rc.bt_adapter,
                    motion_sensor_entity=rc.motion_sensor,
                )

        # Also build zone_manager from rooms config
        if not zone_manager:
            ma1260_cfg = rooms_config.get("ma1260", {})
            ma1260 = MA1260Controller(
                connection_type=ma1260_cfg.get("connection_type", "simulation"),
                serial_port=ma1260_cfg.get("serial_port", "/dev/ttyUSB0"),
                baudrate=ma1260_cfg.get("baudrate", 9600),
                ip_address=ma1260_cfg.get("ip_address"),
                ip_port=ma1260_cfg.get("ip_port", 8080),
                audio_output_device=ma1260_cfg.get("audio_output_device"),
                default_source=MA1260Source(ma1260_cfg.get("default_source", 1)),
            )

            zone_list = []
            for room_id, rs in room_streams.items():
                rc = room_context_manager.get_room_config(room_id)
                if rc and rc.ma1260_zone is not None:
                    zone = Zone(
                        id=f"zone_{room_id}",
                        name=rc.name,
                        mic_device_index=rc.mic_device_index,
                        ma1260_zone=rc.ma1260_zone,
                        volume=rc.default_volume,
                        noise_floor=rc.noise_floor,
                        detection_threshold=room_detection.get("vad_threshold", 0.02),
                    )
                    zone_list.append(zone)

            zone_manager = ZoneManager(
                zones=zone_list,
                ma1260_controller=ma1260,
                detection_window_ms=room_detection.get("detection_window_ms", 500),
                priority_mode=room_detection.get("priority_mode", "loudest"),
            )

        # Build MultiRoomAudioLoop
        if room_streams:
            multi_room_loop = MultiRoomAudioLoop(
                room_streams=room_streams,
                follow_up=follow_up,
                sample_rate=16000,
                dedup_window_ms=rooms_config.get("dedup_window_ms", 200),
            )
            logger.info(f"Multi-room audio: {len(room_streams)} rooms configured")
```

**Step 3: Add WakeWordDetector import**

```python
from src.wakeword.detector import WakeWordDetector
```

**Step 4: Inject room_context_manager into RequestRouter**

Modify the `RequestRouter(...)` instantiation (~line 343) to include:

```python
    request_router = RequestRouter(
        command_processor=command_processor,
        response_handler=response_handler,
        audio_manager=audio_manager,
        room_context_manager=room_context_manager,  # NEW
        orchestrator=orchestrator,
        # ... rest unchanged
    )
```

**Step 5: Use MultiRoomAudioLoop instead of AudioLoop when available**

Replace the AudioLoop creation (~line 319-325) and pipeline assembly (~line 409-418):

```python
    # Use MultiRoomAudioLoop if rooms are configured, else fall back to AudioLoop
    active_audio_loop = multi_room_loop if multi_room_loop else audio_loop

    pipeline = VoicePipeline(
        audio_loop=active_audio_loop,
        command_processor=command_processor,
        request_router=request_router,
        response_handler=response_handler,
        feature_manager=feature_manager,
        chroma_sync=chroma,
        memory_manager=memory_manager,
        orchestrator=orchestrator,
    )
```

**Step 6: Start presence detector**

Before `await pipeline.run()`, add:

```python
    if presence_detector:
        await presence_detector.start()
        logger.info("Presence detector started")
```

**Step 7: Commit**

```bash
git add src/main.py
git commit -m "feat: wire MultiRoomAudioLoop, RoomContextManager, PresenceDetector in main.py"
```

---

### Task 9: Integration test — concurrent commands from 2 rooms

**Files:**
- Create: `tests/integration/test_multi_room_concurrent.py`

**Step 1: Write integration test**

```python
# tests/integration/test_multi_room_concurrent.py
"""
Integration test: two rooms process commands concurrently.
Verifies that commands from different rooms don't block each other.
"""
import asyncio
import numpy as np
import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from src.pipeline.command_event import CommandEvent
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream


@pytest.mark.asyncio
async def test_concurrent_commands_from_different_rooms():
    """Two commands from different rooms should process in parallel."""
    results = []
    processing_times = []

    async def mock_process(event: CommandEvent) -> dict:
        start = time.time()
        await asyncio.sleep(0.1)  # Simulate 100ms processing
        elapsed = time.time() - start
        processing_times.append(elapsed)
        results.append(event.room_id)
        return {"text": f"processed_{event.room_id}", "success": True}

    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(is_active=False),
    )
    loop.on_command(mock_process)

    # Dispatch two commands concurrently
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

    # Both should have processed
    assert len(results) == 2
    assert "cocina" in results
    assert "escritorio" in results

    # Total time should be ~100ms (parallel), not ~200ms (serial)
    assert total < 0.18, f"Commands took {total:.3f}s — should be parallel"


@pytest.mark.asyncio
async def test_dedup_prevents_echo_but_allows_concurrent():
    """Echo dedup should block echoes but allow genuine concurrent commands."""
    loop = MultiRoomAudioLoop(
        room_streams={},
        follow_up=MagicMock(),
        dedup_window_ms=200,
    )

    now = time.time()

    # Genuine concurrent command (>200ms apart)
    assert loop._should_accept_wakeword("cocina", 0.5, now) is True
    assert loop._should_accept_wakeword("escritorio", 0.5, now + 0.3) is True

    # Echo (within 200ms, weaker)
    now2 = time.time()
    assert loop._should_accept_wakeword("living", 0.8, now2) is True
    assert loop._should_accept_wakeword("hall", 0.3, now2 + 0.05) is False  # echo
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_multi_room_concurrent.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_multi_room_concurrent.py
git commit -m "test: add integration tests for concurrent multi-room commands"
```

---

### Task 10: Run full test suite and verify no regressions

**Step 1: Run all existing tests**

Run: `pytest tests/ -v --tb=short`
Expected: All 617+ tests PASS, no regressions from new code

**Step 2: If failures, fix them**

Common issues:
- Tests that mock `AudioLoop` may need updating if VoicePipeline constructor changed
- Tests that call `process_command(audio)` should still work (backward compat)

**Step 3: Final commit**

```bash
git add -A
git commit -m "fix: resolve test regressions from multi-room integration"
```
