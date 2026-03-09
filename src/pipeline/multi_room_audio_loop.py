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
from typing import Callable, Awaitable

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
        room_streams: dict[str, RoomStream],
        follow_up: FollowUpMode,
        sample_rate: int = 16000,
        command_duration: float = 2.0,
        silence_threshold: float = 0.015,
        silence_duration_ms: int = 300,
        min_speech_ms: int = 300,
        dedup_window_ms: int = 500,
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
        self._on_command_callback: Callable[[CommandEvent], Awaitable[dict]] | None = None
        self._on_post_command_callback: Callable[[dict, CommandEvent], Awaitable[None]] | None = None

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
                logger.info(
                    f"Dedup: {room_id} (rms={rms:.3f}) replaces "
                    f"{self._last_wakeword_room} (rms={self._last_wakeword_rms:.3f})"
                )
                self._last_wakeword_time = timestamp
                self._last_wakeword_room = room_id
                self._last_wakeword_rms = rms
                return True
            else:
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

    def _check_vad_completion(self, rs: RoomStream) -> tuple[bool, np.ndarray | None]:
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
