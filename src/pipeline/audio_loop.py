"""
Audio Loop Module
Main audio capture and wake word detection loop, extracted from VoicePipeline.

Handles:
- Audio capture via sounddevice InputStream
- Echo suppression (skip audio while TTS plays)
- Wake word detection
- Follow-up mode (accept speech without wake word)
- VAD-based command capture with early exit
- Ambient audio detection (fire-and-forget background tasks)
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable

import numpy as np
import sounddevice as sd

from src.pipeline.audio_manager import AudioManager
from src.audio import EchoSuppressor
from src.conversation import FollowUpMode

logger = logging.getLogger(__name__)


class AudioLoop:
    """
    Main audio capture and wake word detection loop.

    Encapsulates the audio callback closure, sounddevice InputStream,
    wake word detection, follow-up mode, VAD-based command capture,
    echo suppression, and ambient audio detection. Designed to be
    composed into VoicePipeline rather than subclassed.

    Attributes:
        audio_manager: AudioManager for wake word and VAD.
        echo_suppressor: EchoSuppressor for echo detection.
        follow_up: FollowUpMode for conversation continuity.
        sample_rate: Audio sample rate in Hz.
        ambient_detector: Optional AudioEventDetector for ambient events.
    """

    def __init__(
        self,
        audio_manager: AudioManager,
        echo_suppressor: EchoSuppressor,
        follow_up: FollowUpMode,
        sample_rate: int = 16000,
        ambient_detector=None,
    ):
        """
        Initialize AudioLoop.

        Args:
            audio_manager: AudioManager instance for wake word + VAD.
            echo_suppressor: EchoSuppressor instance for echo detection.
            follow_up: FollowUpMode instance for conversation continuity.
            sample_rate: Audio sample rate in Hz.
            ambient_detector: Optional AudioEventDetector for ambient events.
        """
        self.audio_manager = audio_manager
        self.echo_suppressor = echo_suppressor
        self.follow_up = follow_up
        self.sample_rate = sample_rate
        self.ambient_detector = ambient_detector

        self._running = False
        self._on_command_callback: Optional[Callable[[np.ndarray], Awaitable[dict]]] = None
        self._on_post_command_callback: Optional[Callable[[dict, np.ndarray], Awaitable[None]]] = None

    def on_command(self, callback: Callable[[np.ndarray], Awaitable[dict]]):
        """
        Register callback for when command audio is captured.

        The callback receives the captured audio data as a numpy array
        and must return a dict with at least 'text', 'success', 'action',
        and 'response' keys.

        Args:
            callback: Async function(audio_data: np.ndarray) -> dict.
        """
        self._on_command_callback = callback

    def on_post_command(self, callback: Callable[[dict, np.ndarray], Awaitable[None]]):
        """
        Register callback for post-processing after command result.

        Called after the command callback returns, for learning,
        follow-up notifications, etc.

        Args:
            callback: Async function(result: dict, audio_data: np.ndarray) -> None.
        """
        self._on_post_command_callback = callback

    async def start(self):
        """Initialize audio subsystems (wake word, ambient detector)."""
        self.audio_manager.load_wake_word()
        if self.ambient_detector:
            await self.ambient_detector.initialize()
            await self.ambient_detector.start()

    async def run(self):
        """
        Main audio loop. Blocks until stop() is called.

        Opens a sounddevice InputStream and continuously:
        1. Captures audio via callback (runs on C thread)
        2. Feeds chunks to wake word detector or follow-up mode
        3. Captures command audio with VAD-based early exit
        4. Dispatches captured commands via on_command callback
        5. Runs post-command processing via on_post_command callback
        6. Analyzes ambient audio in fire-and-forget background tasks
        """
        CHUNK_SIZE = 1280
        audio_buffer = []
        listening_for_command = False
        command_start_time = None
        ambient_buffer = []

        logger.info(
            f"AudioLoop ready. Wake word: '{self.audio_manager.wake_word_model}'"
        )
        logger.info(
            f"  Follow-up mode: {self.follow_up.follow_up_window}s window"
        )
        logger.info(
            f"  Ambient detection: {'ON' if self.ambient_detector else 'OFF'}"
        )
        logger.info(
            f"  Echo suppression: ON "
            f"(cooldown={self.echo_suppressor.config.post_speech_buffer_ms}ms)"
        )

        self._running = True

        def audio_callback(indata, frames, time_info, status):
            """
            Sounddevice callback — runs on a C thread.

            Must NOT contain any async calls. Only manipulates
            nonlocal buffers and flags.
            """
            nonlocal audio_buffer, listening_for_command, command_start_time, ambient_buffer

            audio_chunk = indata[:, 0].copy()

            # === Echo suppression ===
            # If KZA is speaking, ignore mic audio
            if not self.echo_suppressor.is_safe_to_listen:
                return

            # Check if captured audio is TTS echo
            should_process, reason = self.echo_suppressor.should_process_audio(audio_chunk)
            if not should_process:
                if reason == "echo_detected":
                    logger.debug("Echo detected, ignoring audio")
                return

            # Accumulate for ambient detection (always active)
            ambient_buffer.extend(audio_chunk)

            if not listening_for_command:
                # Detect wake word
                detection = self.audio_manager.detect_wake_word(audio_chunk)

                if detection:
                    listening_for_command = True
                    command_start_time = time.time()
                    audio_buffer = []
                    # Start conversation for follow-up mode
                    self.follow_up.start_conversation()

                # Follow-up mode: accept without wake word if conversation active
                elif self.follow_up.is_active:
                    # Simple VAD to detect speech
                    rms = np.sqrt(np.mean(audio_chunk ** 2))
                    if rms > 0.02:  # Activity threshold
                        # Verify it's not KZA's own voice
                        if self.echo_suppressor.is_human_voice(audio_chunk):
                            listening_for_command = True
                            command_start_time = time.time()
                            audio_buffer = []
                            logger.debug("Follow-up: capturing without wake word")
            else:
                audio_buffer.extend(audio_chunk)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )

        with stream:
            while self._running:
                await asyncio.sleep(0.05)

                # === Ambient detection (in parallel) ===
                if self.ambient_detector and len(ambient_buffer) >= self.sample_rate:
                    # Process 1 second of audio
                    chunk = np.array(ambient_buffer[:self.sample_rate], dtype=np.float32)
                    ambient_buffer = ambient_buffer[self.sample_rate:]

                    # Analyze in background (non-blocking)
                    task = asyncio.create_task(
                        self.ambient_detector.analyze_chunk(chunk)
                    )
                    task.add_done_callback(self._handle_task_error)

                # === Command processing ===
                if listening_for_command:
                    # Use early VAD to detect end of speech
                    is_complete, elapsed_ms, audio_data, early_exit = (
                        self.audio_manager.capture_command_with_vad(
                            audio_buffer,
                            command_start_time,
                            silence_threshold=0.015,
                            silence_duration_ms=300,
                            min_speech_ms=300
                        )
                    )

                    if is_complete and audio_data is not None:
                        if early_exit:
                            logger.debug(
                                f"VAD early exit: command captured in {elapsed_ms:.0f}ms"
                            )

                        # Dispatch command via callback
                        if self._on_command_callback:
                            result = await self._on_command_callback(audio_data)
                        else:
                            logger.warning("No on_command callback registered")
                            result = {}

                        # Post-command processing
                        if self._on_post_command_callback:
                            await self._on_post_command_callback(result, audio_data)

                        listening_for_command = False
                        audio_buffer = []

    async def stop(self):
        """Stop the audio loop and ambient detector."""
        self._running = False
        if self.ambient_detector:
            await self.ambient_detector.stop()

    @staticmethod
    def _handle_task_error(task: asyncio.Task):
        """Log errors from fire-and-forget background tasks."""
        if not task.cancelled() and task.exception():
            logger.error(f"Background audio task failed: {task.exception()}")
