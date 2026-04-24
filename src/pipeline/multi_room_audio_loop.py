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
from typing import Callable, Awaitable, Optional

import numpy as np
import sounddevice as sd

from src.pipeline.command_event import CommandEvent
from src.wakeword.detector import WakeWordDetector
from src.audio.echo_suppressor import EchoSuppressor
from src.conversation.follow_up_mode import FollowUpMode
from src.nlu.command_grammar import PartialCommand, parse_partial_command

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
    # Early dispatch (opt-in vía config): un background task transcribe+parsea
    # el audio acumulado cada N ms y settea `early_command` apenas el parser
    # tiene intent+entity. El polling loop dispatcha sin esperar silencio.
    early_task: Optional[asyncio.Task] = None
    early_command: Optional[PartialCommand] = None


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
        early_dispatch_enabled: bool = False,
        early_dispatch_interval_ms: int = 400,
        early_dispatch_min_audio_s: float = 0.6,
        stt=None,
        endpointing_enabled: bool = True,
        endpointing_short_ms: int = 150,
        endpointing_medium_ms: int = 300,
        endpointing_long_ms: int = 500,
    ):
        self.room_streams = room_streams
        self.follow_up = follow_up
        self.sample_rate = sample_rate
        self.command_duration = command_duration
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms
        self.dedup_window_ms = dedup_window_ms

        self.early_dispatch_enabled = early_dispatch_enabled
        self.early_dispatch_interval_ms = early_dispatch_interval_ms
        self.early_dispatch_min_audio_s = early_dispatch_min_audio_s
        self._stt = stt  # FastWhisperSTT — usado por el worker early parse

        # Endpointing adaptativo (S5): usa la señal del parser streaming
        # (`rs.early_command`) para decidir cuánto silencio esperar antes
        # de cerrar la captura. Con parser ready → corte rápido (short_ms).
        # Sin parser ready → corte normal (medium_ms). Sin señal alguna del
        # parser → espera más (long_ms, reservado para futuro uso).
        self.endpointing_enabled = endpointing_enabled
        self.endpointing_short_ms = endpointing_short_ms
        self.endpointing_medium_ms = endpointing_medium_ms
        self.endpointing_long_ms = endpointing_long_ms

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

                    # 1. Start early parse worker si corresponde (una sola vez por captura)
                    if (
                        self.early_dispatch_enabled
                        and self._stt is not None
                        and rs.early_task is None
                    ):
                        rs.early_task = asyncio.create_task(
                            self._early_parse_worker(rs)
                        )

                    # 2. Early dispatch si el worker detectó comando completo
                    if rs.early_command is not None:
                        audio_data = np.array(rs.audio_buffer, dtype=np.float32)
                        pc = rs.early_command
                        logger.info(
                            f"⚡ Early dispatch in {rs.room_id}: "
                            f"intent={pc.intent} entity={pc.entity} room={pc.room} "
                            f"({len(audio_data) / self.sample_rate * 1000:.0f}ms captured)"
                        )
                        event = CommandEvent(
                            audio=audio_data,
                            room_id=room_id,
                            mic_device_index=rs.device_index,
                            partial_command=pc,
                            early_dispatch=True,
                        )
                        asyncio.create_task(self._dispatch_command(event))
                        self._reset_listening(rs)
                        continue

                    # 3. Fallback: VAD silencio normal
                    is_complete, audio_data = self._check_vad_completion(rs)
                    if is_complete and audio_data is not None:
                        event = CommandEvent(
                            audio=audio_data,
                            room_id=room_id,
                            mic_device_index=rs.device_index,
                        )
                        asyncio.create_task(self._dispatch_command(event))
                        self._reset_listening(rs)
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

                        # Si el detector es WhisperWake y ya tiene el audio del comando
                        # inline (la misma utterance del wake word), lo usamos directo.
                        pop_fn = getattr(rs.wake_detector, "pop_pending_command_audio", None)
                        if callable(pop_fn):
                            inline_audio = pop_fn()
                            if inline_audio is not None and len(inline_audio) > 0:
                                rs.audio_buffer = list(inline_audio)
                                # Simular que ya terminó la captura (silencio final)
                                rs.command_start_time = time.time() - self.min_speech_ms / 1000 - 0.1
                                logger.info(
                                    f"Usando audio inline del wake word ({len(inline_audio)/16000:.2f}s) "
                                    f"— saltando captura post-wake"
                                )

                elif self.follow_up.is_active:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if rms > 0.02 and rs.echo_suppressor.is_human_voice(audio_chunk):
                        rs.listening = True
                        rs.command_start_time = time.time()
                        rs.audio_buffer = []
            else:
                rs.audio_buffer.extend(audio_chunk)

        return audio_callback

    def _adaptive_endpoint_threshold(self, rs: RoomStream) -> int:
        """
        Decide cuánto silencio esperar antes de cerrar una captura, usando la
        señal del parser streaming (`rs.early_command`) como heurística.

        - Parser ready_to_dispatch (intent+entity ya capturados) → short_ms
          (cerrar YA; ganancia ~150ms de latencia percibida en comandos
          completos).
        - Parser parcial o sin señal → medium_ms (comportamiento equivalente
          al silence_duration_ms clásico).

        Returns:
            Silencio requerido en ms antes de cerrar la captura.
        """
        if rs.early_command is not None and rs.early_command.ready_to_dispatch():
            return self.endpointing_short_ms
        return self.endpointing_medium_ms

    def _check_vad_completion(self, rs: RoomStream) -> tuple[bool, np.ndarray | None]:
        """Check if a room's command capture is complete (VAD or timeout)."""
        elapsed = time.time() - rs.command_start_time
        elapsed_ms = elapsed * 1000

        if elapsed_ms < self.min_speech_ms:
            return False, None

        if not rs.audio_buffer:
            return False, None

        samples_per_ms = self.sample_rate // 1000

        # Endpointing adaptativo (S5): intentar cerrar con threshold menor
        # cuando el parser streaming ya tiene comando completo. Si no logra
        # cerrar acá, cae al silence check clásico abajo.
        if self.endpointing_enabled:
            threshold_ms = self._adaptive_endpoint_threshold(rs)
            silence_needed = int(threshold_ms * samples_per_ms)
            if len(rs.audio_buffer) >= silence_needed and silence_needed > 0:
                recent_adaptive = rs.audio_buffer[-silence_needed:]
                recent_array = np.array(recent_adaptive, dtype=np.float32)
                rms_adaptive = float(np.sqrt(np.mean(recent_array ** 2)))
                if rms_adaptive < self.silence_threshold:
                    audio_data = np.array(rs.audio_buffer, dtype=np.float32)
                    logger.debug(
                        f"Adaptive endpoint in {rs.room_id}: "
                        f"threshold={threshold_ms}ms rms={rms_adaptive:.3f} "
                        f"elapsed={elapsed_ms:.0f}ms"
                    )
                    return True, audio_data

        # Silence check clásico (fallback si el adaptativo no cortó).
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

    def _reset_listening(self, rs: RoomStream) -> None:
        """Cierra el estado de captura y cancela el worker early si está activo."""
        rs.listening = False
        rs.audio_buffer = []
        rs.early_command = None
        if rs.early_task is not None:
            rs.early_task.cancel()
            rs.early_task = None

    async def _early_parse_worker(self, rs: RoomStream) -> None:
        """
        Corre mientras `rs.listening`. Cada `early_dispatch_interval_ms`:
          1. Snapshot del audio acumulado.
          2. Transcribe con el STT compartido.
          3. Parsea con `parse_partial_command`.
          4. Si `ready_to_dispatch()` → setea `rs.early_command` y sale.

        El polling loop principal ve `rs.early_command` y despacha.
        """
        interval_s = self.early_dispatch_interval_ms / 1000
        min_samples = int(self.early_dispatch_min_audio_s * self.sample_rate)
        try:
            while rs.listening and self._running:
                await asyncio.sleep(interval_s)
                if not rs.listening:
                    return
                buf_len = len(rs.audio_buffer)
                if buf_len < min_samples:
                    continue
                audio_snapshot = np.array(rs.audio_buffer, dtype=np.float32)
                try:
                    text, _ms = self._stt.transcribe(
                        audio_snapshot, sample_rate=self.sample_rate,
                    )
                except Exception as e:
                    logger.debug(f"Early transcribe error in {rs.room_id}: {e}")
                    continue
                if not text:
                    continue
                pc = parse_partial_command(text)
                logger.debug(
                    f"Early parse {rs.room_id}: {text!r} → "
                    f"intent={pc.intent} entity={pc.entity} room={pc.room}"
                )
                if pc.ready_to_dispatch():
                    rs.early_command = pc
                    return
        except asyncio.CancelledError:
            return

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
            logger.exception(f"Command dispatch failed for {event.room_id}: {e}")
