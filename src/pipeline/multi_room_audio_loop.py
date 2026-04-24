"""
Multi-Room Audio Loop
Opens one sounddevice InputStream per XVF3800 microphone.
Each room's stream independently detects wake words and captures commands.
Concurrent commands from different rooms are processed in parallel.
"""

from __future__ import annotations

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
    # Barge-in (S3): acumulador de ms de voz sostenida durante TTS activo.
    # Se incrementa por cada chunk con RMS + is_human_voice==True; decae
    # con silencio. Dispara barge-in cuando supera `barge_in_min_duration_ms`.
    barge_in_accum_ms: float = 0.0


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
        response_handler=None,
        barge_in_enabled: bool = False,
        barge_in_rms_threshold: float = 0.03,
        barge_in_min_duration_ms: int = 200,
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

        # Barge-in (S3). `response_handler` puede setearse post-init via
        # `attach_response_handler()` — útil porque el ResponseHandler se
        # construye después del loop en main.py.
        self._response_handler = response_handler
        self.barge_in_enabled = barge_in_enabled
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.barge_in_min_duration_ms = barge_in_min_duration_ms

        self._running = False
        self._on_command_callback: Callable[[CommandEvent], Awaitable[dict]] | None = None
        self._on_post_command_callback: Callable[[dict, CommandEvent], Awaitable[None]] | None = None

        # Event loop capturado al `run()` — usado desde el audio_callback
        # (corre en thread del sounddevice, no en asyncio) para schedular
        # `_trigger_barge_in` via `run_coroutine_threadsafe`.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Deduplication state
        self._last_wakeword_time: float = 0.0
        self._last_wakeword_room: str = ""
        self._last_wakeword_rms: float = 0.0

    def attach_response_handler(self, response_handler) -> None:
        """Inyectar ResponseHandler post-init (útil por orden de DI en main.py)."""
        self._response_handler = response_handler

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
        # Guardamos el event loop para que el audio_callback (thread C del
        # sounddevice) pueda schedular corrutinas via run_coroutine_threadsafe
        # cuando detecta barge-in.
        self._loop = asyncio.get_running_loop()
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

            # Barge-in check (S3) — corre ANTES del echo suppressor porque
            # `is_safe_to_listen` retorna False mientras TTS está activo, lo
            # que bloquearía el flujo normal y nos dejaría sin detectar la
            # interrupción. El threshold + VAD + min_duration_ms filtran los
            # picos espurios y el eco residual del propio TTS.
            if (
                self.barge_in_enabled
                and self._response_handler is not None
                and self._response_handler.is_speaking
            ):
                rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                is_voice = (
                    rms > self.barge_in_rms_threshold
                    and rs.echo_suppressor.is_human_voice(audio_chunk)
                )
                if is_voice:
                    chunk_ms = (frames / self.sample_rate) * 1000
                    rs.barge_in_accum_ms += chunk_ms
                    if rs.barge_in_accum_ms >= self.barge_in_min_duration_ms:
                        # Schedular el cancel+listen en el event loop asyncio
                        # (este callback corre en thread C del sounddevice).
                        if self._loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._trigger_barge_in(rs),
                                self._loop,
                            )
                        rs.barge_in_accum_ms = 0.0
                else:
                    # Decay en silencio — protege contra picos aislados que
                    # no forman voz sostenida.
                    rs.barge_in_accum_ms = max(
                        0.0, rs.barge_in_accum_ms - 20.0
                    )
                # Mientras TTS habla, no procesamos wake word / captura normal.
                return

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

    async def _trigger_barge_in(self, rs: RoomStream) -> None:
        """
        Invocada desde el audio thread via `run_coroutine_threadsafe`.

        Corta el TTS activo y abre el modo listening en el room que detectó
        la interrupción (simula un wake word trigger). Si el ResponseHandler
        ya no estaba hablando (race con fin natural del TTS), sale silencioso.
        """
        if self._response_handler is None:
            return
        try:
            was_speaking = await self._response_handler.cancel()
        except Exception as e:
            logger.warning(f"Barge-in cancel error in {rs.room_id}: {e}")
            return

        if not was_speaking:
            # El TTS ya había terminado; no abrir listening para evitar
            # capturas espurias.
            return

        logger.info(f"⏹  Barge-in detectado en {rs.room_id}")
        rs.listening = True
        rs.command_start_time = time.time()
        rs.audio_buffer = []
        try:
            self.follow_up.start_conversation()
        except Exception as e:
            logger.debug(f"follow_up.start_conversation no-op: {e}")
