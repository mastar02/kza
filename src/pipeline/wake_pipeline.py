"""Per-room wake word detection, pre-roll, AmbientGuard gating, clip capture, and barge-in.

Extracted from ``src/pipeline/multi_room_audio_loop.py`` so the real-time audio
callback, wake gating, and barge-in pipeline are isolated from stream recovery.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import deque
from contextvars import Context
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Event, Lock
from pathlib import Path
from typing import Callable, Awaitable, Optional
from uuid import uuid4

import numpy as np
import sounddevice as sd

from src.ambient.tap import MultiChannelTap
from src.audio.input_group import InputGroup
from src.audio.owned_input import InputNotAdmitted
from src.audio.xvf_controller import XvfController, XvfRebootOutcome
from src.audio.xvf_recovery import (
    REBOOT_DEFERRAL_ALERT_S, XvfRebootBudget, XvfRebootDecision, XvfRebootPolicy,
)
from src.pipeline.ambient_guard import AmbientGuard, GuardState, classify_outcome
from src.pipeline.command_event import CapturedTranscript, CommandEvent, TranscriptSource
from src.wakeword.wake_clip_writer import WakeClipWriter
from src.pipeline.capture_room import CaptureRoom
from src.pipeline.post_wake_endpoint import (
    CaptureTrigger, EndpointDecision, EndpointState, NoiseFloorTracker, PostWakeConfig, frame_rms,
)
from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.core.capture_opening import CaptureOpening, coherent_opening, current_capture_opening
from src.evaluation.provenance import CaptureLane
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission, producer_context
from src.memory.producer_scopes import ProducerScopes, current_producer_scope
from src.memory.prompt_context import SourcesInvalidated, run_private
from src.wakeword.detector import WakeWordDetector
from src.pipeline.text_utils import _text_likely_truncated
from src.audio.echo_suppressor import EchoSuppressor
from src.audio.direction_observer import (
    DirectionCalibration, DirectionObserver, parse_direction_observation,
)
from src.audio import microphone_status
from src.audio.microphone_status import (
    CaptureObservation, MicrophoneRoom, MicrophonesSnapshot, UsbInventory,
    build_microphone_snapshot,
)
from src.conversation.command_turns import CommandTurns
from src.conversation.follow_up_mode import FollowUpMode
from src.nlu.command_grammar import PartialCommand, parse_partial_command
from src.rooms.room_context import resolve_mic_usb_port, usb_port_to_alsa_card

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1280

# Warn after this observation budget without cancelling native publication.
# _audio_health_work retains one real write until it exits, even after timeout.
AUDIO_HEALTH_WRITE_TIMEOUT_S = 1.0

from src.pipeline.room_stream_manager import RoomStream, _CaptureHolder

import sys as _sys

def _facade():
    return _sys.modules.get("src.pipeline.multi_room_audio_loop")


def _pcm_context(parts):
    """Reduce retained PCM contributions; unknown is never a latest-value label."""
    opening, conflict = coherent_opening(part[1] for part in parts)
    channels = {part[2] for part in parts if part[2] is not None}
    conflict = conflict or len(channels) > 1
    if conflict or not parts or any(part[1] is None or part[2] is None for part in parts):
        return None, None, conflict
    return opening, next(iter(channels)), False

def compute_wake_vad(audio_chunk, vad_predict) -> float | None:
    """Prob de Silero sobre el audio del wake-trigger (fail-safe → None).

    vad_predict es el predictor de make_silero_predictor (devuelve prob máx
    sobre las ventanas de 512 del chunk). Cualquier fallo → None: el guard
    usa entonces el umbral fijo (nunca más permisivo por falta de señal).
    """
    if vad_predict is None:
        return None
    try:
        mono = audio_chunk
        if getattr(mono, "ndim", 1) > 1:
            mono = mono[:, 0]
        return float(vad_predict(np.ascontiguousarray(mono)))
    except Exception:
        return None

def _fmt_vad(value: float | None) -> str:
    """Format a Silero probability for logs ('na' when unavailable)."""
    return "na" if value is None else f"{value:.2f}"


class WakePipelineMixin:
    """Audio callback, wake acceptance, pre-roll, clip capture, and barge-in mixin."""

    def _should_accept_wakeword(
        self, room_id: str, rms: float, timestamp: float,
        wake_score: float = 1.0, wake_vad: float | None = None,
    ) -> bool:
        """
        Deduplicate wake words between rooms.

        If two rooms detect wake word within dedup_window_ms, keep
        the one with higher RMS (closer to the speaker). If outside
        the window, both are independent commands.
        """
        # AmbientGuard primero: en COOLDOWN rechaza barato (sin tocar dedup);
        # en STRICT exige score alto. El detector queda en su threshold base —
        # la decisión adaptativa vive acá, en un solo lugar testeable.
        if self._guard is not None:
            decision = self._guard.on_wake(room_id, wake_score, rms, wake_vad=wake_vad)
            if not decision.accept:
                logger.info(
                    f"[AmbientGuard] wake rechazado en {room_id} "
                    f"({decision.reason}, state={decision.state.value}, "
                    f"score={wake_score:.2f}, rms={rms:.4f}, vad={getattr(_facade(), '_fmt_vad', _fmt_vad)(wake_vad)})"
                )
                # Liberar el refractario del detector: el rechazo del guard NO
                # debe consumir la ventana de 2s. Visto en vivo (2026-06-05,
                # escenario 2): un frame de TV a 0.53 abría el refractario y el
                # "Nexa" real a 0.91 que llegaba 80ms después era suprimido por
                # detect() sin llegar nunca al guard. SOLO acá — el rechazo por
                # dedup (eco cross-room) sí debe mantener el refractario.
                rs = self.room_streams.get(room_id)
                if rs is not None:
                    reset_fn = getattr(rs.wake_detector, "reset_refractory", None)
                    if callable(reset_fn):
                        reset_fn()
                return False

        # Señal shadow anti-TV (spec 2026-06-06 §5.1, Fase 2): si el ambient
        # path vio una utterance 'tv' reciente, loguear qué haría el guard.
        # SOLO log — el flip a enforcement es Fase 3, con una semana de datos.
        # Fail-open: error del transcriber jamás toca el wake.
        if self._ambient_transcriber is not None:
            try:
                if self._ambient_transcriber.tv_active_recent(room_id):
                    logger.info(
                        f"[Ambient-shadow] wake en {room_id} con TV activa "
                        f"(score={wake_score:.2f}, rms={rms:.4f}) — "
                        f"enforcement habría exigido strict_wake_score"
                    )
            except Exception as e:
                logger.debug(f"[Ambient-shadow] señal no disponible: {e}")

        # Pre-gate de energía: descarta near-silence antes de capturar/transcribir.
        # Default 0.0 = off. Ver __init__ (calibrar en repro; AGC infla el piso).
        if self.min_wake_rms > 0.0 and rms < self.min_wake_rms:
            logger.debug(
                f"Wake en {room_id} rechazado por RMS bajo: "
                f"{rms:.4f} < {self.min_wake_rms}"
            )
            return False

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

    def _make_audio_callback(self, rs: RoomStream, holder: _CaptureHolder | None = None):
        """Create a sounddevice callback closure for one room."""
        if holder is None:
            holder = self._microphone_captures.setdefault(rs.room_id, _CaptureHolder(CaptureObservation()))

        room = self._capture_rooms[rs.room_id]

        def audio_callback(indata, frames, time_info, status):
            opening = holder.opening
            tap = self._ambient_tap
            ambient_admitted = not isinstance(tap, MultiChannelTap) or tap.accepting
            ambient_boundary = tap.input_boundary(rs.room_id) if isinstance(tap, MultiChannelTap) else None
            ambient_epoch = tap.input_epoch(rs.room_id) if isinstance(tap, MultiChannelTap) else None
            voice_epoch = room.input_epoch
            ticket = room.callbacks.enter(room.input_boundary,
                branch=tap.ingress_token if isinstance(tap, MultiChannelTap) and ambient_admitted else None)
            ambient = None
            try:
                if ticket is None or not room.callbacks.accepting or self._closed or not self._inputs.accepting or self._microphone_captures.get(rs.room_id) is not holder:
                    return
                # Sample acquisition time before any array method can stall.
                now_wall = self._microphone_wall_clock()
                timestamp = now_wall - frames / self.sample_rate
                room.callbacks.stamp(ticket, timestamp, room.source)
                if self._closed:
                    return
                if self._microphone_captures.get(rs.room_id) is not holder:
                    return
                nonzero = bool(indata.any())
                if self._microphone_captures.get(rs.room_id) is not holder:
                    return
                now_mono = self._microphone_monotonic()
                observation = holder.observation
                holder.observation = replace(
                    observation, frames_received=observation.frames_received + frames,
                    last_callback_at=now_wall, last_callback_monotonic=now_mono,
                    last_nonzero_at=now_wall if nonzero else observation.last_nonzero_at,
                    nonzero_seen=observation.nonzero_seen or nonzero,
                )
                # Watchdog heartbeat: marca que el stream entregó AUDIO. Primera
                # línea, nunca lanza — si esto deja de actualizarse, el mic murió
                # y _stream_watchdog dispara recovery.
                #
                # `indata.any()` y no un refresh incondicional (2026-07-29): un
                # XVF3800 con el endpoint isócrono degradado enumera, deja abrir el
                # stream y hace que PortAudio siga invocando este callback — pero
                # con buffers de CEROS. Marcando cada invocación, el watchdog veía
                # un stream sano y el sistema quedó sordo ~7 min sin una alerta,
                # con el servicio en `active`. Misma lección que "arecord exit 0 ≠
                # audio": contar muestras no-cero, no llamadas.
                #
                # El criterio es no-cero, NO un umbral de RMS: el silencio real de
                # un mic vivo trae un piso de ruido diminuto pero distinto de cero,
                # y no debe confundirse con un mic muerto.
                if nonzero:
                    rs.last_frame_ts = getattr(_facade(), 'time', time).monotonic()

                if not room.accepts(timestamp, boundary=ticket.boundary):
                    return

                # Tee al ambient path (spec 2026-06-06): SIEMPRE primero — el
                # ambient quiere todo el audio, incluso lo que el barge-in o el
                # echo suppressor descartan para el command path. O(1). El
                # try/except hace el fail-open EXPLÍCITO: una excepción acá
                # abortaría el stream de audio del command path (thread C).
                if tap is not None and ambient_admitted and (not isinstance(tap, MultiChannelTap)
                        or tap.accepts(rs.room_id, timestamp, ambient_boundary,epoch=ambient_epoch)):
                    try:
                        tts_now = (
                            self._response_handler is not None
                            and self._response_handler.is_speaking
                        )
                        ambient = indata.copy()
                        if self._closed or self._microphone_captures.get(rs.room_id) is not holder or not room.accepts(timestamp, boundary=ticket.boundary):
                            return
                        if isinstance(tap, MultiChannelTap):
                            tap.push(rs.room_id, ambient, ts=timestamp, tts_active=tts_now,
                                boundary=ambient_boundary, epoch=ambient_epoch, opening=opening)
                        else:
                            tap.push(rs.room_id, ambient, ts=timestamp, tts_active=tts_now)
                        ambient = None
                    except Exception:
                        pass  # perder un chunk ambiental es aceptable; el stream no
                    finally:
                        ambient = None
                        room.callbacks.release_branch(ticket)

                room.callbacks.release_branch(ticket)

                if self._microphone_captures.get(rs.room_id) is holder:
                    self._capture_rooms[rs.room_id].push(
                        timestamp, indata, frames, boundary=ticket.boundary, epoch=voice_epoch, opening=opening)
            except Exception:
                room.callbacks.failed()
            finally:
                indata = ambient = None
                if ticket is not None:
                    room.callbacks.leave(ticket)
                ticket = None

        return audio_callback

    def _make_audio_processor(self, rs: RoomStream):
        """Build the native, serialized state processor; never a device callback."""
        def process(indata, frames, time_info, status):
            self._capture_rooms[rs.room_id].require_live()
            # Canal configurado per-room con fallback seguro: si el device no
            # tiene ese canal (mic mono UAC1.0), usar ch0 y avisar una vez.
            ch = rs.capture_channel
            if ch and indata.shape[1] <= ch:
                if not rs.channel_warned:
                    rs.channel_warned = True
                    logger.warning(
                        f"Room {rs.room_id}: capture_channel={ch} no existe "
                        f"(device de {indata.shape[1]}ch) — fallback a canal 0"
                    )
                ch = 0
            audio_chunk = indata[:, ch].copy()
            opening = current_capture_opening()
            opening = replace(opening) if opening is not None else None
            if opening is not None and (opening.channels != indata.shape[1]
                    or opening.sample_rate_hz != self.sample_rate):
                opening = None
            part = (audio_chunk.size, opening, ch)

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
                                self._tracked_barge_in(rs, self._capture_rooms[rs.room_id].source),
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
            # Wake models need continuous time, including quiet frames. Command
            # endpointing also needs trailing silence; dropping it concatenates
            # speech and forces capture to wait for the timeout.
            if not should_process and reason != "silence":
                return

            retained = (rs.audio_metadata if rs.listening else list(rs.preroll_metadata or ()))
            recurrent_conflict = (opening is not None and rs.wake_opening is not None
                and (opening != rs.wake_opening or ch != rs.wake_channel))
            if recurrent_conflict or getattr(_facade(), '_pcm_context', _pcm_context)([*retained, part])[2]:
                self._clear_capture(rs)
            if opening is not None:
                rs.wake_opening, rs.wake_channel = opening, ch

            if not rs.listening:
                # Pre-roll: acumular el audio reciente para sembrar el buffer al
                # disparar el wake (recupera el comando dicho durante la latencia
                # de detección de openwakeword). 0 chunks = desactivado.
                if self._preroll_maxchunks > 0:
                    if rs.preroll is None:
                        rs.preroll = deque(maxlen=self._preroll_maxchunks)
                    if rs.preroll_metadata is None:
                        rs.preroll_metadata = deque(
                            [(len(chunk), None, None) for chunk in rs.preroll], maxlen=self._preroll_maxchunks)
                    rs.preroll.append(audio_chunk)
                    rs.preroll_metadata.append(part)
                if self._post_wake is not None:
                    if rs.noise_floor is None:
                        rs.noise_floor = NoiseFloorTracker(self._post_wake, self.sample_rate, CHUNK_SIZE)
                    rs.noise_floor.observe(float(np.sqrt(np.mean(audio_chunk ** 2))))
                room = self._capture_rooms[rs.room_id]
                room.require_live()
                backend = getattr(rs.wake_detector, 'whisper', None)
                native_lock = self.native_model_locks.get(id(backend))
                if native_lock is not None:
                    with model_admission(native_lock, room.require_live):
                        detect = (getattr(rs.wake_detector, 'detect_with_provenance', None)
                            if callable(getattr(type(rs.wake_detector), 'detect_with_provenance', None)) else None)
                        detection = (detect(audio_chunk, opening=opening, selected_channel=ch)
                            if callable(detect) else rs.wake_detector.detect(audio_chunk))
                else:
                    detect = (getattr(rs.wake_detector, 'detect_with_provenance', None)
                        if callable(getattr(type(rs.wake_detector), 'detect_with_provenance', None)) else None)
                    detection = (detect(audio_chunk, opening=opening, selected_channel=ch)
                        if callable(detect) else rs.wake_detector.detect(audio_chunk))
                room.require_live()
                if detection:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    predict = self._wake_vad_predictors.get(rs.room_id) or None
                    wake_vad = getattr(_facade(), 'compute_wake_vad', compute_wake_vad)(audio_chunk, predict)
                    room.require_live()
                    if self._should_accept_wakeword(
                        rs.room_id, rms, getattr(_facade(), 'time', time).time(),
                        wake_score=detection[1], wake_vad=wake_vad,
                    ):
                        rs.capture_generation += 1
                        rs.transcript = None
                        rs.capture_complete = False
                        rs.listening = True
                        rs.wake_score = detection[1]
                        rs.wake_vad = wake_vad
                        rs.capture_trigger = CaptureTrigger.WAKE
                        rs.endpoint = None
                        rs.command_start_time = getattr(_facade(), 'time', time).time()
                        # Sembrar con el pre-roll (en vez de []) para no perder el
                        # verbo dicho durante la latencia. Backdate command_start_time
                        # para que el endpointing cuente la duración real del buffer.
                        if rs.preroll:
                            rs.audio_buffer = list(np.concatenate(list(rs.preroll)))
                            rs.audio_metadata = list(rs.preroll_metadata)
                            rs.inline_provenance = None
                            rs.command_start_time = getattr(_facade(), 'time', time).time() - len(rs.audio_buffer) / self.sample_rate
                            rs.preroll.clear()
                            rs.preroll_metadata.clear()
                        else:
                            rs.audio_buffer = []
                            rs.audio_metadata = []
                            rs.inline_provenance = None
                        # Endpoint timing starts at the wake, not at the pre-roll.
                        rs.wake_sample_index = len(rs.audio_buffer)
                        # Persistir el clip del trigger para entrenamiento
                        # (fail-open: un writer roto jamás corta la captura).
                        if self._wake_clip_writer is not None and rs.audio_buffer:
                            try:
                                self._wake_clip_writer.submit(
                                    rs.room_id, detection[1], rs.audio_buffer,
                                )
                            except Exception as e:
                                logger.warning(f"[WakeClip] submit error: {e}")
                        # En STRICT/COOLDOWN no abrir follow_up: la ventana
                        # abierta con TV era parte de la cascada del 06-04.
                        if self._guard is None or self._guard.follow_up_allowed(rs.room_id):
                            self.follow_up.start_conversation()
                            self._conversation_room = rs.room_id
                        logger.info(f"Wake word in {rs.room_id} ({detection[0]}: {detection[1]:.2f}, "
                                    f"vad={getattr(_facade(), '_fmt_vad', _fmt_vad)(wake_vad)})")

                        # Peek wake text first to decide if it was truncated.
                        # Si el wake terminó con preposición/artículo/elipsis,
                        # NO saltamos la captura post-wake — el comando real
                        # probablemente sigue. (Fix 1A — visto 2026-05-02 06:36:51:
                        # 'Nexa prendé la luz del...' → light.bano por adivinanza
                        # del LLM con texto cortado.)
                        peek_text_fn = getattr(rs.wake_detector, "peek_pending_text", None)
                        wake_text_peek = peek_text_fn() if callable(peek_text_fn) else None
                        wake_was_truncated = getattr(_facade(), '_text_likely_truncated', _text_likely_truncated)(wake_text_peek or "")

                        # Si el detector es WhisperWake y ya tiene el audio del comando
                        # inline (la misma utterance del wake word), lo usamos directo.
                        pop_fn = getattr(rs.wake_detector, "pop_pending_command_audio", None)
                        inline_audio = None
                        if callable(pop_fn):
                            pop_metadata = getattr(rs.wake_detector, 'pop_pending_command_metadata', None)
                            inline_metadata = pop_metadata() if callable(pop_metadata) else (None, None, None)
                            inline_audio = pop_fn()
                            if inline_audio is not None and len(inline_audio) > 0:
                                rs.audio_buffer = list(inline_audio)
                                rs.wake_sample_index = 0
                                rs.capture_trigger = CaptureTrigger.INLINE
                                rs.endpoint = None
                                inline_opening, inline_channel, rs.inline_provenance = inline_metadata
                                rs.audio_metadata = [(len(inline_audio), inline_opening, inline_channel)]
                                if wake_was_truncated:
                                    logger.info(
                                        f"Audio inline ({len(inline_audio)/16000:.2f}s) "
                                        f"+ continuando captura post-wake — "
                                        f"texto truncado: {wake_text_peek!r}"
                                    )
                                    # No simulamos fin de captura → sigue capturando audio.
                                else:
                                    rs.command_start_time = getattr(_facade(), 'time', time).time() - self.min_speech_ms / 1000 - 0.1
                                    logger.info(
                                        f"Usando audio inline del wake word ({len(inline_audio)/16000:.2f}s) "
                                        f"— saltando captura post-wake"
                                    )
                        # Bind text only to the detector's inline snapshot.
                        # Any subsequently captured samples invalidate reuse.
                        pop_text_fn = getattr(rs.wake_detector, "pop_pending_text", None)
                        if callable(pop_text_fn):
                            pop_metadata = getattr(rs.wake_detector, 'pop_pending_transcript_metadata', None)
                            text_metadata = pop_metadata() if callable(pop_metadata) else (None, None, None)
                            popped = pop_text_fn()
                            rs.wake_text = popped if not wake_was_truncated else None
                            if isinstance(rs.wake_text, str) and callable(pop_fn) and inline_audio is not None and len(inline_audio) > 0:
                                rs.transcript = CapturedTranscript.from_audio(
                                    rs.wake_text, TranscriptSource.WAKE,
                                    np.asarray(rs.audio_buffer, dtype=np.float32),
                                    transcript_provenance=text_metadata[0], transcript_span=text_metadata[1],
                                    transcript_truncated=text_metadata[2],
                                )
                    else:
                        # Wake RECHAZADO por el guard (STRICT score / COOLDOWN):
                        # persistir igual para el dataset de re-entrenamiento. Los
                        # 0.40-0.45 que STRICT mata son comandos far-field REALES
                        # (voz del usuario 0.41-0.65) = positivos hoy perdidos; los
                        # de TV son hard-negatives. Fail-open: jamás bloquea ni
                        # muta el estado del room (no toca preroll/listening).
                        if self._wake_clip_writer is not None and rs.preroll:
                            try:
                                self._wake_clip_writer.submit(
                                    rs.room_id, detection[1],
                                    np.concatenate(list(rs.preroll)),
                                    accepted=False,
                                )
                            except Exception as e:
                                logger.warning(f"[WakeClip] submit (rejected) error: {e}")

                elif self.follow_up.is_active and (
                    self._guard is None or self._guard.follow_up_allowed(rs.room_id)
                ):
                    if self._post_wake is not None:
                        self._maybe_start_follow_up(rs, audio_chunk, part)
                    else:
                        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                        if rms > 0.02 and rs.echo_suppressor.is_human_voice(audio_chunk):
                            rs.listening = True
                            rs.capture_generation += 1
                            rs.command_start_time = getattr(_facade(), 'time', time).time()
                            rs.audio_buffer = []
                            rs.audio_metadata = []
                            rs.inline_provenance = None
                            rs.capture_trigger = CaptureTrigger.FOLLOW_UP
            else:
                rs.audio_buffer.extend(audio_chunk)
                rs.audio_metadata.append(part)
                rs.inline_provenance = None

        return process

    async def _tracked_barge_in(self, rs: RoomStream, source) -> None:
        task = asyncio.current_task()
        self._capture_tasks.add(task)
        try:
            with self._scopes.operation((source,) if source is not None else ()):
                await run_private(self._trigger_barge_in(rs), lambda: self._source_live(source))
        except SourcesInvalidated:
            pass
        finally:
            self._capture_tasks.discard(task)
            rs = None

    def _begin_barge_capture(self, rs: RoomStream) -> None:
        self._capture_rooms[rs.room_id].require_live()
        self._reset_listening(rs)
        rs.command_start_time = getattr(_facade(), 'time', time).time()
        rs.capture_trigger = CaptureTrigger.BARGE_IN
        try:
            self.follow_up.start_conversation()
            self._conversation_room = rs.room_id
        except Exception as error:
            logger.debug('Follow-up unavailable after barge-in: %s', type(error).__name__)
        self._capture_rooms[rs.room_id].require_live()
        rs.listening = True

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
            was_speaking = await self._response_handler.cancel(zone_id=f"zone_{rs.room_id}")
        except Exception as e:
            logger.warning(f"Barge-in cancel error in {rs.room_id}: {e}")
            return

        if not was_speaking:
            # El TTS ya había terminado; no abrir listening para evitar
            # capturas espurias.
            return

        logger.info(f"⏹  Barge-in detectado en {rs.room_id}")
        await self._capture_rooms[rs.room_id].call(self._begin_barge_capture, rs)
