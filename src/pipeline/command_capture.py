"""Post-wake command endpointing, follow-up capture, early parse, and command dispatch.

Extracted from ``src/pipeline/multi_room_audio_loop.py`` so capture completion,
follow-up onset, and ``CommandEvent`` dispatch live in a dedicated cohesive module.
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
from src.wakeword.whisper_wake import _text_likely_truncated
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

from src.pipeline.room_stream_manager import RoomStream
from src.pipeline.wake_pipeline import _fmt_vad

import sys as _sys

def _facade():
    return _sys.modules.get("src.pipeline.multi_room_audio_loop")



class CommandCaptureMixin:
    """Endpointing, follow-up onset, early parsing, and dispatch mixin."""

    def _room_vad(self, rs: RoomStream):
        """Per-room Silero predictor already loaded for the wake, or None."""
        return self._wake_vad_predictors.get(rs.room_id) or None

    def _room_floor(self, rs: RoomStream) -> float:
        """Current noise floor of the room (post-wake rule only)."""
        if rs.noise_floor is None:
            rs.noise_floor = NoiseFloorTracker(self._post_wake, self.sample_rate, CHUNK_SIZE)
        return rs.noise_floor.floor()

    def _maybe_start_follow_up(self, rs: RoomStream, audio_chunk, part) -> None:
        """Open a follow-up capture on a sustained speech onset in the conversation room.

        Replaces the absolute ``rms > 0.02`` start, unreachable in quiet rooms.
        Only the room that opened the conversation may start one (the
        follow-up window is global). The onset needs ``min_speech_ms`` of
        consecutive frames that pass the level test relative to the room's
        floor; Silero never starts a follow-up alone (it cannot tell TV from
        people), it only helps the endpoint once the capture is open. The
        ambient guard check stays with the caller; the CommandGate still judges
        the transcript. A short pre-roll keeps the onset syllable.
        """
        config = self._post_wake
        if self._conversation_room != rs.room_id:
            rs.onset_frames = 0
            return
        level = frame_rms(audio_chunk) >= self._room_floor(rs) * config.noise_ratio
        if not level or not rs.echo_suppressor.is_human_voice(audio_chunk):
            rs.onset_frames = 0
            return
        rs.onset_frames += 1
        needed = max(1, math.ceil(config.min_speech_ms * self.sample_rate / (1000 * CHUNK_SIZE) - 1e-9))
        if rs.onset_frames < needed:
            return
        onset, rs.onset_frames = rs.onset_frames, 0
        keep = onset + int(round(config.follow_up_preroll_s * self.sample_rate / CHUNK_SIZE))
        if rs.preroll:
            chunks = list(rs.preroll)[-keep:]
            metadata = list(rs.preroll_metadata or ())[-len(chunks):]
            if len(metadata) != len(chunks):
                chunks, metadata = [audio_chunk], [part]
            rs.preroll.clear()
            rs.preroll_metadata.clear()
        else:
            chunks, metadata = [audio_chunk], [part]
        rs.listening = True
        rs.capture_generation += 1
        rs.audio_buffer = list(np.concatenate(chunks))
        rs.audio_metadata = metadata
        rs.inline_provenance = None
        # The onset chunks themselves count as post-"wake" speech.
        onset_samples = sum(len(chunk) for chunk in chunks[-onset:])
        rs.wake_sample_index = len(rs.audio_buffer) - onset_samples
        rs.command_start_time = getattr(_facade(), 'time', time).time() - len(rs.audio_buffer) / self.sample_rate
        rs.capture_trigger = CaptureTrigger.FOLLOW_UP
        rs.wake_vad = None
        rs.endpoint = None
        logger.info("[endpoint] follow-up capture start room=%s", rs.room_id)

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

    def _passes_spenergy_gate(self, rs: RoomStream) -> bool:
        """True si hubo voz (SPENERGY ≥ umbral) durante la captura, o si el gate
        no aplica. FAIL-OPEN: sin controller o sin muestras → True (procesa).

        Bloquea solo cuando el pico de SPENERGY durante [command_start_time, now]
        quedó por debajo del umbral = secador/silencio → Whisper alucinaría.
        """
        ctrl = self._xvf_by_room.get(rs.room_id)
        if ctrl is None or not self.spenergy_gate_enabled:
            return True
        try:
            peak = ctrl.peak_since(rs.command_start_time)
        except Exception as e:  # fail-open ante cualquier error del controller
            logger.debug(f"SPENERGY gate fail-open ({e})")
            return True
        if peak is None:
            return True
        if peak < self.spenergy_threshold:
            logger.info(
                f"[SPENERGY-gate] descartado en {rs.room_id}: pico {peak:.0f} < "
                f"{self.spenergy_threshold:.0f} (secador/silencio, no se transcribe)"
            )
            return False
        return True

    def _check_vad_completion(self, rs: RoomStream) -> tuple[bool, np.ndarray | None]:
        """Check if a room's command capture is complete (VAD or timeout)."""
        rs.capture_complete = False
        if self._post_wake is not None:
            return self._check_post_wake_completion(rs)
        elapsed = getattr(_facade(), 'time', time).time() - rs.command_start_time
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
                    rs.capture_complete = True
                    return True, audio_data

        # Silence check clásico (fallback si el adaptativo no cortó).
        silence_samples = int(self.silence_duration_ms * samples_per_ms)
        recent = rs.audio_buffer[-silence_samples:] if len(rs.audio_buffer) > silence_samples else rs.audio_buffer

        if recent and len(rs.audio_buffer) >= silence_samples:
            recent_array = np.array(recent, dtype=np.float32)
            rms = float(np.sqrt(np.mean(recent_array ** 2)))
            if rms < self.silence_threshold:
                audio_data = np.array(rs.audio_buffer, dtype=np.float32)
                logger.debug(f"VAD early exit in {rs.room_id}: {elapsed_ms:.0f}ms")
                rs.capture_complete = True
                return True, audio_data

        # Timeout
        if elapsed >= self.command_duration:
            audio_data = np.array(rs.audio_buffer, dtype=np.float32)
            rs.capture_complete = False
            return True, audio_data

        return False, None

    def _check_post_wake_completion(self, rs: RoomStream) -> tuple[bool, np.ndarray | None]:
        """Post-wake rule: listen >= min after the wake, close on speech then silence.

        Every duration is measured in captured samples after the wake (the
        pre-roll is excluded). A capture without speech is dropped without
        STT; ``_poll_command`` sees ``(True, None)`` for a drop. A capture
        that reaches the cap is returned incomplete unless
        ``dispatch_at_max_utterance`` is set.
        """
        config = self._post_wake
        if rs.endpoint is None:
            wake_index = min(max(0, rs.wake_sample_index), len(rs.audio_buffer))
            rs.endpoint = EndpointState(wake_index=wake_index, floor=self._room_floor(rs),
                                        next_frame=wake_index, trigger=rs.capture_trigger)
        state = rs.endpoint
        state.analyze(rs.audio_buffer, config, self._room_vad(rs))
        wake_wall = rs.command_start_time + state.wake_index / self.sample_rate
        parser_ready = (self.endpointing_enabled and rs.early_command is not None
                        and rs.early_command.ready_to_dispatch())
        decision = state.decide(config, self.sample_rate, parser_ready, getattr(_facade(), 'time', time).time() - wake_wall)
        if not decision.closes:
            return False, None
        state.decision = decision
        self.endpoint_counts[decision.value] = self.endpoint_counts.get(decision.value, 0) + 1
        summary = state.summary(self.sample_rate)
        logger.info("[endpoint] room=%s code=%s trigger=%s post_wake_s=%.2f speech_s=%.2f floor=%.4f "
                    "peak_rms=%.4f vad_max=%s", rs.room_id, decision.value, summary["trigger"],
                    summary["post_wake_s"], summary["speech_s"], summary["floor"], summary["peak_rms"],
                    _fmt_vad(summary["vad_max"]))
        if not decision.dispatches:
            return True, None
        # At the cap the capture stays incomplete unless the household chose to
        # dispatch it (VC-Q2): the router then rejects it before STT, as the
        # legacy timeout did, so a false wake into a call costs no STT.
        rs.capture_complete = (decision is not EndpointDecision.CLOSE_MAX
                               or config.dispatch_at_max_utterance)
        return True, np.array(rs.audio_buffer, dtype=np.float32)

    def _reset_listening(self, rs: RoomStream) -> None:
        """Cierra el estado de captura y cancela el worker early si está activo."""
        rs.listening = False
        rs.audio_buffer = []
        rs.audio_metadata = []
        rs.inline_provenance = None
        rs.early_command = None
        rs.wake_text = None
        rs.transcript = None
        rs.capture_complete = False
        rs.capture_generation += 1
        rs.wake_sample_index = 0
        rs.capture_trigger = CaptureTrigger.UNKNOWN
        rs.wake_vad = None
        rs.endpoint = None
        rs.onset_frames = 0
        if rs.early_task is not None:
            task, rs.early_task = rs.early_task, None
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    def _early_snapshot(self, rs: RoomStream, generation: int):
        room = self._capture_rooms[rs.room_id]
        room.require_live()
        if not rs.listening or generation != rs.capture_generation:
            raise SourcesInvalidated('capture_generation_changed')
        audio = np.array(rs.audio_buffer, dtype=np.float32)
        return audio, self._capture_provenance(rs, audio)

    def _require_early(self, rs: RoomStream, generation: int, cancelled=None) -> None:
        self._capture_rooms[rs.room_id].require_live()
        if not self._running or not rs.listening or generation != rs.capture_generation:
            raise SourcesInvalidated('capture_generation_changed')
        if cancelled is not None and cancelled.is_set():
            raise SourcesInvalidated('capture_observer_cancelled')

    def _transcribe_early(self, rs: RoomStream, generation: int, audio, cancelled, audio_provenance=None):
        text = partial = transcript = None
        invalidated = False
        try:
            with model_admission(self.native_model_locks[id(self._stt)],
                                 lambda: self._require_early(rs, generation, cancelled)):
                text, _ = self._stt.transcribe(audio, sample_rate=self.sample_rate)
                self._require_early(rs, generation, cancelled)
                if text:
                    partial = getattr(_facade(), 'parse_partial_command', parse_partial_command)(text)
                    from src.evaluation.provenance import FrameInterval
                    transcript = CapturedTranscript.from_audio(text, TranscriptSource.PARTIAL, audio,
                        transcript_provenance=audio_provenance,
                        transcript_span=FrameInterval(0, audio.size) if audio_provenance is not None else None,
                        transcript_truncated=False)
                return partial, transcript
        except SourcesInvalidated:
            invalidated = True
        except Exception as error:
            logger.debug('Early STT unavailable: %s', type(error).__name__)
        finally:
            audio = text = partial = transcript = cancelled = None
            audio_provenance = None
        if invalidated:
            raise SourcesInvalidated('capture_sources_invalidated')
        return None, None

    async def _owned_early(self, rs: RoomStream, generation: int, audio, audio_provenance=None):
        cancelled = Event()
        try:
            return await self._capture_native.run(self._transcribe_early, rs, generation, audio, cancelled, audio_provenance)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            audio = cancelled = None
            audio_provenance = None

    def _publish_early(self, rs: RoomStream, generation: int, partial, transcript):
        try:
            self._require_early(rs, generation)
            rs.early_command, rs.transcript = partial, transcript
        finally:
            partial = transcript = None

    async def _early_parse_worker(self, rs: RoomStream) -> None:
        """Own speculative native STT and publish only to the same live capture."""
        interval_s = self.early_dispatch_interval_ms / 1000
        min_samples = int(self.early_dispatch_min_audio_s * self.sample_rate)
        generation, last_samples = rs.capture_generation, -1
        audio = partial = transcript = audio_provenance = None
        room = self._capture_rooms[rs.room_id]
        try:
            while rs.listening and self._running and not self._closed:
                await asyncio.sleep(interval_s)
                audio, audio_provenance = await room.call(self._early_snapshot, rs, generation)
                if len(audio) < min_samples or len(audio) == last_samples:
                    audio = audio_provenance = None
                    continue
                last_samples = len(audio)
                partial, transcript = await self._owned_early(rs, generation, audio, audio_provenance)
                await room.call(self._publish_early, rs, generation, partial, transcript)
                audio = partial = transcript = None
                audio_provenance = None
        except SourcesInvalidated:
            return
        finally:
            audio = partial = transcript = None
            audio_provenance = None

    async def _dispatch_command(self, event: CommandEvent):
        """Dispatch a captured command via registered callback."""
        # Wake textual (spec 2026-07-05, fix review final): registrar el ts
        # del dispatch ACÚSTICO por room ANTES de awaitear el callback. Un
        # comando slow-path (reasoner cloud, segundos) deja una ventana
        # abierta mientras se procesa — si el ts se registrara solo al final,
        # el canal textual podría evaluar la MISMA utterance sin ver el
        # dedup acústico (el ts todavía no existiría) y disparar un segundo
        # comando en paralelo. Registrarlo acá hace que el canal textual
        # gane la carrera de dedup desde el primer instante del dispatch.
        self._last_command_dispatch_ts[event.room_id] = getattr(_facade(), 'time', time).monotonic()
        try:
            self._require_dispatch_sources()
            if self._on_command_callback:
                result = await self._on_command_callback(event)
            else:
                logger.warning("No on_command callback registered")
                result = {}

            self._require_dispatch_sources()

            # Refrescar el ts tras el callback: ancla también la ventana de
            # dedup al momento de FINALIZACIÓN de comandos largos, no solo
            # al de inicio.
            self._last_command_dispatch_ts[event.room_id] = getattr(_facade(), 'time', time).monotonic()

            outcome = getattr(_facade(), 'classify_outcome', classify_outcome)(result) if isinstance(result, dict) else None

            # Captura vacía → liberar la supresión del canal textual
            # (incidente 2026-07-25). El ts se setea al DISPARAR el wake, así
            # que una captura que no produjo comando —p.ej. el [STT-veto]
            # dejando Text='' sobre un "prendé la luz" real— dejaba la red de
            # seguridad textual abajo durante dedup_window_s enteros, justo
            # después de que el camino acústico falló en silencio. Eso era el
            # "al tercer intento recién agarró".
            #
            # Sólo se libera en "empty": ruido/TV y alucinaciones MANTIENEN la
            # supresión (ahí el canal textual no debe entrar). Ante cualquier
            # otra cosa —incluido un callback que explotó— se mantiene.
            error_code = result.get("error") if isinstance(result, dict) else None
            if outcome == "empty" or error_code == "gate_rejected:incomplete_capture":
                self._last_command_dispatch_ts.pop(event.room_id, None)
                logger.info(
                    f"[dedup] captura vacía/incompleta en {event.room_id} — "
                    f"libero el canal textual (no hubo comando)"
                )

            # AmbientGuard: el resultado de la captura alimenta la escalera
            # (noise/empty/timeout escalan; accepted/other_fail/hallucination
            # no — alucinaciones de silencio de Whisper no son ambiente
            # hostil y no deben bloquear el quiet timer, 2026-07-05).
            if self._guard is not None and outcome is not None:
                self._guard.on_capture_result(event.room_id, outcome)
                logger.debug(
                    f"[AmbientGuard] capture outcome en {event.room_id}: {outcome}"
                )
                # Gracia post-éxito (2026-06-06): en STRICT el follow_up no se
                # abre al wake; tras un comando ACEPTADO se abre acá — salvo si
                # vino incrustado en una charla continua (inline_conversation).
                if (
                    outcome == "accepted"
                    and not getattr(event, "inline_conversation", False)
                    and self._guard.follow_up_allowed(event.room_id)
                ):
                    try:
                        self.follow_up.start_conversation()
                        self._conversation_room = event.room_id
                    except Exception as e:
                        logger.debug(f"follow_up post-éxito no-op: {e}")

            if self._on_post_command_callback:
                await self._on_post_command_callback(result, event)
        except SourcesInvalidated:
            pass
        except Exception as error:
            logger.error('Command dispatch failed: %s', type(error).__name__)
        finally:
            event = result = None

    def _require_dispatch_sources(self):
        if not self._dispatch_live():
            raise SourcesInvalidated('dispatch_sources_invalidated')

    def _dispatch_live(self):
        receipt = current_producer_scope()
        sources = receipt.snapshot()[0] if receipt is not None else PRODUCER_CONTEXT.get().sources
        return not self._closed and not self._paused and (self._privacy is None or self._privacy.sources_available(sources))
