"""Room stream descriptors, capture channel resolution, and unified stream manager facade.

Composes ``StreamRecoveryMixin`` and ``XvfTelemetryMixin`` while keeping stream
construction (``_make_stream``, ``_configure_stream``) and core DTOs in one
compact entrypoint.
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


import sys as _sys


from src.pipeline.stream_recovery import PortAudioReset, StreamRecoveryMixin, detect_stale_streams
from src.pipeline.xvf_telemetry import XvfRebootReport, XvfTelemetryMixin, _CaptureHolder, _RebootDeferral

def _facade():
    return _sys.modules.get("src.pipeline.multi_room_audio_loop")


def _resolve_capture_channels(max_input_channels: int) -> int:
    """Channels to open for the InputStream capture.

    Some mics (UAC1.0 ReSpeaker legacy) report max_input_channels=0 in
    PortAudio and require forcing 1. Others (XVF3800, 2ch) must be opened
    with their native count: opening a 2ch device as 1ch reads interleaved
    data into a 1ch buffer and garbles the audio, causing Whisper to
    hallucinate. The channel actually consumed downstream is NOT hardcoded
    to 0: it's `rs.capture_channel` (per-room config, e.g. `capture_channel: 1`
    for cocina/escritorio), read in the InputStream callback with a
    fallback to channel 0 only if the device doesn't have that many
    channels. This function only decides how many channels to *open*.

    Args:
        max_input_channels: Value from sd.query_devices(index)['max_input_channels'].

    Returns:
        Number of channels to pass to sd.InputStream.
    """
    return max_input_channels if max_input_channels and max_input_channels >= 1 else 1


@dataclass
class RoomStream:
    """Per-room audio capture state."""
    room_id: str
    device_index: int | None
    wake_detector: WakeWordDetector
    echo_suppressor: EchoSuppressor
    listening: bool = False
    audio_buffer: list = field(default_factory=list)
    audio_metadata: list = field(default_factory=list)
    inline_provenance: object = None
    command_start_time: float = 0.0
    # Speculative parsing can shorten endpointing, but never authorize sending
    # an action before capture ends. The snapshot must cover the final audio.
    early_task: Optional[asyncio.Task] = None
    early_command: Optional[PartialCommand] = None
    transcript: CapturedTranscript | None = None
    capture_complete: bool = False
    capture_generation: int = 0
    # Diagnostic wake text; reuse requires a matching transcript snapshot.
    wake_text: Optional[str] = None
    # Barge-in (S3): acumulador de ms de voz sostenida durante TTS activo.
    # Se incrementa por cada chunk con RMS + is_human_voice==True; decae
    # con silencio. Dispara barge-in cuando supera `barge_in_min_duration_ms`.
    barge_in_accum_ms: float = 0.0
    # Pre-roll: ring buffer (deque) de chunks de audio previos al wake. Al
    # disparar el wake se usa para sembrar `audio_buffer` y recuperar el comando
    # dicho durante la latencia de detección de openwakeword. None hasta que el
    # loop lo inicializa con maxlen según wake_preroll_s.
    preroll: object = None
    preroll_metadata: object = None
    wake_opening: CaptureOpening | None = None
    wake_channel: int | None = None
    # Canal de captura del device (L-3 prep 2026-06-04). El XVF3800 UA expone
    # 2 canales (doc Seeed: ch0=Conference post-procesado para oído humano,
    # ch1=ASR del beam auto-select). Default 0 = comportamiento histórico.
    # Configurable per-room (rooms.<room>.capture_channel) para el A/B ch0/ch1
    # SIN swap global — el mic UAC1.0 del escritorio es mono (fallback a 0).
    capture_channel: int = 0
    # Warning de canal faltante emitido (una sola vez por stream).
    channel_warned: bool = False
    # Score del wake que abrió la captura actual. Se fija al aceptar el wake
    # y se propaga al CommandEvent para que el earcon gate decida humano-plausible.
    wake_score: float = 1.0
    # Puerto USB físico estable (ej "3-1.4") para re-resolver el índice de
    # PortAudio si el device se re-enumera. None = no re-resolver por puerto.
    mic_usb_port: Optional[str] = None
    # Timestamp monotónico del último frame recibido por el callback. 0.0 hasta
    # que el stream se abre/recibe el primer frame. Lo vigila _stream_watchdog
    # para detectar un mic muerto por re-enumeración USB.
    last_frame_ts: float = 0.0
    # Momento de apertura del stream. Separado de last_frame_ts para que el
    # watchdog distinga "abrió y todavía no entregó" (gracia larga) de
    # "entregaba y dejó de entregar" (timeout corto).
    opened_ts: float = 0.0
    # Post-wake endpoint (V1 2026-11-17): buffer index of the first sample
    # after the wake (pre-roll length), what opened the capture, the Silero
    # probability of the wake frame and the per-capture endpoint state.
    wake_sample_index: int = 0
    capture_trigger: CaptureTrigger = CaptureTrigger.UNKNOWN
    wake_vad: float | None = None
    endpoint: EndpointState | None = None
    # Ambient level of this room while not capturing (post-wake rule only).
    noise_floor: NoiseFloorTracker | None = None
    # Consecutive level-speech frames seen while a follow-up may start.
    onset_frames: int = 0


# Preserve canonical nominal identity expected by runtime_output_census.
RoomStream.__module__ = "src.pipeline.multi_room_audio_loop"


class RoomStreamManagerMixin(StreamRecoveryMixin, XvfTelemetryMixin):
    """RoomStream construction composed with recovery and XVF telemetry mixins."""

    def _make_stream(self, rs: "RoomStream", *, report_not_admitted: bool = False):
        """Construct only; its retained owner configures and starts the handle."""
        if rs.mic_usb_port:
            rs.device_index = getattr(_facade(), 'resolve_mic_usb_port', resolve_mic_usb_port)(rs.mic_usb_port)
            self._note_catalog_gap(rs)
        if rs.device_index is None or self._closed or not self._inputs.accepting:
            if report_not_admitted:
                raise InputNotAdmitted()
            return None
        with self._direction_lock:
            self._direction_opening.add(rs.room_id)
        self._invalidate_direction(rs.room_id)
        native_admitted = False
        try:
            dev_info = getattr(_facade(), 'sd', sd).query_devices(rs.device_index)
            if self._closed or not self._inputs.accepting:
                if report_not_admitted:
                    raise InputNotAdmitted()
                return None
            capture_channels = getattr(_facade(), '_resolve_capture_channels', _resolve_capture_channels)(
                int(dev_info.get("max_input_channels", 0))
            )
            previous = self._microphone_captures.get(rs.room_id)
            generation = previous.observation.generation + 1 if previous else 1
            match = re.search(r"\(hw:(\d+),", dev_info.get("name", ""))
            card = int(match[1]) if match else None
            inventory = self._microphone_inventory
            identities = [device for device in inventory.devices if device.alsa_card == card] if (
                inventory is not None and inventory.success and card is not None
                and self._microphone_monotonic() - inventory.observed_monotonic <= self._microphone_stale_after_s
            ) else []
            identity = identities[0] if len(identities) == 1 else None
            holder = _CaptureHolder(CaptureObservation(
                generation=generation, opened=True,
                usb_port=identity.usb_port if identity else None,
                serial=identity.serial if identity else None,
                alsa_card=card, device_index=rs.device_index, channels=capture_channels,
            ))
            self._microphone_captures[rs.room_id] = holder
            callback = self._make_audio_callback(rs, holder)
            native_admitted = True
            stream = getattr(_facade(), 'sd', sd).InputStream(
                device=rs.device_index,
                samplerate=self.sample_rate,
                channels=capture_channels,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=callback,
            )
            return stream
        except getattr(_facade(), 'sd', sd).PortAudioError as e:
            self._retire_microphone_capture(rs.room_id)
            logger.error(
                f"Room {rs.room_id}: failed to open device {rs.device_index}: {e}"
            )
            if report_not_admitted and not native_admitted:
                raise InputNotAdmitted() from None
            return None
        finally:
            with self._direction_lock:
                self._direction_opening.discard(rs.room_id)

    def _configure_stream(self, rs: RoomStream, stream) -> None:
        """Read host properties only after the driver owner retains the handle."""
        holder = self._microphone_captures[rs.room_id]
        rate = getattr(stream, 'samplerate', None)
        channels = getattr(stream, 'channels', None)
        dtype = getattr(stream, 'dtype', None)
        index = getattr(stream, 'device', None)
        actual_channels = channels if isinstance(channels, int) else holder.observation.channels
        # Earlier callbacks keep None; no requested-value fallback establishes provenance.
        holder.opening = None
        if (type(rate) in (int, float) and math.isfinite(rate) and float(rate).is_integer()
                and type(channels) is int):
            try:
                from src.evaluation.provenance import CaptureSource
                holder.opening = CaptureOpening(CaptureSource.LIVE_INPUT, uuid4().hex,
                    holder.observation.generation, int(rate), channels)
            except (ValueError, TypeError, OverflowError):
                pass
        holder.observation = replace(holder.observation,
            sample_rate_hz=float(rate) if isinstance(rate, (int, float)) else float(self.sample_rate),
            channels=actual_channels,
            selected_channel=rs.capture_channel if 0 <= rs.capture_channel < actual_channels else 0,
            host_dtype=dtype if isinstance(dtype, str) else 'float32',
            device_index=index if isinstance(index, int) else rs.device_index)
        # Publish within the retained native owner, even when the observer
        # awaiting open has disappeared. start() may already deliver frames.
        rs.opened_ts = getattr(_facade(), 'time', time).monotonic()
        rs.last_frame_ts = 0.0
