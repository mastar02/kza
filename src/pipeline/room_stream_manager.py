"""Room stream lifecycle, PortAudio catalog recovery, watchdog, and XVF DSP management.

Extracted from ``src/pipeline/multi_room_audio_loop.py`` so microphone hardware,
watchdog, telemetry, and XVF recovery live in a dedicated cohesive module.
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


import sys as _sys

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

def detect_stale_streams(
    states: list[tuple[str, float, float]],
    now: float,
    timeout_s: float,
    first_frame_grace_s: float,
) -> list[str]:
    """Return room_ids whose audio stream is not delivering audio.

    Dos regímenes distintos, porque "todavía no arrancó" no es lo mismo que
    "dejó de entregar":

    - Ya entregó audio alguna vez (``last_frame_ts > 0``): stale si pasaron
      más de ``timeout_s`` desde el último frame con señal.
    - Abierto pero sin entregar nunca (``last_frame_ts == 0``): stale recién
      pasado ``first_frame_grace_s`` desde la apertura. Medido sobre arranques
      reales (2026-07-29/30): lo normal son 1.5-2s, pero se observó uno de
      135s. Sin esta gracia el watchdog dispararía recovery sobre un mic que
      todavía está despertando, y como el recovery cierra TODOS los streams,
      se llevaría puestos a los sanos.
    - Nunca abierto (``opened_ts == 0``): se ignora, no hay nada que recuperar.

    Args:
        states: list of (room_id, last_frame_ts, opened_ts), monotónicos.
        now: current monotonic time.
        timeout_s: segundos sin audio, tras haber entregado, para darlo por muerto.
        first_frame_grace_s: margen para el PRIMER frame tras abrir el stream.
    """
    stale = []
    for room_id, last_frame_ts, opened_ts in states:
        if last_frame_ts > 0.0:
            if (now - last_frame_ts) > timeout_s:
                stale.append(room_id)
        elif opened_ts > 0.0 and (now - opened_ts) > first_frame_grace_s:
            stale.append(room_id)
    return stale

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

class PortAudioReset(Enum):
    """Outcome of a PortAudio terminate+initialize (T025-U2-RV3-5)."""

    # Terminated and initialized again: the device list was re-read.
    RESCANNED = "rescanned"
    # Terminate failed while still initialized: the old device list is kept.
    UNCHANGED = "unchanged"
    # Initialize failed: no stream can open until a later reset.
    UNINITIALIZED = "uninitialized"

@dataclass(frozen=True)
class XvfRebootReport:
    """What the native part of one XVF firmware reboot observed (T048-U1).

    Attributes:
        outcome: Result of the REBOOT control transfer.
        reenumerated: The device came back on its port within the settle wait.
        alsa_card: ALSA card it exposes again, or None if none appeared in time.
        devnum: Kernel device number it came back with (sysfs), when readable.
    """

    outcome: XvfRebootOutcome
    reenumerated: bool = False
    alsa_card: int | None = None
    devnum: int | None = None

@dataclass
class _RebootDeferral:
    """A due firmware reboot waiting for an activity to end (T048-U1-RV1 F3).

    Attributes:
        reason: Blocker last logged (``command``, ``conversation``, ``response``).
        since: Microphone-clock time the deferral started.
        escalated: The long-deferral error was already logged.
    """

    reason: str
    since: float
    escalated: bool = False

class _CaptureHolder:
    """One callback-owned observation pointer; retired holders stay isolated."""

    def __init__(self, observation: CaptureObservation):
        self.observation = observation
        self.opening: CaptureOpening | None = None


class RoomStreamManagerMixin:
    """PortAudio stream opening, watchdog, XVF recovery, and telemetry mixin."""

    def _apply_xvf_tuning(self) -> None:
        """Aplica el tuning configurado al DSP (EN RAM). Fail-open por param.

        Un nombre/valor inválido en el yaml NO tira el servicio: se loguea y
        se continúa con el resto. Loguea valor previo → nuevo para auditoría.
        """
        if not self._xvf_tuning.get("apply_on_start", False):
            return
        if not self._xvf_by_room:
            logger.warning(
                "[XVF-tuning] xvf_tuning.apply_on_start=true pero no hay "
                "XvfController — tuning NO aplicado (¿mic XVF3800 presente?)"
            )
            return
        # El tuning se aplica al DSP de CADA mic: son chips distintos y cada uno
        # arranca con el preset persistido de fábrica al re-enchufarse.
        for room_id, ctrl in self._xvf_by_room.items():
            self._apply_xvf_tuning_room(room_id, ctrl)

    def _xvf_tuning_enabled(self) -> bool:
        """Whether the configured tuning is written to the DSP (``apply_on_start``)."""
        return bool(self._xvf_tuning.get("apply_on_start", False)
                    and self._xvf_tuning.get("params"))

    def _apply_xvf_tuning_room(self, room_id: str, ctrl) -> None:
        """Write the configured tuning to one room's DSP (RAM). Fail-open per param.

        Blocking USB I/O: runs on the ``_xvf_native`` owner, never on the loop.

        Args:
            room_id: Room whose controller is written (for the audit log).
            ctrl: That room's port-bound XVF controller.
        """
        if not self._xvf_tuning.get("apply_on_start", False):
            return
        params = self._xvf_tuning.get("params") or {}
        for name, values in params.items():
            try:
                before = ctrl.read_param(name)
            except ValueError:
                before = None
            try:
                ok = ctrl.write_param(name, values)
            except ValueError as e:  # typo en el yaml: log y seguir
                logger.error(f"[XVF-tuning] parámetro inválido en config: {e}")
                continue
            except Exception as e:  # cualquier otra cosa: fail-open
                logger.warning(f"[XVF-tuning] {room_id}: {name} no aplicado: {e}")
                continue
            if ok:
                logger.info(
                    f"[XVF-tuning] {room_id}: {name}: {before} → {values} (RAM)"
                )
            else:
                logger.warning(
                    f"[XVF-tuning] {room_id}: {name} no aplicado (write fail-open)"
                )

    def _reapply_xvf_tuning(self, room_id: str, ctrl) -> None:
        """Reapply the tuning after the room's device re-enumerated (native).

        The controller may still cache the handle of the device that left the
        bus, so it is dropped first and the next transfer finds the new one.

        Args:
            room_id: Room whose device re-enumerated.
            ctrl: That room's port-bound XVF controller.
        """
        invalidate = getattr(ctrl, "invalidate_device", None)
        if callable(invalidate):
            invalidate()
        self._apply_xvf_tuning_room(room_id, ctrl)

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

    def _reinit_portaudio(self) -> PortAudioReset:
        """Force PortAudio to re-scan the device list (sync; fail-open).

        PortAudio snapshots devices at Pa_Initialize and never re-scans. After a
        USB re-enumeration the cached indices point to dead devices, so we must
        terminate+initialize to see the new ones. This is GLOBAL: it invalidates
        every open stream, which is why _recover_streams reopens all of them.

        Terminate and initialize fail separately. A failed initialize leaves
        PortAudio uninitialized, and sounddevice then refuses to terminate it
        ('PortAudio not initialized'); initialize must still run on the next
        reset or no microphone could ever open again (T025-U2-RV2-1).

        Returns:
            ``RESCANNED`` when the device list was re-read; ``UNCHANGED``
            when terminate failed while PortAudio stayed initialized, so its
            old list is kept (no re-scan happened); ``UNINITIALIZED`` when it
            was left uninitialized and no stream can open until a later reset.
        """
        try:
            getattr(_facade(), 'sd', sd)._terminate()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio terminate failed: {e}")
            initialized = getattr(getattr(_facade(), 'sd', sd), '_initialized', None)
            if type(initialized) is int and initialized > 0:
                # Still initialized with its old catalog: initializing again
                # would only nest sounddevice's count, and a later terminate
                # would then never re-scan.
                return PortAudioReset.UNCHANGED
        try:
            getattr(_facade(), 'sd', sd)._initialize()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio reinit failed: {e}")
            return PortAudioReset.UNINITIALIZED
        return PortAudioReset.RESCANNED

    def _note_catalog_gap(self, rs: RoomStream) -> None:
        """Record whether a configured card is in sysfs but not in PortAudio.

        Runs in the native open owner right after the port was resolved, so
        the sysfs read never blocks the event loop. An unresolved port whose
        card is also absent from sysfs (an unplugged microphone) is not a
        catalog gap: refreshing PortAudio cannot bring it back.

        Args:
            rs: Room whose ``device_index`` was just resolved from its port.
        """
        if self._catalog_refresh_interval_s <= 0:
            return
        card = None
        if rs.device_index is None:
            card = getattr(_facade(), 'usb_port_to_alsa_card', usb_port_to_alsa_card)(rs.mic_usb_port, str(self._microphone_usb_root))
        if card is None:
            self._catalog_missing.pop(rs.room_id, None)
            self._catalog_gap_seen.pop(rs.room_id, None)
        else:
            self._catalog_missing[rs.room_id] = card
            seen = self._catalog_gap_seen.get(rs.room_id)
            if seen is None or seen[0] != card:
                self._catalog_gap_seen[rs.room_id] = (card, getattr(_facade(), 'time', time).monotonic())

    def _catalog_refresh_blocker(self) -> str | None:
        """Name the activity a brief all-room capture cut would interrupt.

        Returns:
            ``'command'`` while a command is captured or dispatched in some
            room, ``'conversation'`` while the follow-up conversation is
            active, ``'response'`` while voice output is owned (a PortAudio
            reset would also stop playback), or None when a cut is allowed.
        """
        # Probe directly: CommandTurns.active would mark the turn as observed
        # and notify its end listeners for a check that is not a consumer.
        if self._command_turn_busy():
            return 'command'
        if bool(getattr(self.follow_up, 'is_active', False)):
            return 'conversation'
        handler = self._response_handler
        if handler is not None and bool(getattr(handler, 'is_responding', False)):
            return 'response'
        return None

    def _request_catalog_refresh(self, rs: RoomStream) -> bool:
        """Ask the global recovery to refresh PortAudio for a late microphone.

        Called by a room's reopen waiter after a failed open. The refresh is
        admitted only when that room's card is in sysfs but missing from
        PortAudio, that gap was first seen at least the reopen backoff
        ceiling ago (``settling``: a card that just enumerated may not be
        openable yet), no command, conversation or response is active, and
        the previous refresh is older than the configured interval.

        Args:
            rs: Room still waiting for its microphone.

        Returns:
            True when a global catalog recovery was scheduled.
        """
        if self._catalog_refresh_interval_s <= 0:
            return False
        card = self._catalog_missing.get(rs.room_id)
        futile = self._catalog_futile.get(rs.room_id)
        if futile is not None and futile[0] != card:
            # The card left sysfs or came back with another number: start over.
            del self._catalog_futile[rs.room_id]
        if card is None:
            return False
        reason = self._catalog_refresh_blocker()
        seen = self._catalog_gap_seen.get(rs.room_id)
        if reason is None and (seen is None or seen[0] != card
                               or getattr(_facade(), 'time', time).monotonic() - seen[1] < self._watchdog_backoff_max_s):
            reason = 'settling'
        last = self._catalog_refreshed_at
        if (reason is None and last is not None
                and self._microphone_monotonic() - last < self._catalog_refresh_wait_s(rs.room_id)):
            reason = 'rate_limit'
        if reason is not None:
            if self._catalog_refresh_deferred != reason:
                self._catalog_refresh_deferred = reason
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: PortAudio catalog refresh deferred "
                    f"({reason}); ALSA card {card} is present but not listed"
                )
            return False
        return self._schedule_recovery([rs.room_id], catalog_refresh=True)

    def _hold_output(self) -> tuple[object | None, bool]:
        """Stop voice output from opening a stream while PortAudio is reset.

        A timer, reminder notice or alert speaks on its own schedule and
        would open an output stream on another thread in the middle of
        ``Pa_Terminate``/``Pa_Initialize`` (T025-U2-RV2-2). Only output that
        goes through the attached ResponseHandler is held.

        Returns:
            ``(token, output_busy)``: the hold token (None when nothing was
            held) and whether a response owns the output right now, so the
            reset would cut it.
        """
        handler = self._response_handler
        hold = getattr(handler, 'try_hold_output', None) if handler is not None else None
        if not callable(hold):
            return None, False
        token = hold()
        return token, token is None

    def _release_output(self, token: object | None) -> None:
        """Release an output hold from ``_hold_output``; safe from any thread."""
        if token is None:
            return
        release = getattr(self._response_handler, 'release_output_hold', None)
        if callable(release):
            release(token)

    def _capture_working(self) -> bool:
        """Whether a brief all-room capture cut would interrupt a microphone.

        Returns:
            True when some room has an open stream that delivered audio within
            the watchdog timeout, or was opened and is still inside its
            first-frame grace; False when every room is deaf.
        """
        now = getattr(_facade(), 'time', time).monotonic()
        for room_id, rs in self.room_streams.items():
            if room_id not in self._streams:
                continue
            if rs.last_frame_ts > 0.0:
                if now - rs.last_frame_ts <= self._watchdog_timeout_s:
                    return True
            elif rs.opened_ts > 0.0 and now - rs.opened_ts <= self._watchdog_first_frame_grace_s:
                return True
        return False

    def _catalog_short_wait_s(self, attempts: int) -> float:
        """Reopen backoff ceiling doubled per consecutive attempt, capped at the interval."""
        exponent = min(max(attempts, 1) - 1, 16)
        return min(self._watchdog_backoff_max_s * 2 ** exponent, self._catalog_refresh_interval_s)

    def _catalog_futile_wait_s(self, room_id: str | None) -> float:
        """The interval, doubled per refresh that left this room's card unlisted, up to the ceiling."""
        interval = self._catalog_refresh_interval_s
        futile = self._catalog_futile.get(room_id) if room_id is not None else None
        if futile is None:
            return interval
        return min(interval * 2 ** min(futile[1] - 1, 16), self._catalog_refresh_ceiling_s)

    def _catalog_refresh_wait_s(self, room_id: str | None = None) -> float:
        """Minimum time between two catalog refresh cuts.

        The interval and the futile backoff only protect working rooms from
        the cut. When nothing captures, a refresh cuts nothing and the house
        stays deaf until one succeeds.

        Args:
            room_id: Room asking for the refresh; its card may have survived
                earlier refreshes unlisted.

        Returns:
            A short backoff (the reopen backoff ceiling, doubling per
            consecutive failure, capped at the interval) while the last reset
            left PortAudio uninitialized (RV2-1), or while no room has a
            working stream (RV3-1, doubling per refresh that ran deaf and left
            a card unlisted). Otherwise the configured interval, doubled per
            further refresh that left this room's card unlisted, up to the
            ceiling.
        """
        if self._catalog_reinit_failures:
            return self._catalog_short_wait_s(self._catalog_reinit_failures)
        if not self._capture_working():
            return self._catalog_short_wait_s(self._catalog_deaf_refreshes)
        return self._catalog_futile_wait_s(room_id)

    def _note_catalog_refresh_result(self, before: dict[str, int], *, deaf: bool = False) -> None:
        """Count refreshes after which PortAudio still did not list a card.

        Called once every room was reopened after a successful reset, so
        ``_catalog_missing`` holds what the refreshed catalog lists.

        Args:
            before: ``_catalog_missing`` as it was when the refresh started.
            deaf: No room was capturing when the refresh started, so the cut
                interrupted nothing: count it toward the short deaf backoff
                instead of the per-room futile streak.
        """
        unlisted = {room_id: card for room_id, card in before.items()
                    if self._catalog_missing.get(room_id) == card}
        self._catalog_deaf_refreshes = self._catalog_deaf_refreshes + 1 if deaf and unlisted else 0
        if self._catalog_deaf_refreshes == 2:
            logger.warning(
                f"[audio-watchdog] PortAudio still does not list ALSA cards {unlisted} after "
                f"{self._catalog_deaf_refreshes} catalog refreshes with no microphone capturing; "
                f"retrying the refresh in {self._catalog_short_wait_s(self._catalog_deaf_refreshes):.1f}s, "
                f"doubling up to {self._catalog_refresh_interval_s:.0f}s"
            )
        for room_id, card in before.items():
            if self._catalog_missing.get(room_id) != card:
                self._catalog_futile.pop(room_id, None)
                continue
            if deaf:
                continue
            previous = self._catalog_futile.get(room_id)
            streak = previous[1] + 1 if previous is not None and previous[0] == card else 1
            self._catalog_futile[room_id] = (card, streak)
            if streak == 2:
                logger.warning(
                    f"[audio-watchdog] {room_id}: PortAudio still does not list ALSA card {card} "
                    f"after {streak} catalog refreshes; backing off the next refresh to "
                    f"{self._catalog_futile_wait_s(room_id):.0f}s, doubling up to "
                    f"{self._catalog_refresh_ceiling_s:.0f}s"
                )
        for room_id, (card, _streak) in tuple(self._catalog_futile.items()):
            if self._catalog_missing.get(room_id) != card:
                del self._catalog_futile[room_id]

    def _note_catalog_reinit(self, ready: bool) -> None:
        """Record whether the last PortAudio reset left it initialized.

        Args:
            ready: Whether ``_reinit_portaudio`` left PortAudio initialized.
        """
        if ready:
            self._catalog_reinit_failures = 0
            return
        self._catalog_reinit_failures += 1
        if self._catalog_refresh_interval_s > 0:
            logger.error(
                "[audio-watchdog] PortAudio is not initialized after the catalog reset; "
                f"no microphone can open; retrying the refresh in {self._catalog_refresh_wait_s():.1f}s"
            )
        else:
            logger.error(
                "[audio-watchdog] PortAudio is not initialized after the catalog reset; "
                "no microphone can open and the catalog refresh is disabled"
            )

    def _input_epoch_current(self, epoch: int) -> bool:
        return not self._closed and self._inputs.accepting and epoch == self._recovery_epoch

    async def _try_reopen_once(self, rs: "RoomStream") -> bool:
        """Join one retained native open without crossing an input pause epoch."""
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch) or self._global_recovery:
            return False
        stream = await self._inputs.open(rs.room_id)
        if stream is None or not self._input_epoch_current(epoch) or self._global_recovery:
            return False
        await self._sync_direction_observers()
        return self._input_epoch_current(epoch) and not self._global_recovery

    async def _reopen_room(self, rs: "RoomStream", *, epoch: int | None = None) -> None:
        """Reintentar la reapertura con backoff hasta que el device vuelva.

        Waits indefinitely (while self._running) for the device to reappear in
        sysfs — the service stays alive; only this mic waits. Never raises.

        ⚠️ Por eso NUNCA debe awaitearse en serie sobre varias rooms: ver
        `_recover_streams`.
        """
        epoch = self._recovery_epoch if epoch is None else epoch
        backoff = self._watchdog_backoff_min_s
        while self._running and self._input_epoch_current(epoch) and not self._global_recovery:
            if await self._try_reopen_once(rs):
                if not self._input_epoch_current(epoch):
                    return
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: recovered "
                    f"(device={rs.device_index})"
                )
                return
            if not self._input_epoch_current(epoch) or self._global_recovery:
                return
            # An admitted global recovery cancels this waiter and reopens every
            # room after the refresh. Keep waiting meanwhile: the recovery may
            # still decline the cut if a command starts before it runs.
            self._request_catalog_refresh(rs)
            logger.warning(
                f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} "
                f"absent, retry in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._watchdog_backoff_max_s)

    def _cancel_reopen_tasks(self) -> list:
        """Cancelar las esperas de reapertura en curso (shutdown).

        `cancel()` solo la SOLICITA: la task no muere hasta que el loop la
        procesa. Devuelve las tasks para que quien pueda awaitear (`stop`)
        confirme que murieron antes de seguir.
        """
        tasks = list(self._reopen_tasks.values())
        for task in tasks:
            task.cancel()
        self._reopen_tasks.clear()
        return tasks

    def _is_awaiting_reopen(self, room_id: str) -> bool:
        """True si `room_id` ya tiene una espera de reapertura en curso."""
        task = self._reopen_tasks.get(room_id)
        return task is not None and not task.done()

    def _schedule_reopen(self, rs: "RoomStream") -> None:
        """Dejar a `rs` esperando su device en una task propia (idempotente)."""
        if (self._closed or not self._inputs.accepting or self._global_recovery
                or self._is_awaiting_reopen(rs.room_id)):
            return  # ya hay una espera viva para esta room
        logger.warning(
            f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} ausente "
            f"— esperando en background (las demás rooms siguen operativas)"
        )
        task = asyncio.create_task(self._reopen_room(rs, epoch=self._recovery_epoch), context=Context())
        self._reopen_tasks[rs.room_id] = task
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())

    def _schedule_recovery(self, trigger_room_ids: list, *, catalog_refresh: bool = False,
                           xvf_reboot: bool = False) -> bool:
        """Admit one finite controller; native effects remain owned by InputGroup.

        Args:
            trigger_room_ids: Rooms that need recovery.
            catalog_refresh: Refresh the PortAudio catalog for a late
                microphone even while other rooms are healthy; re-checked
                against active commands and conversations when it starts.
            xvf_reboot: Firmware-reboot the single trigger room's XVF3800
                (T048-U1); re-checked against activity when it starts.

        Returns:
            True when a recovery controller was admitted.
        """
        if (not self._running or not self._recovery_ready or self._closed
                or not self._inputs.accepting
                or self._recovery_task is not None and not self._recovery_task.done()):
            return False
        epoch = self._recovery_epoch
        rooms = tuple(room for room in self.room_streams if room in trigger_room_ids)
        if not rooms:
            return False

        async def recover():
            if not (self._input_epoch_current(epoch) and self._running and self._recovery_ready):
                return
            if xvf_reboot:
                await self._recover_with_xvf_reboot(rooms[0])
            elif catalog_refresh:
                # A wake or a response may have started since admission.
                if self._catalog_refresh_blocker() is not None:
                    return
                await self._recover_streams(list(rooms), catalog_refresh=True)
            else:
                await self._recover_streams(list(rooms))

        self._recovery_task = asyncio.create_task(recover(), context=Context())

        def finished(done):
            if done.cancelled():
                # A native admission may survive unexpected owner cancellation.
                # Do not admit replacement recovery until a real pause drains it.
                if self._recovery_task is done:
                    self._recovery_ready = False
                logger.warning('Audio recovery owner cancelled; admission sealed')
            elif done.exception() is not None:
                logger.warning('Audio recovery failed: %s', type(done.exception()).__name__)
            if not done.cancelled() and self._recovery_task is done:
                # Native work was awaited to its real exit. Drop exception
                # frames and RoomStream references instead of caching a task.
                self._recovery_task = None

        self._recovery_task.add_done_callback(finished, context=Context())
        return True

    def _reboot_xvf_room(self, ctrl, usb_port: str, epoch: int) -> XvfRebootReport:
        """Send the firmware reboot and wait for the device and its ALSA card (native).

        Runs on the ``_xvf_native`` owner. Both waits share the policy's
        settle bound and stop early when a pause or close supersedes the
        recovery epoch.

        Args:
            ctrl: The room's port-bound XVF controller.
            usb_port: The room's mic port, where the card must reappear.
            epoch: Recovery epoch that admitted this reboot.

        Returns:
            What was observed; never raises for USB problems (fail-open).
        """
        deadline = getattr(_facade(), 'time', time).monotonic() + self._xvf_reboot_budget.policy.settle_s

        def superseded() -> bool:
            return not self._input_epoch_current(epoch)

        result = ctrl.reboot()
        if result.outcome is XvfRebootOutcome.NO_DEVICE:
            return XvfRebootReport(result.outcome)
        # A transfer error may be the device resetting before it answered.
        if not ctrl.wait_reenumerated(result.previous_address, deadline - getattr(_facade(), 'time', time).monotonic(),
                                      abort=superseded):
            return XvfRebootReport(result.outcome)
        card = None
        while not superseded():
            card = getattr(_facade(), 'usb_port_to_alsa_card', usb_port_to_alsa_card)(usb_port, str(self._microphone_usb_root))
            if card is not None or getattr(_facade(), 'time', time).monotonic() >= deadline:
                break
            getattr(_facade(), 'time', time).sleep(0.05)
        try:
            devnum_text = (self._microphone_usb_root / usb_port / "devnum").read_text().strip()
        except OSError:
            devnum_text = ""
        devnum = int(devnum_text) if devnum_text.isdigit() else None
        return XvfRebootReport(result.outcome, reenumerated=True, alsa_card=card, devnum=devnum)

    async def _recover_with_xvf_reboot(self, room_id: str) -> None:
        """Firmware-reboot one room's stuck XVF3800 and bring its capture back.

        Incident 2026-11-19: the isochronous endpoint of every XVF3800 kept
        sending oversized packets after a warm host reboot. The stream opened
        and never delivered audio, and only the firmware REBOOT recovered it.
        Sequence: retire this room's stream, reboot its device (located by
        its mic port), wait for the re-enumeration, reapply the RAM tuning
        (the reboot restores the factory preset), then reopen; when PortAudio
        does not list the new card yet, refresh its catalog (T025-U2 path).
        The other rooms keep capturing unless that refresh is needed.

        Args:
            room_id: Room whose ordinary recovery failed the configured times.
        """
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch):
            return
        reason = self._catalog_refresh_blocker()
        if reason is not None:
            # A wake or a response may have started since admission.
            self._note_xvf_reboot_deferred(room_id, reason)
            return
        ctrl = self._xvf_reboot_controller(room_id)
        if ctrl is None:
            return
        rs = self.room_streams[room_id]
        port = self._xvf_room_port(room_id)
        budget = self._xvf_reboot_budget
        failures = budget.failures(room_id)
        attempt = budget.charge(room_id)
        self._xvf_reboot_deferred.pop(room_id, None)
        logger.error(
            f"[audio-watchdog] {room_id}: isochronous capture dead after {failures} reopens; "
            f"XVF firmware reboot {attempt}/{budget.policy.max_per_hour} (USB port {port})"
        )
        report = None
        self._xvf_rebooting.add(room_id)
        try:
            await self._inputs.retire(room_id)
            if not self._input_epoch_current(epoch):
                return
            try:
                report = await self._xvf_native.run(self._reboot_xvf_room, ctrl, port, epoch)
            except Exception as error:
                logger.warning(f"[audio-watchdog] {room_id}: XVF firmware reboot failed: {type(error).__name__}")
            if report is not None and report.reenumerated:
                logger.info(
                    f"[audio-watchdog] {room_id}: XVF3800 re-enumerated at {port} "
                    f"(ALSA card {report.alsa_card}); reapplying the tuning"
                )
                try:
                    await self._xvf_native.run(self._apply_xvf_tuning_room, room_id, ctrl)
                except Exception as error:
                    logger.warning(f"[XVF-tuning] {room_id}: tuning not reapplied: {type(error).__name__}")
                if room_id in self._xvf_port_seen and report.alsa_card is not None:
                    # This reboot already retuned the new device: record it as
                    # the baseline so a later inventory poll does not retune
                    # it a second time (T048-U1-RV1 F4).
                    self._xvf_port_seen[room_id] = (report.devnum, report.alsa_card)
            elif report is not None and report.outcome is not XvfRebootOutcome.NO_DEVICE:
                logger.warning(
                    f"[audio-watchdog] {room_id}: XVF3800 at {port} did not re-enumerate within "
                    f"{budget.policy.settle_s:.0f}s after the firmware reboot"
                )
        finally:
            self._xvf_rebooting.discard(room_id)
        if not self._input_epoch_current(epoch):
            return
        if await self._try_reopen_once(rs):
            logger.info(
                f"[audio-watchdog] {room_id}: stream reopened after the XVF firmware reboot "
                f"(device={rs.device_index}); awaiting audio frames"
            )
            return
        if not self._input_epoch_current(epoch):
            return
        if report is not None and report.alsa_card is not None:
            # The card is back in sysfs but PortAudio still lists its old
            # catalog: refresh it (brief cut in every room). The reboot took
            # seconds, so a wake or a response may have started meanwhile:
            # check again and never cut the other rooms under it
            # (T048-U1-RV1 F1). The room's reopen waiter asks for the refresh
            # through _request_catalog_refresh once the activity ends.
            blocker = self._catalog_refresh_blocker()
            if blocker is None:
                await self._recover_streams([room_id], catalog_refresh=True)
                return
            logger.info(
                f"[audio-watchdog] {room_id}: PortAudio catalog refresh after the XVF firmware "
                f"reboot deferred ({blocker}); waiting for ALSA card {report.alsa_card}"
            )
        self._schedule_reopen(rs)

    async def _recover_streams(self, trigger_room_ids: list, *, catalog_refresh: bool = False) -> None:
        """Recover failed rooms without interrupting rooms still receiving audio.

        Reinitialize the shared PortAudio catalog only when no other stream is
        healthy, or when ``catalog_refresh`` asks for it because a configured
        card is in sysfs but missing from PortAudio's device list. Native
        stop/open calls must not block the asyncio event loop.

        Args:
            trigger_room_ids: Rooms that stopped delivering or never opened.
            catalog_refresh: Force the global path: retire every room,
                refresh the catalog and reopen every room.
        """
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch):
            return
        if catalog_refresh:
            blocker = self._catalog_refresh_blocker()
            if blocker is not None:
                # Check before anything is retired: the cut itself is what
                # must not happen under a command, conversation or response
                # (T048-U1-RV1 F1). The trigger rooms keep (or get) their
                # reopen waiter, which asks for the refresh again later.
                logger.info(
                    f"[audio-watchdog] PortAudio catalog refresh for {list(trigger_room_ids)} "
                    f"deferred ({blocker}) before any capture cut"
                )
                for room_id in trigger_room_ids:
                    rs = self.room_streams.get(room_id)
                    if rs is not None and self._streams.get(room_id) is None:
                        self._schedule_reopen(rs)
                return
        now = getattr(_facade(), 'time', time).monotonic()
        healthy_other = not catalog_refresh and any(
            room_id not in trigger_room_ids and rs.last_frame_ts > 0
            and now - rs.last_frame_ts <= self._watchdog_timeout_s
            for room_id, rs in self.room_streams.items()
        )
        affected = trigger_room_ids if healthy_other else list(self.room_streams)
        missing_before = dict(self._catalog_missing)
        # Whether the cut interrupts a working microphone (RV3-1).
        deaf_before = catalog_refresh and not self._capture_working()
        refreshed = False
        if catalog_refresh:
            cards = {room: self._catalog_missing.get(room) for room in trigger_room_ids}
            logger.warning(
                f"[audio-watchdog] PortAudio does not list ALSA cards {cards} present in sysfs; "
                f"refreshing the catalog with a brief capture cut in {affected}"
            )
        else:
            logger.error(f"[audio-watchdog] streams {trigger_room_ids} stopped delivering audio; recovering {affected}")
        if not healthy_other:
            # Catalog reset invalidates every native handle. Quiesce retry
            # controllers before the first await, then drain their retained
            # InputGroup opens as well as already visible native owners.
            self._global_recovery = True
        try:
            if not healthy_other:
                await asyncio.gather(*self._cancel_reopen_tasks(), return_exceptions=True)
                if not self._input_epoch_current(epoch):
                    return
            for room_id in affected:
                if not self._input_epoch_current(epoch):
                    return
                await self._inputs.retire(room_id)
            if not self._input_epoch_current(epoch):
                return
            if catalog_refresh:
                # Charge the rate limit only once the cut really happened: a
                # pause that superseded this recovery cut nothing on its behalf.
                self._catalog_refreshed_at = self._microphone_monotonic()
                self._catalog_refresh_deferred = None
            if not healthy_other:
                # A retained open may not yet have constructed its owner when
                # the first retirement pass ran. No new retry can enter now.
                await asyncio.gather(*(asyncio.shield(task) for task in self._inputs.snapshot_operations()),
                                     return_exceptions=True)
                if not self._input_epoch_current(epoch):
                    return
                for room_id in affected:
                    await self._inputs.retire(room_id)
                    if not self._input_epoch_current(epoch):
                        return
                blocker = self._catalog_refresh_blocker() if catalog_refresh else None
                hold = None
                if blocker is None:
                    hold, output_busy = self._hold_output()
                    if output_busy and catalog_refresh:
                        blocker = 'response'
                if blocker is None:
                    outcome: list[PortAudioReset] = []

                    def reset_catalog() -> None:
                        try:
                            outcome.append(self._reinit_portaudio())
                        finally:
                            self._release_output(hold)

                    try:
                        await self._inputs.catalog_call(reset_catalog)
                    finally:
                        # Also when the reset was not admitted; idempotent.
                        self._release_output(hold)
                    if outcome:
                        self._note_catalog_reinit(outcome[0] is not PortAudioReset.UNINITIALIZED)
                        # Only a real re-scan can tell whether a card is
                        # still unlisted (RV3-5).
                        refreshed = catalog_refresh and outcome[0] is PortAudioReset.RESCANNED
                else:
                    # Never reset PortAudio under a starting command or a
                    # playing response; the cut already happened, so reopen.
                    logger.warning(
                        f"[audio-watchdog] PortAudio catalog refresh skipped ({blocker}); reopening {affected}"
                    )
                if not self._input_epoch_current(epoch):
                    return
        finally:
            if not healthy_other:
                self._global_recovery = False
        for room_id in affected:
            if not self._input_epoch_current(epoch):
                return
            rs = self.room_streams[room_id]
            if await self._try_reopen_once(rs):
                logger.info(f"[audio-watchdog] {room_id}: stream reopened (device={rs.device_index}); awaiting audio frames")
            elif self._input_epoch_current(epoch):
                self._schedule_reopen(rs)
        if refreshed:
            self._note_catalog_refresh_result(missing_before, deaf=deaf_before)

    def _update_audio_anchors(self, now_mono: float, now_wall: float) -> None:
        """Refrescar las anclas de salud que la reapertura no puede resetear.

        ⚠️ Por qué existe esto en vez de leer `rs.last_frame_ts`/`rs.opened_ts`
        directo: `_try_reopen_once` pone `opened_ts = now` y
        `last_frame_ts = 0.0` en CADA reapertura exitosa. Un XVF3800 con el
        endpoint isócrono muerto —el modo de falla de los incidentes de 27h y
        7h— abre perfecto en PortAudio y entrega ceros, así que el watchdog lo
        detecta a los ~180s, "recupera", y el reset deja `ever=False` con
        `age_s` en 0. Bucle eterno de ~182s en el que `age_s` solo supera el
        umbral ~2s por vuelta: la alerta quedaba MÁS DÉBIL justo en el caso
        primario (simulado: mediana 1,67h hasta detectar, máximo 15,98h,
        1/200 sin detectar en 24h).

        ⚠️ Y el fix intuitivo empeora las cosas: subir `first_frame_grace_s`
        del poller hace que NUNCA detecte, porque `age_s` está acotado por el
        ciclo de recovery. La salida es esta: un ancla que la reapertura no
        toca, con la que la detección pasa a ser determinística a los
        `deaf_after_s`.

        Args:
            now_mono: getattr(_facade(), 'time', time).monotonic() del ciclo actual.
            now_wall: getattr(_facade(), 'time', time).time() del ciclo actual.
        """
        for room_id, rs in self.room_streams.items():
            self._room_since_wall.setdefault(room_id, now_wall)
            if rs.last_frame_ts <= 0.0:
                continue  # todavía no entregó (o lo acaban de resetear)
            # Convertir el monotónico del último frame a wall clock. Exacto:
            # no depende del período del watchdog. El max() es lo que hace el
            # ancla monótona — una reapertura que pone last_frame_ts=0 cae en
            # el `continue` de arriba y no puede hacerla retroceder.
            frame_wall = now_wall - (now_mono - rs.last_frame_ts)
            previo = self._last_audio_wall.get(room_id)
            if previo is None or frame_wall > previo:
                self._last_audio_wall[room_id] = frame_wall

    def _audio_health_rooms(self) -> dict[str, tuple[float, bool]]:
        """(ancla_wall, ever) por room, tal como lo espera write_audio_health."""
        return {
            room_id: (
                self._last_audio_wall.get(
                    room_id, self._room_since_wall.get(room_id, 0.0)
                ),
                room_id in self._last_audio_wall,
            )
            for room_id in self.room_streams
        }

    def _publish_microphone_inventory(self, inventory: UsbInventory) -> None:
        """Retain disappeared identities without treating old data as present."""
        previous = self._microphone_inventory
        known = {}
        if previous is not None:
            known.update((device.usb_port, device) for device in previous.previous_devices)
            known.update((device.usb_port, device) for device in previous.devices)
        self._microphone_inventory = replace(inventory, previous_devices=tuple(known.values()))
        self._microphone_inventory_ready.set()
        snapshot = self._base_microphone_snapshot()
        if snapshot is not None:
            current = {mic.room_id: mic for mic in snapshot.microphones}
            with self._direction_lock:
                changed = [room for room in self._direction_observers
                           if self._direction_key(current.get(room)) != self._direction_keys.get(room)]
            for room in changed:
                self._invalidate_direction(room)

    async def _refresh_microphone_inventory(self) -> None:
        """Share one actual read; reject late publication and reconciliation."""
        epoch = self._auxiliary_epoch
        if not self._auxiliary_allowed(epoch):
            return
        worker = self._microphone_read_task
        if worker is None:
            reader = self._microphone_reader or getattr(_facade(), 'microphone_status', microphone_status).read_usb_microphones
            worker = asyncio.create_task(
                asyncio.to_thread(reader, self._microphone_usb_root), context=Context())
            self._microphone_read_task = worker
        try:
            inventory = await asyncio.shield(worker)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning('Microphone inventory failed: %s', type(error).__name__)
            inventory = UsbInventory(
                observed_at=self._microphone_wall_clock(),
                observed_monotonic=self._microphone_monotonic(), success=False)
        if not self._auxiliary_allowed(epoch) or self._microphone_read_task is not worker:
            return
        self._microphone_read_task = None
        self._publish_microphone_inventory(inventory)
        reenumerated = self._note_xvf_reenumerations(inventory)
        if self._auxiliary_allowed(epoch):
            await self._sync_direction_observers()
        for room_id in reenumerated:
            if not self._auxiliary_allowed(epoch):
                return
            await self._retune_reenumerated(room_id)

    def _xvf_room_port(self, room_id: str) -> str | None:
        """USB port the room's XVF controller itself is bound to, or None.

        Deliberately no fallback to the room's ``mic_usb_port``: a controller
        without its own port discovers whichever XVF3800 comes first on the
        bus, so rebooting or retuning through it could hit another room's
        microphone (T048-U1-RV1 F2).
        """
        port = getattr(self._xvf_by_room.get(room_id), "usb_port", None)
        return port if isinstance(port, str) and port else None

    def _note_xvf_reenumerations(self, inventory: UsbInventory) -> tuple[str, ...]:
        """Find rooms whose XVF3800 re-enumerated since the previous inventory.

        A re-enumeration (physical replug, firmware reboot, bus reset) resets
        the DSP to its factory preset. It shows up as the device appearing at
        the room's port, or with another kernel device number or ALSA card.
        The first observation of a room only records a baseline: the start
        already applied the tuning.

        Args:
            inventory: Inventory just published.

        Returns:
            Rooms whose tuning must be reapplied (none while tuning is off,
            and never one whose own firmware reboot is in flight).
        """
        if not inventory.success or not self._xvf_tuning_enabled():
            return ()
        by_port = {device.usb_port: (device.devnum, device.alsa_card)
                   for device in inventory.devices}
        changed = []
        for room_id in self._xvf_by_room:
            port = self._xvf_room_port(room_id)
            if not port:
                continue
            current = by_port.get(port)
            known = room_id in self._xvf_port_seen
            previous = self._xvf_port_seen.get(room_id)
            self._xvf_port_seen[room_id] = current
            if (known and current is not None and current != previous
                    and room_id not in self._xvf_rebooting):
                changed.append(room_id)
        return tuple(changed)

    async def _retune_reenumerated(self, room_id: str) -> None:
        """Reapply the tuning to one re-enumerated room; never raises.

        Args:
            room_id: Room reported by ``_note_xvf_reenumerations``.
        """
        ctrl = self._xvf_by_room.get(room_id)
        if ctrl is None:
            return
        logger.info(
            f"[XVF-tuning] {room_id}: device re-enumerated at {self._xvf_room_port(room_id)}; "
            "reapplying the tuning"
        )
        try:
            await self._xvf_native.run(self._reapply_xvf_tuning, room_id, ctrl)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(f"[XVF-tuning] {room_id}: tuning not reapplied: {type(error).__name__}")

    async def _microphone_telemetry_loop(self) -> None:
        """Poll only while capture auxiliaries may admit native work."""
        while self._running and self._auxiliary_allowed():
            await self._refresh_microphone_inventory()
            if not self._auxiliary_allowed():
                return
            await asyncio.sleep(self._microphone_interval_s)

    async def _stop_microphone_telemetry(self) -> None:
        """Join a retained stop owner without cancelling the actual read."""
        if self._microphone_stop_task is None:
            async def finish():
                task = self._microphone_telemetry_task
                if task is not None:
                    task.cancel()
                    await asyncio.gather(asyncio.shield(task), return_exceptions=True)
                worker = self._microphone_read_task
                if worker is not None:
                    await asyncio.shield(worker)
                    if worker.cancelled():
                        raise RuntimeError('microphone_read_unconfirmed')
                    if self._microphone_read_task is worker:
                        self._microphone_read_task = None
                if self._microphone_telemetry_task is task:
                    self._microphone_telemetry_task = None
            self._microphone_stop_task = asyncio.create_task(finish(), context=Context())
            self._microphone_stop_task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._microphone_stop_task)

    def _retire_microphone_capture(self, room_id: str) -> None:
        self._invalidate_direction(room_id)
        room = self._capture_rooms.get(room_id)
        if room is not None:
            room.retire_stream(self._microphone_wall_clock())
        holder = self._microphone_captures.get(room_id)
        if holder is not None:
            # Swap holders rather than mutate the callback-owned one. Even an
            # already executing old callback can only update retired memory.
            self._microphone_captures[room_id] = _CaptureHolder(replace(holder.observation, opened=False))

    def _base_microphone_snapshot(self) -> MicrophonesSnapshot | None:
        """Project the existing capture contract without direction lifecycle work."""
        inventory = self._microphone_inventory
        if inventory is None:
            return None
        return getattr(_facade(), 'build_microphone_snapshot', build_microphone_snapshot)(
            rooms=self._microphone_rooms, inventory=inventory,
            captures={room: holder.observation for room, holder in self._microphone_captures.items()},
            now_monotonic=self._microphone_monotonic(), now_wall=self._microphone_wall_clock(),
            stale_after_s=self._microphone_stale_after_s,
        )

    def _direction_key(self, mic) -> tuple | None:
        if (mic is None or mic.room_id not in self._direction_room_ids
                or mic.room_id in self._direction_opening):
            return None
        controller = self._xvf_by_room.get(mic.room_id)
        if (mic.connection != 'connected' or mic.binding != 'matched'
                or not mic.usb_port or mic.usb_port != mic.configured_usb_port
                or mic.usb_port != getattr(controller, 'usb_port', None)):
            return None
        return mic.mic_id, mic.usb_port, mic.capture.generation

    def _invalidate_direction(self, room_id: str) -> None:
        """Immediately hide an old generation; defer joins to async lifecycle."""
        with self._direction_lock:
            observer = self._direction_observers.pop(room_id, None)
            self._direction_keys.pop(room_id, None)
            if observer is not None:
                observer.invalidate()
                self._direction_retired.append(observer)
                self._direction_needs_reset.add(room_id)

    @staticmethod
    def _drain_direction_observer(observer: DirectionObserver) -> None:
        observer.stop()
        observer.wait_stopped()

    def _room_direction_calibration(self, room_id: str) -> DirectionCalibration | None:
        raw = self._direction_calibrations.get(room_id) if isinstance(self._direction_calibrations, dict) else False
        if raw is None or isinstance(raw, DirectionCalibration):
            return raw
        try:
            return DirectionCalibration(**raw)
        except (TypeError, ValueError):
            # Preserve the distinction between no calibration and an invalid one.
            return DirectionCalibration(0, 0, '', '', '')

    async def _sync_direction_observers(self) -> None:
        """Reconcile identity-bound workers outside GET and the audio callback."""
        epoch = self._auxiliary_epoch
        if not self._direction_started or not self._direction_room_ids or not self._auxiliary_allowed(epoch):
            return
        async with self._direction_lifecycle_lock:
            if not self._direction_started or not self._auxiliary_allowed(epoch):
                return
            snapshot = self._base_microphone_snapshot()
            current = {mic.room_id: mic for mic in snapshot.microphones} if snapshot else {}
            with self._direction_lock:
                changed = [room for room in self._direction_observers
                           if self._direction_key(current.get(room)) != self._direction_keys.get(room)]
            for room in changed:
                self._invalidate_direction(room)
            await self._drain_retired_directions()
            if not self._direction_started or not self._auxiliary_allowed(epoch):
                return
            for room_id in self._direction_room_ids:
                if not self._direction_started or not self._auxiliary_allowed(epoch):
                    return
                with self._direction_lock:
                    needs_reset = room_id in self._direction_needs_reset
                if needs_reset:
                    await self._xvf_native.run(self._xvf_by_room[room_id].invalidate_device)
                    if not self._direction_started or not self._auxiliary_allowed(epoch):
                        return
                    with self._direction_lock:
                        self._direction_needs_reset.discard(room_id)
                # Identity/generation can change while an old worker drains.
                snapshot = self._base_microphone_snapshot()
                mic = next((mic for mic in snapshot.microphones if mic.room_id == room_id), None) if snapshot else None
                key = self._direction_key(mic)
                if key is None:
                    continue
                with self._direction_lock:
                    fresh = self._base_microphone_snapshot()
                    fresh_mic = next((mic for mic in fresh.microphones if mic.room_id == room_id), None) if fresh else None
                    if (not self._direction_started or not self._auxiliary_allowed(epoch)
                            or room_id in self._direction_observers
                            or self._direction_key(fresh_mic) != key):
                        continue
                    observer = getattr(_facade(), 'DirectionObserver', DirectionObserver)(
                        controller=self._xvf_by_room[room_id], room_id=room_id, usb_port=mic.usb_port,
                        mic_id=mic.mic_id, config=self._direction_config,
                        calibration=self._room_direction_calibration(room_id),
                        monotonic=self._microphone_monotonic,
                        utc_now=lambda: datetime.fromtimestamp(self._microphone_wall_clock(), timezone.utc),
                    )
                    self._direction_observers[room_id] = observer
                    self._direction_keys[room_id] = key
                    observer.start()

    async def _drain_retired_directions(self) -> None:
        """Keep observers and shared real join tasks until successful return."""
        with self._direction_lock:
            retired = tuple(self._direction_retired)
        for observer in retired:
            if observer not in self._direction_join_tasks:
                self._direction_join_tasks[observer] = asyncio.create_task(
                    asyncio.to_thread(self._drain_direction_observer, observer), context=Context())
        tasks = tuple(self._direction_join_tasks[observer] for observer in retired)
        results = await asyncio.gather(*(asyncio.shield(task) for task in tasks),
                                       return_exceptions=True)
        errors = []
        for observer, result in zip(retired, results):
            if isinstance(result, BaseException):
                errors.append(result)
                continue
            with self._direction_lock:
                if observer in self._direction_retired:
                    self._direction_retired.remove(observer)
            self._direction_join_tasks.pop(observer, None)
        if errors:
            raise errors[0]

    async def _stop_direction_observers(self) -> None:
        """Invalidate all generations before awaiting complete worker drainage."""
        self._direction_started = False
        with self._direction_lock:
            room_ids = tuple(self._direction_observers)
        for room_id in room_ids:
            self._invalidate_direction(room_id)
        async with self._direction_lifecycle_lock:
            await self._drain_retired_directions()

    def get_microphone_snapshot(self) -> MicrophonesSnapshot | None:
        """Project cached capture and direction snapshots without any I/O."""
        snapshot = self._base_microphone_snapshot()
        if snapshot is None:
            return None
        microphones = []
        with self._direction_lock:
            for mic in snapshot.microphones:
                direction = getattr(_facade(), 'microphone_status', microphone_status).DirectionSnapshot(
                    state=getattr(_facade(), 'microphone_status', microphone_status).DirectionState.DISABLED,
                    reason=('capture_retirement_failed' if self._auxiliary_phase == 'failed'
                            else 'capture_retiring' if self._auxiliary_phase in ('retiring', 'closing')
                            else 'capture_paused') if self._paused or self._closed else self._direction_reason,
                )
                if mic.room_id in self._direction_room_ids and self._direction_started:
                    observer = self._direction_observers.get(mic.room_id)
                    key = self._direction_key(mic)
                    if observer is not None and key is not None and key == self._direction_keys.get(mic.room_id):
                        direction = observer.snapshot()
                    else:
                        direction = replace(direction, state=getattr(_facade(), 'microphone_status', microphone_status).DirectionState.UNKNOWN,
                                            reason='identity_unavailable')
                microphones.append(replace(mic, direction=direction))
        return replace(snapshot, microphones=tuple(microphones))

    def get_audio_health(self) -> dict:
        """Return room frame health without file I/O or audio content."""
        now_wall = getattr(_facade(), 'time', time).time()
        self._update_audio_anchors(now_mono=getattr(_facade(), 'time', time).monotonic(), now_wall=now_wall)
        return {
            "rooms": {
                room: {"age_s": max(0.0, now_wall - anchor), "ever": ever}
                for room, (anchor, ever) in self._audio_health_rooms().items()
            },
            "deaf_after_s": self._watchdog_timeout_s,
            "first_frame_grace_s": self._watchdog_first_frame_grace_s,
        }

    def _publish_audio_health(self) -> None:
        """Refresh anchors and admit at most one retained native publication."""
        now = getattr(_facade(), 'time', time).monotonic()
        now_wall = getattr(_facade(), 'time', time).time()
        self._update_audio_anchors(now_mono=now, now_wall=now_wall)
        task = self._audio_health_task
        if (task is not None and not task.done()) or self._audio_health_work._tasks:
            if not self._audio_health_timed_out and now - self._audio_health_started >= AUDIO_HEALTH_WRITE_TIMEOUT_S:
                self._audio_health_timed_out = True
                logger.warning('Audio health publication pending: native write still owned')
            return
        if not self._auxiliary_allowed():
            return
        from src.monitoring.audio_health import write_audio_health

        self._audio_health_started = now
        self._audio_health_timed_out = False
        task = self._audio_health_task = asyncio.create_task(
            self._audio_health_work.run(write_audio_health, self._audio_health_path,
                self._audio_health_rooms(), now_wall=now_wall), context=Context())

        def finished(done):
            if done.cancelled():
                logger.warning('Audio health publication observer cancelled')
            elif done.exception() is not None:
                logger.warning('Audio health publication failed: %s', type(done.exception()).__name__)
            if not done.cancelled() and self._audio_health_task is done:
                self._audio_health_task = None

        task.add_done_callback(finished, context=Context())

    async def _stream_watchdog(self) -> None:
        """Publish health independently of native I/O and one recovery owner."""
        while self._running:
            await asyncio.sleep(self._watchdog_check_interval_s)
            if not self._running:
                break
            try:
                self._publish_audio_health()
            except Exception as error:
                logger.warning('Audio health publication failed: %s', type(error).__name__)
            if not self._recovery_ready or not self._inputs.accepting:
                continue
            now = getattr(_facade(), 'time', time).monotonic()
            states = []
            for room_id, rs in self.room_streams.items():
                if rs.last_frame_ts > 0.0 and now - rs.last_frame_ts <= self._watchdog_timeout_s:
                    # Real audio ends the dead-capture streak (T048-U1).
                    self._xvf_reboot_budget.note_audio(room_id)
                    self._xvf_reboot_deferred.pop(room_id, None)
                if not self._is_awaiting_reopen(room_id):
                    states.append((room_id, rs.last_frame_ts, rs.opened_ts))
            stale = getattr(_facade(), 'detect_stale_streams', detect_stale_streams)(states, now,
                self._watchdog_timeout_s, self._watchdog_first_frame_grace_s)
            if stale:
                self._recover_dead_streams(stale)

    def _xvf_reboot_controller(self, room_id: str):
        """The room's controller when it can send a firmware reboot, else None."""
        ctrl = self._xvf_by_room.get(room_id)
        if (ctrl is None or not callable(getattr(ctrl, "reboot", None))
                or not callable(getattr(ctrl, "wait_reenumerated", None))
                or not self._xvf_room_port(room_id)):
            return None
        return ctrl

    def _note_xvf_reboot_deferred(self, room_id: str, reason: str) -> None:
        """Log once per blocker that a due firmware reboot waits for it.

        A deferral longer than ``REBOOT_DEFERRAL_ALERT_S`` is logged once as
        an error: an activity flag that never clears would otherwise keep the
        reboot away forever with a single info line (T048-U1-RV1 F3).
        """
        now = self._microphone_monotonic()
        deferral = self._xvf_reboot_deferred.get(room_id)
        if deferral is None:
            deferral = self._xvf_reboot_deferred[room_id] = _RebootDeferral(reason, now)
            logger.info(f"[audio-watchdog] {room_id}: XVF firmware reboot deferred ({reason})")
        elif deferral.reason != reason:
            deferral.reason = reason
            logger.info(f"[audio-watchdog] {room_id}: XVF firmware reboot deferred ({reason})")
        if not deferral.escalated and now - deferral.since >= REBOOT_DEFERRAL_ALERT_S:
            deferral.escalated = True
            logger.error(
                f"[audio-watchdog] {room_id}: XVF firmware reboot deferred for "
                f"{now - deferral.since:.0f}s ({reason}); capture is still dead and only the "
                "ordinary reopen runs; check whether that activity is stuck"
            )

    def _recover_dead_streams(self, stale: list[str]) -> None:
        """Route rooms whose stream delivers no audio to a reboot or the recovery.

        A room whose ordinary recovery already failed the configured number
        of times gets a firmware reboot of its own XVF3800 instead, unless a
        command, conversation or response is active: then the reboot waits
        and the room keeps the ordinary recovery it had before T048-U1.
        Rooms over the hourly cap, or without a usable controller, keep the
        ordinary recovery too.

        Args:
            stale: Rooms the watchdog found without audio.
        """
        budget = self._xvf_reboot_budget
        ordinary = []
        for room_id in stale:
            decision = budget.decide(room_id)
            if decision is XvfRebootDecision.REBOOT:
                if self._xvf_reboot_controller(room_id) is None:
                    if room_id not in self._xvf_reboot_unavailable:
                        self._xvf_reboot_unavailable.add(room_id)
                        logger.warning(
                            f"[audio-watchdog] {room_id}: capture dead after "
                            f"{budget.failures(room_id)} reopens but the room has no XVF "
                            "controller bound to its mic port; firmware reboot unavailable"
                        )
                    ordinary.append(room_id)
                    continue
                reason = self._catalog_refresh_blocker()
                if reason is not None:
                    # Keep the ordinary recovery meanwhile, exactly as before
                    # T048-U1: the room is never left worse off than without
                    # the reboot, even if the activity flag sticks (RV1 F3).
                    self._note_xvf_reboot_deferred(room_id, reason)
                    ordinary.append(room_id)
                    continue
                # One recovery owner at a time: a busy owner retries next check.
                self._schedule_recovery([room_id], xvf_reboot=True)
                continue
            if decision is XvfRebootDecision.CAPPED and budget.report_cap(room_id):
                logger.error(
                    f"[audio-watchdog] {room_id}: isochronous capture still dead after "
                    f"{budget.reboots_in_window(room_id)} XVF firmware reboots in the last hour "
                    f"(cap {budget.policy.max_per_hour}); no more reboots for now: physically "
                    f"unplug and replug the microphone at USB port "
                    f"{self._xvf_room_port(room_id) or self.room_streams[room_id].mic_usb_port}"
                )
            ordinary.append(room_id)
        if not ordinary:
            return
        openings = {room_id: self.room_streams[room_id].opened_ts for room_id in ordinary}
        if self._schedule_recovery(ordinary):
            for room_id, opening in openings.items():
                budget.note_recovery(room_id, opening)
