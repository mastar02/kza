"""
Multi-Room Audio Loop
Opens one sounddevice InputStream per XVF3800 microphone.
Each room's stream independently detects wake words and captures commands.
Concurrent commands from different rooms are processed in parallel.
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

from src.pipeline.room_stream_manager import (
    PortAudioReset,
    RoomStream,
    RoomStreamManagerMixin,
    XvfRebootReport,
    _CaptureHolder,
    _RebootDeferral,
    _resolve_capture_channels,
    detect_stale_streams,
)
from src.pipeline.wake_pipeline import (
    WakePipelineMixin,
    _fmt_vad,
    _pcm_context,
    compute_wake_vad,
)
from src.pipeline.command_capture import CommandCaptureMixin


class MultiRoomAudioLoop(RoomStreamManagerMixin, WakePipelineMixin, CommandCaptureMixin):
    """
    Parallel audio capture from multiple XVF3800 microphones.

    Opens one sd.InputStream per room. Each stream runs wake word
    detection independently. When a stream captures a complete command,
    it dispatches a CommandEvent via asyncio.create_task (non-blocking),
    allowing concurrent processing of commands from different rooms.
    """

    # Post-wake rule (V1); the class default keeps partially built loops
    # (tests that bypass __init__) on the legacy endpoint.
    _post_wake: PostWakeConfig | None = None
    # Room whose wake, accepted command or barge-in opened the current
    # follow-up conversation; follow-ups only start there (post-wake rule).
    _conversation_room: str | None = None

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
        response_handler=None,
        barge_in_enabled: bool = False,
        barge_in_rms_threshold: float = 0.03,
        barge_in_min_duration_ms: int = 200,
        min_wake_rms: float = 0.0,
        wake_preroll_s: float = 0.0,
        xvf_controllers: dict | None = None,
        spenergy_threshold: float = 100.0,
        spenergy_gate_enabled: bool = True,
        xvf_tuning: dict | None = None,
        ambient_guard: AmbientGuard | None = None,
        wake_clip_writer=None,
        stream_watchdog_enabled: bool = False,
        stream_watchdog_no_frames_timeout_s: float = 8.0,
        stream_watchdog_check_interval_s: float = 2.0,
        stream_watchdog_reopen_backoff_min_s: float = 1.0,
        stream_watchdog_reopen_backoff_max_s: float = 10.0,
        stream_watchdog_first_frame_grace_s: float = 180.0,
        stream_watchdog_catalog_refresh_min_interval_s: float = 0.0,
        stream_watchdog_catalog_refresh_max_interval_s: float = 3600.0,
        xvf_reboot_policy: XvfRebootPolicy | None = None,
        audio_health_path: str = "./data/audio_health.json",
        microphone_rooms: tuple[MicrophoneRoom, ...] | None = None,
        microphone_usb_root: Path = Path("/sys/bus/usb/devices"),
        microphone_reader: Callable[[Path], UsbInventory] | None = None,
        microphone_monotonic: Callable[[], float] | None = None,
        microphone_wall_clock: Callable[[], float] | None = None,
        microphone_interval_s: float = 2.0,
        microphone_stale_after_s: float = 6.0,
        direction_observation: object = None,
        direction_calibrations: dict | None = None,
        privacy=None,
        close_timeout_s: float = 1,
        post_wake: PostWakeConfig | None = None,
    ):
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (microphone_interval_s, microphone_stale_after_s))
            or microphone_stale_after_s < microphone_interval_s
        ):
            raise ValueError("invalid microphone telemetry interval/stale window")
        refresh_interval = stream_watchdog_catalog_refresh_min_interval_s
        refresh_ceiling = stream_watchdog_catalog_refresh_max_interval_s
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (refresh_interval, refresh_ceiling)):
            raise ValueError("invalid PortAudio catalog refresh interval")
        self.room_streams = room_streams
        self.follow_up = follow_up
        self.sample_rate = sample_rate
        self.command_duration = command_duration
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms
        self.dedup_window_ms = dedup_window_ms
        # Post-wake listening rule (V1 2026-11-17). None keeps the legacy
        # endpoint (absolute RMS, timeout counted from the pre-roll start);
        # settings.yaml enables it under rooms.endpointing.post_wake.
        self._post_wake = post_wake
        self.endpoint_counts: dict[str, int] = {}

        self.early_dispatch_enabled = early_dispatch_enabled
        self.early_dispatch_interval_ms = early_dispatch_interval_ms
        self.early_dispatch_min_audio_s = early_dispatch_min_audio_s
        self._stt = stt  # FastWhisperSTT — usado por el worker early parse
        self._privacy = privacy
        self._closed = False
        self._closing = None
        self._close_timeout_s = close_timeout_s
        self._capture_controllers = {}
        self._capture_tasks = set()
        self._capture_batches = set()
        self._dispatch_tasks = set()
        # Q33 (review 3): a command being captured in some room or dispatched, which a held reminder
        # notice must not talk over even after the follow-up window timer ended the conversation.
        self.command_turns = CommandTurns(self._command_turn_busy)
        self._early_tasks = set()
        self._capture_native = OwnedBlockingWork(capacity=8)
        self._scopes = ProducerScopes(self.snapshot_operations, capacity=min(256, max(64, len(room_streams)*4+24)))
        self.native_model_locks = {id(stt): Lock()} if stt is not None else {}

        # Endpointing adaptativo (S5): usa la señal del parser streaming
        # (`rs.early_command`) para decidir cuánto silencio esperar antes
        # de cerrar la captura. Con parser ready → corte rápido (short_ms).
        # Sin parser ready → corte normal (medium_ms). Sin señal alguna del
        # parser → espera más (long_ms, reservado para futuro uso).
        self.endpointing_enabled = endpointing_enabled
        self.endpointing_short_ms = endpointing_short_ms
        self.endpointing_medium_ms = endpointing_medium_ms
        self.endpointing_long_ms = endpointing_long_ms

        # Barge-in (S3). `response_handler` puede setearse post-init via
        # `attach_response_handler()` — útil porque el ResponseHandler se
        # construye después del loop en main.py.
        self._response_handler = response_handler
        self.barge_in_enabled = barge_in_enabled
        self.barge_in_rms_threshold = barge_in_rms_threshold
        self.barge_in_min_duration_ms = barge_in_min_duration_ms

        # Pre-gate de energía post-wake (2026-06-02). Descarta activaciones de
        # muy baja RMS (near-silence) ANTES de transcribir → ataca las capturas
        # Text='' y las alucinaciones 'Gracias.' sobre silencio. Default 0.0 =
        # desactivado (sin regresión). CALIBRAR en repro: el AGC ×64 del XVF3800
        # infla el piso de ruido (~0.025-0.05), así que el valor útil se mide
        # con voz real vs ambiente, idealmente tras bajar el AGC.
        self.min_wake_rms = min_wake_rms

        # Pre-roll (2026-06-02): cuántos chunks previos al wake conservar para
        # sembrar el buffer del comando. openwakeword tiene latencia de detección
        # (~0.5-1s): el verbo dicho justo tras "Nexa" se pierde porque la captura
        # arranca recién al disparar. El pre-roll lo recupera. 0.0 = desactivado
        # (sin regresión); settings.yaml lo activa (rooms.wake_word.wake_preroll_s).
        self.wake_preroll_s = wake_preroll_s
        self._preroll_maxchunks = (
            int(round(wake_preroll_s * sample_rate / CHUNK_SIZE))
            if wake_preroll_s > 0 else 0
        )

        # Captura de clips de wake (2026-06-12): cada wake ACEPTADO persiste su
        # preroll como WAV (dataset de re-entrenamiento: falsos de TV = hard
        # negatives; comandos reales = positivos far-field). Requiere preroll
        # activo (sin preroll no hay audio que guardar). None = apagado.
        self._wake_clip_writer = wake_clip_writer

        # Pre-gate SPENERGY (2026-06-02): VAD por hardware del XVF3800. Si el
        # pico de SPENERGY durante la captura quedó por debajo del umbral, era
        # secador/silencio → no transcribir (mata alucinaciones de Whisper).
        # Fail-open: sin controller/datos, procesa siempre. Medido: secador/
        # silencio=0, voz≥52k → umbral 100 separa con margen enorme.
        # UN controller POR ROOM, bindeado al mic_usb_port de esa habitación
        # (2026-07-25): con dos XVF3800 conectados, un controller compartido
        # tunea/gatea con el mic de OTRA room — usb.core.find() devuelve el
        # primero que enumere, y el orden no es estable. Una room sin
        # controller propio NO se gatea (fail-open) en vez de usar el ajeno.
        self._xvf_by_room: dict = dict(xvf_controllers or {})
        self._xvf_native = OwnedBlockingWork(capacity=8, isolate_context=True)
        self.spenergy_threshold = spenergy_threshold
        # Gate y tuning son features ORTOGONALES sobre el mismo controller
        # (review Fase 1): se puede tunear el DSP con el gate apagado. El
        # poller solo corre si el gate está habilitado (el tuning no lo usa).
        self.spenergy_gate_enabled = spenergy_gate_enabled

        # Tuning del DSP XVF3800 (L-2 2026-06-04): writes EN RAM aplicados al
        # arrancar (reversibles al re-enchufar; SAVE_CONFIGURATION no existe en
        # el command-map a propósito). Default None/apply_on_start=False = cero
        # writes. Como es RAM, un re-enchufe o un REBOOT de firmware restauran
        # el preset de fábrica: desde T048-U1 el tuning se re-aplica a la room
        # cuyo device re-enumera (inventario USB) y tras cada REBOOT propio.
        self._xvf_tuning = xvf_tuning or {}
        # room_id -> (devnum, alsa_card) last seen at the controller's port
        # (None = absent). A change means the device re-enumerated.
        self._xvf_port_seen: dict[str, tuple | None] = {}
        # Rooms whose firmware reboot is in flight; that reboot reapplies the
        # tuning itself, so the inventory does not do it a second time.
        self._xvf_rebooting: set[str] = set()

        # AmbientGuard (spec 2026-06-05): compuerta acústica integral por room.
        # None = sin guard (comportamiento previo exacto). La escalera
        # NORMAL/STRICT/COOLDOWN se alimenta de los resultados de captura
        # (_dispatch_command) y decide sobre cada wake (_should_accept_wakeword).
        self._guard = ambient_guard

        # Ambient path (spec 2026-06-06): tap multicanal + transcriber para la
        # señal shadow anti-TV. attach_ambient() post-init (orden de DI en main,
        # mismo patrón que attach_response_handler). None = feature apagada.
        self._ambient_tap = None
        self._ambient_transcriber = None
        self._wake_vad_predictors: dict[str, object] = {}  # one native recurrent state per room
        self._wake_vad_owned: dict[int, object] = {}  # includes late, unpublished loads

        self._watchdog_enabled = stream_watchdog_enabled
        self._watchdog_timeout_s = stream_watchdog_no_frames_timeout_s
        self._watchdog_check_interval_s = stream_watchdog_check_interval_s
        self._watchdog_backoff_min_s = stream_watchdog_reopen_backoff_min_s
        self._watchdog_backoff_max_s = stream_watchdog_reopen_backoff_max_s
        # Margen para el PRIMER frame tras abrir (arranques medidos:
        # 1.5-2s lo normal, 135s el peor observado el 2026-07-29).
        self._watchdog_first_frame_grace_s = stream_watchdog_first_frame_grace_s
        # Late microphone (T025-U2, Cocina 2026-11-15): PortAudio lists only the
        # devices present at Pa_Initialize. A configured card that enumerates
        # later stays invisible while the other rooms are healthy, so the reopen
        # waiter requests one global catalog refresh (a brief capture cut in
        # every room), at most once per interval and never during a command,
        # a conversation or a response. 0 disables it.
        self._catalog_refresh_interval_s = float(refresh_interval)
        # A card PortAudio still does not list after a refresh (busy device,
        # failed capture interface) backs the next refresh off: the interval,
        # then doubling per refresh that left the same card unlisted, up to
        # this ceiling (never below the interval). T025-U2-RV2-3.
        self._catalog_refresh_ceiling_s = max(float(refresh_ceiling), self._catalog_refresh_interval_s)
        # room_id -> ALSA card seen in sysfs but not listed by PortAudio, as
        # observed by the latest open attempt (written by the native factory).
        self._catalog_missing: dict[str, int] = {}
        # room_id -> (card, time.monotonic() when an open attempt first saw
        # that card in sysfs but not in PortAudio). A card that just appeared
        # may not be openable yet (udev permissions, ALSA readiness) and
        # PortAudio skips devices it cannot open, so the first refresh waits
        # until the gap was seen at least the reopen backoff ceiling apart
        # (T025-U2-RV3-4).
        self._catalog_gap_seen: dict[str, tuple[int, float]] = {}
        self._catalog_refreshed_at: float | None = None
        self._catalog_refresh_deferred: str | None = None
        # Consecutive catalog resets that left PortAudio uninitialized. The
        # house is deaf meanwhile, so the next refresh waits only a short
        # backoff instead of the full interval.
        self._catalog_reinit_failures = 0
        # room_id -> (card, consecutive refreshes after which PortAudio still
        # did not list that card). Dropped when the card leaves sysfs, comes
        # back with another number, or is listed.
        self._catalog_futile: dict[str, tuple[int, int]] = {}
        # Consecutive refreshes that ran while no microphone was capturing and
        # still left a card unlisted. The cut interrupts nothing then, so the
        # next refresh waits only the short backoff and the futile streak
        # above, which protects working rooms, does not grow (T025-U2-RV3-1).
        self._catalog_deaf_refreshes = 0
        # Stuck XVF3800 (T048-U1, incident 2026-11-19): a stream that keeps
        # opening dead after the ordinary recovery failed N times gets a
        # firmware REBOOT of that room's device only, bounded per hour.
        self._xvf_reboot_budget = XvfRebootBudget(
            xvf_reboot_policy or XvfRebootPolicy(), clock=lambda: self._microphone_monotonic())
        # room_id -> a due reboot waiting for an activity (log once per
        # blocker; an error once if it waits too long).
        self._xvf_reboot_deferred: dict[str, _RebootDeferral] = {}
        self._xvf_reboot_unavailable: set[str] = set()
        self._watchdog_task = None
        self._recovery_task = None
        self._recovery_epoch = 0
        self._recovery_ready = True
        self._global_recovery = False
        self._input_recovery_pause = None
        self._audio_health_work = OwnedBlockingWork(capacity=1)
        self._audio_health_task = None
        self._audio_health_started = 0.0
        self._audio_health_timed_out = False
        # Snapshot del heartbeat para el poller externo (2026-07-31): el
        # watchdog interno recupera mics pero no puede AVISAR — si el
        # proceso se traba, se traba con él. Ver tools/audio_watchdog_alert.py.
        self._audio_health_path = audio_health_path
        # Anclas de salud en WALL CLOCK, fuera de RoomStream a propósito: son
        # lo único que la reapertura de un stream NO puede resetear. Ver
        # `_update_audio_anchors` para por qué es imprescindible.
        # room_id -> time.time() del último audio REAL (presencia ⇒ ever=True)
        self._last_audio_wall: dict[str, float] = {}
        # room_id -> time.time() en que la room entró en observación; ancla de
        # la room que todavía no entregó nada.
        self._room_since_wall: dict[str, float] = {}

        self._running = False
        self._paused = False
        self._pausing = None
        self._auxiliary_epoch = 0
        self._auxiliary_phase = 'idle'
        self._auxiliary_resume_task = None
        self._auxiliary_resume_epoch = None
        self._auxiliary_started_epoch = None
        self._auxiliary_pause_task = None
        self._capture_start_task = None
        self._start_requested = False
        self._run_started = False
        self._auxiliary_start_failed = False
        self._microphone_stop_task = None
        self._direction_join_tasks = {}
        self._xvf_pause_joins = {}
        self._streams: dict = {}
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

        # Wake textual (spec 2026-07-05): timestamp monotónico del último
        # dispatch ACÚSTICO por room — consumido por TextualWakeDetector
        # (last_acoustic_command_ts_fn) para el dedup cruzado acústico/textual.
        self._last_command_dispatch_ts: dict[str, float] = {}

        # Esperas de reapertura en curso, una por room (incidente 2026-07-28).
        # Viven fuera de _recover_streams para que un mic ausente no bloquee
        # ni a las otras rooms ni al watchdog.
        self._reopen_tasks: dict[str, asyncio.Task] = {}
        self._microphone_rooms = microphone_rooms if microphone_rooms is not None else tuple(
            MicrophoneRoom(rs.room_id, rs.room_id, rs.mic_usb_port, device_index=rs.device_index)
            for rs in room_streams.values()
        )
        self._microphone_usb_root = microphone_usb_root
        self._microphone_reader = microphone_reader
        self._microphone_monotonic = microphone_monotonic or time.monotonic
        self._microphone_wall_clock = microphone_wall_clock or time.time
        self._microphone_inventory: UsbInventory | None = None
        self._microphone_captures: dict[str, _CaptureHolder] = {}
        self._microphone_telemetry_task: asyncio.Task | None = None
        self._microphone_read_task: asyncio.Task | None = None
        self._microphone_interval_s = float(microphone_interval_s)
        self._microphone_stale_after_s = float(microphone_stale_after_s)
        self._microphone_startup_budget_s = 0.25
        self._microphone_inventory_ready = asyncio.Event()
        self._direction_config, self._direction_room_ids, self._direction_reason = parse_direction_observation(
            direction_observation, {room.room_id for room in self._microphone_rooms},
        )
        self._direction_calibrations = direction_calibrations or {}
        self._direction_observers: dict[str, DirectionObserver] = {}
        self._direction_keys: dict[str, tuple] = {}
        self._direction_retired: list[DirectionObserver] = []
        self._direction_needs_reset: set[str] = set()
        self._direction_opening: set[str] = set()
        self._direction_lock = Lock()
        self._direction_lifecycle_lock = asyncio.Lock()
        self._direction_started = False
        if self._direction_reason:
            logger.warning('[direction-observation] Pilot disabled: %s', self._direction_reason)
        self._capture_rooms = {
            room_id: CaptureRoom(room_id, process=self._make_audio_processor(rs),
                poll=lambda rs=rs: self._poll_command(rs),
                clear=lambda rs=rs: self._clear_capture(rs),
                listening=lambda rs=rs: rs.listening, privacy=privacy)
            for room_id, rs in room_streams.items()
        }
        self._inputs = InputGroup(room_streams,
            factory=lambda key: self._make_stream(self.room_streams[key], report_not_admitted=True),
            configure=lambda key, stream: self._configure_stream(self.room_streams[key], stream),
            retire=self._retire_microphone_capture,
            callbacks=lambda key: self._capture_rooms[key].callbacks)
        self._streams = self._inputs.streams
        if privacy is not None:
            privacy.add_borrower(self.begin_close)

    def attach_response_handler(self, response_handler) -> None:
        """Inyectar ResponseHandler post-init (útil por orden de DI en main.py)."""
        self._response_handler = response_handler

    def _clear_capture(self, rs: RoomStream) -> None:
        self._reset_listening(rs)
        rs.preroll = None
        rs.preroll_metadata = None
        rs.barge_in_accum_ms = 0.0
        reset = getattr(rs.wake_detector, 'reset', None)
        if not callable(reset):
            raise RuntimeError('unsupported_capture_reset')
        reset()
        rs.wake_opening = rs.wake_channel = None
        predictor = self._wake_vad_predictors.get(rs.room_id)
        reset = getattr(predictor, 'reset', None)
        if callable(reset):
            reset()

    def _poll_command(self, rs: RoomStream) -> CommandEvent | None:
        if not rs.listening:
            return None
        complete, audio = self._check_vad_completion(rs)
        if complete and audio is None and self._post_wake is not None:
            self._drop_capture(rs)
            return None
        if not complete or audio is None:
            return None
        if not self._passes_spenergy_gate(rs):
            self._clear_capture(rs)
            return None
        return CommandEvent(audio=audio, room_id=rs.room_id,
            mic_device_index=rs.device_index, wake_text=rs.wake_text,
            partial_command=rs.early_command, transcript=rs.transcript,
            capture_complete=rs.capture_complete,
            ambient_strict=self._guard is not None and self._guard.state_for(rs.room_id) is GuardState.STRICT,
            wake_score=rs.wake_score, captured_at=rs.command_start_time,
            audio_provenance=self._capture_provenance(rs, audio),
            wake_vad=rs.wake_vad, capture_trigger=rs.capture_trigger.value,
            endpoint_code=(rs.endpoint.decision.value
                           if rs.endpoint is not None and rs.endpoint.decision is not None else None))

    def _drop_capture(self, rs: RoomStream) -> None:
        """Discard a capture without speech after the wake (no STT).

        An acoustic wake that captured nothing feeds the ambient guard as an
        empty capture, like the empty transcript it used to produce. The
        follow-up window opened by the wake stays as it was.
        """
        trigger = rs.capture_trigger
        room_id = rs.room_id
        self._clear_capture(rs)
        if self._guard is not None and trigger is CaptureTrigger.WAKE:
            try:
                self._guard.on_capture_result(room_id, "empty")
            except Exception as error:
                logger.debug("[endpoint] guard update skipped: %s", type(error).__name__)

    def _capture_provenance(self, rs: RoomStream, audio):
        if sum(part[0] for part in rs.audio_metadata) != audio.size:
            return None
        opening, channel, conflict = _pcm_context(rs.audio_metadata)
        if conflict or opening is None:
            return None
        if rs.inline_provenance is not None and rs.inline_provenance.frame_count == audio.size:
            return rs.inline_provenance
        return opening.project(lane=CaptureLane.COMMAND, segment_generation=rs.capture_generation,
            selected_channel=channel, frame_count=audio.size, input_ref=uuid4().hex)

    def _track_capture(self, task: asyncio.Task, tasks: set) -> asyncio.Task:
        tasks.add(task)
        def finished(done):
            tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    def _source_live(self, source) -> bool:
        return not self._closed and not self._paused and (self._privacy is None or source is not None and
            self._privacy.sources_available((source,)))

    def _start_dispatch(self, event: CommandEvent) -> None:
        from src.memory.fact_ledger import MemoryOrigin, MemorySource
        source = None
        try:
            if self._privacy is not None:
                source = MemorySource(event.capture_id, event.conversation_id, event.captured_at,
                                      event.source_expires_at, MemoryOrigin.VOICE)
            if not self._source_live(source):
                return
            if len(self._dispatch_tasks) >= 8:
                logger.warning('Command dispatch capacity reached')
                return
            sources = (source,) if source is not None else ()
            with self._scopes.operation(sources), producer_context(event.room_id, sources):
                dispatch = self._track_capture(asyncio.create_task(run_private(
                    self._dispatch_command(event), self._dispatch_live)), self._dispatch_tasks)
            dispatch.add_done_callback(self._settle_turn)
        finally:
            event = None

    def _command_turn_busy(self) -> bool:
        return bool(self._dispatch_tasks) or any(rs.listening for rs in tuple(self.room_streams.values()))

    def _settle_turn(self, _done=None) -> None:
        self.command_turns.settle()

    async def _consume_room(self, rs: RoomStream) -> None:
        room = self._capture_rooms[rs.room_id]
        batch = None
        try:
            while self._running and not self._closed:
                if self._paused:
                    await asyncio.sleep(.05)
                    continue
                batch = self._track_capture(asyncio.create_task(self._consume_once(rs)), self._capture_batches)
                try:
                    await batch
                except SourcesInvalidated:
                    if room.resource_status()['closed'] or room.resource_status()['reset_failed']:
                        raise
                except asyncio.CancelledError:
                    if self._closed or not self._paused or asyncio.current_task().cancelling():
                        raise
                finally:
                    batch = None
                await asyncio.sleep(.05)
        except Exception as error:
            logger.error('Command capture controller stopped: %s', type(error).__name__)
            room.begin_close()
        finally:
            batch = None

    async def _consume_once(self, rs):
        event = step = None
        room = self._capture_rooms[rs.room_id]
        with self._scopes.operation(()) as handoff:
            try:
                step = self._track_capture(asyncio.create_task(room.step(handoff=handoff)), self._capture_tasks)
                event = await step
                if event is not None:
                    self._start_dispatch(event)
                elif rs.listening and self.early_dispatch_enabled and self._stt is not None and rs.early_task is None:
                    source = room.source
                    sources = (source,) if source is not None else ()
                    with self._scopes.operation(sources), producer_context(rs.room_id, sources):
                        rs.early_task = self._track_capture(asyncio.create_task(run_private(
                            self._early_parse_worker(rs), lambda source=source: self._source_live(source))), self._early_tasks)
            finally:
                event = step = None
                self.command_turns.settle()

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose finite capture and auxiliary owners without idle loops."""
        tasks = self._capture_tasks | self._capture_batches | self._early_tasks | self._dispatch_tasks | self._capture_native._tasks
        tasks |= set(self._xvf_native._tasks)
        for room in self._capture_rooms.values():
            tasks |= set(room.snapshot_operations())
        tasks |= set(self._inputs.snapshot_operations())
        tasks.update(task for task in (
            self._recovery_task, self._input_recovery_pause, self._capture_start_task,
            self._auxiliary_resume_task, self._auxiliary_pause_task, self._microphone_read_task,
        ) if task is not None and not task.done())
        tasks.update(task for task in self._direction_join_tasks.values() if not task.done())
        tasks.update(task for task in self._xvf_pause_joins.values() if not task.done())
        if self._pausing is not None and not self._pausing.done():
            tasks.add(self._pausing)
        if isinstance(self._wake_clip_writer, WakeClipWriter):
            tasks |= set(self._wake_clip_writer.snapshot_operations())
        return tuple(tasks)

    def conversation_busy(self, conversation_id: str) -> bool:
        """Include finite work, actual input callbacks and room idle buffers."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        return self._scopes.busy(conversation_id, self._privacy.ledger) or any(
            room.conversation_busy(conversation_id) for room in self._capture_rooms.values())

    async def prune_private(self) -> None:
        """Retire invalid room buffers without waiting for another input frame."""
        if self._paused:
            return
        await asyncio.gather(*(room.prune_private() for room in self._capture_rooms.values()))

    def capture_status(self) -> dict:
        """Report real auxiliary admission/ownership without private media."""
        return {'closed': self._closed, 'paused': self._paused, 'operations': len(self.snapshot_operations()),
                'batches': len(self._capture_batches), 'source_scopes': self._scopes.resource_status(),
                'inputs': self._inputs.resource_status(),
                'audio_health': self._audio_health_work.status(),
                'auxiliary_phase': self._auxiliary_phase,
                'auxiliary_epoch': self._auxiliary_epoch,
                'auxiliary_resume_pending': (self._auxiliary_resume_task is not None
                                             and not self._auxiliary_resume_task.done()),
                'rooms': [room.resource_status() for room in self._capture_rooms.values()]}

    def begin_pause_inputs(self) -> asyncio.Task:
        """Seal recovery now and retain its real input/native pause barrier."""
        if self._input_recovery_pause is None:
            self._recovery_epoch += 1
            pending = self._cancel_reopen_tasks()
            recovery = self._recovery_task
            pause = self._inputs.begin_pause()

            async def finish():
                await asyncio.gather(*pending, return_exceptions=True)
                if recovery is not None:
                    await asyncio.gather(asyncio.shield(recovery), return_exceptions=True)
                await asyncio.shield(pause)

            self._input_recovery_pause = asyncio.create_task(finish(), context=Context())
            self._input_recovery_pause.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        return self._input_recovery_pause

    def _auxiliary_allowed(self, epoch=None) -> bool:
        return (not self._closed and not self._paused
                and (epoch is None or epoch == self._auxiliary_epoch))

    def _seal_auxiliaries(self) -> None:
        self._direction_started = False
        with self._direction_lock:
            rooms = tuple(self._direction_observers)
        for room_id in rooms:
            self._invalidate_direction(room_id)
        for ctrl in self._xvf_by_room.values():
            if isinstance(ctrl, XvfController):
                ctrl.request_stop()
        telemetry = self._microphone_telemetry_task
        if telemetry is not None and not telemetry.done():
            telemetry.cancel()
        self._microphone_inventory_ready.set()

    def _start_microphone_telemetry(self, epoch) -> None:
        if not self._running or not self._run_started or not self._auxiliary_allowed(epoch):
            return
        if self._microphone_telemetry_task is None or self._microphone_telemetry_task.done():
            self._microphone_stop_task = None
            self._microphone_telemetry_task = asyncio.create_task(
                self._microphone_telemetry_loop(), context=Context())

    async def _ensure_capture_models(self) -> None:
        if self._capture_start_task is None:
            async def initialize():
                for rs in self.room_streams.values():
                    if self._closed:
                        return
                    try:
                        with producer_context(rs.room_id):
                            await self._capture_native.run(self._load_capture_models, rs)
                    except Exception as error:
                        logger.error('Room wake unavailable: %s', type(error).__name__)
                        await self._capture_rooms[rs.room_id].begin_close()
            self._capture_start_task = asyncio.create_task(initialize(), context=Context())
            self._capture_start_task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._capture_start_task)

    async def _start_auxiliaries(self, epoch) -> None:
        try:
            if self._start_requested:
                await self._ensure_capture_models()
                if not self._auxiliary_allowed(epoch):
                    self._seal_auxiliaries()
                    return
                await self._xvf_native.run(self._apply_xvf_tuning)
                if not self._auxiliary_allowed(epoch):
                    self._seal_auxiliaries()
                    return
                if self.spenergy_gate_enabled:
                    for room_id, ctrl in self._xvf_by_room.items():
                        if not self._auxiliary_allowed(epoch):
                            self._seal_auxiliaries()
                            return
                        try:
                            started = await self._xvf_native.run(ctrl.start)
                        except Exception as error:
                            self._auxiliary_start_failed = True
                            logger.warning('SPENERGY start failed in %s: %s',
                                           room_id, type(error).__name__)
                        else:
                            if not started:
                                logger.warning('SPENERGY unavailable in %s; gate remains fail-open', room_id)
                        if not self._auxiliary_allowed(epoch):
                            self._seal_auxiliaries()
                            return
                self._direction_started = True
                await self._sync_direction_observers()
                if not self._auxiliary_allowed(epoch):
                    self._seal_auxiliaries()
                    return
                self._auxiliary_started_epoch = epoch
            self._start_microphone_telemetry(epoch)
            if self._run_started and self._auxiliary_allowed(epoch):
                await self._refresh_microphone_inventory()
                if not self._auxiliary_allowed(epoch):
                    self._seal_auxiliaries()
                    return
            if self._auxiliary_allowed(epoch):
                self._auxiliary_phase = 'unknown' if self._auxiliary_start_failed else 'running'
        except BaseException:
            if self._auxiliary_allowed(epoch):
                self._auxiliary_phase = 'failed'
                self.pause_admission()
                self._auxiliary_phase = 'failed'
            raise

    def _schedule_auxiliary_resume(self) -> asyncio.Task:
        epoch = self._auxiliary_epoch
        task = self._auxiliary_resume_task
        needs_first_start = (task is not None and task.done() and self._start_requested
                             and self._auxiliary_started_epoch != epoch)
        if task is None or self._auxiliary_resume_epoch != epoch or needs_first_start:
            self._auxiliary_resume_epoch = epoch
            self._auxiliary_resume_task = asyncio.create_task(
                self._start_auxiliaries(epoch), context=Context())
            self._auxiliary_resume_task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        return self._auxiliary_resume_task

    async def drain_resume(self) -> None:
        """Observe retained restart, including startup requested during a resume."""
        if self._closed or self._paused or self._auxiliary_resume_task is None:
            raise RuntimeError('capture_resume_unavailable')
        epoch = self._auxiliary_epoch
        while True:
            task = self._auxiliary_resume_task
            await asyncio.shield(task)
            if self._closed or self._paused or epoch != self._auxiliary_epoch:
                raise RuntimeError('capture_resume_superseded')
            if task is not self._auxiliary_resume_task:
                continue
            if self._start_requested and self._auxiliary_started_epoch != epoch:
                self._schedule_auxiliary_resume()
                continue
            if self._auxiliary_start_failed or self._auxiliary_phase != 'running':
                raise RuntimeError('capture_auxiliary_resume_unconfirmed')
            return

    async def _drain_auxiliaries(self) -> None:
        errors = []
        startup = self._auxiliary_resume_task
        if startup is not None:
            results = await asyncio.gather(asyncio.shield(startup), return_exceptions=True)
            errors.extend(result for result in results if isinstance(result, BaseException))
        native = tuple(self._xvf_native._tasks)
        health = tuple(self._audio_health_work._tasks)
        await asyncio.gather(self._xvf_native.drain_admitted(),
                             self._audio_health_work.drain_admitted())
        for task in native + health:
            if task.cancelled():
                errors.append(RuntimeError('capture_auxiliary_work_unconfirmed'))
            elif task.exception() is not None:
                errors.append(task.exception())
        # An admitted start can enter after the first memory-only stop.
        self._seal_auxiliaries()
        for ctrl in self._xvf_by_room.values():
            if not isinstance(ctrl, XvfController):
                errors.append(RuntimeError('unsupported_capture_pause_controller'))
                continue
            key = id(ctrl)
            if key not in self._xvf_pause_joins:
                self._xvf_pause_joins[key] = asyncio.create_task(
                    asyncio.to_thread(ctrl.wait_stopped), context=Context())
        drains = [self._stop_microphone_telemetry(), self._stop_direction_observers()]
        drains.extend(asyncio.shield(task) for task in self._xvf_pause_joins.values())
        results = await asyncio.gather(*drains, return_exceptions=True)
        errors.extend(result for result in results if isinstance(result, BaseException))
        if self._auxiliary_start_failed:
            errors.append(RuntimeError('capture_auxiliary_start_unconfirmed'))
        if errors:
            raise errors[0]

    def pause_admission(self) -> None:
        """Seal capture and auxiliary generations before any wait."""
        if not self._paused:
            self._auxiliary_epoch += 1
            self._auxiliary_phase = 'retiring'
        self._paused = True
        self._seal_auxiliaries()
        if self._privacy is not None:
            from src.memory.fact_ledger import MemoryOrigin
            self._privacy.scopes.suspend_origin(MemoryOrigin.VOICE)
        for room in self._capture_rooms.values():
            room.pause_admission()
        self.begin_pause_inputs()
        if isinstance(self._wake_clip_writer, WakeClipWriter):
            self._wake_clip_writer.pause_admission()

    async def drain_pause(self) -> None:
        """Retain one complete reusable pause, including real USB/IO exits."""
        if not self._paused or self._closed:
            raise RuntimeError('capture_pause_unavailable')
        retry = (self._pausing is not None and self._pausing.done()
                 and (self._pausing.cancelled() or self._pausing.exception() is not None))
        if self._pausing is None or retry:
            if retry:
                self._auxiliary_phase = 'retiring'
                self._xvf_pause_joins = {key: task for key, task in self._xvf_pause_joins.items()
                                         if not task.done() or not task.cancelled() and task.exception() is None}
                self._direction_join_tasks = {key: task for key, task in self._direction_join_tasks.items()
                                              if not task.done() or not task.cancelled() and task.exception() is None}
                if self._microphone_stop_task is not None and self._microphone_stop_task.done():
                    if self._microphone_stop_task.cancelled() or self._microphone_stop_task.exception() is not None:
                        self._microphone_stop_task = None
            async def finish():
                errors = []
                self._auxiliary_pause_task = asyncio.create_task(
                    self._drain_auxiliaries(), context=Context())
                try:
                    while True:
                        tasks = {task for task in self.snapshot_operations()
                                 if task not in (asyncio.current_task(), self._auxiliary_pause_task)
                                 and not task.done()}
                        if not tasks:
                            break
                        await asyncio.gather(*(asyncio.shield(task) for task in tasks),
                                             return_exceptions=True)
                    try:
                        await asyncio.shield(self.begin_pause_inputs())
                    except BaseException as error:
                        errors.append(error)
                    results = await asyncio.gather(
                        *(room.drain_pause() for room in self._capture_rooms.values()),
                        return_exceptions=True)
                    errors.extend(result for result in results if isinstance(result, BaseException))
                    if isinstance(self._wake_clip_writer, WakeClipWriter):
                        try:
                            await self._wake_clip_writer.drain_pause()
                        except BaseException as error:
                            errors.append(error)
                    elif self._wake_clip_writer is not None:
                        errors.append(RuntimeError('unsupported_capture_pause_writer'))
                    self.follow_up.end_conversation(reason='capture_paused')
                    self.command_turns.settle()
                except BaseException as error:
                    errors.append(error)
                results = await asyncio.gather(
                    asyncio.shield(self._auxiliary_pause_task), return_exceptions=True)
                errors.extend(result for result in results if isinstance(result, BaseException))
                if errors:
                    self._auxiliary_phase = 'failed'
                    raise errors[0]
                self._auxiliary_phase = 'paused'
            self._pausing = asyncio.create_task(finish(), context=Context())
            self._pausing.add_done_callback(
                lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._pausing)

    def resume_admission(self) -> None:
        """Schedule fresh capture only after every old pause receipt succeeds."""
        if self._closed or self._pausing is None or not self._pausing.done():
            raise RuntimeError('capture_pause_still_owned')
        self._pausing.result()
        if any(not task.done() for task in self.snapshot_operations()):
            raise RuntimeError('capture_pause_still_owned')
        if self._privacy is not None:
            from src.memory.fact_ledger import MemoryOrigin
            self._privacy.scopes.resume_origin(MemoryOrigin.VOICE)
        for room in self._capture_rooms.values():
            room.resume_admission()
        if isinstance(self._wake_clip_writer, WakeClipWriter):
            self._wake_clip_writer.resume_admission()
        self._auxiliary_epoch += 1
        self._xvf_pause_joins.clear()
        self._auxiliary_pause_task = None
        self._paused = False
        self._pausing = None
        self._auxiliary_phase = 'resuming'
        self.resume_inputs()
        self._schedule_auxiliary_resume()

    def resume_inputs(self) -> None:
        """Reallow driver opening after actual close; never reuse an old handle."""
        if self._closed:
            raise RuntimeError('capture_closed')
        if self._input_recovery_pause is not None:
            if not self._input_recovery_pause.done():
                raise RuntimeError('input_group_still_owned')
            self._input_recovery_pause.result()
        if self._recovery_task is not None:
            if (not self._recovery_task.done()
                    or self._recovery_task.cancelled() and self._input_recovery_pause is None):
                raise RuntimeError('input_group_still_owned')
        self._inputs.resume()
        self._input_recovery_pause = None
        self._recovery_ready = True
        if self._running:
            for rs in self.room_streams.values():
                self._schedule_reopen(rs)

    def begin_close(self) -> asyncio.Task:
        """Seal admission and retain all capture/XVF work until its real exit."""
        self._closed = True
        self._running = False
        self._auxiliary_epoch += 1
        self._auxiliary_phase = 'closing'
        self._seal_auxiliaries()
        if self._closing is None:
            for ctrl in self._xvf_by_room.values():
                if isinstance(ctrl, XvfController):
                    ctrl.request_stop()
            xvf_close = self._xvf_native.begin_close()
            self._direction_started = False
            with self._direction_lock:
                direction_rooms = tuple(self._direction_observers)
            for room_id in direction_rooms:
                self._invalidate_direction(room_id)
            input_pause = self.begin_pause_inputs()
            watchdog = self._watchdog_task
            if watchdog is not None:
                watchdog.cancel()
                self._watchdog_task = None
            health = self._audio_health_task
            health_close = self._audio_health_work.begin_close()
            controllers = set(self._capture_controllers.values()) | self._capture_tasks | self._capture_batches | self._early_tasks | self._dispatch_tasks
            for task in controllers:
                task.cancel()
            drains = [room.begin_close() for room in self._capture_rooms.values()]
            drains.append(self._inputs.begin_close())
            drains.extend((input_pause, health_close, self._capture_native.begin_close()))
            drains.extend(task for task in (self._capture_start_task, self._auxiliary_resume_task)
                          if task is not None)
            if isinstance(self._wake_clip_writer, WakeClipWriter):
                drains.append(self._wake_clip_writer.begin_close())
            self._microphone_inventory_ready.set()
            for room_id in self._microphone_captures:
                self._retire_microphone_capture(room_id)

            async def finish_capture():
                if watchdog is not None:
                    await asyncio.gather(watchdog, return_exceptions=True)
                if health is not None:
                    await asyncio.gather(asyncio.shield(health), return_exceptions=True)
                results = await asyncio.gather(*controllers, *drains, return_exceptions=True)
                failures = [r for r in results[len(controllers):] if isinstance(r, BaseException)]
                if self._pausing is not None:
                    # Terminal room drains repeat and acknowledge reset after a
                    # pause superseded by close; its controller may be rejected.
                    await asyncio.gather(asyncio.shield(self._pausing), return_exceptions=True)
                if self._wake_clip_writer is not None and not isinstance(self._wake_clip_writer, WakeClipWriter):
                    try:
                        await asyncio.to_thread(self._wake_clip_writer.stop)
                    except Exception:
                        pass
                if failures:
                    raise failures[0]
                await self._retire_wake_predictors()

            async def finish_xvf():
                await asyncio.shield(xvf_close)
                # An admitted start may enter after the first memory-only seal.
                # Drain admission, seal again, then join the actual pollers.
                for ctrl in self._xvf_by_room.values():
                    if isinstance(ctrl, XvfController):
                        ctrl.request_stop()
                errors = []
                for ctrl in self._xvf_by_room.values():
                    if isinstance(ctrl, XvfController):
                        try:
                            await asyncio.to_thread(ctrl.wait_stopped)
                        except Exception as error:
                            errors.append(error)
                    else:
                        try:
                            await asyncio.to_thread(ctrl.stop)
                        except Exception:
                            pass
                if errors:
                    raise errors[0]

            async def finish():
                results = await asyncio.gather(
                    finish_capture(), finish_xvf(),
                    self._stop_direction_observers(), self._stop_microphone_telemetry(),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing

    def last_command_dispatch_ts(self, room_id: str) -> float:
        """Timestamp monotónico (`time.monotonic()`) del último dispatch
        ACÚSTICO en `room_id`, o 0.0 si nunca hubo uno en esta room."""
        return self._last_command_dispatch_ts.get(room_id, 0.0)

    def attach_ambient(self, tap, transcriber=None) -> None:
        """Inyectar el ambient path post-init (tap obligatorio, transcriber
        opcional — habilita la señal shadow anti-TV en el wake)."""
        if isinstance(tap, MultiChannelTap):
            for room in self._capture_rooms.values():
                tap.register_ingress(room.callbacks)
        self._ambient_tap = tap
        self._ambient_transcriber = transcriber

    def on_command(self, callback: Callable[[CommandEvent], Awaitable[dict]]):
        """Register callback for when command audio is captured."""
        self._on_command_callback = callback

    def on_post_command(
        self, callback: Callable[[dict, CommandEvent], Awaitable[None]]
    ):
        """Register callback for post-processing after command result."""
        self._on_post_command_callback = callback


    async def start(self):
        """Initialize models and retain auxiliary startup through cancellation."""
        if self._closed:
            raise RuntimeError('capture_closed')
        self._start_requested = True
        if not self._auxiliary_allowed():
            await self._ensure_capture_models()
        else:
            # A prior resume may have finished before startup was requested.
            # Never replace pending work; after it returns, ensure this epoch
            # also completed its actual configured startup.
            while self._auxiliary_allowed():
                await asyncio.shield(self._schedule_auxiliary_resume())
                if self._closed:
                    raise RuntimeError('capture_closed')
                if self._auxiliary_started_epoch == self._auxiliary_epoch:
                    break
        if self._closed:
            raise RuntimeError('capture_closed')

    def _load_capture_models(self, rs: RoomStream) -> None:
        from src.ambient.segmenter import _SileroPredictor, make_silero_predictor
        if self._closed:
            return
        rs.wake_detector.load()
        if self._closed:
            return
        if rs.room_id not in self._wake_vad_predictors:
            try:
                predictor = make_silero_predictor()
                if issubclass(type(predictor), _SileroPredictor):
                    self._wake_vad_owned[id(predictor)] = predictor
                if self._closed:
                    predictor.reset()
                else:
                    self._wake_vad_predictors[rs.room_id] = predictor
            except Exception as error:
                logger.warning('Wake VAD unavailable: %s', type(error).__name__)
                if not self._closed:
                    self._wake_vad_predictors[rs.room_id] = False

    async def _retire_wake_predictors(self) -> None:
        """Retire owned models only after capture resets and native work drain."""
        owned = tuple(self._wake_vad_owned.items())

        async def close(predictor):
            await predictor.begin_close()

        results = await asyncio.gather(
            *(close(predictor) for _, predictor in owned), return_exceptions=True,
        )
        failures = []
        for (identity, _), result in zip(owned, results):
            if isinstance(result, BaseException):
                failures.append(result)
            else:
                self._wake_vad_owned.pop(identity)
        if failures:
            raise failures[0]
        self._wake_vad_predictors.clear()


    async def run(self):
        """Own capture while paused without admitting auxiliary work."""
        if self._closed:
            raise RuntimeError('capture_closed')
        self._running = True
        self._run_started = True
        self._recovery_ready = False
        self._loop = asyncio.get_running_loop()
        self._microphone_inventory_ready.clear()
        epoch = self._auxiliary_epoch
        self._start_microphone_telemetry(epoch)
        if self._watchdog_enabled:
            self._watchdog_task = asyncio.create_task(self._stream_watchdog(), context=Context())
        try:
            if self._auxiliary_allowed(epoch):
                try:
                    await asyncio.wait_for(self._microphone_inventory_ready.wait(), self._microphone_startup_budget_s)
                except TimeoutError:
                    if self._auxiliary_allowed(epoch):
                        self._publish_microphone_inventory(UsbInventory(
                            self._microphone_wall_clock(), self._microphone_monotonic(), False))
            if self._running:
                await self._run_capture()
        finally:
            self.begin_close()
            await self._stop_direction_observers()
            await self._stop_microphone_telemetry()
            if self._watchdog_task is not None:
                self._watchdog_task.cancel()
                self._watchdog_task = None
            pending = self._cancel_reopen_tasks()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.shield(self._inputs.begin_close())
            await asyncio.shield(self.begin_close())

    async def _run_capture(self):
        """
        Main loop — opens N InputStreams and polls for completed commands.

        Each room's audio callback runs on sounddevice's C thread.
        Command dispatch happens on the asyncio event loop via create_task.
        """
        epoch = self._recovery_epoch
        if self._watchdog_enabled and (self._watchdog_task is None or self._watchdog_task.done()):
            self._recovery_ready = False
            self._watchdog_task = asyncio.create_task(self._stream_watchdog(), context=Context())
        for room_id, rs in self.room_streams.items():
            if self._capture_rooms[room_id].resource_status()['closed']:
                continue
            self._capture_controllers[room_id] = asyncio.create_task(self._consume_room(rs))
        for room_id, rs in self.room_streams.items():
            if not self._input_epoch_current(epoch):
                break
            if self._capture_rooms[room_id].resource_status()['closed']:
                continue
            opened = await self._try_reopen_once(rs)
            if not self._input_epoch_current(epoch):
                break
            if not opened:
                self._schedule_reopen(rs)
        self._recovery_ready = True

        logger.info(
            f"MultiRoomAudioLoop ready "
            f"({len(self._streams)}/{len(self.room_streams)} streams)"
        )

        if self._watchdog_enabled:
            logger.info(
                f"[audio-watchdog] ACTIVO (timeout={self._watchdog_timeout_s}s, "
                f"check={self._watchdog_check_interval_s}s)"
            )

        try:
            while self._running:
                await asyncio.sleep(0.05)

        finally:
            self.begin_close()
            controllers = list(self._capture_controllers.values())
            for task in controllers:
                task.cancel()
            await asyncio.gather(*controllers, return_exceptions=True)
            self._capture_controllers.clear()
            if self._watchdog_task is not None:
                self._watchdog_task.cancel()
                self._watchdog_task = None
            pending = self._cancel_reopen_tasks()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def stop(self):
        """Observe the retained teardown without cancelling its actual owner."""
        closing = self.begin_close()
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(closing)
        except TimeoutError:
            raise WorkCloseTimeout('command_capture_still_running') from None


