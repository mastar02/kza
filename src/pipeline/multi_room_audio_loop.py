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
from threading import Event, Lock
from pathlib import Path
from typing import Callable, Awaitable, Optional
from uuid import uuid4

import numpy as np
import sounddevice as sd

from src.ambient.tap import MultiChannelTap
from src.audio.input_group import InputGroup
from src.audio.xvf_controller import XvfController
from src.pipeline.ambient_guard import AmbientGuard, GuardState, classify_outcome
from src.pipeline.command_event import CapturedTranscript, CommandEvent, TranscriptSource
from src.wakeword.wake_clip_writer import WakeClipWriter
from src.pipeline.capture_room import CaptureRoom
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
from src.conversation.follow_up_mode import FollowUpMode
from src.nlu.command_grammar import PartialCommand, parse_partial_command
from src.rooms.room_context import resolve_mic_usb_port

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1280

# Warn after this observation budget without cancelling native publication.
# _audio_health_work retains one real write until it exits, even after timeout.
AUDIO_HEALTH_WRITE_TIMEOUT_S = 1.0


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


class _CaptureHolder:
    """One callback-owned observation pointer; retired holders stay isolated."""

    def __init__(self, observation: CaptureObservation):
        self.observation = observation
        self.opening: CaptureOpening | None = None


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
    ):
        if (
            any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in (microphone_interval_s, microphone_stale_after_s))
            or microphone_stale_after_s < microphone_interval_s
        ):
            raise ValueError("invalid microphone telemetry interval/stale window")
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
        self._privacy = privacy
        self._closed = False
        self._closing = None
        self._close_timeout_s = close_timeout_s
        self._capture_controllers = {}
        self._capture_tasks = set()
        self._capture_batches = set()
        self._dispatch_tasks = set()
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
        # writes. Como es RAM, un re-enchufe del mic restaura el preset Seeed →
        # re-aplicar requiere reiniciar kza-voice (aceptado: evento raro).
        self._xvf_tuning = xvf_tuning or {}

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
            factory=lambda key: self._make_stream(self.room_streams[key]),
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
            audio_provenance=self._capture_provenance(rs, audio))

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
                self._track_capture(asyncio.create_task(run_private(
                    self._dispatch_command(event), self._dispatch_live)), self._dispatch_tasks)
        finally:
            event = None

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
                    f"score={wake_score:.2f}, rms={rms:.4f})"
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
        params = self._xvf_tuning.get("params") or {}
        # El tuning se aplica al DSP de CADA mic: son chips distintos y cada uno
        # arranca con el preset persistido de fábrica al re-enchufarse.
        for room_id, ctrl in self._xvf_by_room.items():
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

    def _make_stream(self, rs: "RoomStream"):
        """Construct only; its retained owner configures and starts the handle."""
        if rs.mic_usb_port:
            rs.device_index = resolve_mic_usb_port(rs.mic_usb_port)
        if rs.device_index is None or self._closed or not self._inputs.accepting:
            return None
        with self._direction_lock:
            self._direction_opening.add(rs.room_id)
        self._invalidate_direction(rs.room_id)
        try:
            dev_info = sd.query_devices(rs.device_index)
            if self._closed or not self._inputs.accepting:
                return None
            capture_channels = _resolve_capture_channels(
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
            stream = sd.InputStream(
                device=rs.device_index,
                samplerate=self.sample_rate,
                channels=capture_channels,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=callback,
            )
            return stream
        except sd.PortAudioError as e:
            self._retire_microphone_capture(rs.room_id)
            logger.error(
                f"Room {rs.room_id}: failed to open device {rs.device_index}: {e}"
            )
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
        rs.opened_ts = time.monotonic()
        rs.last_frame_ts = 0.0

    def _reinit_portaudio(self) -> None:
        """Force PortAudio to re-scan the device list (sync; fail-open).

        PortAudio snapshots devices at Pa_Initialize and never re-scans. After a
        USB re-enumeration the cached indices point to dead devices, so we must
        terminate+initialize to see the new ones. This is GLOBAL: it invalidates
        every open stream, which is why _recover_streams reopens all of them.
        """
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            logger.warning(f"[audio-watchdog] PortAudio reinit failed: {e}")

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

    def _schedule_recovery(self, trigger_room_ids: list) -> None:
        """Admit one finite controller; native effects remain owned by InputGroup."""
        if (not self._running or not self._recovery_ready or self._closed
                or not self._inputs.accepting
                or self._recovery_task is not None and not self._recovery_task.done()):
            return
        epoch = self._recovery_epoch
        rooms = tuple(room for room in self.room_streams if room in trigger_room_ids)
        if not rooms:
            return

        async def recover():
            if self._input_epoch_current(epoch) and self._running and self._recovery_ready:
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

    async def _recover_streams(self, trigger_room_ids: list) -> None:
        """Recover failed rooms without interrupting rooms still receiving audio.

        Reinitialize the shared PortAudio catalog only when no other stream is
        healthy. Native stop/open calls must not block the asyncio event loop.
        """
        epoch = self._recovery_epoch
        if not self._input_epoch_current(epoch):
            return
        now = time.monotonic()
        healthy_other = any(
            room_id not in trigger_room_ids and rs.last_frame_ts > 0
            and now - rs.last_frame_ts <= self._watchdog_timeout_s
            for room_id, rs in self.room_streams.items()
        )
        affected = trigger_room_ids if healthy_other else list(self.room_streams)
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
                await self._inputs.catalog_call(self._reinit_portaudio)
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
            now_mono: time.monotonic() del ciclo actual.
            now_wall: time.time() del ciclo actual.
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
            reader = self._microphone_reader or microphone_status.read_usb_microphones
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
        if self._auxiliary_allowed(epoch):
            await self._sync_direction_observers()

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
        return build_microphone_snapshot(
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
                    observer = DirectionObserver(
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
                direction = microphone_status.DirectionSnapshot(
                    state=microphone_status.DirectionState.DISABLED,
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
                        direction = replace(direction, state=microphone_status.DirectionState.UNKNOWN,
                                            reason='identity_unavailable')
                microphones.append(replace(mic, direction=direction))
        return replace(snapshot, microphones=tuple(microphones))

    def get_audio_health(self) -> dict:
        """Return room frame health without file I/O or audio content."""
        now_wall = time.time()
        self._update_audio_anchors(now_mono=time.monotonic(), now_wall=now_wall)
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
        now = time.monotonic()
        now_wall = time.time()
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
            states = [
                (room_id, rs.last_frame_ts, rs.opened_ts)
                for room_id, rs in self.room_streams.items()
                if not self._is_awaiting_reopen(room_id)
            ]
            stale = detect_stale_streams(states, time.monotonic(),
                self._watchdog_timeout_s, self._watchdog_first_frame_grace_s)
            if stale:
                self._schedule_recovery(stale)


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
                    rs.last_frame_ts = time.monotonic()

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
            if recurrent_conflict or _pcm_context([*retained, part])[2]:
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
                    wake_vad = compute_wake_vad(audio_chunk, predict)
                    room.require_live()
                    if self._should_accept_wakeword(
                        rs.room_id, rms, time.time(),
                        wake_score=detection[1], wake_vad=wake_vad,
                    ):
                        rs.capture_generation += 1
                        rs.transcript = None
                        rs.capture_complete = False
                        rs.listening = True
                        rs.wake_score = detection[1]
                        rs.command_start_time = time.time()
                        # Sembrar con el pre-roll (en vez de []) para no perder el
                        # verbo dicho durante la latencia. Backdate command_start_time
                        # para que el endpointing cuente la duración real del buffer.
                        if rs.preroll:
                            rs.audio_buffer = list(np.concatenate(list(rs.preroll)))
                            rs.audio_metadata = list(rs.preroll_metadata)
                            rs.inline_provenance = None
                            rs.command_start_time = time.time() - len(rs.audio_buffer) / self.sample_rate
                            rs.preroll.clear()
                            rs.preroll_metadata.clear()
                        else:
                            rs.audio_buffer = []
                            rs.audio_metadata = []
                            rs.inline_provenance = None
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
                        logger.info(f"Wake word in {rs.room_id} ({detection[0]}: {detection[1]:.2f})")

                        # Peek wake text first to decide if it was truncated.
                        # Si el wake terminó con preposición/artículo/elipsis,
                        # NO saltamos la captura post-wake — el comando real
                        # probablemente sigue. (Fix 1A — visto 2026-05-02 06:36:51:
                        # 'Nexa prendé la luz del...' → light.bano por adivinanza
                        # del LLM con texto cortado.)
                        peek_text_fn = getattr(rs.wake_detector, "peek_pending_text", None)
                        wake_text_peek = peek_text_fn() if callable(peek_text_fn) else None
                        wake_was_truncated = _text_likely_truncated(wake_text_peek or "")

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
                                    rs.command_start_time = time.time() - self.min_speech_ms / 1000 - 0.1
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
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if rms > 0.02 and rs.echo_suppressor.is_human_voice(audio_chunk):
                        rs.listening = True
                        rs.capture_generation += 1
                        rs.command_start_time = time.time()
                        rs.audio_buffer = []
                        rs.audio_metadata = []
                        rs.inline_provenance = None
            else:
                rs.audio_buffer.extend(audio_chunk)
                rs.audio_metadata.append(part)
                rs.inline_provenance = None

        return process

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
                    partial = parse_partial_command(text)
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
        self._last_command_dispatch_ts[event.room_id] = time.monotonic()
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
            self._last_command_dispatch_ts[event.room_id] = time.monotonic()

            outcome = classify_outcome(result) if isinstance(result, dict) else None

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
            if outcome == "empty":
                self._last_command_dispatch_ts.pop(event.room_id, None)
                logger.info(
                    f"[dedup] captura vacía en {event.room_id} — "
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
                # abre al wake; tras un comando ACEPTADO se abre acá — el
                # usuario quedó confirmado y puede encadenar ('apagá'→'prendé')
                # sin re-pasar el wake estricto. Las compuertas de texto siguen.
                if outcome == "accepted" and self._guard.follow_up_allowed(
                    event.room_id
                ):
                    try:
                        self.follow_up.start_conversation()
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
        rs.command_start_time = time.time()
        try:
            self.follow_up.start_conversation()
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
