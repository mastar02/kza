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
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

import numpy as np
import sounddevice as sd

from src.pipeline.ambient_guard import AmbientGuard, GuardState, classify_outcome
from src.pipeline.command_event import CommandEvent
from src.wakeword.detector import WakeWordDetector
from src.wakeword.whisper_wake import _text_likely_truncated
from src.audio.echo_suppressor import EchoSuppressor
from src.conversation.follow_up_mode import FollowUpMode
from src.nlu.command_grammar import PartialCommand, parse_partial_command
from src.rooms.room_context import resolve_mic_usb_port

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1280


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
    hallucinate. Channel 0 (indata[:, 0]) is always consumed downstream.

    Args:
        max_input_channels: Value from sd.query_devices(index)['max_input_channels'].

    Returns:
        Number of channels to pass to sd.InputStream.
    """
    return max_input_channels if max_input_channels and max_input_channels >= 1 else 1


def detect_stale_streams(
    states: list[tuple[str, float]], now: float, timeout_s: float
) -> list[str]:
    """Return room_ids whose audio stream stopped delivering frames.

    A stream is stale when it has produced at least one frame (last_frame_ts > 0)
    and more than `timeout_s` seconds elapsed since the last one. Streams that
    never opened (last_frame_ts == 0.0) are ignored — there is nothing to recover
    until `run()` opens them.

    Args:
        states: list of (room_id, last_frame_ts) with monotonic timestamps.
        now: current monotonic time.
        timeout_s: seconds without frames before a stream is considered dead.
    """
    return [
        room_id
        for room_id, last_frame_ts in states
        if last_frame_ts > 0.0 and (now - last_frame_ts) > timeout_s
    ]


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
    # Texto completo transcripto por el wake detector (si está disponible).
    # Se usa como pretranscribed_text del CommandEvent para evitar 2do Whisper.
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
        xvf_controller=None,
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
        self._xvf = xvf_controller
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
        self._wake_vad_predict = None  # lazy: predictor Silero para wake_vad

        self._watchdog_enabled = stream_watchdog_enabled
        self._watchdog_timeout_s = stream_watchdog_no_frames_timeout_s
        self._watchdog_check_interval_s = stream_watchdog_check_interval_s
        self._watchdog_backoff_min_s = stream_watchdog_reopen_backoff_min_s
        self._watchdog_backoff_max_s = stream_watchdog_reopen_backoff_max_s
        self._watchdog_task = None

        self._running = False
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

    def attach_response_handler(self, response_handler) -> None:
        """Inyectar ResponseHandler post-init (útil por orden de DI en main.py)."""
        self._response_handler = response_handler

    def attach_ambient(self, tap, transcriber=None) -> None:
        """Inyectar el ambient path post-init (tap obligatorio, transcriber
        opcional — habilita la señal shadow anti-TV en el wake)."""
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
        """Initialize wake word detectors for all rooms."""
        for room_id, rs in self.room_streams.items():
            rs.wake_detector.load()
            logger.info(
                f"Room {room_id}: wake word loaded "
                f"(device={rs.device_index}, models={rs.wake_detector.get_active_models()})"
            )
        # Pre-cargar el predictor Silero para wake_vad FUERA del audio callback
        # (torch.hub.load hace I/O + init de torch; en el callback C de
        # sounddevice congelaría el stream en la primera detección). Dedicado:
        # NO compartir con los predictores del ambient segmenter (Silero es
        # stateful — GRU — y el segmenter resetea su estado por utterance).
        if self._wake_vad_predict is None:
            from src.ambient.segmenter import make_silero_predictor
            try:
                self._wake_vad_predict = make_silero_predictor()
                logger.info("wake_vad: predictor Silero pre-cargado")
            except Exception as e:
                logger.warning(f"wake_vad: Silero no disponible ({e}) — umbral fijo")
                self._wake_vad_predict = False  # no reintentar
        # Tuning del DSP ANTES de arrancar el poller (review Fase 1): así los
        # writes corren con USB single-threaded (sin transfers concurrentes
        # sobre el mismo handle), y en to_thread para no bloquear el event
        # loop (cada ctrl_transfer con retries puede tardar segundos).
        await asyncio.to_thread(self._apply_xvf_tuning)
        # Pre-gate SPENERGY: poller solo si el gate está habilitado (fail-open).
        if self._xvf is not None and self.spenergy_gate_enabled:
            try:
                if self._xvf.start():
                    logger.info(
                        f"Pre-gate SPENERGY activo (umbral {self.spenergy_threshold:.0f})"
                    )
                else:
                    logger.warning("Pre-gate SPENERGY no pudo iniciar — gate OFF (fail-open)")
            except Exception as e:
                logger.warning(f"Pre-gate SPENERGY error al iniciar — gate OFF: {e}")

    def _apply_xvf_tuning(self) -> None:
        """Aplica el tuning configurado al DSP (EN RAM). Fail-open por param.

        Un nombre/valor inválido en el yaml NO tira el servicio: se loguea y
        se continúa con el resto. Loguea valor previo → nuevo para auditoría.
        """
        if not self._xvf_tuning.get("apply_on_start", False):
            return
        if self._xvf is None:
            logger.warning(
                "[XVF-tuning] xvf_tuning.apply_on_start=true pero no hay "
                "XvfController — tuning NO aplicado (¿mic XVF3800 presente?)"
            )
            return
        params = self._xvf_tuning.get("params") or {}
        for name, values in params.items():
            try:
                before = self._xvf.read_param(name)
            except ValueError:
                before = None
            try:
                ok = self._xvf.write_param(name, values)
            except ValueError as e:  # typo en el yaml: log y seguir
                logger.error(f"[XVF-tuning] parámetro inválido en config: {e}")
                continue
            except Exception as e:  # cualquier otra cosa: fail-open
                logger.warning(f"[XVF-tuning] {name} no aplicado: {e}")
                continue
            if ok:
                logger.info(f"[XVF-tuning] {name}: {before} → {values} (RAM)")
            else:
                logger.warning(f"[XVF-tuning] {name} no aplicado (write fail-open)")

    def _open_stream(self, rs: "RoomStream"):
        """Open and start an InputStream for one room. None on PortAudioError."""
        callback = self._make_audio_callback(rs)
        try:
            dev_info = sd.query_devices(rs.device_index)
            capture_channels = _resolve_capture_channels(
                int(dev_info.get("max_input_channels", 0))
            )
            stream = sd.InputStream(
                device=rs.device_index,
                samplerate=self.sample_rate,
                channels=capture_channels,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=callback,
            )
            stream.start()
            logger.info(
                f"Room {rs.room_id}: audio stream started "
                f"(device={rs.device_index}, channels={capture_channels})"
            )
            return stream
        except sd.PortAudioError as e:
            logger.error(
                f"Room {rs.room_id}: failed to open device {rs.device_index}: {e}"
            )
            return None

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

    async def _reopen_room(self, rs: "RoomStream") -> None:
        """Re-resolve the device by USB port and reopen its stream, with backoff.

        Waits indefinitely (while self._running) for the device to reappear in
        sysfs — the service stays alive; only this mic waits. Never raises.
        """
        backoff = self._watchdog_backoff_min_s
        while self._running:
            new_index = rs.device_index
            if rs.mic_usb_port:
                resolved = await asyncio.to_thread(resolve_mic_usb_port, rs.mic_usb_port)
                if resolved is None:
                    logger.warning(
                        f"[audio-watchdog] {rs.room_id}: device {rs.mic_usb_port} "
                        f"absent, retry in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._watchdog_backoff_max_s)
                    continue
                new_index = resolved
            rs.device_index = new_index
            stream = await asyncio.to_thread(self._open_stream, rs)
            if stream is not None:
                self._streams[rs.room_id] = stream
                rs.last_frame_ts = time.monotonic()
                logger.info(
                    f"[audio-watchdog] {rs.room_id}: recovered (device={new_index})"
                )
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._watchdog_backoff_max_s)

    async def _recover_streams(self, trigger_room_ids: list) -> None:
        """Close all streams, reinit PortAudio, reopen all rooms.

        sd._terminate() invalidates every stream, so recovery is all-or-nothing
        even if only one room went stale.
        """
        logger.error(
            f"[audio-watchdog] streams {trigger_room_ids} stopped delivering "
            f"audio → recovering all streams"
        )
        for room_id, stream in list(self._streams.items()):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()
        await asyncio.to_thread(self._reinit_portaudio)
        for room_id, rs in self.room_streams.items():
            await self._reopen_room(rs)

    async def _stream_watchdog(self) -> None:
        """Periodically detect mics that stopped delivering frames and recover.

        Reuses detect_stale_streams (pure) for the decision. Runs only while
        self._running; recovery is awaited so we never overlap two recoveries.
        """
        while self._running:
            await asyncio.sleep(self._watchdog_check_interval_s)
            if not self._running:
                break
            now = time.monotonic()
            states = [
                (room_id, rs.last_frame_ts)
                for room_id, rs in self.room_streams.items()
            ]
            stale = detect_stale_streams(states, now, self._watchdog_timeout_s)
            if stale:
                await self._recover_streams(stale)

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

        self._streams = {}
        for room_id, rs in self.room_streams.items():
            stream = self._open_stream(rs)
            if stream is not None:
                self._streams[room_id] = stream
                rs.last_frame_ts = time.monotonic()

        logger.info(
            f"MultiRoomAudioLoop ready "
            f"({len(self._streams)}/{len(self.room_streams)} streams)"
        )

        if self._watchdog_enabled:
            self._watchdog_task = asyncio.create_task(self._stream_watchdog())
            logger.info(
                f"[audio-watchdog] ACTIVO (timeout={self._watchdog_timeout_s}s, "
                f"check={self._watchdog_check_interval_s}s)"
            )

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
                        # Pre-gate SPENERGY también acá (QW-1 2026-06-04): sin
                        # este chequeo, una alucinación con forma de comando
                        # (grammar full sobre secador/ruido) se despachaba
                        # saltándose el VAD por hardware — early_dispatch es el
                        # path más usado en prod.
                        if not self._passes_spenergy_gate(rs):
                            self._reset_listening(rs)
                            continue
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
                            wake_text=rs.wake_text,
                            ambient_strict=(
                                self._guard is not None
                                and self._guard.state_for(room_id) is GuardState.STRICT
                            ),
                            wake_score=rs.wake_score,
                        )
                        asyncio.create_task(self._dispatch_command(event))
                        self._reset_listening(rs)
                        continue

                    # 3. Fallback: VAD silencio normal
                    is_complete, audio_data = self._check_vad_completion(rs)
                    if is_complete and audio_data is not None:
                        # Pre-gate SPENERGY: descartar capturas sin voz real
                        # (secador/silencio) antes de gastar Whisper.
                        if not self._passes_spenergy_gate(rs):
                            self._reset_listening(rs)
                            continue
                        event = CommandEvent(
                            audio=audio_data,
                            room_id=room_id,
                            mic_device_index=rs.device_index,
                            wake_text=rs.wake_text,
                            ambient_strict=(
                                self._guard is not None
                                and self._guard.state_for(room_id) is GuardState.STRICT
                            ),
                            wake_score=rs.wake_score,
                        )
                        asyncio.create_task(self._dispatch_command(event))
                        self._reset_listening(rs)
        finally:
            if self._watchdog_task is not None:
                self._watchdog_task.cancel()
                self._watchdog_task = None
            for stream in self._streams.values():
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    async def stop(self):
        """Stop all audio streams."""
        self._running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None
        if self._xvf is not None:
            try:
                # XvfController.stop() hace thread.join(timeout=1.0) — síncrono.
                # to_thread evita bloquear el event loop durante el shutdown
                # (regla del proyecto: nunca llamadas sync en el loop).
                await asyncio.to_thread(self._xvf.stop)
            except Exception:
                pass
        if self._wake_clip_writer is not None:
            try:
                # stop() drena la cola y hace thread.join — síncrono.
                await asyncio.to_thread(self._wake_clip_writer.stop)
            except Exception:
                pass

    def _make_audio_callback(self, rs: RoomStream):
        """Create a sounddevice callback closure for one room."""

        def audio_callback(indata, frames, time_info, status):
            # Watchdog heartbeat: marca que el stream entregó un frame. Primera
            # línea, O(1), nunca lanza — si esto deja de actualizarse, el mic
            # murió (re-enumeración USB) y _stream_watchdog dispara recovery.
            rs.last_frame_ts = time.monotonic()

            # Tee al ambient path (spec 2026-06-06): SIEMPRE primero — el
            # ambient quiere todo el audio, incluso lo que el barge-in o el
            # echo suppressor descartan para el command path. O(1). El
            # try/except hace el fail-open EXPLÍCITO: una excepción acá
            # abortaría el stream de audio del command path (thread C).
            if self._ambient_tap is not None:
                try:
                    tts_now = (
                        self._response_handler is not None
                        and self._response_handler.is_speaking
                    )
                    self._ambient_tap.push(
                        rs.room_id, indata.copy(), tts_active=tts_now
                    )
                except Exception:
                    pass  # perder un chunk ambiental es aceptable; el stream no

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
                # Pre-roll: acumular el audio reciente para sembrar el buffer al
                # disparar el wake (recupera el comando dicho durante la latencia
                # de detección de openwakeword). 0 chunks = desactivado.
                if self._preroll_maxchunks > 0:
                    if rs.preroll is None:
                        rs.preroll = deque(maxlen=self._preroll_maxchunks)
                    rs.preroll.append(audio_chunk)
                detection = rs.wake_detector.detect(audio_chunk)
                if detection:
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    predict = self._wake_vad_predict or None
                    wake_vad = compute_wake_vad(audio_chunk, predict)
                    if self._should_accept_wakeword(
                        rs.room_id, rms, time.time(),
                        wake_score=detection[1], wake_vad=wake_vad,
                    ):
                        rs.listening = True
                        rs.wake_score = detection[1]
                        rs.command_start_time = time.time()
                        # Sembrar con el pre-roll (en vez de []) para no perder el
                        # verbo dicho durante la latencia. Backdate command_start_time
                        # para que el endpointing cuente la duración real del buffer.
                        if rs.preroll:
                            rs.audio_buffer = list(np.concatenate(list(rs.preroll)))
                            rs.command_start_time = time.time() - len(rs.audio_buffer) / self.sample_rate
                            rs.preroll.clear()
                        else:
                            rs.audio_buffer = []
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
                        if callable(pop_fn):
                            inline_audio = pop_fn()
                            if inline_audio is not None and len(inline_audio) > 0:
                                rs.audio_buffer = list(inline_audio)
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
                        # Guardar texto del wake como `pretranscribed_text` SOLO
                        # si no estaba truncado. Si estaba truncado, dejamos que
                        # el STT post-wake re-transcriba el audio completo
                        # (inline + continuación) y obtenga el comando entero.
                        pop_text_fn = getattr(rs.wake_detector, "pop_pending_text", None)
                        if callable(pop_text_fn):
                            popped = pop_text_fn()
                            rs.wake_text = popped if not wake_was_truncated else None
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

    def _passes_spenergy_gate(self, rs: RoomStream) -> bool:
        """True si hubo voz (SPENERGY ≥ umbral) durante la captura, o si el gate
        no aplica. FAIL-OPEN: sin controller o sin muestras → True (procesa).

        Bloquea solo cuando el pico de SPENERGY durante [command_start_time, now]
        quedó por debajo del umbral = secador/silencio → Whisper alucinaría.
        """
        if self._xvf is None or not self.spenergy_gate_enabled:
            return True
        try:
            peak = self._xvf.peak_since(rs.command_start_time)
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
        rs.wake_text = None
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

            # AmbientGuard: el resultado de la captura alimenta la escalera
            # (noise/empty/timeout escalan; accepted/other_fail/hallucination
            # no — alucinaciones de silencio de Whisper no son ambiente
            # hostil y no deben bloquear el quiet timer, 2026-07-05).
            if self._guard is not None and isinstance(result, dict):
                outcome = classify_outcome(result)
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
