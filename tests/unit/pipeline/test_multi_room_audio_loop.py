"""
Tests for MultiRoomAudioLoop — parallel audio capture from multiple rooms.

Tests ensure that:
1. RoomStream holds per-room state correctly
2. MultiRoomAudioLoop initializes with room streams
3. Deduplication keeps strongest RMS within window
4. Deduplication allows independent commands after window
5. on_command registers callback
6. _dispatch_command calls the registered callback with CommandEvent
"""

import sys
import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import numpy as np
import pytest

import src.pipeline.multi_room_audio_loop as mra_mod
from src.pipeline.multi_room_audio_loop import (
    MultiRoomAudioLoop,
    RoomStream,
    CHUNK_SIZE,
    _resolve_capture_channels,
)
from src.pipeline.command_event import CommandEvent
import src.monitoring.audio_health as audio_health_mod
from src.monitoring.audio_health import evaluate_health


# ============================================================
# Helpers
# ============================================================

def _make_wake_detector():
    """Create a mock WakeWordDetector."""
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(return_value=None)
    m.get_active_models = MagicMock(return_value=["hey_jarvis"])
    return m


def _make_echo_suppressor():
    """Create a mock EchoSuppressor."""
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=True)
    m.config = MagicMock()
    m.config.post_speech_buffer_ms = 400
    return m


def _make_follow_up():
    """Create a mock FollowUpMode."""
    m = MagicMock()
    m.is_active = False
    m.follow_up_window = 8.0
    m.start_conversation = MagicMock()
    return m


def _make_room_stream(room_id: str, device_index: int = 0) -> RoomStream:
    """Create a RoomStream with mock dependencies."""
    return RoomStream(
        room_id=room_id,
        device_index=device_index,
        wake_detector=_make_wake_detector(),
        echo_suppressor=_make_echo_suppressor(),
    )


def _make_multi_room_loop(rooms=None, **kwargs) -> MultiRoomAudioLoop:
    """Create a MultiRoomAudioLoop with mock dependencies."""
    if rooms is None:
        rooms = {
            "cocina": _make_room_stream("cocina", device_index=2),
            "living": _make_room_stream("living", device_index=3),
        }
    return MultiRoomAudioLoop(
        room_streams=rooms,
        follow_up=_make_follow_up(),
        **kwargs,
    )


# ============================================================
# Tests
# ============================================================

class TestRoomStream:
    """Test RoomStream dataclass."""

    def test_room_stream_creation(self):
        """RoomStream holds per-room state."""
        wake = _make_wake_detector()
        echo = _make_echo_suppressor()

        rs = RoomStream(
            room_id="cocina",
            device_index=2,
            wake_detector=wake,
            echo_suppressor=echo,
        )

        assert rs.room_id == "cocina"
        assert rs.device_index == 2
        assert rs.wake_detector is wake
        assert rs.echo_suppressor is echo
        assert rs.listening is False
        assert rs.audio_buffer == []
        assert rs.command_start_time == 0.0

    def test_room_stream_mutable_state(self):
        """RoomStream state can be modified during capture."""
        rs = _make_room_stream("living", device_index=5)

        rs.listening = True
        rs.command_start_time = time.time()
        rs.audio_buffer = [0.1, 0.2, 0.3]

        assert rs.listening is True
        assert rs.command_start_time > 0
        assert len(rs.audio_buffer) == 3


class TestMultiRoomAudioLoopInit:
    """Test MultiRoomAudioLoop initialization."""

    def test_multi_room_audio_loop_init(self):
        """MultiRoomAudioLoop initializes with room streams."""
        rooms = {
            "cocina": _make_room_stream("cocina", 2),
            "living": _make_room_stream("living", 3),
        }
        follow_up = _make_follow_up()

        loop = MultiRoomAudioLoop(
            room_streams=rooms,
            follow_up=follow_up,
            sample_rate=16000,
            dedup_window_ms=500,
        )

        assert loop.room_streams is rooms
        assert loop.follow_up is follow_up
        assert loop.sample_rate == 16000
        assert loop.dedup_window_ms == 500
        assert loop._running is False
        assert loop._on_command_callback is None
        assert loop._on_post_command_callback is None
        assert len(loop.room_streams) == 2

    def test_multi_room_audio_loop_defaults(self):
        """MultiRoomAudioLoop uses sensible defaults."""
        loop = _make_multi_room_loop()

        assert loop.sample_rate == 16000
        assert loop.command_duration == 2.0
        assert loop.silence_threshold == 0.015
        assert loop.silence_duration_ms == 300
        assert loop.min_speech_ms == 300
        assert loop.dedup_window_ms == 500


class TestDeduplication:
    """Test wake word deduplication between rooms."""

    def test_dedup_same_wakeword_within_window(self):
        """Within 200ms: keep strongest RMS, discard weaker."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # First room detects wake word
        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        # Second room detects within window but weaker RMS — echo
        result2 = loop._should_accept_wakeword("living", rms=0.02, timestamp=now + 0.05)
        assert result2 is False

        # Verify cocina is still the accepted room
        assert loop._last_wakeword_room == "cocina"

    def test_dedup_stronger_replaces_weaker(self):
        """Within window: stronger RMS from second room replaces first."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # Weaker room detects first
        result1 = loop._should_accept_wakeword("cocina", rms=0.02, timestamp=now)
        assert result1 is True

        # Stronger room detects within window — replaces
        result2 = loop._should_accept_wakeword("living", rms=0.08, timestamp=now + 0.05)
        assert result2 is True
        assert loop._last_wakeword_room == "living"
        assert loop._last_wakeword_rms == 0.08

    def test_dedup_allows_after_window(self):
        """After 200ms: both accepted as independent commands."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        # First room
        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        # Second room well after window
        result2 = loop._should_accept_wakeword("living", rms=0.03, timestamp=now + 0.5)
        assert result2 is True

        # Both are independent, living is now the latest
        assert loop._last_wakeword_room == "living"

    def test_dedup_same_room_within_window(self):
        """Same room within window is always accepted."""
        loop = _make_multi_room_loop(dedup_window_ms=200)

        now = time.time()

        result1 = loop._should_accept_wakeword("cocina", rms=0.05, timestamp=now)
        assert result1 is True

        result2 = loop._should_accept_wakeword("cocina", rms=0.04, timestamp=now + 0.05)
        assert result2 is True


class TestMinWakeRmsGate:
    """Pre-gate de RMS post-wake (2026-06-02): rechaza activaciones de muy baja
    energía (near-silence) antes de transcribir. Default 0.0 = desactivado (no
    regresión); se calibra en repro porque el AGC ×64 infla el piso de ruido."""

    def test_rms_below_min_wake_rms_rejected(self):
        loop = _make_multi_room_loop(min_wake_rms=0.02)
        assert loop._should_accept_wakeword("cocina", rms=0.005, timestamp=1.0) is False

    def test_rms_at_or_above_min_wake_rms_accepted(self):
        loop = _make_multi_room_loop(min_wake_rms=0.02)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=2.0) is True

    def test_min_wake_rms_zero_disables_gate(self):
        # Default 0.0 → gate off → comportamiento idéntico al baseline (dedup).
        loop = _make_multi_room_loop(min_wake_rms=0.0)
        assert loop._should_accept_wakeword("cocina", rms=0.0001, timestamp=3.0) is True


def _detector_seq(detect_returns):
    """Wake detector mock SIN inline audio (simula el path openwakeword)."""
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(side_effect=list(detect_returns))
    m.get_active_models = MagicMock(return_value=[])
    # openwakeword no tiene estos métodos (son del WhisperWake) -> None para que
    # getattr(...) los saltee y se ejecute el path acústico.
    m.peek_pending_text = None
    m.pop_pending_command_audio = None
    m.pop_pending_text = None
    return m


class TestWakePreroll:
    """Pre-roll (2026-06-02): al disparar el wake, sembrar el buffer con el audio
    previo para no perder el comando dicho durante la latencia de openwakeword
    ('Nexa apagá la luz' -> 'apagá' se decía mientras el detector aún procesaba).
    """

    def _loop_with_room(self, detector, **kwargs):
        rs = RoomStream(
            room_id="escritorio", device_index=0,
            wake_detector=detector, echo_suppressor=_make_echo_suppressor(),
        )
        loop = _make_multi_room_loop(rooms={"escritorio": rs}, **kwargs)
        return loop, rs

    def test_preroll_seeds_command_buffer_on_wake(self):
        det = _detector_seq([None, None, None, ("nexa", 0.8)])
        loop, rs = self._loop_with_room(det, wake_preroll_s=0.24)  # ~3 chunks @ 80ms
        cb = loop._make_audio_callback(rs)
        for i in range(3):
            cb(np.full((CHUNK_SIZE, 2), 0.01 * (i + 1), dtype=np.float32), CHUNK_SIZE, None, None)
            assert rs.listening is False
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True
        # el buffer arranca con el pre-roll (≥3 chunks) en vez de vacío
        assert len(rs.audio_buffer) >= 3 * CHUNK_SIZE

    def test_preroll_off_keeps_empty_buffer(self):
        det = _detector_seq([None, None, ("nexa", 0.8)])
        loop, rs = self._loop_with_room(det, wake_preroll_s=0.0)  # default = off
        cb = loop._make_audio_callback(rs)
        for _ in range(2):
            cb(np.full((CHUNK_SIZE, 2), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True
        assert len(rs.audio_buffer) == 0  # sin pre-roll = comportamiento actual


class TestWakeClipCapture:
    """Captura de clips de wake (2026-06-12): cada wake ACEPTADO persiste su
    audio (preroll) vía WakeClipWriter para el dataset de re-entrenamiento
    (hard negatives de TV + positivos far-field). El submit no debe bloquear
    el audio callback — el writer es un colaborador inyectado y mockeable."""

    def _loop_with_room(self, detector, **kwargs):
        rs = RoomStream(
            room_id="escritorio", device_index=0,
            wake_detector=detector, echo_suppressor=_make_echo_suppressor(),
        )
        loop = _make_multi_room_loop(rooms={"escritorio": rs}, **kwargs)
        return loop, rs

    def test_accepted_wake_submits_clip(self):
        det = _detector_seq([None, None, None, ("nexa", 0.8)])
        writer = MagicMock()
        loop, rs = self._loop_with_room(
            det, wake_preroll_s=0.24, wake_clip_writer=writer,
        )
        cb = loop._make_audio_callback(rs)
        for i in range(3):
            cb(np.full((CHUNK_SIZE, 2), 0.01 * (i + 1), dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True
        writer.submit.assert_called_once()
        room_id, score, audio = writer.submit.call_args.args
        assert room_id == "escritorio"
        assert score == 0.8
        assert len(audio) >= 3 * CHUNK_SIZE  # el preroll sembrado

    def test_no_writer_is_fine(self):
        det = _detector_seq([None, ("nexa", 0.8)])
        loop, rs = self._loop_with_room(det, wake_preroll_s=0.24)
        cb = loop._make_audio_callback(rs)
        cb(np.full((CHUNK_SIZE, 2), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True  # sin writer no rompe nada

    def test_rejected_wake_submits_as_rejected(self):
        # 2026-06-14: el wake RECHAZADO por el guard también se persiste (desde
        # el preroll) con accepted=False → subcarpeta rejected/. Recupera los
        # 0.40-0.45 que STRICT mata (positivos far-field reales) + hard-negatives
        # de TV. NO entra en captura (rs.listening sigue False).
        # min_wake_rms imposible → _should_accept_wakeword rechaza.
        det = _detector_seq([None, ("nexa", 0.8)])
        writer = MagicMock()
        loop, rs = self._loop_with_room(
            det, wake_preroll_s=0.24, wake_clip_writer=writer, min_wake_rms=9.9,
        )
        cb = loop._make_audio_callback(rs)
        cb(np.full((CHUNK_SIZE, 2), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is False  # rechazado: no entra en captura
        writer.submit.assert_called_once()
        assert writer.submit.call_args.kwargs.get("accepted") is False
        assert writer.submit.call_args.args[1] == 0.8  # score

    def test_writer_exception_does_not_break_capture(self):
        det = _detector_seq([None, ("nexa", 0.8)])
        writer = MagicMock()
        writer.submit.side_effect = RuntimeError("disk on fire")
        loop, rs = self._loop_with_room(
            det, wake_preroll_s=0.24, wake_clip_writer=writer,
        )
        cb = loop._make_audio_callback(rs)
        cb(np.full((CHUNK_SIZE, 2), 0.01, dtype=np.float32), CHUNK_SIZE, None, None)
        cb(np.full((CHUNK_SIZE, 2), 0.05, dtype=np.float32), CHUNK_SIZE, None, None)
        assert rs.listening is True  # fail-open: la captura del comando sigue


class _FakeXvf:
    """XvfController falso: peak_since devuelve un valor fijo (o None)."""
    def __init__(self, peak):
        self._peak = peak
        self.started = False
    def start(self):
        self.started = True
        return True
    def stop(self):
        pass
    def peak_since(self, since_ts):
        return self._peak


class TestSpenergyGate:
    """Pre-gate SPENERGY (2026-06-02): no transcribir si el pico de SPENERGY
    durante la captura < umbral (secador/silencio → alucinación de Whisper).
    Fail-open: sin controller o sin datos → procesa."""

    def _rs(self):
        rs = _make_room_stream("escritorio")
        rs.command_start_time = 100.0
        return rs

    def test_no_controller_passes(self):
        loop = _make_multi_room_loop()  # xvf_controller None por defecto
        assert loop._passes_spenergy_gate(self._rs()) is True

    def test_peak_none_fail_open_passes(self):
        loop = _make_multi_room_loop(
            xvf_controllers={"escritorio": _FakeXvf(None)}, spenergy_threshold=100.0
        )
        assert loop._passes_spenergy_gate(self._rs()) is True

    def test_low_peak_blocks(self):
        # secador/silencio = 0 < 100 → descarta
        loop = _make_multi_room_loop(
            xvf_controllers={"escritorio": _FakeXvf(0.0)}, spenergy_threshold=100.0
        )
        assert loop._passes_spenergy_gate(self._rs()) is False

    def test_voice_peak_passes(self):
        # voz medida ~335k ≥ 100 → procesa
        loop = _make_multi_room_loop(
            xvf_controllers={"escritorio": _FakeXvf(335000.0)}, spenergy_threshold=100.0
        )
        assert loop._passes_spenergy_gate(self._rs()) is True


class TestSpenergyGateEarlyDispatch:
    """El pre-gate SPENERGY debe cubrir TAMBIÉN el path early_dispatch (QW-1
    2026-06-04): el bloque early en run() despachaba sin consultar el gate, así
    que una alucinación con forma de comando (grammar full sobre ruido) se
    ejecutaba saltándose el VAD por hardware. Con early_dispatch:true ese es el
    path más usado en prod."""

    def _make_ready_partial_command(self):
        """PartialCommand-like ya listo para despachar (intent+entity)."""
        pc = MagicMock()
        pc.intent = "turn_on"
        pc.entity = "luz"
        pc.room = "escritorio"
        pc.ready_to_dispatch = MagicMock(return_value=True)
        return pc

    async def _run_one_early_dispatch(self, xvf_peak: float) -> tuple[list, RoomStream]:
        """Corre run() con un room en estado early-ready y SPENERGY=xvf_peak.

        Devuelve (eventos despachados, room stream) tras ~3 iteraciones del
        polling loop.
        """
        rs = _make_room_stream("escritorio")
        rs.listening = True
        rs.command_start_time = time.time()
        rs.audio_buffer = [0.05] * CHUNK_SIZE
        rs.early_command = self._make_ready_partial_command()

        loop = _make_multi_room_loop(
            rooms={"escritorio": rs},
            xvf_controllers={"escritorio": _FakeXvf(xvf_peak)},
            spenergy_threshold=100.0,
        )

        received = []

        async def on_cmd(event):
            received.append(event)
            return {}

        loop.on_command(on_cmd)

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd):
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.15)
            await loop.stop()
            await asyncio.wait_for(run_task, timeout=2.0)
        await asyncio.sleep(0)  # drenar el create_task del dispatch si lo hubo
        return received, rs

    @pytest.mark.asyncio
    async def test_early_dispatch_blocked_when_spenergy_low(self):
        """SPENERGY bajo umbral (secador/silencio) → early_dispatch NO despacha."""
        received, rs = await self._run_one_early_dispatch(xvf_peak=0.0)
        assert received == []
        assert rs.listening is False  # captura reseteada igual
        assert rs.early_command is None

    @pytest.mark.asyncio
    async def test_early_dispatch_passes_when_spenergy_high(self):
        """SPENERGY de voz real (≥ umbral) → early_dispatch despacha normal."""
        received, rs = await self._run_one_early_dispatch(xvf_peak=335000.0)
        assert len(received) == 1
        assert received[0].early_dispatch is True
        assert rs.listening is False


class TestCallbacks:
    """Test callback registration."""

    def test_on_command_callback(self):
        """on_command registers callback."""
        loop = _make_multi_room_loop()

        async def my_callback(event: CommandEvent) -> dict:
            return {"text": "test"}

        loop.on_command(my_callback)
        assert loop._on_command_callback is my_callback

    def test_on_post_command_callback(self):
        """on_post_command registers callback."""
        loop = _make_multi_room_loop()

        async def my_post_callback(result: dict, event: CommandEvent) -> None:
            pass

        loop.on_post_command(my_post_callback)
        assert loop._on_post_command_callback is my_post_callback

    def test_both_callbacks(self):
        """Both callbacks can be registered independently."""
        loop = _make_multi_room_loop()

        async def cmd_cb(event):
            return {}

        async def post_cb(result, event):
            pass

        loop.on_command(cmd_cb)
        loop.on_post_command(post_cb)

        assert loop._on_command_callback is cmd_cb
        assert loop._on_post_command_callback is post_cb


class TestDispatchCommand:
    """Test _dispatch_command async method."""

    @pytest.mark.asyncio
    async def test_dispatch_command_calls_callback(self):
        """_dispatch_command calls the registered callback with CommandEvent."""
        loop = _make_multi_room_loop()

        received_events = []

        async def mock_callback(event: CommandEvent) -> dict:
            received_events.append(event)
            return {"text": "luz encendida", "success": True}

        loop.on_command(mock_callback)

        audio = np.zeros(16000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina", mic_device_index=2)

        await loop._dispatch_command(event)

        assert len(received_events) == 1
        assert received_events[0].room_id == "cocina"
        assert received_events[0].mic_device_index == 2

    @pytest.mark.asyncio
    async def test_dispatch_command_calls_post_callback(self):
        """_dispatch_command calls both on_command and on_post_command."""
        loop = _make_multi_room_loop()

        post_results = []

        async def mock_cmd(event: CommandEvent) -> dict:
            return {"text": "ok", "success": True}

        async def mock_post(result: dict, event: CommandEvent) -> None:
            post_results.append((result, event.room_id))

        loop.on_command(mock_cmd)
        loop.on_post_command(mock_post)

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="living", mic_device_index=3)

        await loop._dispatch_command(event)

        assert len(post_results) == 1
        assert post_results[0][0] == {"text": "ok", "success": True}
        assert post_results[0][1] == "living"

    @pytest.mark.asyncio
    async def test_dispatch_command_no_callback(self, caplog):
        """_dispatch_command warns if no callback registered."""
        import logging

        loop = _make_multi_room_loop()

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        with caplog.at_level(logging.WARNING):
            await loop._dispatch_command(event)

        assert "No on_command callback registered" in caplog.text

    @pytest.mark.asyncio
    async def test_dispatch_command_handles_exception(self, caplog):
        """_dispatch_command catches and logs callback exceptions."""
        import logging

        loop = _make_multi_room_loop()

        async def failing_callback(event):
            raise RuntimeError("STT failed")

        loop.on_command(failing_callback)

        audio = np.zeros(8000, dtype=np.float32)
        event = CommandEvent(audio=audio, room_id="cocina")

        with caplog.at_level(logging.ERROR):
            await loop._dispatch_command(event)

        assert "Command dispatch failed for cocina" in caplog.text


class TestStart:
    """Test start() method."""

    @pytest.mark.asyncio
    async def test_start_loads_wake_word_all_rooms(self):
        """start() calls wake_detector.load() for every room."""
        loop = _make_multi_room_loop()

        await loop.start()

        for room_id, rs in loop.room_streams.items():
            rs.wake_detector.load.assert_called_once()


class TestStop:
    """Test stop() method."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() sets _running to False."""
        loop = _make_multi_room_loop()
        loop._running = True

        await loop.stop()

        assert loop._running is False

    @pytest.mark.asyncio
    async def test_stop_stops_xvf_controller(self):
        """stop() detiene el XvfController (vía to_thread — el join sincrónico
        del poller no debe correr en el event loop; review 2026-06-04)."""
        xvf = _FakeXvf(0.0)
        xvf.stopped = False
        xvf.stop = lambda: setattr(xvf, "stopped", True)
        loop = _make_multi_room_loop(xvf_controllers={"cocina": xvf})
        loop._running = True

        await loop.stop()

        assert xvf.stopped is True
        assert loop._running is False


class TestResolveCapturChannels:
    """Test _resolve_capture_channels pure function."""

    @pytest.mark.parametrize("reported,expected", [
        (0, 1),
        (1, 1),
        (2, 2),
        (6, 6),
    ])
    def test_resolve_capture_channels(self, reported, expected):
        assert _resolve_capture_channels(reported) == expected


class TestCaptureChannel:
    """L-3 prep (2026-06-04): canal de captura configurable per-room.

    El XVF3800 UA expone 2 canales (doc Seeed: ch0=Conference con post-proceso
    para oído humano, ch1=ASR del beam auto-select). Hoy se consume ch0 fijo;
    capture_channel permite el A/B per-device SIN swap global (el mic UAC1.0
    del escritorio es mono → un swap ciego daría IndexError)."""

    def _loop_with_channel(self, capture_channel):
        det = _detector_seq([("nexa", 0.8)] * 10)
        rs = RoomStream(
            room_id="living", device_index=0,
            wake_detector=det, echo_suppressor=_make_echo_suppressor(),
            capture_channel=capture_channel,
        )
        loop = _make_multi_room_loop(rooms={"living": rs})
        return loop, rs, det

    def test_callback_uses_configured_channel(self):
        loop, rs, det = self._loop_with_channel(capture_channel=1)
        cb = loop._make_audio_callback(rs)
        indata = np.zeros((CHUNK_SIZE, 2), dtype=np.float32)
        indata[:, 0] = 0.01
        indata[:, 1] = 0.99
        cb(indata, CHUNK_SIZE, None, None)
        chunk = det.detect.call_args[0][0]
        assert chunk == pytest.approx(np.full(CHUNK_SIZE, 0.99))

    def test_default_channel_zero_preserved(self):
        loop, rs, det = self._loop_with_channel(capture_channel=0)
        cb = loop._make_audio_callback(rs)
        indata = np.zeros((CHUNK_SIZE, 2), dtype=np.float32)
        indata[:, 0] = 0.01
        indata[:, 1] = 0.99
        cb(indata, CHUNK_SIZE, None, None)
        chunk = det.detect.call_args[0][0]
        assert chunk == pytest.approx(np.full(CHUNK_SIZE, 0.01))

    def test_missing_channel_falls_back_to_zero(self):
        # Mic mono (UAC1.0 escritorio): capture_channel=1 NO debe explotar.
        loop, rs, det = self._loop_with_channel(capture_channel=1)
        cb = loop._make_audio_callback(rs)
        indata = np.full((CHUNK_SIZE, 1), 0.07, dtype=np.float32)
        cb(indata, CHUNK_SIZE, None, None)  # sin IndexError
        chunk = det.detect.call_args[0][0]
        assert chunk == pytest.approx(np.full(CHUNK_SIZE, 0.07))

    def test_room_stream_default_capture_channel(self):
        rs = _make_room_stream("cocina")
        assert rs.capture_channel == 0


class _FakeXvfRW(_FakeXvf):
    """FakeXvf con write/read de parámetros (L-2 apply-on-start)."""

    def __init__(self, peak=0.0, reads=None):
        super().__init__(peak)
        self.writes = []
        self._reads = reads or {}

    def read_param(self, name):
        if name == "NO_EXISTE":
            raise ValueError(f"parámetro desconocido: {name!r}")
        return self._reads.get(name)

    def write_param(self, name, values):
        if name == "NO_EXISTE":
            raise ValueError(f"parámetro desconocido: {name!r}")
        self.writes.append((name, list(values)))
        return True


class TestXvfTuningOnStart:
    """L-2 prep (2026-06-04): tuning del DSP aplicado al arrancar el loop.

    EN RAM (reversible al re-enchufar). Default apply_on_start=False → cero
    writes (sin regresión). Un param inválido en el yaml NO debe tirar el
    servicio (fail-open de config: log + continuar)."""

    @pytest.mark.asyncio
    async def test_tuning_applied_on_start(self):
        xvf = _FakeXvfRW(reads={"PP_AGCMAXGAIN": (64.0,)})
        loop = _make_multi_room_loop(
            xvf_controllers={"cocina": xvf},
            xvf_tuning={
                "apply_on_start": True,
                "params": {"PP_AGCMAXGAIN": [16.0], "PP_AGCONOFF": [1]},
            },
        )
        await loop.start()
        assert ("PP_AGCMAXGAIN", [16.0]) in xvf.writes
        assert ("PP_AGCONOFF", [1]) in xvf.writes

    @pytest.mark.asyncio
    async def test_tuning_off_by_default_no_writes(self):
        xvf = _FakeXvfRW()
        loop = _make_multi_room_loop(
            xvf_controllers={"cocina": xvf},
            xvf_tuning={"params": {"PP_AGCMAXGAIN": [16.0]}},  # sin apply_on_start
        )
        await loop.start()
        assert xvf.writes == []

    @pytest.mark.asyncio
    async def test_tuning_invalid_param_does_not_break_start(self):
        xvf = _FakeXvfRW()
        loop = _make_multi_room_loop(
            xvf_controllers={"cocina": xvf},
            xvf_tuning={
                "apply_on_start": True,
                "params": {"NO_EXISTE": [1], "PP_AGCMAXGAIN": [16.0]},
            },
        )
        await loop.start()  # no explota
        assert ("PP_AGCMAXGAIN", [16.0]) in xvf.writes  # el válido se aplicó

    @pytest.mark.asyncio
    async def test_tuning_without_controller_noop(self):
        loop = _make_multi_room_loop(
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]}},
        )
        await loop.start()  # sin xvf_controller → no explota


class TestXvfReviewFixes:
    """Fixes de la review adversarial de Fase 1 (2026-06-04):
    - el tuning se aplica ANTES de arrancar el poller (sin ventana de USB
      concurrente en el arranque) y fuera del event loop;
    - spenergy_gate_enabled desacoplado del tuning (dos features ortogonales);
    - tuning configurado sin controller → warning, no silencio."""

    def _xvf_with_events(self):
        xvf = _FakeXvfRW()
        xvf.events = []
        orig_start, orig_write = xvf.start, xvf.write_param

        def tracked_start():
            xvf.events.append("poller_start")
            return orig_start()

        def tracked_write(name, values):
            xvf.events.append(("write", name))
            return orig_write(name, values)

        xvf.start = tracked_start
        xvf.write_param = tracked_write
        return xvf

    @pytest.mark.asyncio
    async def test_tuning_applied_before_poller_starts(self):
        # Sin esto, el write corre con el poller ya leyendo SPENERGY cada 40ms
        # sobre el mismo device handle (transfers USB concurrentes sin lock).
        xvf = self._xvf_with_events()
        loop = _make_multi_room_loop(
            xvf_controllers={"cocina": xvf},
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]}},
        )
        await loop.start()
        assert xvf.events == [("write", "PP_AGCMAXGAIN"), "poller_start"]

    @pytest.mark.asyncio
    async def test_gate_disabled_passes_even_with_low_peak(self):
        rs = _make_room_stream("escritorio")
        rs.command_start_time = 100.0
        loop = _make_multi_room_loop(
            xvf_controllers={"escritorio": _FakeXvf(0.0)},  # pico bajo umbral
            spenergy_threshold=100.0,
            spenergy_gate_enabled=False,
        )
        assert loop._passes_spenergy_gate(rs) is True

    @pytest.mark.asyncio
    async def test_gate_disabled_skips_poller_but_applies_tuning(self):
        # spenergy off + tuning on: el controller sirve SOLO para los writes;
        # el poller (que alimenta el gate) no debe arrancar.
        xvf = self._xvf_with_events()
        loop = _make_multi_room_loop(
            xvf_controllers={"cocina": xvf},
            spenergy_gate_enabled=False,
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCONOFF": [0]}},
        )
        await loop.start()
        assert ("write", "PP_AGCONOFF") in xvf.events
        assert "poller_start" not in xvf.events

    @pytest.mark.asyncio
    async def test_tuning_without_controller_warns(self, caplog):
        import logging
        loop = _make_multi_room_loop(
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]}},
        )
        with caplog.at_level(logging.WARNING):
            await loop.start()
        assert "xvf_tuning" in caplog.text.lower()


# ============================================================
# AmbientGuard integration (spec 2026-06-05)
# ============================================================

from src.pipeline.ambient_guard import (
    AmbientGuard,
    AmbientGuardConfig,
    GuardState,
)


def _make_enabled_guard(**overrides) -> AmbientGuard:
    cfg = AmbientGuardConfig(
        enabled=True,
        strict_entry_rejects=2,
        strict_entry_window_s=60.0,
        strict_wake_score=0.65,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return AmbientGuard(config=cfg)


class TestAmbientGuardIntegration:
    def test_no_guard_keeps_current_behavior(self):
        loop = _make_multi_room_loop()
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.41) is True

    def test_guard_rejects_low_score_in_strict(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")  # → STRICT
        loop = _make_multi_room_loop(ambient_guard=guard)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.50) is False

    def test_guard_accepts_high_score_in_strict(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")
        loop = _make_multi_room_loop(ambient_guard=guard)
        assert loop._should_accept_wakeword("cocina", rms=0.05, timestamp=time.time(),
                                            wake_score=0.80) is True

    @pytest.mark.asyncio
    async def test_dispatch_reports_outcome_to_guard(self):
        guard = _make_enabled_guard()
        loop = _make_multi_room_loop(ambient_guard=guard)
        # Callback que simula rechazo del gate por ruido real (TV/frase noise,
        # no una alucinación de silencio — ver TestClassifyOutcomeGateReasons).
        loop.on_command(AsyncMock(return_value={
            "success": False, "text": "gracias por ver",
            "intent": "gate_rejected:noise_phrase:'gracias por ver'",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        await loop._dispatch_command(event)
        await loop._dispatch_command(event)
        # 2 rechazos con strict_entry_rejects=2 → STRICT
        assert guard.state_for("cocina") is GuardState.STRICT

    @pytest.mark.asyncio
    async def test_dispatch_accepted_does_not_escalate(self):
        guard = _make_enabled_guard()
        loop = _make_multi_room_loop(ambient_guard=guard)
        loop.on_command(AsyncMock(return_value={
            "success": True, "text": "prende la luz", "intent": "domotics",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        for _ in range(5):
            await loop._dispatch_command(event)
        assert guard.state_for("cocina") is GuardState.NORMAL

    def test_command_event_carries_ambient_strict_default_false(self):
        event = CommandEvent(audio=np.zeros(10, dtype=np.float32), room_id="cocina")
        assert event.ambient_strict is False


class TestGuardRejectionClearsRefractory:
    """Bug encontrado en validación en vivo 2026-06-05 (escenario 2): un frame
    de TV a 0.528 disparó el detector, el guard lo rechazó (STRICT), pero el
    refractario de 2s del detector quedó abierto → el "Nexa" real del usuario
    a 0.907 80ms después fue suprimido por detect() y nunca llegó al guard.
    El rechazo del guard NO debe consumir la ventana refractaria."""

    def test_guard_rejection_resets_detector_refractory(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")  # → STRICT
        loop = _make_multi_room_loop(ambient_guard=guard)
        rs = loop.room_streams["cocina"]
        accepted = loop._should_accept_wakeword(
            "cocina", rms=0.05, timestamp=time.time(), wake_score=0.50
        )
        assert accepted is False
        rs.wake_detector.reset_refractory.assert_called_once()

    def test_accepted_wake_does_not_reset_refractory(self):
        guard = _make_enabled_guard()
        loop = _make_multi_room_loop(ambient_guard=guard)
        rs = loop.room_streams["cocina"]
        accepted = loop._should_accept_wakeword(
            "cocina", rms=0.05, timestamp=time.time(), wake_score=0.90
        )
        assert accepted is True
        rs.wake_detector.reset_refractory.assert_not_called()

    def test_dedup_rejection_does_not_reset_refractory(self):
        # Solo el rechazo del GUARD libera el refractario. El rechazo por
        # dedup (eco cross-room) debe dejarlo intacto — si no, el eco
        # re-dispararía cada frame durante la ventana de dedup.
        loop = _make_multi_room_loop()
        now = time.time()
        assert loop._should_accept_wakeword("cocina", rms=0.5, timestamp=now) is True
        accepted = loop._should_accept_wakeword("living", rms=0.01, timestamp=now)
        assert accepted is False  # eco más débil dentro de la ventana
        loop.room_streams["living"].wake_detector.reset_refractory.assert_not_called()


class TestPostSuccessFollowUp:
    """Gracia post-éxito (2026-06-06): en STRICT el follow_up no se abre al
    wake; tras un resultado ACEPTADO se abre acá (el guard ya registró
    last_accept_at en on_capture_result → follow_up_allowed=True)."""

    @pytest.mark.asyncio
    async def test_accepted_dispatch_opens_follow_up(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")  # → STRICT
        loop = _make_multi_room_loop(ambient_guard=guard)
        loop.on_command(AsyncMock(return_value={
            "success": True, "text": "apaga la luz", "intent": "domotics",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        await loop._dispatch_command(event)
        loop.follow_up.start_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejected_dispatch_does_not_open_follow_up(self):
        guard = _make_enabled_guard()
        guard.on_capture_result("cocina", "noise")
        guard.on_capture_result("cocina", "noise")
        loop = _make_multi_room_loop(ambient_guard=guard)
        loop.on_command(AsyncMock(return_value={
            "success": False, "text": "gracias por ver",
            "intent": "gate_rejected:noise_phrase:'gracias por ver'",
        }))
        event = CommandEvent(audio=np.zeros(16000, dtype=np.float32), room_id="cocina")
        await loop._dispatch_command(event)
        loop.follow_up.start_conversation.assert_not_called()


from src.pipeline.multi_room_audio_loop import detect_stale_streams


class TestCallbackStampsFrameTimestamp:
    def test_fields_default(self):
        rs = _make_room_stream("escritorio", device_index=4)
        assert rs.last_frame_ts == 0.0
        assert rs.mic_usb_port is None

    def test_callback_updates_last_frame_ts(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        callback = loop._make_audio_callback(rs)
        # Buffer CON señal: desde 2026-07-29 el heartbeat exige audio, no la
        # mera invocación del callback (ver TestWatchdogHeartbeatRequiresSignal).
        indata = np.zeros((160, 2), dtype="float32")
        indata[:, 0] = 0.02
        assert rs.last_frame_ts == 0.0
        callback(indata, 160, None, None)
        assert rs.last_frame_ts > 0.0


class TestDetectStaleStreams:
    # states = (room_id, last_frame_ts, opened_ts)
    GRACE = 180.0

    def test_marks_stream_past_timeout(self):
        # last_frame_ts=100.0, now=109.0 → 9s sin frames > 8s
        assert detect_stale_streams(
            [("escritorio", 100.0, 90.0)], now=109.0, timeout_s=8.0,
            first_frame_grace_s=self.GRACE,
        ) == ["escritorio"]

    def test_ignores_fresh_stream(self):
        # 2s sin frames < 8s
        assert detect_stale_streams(
            [("escritorio", 100.0, 90.0)], now=102.0, timeout_s=8.0,
            first_frame_grace_s=self.GRACE,
        ) == []

    def test_ignores_never_opened_stream(self):
        # opened_ts=0.0 → nunca se abrió, no se marca
        assert detect_stale_streams(
            [("escritorio", 0.0, 0.0)], now=999.0, timeout_s=8.0,
            first_frame_grace_s=self.GRACE,
        ) == []

    def test_multiple_streams_only_stale_returned(self):
        states = [("a", 100.0, 90.0), ("b", 108.5, 90.0), ("c", 0.0, 0.0)]
        # now=110: a=10s stale, b=1.5s fresh, c=never opened
        assert detect_stale_streams(
            states, now=110.0, timeout_s=8.0, first_frame_grace_s=self.GRACE,
        ) == ["a"]


class TestFirstFrameGrace:
    """Distinguir "todavía no arrancó" de "dejó de entregar".

    Medido sobre arranques reales (2026-07-29/30): el primer frame llega
    normalmente en 1.5-2s, pero se observó un arranque que tardó 135s. Sin
    período de gracia, un mic que todavía está despertando se leería como
    muerto y el watchdog dispararía recovery cada `timeout_s` — y como el
    recovery cierra TODOS los streams, se llevaría puestos a los sanos.
    """

    def test_just_opened_without_frames_is_not_stale(self):
        # abrió hace 10s, nunca entregó, gracia 180s → todavía despertando
        assert detect_stale_streams(
            [("escritorio", 0.0, 100.0)],
            now=110.0, timeout_s=8.0, first_frame_grace_s=180.0,
        ) == []

    def test_never_delivering_past_grace_is_stale(self):
        # abrió hace 200s y nunca entregó un frame → el mic no va a arrancar
        assert detect_stale_streams(
            [("escritorio", 0.0, 100.0)],
            now=300.0, timeout_s=8.0, first_frame_grace_s=180.0,
        ) == ["escritorio"]

    def test_was_delivering_then_stopped_uses_short_timeout(self):
        # ya había entregado (ts=150): no hereda la gracia, vale el timeout
        assert detect_stale_streams(
            [("escritorio", 150.0, 100.0)],
            now=160.0, timeout_s=8.0, first_frame_grace_s=180.0,
        ) == ["escritorio"]

    def test_never_opened_is_still_ignored(self):
        # opened_ts=0.0 → el stream nunca se abrió, no hay nada que recuperar
        assert detect_stale_streams(
            [("escritorio", 0.0, 0.0)],
            now=999.0, timeout_s=8.0, first_frame_grace_s=180.0,
        ) == []


class TestOpenStream:
    def test_open_stream_returns_started_stream(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        fake_stream = MagicMock()
        mock_sd.InputStream.return_value = fake_stream
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd):
            result = loop._open_stream(rs)
        assert result is fake_stream
        fake_stream.start.assert_called_once()

    def test_open_stream_returns_none_on_portaudio_error(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        rs = loop.room_streams["escritorio"]
        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.side_effect = mock_sd.PortAudioError("no device")
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd):
            result = loop._open_stream(rs)
        assert result is None


class TestWatchdogHeartbeatRequiresSignal:
    """El heartbeat del watchdog debe medir AUDIO, no llamadas al callback.

    Incidente 2026-07-29: el XVF3800 quedó con el endpoint isócrono degradado
    —enumera, el stream abre, PortAudio sigue invocando el callback— pero
    entregando ceros. Como `last_frame_ts` se refrescaba incondicionalmente en
    la primera línea, el watchdog vio un stream sano durante ~7 minutos y el
    sistema quedó sordo sin una sola alerta, con el servicio en `active`.

    Misma lección que [arecord exit 0 ≠ audio]: contar muestras no-cero, no
    invocaciones.
    """

    def _loop_and_room(self):
        rs = _make_room_stream("escritorio", device_index=4)
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        return loop, rs

    def test_silent_buffer_does_not_refresh_heartbeat(self):
        loop, rs = self._loop_and_room()
        cb = loop._make_audio_callback(rs)
        rs.last_frame_ts = 0.0

        cb(np.zeros((CHUNK_SIZE, 2), dtype=np.float32), CHUNK_SIZE, None, None)

        assert rs.last_frame_ts == 0.0, (
            "un buffer de ceros no es audio: el watchdog tiene que poder verlo"
        )

    def test_real_audio_refreshes_heartbeat(self):
        loop, rs = self._loop_and_room()
        cb = loop._make_audio_callback(rs)
        rs.last_frame_ts = 0.0

        indata = np.zeros((CHUNK_SIZE, 2), dtype=np.float32)
        indata[:, 0] = 0.02  # ruido de fondo de un mic vivo
        cb(indata, CHUNK_SIZE, None, None)

        assert rs.last_frame_ts > 0.0

    def test_faint_noise_still_counts_as_signal(self):
        """El piso de ruido de un mic sano cuenta: el criterio es no-cero.

        No un umbral de RMS — el silencio real de una habitación vacía trae
        ruido diminuto pero distinto de cero, y no debe leerse como mic muerto.
        """
        loop, rs = self._loop_and_room()
        cb = loop._make_audio_callback(rs)
        rs.last_frame_ts = 0.0

        indata = np.zeros((CHUNK_SIZE, 2), dtype=np.float32)
        indata[0, 0] = 1e-6
        cb(indata, CHUNK_SIZE, None, None)

        assert rs.last_frame_ts > 0.0


class TestStreamWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_recovers_when_stream_stale(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 0.05
        # frame "viejo": monotonic muy atrás → stale
        rs.last_frame_ts = time.monotonic() - 10.0

        called = {}
        async def fake_recover(ids):
            called["ids"] = ids
            loop._running = False  # corta el loop tras una recuperación
        loop._recover_streams = fake_recover

        await asyncio.wait_for(loop._stream_watchdog(), timeout=1.0)
        assert called.get("ids") == ["escritorio"]

    @pytest.mark.asyncio
    async def test_watchdog_ignores_room_already_awaiting_reopen(self):
        """Una room que ya espera su device no debe re-disparar el recovery.

        Su `last_frame_ts` no se actualiza nunca (no hay device que entregue
        frames), así que queda stale para siempre. Sin este filtro el watchdog
        llamaría a `_recover_streams` en cada ciclo, y como el recovery cierra
        TODOS los streams, el mic sano se cerraría cada `check_interval`.
        """
        ausente = _make_room_stream("cocina", device_index=10)
        ausente.mic_usb_port = "5-5.3"
        sana = _make_room_stream("escritorio", device_index=4)
        loop = _make_multi_room_loop(
            rooms={"cocina": ausente, "escritorio": sana}
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        # timeout amplio: la room sana no puede volverse stale por el propio
        # tiempo que corre el test (si no, la aserción mediría una carrera).
        loop._watchdog_timeout_s = 30.0
        ausente.last_frame_ts = time.monotonic() - 600.0  # stale permanente
        sana.last_frame_ts = time.monotonic()             # sana y fresca

        # La cocina ya tiene una espera de reapertura viva.
        espera = asyncio.get_running_loop().create_future()
        loop._reopen_tasks = {"cocina": asyncio.ensure_future(espera)}

        llamadas = []
        async def fake_recover(ids):
            llamadas.append(ids)
            loop._running = False
        loop._recover_streams = fake_recover

        async def parar():
            await asyncio.sleep(0.05)
            loop._running = False
        await asyncio.gather(loop._stream_watchdog(), parar())

        espera.cancel()
        assert llamadas == [], (
            f"no debía dispararse el recovery, pero se llamó con {llamadas}"
        )

    @pytest.mark.asyncio
    async def test_watchdog_noop_when_fresh(self):
        rs = _make_room_stream("escritorio", device_index=4)
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 5.0
        rs.last_frame_ts = time.monotonic()  # fresco

        called = {"n": 0}
        async def fake_recover(ids):
            called["n"] += 1
        loop._recover_streams = fake_recover

        async def stop_soon():
            await asyncio.sleep(0.05)
            loop._running = False
        await asyncio.gather(loop._stream_watchdog(), stop_soon())
        assert called["n"] == 0


class TestRecoverStreams:
    @pytest.mark.asyncio
    async def test_recover_reinits_portaudio_and_reopens(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        old_stream = MagicMock()
        loop._streams = {"escritorio": old_stream}

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        new_stream = MagicMock()
        mock_sd.InputStream.return_value = new_stream
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            return_value=7,
        ):
            await loop._recover_streams(["escritorio"])

        old_stream.close.assert_called_once()          # cerró el muerto
        assert mock_sd._terminate.called and mock_sd._initialize.called  # reinit
        assert rs.device_index == 7                    # re-resolvió por puerto
        assert loop._streams["escritorio"] is new_stream  # reabrió
        # Stream nuevo: arranca la gracia del primer frame, y el heartbeat
        # queda en 0 hasta que llegue audio REAL (no la apertura del stream).
        assert rs.opened_ts > 0.0
        assert rs.last_frame_ts == 0.0

    @pytest.mark.asyncio
    async def test_reopen_waits_with_backoff_when_device_absent(self):
        rs = _make_room_stream("escritorio", device_index=4)
        rs.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(rooms={"escritorio": rs})
        loop._running = True
        loop._watchdog_backoff_min_s = 0.001
        loop._watchdog_backoff_max_s = 0.004

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        mock_sd.InputStream.return_value = MagicMock()
        # 1ra resolución None (ausente), 2da devuelve índice → 1 reintento
        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            side_effect=[None, 7],
        ):
            await loop._reopen_room(rs)

        assert rs.device_index == 7
        assert "escritorio" in loop._streams

    @pytest.mark.asyncio
    async def test_recover_reopens_healthy_room_when_another_device_absent(self):
        """Un mic ausente NO debe impedir que los sanos vuelvan.

        Incidente 2026-07-28: la cocina se desenchufó del USB y
        `_recover_streams` quedó esperando su device indefinidamente
        (`_reopen_room` loopea `while self._running`), así que el `for`
        nunca llegó al escritorio —presente y sano— y el sistema quedó
        27h sin capturar audio con el servicio `active`.

        La room ausente va PRIMERO en el dict a propósito: reproduce el
        orden real de `room_streams` que disparó el incidente.
        """
        ausente = _make_room_stream("cocina", device_index=10)
        ausente.mic_usb_port = "5-5.3"
        sana = _make_room_stream("escritorio", device_index=4)
        sana.mic_usb_port = "3-1.4"
        loop = _make_multi_room_loop(
            rooms={"cocina": ausente, "escritorio": sana}
        )
        loop._running = True
        loop._watchdog_backoff_min_s = 0.001
        loop._watchdog_backoff_max_s = 0.002
        loop._streams = {"cocina": MagicMock(), "escritorio": MagicMock()}

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}
        nuevo = MagicMock()
        mock_sd.InputStream.return_value = nuevo

        def resolve(port):
            return None if port == "5-5.3" else 4  # la cocina nunca vuelve

        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            side_effect=resolve,
        ):
            await asyncio.wait_for(loop._recover_streams(["cocina"]), timeout=2.0)

        assert loop._streams.get("escritorio") is nuevo, (
            "el mic sano tiene que reabrirse aunque otro siga ausente"
        )
        assert "cocina" not in loop._streams  # la ausente sigue sin stream

    @pytest.mark.asyncio
    async def test_recover_returns_without_waiting_for_absent_device(self):
        """`_recover_streams` no puede quedarse colgado por un mic ausente.

        Si no retorna, el `await` de `_stream_watchdog` tampoco vuelve y el
        watchdog deja de vigilar al resto: una sola falla lo apaga entero.
        """
        ausente = _make_room_stream("cocina", device_index=10)
        ausente.mic_usb_port = "5-5.3"
        loop = _make_multi_room_loop(rooms={"cocina": ausente})
        loop._running = True
        loop._watchdog_backoff_min_s = 0.001
        loop._watchdog_backoff_max_s = 0.002
        loop._streams = {"cocina": MagicMock()}

        mock_sd = MagicMock()
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"max_input_channels": 2}

        with patch("src.pipeline.multi_room_audio_loop.sd", mock_sd), patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            return_value=None,
        ):
            await asyncio.wait_for(loop._recover_streams(["cocina"]), timeout=2.0)

        loop._running = False  # corta cualquier espera en background

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_reopen_tasks(self):
        """`stop()` no puede dejar esperas de reapertura vivas.

        Si sobreviven, siguen intentando abrir devices mientras el proceso
        se apaga (y en un restart rápido, contra el proceso nuevo).
        """
        ausente = _make_room_stream("cocina", device_index=10)
        ausente.mic_usb_port = "5-5.3"
        loop = _make_multi_room_loop(rooms={"cocina": ausente})
        loop._running = True
        loop._watchdog_backoff_min_s = 0.01
        loop._watchdog_backoff_max_s = 0.02

        with patch(
            "src.pipeline.multi_room_audio_loop.resolve_mic_usb_port",
            return_value=None,
        ):
            loop._schedule_reopen(ausente)
            await asyncio.sleep(0)  # dejar arrancar la task
            tarea = loop._reopen_tasks["cocina"]
            assert not tarea.done()

            await loop.stop()

        assert tarea.cancelled() or tarea.done()
        assert not loop._reopen_tasks


class TestAudioHealthPublication:
    """El snapshot de salud tiene que SALIR del proceso, no solo calcularse.

    Cobertura que faltaba: borrar el bloque entero que publica el snapshot
    desde `_stream_watchdog` sobrevivía la suite sin que cayera un solo test
    (ningún test tocaba `_audio_health_path`, ni la publicación desde el loop,
    ni la lectura de `rooms.stream_watchdog.health_path` en main.py). O sea,
    la alerta de sordera entera podía desaparecer en un merge y nadie se
    enteraba — el mismo fallo silencioso que la alerta existe para atrapar.
    """

    async def _run_one_watchdog_iteration(self, loop):
        """Correr exactamente un ciclo del watchdog y frenar."""
        async def parar():
            await asyncio.sleep(0.05)
            loop._running = False

        await asyncio.gather(loop._stream_watchdog(), parar())

    @pytest.mark.asyncio
    async def test_watchdog_publishes_snapshot_with_every_room(self, tmp_path):
        health = tmp_path / "audio_health.json"
        cocina = _make_room_stream("cocina", device_index=2)
        escritorio = _make_room_stream("escritorio", device_index=4)
        loop = _make_multi_room_loop(
            rooms={"cocina": cocina, "escritorio": escritorio},
            audio_health_path=str(health),
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 30.0
        loop._watchdog_first_frame_grace_s = 30.0
        cocina.last_frame_ts = time.monotonic()
        escritorio.last_frame_ts = time.monotonic()

        await self._run_one_watchdog_iteration(loop)

        assert health.exists(), (
            "el watchdog no publicó el snapshot: el poller externo "
            "(tools/audio_watchdog_alert.py) se queda ciego para siempre"
        )
        data = json.loads(health.read_text())
        assert sorted(data["rooms"]) == ["cocina", "escritorio"]
        assert data["rooms"]["cocina"]["ever"] is True
        assert data["wall"] > 0

    @pytest.mark.asyncio
    async def test_published_snapshot_is_readable_by_the_external_poller(
        self, tmp_path
    ):
        """No basta con escribir un archivo: `evaluate_health` —la función que
        el poller externo usa— tiene que poder leerlo y dar un veredicto. Ata
        el formato que escribe el loop al que consume el poller."""
        health = tmp_path / "audio_health.json"
        sorda = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(
            rooms={"cocina": sorda}, audio_health_path=str(health)
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        # timeout alto: no queremos que dispare recovery durante el test, solo
        # que el snapshot refleje una room callada hace rato.
        loop._watchdog_timeout_s = 9999.0
        loop._watchdog_first_frame_grace_s = 9999.0
        sorda.last_frame_ts = time.monotonic() - 600.0

        await self._run_one_watchdog_iteration(loop)

        snapshot = json.loads(health.read_text())
        deaf = evaluate_health(
            snapshot, now_wall=snapshot["wall"], deaf_after_s=120.0
        )
        assert deaf == ["cocina"]


class TestAudioHealthWriteIsOffTheEventLoop:
    """La escritura del snapshot no puede bloquear el loop del fast path.

    `write_audio_health` hace mkstemp + json.dump + os.replace SÍNCRONOS, y
    `_stream_watchdog` corre en el mismo event loop que el camino de voz
    (<300ms). Con check_interval=2s son ~43.000 creaciones+renames por día
    sobre ./data/, el mismo disco que events.db, latency.db, ChromaDB y el
    entrenamiento nocturno.
    """

    @pytest.mark.asyncio
    async def test_write_runs_in_a_worker_thread(self, tmp_path):
        health = tmp_path / "audio_health.json"
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(
            rooms={"cocina": rs}, audio_health_path=str(health)
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 9999.0
        loop._watchdog_first_frame_grace_s = 9999.0
        rs.last_frame_ts = time.monotonic()

        hilos = []
        real_write = audio_health_mod.write_audio_health

        def spy(*args, **kwargs):
            hilos.append(threading.current_thread())
            return real_write(*args, **kwargs)

        async def parar():
            await asyncio.sleep(0.05)
            loop._running = False

        with patch.object(audio_health_mod, "write_audio_health", spy):
            await asyncio.gather(loop._stream_watchdog(), parar())

        assert hilos, "no se llegó a escribir el snapshot"
        principal = threading.current_thread()
        assert all(h is not principal for h in hilos), (
            "write_audio_health corrió en el event loop: mkstemp+json.dump+"
            "os.replace síncronos delante del fast path"
        )

    @pytest.mark.asyncio
    async def test_cancellation_is_not_swallowed_by_the_write_guard(self, tmp_path):
        """La cancelación tiene que propagar aunque caiga dentro del try.

        Al meter un `await` dentro del bloque protegido, CancelledError pasa a
        ser inyectable justo ahí. Es BaseException, así que `except Exception`
        no la toca — pero si alguien "endurece" ese guard a BaseException o a
        un `except:` pelado, `stop()` dejaría de poder terminar la task y el
        watchdog quedaría vivo para siempre. Este test fija el contrato.
        """
        health = tmp_path / "audio_health.json"
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(
            rooms={"cocina": rs}, audio_health_path=str(health)
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 9999.0
        loop._watchdog_first_frame_grace_s = 9999.0
        rs.last_frame_ts = time.monotonic()

        entro = asyncio.Event()

        def bloquear(*_args, **_kwargs):
            """Frena dentro del try para que la cancelación caiga ahí."""
            loop._loop.call_soon_threadsafe(entro.set)
            time.sleep(0.5)

        with patch.object(audio_health_mod, "write_audio_health", bloquear):
            loop._loop = asyncio.get_running_loop()
            tarea = asyncio.create_task(loop._stream_watchdog())
            await asyncio.wait_for(entro.wait(), timeout=2.0)
            tarea.cancel()
            # `asyncio.wait` y no `await tarea` ni `wait_for`: si el guard SÍ
            # se traga la cancelación, la task sigue girando para siempre —
            # `await` colgaría, y `wait_for` también (al vencer el timeout
            # re-cancela y AWAITEA la misma task insensible). `wait` solo
            # observa, así que el test falla rápido en vez de colgarse.
            done, _ = await asyncio.wait({tarea}, timeout=2.0)
            loop._running = False  # cleanup si la mutación la dejó viva

        assert tarea in done and tarea.cancelled(), (
            "el guard se tragó la cancelación: stop() no podría terminar el "
            "watchdog y la task quedaría viva"
        )


class TestAudioHealthWriteTimeoutDoesNotBlockRecovery:
    """Un write colgado no puede apagar la recuperación de mics.

    `detect_stale_streams` + `_recover_streams` corren DESPUÉS del bloque
    try que publica `audio_health`. Antes del fix, ese `await
    asyncio.to_thread(write_audio_health, ...)` no tenía cota: un thread que
    nunca retorna (fs trabado, disco lleno bloqueando I/O) dejaba el await
    esperando para siempre, y con él, la recuperación de mics enteros —
    aunque el proceso siguiera reportando `active`. Mutación probada:
    sacar el `asyncio.wait_for(...)` y dejar el `await asyncio.to_thread(...)`
    a secas (el código pre-fix) hace que este test falle (TimeoutError a
    los 0.5s) en vez de que `_recover_streams` se llame.
    """

    @pytest.mark.asyncio
    async def test_hung_write_does_not_block_stale_stream_recovery(
        self, tmp_path, monkeypatch
    ):
        # Timeout de escritura chico para que el test sea rápido: la
        # constante real (1.0s) sigue probada indirectamente por el hecho
        # de que este test pasa con cualquier valor finito bien por debajo
        # del sleep colgado de abajo.
        monkeypatch.setattr(mra_mod, "AUDIO_HEALTH_WRITE_TIMEOUT_S", 0.05)

        health = tmp_path / "audio_health.json"
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(
            rooms={"cocina": rs}, audio_health_path=str(health)
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 0.01
        loop._watchdog_first_frame_grace_s = 0.01
        rs.last_frame_ts = time.monotonic() - 10.0  # ya stale al arrancar

        def colgado(*_args, **_kwargs):
            # Sleep finito (no un Event que nunca se setea): así el hilo
            # real del executor termina solo y no deja el proceso de test
            # colgado en el join de atexit de ThreadPoolExecutor. Alcanza
            # con que sea bien mayor al timeout de escritura (0.05s) y al
            # límite del wait_for de abajo (0.5s) para simular "colgado".
            time.sleep(1.0)

        recovered = {}

        async def fake_recover(ids):
            recovered["ids"] = ids
            loop._running = False

        loop._recover_streams = fake_recover

        with patch.object(audio_health_mod, "write_audio_health", colgado):
            # `wait_for` acá es la red de seguridad del TEST: contra el
            # código pre-fix el write cuelga >= 1.0s y este límite de 0.5s
            # se cumple primero -> TimeoutError, falla rápido en vez de
            # trabar la suite. Contra el código con el fix, el ciclo
            # entero (write cortado a los 0.05s + recovery) termina en
            # milisegundos, muy por debajo de 0.5s.
            await asyncio.wait_for(loop._stream_watchdog(), timeout=0.5)

        assert recovered.get("ids") == ["cocina"], (
            "un write colgado no debería impedir _recover_streams: el hilo "
            "bloqueado queda huérfano, pero el watchdog tiene que seguir"
        )


class TestAudioHealthSurvivesRecovery:
    """La señal de sordera no puede resetearse cada vez que el mic se reabre.

    Modo de falla que motivó la alerta (incidentes de 27h y 7h): un XVF3800
    con el endpoint isócrono muerto ABRE perfecto en PortAudio y entrega
    ceros. El watchdog lo detecta a los ~180s y "recupera", y
    `_try_reopen_once` pone `opened_ts = now` y `last_frame_ts = 0.0`.

    Derivar `age_s`/`ever` de esos dos campos hacía que el snapshot volviera a
    cero en cada ciclo: un bucle eterno de ~182s donde `age_s` solo superaba
    el umbral ~2s por vuelta. Simulado por el revisor (200 corridas, poll cada
    60s, 24h): mediana 1,67h hasta detectar, máximo 15,98h, 1/200 sin detectar
    en 24h. O sea, la alerta era MÁS DÉBIL justo en el caso primario.

    ⚠️ El fix intuitivo (subir `first_frame_grace_s` del poller) EMPEORA esto:
    `age_s` está acotado por el ciclo de recovery, así que un umbral más alto
    hace que no se detecte NUNCA. Por eso las anclas de acá viven fuera de
    `RoomStream`, donde la reapertura no las alcanza.
    """

    def _reabrir(self, rs, now_mono):
        """Lo que `_try_reopen_once` le hace al RoomStream al reabrir."""
        rs.opened_ts = now_mono
        rs.last_frame_ts = 0.0

    def test_never_delivering_mic_keeps_aging_across_recoveries(self):
        """El mic que nunca entrega: `age_s` tiene que crecer igual."""
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(rooms={"cocina": rs})
        rs.last_frame_ts = 0.0
        rs.opened_ts = 500.0

        # t=0: arranca. Nunca entrega un frame.
        loop._update_audio_anchors(now_mono=500.0, now_wall=1000.0)

        # Tres ciclos de recovery de ~182s: abre bien, entrega ceros, el
        # watchdog recupera, la reapertura pisa opened_ts/last_frame_ts.
        for i in range(1, 4):
            mono = 500.0 + 182.0 * i
            self._reabrir(rs, mono)
            loop._update_audio_anchors(now_mono=mono, now_wall=1000.0 + 182.0 * i)

        rooms = loop._audio_health_rooms()
        anchor, ever = rooms["cocina"]
        assert ever is False, "nunca entregó audio: `ever` debe seguir en False"
        age_s = (1000.0 + 182.0 * 3) - anchor
        assert age_s == pytest.approx(546.0), (
            f"age_s={age_s}: la reapertura reseteó el ancla. Con el reset, la "
            f"sordera solo es visible ~2s cada ~182s y el poller la pierde."
        )

    def test_reopen_does_not_erase_that_the_room_ever_delivered(self):
        """`ever` no puede volver a False: cambia el umbral del poller.

        `evaluate_health` usa `first_frame_grace_s` (180s) cuando `ever` es
        False y `deaf_after_s` (300s en producción) cuando es True. Si una
        reapertura borrara el `ever`, el veredicto cambiaría de umbral solo
        por haberse reabierto, no por lo que el mic esté haciendo.
        """
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(rooms={"cocina": rs})

        # La room entregó audio real en t_mono=600 (wall 1100).
        rs.last_frame_ts = 600.0
        loop._update_audio_anchors(now_mono=600.0, now_wall=1100.0)
        assert loop._audio_health_rooms()["cocina"][1] is True

        # Se queda muda y el watchdog la reabre 200s después.
        self._reabrir(rs, 800.0)
        loop._update_audio_anchors(now_mono=800.0, now_wall=1300.0)

        anchor, ever = loop._audio_health_rooms()["cocina"]
        assert ever is True, "la reapertura borró que la room ya había entregado audio"
        assert 1300.0 - anchor == pytest.approx(200.0), (
            "el ancla debe seguir marcando el ÚLTIMO audio real, no la reapertura"
        )

    def test_anchor_advances_when_real_audio_arrives(self):
        """El ancla no es un cero permanente: audio real la mueve hacia adelante."""
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(rooms={"cocina": rs})

        rs.last_frame_ts = 600.0
        loop._update_audio_anchors(now_mono=600.0, now_wall=1100.0)
        primero = loop._audio_health_rooms()["cocina"][0]

        rs.last_frame_ts = 900.0  # llegó audio nuevo
        loop._update_audio_anchors(now_mono=900.0, now_wall=1400.0)
        segundo = loop._audio_health_rooms()["cocina"][0]

        assert segundo > primero
        assert 1400.0 - segundo == pytest.approx(0.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_poller_flags_the_recovery_cycle_end_to_end(self, tmp_path):
        """El escenario completo, hasta el veredicto del poller externo.

        Los tests de arriba ejercitan las anclas; este corre el watchdog real,
        lee el snapshot que quedó en disco y lo pasa por `evaluate_health`
        —igual que el poller— para confirmar que la room del ciclo de recovery
        se reporta SORDA en vez de eternamente joven.
        """
        health = tmp_path / "audio_health.json"
        rs = _make_room_stream("cocina", device_index=2)
        loop = _make_multi_room_loop(
            rooms={"cocina": rs}, audio_health_path=str(health)
        )
        loop._running = True
        loop._watchdog_check_interval_s = 0.001
        loop._watchdog_timeout_s = 9999.0
        loop._watchdog_first_frame_grace_s = 9999.0

        # La room nunca entregó audio y ya lleva 40 min bajo observación: es
        # el estado en que la deja el ciclo de recovery (opened_ts recién
        # pisado, last_frame_ts en 0).
        rs.last_frame_ts = 0.0
        rs.opened_ts = time.monotonic()  # "recién reabierta"
        loop._room_since_wall = {"cocina": time.time() - 2400.0}

        async def parar():
            await asyncio.sleep(0.05)
            loop._running = False

        await asyncio.gather(loop._stream_watchdog(), parar())

        snapshot = json.loads(health.read_text())
        deaf = evaluate_health(
            snapshot,
            now_wall=snapshot["wall"],
            deaf_after_s=300.0,
            first_frame_grace_s=180.0,
        )
        assert deaf == ["cocina"], (
            "el poller no ve la sordera: el snapshot se rearmó con el "
            "opened_ts que la reapertura acaba de pisar"
        )


class TestWatchdogConfigContract:
    def test_disabled_by_default(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)}
        )
        assert loop._watchdog_enabled is False

    def test_enabled_via_kwarg(self):
        loop = _make_multi_room_loop(
            rooms={"escritorio": _make_room_stream("escritorio", device_index=4)},
            stream_watchdog_enabled=True,
            stream_watchdog_no_frames_timeout_s=8.0,
        )
        assert loop._watchdog_enabled is True
        assert loop._watchdog_timeout_s == 8.0


# ============================================================
# Controllers per-room (2026-07-25): al conectar el SEGUNDO XVF3800 (cocina),
# un controller compartido tunea/gatea con el mic de OTRA habitación —
# usb.core.find() devuelve el primero que enumere. Cada room usa el suyo,
# bindeado por su mic_usb_port.
# ============================================================

def _two_rooms():
    return {
        "cocina": _make_room_stream("cocina", device_index=2),
        "escritorio": _make_room_stream("escritorio", device_index=3),
    }


class TestPerRoomXvfControllers:

    @pytest.mark.asyncio
    async def test_tuning_applied_to_each_rooms_own_controller(self):
        # El bug: con un solo controller, el escritorio se quedaba en preset
        # de fábrica (MAXGAIN=64) mientras la cocina recibía el tuning.
        coc = _FakeXvfRW(reads={"PP_AGCMAXGAIN": (64.0,)})
        esc = _FakeXvfRW(reads={"PP_AGCMAXGAIN": (64.0,)})
        loop = _make_multi_room_loop(
            rooms=_two_rooms(),
            xvf_controllers={"cocina": coc, "escritorio": esc},
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]}},
        )
        await loop.start()
        assert ("PP_AGCMAXGAIN", [16.0]) in coc.writes
        assert ("PP_AGCMAXGAIN", [16.0]) in esc.writes

    def test_gate_uses_the_rooms_own_controller(self):
        # cocina en silencio, escritorio con voz: cada gate mira SU mic.
        loop = _make_multi_room_loop(
            rooms=_two_rooms(),
            xvf_controllers={
                "cocina": _FakeXvf(0.0),
                "escritorio": _FakeXvf(335000.0),
            },
            spenergy_threshold=100.0,
        )
        rs_coc = loop.room_streams["cocina"]
        rs_coc.command_start_time = 100.0
        rs_esc = loop.room_streams["escritorio"]
        rs_esc.command_start_time = 100.0
        assert loop._passes_spenergy_gate(rs_coc) is False
        assert loop._passes_spenergy_gate(rs_esc) is True

    def test_room_without_controller_fails_open(self):
        # Sin controller propio NO se gatea con el de otra room: fail-open.
        loop = _make_multi_room_loop(
            rooms=_two_rooms(),
            xvf_controllers={"escritorio": _FakeXvf(0.0)},
            spenergy_threshold=100.0,
        )
        rs = loop.room_streams["cocina"]
        rs.command_start_time = 100.0
        assert loop._passes_spenergy_gate(rs) is True

    @pytest.mark.asyncio
    async def test_stop_stops_every_controller(self):
        coc, esc = _FakeXvf(0.0), _FakeXvf(0.0)
        for x in (coc, esc):
            x.stopped = False
            x.stop = lambda x=x: setattr(x, "stopped", True)
        loop = _make_multi_room_loop(
            rooms=_two_rooms(),
            xvf_controllers={"cocina": coc, "escritorio": esc},
        )
        await loop.start()
        await loop.stop()
        assert coc.stopped is True
        assert esc.stopped is True
