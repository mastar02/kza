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

from src.pipeline.multi_room_audio_loop import (
    MultiRoomAudioLoop,
    RoomStream,
    CHUNK_SIZE,
    _resolve_capture_channels,
)
from src.pipeline.command_event import CommandEvent


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
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(None), spenergy_threshold=100.0)
        assert loop._passes_spenergy_gate(self._rs()) is True

    def test_low_peak_blocks(self):
        # secador/silencio = 0 < 100 → descarta
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(0.0), spenergy_threshold=100.0)
        assert loop._passes_spenergy_gate(self._rs()) is False

    def test_voice_peak_passes(self):
        # voz medida ~335k ≥ 100 → procesa
        loop = _make_multi_room_loop(xvf_controller=_FakeXvf(335000.0), spenergy_threshold=100.0)
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
            xvf_controller=_FakeXvf(xvf_peak),
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
        loop = _make_multi_room_loop(xvf_controller=xvf)
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
            xvf_controller=xvf,
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
            xvf_controller=xvf,
            xvf_tuning={"params": {"PP_AGCMAXGAIN": [16.0]}},  # sin apply_on_start
        )
        await loop.start()
        assert xvf.writes == []

    @pytest.mark.asyncio
    async def test_tuning_invalid_param_does_not_break_start(self):
        xvf = _FakeXvfRW()
        loop = _make_multi_room_loop(
            xvf_controller=xvf,
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
            xvf_controller=xvf,
            xvf_tuning={"apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]}},
        )
        await loop.start()
        assert xvf.events == [("write", "PP_AGCMAXGAIN"), "poller_start"]

    @pytest.mark.asyncio
    async def test_gate_disabled_passes_even_with_low_peak(self):
        rs = _make_room_stream("escritorio")
        rs.command_start_time = 100.0
        loop = _make_multi_room_loop(
            xvf_controller=_FakeXvf(0.0),  # pico bajo umbral
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
            xvf_controller=xvf,
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
        # Callback que simula rechazo del gate (texto ruido)
        loop.on_command(AsyncMock(return_value={
            "success": False, "text": "gracias por ver", "intent": "gate_rejected",
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
            "success": False, "text": "gracias por ver", "intent": "gate_rejected",
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
        indata = np.zeros((160, 2), dtype="float32")
        assert rs.last_frame_ts == 0.0
        callback(indata, 160, None, None)
        assert rs.last_frame_ts > 0.0


class TestDetectStaleStreams:
    def test_marks_stream_past_timeout(self):
        # last_frame_ts=100.0, now=109.0 → 9s sin frames > 8s
        assert detect_stale_streams([("escritorio", 100.0)], now=109.0, timeout_s=8.0) == ["escritorio"]

    def test_ignores_fresh_stream(self):
        # 2s sin frames < 8s
        assert detect_stale_streams([("escritorio", 100.0)], now=102.0, timeout_s=8.0) == []

    def test_ignores_never_opened_stream(self):
        # last_frame_ts=0.0 → nunca recibió/abrió, no se marca
        assert detect_stale_streams([("escritorio", 0.0)], now=999.0, timeout_s=8.0) == []

    def test_multiple_streams_only_stale_returned(self):
        states = [("a", 100.0), ("b", 108.5), ("c", 0.0)]
        # now=110: a=10s stale, b=1.5s fresh, c=never
        assert detect_stale_streams(states, now=110.0, timeout_s=8.0) == ["a"]


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
        assert rs.last_frame_ts > 0.0                  # re-estampó

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
