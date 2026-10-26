"""
Tests for S3 — Barge-in: interrumpir TTS al detectar voz del user.

Cubre:
1. `ResponseHandler.is_speaking` flag (property) antes/durante/después de
   un `speak()`.
2. `ResponseHandler.cancel()` idempotent — llamadas múltiples sin efectos.
3. `MultiRoomAudioLoop` — barge-in accum_ms decae con silencio.
4. `MultiRoomAudioLoop` — voz sostenida ≥ min_duration_ms dispara cancel.
5. `MultiRoomAudioLoop` — sin TTS activo, no dispara barge-in.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np
import pytest

from tests.mocks.responses import response_handler_mock
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from src.pipeline.response_handler import ResponseHandler


# =========================================================================
# Helpers — mocks compartidos con test_multi_room_audio_loop.py
# =========================================================================

def _make_wake_detector():
    m = MagicMock()
    m.load = MagicMock()
    m.detect = MagicMock(return_value=None)
    m.get_active_models = MagicMock(return_value=["hey_jarvis"])
    return m


def _make_echo_suppressor(is_human: bool = True):
    m = MagicMock()
    m.is_safe_to_listen = True
    m.should_process_audio = MagicMock(return_value=(True, "ok"))
    m.is_human_voice = MagicMock(return_value=is_human)
    return m


def _make_follow_up():
    m = MagicMock()
    m.is_active = False
    m.start_conversation = MagicMock()
    return m


def _make_room_stream(room_id: str = "escritorio", echo=None) -> RoomStream:
    return RoomStream(
        room_id=room_id,
        device_index=0,
        wake_detector=_make_wake_detector(),
        echo_suppressor=echo or _make_echo_suppressor(),
    )


def _make_response_handler_mock(is_speaking: bool = False):
    """Mock de ResponseHandler para tests de MultiRoomAudioLoop."""
    rh = response_handler_mock()
    rh.is_speaking = is_speaking
    rh.cancel = AsyncMock(return_value=is_speaking)
    return rh


def _make_loop(response_handler=None, **kwargs) -> MultiRoomAudioLoop:
    rooms = {"escritorio": _make_room_stream("escritorio")}
    defaults = dict(
        room_streams=rooms,
        follow_up=_make_follow_up(),
        sample_rate=16000,
        barge_in_enabled=True,
        barge_in_rms_threshold=0.02,
        barge_in_min_duration_ms=100,
    )
    defaults.update(kwargs)
    return MultiRoomAudioLoop(response_handler=response_handler, **defaults)


# =========================================================================
# 1. ResponseHandler.is_speaking
# =========================================================================

class TestResponseHandlerIsSpeaking:
    def test_is_speaking_false_at_init(self):
        handler = ResponseHandler(tts=MagicMock())
        assert not handler.is_speaking and not handler.is_responding

    @pytest.mark.asyncio
    async def test_is_speaking_tracks_owned_device_and_cancel(self, monkeypatch):
        from src.tts import piper_tts
        from tests.mocks.audio_output import AudioBackend
        from tests.mocks.speech import PCMEngine
        from src.pipeline.response_handler import DeliveryState
        backend = AudioBackend(stall=True)
        monkeypatch.setattr(piper_tts, 'sd', backend)
        handler = ResponseHandler(PCMEngine())
        task = asyncio.create_task(handler.aspeak('Synthetic response'))
        async with asyncio.timeout(1):
            while not backend.created.is_set():
                await asyncio.sleep(0.001)
        assert handler.is_speaking and handler.is_responding
        assert await handler.cancel()
        assert (await task).delivery is DeliveryState.CANCELLED
        assert backend.outputs[0].aborted.is_set()
        assert not await handler.cancel()
        assert not handler.is_speaking

    @pytest.mark.asyncio
    async def test_failure_is_explicit_and_clears_owner(self):
        from tests.mocks.speech import PCMEngine
        from src.pipeline.response_handler import DeliveryState
        engine = PCMEngine()
        engine.failure = RuntimeError('synthetic TTS failure')
        handler = ResponseHandler(engine, streaming_enabled=False)
        result = await handler.aspeak('Synthetic response')
        assert result.delivery is DeliveryState.FAILED
        assert not result.success and not handler.is_responding and not handler.is_speaking


class TestResponseHandlerCancel:
    @pytest.mark.asyncio
    async def test_idle_cancel_is_idempotent(self):
        handler = ResponseHandler(tts=MagicMock())
        assert not await handler.cancel()
        assert not await handler.cancel()


class TestBargeInAccumulator:
    """Verificar el acumulador `barge_in_accum_ms` en RoomStream."""

    def test_accumulator_defaults_to_zero(self):
        """RoomStream arranca con accum_ms en 0."""
        rs = _make_room_stream()
        assert rs.barge_in_accum_ms == 0.0

    def test_barge_in_accumulator_resets_on_silence(self):
        """
        El acumulador decae (−20ms por chunk) cuando el audio es silencio
        (RMS < threshold), protegiendo contra triggers espurios.
        """
        response_handler = _make_response_handler_mock(is_speaking=True)
        loop = _make_loop(
            response_handler=response_handler,
            barge_in_min_duration_ms=500,  # alto para no triggerar
        )
        rs = loop.room_streams["escritorio"]

        # Primero acumulamos con audio alto
        callback = loop._make_audio_callback(rs)

        loud_audio = np.full((320, 1), 0.5, dtype=np.float32)
        callback(loud_audio, frames=320, time_info=None, status=MagicMock(output_underflow=False))

        # 320 frames @ 16kHz = 20ms; debió acumular
        assert rs.barge_in_accum_ms > 0
        prev = rs.barge_in_accum_ms

        # Ahora silencio: accum debe decaer (-20ms por chunk)
        silent_audio = np.zeros((320, 1), dtype=np.float32)
        callback(silent_audio, frames=320, time_info=None, status=MagicMock(output_underflow=False))

        assert rs.barge_in_accum_ms < prev
        # Un chunk más de silencio → llega a 0 (floor)
        callback(silent_audio, frames=320, time_info=None, status=MagicMock(output_underflow=False))
        assert rs.barge_in_accum_ms == 0.0


# =========================================================================
# 4. Barge-in trigger after min duration
# =========================================================================

class TestBargeInTrigger:
    """Verificar que voz sostenida ≥ min_duration_ms dispara cancel."""

    def test_barge_in_triggers_after_min_duration(self):
        """
        Audio alto sostenido durante TTS activo, acumulando más de
        `min_duration_ms`, debe schedular `_trigger_barge_in`.
        """
        response_handler = _make_response_handler_mock(is_speaking=True)
        loop = _make_loop(
            response_handler=response_handler,
            barge_in_min_duration_ms=100,  # 100ms = 5 chunks de 20ms
            barge_in_rms_threshold=0.02,
        )
        rs = loop.room_streams["escritorio"]

        # Simular que el loop ya arrancó — capturamos un event loop falso
        scheduled_coroutines = []

        class _FakeLoop:
            def call_soon_threadsafe(self, callback, *args):
                callback(*args)

        # Evitamos el run_coroutine_threadsafe real (necesita un loop
        # corriendo en otro thread). Reemplazamos por un recorder.
        import src.pipeline.multi_room_audio_loop as mral
        original_run_threadsafe = mral.asyncio.run_coroutine_threadsafe

        def _fake_run_threadsafe(coro, loop):
            scheduled_coroutines.append(coro)
            # Cerramos la coro para evitar "coroutine never awaited" warnings.
            coro.close()
            return MagicMock()

        mral.asyncio.run_coroutine_threadsafe = _fake_run_threadsafe
        loop._loop = MagicMock()  # cualquier valor no-None basta

        try:
            callback = loop._make_audio_callback(rs)

            # 5 chunks × 20ms = 100ms → debería disparar barge-in
            loud_audio = np.full((320, 1), 0.5, dtype=np.float32)
            status = MagicMock(output_underflow=False)
            for _ in range(5):
                callback(loud_audio, frames=320, time_info=None, status=status)

            # Verificar que se scheduló al menos un _trigger_barge_in
            assert len(scheduled_coroutines) >= 1
            # El accumulator se resetea después del trigger
            assert rs.barge_in_accum_ms == 0.0
        finally:
            mral.asyncio.run_coroutine_threadsafe = original_run_threadsafe

    def test_barge_in_ignored_when_not_speaking(self):
        """
        Sin TTS activo (`is_speaking=False`), el barge-in check no corre
        y el accum no incrementa.
        """
        response_handler = _make_response_handler_mock(is_speaking=False)
        loop = _make_loop(
            response_handler=response_handler,
            barge_in_min_duration_ms=100,
        )
        rs = loop.room_streams["escritorio"]

        callback = loop._make_audio_callback(rs)
        loud_audio = np.full((320, 1), 0.5, dtype=np.float32)
        status = MagicMock(output_underflow=False)

        for _ in range(10):
            callback(loud_audio, frames=320, time_info=None, status=status)

        # Nunca se acumuló — el check falló en `is_speaking`
        assert rs.barge_in_accum_ms == 0.0
        response_handler.cancel.assert_not_called()

    def test_barge_in_disabled_does_not_trigger(self):
        """Con `barge_in_enabled=False`, el check se saltea."""
        response_handler = _make_response_handler_mock(is_speaking=True)
        loop = _make_loop(
            response_handler=response_handler,
            barge_in_enabled=False,
            barge_in_min_duration_ms=100,
        )
        rs = loop.room_streams["escritorio"]

        callback = loop._make_audio_callback(rs)
        loud_audio = np.full((320, 1), 0.5, dtype=np.float32)
        status = MagicMock(output_underflow=False)

        # Muchos chunks → si estuviera activo, triggeraría muchas veces
        for _ in range(10):
            callback(loud_audio, frames=320, time_info=None, status=status)

        assert rs.barge_in_accum_ms == 0.0


# =========================================================================
# 5. _trigger_barge_in logic
# =========================================================================

class TestTriggerBargeIn:
    """Tests para `_trigger_barge_in` — el handler async del cancel+listen."""

    @pytest.mark.asyncio
    async def test_trigger_barge_in_cancels_tts_and_opens_listening(self):
        """cancel() → True abre listening en el room."""
        response_handler = _make_response_handler_mock(is_speaking=True)
        response_handler.cancel = AsyncMock(return_value=True)

        loop = _make_loop(response_handler=response_handler)
        rs = loop.room_streams["escritorio"]

        assert rs.listening is False
        await loop._trigger_barge_in(rs)

        response_handler.cancel.assert_awaited_once()
        assert rs.listening is True
        assert rs.audio_buffer == []
        loop.follow_up.start_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_barge_in_skips_if_not_speaking(self):
        """
        Si cancel() retorna False (race: TTS ya terminó), NO abrimos
        listening — evita capturas espurias.
        """
        response_handler = _make_response_handler_mock()
        response_handler.cancel = AsyncMock(return_value=False)

        loop = _make_loop(response_handler=response_handler)
        rs = loop.room_streams["escritorio"]

        await loop._trigger_barge_in(rs)

        response_handler.cancel.assert_awaited_once()
        assert rs.listening is False
        loop.follow_up.start_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_barge_in_no_response_handler(self):
        """Sin response_handler attached, trigger es no-op."""
        loop = _make_loop(response_handler=None)
        rs = loop.room_streams["escritorio"]

        # No debe crashear
        await loop._trigger_barge_in(rs)
        assert rs.listening is False


# =========================================================================
# 6. attach_response_handler (post-init DI)
# =========================================================================

class TestAttachResponseHandler:
    """Verificar el patrón de DI tardía para el response_handler."""

    def test_attach_response_handler_sets_reference(self):
        """`attach_response_handler` inyecta el handler post-construcción."""
        loop = _make_loop(response_handler=None)
        assert loop._response_handler is None

        rh = _make_response_handler_mock()
        loop.attach_response_handler(rh)

        assert loop._response_handler is rh
