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

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# StrEnum shim para Python 3.9 (StrEnum llega en 3.11). El server corre 3.13
# y no lo necesita, pero el dev local sí.
import enum as _enum

if not hasattr(_enum, "StrEnum"):
    class _StrEnumShim(str, _enum.Enum):
        pass
    _enum.StrEnum = _StrEnumShim

# Mock system-level modules ANTES de cualquier import — sounddevice/torch/etc
# no están disponibles en macOS local y romperían la carga de los módulos.
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())


# El paquete `src.pipeline.__init__` carga transitivamente muchos módulos que
# usan sintaxis de tipos de Python 3.10+ sin `from __future__ import
# annotations` (e.g. `src.conversation.follow_up_mode` con `str | None` a
# nivel de clase). En Python 3.13 (server) no hay problema, pero macOS local
# tiene Python 3.9. Para que este test corra en ambos:
#   1. Stubbeamos `src.pipeline` y las dependencias de MultiRoomAudioLoop
#      que no necesitamos ejercitar (audio_loop, conversation, etc).
#   2. Cargamos `multi_room_audio_loop.py` y `response_handler.py` directo
#      del archivo vía importlib, saltando `__init__.py`.

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Registra un módulo fake en sys.modules si aún no existe."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# src.audio — solo necesitamos EchoSuppressor (mock)
_stub_module(
    "src.audio",
    EchoSuppressor=MagicMock,
    ZoneManager=MagicMock,
    Zone=MagicMock,
)
_stub_module(
    "src.audio.echo_suppressor",
    EchoSuppressor=MagicMock,
)

# src.conversation — FollowUpMode (mock)
_stub_module(
    "src.conversation",
    FollowUpMode=MagicMock,
    ConversationState=MagicMock,
    ConversationContext=MagicMock,
)
_stub_module(
    "src.conversation.follow_up_mode",
    FollowUpMode=MagicMock,
    ConversationState=MagicMock,
    ConversationContext=MagicMock,
)

# src.wakeword — WakeWordDetector (mock)
_stub_module("src.wakeword", WakeWordDetector=MagicMock)
_stub_module("src.wakeword.detector", WakeWordDetector=MagicMock)

# src.llm.buffered_streamer — usado por response_handler
_stub_module(
    "src.llm.buffered_streamer",
    BufferedLLMStreamer=MagicMock,
    BufferConfig=MagicMock,
    create_buffered_streamer=MagicMock(),
)

# src.tts.response_cache — usado por response_handler
_stub_module(
    "src.tts.response_cache",
    CachedAudio=MagicMock,
    ResponseCache=MagicMock,
)

# src.pipeline.command_event
_stub_module("src.pipeline.command_event", CommandEvent=MagicMock)

# src.nlu.command_grammar — el módulo real es seguro de importar; dejamos
# intentar el import real antes de stubear.
try:
    import src.nlu.command_grammar  # noqa: F401
except Exception:
    _stub_module(
        "src.nlu.command_grammar",
        PartialCommand=MagicMock,
        parse_partial_command=MagicMock(),
    )


def _load_source_module(dotted_name: str, rel_path: str) -> types.ModuleType:
    """Carga un .py directo con importlib sin pasar por el __init__ del paquete."""
    # Asegurar que el paquete padre exista como módulo stubbeado.
    parent_name = dotted_name.rsplit(".", 1)[0]
    if parent_name not in sys.modules:
        _stub_module(parent_name)
    spec = importlib.util.spec_from_file_location(
        dotted_name, str(_REPO_ROOT / rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = module
    spec.loader.exec_module(module)
    # Exponer como submódulo del padre para imports encadenados.
    setattr(sys.modules[parent_name], dotted_name.rsplit(".", 1)[1], module)
    return module


# Cargar los dos módulos que realmente queremos ejercitar.
_mral = _load_source_module(
    "src.pipeline.multi_room_audio_loop",
    "src/pipeline/multi_room_audio_loop.py",
)
_rh = _load_source_module(
    "src.pipeline.response_handler",
    "src/pipeline/response_handler.py",
)

MultiRoomAudioLoop = _mral.MultiRoomAudioLoop
RoomStream = _mral.RoomStream
ResponseHandler = _rh.ResponseHandler

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402


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
    rh = MagicMock()
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
    """Verificar el flag `is_speaking` antes/durante/después de speak."""

    def test_is_speaking_false_at_init(self):
        """Arranca en False; cancel() sin TTS activo retorna False."""
        tts = MagicMock()
        tts.sample_rate = 16000
        handler = ResponseHandler(tts=tts, streaming_enabled=False)

        assert handler.is_speaking is False

    def test_response_handler_is_speaking_flag(self):
        """
        `is_speaking` está True durante el speak y False al terminar.

        Usamos un side_effect en `tts.speak` para capturar el flag durante
        la ejecución (no sólo antes/después).
        """
        tts = MagicMock()
        tts.sample_rate = 16000
        # Sin synthesize_stream → cae a _speak_direct sin streaming
        if hasattr(tts, "synthesize_stream"):
            del tts.synthesize_stream
        if hasattr(tts, "speak_stream"):
            del tts.speak_stream

        handler = ResponseHandler(
            tts=tts,
            zone_manager=None,  # fuerza _speak_direct
            streaming_enabled=False,
        )

        seen_during = {"value": None}

        def _capture_flag(text):
            seen_during["value"] = handler.is_speaking

        tts.speak.side_effect = _capture_flag

        assert handler.is_speaking is False
        handler.speak("Hola")

        # Durante el speak, el flag estaba True.
        assert seen_during["value"] is True
        # Al terminar, False (el finally lo resetea).
        assert handler.is_speaking is False

    def test_is_speaking_resets_on_exception(self):
        """Si el TTS lanza, `is_speaking` vuelve a False (finally)."""
        tts = MagicMock()
        tts.sample_rate = 16000
        if hasattr(tts, "synthesize_stream"):
            del tts.synthesize_stream
        if hasattr(tts, "speak_stream"):
            del tts.speak_stream
        tts.speak.side_effect = RuntimeError("TTS crash")

        handler = ResponseHandler(
            tts=tts, zone_manager=None, streaming_enabled=False
        )

        with pytest.raises(RuntimeError):
            handler.speak("Hola")

        assert handler.is_speaking is False


# =========================================================================
# 2. ResponseHandler.cancel() idempotent
# =========================================================================

class TestResponseHandlerCancel:
    """Verificar que cancel() es idempotent y robusto."""

    @pytest.mark.asyncio
    async def test_response_handler_cancel_idempotent(self):
        """Dos llamadas consecutivas a cancel() no fallan ni disparan errores."""
        tts = MagicMock()
        tts.sample_rate = 16000
        handler = ResponseHandler(tts=tts, streaming_enabled=False)

        # 1ra llamada: no había nada que cancelar → False
        result1 = await handler.cancel()
        assert result1 is False

        # 2da llamada: tampoco hay nada → False, no exception
        result2 = await handler.cancel()
        assert result2 is False

    @pytest.mark.asyncio
    async def test_cancel_when_speaking_returns_true(self):
        """Si `is_speaking=True`, cancel() retorna True y apaga el flag."""
        tts = MagicMock()
        tts.sample_rate = 16000
        handler = ResponseHandler(tts=tts, streaming_enabled=False)

        # Simular TTS activo — así sería durante speak()
        handler._is_speaking = True

        result = await handler.cancel()

        assert result is True
        assert handler.is_speaking is False

        # 2da llamada inmediata: idempotent, retorna False sin side effects
        result2 = await handler.cancel()
        assert result2 is False

    @pytest.mark.asyncio
    async def test_cancel_closes_current_stream(self):
        """cancel() llama stop/close sobre `_current_stream` si existe."""
        tts = MagicMock()
        tts.sample_rate = 16000
        handler = ResponseHandler(tts=tts, streaming_enabled=False)

        # Simular un stream activo (ej: StreamingAudioPlayer)
        fake_stream = MagicMock()
        handler._is_speaking = True
        handler._current_stream = fake_stream

        result = await handler.cancel()

        assert result is True
        fake_stream.stop.assert_called_once()
        fake_stream.close.assert_called_once()
        assert handler._current_stream is None

    @pytest.mark.asyncio
    async def test_cancel_cancels_playback_task(self):
        """cancel() cancela el `_playback_task` si está activo."""
        tts = MagicMock()
        tts.sample_rate = 16000
        handler = ResponseHandler(tts=tts, streaming_enabled=False)

        # Simular playback task activo
        async def _long_playback():
            await asyncio.sleep(10)

        task = asyncio.create_task(_long_playback())
        handler._is_speaking = True
        handler._playback_task = task

        result = await handler.cancel()

        assert result is True
        # Dar chance al event loop para que propague el cancel
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()


# =========================================================================
# 3. Barge-in accumulator
# =========================================================================

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
