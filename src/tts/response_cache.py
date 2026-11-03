"""
Cache de respuestas TTS pre-generadas.

Al startup, sintetiza un banco de frases canónicas con el motor TTS (Kokoro
por default en KZA, 24kHz float32) y las mantiene en RAM como np.ndarray.
En runtime, `ResponseHandler` consulta el cache antes del stream TTS en vivo
y, si hay match, reproduce directo desde RAM (~5-10ms vs ~80ms de Kokoro).

Patrón Alexa/Google: pre-generan miles de respuestas comunes. Acá adaptamos
con ~30 frases canónicas + templates por room — suficiente para cubrir el
grueso del fast path de domótica.
"""

from __future__ import annotations

import asyncio
from contextvars import Context, ContextVar
import logging
import threading
import weakref
import re
import time
import wave as _wave
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

import numpy as np

from src.core.owned_work import OwnedBlockingWork

logger = logging.getLogger(__name__)


# Frases canónicas cortas — acknowledgements genéricos del asistente.
CANONICAL_PHRASES: list[str] = [
    "Listo",
    "Dale",
    "Hecho",
    "Perdón, no entendí",
    "Un segundo",
    "No pude hacerlo",
    "Ya está",
    "Cómo dijiste",
    "Sí",
    "No",
    "OK",
    "Nada más",
]

# Templates con slot de room — se sintetiza cada combinación una única vez.
# Los rooms deben mantenerse consistentes con `src/rooms/room_context.py` y
# el naming HA (ver decisiones.md → ha_light_naming_convention.md).
TEMPLATES: dict[str, list[str]] = {
    "turn_on_light_room": [
        "Prendo la luz del escritorio",
        "Prendo la luz del living",
        "Prendo la luz de la cocina",
        "Prendo la luz del baño",
        "Prendo la luz del hall",
        "Prendo la luz del cuarto",
    ],
    "turn_off_light_room": [
        "Apago la luz del escritorio",
        "Apago la luz del living",
        "Apago la luz de la cocina",
        "Apago la luz del baño",
        "Apago la luz del hall",
        "Apago la luz del cuarto",
    ],
}


@dataclass
class CachedAudio:
    """Audio pre-generado listo para playback directo."""

    audio: np.ndarray
    sample_rate: int
    duration_s: float
    text: str


def _normalize_key(text: str) -> str:
    """Normaliza un texto para lookup case-insensitive y sin puntuación.

    Colapsa espacios múltiples, elimina signos no alfanuméricos (incluye
    `¿ ? ! , .`) y baja a lowercase. Respeta los caracteres con acento para
    no perder diferenciación entre "baño" y "bano" (Unicode \\w).
    """
    t = text.lower().strip()
    # Quitar puntuación estándar y caracteres especiales (mantener \w unicode).
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_earcon(path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Cargar un WAV de earcon a float32 mono. Fail-open: (None, None) si falla."""
    try:
        with _wave.open(path, "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr
    except Exception as e:
        logger.warning(f"[earcon] no se pudo cargar {path!r}: {e}")
        return None, None


class _CacheOperation:
    def __init__(self):
        self._lock = threading.Lock()
        self._references = 1

    @property
    def active(self) -> bool:
        with self._lock:
            return self._references > 0

    def retain(self):
        with self._lock:
            if not self._references:
                return None
            self._references += 1
        return _CacheOperationPin(self)

    def release(self) -> None:
        with self._lock:
            self._references -= 1


class _CacheOperationPin:
    def __init__(self, operation):
        self._lock = threading.Lock()
        self._operation = operation

    def release(self) -> None:
        """Release once, after the retaining caller's actual native receipt."""
        with self._lock:
            operation, self._operation = self._operation, None
        if operation is not None:
            operation.release()


class ResponseCache:
    """Cache en RAM de respuestas TTS pre-generadas.

    Uso típico:
        cache = ResponseCache(tts)
        await cache.build()              # startup, ~3-5s

        # runtime:
        cached = cache.get("Listo")
        if cached is not None:
            playback(cached.audio, cached.sample_rate)    # ~5ms
        else:
            tts.speak(text)              # fallback camino normal
    """

    def __init__(self, tts, sample_rate: int = 24000, *,
                 synthesize_callback: Callable[[str], Awaitable[object]] | None = None):
        """Crear cache.

        Args:
            tts: Instancia TTS con alguno de: `synthesize_to_array`,
                `synthesize`, `generate`. Puede retornar np.ndarray directo
                o tupla `(audio, elapsed, engine)` (DualTTS) — tomamos el
                primer elemento ndarray.
            sample_rate: Sample rate a guardar en `CachedAudio`. Kokoro
                produce 24kHz. Si el TTS expone `.sample_rate`, se respeta.
            synthesize_callback: Shared owned synthesis for application startup;
                a failed/cancelled call never falls through to an unowned engine.
        """
        self._tts = tts
        self._synthesize_callback = synthesize_callback
        # Respetar sample_rate del TTS si está disponible (DualTTS y Kokoro
        # ambos exponen .sample_rate = 24000).
        self._sample_rate = getattr(tts, "sample_rate", sample_rate)
        self._cache: dict[str, CachedAudio] = {}
        self._state_lock = threading.Lock()
        self._closed = False
        self._work = OwnedBlockingWork(capacity=1, isolate_context=True)
        self._building: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._close_owner_thread = None
        self._close_context = ContextVar("response_cache_operation", default=None)
        self._sequence_context = ContextVar("response_cache_sequence", default=None)

    async def build(self) -> None:
        """Retain a complete build; overlapping observers join, later calls rebuild."""
        operation = self._close_context.get()
        if operation is not None and operation.active:
            raise RuntimeError("response_cache_build_from_operation")
        task = None
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("response_cache_closed")
                if self._building is None or self._building.done():
                    async def invoke():
                        await asyncio.sleep(0)
                        if self._is_closed():
                            return
                        await self._work.run_async(self._run_owned, self._build_owned)
                    self._building = asyncio.create_task(invoke(), context=Context())
                    self._building.add_done_callback(self._build_finished, context=Context())
                task = self._building
            await asyncio.shield(task)
        finally:
            task = None

    async def _synthesize(self, text: str):
        """Preserve callback/fallback ordering inside a retained actual sequence."""
        reference = self._sequence_context.get()
        if reference is not None and reference() is asyncio.current_task():
            return await self._synthesize_owned(text)
        self._require_open()
        try:
            return await self._work.run_async(self._run_owned, self._synthesize_owned, text)
        finally:
            text = None

    def _is_closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def _require_open(self) -> None:
        if self._is_closed():
            raise RuntimeError("response_cache_closed")

    def require_close_caller(self) -> None:
        """Reject self-retirement without sealing or creating any task.

        Context copies keep the origin marker while a callback's actual native
        worker outlives an early CancelledError from its async observer.
        """
        operation = self._close_context.get()
        if operation is not None and operation.active:
            raise RuntimeError("response_cache_close_from_operation")
        with self._state_lock:
            if self._close_owner_thread is threading.current_thread():
                raise RuntimeError("response_cache_close_from_owner")
            closing = self._closing
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if current is not None and current is closing:
            raise RuntimeError("response_cache_close_from_owner")

    def request_close(self) -> None:
        """Seal admission/publication using memory only."""
        with self._state_lock:
            self._closed = True

    def retain_current_operation(self):
        """Pin inherited guard metadata until an external native receipt exits.

        Capture before dispatch; release on actual native completion or failed
        dispatch. Pins contain no cache, task, TTS or PCM references.
        """
        operation = self._close_context.get()
        return None if operation is None else operation.retain()

    def _build_finished(self, task) -> None:
        if not task.cancelled():
            task.exception()
        with self._state_lock:
            if self._building is task:
                self._building = None

    async def _run_owned(self, function, *args):
        operation = _CacheOperation()
        close_token = self._close_context.set(operation)
        sequence_token = self._sequence_context.set(weakref.ref(asyncio.current_task()))
        try:
            return await function(*args)
        finally:
            function, args = None, ()
            operation.release()
            try:
                # A callback may finish observing before its actual native
                # worker. Its pins retain this sequence until real exit.
                while operation.active:
                    try:
                        await asyncio.sleep(0.005)
                    except asyncio.CancelledError:
                        continue
            finally:
                self._sequence_context.reset(sequence_token)
                self._close_context.reset(close_token)

    async def _build_owned(self) -> None:
        audio = record = None
        t0 = time.time()
        phrases: list[str] = list(CANONICAL_PHRASES)
        for template_phrases in TEMPLATES.values():
            phrases.extend(template_phrases)
        try:
            for phrase in phrases:
                if self._is_closed():
                    return
                try:
                    audio = await self._synthesize(phrase)
                    if self._is_closed():
                        return
                    if audio is None or len(audio) == 0:
                        logger.warning(f"TTS cache: audio vacío para {phrase!r}, se omite")
                        continue
                    key = _normalize_key(phrase)
                    record = CachedAudio(audio=audio, sample_rate=self._sample_rate,
                                         duration_s=len(audio) / self._sample_rate, text=phrase)
                    with self._state_lock:
                        if self._closed:
                            return
                        self._cache[key] = record
                except asyncio.CancelledError:
                    if self._is_closed():
                        return
                    raise
                except Exception as error:
                    logger.warning(f"TTS cache: fallo sintetizando {phrase!r}: {error}")
                finally:
                    audio = record = None
            elapsed = time.time() - t0
            logger.info(f"TTS response cache listo: {self.size()} frases cacheadas en {elapsed:.1f}s")
        finally:
            audio = record = phrases = None

    def _call_native(self, function, text):
        try:
            return None if self._is_closed() else function(text)
        finally:
            function = text = None

    def _convert_audio(self, result):
        audio = None
        try:
            audio = self._extract_ndarray(result)
            return audio.astype(np.float32) if audio is not None else None
        finally:
            result = audio = None

    async def _synthesize_owned(self, text: str):
        tts = callback = function = result = audio = None
        try:
            with self._state_lock:
                if self._closed:
                    return None
                tts, callback = self._tts, self._synthesize_callback
            if callback is not None:
                result = await callback(text)
                if self._is_closed():
                    return None
                audio = await self._work.native_step(self._convert_audio, result)
                return None if self._is_closed() else audio
            for method_name in ("synthesize_to_array", "synthesize", "generate"):
                if self._is_closed():
                    return None
                function = getattr(tts, method_name, None)
                if function is None:
                    continue
                try:
                    if asyncio.iscoroutinefunction(function):
                        result = await function(text)
                    else:
                        result = await self._work.native_step(self._call_native, function, text)
                    if asyncio.iscoroutine(result):
                        if self._is_closed():
                            result.close()
                            result = None
                            return None
                        result = await result
                except Exception as error:
                    logger.warning(f"TTS cache: método {method_name!r} falló para {text!r}: "
                                   f"{type(error).__name__}: {error}", exc_info=True)
                    continue
                if self._is_closed():
                    return None
                audio = await self._work.native_step(self._convert_audio, result)
                result = None
                if self._is_closed():
                    return None
                if audio is not None:
                    return audio
            return None
        finally:
            tts = callback = function = result = audio = text = None

    def _drop_references(self) -> None:
        """Dispose own references off-loop, without locks across destructors."""
        with self._state_lock:
            self._close_owner_thread = threading.current_thread()
            retired = (self._cache, self._tts, self._synthesize_callback)
            self._cache = {}
            self._tts = self._synthesize_callback = None
        try:
            retired[0].clear()
        finally:
            retired = None
            with self._state_lock:
                self._close_owner_thread = None

    def begin_close(self) -> asyncio.Task:
        """Retain actual build/native/callback drain before dropping own data.

        Owners must first drain external playback borrowers. Returned arrays
        are never mutated, and the borrowed TTS engine is never closed here.
        """
        self.require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._state_lock:
            if self._closing is None:
                building = self._building
                native = self._work.begin_close()

                async def finish():
                    await asyncio.sleep(0)
                    if building is not None:
                        await asyncio.gather(building, return_exceptions=True)
                    await asyncio.shield(native)
                    await asyncio.to_thread(self._drop_references)

                self._closing = loop.create_task(finish(), context=Context())
                self._closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(), context=Context())
            return self._closing

    async def aclose(self) -> None:
        """Wait for actual retirement through repeated observer cancellation."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _extract_ndarray(result):
        """Extraer np.ndarray de un resultado TTS arbitrario.

        Acepta: ndarray directo o tupla/lista cuyo primer elemento ndarray
        se toma como audio (resto son elapsed/engine metadata).
        """
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, np.ndarray):
                    return item
        return None

    def get(self, text: str) -> CachedAudio | None:
        """Return a borrowed cached record, or a miss after terminal sealing."""
        if self._is_closed():
            return None
        key = _normalize_key(text)
        with self._state_lock:
            return None if self._closed else self._cache.get(key)

    def size(self) -> int:
        """Count entries retained by the cache; zero after terminal disposal."""
        with self._state_lock:
            return len(self._cache)
