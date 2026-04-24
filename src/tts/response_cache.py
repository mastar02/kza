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
import logging
import re
import time
from dataclasses import dataclass

import numpy as np

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

    def __init__(self, tts, sample_rate: int = 24000):
        """Crear cache.

        Args:
            tts: Instancia TTS con alguno de: `synthesize_to_array`,
                `synthesize`, `generate`. Puede retornar np.ndarray directo
                o tupla `(audio, elapsed, engine)` (DualTTS) — tomamos el
                primer elemento ndarray.
            sample_rate: Sample rate a guardar en `CachedAudio`. Kokoro
                produce 24kHz. Si el TTS expone `.sample_rate`, se respeta.
        """
        self._tts = tts
        # Respetar sample_rate del TTS si está disponible (DualTTS y Kokoro
        # ambos exponen .sample_rate = 24000).
        self._sample_rate = getattr(tts, "sample_rate", sample_rate)
        self._cache: dict[str, CachedAudio] = {}

    async def build(self) -> None:
        """Sintetiza cada frase canónica + template y la guarda. Llamar al startup."""
        t0 = time.time()
        phrases: list[str] = list(CANONICAL_PHRASES)
        for template_phrases in TEMPLATES.values():
            phrases.extend(template_phrases)

        for phrase in phrases:
            try:
                audio = await self._synthesize(phrase)
                if audio is None or len(audio) == 0:
                    logger.warning(
                        f"TTS cache: audio vacío para {phrase!r}, se omite"
                    )
                    continue
                key = _normalize_key(phrase)
                self._cache[key] = CachedAudio(
                    audio=audio,
                    sample_rate=self._sample_rate,
                    duration_s=len(audio) / self._sample_rate,
                    text=phrase,
                )
            except Exception as e:
                # No tirar al startup por una frase rota — solo skippear.
                logger.warning(f"TTS cache: fallo sintetizando {phrase!r}: {e}")

        elapsed = time.time() - t0
        logger.info(
            f"TTS response cache listo: {len(self._cache)} frases "
            f"cacheadas en {elapsed:.1f}s"
        )

    async def _synthesize(self, text: str):
        """Llamar al TTS intentando varios métodos compatibles.

        Prueba `synthesize_to_array`, `synthesize` y `generate` en ese
        orden. Soporta:
        - Retorno np.ndarray directo.
        - Retorno tupla `(audio, elapsed_ms)` (Kokoro).
        - Retorno tupla `(audio, elapsed_ms, engine)` (DualTTS).
        - Método sync o async.
        """
        for method_name in ("synthesize_to_array", "synthesize", "generate"):
            fn = getattr(self._tts, method_name, None)
            if fn is None:
                continue
            try:
                result = fn(text)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as e:
                logger.debug(
                    f"TTS cache: método {method_name!r} falló para "
                    f"{text!r}: {e}"
                )
                continue

            audio = self._extract_ndarray(result)
            if audio is not None:
                return audio.astype(np.float32)
        return None

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
        """Retorna el audio cacheado si existe, None si no (cache miss)."""
        return self._cache.get(_normalize_key(text))

    def size(self) -> int:
        """Número de frases cacheadas."""
        return len(self._cache)
