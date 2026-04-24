# S2 — TTS pre-generated responses

**Effort**: 🟡 2-3 h
**Depende**: S1 (warmup garantiza Kokoro listo para pre-generar en startup)
**Branch sugerido**: `feat/s2-tts-response-cache`

## Objetivo

Respuestas comunes ("Dale", "Listo", "Prendo la luz del escritorio") no deben
pasar por Kokoro TTS en tiempo real. Al startup generamos un banco de frases
canónicas y las servimos desde RAM con latencia cuasi-cero.

## Arquitectura

```
ResponseHandler.speak(text):
  1. Normalizar text → key (lowercase, strip punct).
  2. Buscar en ResponseCache:
     - Match exact → playback del WAV cacheado (~5ms).
     - Match template "Prendo la luz del {room}" → concat de chunks cacheados.
     - No match → stream Kokoro en vivo (camino actual).
```

El cache se inicializa al startup: ~30 frases + templates → 3-5 s total de
síntesis una vez, luego 0 latencia forever.

## Archivos a crear/modificar

### Nuevo: `src/tts/response_cache.py`

```python
"""
Cache de respuestas TTS pre-generadas.

Al startup, sintetiza un banco de frases canónicas con Kokoro y las mantiene
en RAM como np.ndarray (float32 @ 24kHz, sample rate de Kokoro). Al momento
de hablar, ResponseHandler busca el match; si existe, playback directo.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# Frases canónicas — cacheadas al startup
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

# Templates con slot de room — Kokoro sintetiza cada combo una vez.
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
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    text: str


def _normalize_key(text: str) -> str:
    """Lowercase + quitar puntuación + colapsar espacios — para lookup."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class ResponseCache:
    """
    Cache TTS en RAM. Uso:
        cache = ResponseCache(tts)
        await cache.build()   # al startup, ~3-5s

        # runtime:
        audio = cache.get("Listo")   # inmediato (~5ms)
        if audio is None:
            # fallback a stream TTS en vivo
    """

    def __init__(self, tts, sample_rate: int = 24000):
        self._tts = tts
        self._sample_rate = sample_rate
        self._cache: dict[str, CachedAudio] = {}

    async def build(self) -> None:
        """Sintetiza cada frase canónica y la guarda. Llamar al startup."""
        t0 = time.time()
        phrases = list(CANONICAL_PHRASES)
        for template_phrases in TEMPLATES.values():
            phrases.extend(template_phrases)

        for phrase in phrases:
            try:
                audio = await self._synthesize(phrase)
                if audio is None or len(audio) == 0:
                    continue
                key = _normalize_key(phrase)
                self._cache[key] = CachedAudio(
                    audio=audio,
                    sample_rate=self._sample_rate,
                    duration_s=len(audio) / self._sample_rate,
                    text=phrase,
                )
            except Exception as e:
                logger.warning(f"TTS cache: fallo sintetizando {phrase!r}: {e}")

        elapsed = time.time() - t0
        logger.info(
            f"TTS response cache listo: {len(self._cache)} frases "
            f"cacheadas en {elapsed:.1f}s"
        )

    async def _synthesize(self, text: str) -> np.ndarray | None:
        """Llama al TTS pasando el método correcto (sync o async)."""
        import asyncio
        for method_name in ("synthesize_to_array", "synthesize", "generate"):
            fn = getattr(self._tts, method_name, None)
            if fn is None:
                continue
            try:
                result = fn(text)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, tuple):
                    audio = result[0]
                else:
                    audio = result
                if isinstance(audio, np.ndarray):
                    return audio.astype(np.float32)
            except Exception:
                continue
        return None

    def get(self, text: str) -> CachedAudio | None:
        """Devuelve el audio cacheado si existe, None si no."""
        return self._cache.get(_normalize_key(text))

    def size(self) -> int:
        return len(self._cache)
```

### Modificar: `src/pipeline/response_handler.py`

Agregar un campo `_cache: ResponseCache | None` en `__init__`. En el método
`speak` (o equivalente), chequear cache primero:

```python
def __init__(self, tts, ..., response_cache=None):
    ...
    self._cache = response_cache

async def speak(self, text: str):
    if self._cache is not None:
        cached = self._cache.get(text)
        if cached is not None:
            logger.info(f"TTS cache HIT: {text!r} ({cached.duration_s*1000:.0f}ms)")
            await self._playback_audio(cached.audio, cached.sample_rate)
            return
    # camino normal
    ...
```

### Modificar: `src/main.py`

Después de `tts.load()` y del warmup (S1), antes del pipeline:

```python
# TTS response cache (S2)
response_cache = None
tts_cache_cfg = tts_config.get("response_cache", {})
if tts_cache_cfg.get("enabled", False):
    from src.tts.response_cache import ResponseCache
    response_cache = ResponseCache(tts)
    await response_cache.build()

# Al construir ResponseHandler:
response_handler = ResponseHandler(
    tts=tts,
    ...
    response_cache=response_cache,
)
```

### Modificar: `config/settings.yaml`

En la sección `tts:`:
```yaml
tts:
  ...
  response_cache:
    enabled: true   # pre-genera ~30 frases al startup (~3-5s extra)
```

## Validación

1. Restart service → log `TTS response cache listo: 24 frases cacheadas en 4.2s`.
2. Comando "nexa apagá la luz del escritorio" → respuesta "Apago la luz del escritorio" con latencia del lado respuesta <10ms (antes ~80ms stream TTS).
3. Log: `TTS cache HIT: 'Apago la luz del escritorio' (1200ms)`.
4. Comando raro ("nexa poné la luz en fucsia al 13 por ciento") → fallback al TTS normal, camino actual sin regressión.

## Test unitario

`tests/unit/tts/test_response_cache.py`:
```python
from unittest.mock import MagicMock
import numpy as np
import pytest
from src.tts.response_cache import ResponseCache, _normalize_key

def test_normalize_key():
    assert _normalize_key("Listo!") == "listo"
    assert _normalize_key("¿Cómo dijiste?") == "como dijiste"

@pytest.mark.asyncio
async def test_build_populates_cache():
    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=np.zeros(12000, dtype=np.float32))
    cache = ResponseCache(tts)
    await cache.build()
    assert cache.size() >= 12  # al menos las CANONICAL_PHRASES
    audio = cache.get("Listo")
    assert audio is not None
    assert audio.audio.dtype == np.float32

@pytest.mark.asyncio
async def test_get_miss_returns_none():
    tts = MagicMock()
    tts.synthesize = MagicMock(return_value=np.zeros(12000, dtype=np.float32))
    cache = ResponseCache(tts)
    await cache.build()
    assert cache.get("frase nunca cacheada 42") is None
```

## Edge cases

- **Kokoro genera a 24kHz, pipeline espera 16kHz**: resample en `_playback_audio`
  o dejar que el output device lo maneje. Verificar sample rate.
- **VRAM**: 24 frases de ~1s @ 24kHz float32 = 24 * 24000 * 4 = ~2.3 MB. Trivial.
- **Phrase drift**: si el engine cambia (Kokoro update), los WAVs cacheados
  pueden sonar "diferente" a los nuevos. No es problema — todos suenan igual,
  vienen del mismo modelo al startup.
- **Slots dinámicos no cacheados**: cualquier texto con slot no templatizado
  cae al camino normal. Es opt-in vía cache hit; siempre hay fallback.

## Commit message sugerido

```
feat(tts): response cache — frases comunes pre-generadas al startup

Alexa/Google pre-generan ~200 respuestas comunes para latencia cero.
Adaptamos con ~30 frases canónicas + templates por room.

- src/tts/response_cache.py: ResponseCache con CANONICAL_PHRASES y
  TEMPLATES. build() sintetiza al startup, get(text) devuelve CachedAudio.
- ResponseHandler: cache lookup antes del stream TTS normal.
- main.py: init + build post-load de Kokoro.
- settings.yaml: tts.response_cache.enabled (true default).

Latencia de respuesta común cae de ~80ms Kokoro stream a <10ms.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Checklist

- [ ] Crear `src/tts/response_cache.py` con ResponseCache
- [ ] Agregar parámetro `response_cache` a `ResponseHandler.__init__`
- [ ] Modificar `ResponseHandler.speak` (o equivalente) para cache hit
- [ ] Init + build en `src/main.py` post-load de TTS
- [ ] Flag `tts.response_cache.enabled` en settings.yaml
- [ ] Tests en `tests/unit/tts/test_response_cache.py`
- [ ] Correr regression tests existentes
- [ ] Commit + push
