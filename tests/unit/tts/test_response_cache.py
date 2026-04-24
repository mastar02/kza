"""Tests para el TTS response cache (S2)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# sounddevice no está disponible en macOS de dev local — mockearlo antes
# de cualquier import que lo pueda tocar indirectamente.
sys.modules.setdefault("sounddevice", MagicMock())

from src.tts.response_cache import (
    CANONICAL_PHRASES,
    TEMPLATES,
    CachedAudio,
    ResponseCache,
    _normalize_key,
)


# =========================================================================
# _normalize_key
# =========================================================================

class TestNormalizeKey:
    """Lookup case-insensitive + sin puntuación."""

    def test_strips_trailing_punct(self):
        assert _normalize_key("Listo!") == "listo"
        assert _normalize_key("Listo.") == "listo"

    def test_lowercases(self):
        assert _normalize_key("LISTO") == "listo"
        assert _normalize_key("Listo") == "listo"

    def test_strips_spanish_punct(self):
        # Signos de apertura/cierre español.
        assert _normalize_key("¿Cómo dijiste?") == "cómo dijiste"
        assert _normalize_key("¡Dale!") == "dale"

    def test_collapses_whitespace(self):
        assert _normalize_key("  Prendo    la    luz   ") == "prendo la luz"

    def test_preserves_accents(self):
        # baño (ñ) vs bano — deben ser distintos para no colapsar
        # habitaciones distintas en el lookup.
        assert _normalize_key("baño") != _normalize_key("bano")

    def test_handles_commas(self):
        assert _normalize_key("Perdón, no entendí") == "perdón no entendí"


# =========================================================================
# ResponseCache.build / get / size
# =========================================================================

def _make_mock_tts(audio_samples: int = 12000, as_tuple: bool = False):
    """Helper: TTS que devuelve np.zeros o una tupla (DualTTS-style)."""
    tts = MagicMock()
    tts.sample_rate = 24000
    audio = np.zeros(audio_samples, dtype=np.float32)
    if as_tuple:
        # DualTTS.synthesize -> (audio, elapsed_ms, engine)
        tts.synthesize = MagicMock(return_value=(audio, 12.3, "kokoro"))
    else:
        tts.synthesize = MagicMock(return_value=audio)
    # Asegurar que fallback methods no estén presentes
    del tts.synthesize_to_array
    del tts.generate
    return tts


@pytest.mark.asyncio
async def test_build_populates_cache():
    """build() sintetiza todas las CANONICAL_PHRASES + TEMPLATES."""
    tts = _make_mock_tts()
    cache = ResponseCache(tts)
    await cache.build()

    expected = len(CANONICAL_PHRASES) + sum(len(v) for v in TEMPLATES.values())
    assert cache.size() == expected
    # sanity: al menos 12 canónicas
    assert cache.size() >= 12
    # El TTS fue llamado una vez por frase.
    assert tts.synthesize.call_count == expected


@pytest.mark.asyncio
async def test_get_hit_returns_cached_audio():
    """Una frase cacheada se devuelve como CachedAudio con ndarray float32."""
    tts = _make_mock_tts(audio_samples=24000)  # 1s @ 24kHz
    cache = ResponseCache(tts)
    await cache.build()

    cached = cache.get("Listo")
    assert cached is not None
    assert isinstance(cached, CachedAudio)
    assert isinstance(cached.audio, np.ndarray)
    assert cached.audio.dtype == np.float32
    assert cached.sample_rate == 24000
    # 24000 samples / 24000 = 1.0s
    assert abs(cached.duration_s - 1.0) < 0.01
    assert cached.text == "Listo"


@pytest.mark.asyncio
async def test_get_is_case_insensitive_and_punct_insensitive():
    """Lookup ignora puntuación y caps."""
    tts = _make_mock_tts()
    cache = ResponseCache(tts)
    await cache.build()

    # "Listo" fue cacheado — probar variantes.
    assert cache.get("Listo") is not None
    assert cache.get("listo") is not None
    assert cache.get("LISTO!") is not None
    assert cache.get("  listo.  ") is not None


@pytest.mark.asyncio
async def test_get_miss_returns_none():
    """Cache miss → None. Fallback a TTS normal es responsabilidad del caller."""
    tts = _make_mock_tts()
    cache = ResponseCache(tts)
    await cache.build()

    assert cache.get("frase nunca cacheada 42") is None
    assert cache.get("poné la luz en fucsia al 13 por ciento") is None


def test_templates_cover_all_rooms():
    """Los templates deben incluir los 6 rooms del proyecto."""
    expected_rooms = {"escritorio", "living", "cocina", "baño", "hall", "cuarto"}

    for template_name, phrases in TEMPLATES.items():
        rooms_in_template = set()
        for phrase in phrases:
            for room in expected_rooms:
                if room in phrase.lower():
                    rooms_in_template.add(room)
        assert rooms_in_template == expected_rooms, (
            f"Template {template_name!r} no cubre todos los rooms: "
            f"faltan {expected_rooms - rooms_in_template}"
        )


@pytest.mark.asyncio
async def test_synthesize_handles_tuple_return():
    """DualTTS.synthesize devuelve (audio, elapsed_ms, engine) — debe extraer ndarray."""
    tts = _make_mock_tts(as_tuple=True)
    cache = ResponseCache(tts)
    await cache.build()

    cached = cache.get("Dale")
    assert cached is not None
    assert isinstance(cached.audio, np.ndarray)
    assert cached.audio.dtype == np.float32


@pytest.mark.asyncio
async def test_synthesize_prefers_synthesize_to_array():
    """Si el TTS expone synthesize_to_array, lo intenta primero."""
    tts = MagicMock()
    tts.sample_rate = 24000
    audio = np.ones(1000, dtype=np.float32)
    tts.synthesize_to_array = MagicMock(return_value=audio)
    tts.synthesize = MagicMock(return_value=np.zeros(1000, dtype=np.float32))

    cache = ResponseCache(tts)
    await cache.build()

    # Debe haber usado synthesize_to_array (valor=1.0), no synthesize (0.0).
    cached = cache.get("Listo")
    assert cached is not None
    assert cached.audio[0] == 1.0
    assert tts.synthesize_to_array.call_count > 0


@pytest.mark.asyncio
async def test_build_handles_exception_gracefully():
    """Si una frase falla, las demás siguen cacheándose."""
    tts = MagicMock()
    tts.sample_rate = 24000
    call_count = {"n": 0}

    def flaky_synth(text):
        call_count["n"] += 1
        # Falla la segunda llamada pero continúa el resto
        if call_count["n"] == 2:
            raise RuntimeError("kokoro glitch")
        return np.zeros(12000, dtype=np.float32)

    tts.synthesize = MagicMock(side_effect=flaky_synth)
    del tts.synthesize_to_array
    del tts.generate

    cache = ResponseCache(tts)
    await cache.build()

    expected_total = len(CANONICAL_PHRASES) + sum(len(v) for v in TEMPLATES.values())
    # Todas menos 1 (la que falló).
    assert cache.size() == expected_total - 1


@pytest.mark.asyncio
async def test_size_zero_before_build():
    """Antes de build(), el cache está vacío."""
    tts = _make_mock_tts()
    cache = ResponseCache(tts)
    assert cache.size() == 0
    assert cache.get("Listo") is None
