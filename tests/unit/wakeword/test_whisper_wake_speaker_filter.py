"""
Tests para WhisperWakeDetector con speaker filter (Fase 1 del roadmap).

Verifica que:
1. El filter rechaza utterances con similarity baja (TV / otros usuarios).
2. El filter acepta utterances con similarity alta (user enrolado).
3. Audio corto (< speaker_min_audio_s) pasa el filter sin consultar embedding.
4. Sin filter configurado, el pipeline se comporta como antes.
5. Excepción del speaker_identifier no rompe la detección (graceful fallback).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock heavy deps antes de importar el módulo
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

from src.wakeword.whisper_wake import WhisperWakeDetector, SAMPLE_RATE


class _FakeWhisperSTT:
    """Stub de FastWhisperSTT que devuelve transcripción controlada."""
    def __init__(self, text: str):
        self._text = text

    def transcribe(self, audio, **kwargs):
        seg = MagicMock()
        seg.text = self._text
        return [seg], None


def _make_speaker_identifier(sim_returns: float) -> MagicMock:
    spk = MagicMock()
    spk.get_embedding = MagicMock(return_value=np.random.rand(192).astype(np.float32))
    spk.compute_similarity = MagicMock(return_value=sim_returns)
    return spk


def _make_detector(
    whisper_text: str,
    speaker_identifier=None,
    speaker_embedding=None,
    speaker_threshold: float = 0.65,
    speaker_min_audio_s: float = 0.8,
) -> WhisperWakeDetector:
    det = WhisperWakeDetector(
        whisper_stt=_FakeWhisperSTT(whisper_text),
        wake_words=["nexa"],
        speaker_identifier=speaker_identifier,
        speaker_embedding=speaker_embedding,
        speaker_threshold=speaker_threshold,
        speaker_min_audio_s=speaker_min_audio_s,
    )
    det._loaded = True
    det._vad = None  # usa RMS fallback
    det._torch = None
    return det


def _audio(duration_s: float, amplitude: float = 0.1) -> np.ndarray:
    samples = int(duration_s * SAMPLE_RATE)
    return (np.random.randn(samples) * amplitude).astype(np.float32)


# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------

def test_filter_rejects_low_similarity():
    """sim < threshold → la utterance no llega a Whisper."""
    whisper = MagicMock()
    whisper.transcribe = MagicMock()
    spk = _make_speaker_identifier(sim_returns=0.4)
    emb = np.random.rand(192).astype(np.float32)

    det = WhisperWakeDetector(
        whisper_stt=whisper, wake_words=["nexa"],
        speaker_identifier=spk, speaker_embedding=emb,
        speaker_threshold=0.65, speaker_min_audio_s=0.8,
    )

    audio = _audio(1.5)
    passed, sim = det._speaker_match(audio)
    assert passed is False
    assert sim == pytest.approx(0.4)
    spk.get_embedding.assert_called_once()
    whisper.transcribe.assert_not_called()


def test_filter_accepts_high_similarity():
    """sim >= threshold → la utterance pasa al pipeline."""
    spk = _make_speaker_identifier(sim_returns=0.85)
    emb = np.random.rand(192).astype(np.float32)

    det = _make_detector(
        whisper_text="nexa prendé la luz",
        speaker_identifier=spk, speaker_embedding=emb,
        speaker_threshold=0.65,
    )

    passed, sim = det._speaker_match(_audio(1.5))
    assert passed is True
    assert sim == pytest.approx(0.85)


def test_short_audio_bypasses_filter():
    """Audio < speaker_min_audio_s → filter devuelve True sin consultar embedding."""
    spk = _make_speaker_identifier(sim_returns=0.0)
    emb = np.random.rand(192).astype(np.float32)

    det = _make_detector(
        whisper_text="nexa",
        speaker_identifier=spk, speaker_embedding=emb,
        speaker_min_audio_s=0.8,
    )

    short_audio = _audio(0.3)  # 300ms, menos que 800ms
    passed, sim = det._speaker_match(short_audio)
    assert passed is True
    assert sim == 0.0
    spk.get_embedding.assert_not_called()


def test_no_filter_when_disabled():
    """Sin speaker_identifier/embedding → _speaker_filter_active=False, sim fija 1.0."""
    det = _make_detector(whisper_text="nexa")
    assert det._speaker_filter_active is False
    passed, sim = det._speaker_match(_audio(2.0))
    assert passed is True
    assert sim == 1.0


def test_exception_in_speaker_identifier_allows_pass():
    """Si ECAPA falla, logueamos y dejamos pasar (no bloqueamos el trigger)."""
    spk = MagicMock()
    spk.get_embedding = MagicMock(side_effect=RuntimeError("CUDA OOM"))
    emb = np.random.rand(192).astype(np.float32)

    det = _make_detector(
        whisper_text="nexa",
        speaker_identifier=spk, speaker_embedding=emb,
    )

    passed, sim = det._speaker_match(_audio(2.0))
    assert passed is True
    assert sim == 0.0


def test_filter_flag_active_requires_both():
    """_speaker_filter_active solo es True si ambos (identifier y embedding) están."""
    assert _make_detector("nexa")._speaker_filter_active is False
    spk = _make_speaker_identifier(0.9)
    det_no_emb = _make_detector("nexa", speaker_identifier=spk, speaker_embedding=None)
    assert det_no_emb._speaker_filter_active is False
    emb = np.random.rand(192).astype(np.float32)
    det_no_spk = _make_detector("nexa", speaker_identifier=None, speaker_embedding=emb)
    assert det_no_spk._speaker_filter_active is False
    det_both = _make_detector("nexa", speaker_identifier=spk, speaker_embedding=emb)
    assert det_both._speaker_filter_active is True


def test_threshold_boundary():
    """sim == threshold debe considerarse pass (>=, no >)."""
    spk = _make_speaker_identifier(sim_returns=0.65)
    emb = np.random.rand(192).astype(np.float32)
    det = _make_detector(
        whisper_text="nexa", speaker_identifier=spk, speaker_embedding=emb,
        speaker_threshold=0.65,
    )
    passed, sim = det._speaker_match(_audio(1.5))
    assert passed is True
    assert sim == pytest.approx(0.65)


# -----------------------------------------------------------------
# Fuzzy match tests (Whisper produce variantes "next", "nexia", etc)
# -----------------------------------------------------------------

def test_fuzzy_match_next_up():
    """Whisper oye 'next up apague la luz' → fuzzy fonético: next /nekst/ ~ /neksa/."""
    det = _make_detector(whisper_text="next up apague la luz")
    det.fuzzy_threshold = 0.75
    # Refresh wake_words_phon (already computed in __init__ from wake_words)
    match, text = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match == "nexa"


def test_fuzzy_match_nexia():
    """'nexia' /neksia/ ~ /neksa/ ratio ~0.91 → match."""
    det = _make_detector(whisper_text="nexia prendé la luz")
    det.fuzzy_threshold = 0.75
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match == "nexa"


def test_fuzzy_match_rejects_far_words():
    """'milagro' / 'gracias' tienen ratio fonético <0.3 → no matchean."""
    det = _make_detector(whisper_text="milagro gracias amigos")
    det.fuzzy_threshold = 0.75
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match is None


def test_fuzzy_rejects_nena_false_positive():
    """'nena' /nena/ vs 'nexa' /neksa/ = 0.67 — rechazado. Este es el caso clave
    que motiva usar fonético en vez de char-level Levenshtein (que daría 0.75)."""
    det = _make_detector(whisper_text="nena apagá la luz")
    det.fuzzy_threshold = 0.75
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match is None


def test_fuzzy_accepts_necks_better_than_char():
    """'necks' /neks/ ~ /neksa/ = 0.89 fonético vs 0.44 char-level.
    Fonético captura mejor este caso."""
    det = _make_detector(whisper_text="necks apagá la luz")
    det.fuzzy_threshold = 0.75
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match == "nexa"


def test_fuzzy_match_only_first_words():
    """fuzzy_start_words=3 ignora palabras posteriores para evitar FP."""
    det = _make_detector(whisper_text="hola amigos cómo están nexa apagá la luz")
    det.fuzzy_threshold = 0.75
    det.fuzzy_start_words = 3
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match is None


def test_fuzzy_disabled_when_threshold_zero():
    """fuzzy_threshold=0 desactiva el paso fuzzy completamente."""
    det = _make_detector(whisper_text="next up apague la luz")
    det.fuzzy_threshold = 0.0
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    assert match is None  # 'next' no es alias exact + fuzzy off → no match


def test_phonetic_encoding():
    """Sanity check del encoding fonético para casos clave."""
    from src.wakeword.whisper_wake import _phonetic_es
    assert _phonetic_es("nexa") == "neksa"
    assert _phonetic_es("next") == "nekst"
    assert _phonetic_es("nena") == "nena"
    assert _phonetic_es("alexa") == "aleksa"
    assert _phonetic_es("nexia") == "neksia"
    assert _phonetic_es("necks") == "neks"
    assert _phonetic_es("café") == "kafe"
    assert _phonetic_es("zapato") == "sapato"


def test_exact_alias_match_beats_fuzzy():
    """Si 'alexa' está en aliases, match exact se usa antes que fuzzy."""
    det = WhisperWakeDetector(
        whisper_stt=_FakeWhisperSTT("alexa apagá la luz"),
        wake_words=["nexa", "alexa"],  # alexa es alias
    )
    det._loaded = True
    det._vad = None
    det._torch = None
    match, _ = det._transcribe_and_match(_audio(2.0), dur_ms=2000)
    # El paso 1 matchea 'alexa' (exact substring)
    assert match == "alexa"
