"""Tests para la ventana de follow-up post-wake-solo.

Cuando el usuario dice "Nexa..." (pausa) "prendé luz", el detector ve dos
utterances separadas. Sin follow-up, la primera se rechaza por sin-verbo y
la segunda por sin-wake. Con follow-up, la primera arma una ventana y la
segunda se acepta como comando implícito.
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock

import numpy as np

# Mock heavy deps antes de importar el módulo
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

from src.wakeword.whisper_wake import WhisperWakeDetector, SAMPLE_RATE


class _FakeWhisperSTT:
    """Whisper stub que devuelve textos en secuencia (uno por llamada)."""
    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self._calls = 0

    def transcribe(self, audio, **kwargs):
        text = self._texts[self._calls] if self._calls < len(self._texts) else ""
        self._calls += 1
        seg = MagicMock()
        seg.text = text
        return [seg], None


def _make_detector(
    texts: list[str],
    follow_up_window_s: float = 4.0,
    follow_up_max_words: int = 3,
) -> WhisperWakeDetector:
    det = WhisperWakeDetector(
        whisper_stt=_FakeWhisperSTT(texts),
        wake_words=["nexa"],
        follow_up_window_s=follow_up_window_s,
        follow_up_max_words=follow_up_max_words,
    )
    det._loaded = True
    det._vad = None
    det._torch = None
    return det


def _audio(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


# -------------------- arming behavior --------------------

def test_wake_only_arms_follow_up():
    """'Nexa' solo (sin verbo) → no triggerea pero arma la ventana."""
    det = _make_detector(["Nexa"])
    match, _ = det._transcribe_and_match(_audio(0.5), 500.0)
    assert match is None
    assert det._follow_up_until > time.time()


def test_wake_only_with_repetition_arms():
    """'Nexa, Nexa, Nexa' (3 palabras) → arma."""
    det = _make_detector(["Nexa, Nexa, Nexa."])
    det._transcribe_and_match(_audio(1.0), 1000.0)
    assert det._follow_up_until > time.time()


def test_long_wake_utterance_does_not_arm():
    """'Nexa, ¿dónde estás aquí?' (>3 palabras) → no arma."""
    det = _make_detector(["Nexa, dónde estás aquí mi amigo querido"])
    det._transcribe_and_match(_audio(2.0), 2000.0)
    assert det._follow_up_until == 0.0


def test_window_zero_disables_arming():
    """follow_up_window_s=0 → no arma nunca."""
    det = _make_detector(["Nexa"], follow_up_window_s=0.0)
    det._transcribe_and_match(_audio(0.5), 500.0)
    assert det._follow_up_until == 0.0


# -------------------- consumption behavior --------------------

def test_follow_up_command_captured_within_window():
    """Tras 'Nexa', la próxima utterance con verbo (sin wake) dispara."""
    det = _make_detector(["Nexa", "prendé la luz del escritorio"])
    # 1) Wake-only — arma
    match1, _ = det._transcribe_and_match(_audio(0.5), 500.0)
    assert match1 is None
    # 2) Comando dentro de ventana
    match2, text2 = det._transcribe_and_match(_audio(1.5), 1500.0)
    assert match2 == "nexa"
    assert text2 == "nexa prendé la luz del escritorio"
    # Ventana consumida
    assert det._follow_up_until == 0.0


def test_follow_up_window_expires():
    """Si pasó el tiempo, no se acepta el comando sin wake."""
    det = _make_detector(["Nexa", "prendé la luz"], follow_up_window_s=0.05)
    det._transcribe_and_match(_audio(0.5), 500.0)
    time.sleep(0.10)
    match, _ = det._transcribe_and_match(_audio(1.0), 1000.0)
    assert match is None


def test_follow_up_ignores_tv_phrases():
    """Dentro de ventana, frases TV stop NO disparan comando."""
    det = _make_detector(["Nexa", "Gracias por ver el video"])
    det._transcribe_and_match(_audio(0.5), 500.0)
    match, _ = det._transcribe_and_match(_audio(1.0), 1000.0)
    assert match is None


def test_follow_up_skips_text_with_wake():
    """Si la 2da utterance ya contiene wake → flujo normal (no atajo)."""
    det = _make_detector(["Nexa", "Nexa apagá la luz"])
    det._transcribe_and_match(_audio(0.5), 500.0)
    # Segunda utterance contiene wake — debe pasar por el match normal
    # (que también dispara), no por la rama follow_up.
    match, text = det._transcribe_and_match(_audio(1.0), 1000.0)
    assert match == "nexa"
    # texto normal, no synthesized con doble nexa
    assert text == "Nexa apagá la luz"


def test_follow_up_requires_command_verb():
    """Dentro de ventana, utterance sin verbo no dispara."""
    det = _make_detector(["Nexa", "buenas tardes"])
    det._transcribe_and_match(_audio(0.5), 500.0)
    match, _ = det._transcribe_and_match(_audio(1.0), 1000.0)
    assert match is None
    # La ventana sigue armada — esperamos otra utterance con verbo
    assert det._follow_up_until > time.time()


def test_multi_wake_in_utterance_drops_command():
    """Utterance con 2+ 'nexa' es probable alucinación TV → no dispatch."""
    det = _make_detector(
        ["Nexa bajá la luz del escritorio, Nexa bajá la luz al cincuenta por ciento"]
    )
    match, _ = det._transcribe_and_match(_audio(2.0), 2000.0)
    assert match is None


def test_short_multi_wake_arms_follow_up():
    """'Nexa Nexa Nexa' (≤3 palabras) cae a follow-up arming."""
    det = _make_detector(["Nexa Nexa Nexa"])
    match, _ = det._transcribe_and_match(_audio(0.6), 600.0)
    assert match is None
    # Wake-only repetido frustrado → arma ventana
    assert det._follow_up_until > time.time()


def test_long_multi_wake_does_not_arm():
    """Utterance larga con 2+ wakes → rechaza pero NO arma (>max_words)."""
    det = _make_detector(["Nexa bajá luz del escritorio Nexa bajá luz al cincuenta"])
    det._transcribe_and_match(_audio(2.0), 2000.0)
    assert det._follow_up_until == 0.0


def test_single_wake_normal_passes():
    """1x 'nexa' + comando válido — no se afecta por la regla multi-wake."""
    det = _make_detector(["Nexa bajá la luz al cincuenta por ciento"])
    match, _ = det._transcribe_and_match(_audio(2.0), 2000.0)
    assert match == "nexa"


def test_normal_command_not_affected_by_follow_up_logic():
    """'Nexa prendé luz' funciona normal, sin tocar follow_up state."""
    det = _make_detector(["Nexa prendé la luz"])
    match, text = det._transcribe_and_match(_audio(1.0), 1000.0)
    assert match == "nexa"
    assert text == "Nexa prendé la luz"
    assert det._follow_up_until == 0.0
