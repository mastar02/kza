"""Tests for combined-signal phantom detection in whisper_wake.

Reproduce los dos eventos fantasma de 2026-05-03:
- 01:25:05 — TV-mode ACTIVO + audio 1.6s (1.21s removidos por VAD = 0.39s
  voz real) + rms=0.025 → texto coherente 'Nexa bajá la luz al cincuenta
  por ciento' alucinado.
- 08:08:49 — sin TV-mode + audio 0.8s + rms=0.013 → mismo texto exacto,
  9 palabras en 0.8s = 11.25 wps (físicamente imposible).

Hard reject: implausible speech rate (>8.5 wps). Independiente de TV-mode
porque es físicamente imposible que un humano hable así de rápido.

Soft reject: en TV-mode + (audio < 1s effective-voice OR rms < 0.04) +
texto sin mención literal de room/entity → reject. Combina señales para
no crear FPs en comandos legítimos en TV-mode.
"""

import pytest

from src.wakeword.whisper_wake import (
    _is_implausible_speech_rate,
    _has_explicit_entity_or_room,
)


class TestImplausibleSpeechRate:
    """Hard reject: human speech maxes ~7-8 wps. >8.5 wps is impossible."""

    def test_phantom_event_2_words_per_second_is_implausible(self):
        """Evento 08:08:49: 9 palabras en 0.8s = 11.25 wps."""
        text = "Nexa bajá la luz al cincuenta por ciento,"
        assert _is_implausible_speech_rate(text, audio_duration_s=0.8) is True

    def test_phantom_event_1_is_borderline_not_blocked(self):
        """Evento 01:25:05: 9 palabras en 1.6s = 5.6 wps. Plausible (voz rápida)."""
        text = "Nexa bajá la luz al cincuenta por ciento,"
        assert _is_implausible_speech_rate(text, audio_duration_s=1.6) is False

    def test_normal_speech_passes(self):
        """Voz normal ~3-4 wps no se bloquea."""
        text = "Nexa prendé la luz"
        assert _is_implausible_speech_rate(text, audio_duration_s=1.5) is False

    def test_empty_text_does_not_trigger(self):
        assert _is_implausible_speech_rate("", audio_duration_s=0.5) is False

    def test_zero_duration_does_not_trigger(self):
        """No dividir por cero."""
        assert _is_implausible_speech_rate("hola mundo", audio_duration_s=0.0) is False

    def test_single_word_quick_does_not_trigger(self):
        """Una palabra sola en 0.3s — natural ('apagá')."""
        assert _is_implausible_speech_rate("apagá", audio_duration_s=0.3) is False


class TestExplicitEntityOrRoomMention:
    """Detecta si el texto menciona literal una entidad o room conocido.

    Esto sirve para la heurística de TV-mode: en TV-mode, comandos sin
    referente literal son sospechosos (la TV no suele articular 'living').
    """

    def test_text_with_room_alias_matches(self):
        assert _has_explicit_entity_or_room(
            "prendé la luz del escritorio",
            known_aliases=("escritorio", "living", "cocina"),
        ) is True

    def test_text_with_no_alias_does_not_match(self):
        """El texto fantasma no tiene mención de room ni entity específica."""
        assert _has_explicit_entity_or_room(
            "Nexa bajá la luz al cincuenta por ciento",
            known_aliases=("escritorio", "living", "cocina", "cuarto", "baño"),
        ) is False

    def test_match_is_word_boundary_not_substring(self):
        """'baño' no debe matchear inside 'rebañar' (no es comando real, pero
        valida que no usemos substring puro)."""
        assert _has_explicit_entity_or_room(
            "rebañar el campo",
            known_aliases=("baño",),
        ) is False

    def test_accent_insensitive(self):
        """'baño' (con tilde) debe matchear 'bano' normalizado."""
        assert _has_explicit_entity_or_room(
            "prendé la luz del baño",
            known_aliases=("bano",),
        ) is True

    def test_empty_aliases_returns_false(self):
        assert _has_explicit_entity_or_room("cualquier cosa", known_aliases=()) is False


class TestPhantomDetectionIntegration:
    """End-to-end del detector con audio mock — reproduce los 2 eventos."""

    def _detector_with_aliases(self, aliases=("escritorio", "living", "cocina", "cuarto", "bano")):
        from src.wakeword.whisper_wake import WhisperWakeDetector
        whisper = type("FakeWhisper", (), {})()
        d = WhisperWakeDetector(
            whisper_stt=whisper,
            wake_words=["nexa"],
            known_room_aliases=aliases,
        )
        # Skip load() — tests only exercise _transcribe_and_match downstream.
        return d

    def test_event_2_implausible_rate_is_rejected(self, monkeypatch):
        """Evento 08:08:49: 9 palabras en 0.8s. Debe rechazarse aunque NO
        haya TV-mode activo (es físicamente imposible)."""
        import numpy as np
        d = self._detector_with_aliases()
        d._tv_mode_until = 0.0  # NO TV-mode
        # Mock the model.transcribe to return the phantom text
        phantom_text = "Nexa bajá la luz al cincuenta por ciento,"

        class FakeSeg:
            def __init__(self, t): self.text = t
        class FakeModel:
            def transcribe(self, audio, **kw):
                return ([FakeSeg(phantom_text)], None)
        d.whisper = FakeModel()

        audio = np.zeros(int(16000 * 0.8), dtype=np.float32)  # 0.8s
        result = d._transcribe_and_match(audio, dur_ms=800.0)
        assert result[0] is None, (
            f"Implausible speech rate (11.25 wps) must be rejected even "
            f"without TV-mode. Got match={result[0]!r}"
        )

    def test_event_1_tv_mode_low_rms_no_room_is_rejected(self):
        """Evento 01:25:05: TV-mode activo + audio rms 0.025 (debajo de 0.04)
        + texto sin room → reject."""
        import numpy as np
        import time
        d = self._detector_with_aliases()
        d._tv_mode_until = time.time() + 60  # TV-mode ACTIVO
        phantom_text = "Nexa bajá la luz al cincuenta por ciento,"

        class FakeSeg:
            def __init__(self, t): self.text = t
        class FakeModel:
            def transcribe(self, audio, **kw):
                return ([FakeSeg(phantom_text)], None)
        d.whisper = FakeModel()

        # Audio largo para evitar el speech-rate hard-reject (1.6s = 5.6 wps OK).
        # RMS bajo: 0.025 < tv_mode_min_rms=0.04 → soft-reject debe disparar.
        # El detector espera float32 normalizado [-1, 1] (ver _voice_prob).
        n = int(16000 * 1.6)
        audio = (np.random.RandomState(0).randn(n) * 0.025).astype(np.float32)

        result = d._transcribe_and_match(audio, dur_ms=1600.0)
        assert result[0] is None, (
            f"TV-mode + low rms + no room mention must reject. "
            f"Got match={result[0]!r}"
        )

    def test_tv_mode_with_room_mention_passes(self):
        """Comando legítimo en TV-mode con room explícito → accept."""
        import numpy as np
        import time
        d = self._detector_with_aliases()
        d._tv_mode_until = time.time() + 60  # TV-mode ACTIVO
        legit_text = "Nexa prendé la luz del escritorio"

        class FakeSeg:
            def __init__(self, t): self.text = t
        class FakeModel:
            def transcribe(self, audio, **kw):
                return ([FakeSeg(legit_text)], None)
        d.whisper = FakeModel()

        # Audio degradado IGUAL que el evento 1 — pero el texto SÍ menciona
        # 'escritorio' → debe pasar.
        n = int(16000 * 1.6)
        amp_int16 = int(0.025 * 32768)
        audio = (np.random.RandomState(0).randn(n) * amp_int16 * 0.7).astype(np.int16)

        result = d._transcribe_and_match(audio, dur_ms=1600.0)
        assert result[0] == "nexa", (
            f"Legitimate command in TV-mode WITH room mention must pass. "
            f"Got match={result[0]!r}"
        )

    def test_no_tv_mode_low_signal_passes_when_aliases_empty(self):
        """Backward compat: sin known_room_aliases configuradas, soft-reject off."""
        import numpy as np
        import time
        d = self._detector_with_aliases(aliases=())  # vacío
        d._tv_mode_until = time.time() + 60  # TV-mode ACTIVO
        phantom_text = "Nexa bajá la luz"

        class FakeSeg:
            def __init__(self, t): self.text = t
        class FakeModel:
            def transcribe(self, audio, **kw):
                return ([FakeSeg(phantom_text)], None)
        d.whisper = FakeModel()

        n = int(16000 * 1.6)
        # Audio float32 con buen rms (~0.1) para evitar disparar el soft-reject.
        audio = (np.random.RandomState(0).randn(n) * 0.1).astype(np.float32)
        result = d._transcribe_and_match(audio, dur_ms=1600.0)
        assert result[0] == "nexa", (
            "With empty known_room_aliases, soft-reject must be disabled "
            "(backward compat)."
        )
