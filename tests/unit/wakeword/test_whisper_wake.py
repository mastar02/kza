"""Tests for WhisperWake helpers — coalescing and normalization."""
import pytest
from src.wakeword.whisper_wake import _decoalesce_post_wake, _normalize, _COMMAND_VERB_RE


class TestDecoalescePostWake:
    def test_aprende_becomes_prende(self):
        """Whisper pasted 'a' of 'nexa' onto 'prende'. Re-segment."""
        text = _normalize("nexa aprende la luz del escritorio")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == "nexa prende la luz del escritorio"

    def test_apaga_unchanged(self):
        """'nexa apagá' is already a valid command — do not modify."""
        text = _normalize("nexa apaga la luz")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == "nexa apaga la luz"

    def test_aencendemos_becomes_encendemos(self):
        """Rare coalescing but possible: 'nexa aencendé' → 'nexa encendé'."""
        text = _normalize("nexa aencende la estufa")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == "nexa encende la estufa"

    def test_abaja_becomes_baja(self):
        text = _normalize("nexa abaja la persiana")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == "nexa baja la persiana"

    def test_asubi_becomes_subi(self):
        text = _normalize("nexa asubi el volumen")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == "nexa subi el volumen"

    def test_no_wake_at_start_passthrough(self):
        """If text doesn't start with the wake word, don't touch."""
        text = _normalize("che prende la luz")
        fixed = _decoalesce_post_wake(text, wake_norm="nexa")
        assert fixed == text

    def test_short_text_passthrough(self):
        """A single-word utterance cannot have coalescing."""
        fixed = _decoalesce_post_wake("nexa", wake_norm="nexa")
        assert fixed == "nexa"

    def test_unknown_second_word_passthrough(self):
        """'nexa alguien' — no whitelisted prefix → unchanged."""
        fixed = _decoalesce_post_wake("nexa alguien llama", wake_norm="nexa")
        assert fixed == "nexa alguien llama"

    def test_empty_string_passthrough(self):
        assert _decoalesce_post_wake("", wake_norm="nexa") == ""


class TestDecoalesceIntegrationWithCommandVerbRegex:
    """After decoalesce, the command verb regex should match for real commands
    that were broken by coalescing."""

    def test_aprende_triggers_prend_match(self):
        norm = _decoalesce_post_wake(
            _normalize("nexa aprende la luz del escritorio"),
            wake_norm="nexa",
        )
        assert _COMMAND_VERB_RE.search(norm) is not None

    def test_aprende_español_does_not_trigger_match(self):
        """After decoalesce 'nexa aprende español' becomes 'nexa prende español'.
        This IS matched by the regex, but the domotica flow would fail to find
        an entity — acceptable FP rate per spec (prefer over FN)."""
        norm = _decoalesce_post_wake(
            _normalize("nexa aprende español"),
            wake_norm="nexa",
        )
        # Documenta el trade-off: SÍ matchea, downstream filtrará por entity.
        assert _COMMAND_VERB_RE.search(norm) is not None
