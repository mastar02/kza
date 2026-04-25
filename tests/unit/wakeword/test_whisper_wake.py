"""Tests for WhisperWake helpers — coalescing and normalization."""
import pytest
from src.wakeword.whisper_wake import (
    _decoalesce_post_wake,
    _decoalesce_original_text,
    _normalize,
    _COMMAND_VERB_RE,
)


class TestDecoalesceOriginalText:
    """Decoalesce aplicado al texto con acentos/puntuación (el que va al NLU)."""

    def test_aprende_with_accent_becomes_prende_with_accent(self):
        fixed = _decoalesce_original_text(
            "Nexa aprendé la luz del escritorio.", wake_norm="nexa"
        )
        assert fixed == "Nexa prendé la luz del escritorio."

    def test_aprende_without_accent(self):
        fixed = _decoalesce_original_text(
            "Nexa aprende la luz", wake_norm="nexa"
        )
        assert fixed == "Nexa prende la luz"

    def test_apaga_unchanged(self):
        """'apag' está en la lista pero coalesced==real → no toca."""
        fixed = _decoalesce_original_text(
            "Nexa apagá la luz", wake_norm="nexa"
        )
        assert fixed == "Nexa apagá la luz"

    def test_abaja_with_accent(self):
        fixed = _decoalesce_original_text(
            "Nexa abajá las persianas", wake_norm="nexa"
        )
        assert fixed == "Nexa bajá las persianas"

    def test_no_wake_at_start_passthrough(self):
        assert (
            _decoalesce_original_text("che aprendé", wake_norm="nexa")
            == "che aprendé"
        )

    def test_empty_text(self):
        assert _decoalesce_original_text("", wake_norm="nexa") == ""

    def test_only_wake_word(self):
        assert _decoalesce_original_text("Nexa", wake_norm="nexa") == "Nexa"

    def test_preserves_punctuation_after_verb(self):
        fixed = _decoalesce_original_text(
            "Nexa, aprendé la luz del escritorio.", wake_norm="nexa"
        )
        # 'Nexa,' normaliza a 'nexa' → ok; preserva la coma
        assert fixed == "Nexa, prendé la luz del escritorio."


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

    def test_para_arriba_matches(self):
        assert _COMMAND_VERB_RE.search("nexa para arriba") is not None

    def test_para_abajo_matches(self):
        assert _COMMAND_VERB_RE.search("nexa para abajo") is not None

    def test_mas_fuerte_matches(self):
        assert _COMMAND_VERB_RE.search("nexa mas fuerte") is not None

    def test_mas_bajo_matches(self):
        assert _COMMAND_VERB_RE.search("nexa mas bajo") is not None

    def test_aire_acondicionado_matches(self):
        assert _COMMAND_VERB_RE.search("nexa aire acondicionado") is not None

    def test_aire_solo_no_matches(self):
        """'aire' solo no es comando — TV dice 'estás en el aire' frecuentemente."""
        assert _COMMAND_VERB_RE.search("estas en el aire") is None

    def test_persiana_matches(self):
        assert _COMMAND_VERB_RE.search("nexa persiana") is not None

    def test_cortina_matches(self):
        assert _COMMAND_VERB_RE.search("nexa cortina") is not None

    def test_mas_calor_matches(self):
        assert _COMMAND_VERB_RE.search("nexa mas calor") is not None

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
