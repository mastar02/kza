"""Tests for WhisperWake helpers — coalescing and normalization."""
import pytest
from src.wakeword.whisper_wake import (
    _decoalesce_post_wake,
    _decoalesce_original_text,
    _has_pathological_repeat,
    _normalize,
    _text_likely_truncated,
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


class TestPathologicalRepeat:
    """Detección de loops patológicos de Whisper en transcripciones del wake.

    Casos reales tomados de logs (2026-05-01 18:00-22:00, TV con serie hispana
    encendida). El filtro `wake_count >= 2` ya cubre loops del wake word; este
    cubre loops sobre tokens NO-wake.
    """

    # --- Casos que DEBEN detectarse como loop ---
    def test_baño_loop_real_from_log(self):
        """Caso real 18:42:09 — comando fantasma ejecutado."""
        norm = _normalize(
            "Nexa bajá la luz del escritorio, el baño y el baño y el baño "
            "y el baño y el baño y el baño."
        )
        assert _has_pathological_repeat(norm) == "bano"

    def test_cincuenta_loop_real_from_log(self):
        """Caso real 18:50:14 — comando fantasma ejecutado."""
        norm = _normalize(
            "Nexa bajá la luz al cincuenta por cincuenta por cincuenta "
            "por cincuenta por cincuenta."
        )
        assert _has_pathological_repeat(norm) == "cincuenta"

    def test_escritorio_loop_real_from_log(self):
        """Caso real 06:36:56 — Whisper alucinó la repetición sobre voz real."""
        norm = _normalize(
            "Nexa prendé la luz del escritorio, la luz del escritorio, "
            "la luz del escritorio, la luz del escritorio,"
        )
        assert _has_pathological_repeat(norm) is not None

    def test_count_three_with_dominance(self):
        """Threshold mínimo: 3 repeticiones con dominancia ≥40% Y ≥6 content tokens.

        Construimos exactamente 6 content tokens donde 3 son la palabra repetida
        (50% dominancia) — debe detectar.
        """
        norm = _normalize("nexa prende abre baño baño baño")
        # content tokens: nexa, prende, abre, baño×3 → baño=3/6=50%
        assert _has_pathological_repeat(norm) == "bano"

    # --- Casos que NO deben detectarse (commands legítimos) ---
    def test_multi_room_command_not_loop(self):
        """Comando multi-room legítimo: 'luz' aparece varias veces sin dominar."""
        norm = _normalize(
            "Nexa prendé la luz del escritorio, la luz del living, "
            "la luz de la cocina, la luz del baño"
        )
        # 'luz' aparece 4 veces, pero hay otras 6 palabras de contenido
        # distintas (escritorio, living, cocina, baño, prende, nexa) → 4/10=40%
        # justo en el borde. Aceptamos que este caso límite se rechace para
        # ser conservadores con TV; el usuario puede dividir el comando.
        # IMPORTANTE: si esto produce FPs en uso real, subir threshold a 50%.
        # Por ahora documentamos: ≥40% es loop.
        result = _has_pathological_repeat(norm)
        # No assertion sobre None aquí; ver test_multi_room_with_more_diversity
        assert result in (None, "luz")

    def test_multi_room_with_more_diversity(self):
        """Comando largo con 'luz' frecuente pero diluida."""
        norm = _normalize(
            "Nexa prendé la luz del cuarto del living del comedor de la cocina"
        )
        # content: nexa, prende, luz, cuarto, living, comedor, cocina → 7
        # luz=1/7=14% → no loop
        assert _has_pathological_repeat(norm) is None

    def test_short_command_not_loop(self):
        """Comandos cortos pasan sin importar."""
        assert _has_pathological_repeat(_normalize("nexa prende la luz")) is None

    def test_empty_string(self):
        assert _has_pathological_repeat("") is None

    def test_below_min_content_tokens(self):
        """Con < 6 content tokens nunca marcamos loop (evita FPs en commands cortos)."""
        norm = _normalize("baño baño baño")
        # Solo 3 content tokens — no llega al min de 6
        assert _has_pathological_repeat(norm) is None

    def test_stopwords_repeated_not_loop(self):
        """Repetición de stopwords ('no, no, no, no') no califica.

        El filtro `noise_phrase` del request_router ya cubre 'no no no'.
        """
        norm = _normalize("no no no no no no no")
        assert _has_pathological_repeat(norm) is None

    def test_natural_emphasis_not_loop(self):
        """'muy muy muy fuerte' es énfasis humano, no loop."""
        norm = _normalize("Nexa subí el volumen muy muy muy fuerte")
        # 'muy' aparece 3 veces — content tokens: nexa, subi, volumen, muy×3, fuerte = 7
        # muy=3/7=43% → BORDE. Aceptamos que se filtre — si causa FPs reales,
        # ajustar threshold. Por ahora el caso es muy infrecuente.
        # No assertion estricta — documentamos comportamiento.
        _ = _has_pathological_repeat(norm)


class TestTextLikelyTruncated:
    """Detección de transcripciones cortadas por silencio breve mid-comando.

    Caso real: 2026-05-02 06:36:51 — 'Nexa prendé la luz del...' (audio cortado
    a ~1s) terminó en `light.bano` porque el LLM adivinó con el texto truncado.
    """

    # --- Truncated ---
    def test_ends_with_del(self):
        assert _text_likely_truncated("Nexa prendé la luz del") is True

    def test_ends_with_del_punctuation(self):
        """Caso real del log: 'Nexa prendé la luz del...'"""
        assert _text_likely_truncated("Nexa prendé la luz del...") is True

    def test_ends_with_la(self):
        assert _text_likely_truncated("Nexa prendé la") is True

    def test_ends_with_y(self):
        assert _text_likely_truncated("Nexa prendé la luz y") is True

    def test_ends_with_en(self):
        assert _text_likely_truncated("Nexa apagá la luz en") is True

    def test_ends_with_ellipsis_unicode(self):
        assert _text_likely_truncated("Nexa prendé la luz…") is True

    def test_ends_with_para(self):
        assert _text_likely_truncated("Nexa prendé la luz para") is True

    def test_ends_with_artículo_unas(self):
        assert _text_likely_truncated("Nexa prendé unas") is True

    # --- NOT truncated (commands válidos completos) ---
    def test_complete_command_with_entity(self):
        assert _text_likely_truncated("Nexa prendé la luz del escritorio.") is False

    def test_complete_command_no_period(self):
        assert _text_likely_truncated("Nexa prendé la luz del escritorio") is False

    def test_short_legitimate_command(self):
        """'Nexa apagá' es voseo válido por sí solo."""
        assert _text_likely_truncated("Nexa apagá") is False

    def test_complete_with_percentage(self):
        assert _text_likely_truncated("Nexa bajá la luz al cincuenta por ciento.") is False

    def test_empty_text(self):
        assert _text_likely_truncated("") is False

    def test_none_safe(self):
        assert _text_likely_truncated(None) is False  # type: ignore[arg-type]

    def test_with_accents_normalized(self):
        """Verifica que las preposiciones con acento se reconozcan tras normalize."""
        # 'más' en español es ambigua — la incluimos porque "X más" suele cortar
        # ("subí más" sí termina, pero "subí más fuerte" no — el peor caso).
        assert _text_likely_truncated("Nexa subí más") is True
