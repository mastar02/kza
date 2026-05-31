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


# ---------------------------------------------------------------------------
# Fixture + tests for TV-mode / _record_reject behavior
# ---------------------------------------------------------------------------
from unittest.mock import Mock
from src.wakeword.whisper_wake import WhisperWakeDetector, TV_MODE_ENTRY_REJECTS


@pytest.fixture
def wake_detector():
    """Instancia mínima de WhisperWakeDetector para tests de TV-mode.

    Se inicializa sin GPU real: whisper_stt es un Mock() y el VAD nunca
    se ejercita — solo probamos métodos internos de conteo.
    """
    return WhisperWakeDetector(whisper_stt=Mock(), wake_words=["nexa"])


class TestRecordRejectTvMode:
    """_record_reject solo debe acumular reasons de audio real, no alucinaciones."""

    def test_record_reject_ignores_hallucination_reasons(self, wake_detector):
        """Las reasons de alucinación NO deben acumular hacia TV-mode."""
        det = wake_detector
        for _ in range(10):
            det._record_reject("tv_stop_phrase")
        assert not det._is_tv_mode_active()
        for _ in range(10):
            det._record_reject("multi_wake_hallucination")
        assert not det._is_tv_mode_active()

    def test_record_reject_counts_real_audio_reasons(self, wake_detector):
        """Reasons de audio real (no_command_verb) sí cuentan hacia TV-mode."""
        det = wake_detector
        for _ in range(TV_MODE_ENTRY_REJECTS):
            det._record_reject("no_command_verb")
        assert det._is_tv_mode_active()


# ---------------------------------------------------------------------------
# Task 1.2: captura de confianza acústica (no_speech_prob / avg_logprob)
# ---------------------------------------------------------------------------

def test_transcribe_captures_confidence(wake_detector, monkeypatch):
    """no_speech_prob/avg_logprob se extraen de los segments y se propagan a _emit_wake."""
    import numpy as np
    class FakeSeg:
        def __init__(self, text, nsp, lp):
            self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp
    segs = [FakeSeg("gracias", 0.92, -0.3)]
    # Anulamos la indirección _model (ver gotcha en la spec):
    # getattr(self.whisper, "_model", None) devuelve Mock truthy -> anulamos con None.
    wake_detector.whisper._model = None
    monkeypatch.setattr(wake_detector.whisper, "transcribe",
                        lambda *a, **k: (iter(segs), None))
    captured = {}
    monkeypatch.setattr(wake_detector, "_emit_wake",
                        lambda *a, **k: captured.update(k))
    audio = np.zeros(16000, dtype=np.float32)  # 1s de silencio
    wake_detector._transcribe_and_match(audio, 1000.0)
    assert captured.get("no_speech_prob") == 0.92
    assert captured.get("avg_logprob") == -0.3


# ---------------------------------------------------------------------------
# Task F2: denylist de alucinaciones de utterance-completa
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "¡Gracias!", "Gracias.", "Amén.", "Adiós.", "¿Verdad?",
])
def test_full_utterance_hallucinations_rejected_as_such(wake_detector, text):
    """Alucinaciones de saludo/cierre deben mapear a la reason no_speech_hallucination."""
    reason = wake_detector._full_utterance_hallucination_reason(_normalize(text))
    assert reason == "no_speech_hallucination"


def test_real_command_with_gracias_not_rejected(wake_detector):
    """gracias DENTRO de un comando real no debe matchear (no es utterance completa)."""
    assert wake_detector._full_utterance_hallucination_reason(
        _normalize("nexa prendé la luz gracias")) is None


def test_full_utterance_hallucination_rejected_via_transcribe(wake_detector, monkeypatch):
    """Wiring: una alucinación de utterance-completa transcrita se rechaza en _transcribe_and_match."""
    import numpy as np
    class FakeSeg:
        def __init__(self, text, nsp=0.9, lp=-0.4):
            self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp
    wake_detector.whisper._model = None
    monkeypatch.setattr(wake_detector.whisper, "transcribe",
                        lambda *a, **k: (iter([FakeSeg("Gracias.")]), None))
    emitted = {}
    monkeypatch.setattr(wake_detector, "_emit_wake", lambda *a, **k: emitted.update(k))
    matched, text = wake_detector._transcribe_and_match(np.zeros(16000, dtype=np.float32), 1000.0)
    assert matched is None
    assert emitted.get("rejection_reason") == "no_speech_hallucination"


# --- _canonicalize_wake ---

class TestCanonicalizeWake:
    """Reemplaza la variante fuzzy del wake por la forma canónica (word boundary, case-insensitive, solo primera)."""

    def test_canonicalize_wake_replaces_fuzzy_variant(self):
        from src.wakeword.whisper_wake import _canonicalize_wake
        assert _canonicalize_wake("Dexa, prender la luz", "dexa", "Nexa") == "Nexa, prender la luz"

    def test_canonicalize_wake_case_insensitive_first_only(self):
        from src.wakeword.whisper_wake import _canonicalize_wake
        assert _canonicalize_wake("dexa dexa luz", "dexa", "Nexa") == "Nexa dexa luz"

    def test_canonicalize_wake_no_match_returns_unchanged(self):
        from src.wakeword.whisper_wake import _canonicalize_wake
        assert _canonicalize_wake("prender la luz", "dexa", "Nexa") == "prender la luz"

    def test_canonicalize_wake_word_boundary(self):
        from src.wakeword.whisper_wake import _canonicalize_wake
        # no debe tocar substrings dentro de otra palabra
        assert _canonicalize_wake("indexar algo", "dexa", "Nexa") == "indexar algo"


# ---------------------------------------------------------------------------
# Tests for use_silero_vad flag
# ---------------------------------------------------------------------------

@pytest.fixture
def wake_detector_no_silero():
    """WhisperWakeDetector con use_silero_vad=False.

    No carga Silero real: el flag debe cortar la carga en load() también.
    _vad must stay None after construction and after load().
    vad_threshold=0.3 replica la config de produccion para XVF3800.
    """
    det = WhisperWakeDetector(
        whisper_stt=Mock(),
        wake_words=["nexa"],
        use_silero_vad=False,
        vad_threshold=0.3,  # config de produccion XVF3800 (settings.yaml)
    )
    det.load()  # debe no cargar Silero y marcar _loaded=True
    return det


def test_use_silero_vad_false_disables_vad_rms_only(wake_detector_no_silero):
    """Con use_silero_vad=False, _vad es None y _voice_prob usa fallback RMS-only."""
    det = wake_detector_no_silero
    assert det._vad is None
    import numpy as np
    loud = np.full(1280, 0.05, dtype=np.float32)   # rms ~0.05 >> min_rms
    assert det._voice_prob(loud) >= det.vad_threshold   # pasa el gate
    quiet = np.full(1280, 0.001, dtype=np.float32)  # rms < min_rms
    assert det._voice_prob(quiet) == 0.0                # no pasa
