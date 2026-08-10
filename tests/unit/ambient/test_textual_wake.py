"""Tests del canal de wake textual "nexa" (spec 2026-07-05) — matcher + dedup.

Lógica pura: normalize_text, matches_wake, TextualWakeDetector. Reloj
inyectado (FakeClock, sin sleeps) — mismo patrón que test_ambient_guard.py.
"""
import logging
from unittest.mock import AsyncMock

import numpy as np

from src.ambient.textual_wake import TextualWakeDetector, matches_wake, normalize_text
from src.pipeline.command_event import CommandEvent


class FakeClock:
    """Reloj monotónico controlable a mano (patrón de test_ambient_guard.py)."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_audio() -> np.ndarray:
    return np.zeros(160, dtype=np.float32)


# ==================== normalize_text ====================

class TestNormalizeText:
    def test_lowercases_and_strips_accents(self):
        assert normalize_text("Nexa, apagá la luz") == "nexa apaga la luz"

    def test_collapses_whitespace(self):
        assert normalize_text("nexa   apaga    la luz") == "nexa apaga la luz"

    def test_strips_punctuation(self):
        assert normalize_text("¡Nexa! ¿apagás la luz?") == "nexa apagas la luz"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation_normalizes_to_empty(self):
        assert normalize_text("¡¿...?!") == ""


# ==================== matches_wake: exact ====================

class TestMatchesWakeExact:
    def test_exact_token_matches(self):
        assert matches_wake("Nexa, apagá la luz") is True

    def test_bigram_next_up_matches(self):
        assert matches_wake("Next up, apagá la luz") is True

    def test_empty_text_does_not_match(self):
        assert matches_wake("") is False

    def test_no_wake_word_does_not_match(self):
        assert matches_wake("apagá la luz") is False


# ==================== matches_wake: fuzzy (edit distance) ====================

class TestMatchesWakeFuzzy:
    def test_neza_matches_edit_distance_1(self):
        # n-e-z-a vs n-e-x-a: 1 sustitución.
        assert matches_wake("neza apaga") is True

    def test_lexa_matches_edit_distance_1(self):
        # l-e-x-a vs n-e-x-a: 1 sustitución.
        assert matches_wake("lexa apaga") is True

    def test_alexa_does_not_match_edit_distance_2(self):
        # edit_distance("nexa", "alexa") == 2 (verificado a mano con la DP:
        # inserción de 'a' inicial + inserción de 'l') > max_edit_distance=1.
        assert matches_wake("alexa apaga la luz") is False

    def test_anexa_matches_accepted_false_positive_v1(self):
        # "anexa" está a edit-distance 1 de "nexa" (inserción de 'a':
        # a-n-e-x-a vs n-e-x-a). DECISIÓN 2026-07-05: aceptable en v1 (falso
        # positivo raro en habla real, documentado en el docstring del
        # módulo). Este test PINNEA el comportamiento REAL elegido (True),
        # no lo oculta con un assert False cómodo.
        assert matches_wake("la anexa el documento") is True

    def test_next_alone_does_not_match_after_denylist(self):
        # "next" está a distancia 1 de "nexa" pero es falso positivo común
        # en STT ambient en inglés (Parakeet emite spurious english words).
        # Mitigación: denylist. Bare "next" ya no debería matchear.
        assert matches_wake("el next episodio ya empieza") is False

    def test_next_up_bigram_still_matches_unaffected(self):
        # El denylist afecta SOLO al fuzzy per-token; el bigram "next up"
        # se evalúa en el pase exacto y NO consulta el denylist.
        assert matches_wake("Next up, apagá la luz") is True

    def test_nena_does_not_match_common_vocative(self):
        # "nena" (vocativo rioplatense muy común, "nena, apagá la luz") está
        # a distancia 1 de "nexa" (sustitución x↔n) — denylist para evitar
        # ejecutar comandos dirigidos a una persona, no al asistente.
        assert matches_wake("nena apagá la luz") is False

    def test_nexo_does_not_match_common_word(self):
        # "nexo" ("el nexo entre ambos") está a distancia 1 de "nexa"
        # (sustitución a↔o) — palabra española común, denylist.
        assert matches_wake("el nexo entre ambos") is False

    def test_rexa_does_not_match_parakeet_reja(self):
        # "rexa" = transcripción típica de Parakeet para "reja", distancia 1
        # de "nexa" (sustitución n↔r). 2 fantasmas reales en prod
        # (auditoría 2026-07-24) — denylist.
        assert matches_wake("Pasame la rexa y de paso...") is False
        assert matches_wake("pero de una rexa igual") is False

    def test_unrelated_words_do_not_match(self):
        # "apaga"/"la"/"luz" no están a distancia <=1 de "nexa".
        assert matches_wake("apagá la luz") is False


# ==================== matches_wake: parametrización ====================

class TestMatchesWakeCustomization:
    def test_max_edit_distance_zero_disables_fuzzy(self):
        assert matches_wake("neza apaga", max_edit_distance=0) is False
        assert matches_wake("nexa apaga", max_edit_distance=0) is True

    def test_custom_variants_replace_default(self):
        assert matches_wake("hola casa apaga", variants=("casa",)) is True
        assert matches_wake("hola nexa apaga", variants=("casa",)) is False


# ==================== TextualWakeDetector: dispatch ====================

class TestDetectorDispatch:
    async def test_dispatches_on_match_with_human_source(self):
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=FakeClock(),
        )
        audio = make_audio()
        text = "Nexa, apagá la luz"

        result = await detector.maybe_dispatch(
            room_id="salon", text=text, source="human_direct",
            speaker="gabriel", audio=audio,
        )

        assert result is True
        dispatch.assert_awaited_once()
        event = dispatch.call_args.args[0]
        assert isinstance(event, CommandEvent)
        assert event.wake_text == text
        assert event.room_id == "salon"
        assert event.wake_score == 1.0
        assert event.audio is audio


# ==================== TextualWakeDetector: skip conditions ====================

class TestDetectorSkipConditions:
    async def test_no_dispatch_when_source_is_tv(self):
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="tv",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_no_dispatch_when_source_is_self(self):
        # source="self" = eco de la propia TTS del asistente (SourceClassifier
        # durante reproducción). Una respuesta hablada que contenga "nexa" no
        # debe re-disparar un comando.
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa apagó la luz", source="self",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_self_skip_logs_info(self, caplog):
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="salon", text="Nexa apagó la luz", source="self",
                speaker=None, audio=make_audio(),
            )

        assert any("[TextualWake]" in r.message for r in caplog.records)

    async def test_no_dispatch_when_disabled(self):
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            enabled=False,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_no_dispatch_when_no_match(self):
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_tv_skip_logs_info(self, caplog):
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="salon", text="Nexa, apagá la luz", source="tv",
                speaker=None, audio=make_audio(),
            )

        assert any("[TextualWake]" in r.message for r in caplog.records)

    async def test_no_match_emits_no_log(self, caplog):
        # "no match" es el caso común (la enorme mayoría del stream ambient)
        # — no debe loguear nada, a diferencia de tv/dedup.
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="salon", text="apagá la luz", source="human_direct",
                speaker=None, audio=make_audio(),
            )

        assert not any("[TextualWake]" in r.message for r in caplog.records)


# ==================== TextualWakeDetector: gate por vad ====================

class TestDetectorVadGate:
    """Gate por vad_prob (2026-08-06): a vad bajo el texto del ambient es
    garble inglés (87% en el bucket <0.20, medido 08-04) y "Next up." espurio
    dispara el canal. Auditoría de ambient.db 08-06: el garble puro vive en
    vad 0.36-0.49; los comandos reales far-field arrancan en ~0.57. Umbral
    default 0.50. Gate SOLO por vad, JAMÁS por idioma: los comandos reales
    salen con lang='fr'/'en' (py3langid sobre "Nexa, prende la luz") — el
    veto por idioma es NO-GO medido (mata 2/2 comandos reales)."""

    def _detector(self, dispatch, **kwargs):
        return TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=FakeClock(),
            **kwargs,
        )

    async def test_vad_below_threshold_skips(self):
        dispatch = AsyncMock()
        detector = self._detector(dispatch, min_vad=0.50)

        result = await detector.maybe_dispatch(
            room_id="cocina", text="Next up.", source="unknown",
            speaker=None, audio=make_audio(), vad_prob=0.36,
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_vad_at_threshold_dispatches(self):
        # El umbral es inclusivo: vad_prob >= min_vad pasa.
        dispatch = AsyncMock(return_value={"success": True})
        detector = self._detector(dispatch, min_vad=0.50)

        result = await detector.maybe_dispatch(
            room_id="cocina", text="Nexa, prende la luz", source="unknown",
            speaker=None, audio=make_audio(), vad_prob=0.50,
        )

        assert result is True
        dispatch.assert_awaited_once()

    async def test_vad_omitted_fails_open(self):
        # Llamadores que no pasan vad_prob (kwarg con default None) no se
        # bloquean — mismo criterio que is_spanish_keepable: la ausencia del
        # instrumento no puede silenciar la red de seguridad. (Cubre también
        # vad_prob=None explícito: es el mismo branch con el mismo default.)
        dispatch = AsyncMock(return_value={"success": True})
        detector = self._detector(dispatch, min_vad=0.50)

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is True

    async def test_default_threshold_is_050(self):
        # Pinnea el default: 0.49 (garble ambiguo medido) skip, 0.57 (comando
        # real medido "Nexa, prende la luz" en cocina) dispara.
        dispatch = AsyncMock(return_value={"success": True})
        detector = self._detector(dispatch)

        assert await detector.maybe_dispatch(
            room_id="escritorio", text="Next up. Los.", source="unknown",
            speaker=None, audio=make_audio(), vad_prob=0.49,
        ) is False
        assert await detector.maybe_dispatch(
            room_id="cocina", text="Nexa, prende la luz", source="unknown",
            speaker=None, audio=make_audio(), vad_prob=0.57,
        ) is True

    async def test_low_vad_skip_logs_info(self, caplog):
        dispatch = AsyncMock()
        detector = self._detector(dispatch, min_vad=0.50)

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="cocina", text="Next up.", source="unknown",
                speaker=None, audio=make_audio(), vad_prob=0.36,
            )

        msgs = [r.message for r in caplog.records if "[TextualWake]" in r.message]
        assert any("low_vad" in m and "0.36" in m for m in msgs)

    async def test_dedup_acoustic_wins_over_low_vad_in_log(self, caplog):
        # El gate corre DESPUÉS de los dedups (review 2026-08-09): la
        # re-transcripción solapada de un comando ya despachado llega con vad
        # diluido, y loguearla como low_vad contaminaría la señal del journal
        # que el runbook usa para recalibrar el umbral.
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 999.0,  # hace 1s
            min_vad=0.50,
            now_fn=clock,
        )

        with caplog.at_level(logging.INFO):
            result = await detector.maybe_dispatch(
                room_id="salon", text="Nexa, prende la luz", source="unknown",
                speaker=None, audio=make_audio(), vad_prob=0.30,
            )

        assert result is False
        msgs = [r.message for r in caplog.records if "[TextualWake]" in r.message]
        assert any("dedup_acoustic" in m for m in msgs)
        assert not any("low_vad" in m for m in msgs)

    async def test_low_vad_without_match_emits_no_log(self, caplog):
        # vad bajo + sin wake = el caso común del stream — silencio en logs.
        dispatch = AsyncMock()
        detector = self._detector(dispatch, min_vad=0.50)

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="cocina", text="so we got a border", source="unknown",
                speaker=None, audio=make_audio(), vad_prob=0.10,
            )

        assert not any("[TextualWake]" in r.message for r in caplog.records)


# ==================== TextualWakeDetector: dedup acústico ====================

class TestDetectorDedupAcoustic:
    async def test_skips_when_acoustic_dispatched_recently(self):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock()
        last_acoustic_at = clock.t - 3.0  # hace 3s, window default 8s

        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: last_acoustic_at,
            now_fn=clock,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()

    async def test_dispatches_when_acoustic_dispatch_is_stale(self):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        last_acoustic_at = clock.t - 20.0  # hace 20s, window default 8s

        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: last_acoustic_at,
            now_fn=clock,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is True
        dispatch.assert_awaited_once()

    async def test_never_acoustic_dispatched_does_not_dedup(self):
        # 0.0 = "nunca". Con un reloj arrancando en 0.0, now - 0.0 == 0.0 <
        # window: SIN la guarda ">0.0" esto dedupearía por error un dispatch
        # que jamás ocurrió. Pinnea la semántica correcta del sentinel.
        clock = FakeClock(t=0.0)
        dispatch = AsyncMock(return_value={"success": True})

        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is True

    async def test_dedup_acoustic_logs_info(self, caplog):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: clock.t - 3.0,
            now_fn=clock,
        )

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="salon", text="Nexa, apagá la luz", source="human_direct",
                speaker=None, audio=make_audio(),
            )

        assert any("[TextualWake]" in r.message for r in caplog.records)


# ==================== TextualWakeDetector: dedup propio ====================

class TestDetectorDedupSelf:
    async def test_second_utterance_within_window_does_not_redispatch(self):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        first = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        clock.advance(3.0)  # < dedup_window_s=8.0
        second = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, prendé la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert first is True
        assert second is False
        assert dispatch.await_count == 1

    async def test_utterance_after_window_redispatches(self):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        first = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        clock.advance(9.0)  # > dedup_window_s=8.0
        second = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, prendé la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert first is True
        assert second is True
        assert dispatch.await_count == 2

    async def test_self_dedup_is_per_room(self):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        salon = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        clock.advance(1.0)  # dentro del window, pero otra room
        living = await detector.maybe_dispatch(
            room_id="living", text="Nexa, prendé la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert salon is True
        assert living is True
        assert dispatch.await_count == 2

    async def test_dedup_self_logs_info(self, caplog):
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        clock.advance(3.0)  # < dedup_window_s=8.0

        with caplog.at_level(logging.INFO):
            await detector.maybe_dispatch(
                room_id="salon", text="Nexa, prendé la luz", source="human_direct",
                speaker=None, audio=make_audio(),
            )

        assert any(
            "[TextualWake]" in r.message and "dedup_self" in r.message
            for r in caplog.records
        )


# ==================== TextualWakeDetector: errores de dispatch ====================

class TestDetectorDispatchErrors:
    async def test_dispatch_exception_returns_false_without_raising(self):
        dispatch = AsyncMock(side_effect=RuntimeError("router down"))
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_awaited_once()

    async def test_dispatch_error_does_not_poison_self_dedup(self):
        # Un intento fallido no registra ts propio → el reintento inmediato
        # NO debe verse bloqueado por el dedup propio.
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(side_effect=[RuntimeError("boom"), {"success": True}])
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
        )

        first = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        second = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert first is False
        assert second is True
        assert dispatch.await_count == 2

    async def test_dispatch_error_logs_error_level(self, caplog):
        dispatch = AsyncMock(side_effect=RuntimeError("router down"))
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        with caplog.at_level(logging.ERROR):
            await detector.maybe_dispatch(
                room_id="salon", text="Nexa, apagá la luz", source="human_direct",
                speaker=None, audio=make_audio(),
            )

        assert any(
            r.levelno == logging.ERROR and "[TextualWake]" in r.message
            for r in caplog.records
        )


# ==================== TextualWakeDetector: oráculo de misses (Etapa A) ====================

class TestDetectorOnMissedFn:
    """on_missed_fn (spec 2026-07-25 Etapa A): un disparo textual EXITOSO
    (pasó tv/self, match, dedup_acoustic, dedup_self y el gate de vad) es por
    construcción la prueba de que el wake acústico falló esa "nexa" — el
    callback alimenta el bucket missed/ de WakeClipWriter. NO se llama en
    ningún camino de skip: esos casos no prueban nada sobre el acústico."""

    async def test_called_on_successful_dispatch_with_room_and_audio(self):
        dispatch = AsyncMock(return_value={"success": True})
        calls = []
        audio = make_audio()
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=audio,
        )

        assert result is True
        assert calls == [("salon", audio)]

    async def test_not_called_when_dedup_acoustic(self):
        # El acústico SÍ disparó hace poco — no es un miss.
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock()
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: clock.t - 3.0,
            now_fn=clock,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert calls == []

    async def test_not_called_when_low_vad(self):
        # Gate de vad bloqueado = probable garble del STT, no habla dirigida
        # — capturarlo como "miss" contaminaría el dataset con no-positivos.
        dispatch = AsyncMock()
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            min_vad=0.50,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        await detector.maybe_dispatch(
            room_id="cocina", text="Next up.", source="unknown",
            speaker=None, audio=make_audio(), vad_prob=0.36,
        )

        assert calls == []

    async def test_not_called_when_no_match(self):
        dispatch = AsyncMock()
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        await detector.maybe_dispatch(
            room_id="salon", text="apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert calls == []

    async def test_not_called_when_dedup_self(self):
        # El propio canal textual ya disparó hace poco para esta room — la
        # re-transcripción solapada sería un clip casi-duplicado del mismo
        # evento de habla, no un miss nuevo.
        clock = FakeClock(t=1000.0)
        dispatch = AsyncMock(return_value={"success": True})
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            now_fn=clock,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )
        clock.advance(3.0)  # < dedup_window_s=8.0
        second_audio = make_audio()
        await detector.maybe_dispatch(
            room_id="salon", text="Nexa, prendé la luz", source="human_direct",
            speaker=None, audio=second_audio,
        )

        # Solo el primer disparo cuenta como miss — el segundo fue absorbido
        # por dedup_self, no debe agregar un segundo call.
        assert len(calls) == 1
        assert calls[0][0] == "salon"

    async def test_not_called_when_source_is_tv(self):
        dispatch = AsyncMock()
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="tv",
            speaker=None, audio=make_audio(),
        )

        assert calls == []

    async def test_none_is_safe_default(self):
        # Default None (nadie inyectó el writer) no debe romper el dispatch.
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is True

    async def test_exception_in_callback_does_not_break_dispatch(self, caplog):
        # Fail-open, mismo patrón que dispatch_fn: un writer roto jamás debe
        # tumbar el canal textual.
        dispatch = AsyncMock(return_value={"success": True})

        def broken(room_id, audio):
            raise RuntimeError("disco lleno")

        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            on_missed_fn=broken,
        )

        with caplog.at_level(logging.WARNING):
            result = await detector.maybe_dispatch(
                room_id="salon", text="Nexa, apagá la luz", source="human_direct",
                speaker=None, audio=make_audio(),
            )

        assert result is True
        dispatch.assert_awaited_once()
        # No alcanza con "no rompió" — el WARNING es la ÚNICA señal de que
        # on_missed_fn falló (sin esto, un logger.warning borrado por error
        # se volvería silencioso de verdad y ningún test lo detectaría).
        assert any(
            r.levelno == logging.WARNING and "[TextualWake]" in r.message
            and "on_missed_fn error" in r.message
            for r in caplog.records
        )

    async def test_called_even_when_dispatch_fn_raises(self):
        # El miss acústico ya ocurrió en cuanto pasamos los gates — es
        # independiente de que el router downstream falle.
        dispatch = AsyncMock(side_effect=RuntimeError("router down"))
        calls = []
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            on_missed_fn=lambda room_id, aud: calls.append((room_id, aud)),
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="Nexa, apagá la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        assert len(calls) == 1


# ==================== TextualWakeDetector: wiring del matcher ====================

class TestDetectorMatcherWiring:
    async def test_uses_default_variants_and_edit_distance(self):
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="neza apaga la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is True

    async def test_custom_max_edit_distance_zero_rejects_fuzzy(self):
        dispatch = AsyncMock(return_value={"success": True})
        detector = TextualWakeDetector(
            dispatch_fn=dispatch,
            last_acoustic_command_ts_fn=lambda room_id: 0.0,
            max_edit_distance=0,
        )

        result = await detector.maybe_dispatch(
            room_id="salon", text="neza apaga la luz", source="human_direct",
            speaker=None, audio=make_audio(),
        )

        assert result is False
        dispatch.assert_not_awaited()
