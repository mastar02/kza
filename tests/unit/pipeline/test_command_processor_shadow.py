"""Tests: CommandProcessor shadow STT (A/B en vivo, fire-and-forget).

El shadow transcribe el MISMO audio con un 2º motor solo para loguear la
comparación; NUNCA debe bloquear la respuesta ni cambiar el texto primario.
"""
import asyncio
import logging

import numpy as np
import pytest

from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult


class _FakeSTT:
    def transcribe(self, audio, sr=16000):
        return "prendé la luz", 5.0

    def transcribe_with_confidence(self, audio, sr=16000):
        return STTResult("prendé la luz", 5.0, no_speech_prob=0.15, avg_logprob=-0.4)


class _FakeShadowSTT:
    """Shadow que registra que se lo llamó y devuelve otro texto/latencia."""

    def __init__(self):
        self.calls = 0
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe_with_confidence(self, audio, sr=16000):
        self.calls += 1
        return STTResult("prende la luz", 12.0)


AUDIO = np.zeros(16000, dtype="float32")


@pytest.mark.asyncio
async def test_shadow_does_not_change_primary_text():
    shadow = _FakeShadowSTT()
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=shadow)
    result = await cp.process_command(AUDIO, use_parallel=False)
    # el texto/confianza que sale es el del primario, nunca el del shadow
    assert result.text == "prendé la luz"
    assert result.stt_confidence.avg_logprob == pytest.approx(-0.4)


@pytest.mark.asyncio
async def test_shadow_runs_in_background():
    shadow = _FakeShadowSTT()
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=shadow)
    await cp.process_command(AUDIO, use_parallel=False)
    # fire-and-forget: puede no haber corrido aún al volver; drenar las tasks
    await asyncio.gather(*cp._shadow_tasks)
    assert shadow.calls == 1


@pytest.mark.asyncio
async def test_shadow_logs_comparison(caplog):
    shadow = _FakeShadowSTT()
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=shadow)
    with caplog.at_level(logging.INFO):
        await cp.process_command(AUDIO, use_parallel=False)
        await asyncio.gather(*cp._shadow_tasks)
    line = next((r.message for r in caplog.records if "[STT-shadow]" in r.message), None)
    assert line is not None
    assert "prendé la luz" in line and "prende la luz" in line


@pytest.mark.asyncio
async def test_shadow_error_never_breaks_command(caplog):
    class _BoomShadow:
        def transcribe_with_confidence(self, audio, sr=16000):
            raise RuntimeError("boom")

    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=_BoomShadow())
    result = await cp.process_command(AUDIO, use_parallel=False)
    await asyncio.gather(*cp._shadow_tasks)  # no debe propagar
    assert result.text == "prendé la luz"


@pytest.mark.asyncio
async def test_no_shadow_by_default():
    cp = CommandProcessor(stt=_FakeSTT())
    result = await cp.process_command(AUDIO, use_parallel=False)
    assert cp.shadow_stt is None
    assert not cp._shadow_tasks
    assert result.text == "prendé la luz"


def test_load_models_loads_shadow():
    shadow = _FakeShadowSTT()
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=shadow)
    cp.load_models()
    assert shadow.loaded


# ==================== Veto híbrido (shadow_veto=True) ====================
# Si Parakeet (shadow) devuelve vacío y el primario texto → descartar la
# captura. Evidencia: 167/167 vacíos de prod eran basura (auditoría 07-24).


class _EmptyShadowSTT:
    """Shadow que calla (caso alucinación del primario)."""

    def transcribe_with_confidence(self, audio, sr=16000):
        return STTResult("", 12.0)


@pytest.mark.asyncio
async def test_veto_discards_when_shadow_empty(caplog):
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=_EmptyShadowSTT(), shadow_veto=True)
    with caplog.at_level(logging.INFO):
        result = await cp.process_command(AUDIO, use_parallel=False)
    assert result.text == ""
    assert result.success is False
    assert any("[STT-veto] descartado" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_veto_keeps_text_when_shadow_nonempty():
    # Shadow con texto (aunque sea distinto/gibberish) → NO veta: el único
    # caso confiable es el vacío (el gibberish inglés apareció en comandos
    # reales — 'In X are a parallel' sobre 'Nexa apagá la luz').
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=_FakeShadowSTT(), shadow_veto=True)
    result = await cp.process_command(AUDIO, use_parallel=False)
    assert result.text == "prendé la luz"
    assert result.success is True


@pytest.mark.asyncio
async def test_veto_does_not_discard_empty_primary():
    # Primario vacío + shadow vacío → no hay nada que vetar (ya era descarte).
    class _EmptySTT:
        def transcribe_with_confidence(self, audio, sr=16000):
            return STTResult("", 5.0)

    cp = CommandProcessor(stt=_EmptySTT(), shadow_stt=_EmptyShadowSTT(), shadow_veto=True)
    result = await cp.process_command(AUDIO, use_parallel=False)
    assert result.success is False


@pytest.mark.asyncio
async def test_veto_fail_open_on_shadow_error(caplog):
    class _BoomShadow:
        def transcribe_with_confidence(self, audio, sr=16000):
            raise RuntimeError("boom")

    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=_BoomShadow(), shadow_veto=True)
    with caplog.at_level(logging.WARNING):
        result = await cp.process_command(AUDIO, use_parallel=False)
    # error del shadow → SIN veto, el comando pasa intacto
    assert result.text == "prendé la luz"
    assert any("[STT-veto] shadow no disponible" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_veto_fail_open_on_timeout():
    import time as _time

    class _SlowShadow:
        def transcribe_with_confidence(self, audio, sr=16000):
            _time.sleep(0.3)
            return STTResult("", 300.0)

    cp = CommandProcessor(
        stt=_FakeSTT(), shadow_stt=_SlowShadow(),
        shadow_veto=True, shadow_veto_timeout_s=0.05,
    )
    result = await cp.process_command(AUDIO, use_parallel=False)
    # timeout → sin veto (jamás retrasar/descartar por un shadow colgado)
    assert result.text == "prendé la luz"


@pytest.mark.asyncio
async def test_veto_logs_shadow_comparison_same_format(caplog):
    # El modo veto también emite [STT-shadow] (los scripts de cosecha lo leen).
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=_FakeShadowSTT(), shadow_veto=True)
    with caplog.at_level(logging.INFO):
        await cp.process_command(AUDIO, use_parallel=False)
    line = next((r.message for r in caplog.records if "[STT-shadow]" in r.message), None)
    assert line is not None and "primary=" in line and "shadow_ms=" in line


@pytest.mark.asyncio
async def test_observer_mode_unchanged_when_veto_off():
    # shadow_veto=False (default) → comportamiento fire-and-forget previo.
    shadow = _FakeShadowSTT()
    cp = CommandProcessor(stt=_FakeSTT(), shadow_stt=shadow, shadow_veto=False)
    result = await cp.process_command(AUDIO, use_parallel=False)
    assert result.text == "prendé la luz"
    await asyncio.gather(*cp._shadow_tasks)
    assert shadow.calls == 1


@pytest.mark.asyncio
async def test_veto_applies_on_parallel_path():
    # El path paralelo (producción real: speaker_id presente) también veta.
    from unittest.mock import MagicMock

    fake_speaker_id = MagicMock()
    fake_speaker_id.identify.return_value = MagicMock(
        is_known=False, confidence=0.0, user_id=None
    )
    fake_user_manager = MagicMock()
    fake_user_manager.get_all_embeddings.return_value = {}
    cp = CommandProcessor(
        stt=_FakeSTT(), speaker_identifier=fake_speaker_id,
        user_manager=fake_user_manager,
        shadow_stt=_EmptyShadowSTT(), shadow_veto=True,
    )
    result = await cp.process_command(AUDIO, use_parallel=True)
    assert result.text == ""
    assert result.success is False
