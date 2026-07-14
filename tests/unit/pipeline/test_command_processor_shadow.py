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
