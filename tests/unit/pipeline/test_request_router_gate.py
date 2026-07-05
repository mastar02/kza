"""Integración: request_router usa CommandAcceptanceGate (accept/reject)."""
import sys
from unittest.mock import MagicMock, AsyncMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np
import pytest
from src.nlu.command_gate import CommandAcceptanceGate
from src.pipeline.request_router import RequestRouter


def _make_cmd(text="nexa prendé la luz"):
    cmd = MagicMock()
    cmd.text = text
    cmd.user = None
    cmd.emotion = None
    cmd.timings = {}
    cmd.stt_confidence = None
    return cmd


def _router(gate):
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=_make_cmd())
    orch = MagicMock()
    orch.process = AsyncMock(
        return_value=MagicMock(
            intent="domotics",
            response="ok",
            success=True,
            action=None,
            path=None,
            timings={},
            was_queued=False,
            queue_position=None,
        )
    )
    r = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=MagicMock(),
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        command_gate=gate,
    )
    return r, orch


@pytest.mark.asyncio
async def test_accepted_command_reaches_orchestrator():
    r, orch = _router(CommandAcceptanceGate(wake_words=("nexa",)))
    await r.process_command(np.zeros(16000, dtype="float32"))
    assert orch.process.called


@pytest.mark.asyncio
async def test_rejected_command_does_not_reach_orchestrator():
    # gate whose wake word never matches -> missing_wake reject for every text
    r, orch = _router(CommandAcceptanceGate(wake_words=("zzzz",)))
    result = await r.process_command(np.zeros(16000, dtype="float32"))
    assert not orch.process.called
    assert result["success"] is False
    assert result["intent"].startswith("gate_rejected:missing_wake")


# ---- LLM router confidence gate (2026-06-02) ----
# Charla ambiente que el LLMCommandRouter 7B marca is_command=True con confianza
# baja NO debe ejecutar una acción fantasma. La gramática (alta confianza) NO se
# ve afectada porque corre por el fast-path antes del LLM.
from src.nlu.llm_router import CommandClassification


def _router_with_llm(classification, text="hola esto es una charla cualquiera"):
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=_make_cmd(text))
    orch = MagicMock()
    orch.process = AsyncMock(return_value=MagicMock(
        intent="domotics", response="ok", success=True, action=None, path=None,
        timings={}, was_queued=False, queue_position=None,
    ))
    llm = MagicMock()
    llm.classify = AsyncMock(return_value=classification)
    r = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=MagicMock(),
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        command_gate=CommandAcceptanceGate(wake_words=()),  # acepta (prod openwakeword)
        llm_command_router=llm,
    )
    return r, orch, llm


@pytest.mark.asyncio
async def test_low_confidence_llm_command_rejected():
    cls = CommandClassification(is_command=True, confidence=0.4, intent="turn_on", entity_hint="light")
    r, orch, llm = _router_with_llm(cls)
    result = await r.process_command(np.zeros(16000, dtype="float32"))
    assert llm.classify.called
    assert not orch.process.called, "comando de baja confianza no debe dispatchar"
    assert result["success"] is False
    assert result["intent"].startswith("low_confidence")


@pytest.mark.asyncio
async def test_high_confidence_llm_command_dispatched():
    cls = CommandClassification(is_command=True, confidence=0.85, intent="turn_on", entity_hint="light")
    r, orch, llm = _router_with_llm(cls)
    await r.process_command(np.zeros(16000, dtype="float32"))
    assert orch.process.called, "comando de alta confianza SÍ debe dispatchar"


# ---- Earcon wiring tests (2026-06-15) ----
# Validan que el earcon SUENA (o NO suena) según el motivo y el wake, sin
# necesitar el pipeline de audio real. Usan el earcon_cfg explícito para
# habilitar el earcon en el router (el default es disabled).

_EARCON_CFG_ENABLED = {
    "enabled": True,
    "min_wake_score": 0.55,
    "min_rms": 0.02,
    "reasons": ["empty", "empty_after_norm", "high_compression",
                "low_confidence", "unverified_intent"],
}


def _router_with_earcon(text="nexa prendé la luz", gate=None):
    """Construye un router con earcon habilitado y response_handler espiado."""
    command_processor = MagicMock()
    command_processor.process_command = AsyncMock(return_value=_make_cmd(text))
    orch = MagicMock()
    orch.process = AsyncMock(
        return_value=MagicMock(
            intent="domotics",
            response="ok",
            success=True,
            action=None,
            path=None,
            timings={},
            was_queued=False,
            queue_position=None,
        )
    )
    response_handler = MagicMock()
    response_handler.play_earcon = MagicMock()
    if gate is None:
        gate = CommandAcceptanceGate(wake_words=("nexa",))
    r = RequestRouter(
        command_processor=command_processor,
        orchestrator=orch,
        orchestrator_enabled=True,
        response_handler=response_handler,
        audio_manager=MagicMock(),
        wake_words=("nexa",),
        command_gate=gate,
        earcon_cfg=_EARCON_CFG_ENABLED,
    )
    return r, response_handler


@pytest.mark.asyncio
async def test_noise_phrase_rejection_does_not_play_earcon():
    """TV text ('gracias por ver') — gate rechaza con reason noise_phrase:...
    El earcon NO debe sonar aunque el wake sea fuerte (wake_score default=1.0).
    Protege contra TV silenciada por error de sonido."""
    # 'gracias por ver' es un _NOISE_PHRASES canonical, el gate lo rechaza con
    # reason que empieza en 'noise_phrase' → blocked por _NOISE_PREFIXES.
    r, rh = _router_with_earcon(text="gracias por ver el video")
    # Audio con energía real para que rms > 0.02
    audio = np.full(16000, 0.1, dtype="float32")
    await r.process_command(audio)
    rh.play_earcon.assert_not_called()


@pytest.mark.asyncio
async def test_empty_stt_with_strong_wake_plays_earcon():
    """STT vacío con wake fuerte (wake_score=1.0 default) y audio con energía
    real → earcon debe sonar (fix #1: el path antes era silent return)."""
    r, rh = _router_with_earcon(text="")  # STT retorna vacío
    # Audio con energía real para que rms > 0.02
    audio = np.full(16000, 0.1, dtype="float32")
    await r.process_command(audio)
    rh.play_earcon.assert_called_once()
