"""Tests para src/nlu/llm_gate.py — LLMGate binario thin.

Mockeamos FastRouter.generate para validar:
- valid=true → GateResult.valid=True
- valid=false → GateResult.valid=False
- Timeout → GateResult.valid=False (política conservadora)
- JSON inválido → GateResult.valid=False
- Excepción del router → GateResult.valid=False
"""
from __future__ import annotations

import asyncio
import pytest

from src.nlu.llm_gate import LLMGate


class _StubRouter:
    """Mock de FastRouter para tests determinísticos."""

    def __init__(self, response: str | None = None, raises: Exception | None = None,
                 sleep_s: float = 0.0):
        self.response = response
        self.raises = raises
        self.sleep_s = sleep_s
        self.calls: list[tuple] = []

    def generate(self, prompts, max_tokens, temperature, extra_body=None):
        self.calls.append((prompts, max_tokens, temperature, extra_body))
        if self.sleep_s:
            import time as _t
            _t.sleep(self.sleep_s)
        if self.raises:
            raise self.raises
        return [self.response or ""]


@pytest.mark.asyncio
async def test_gate_valid_true() -> None:
    router = _StubRouter(response='{"valid": true}')
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("apagá la luz", "turn_off", "escritorio")
    assert result.valid is True
    # Comprobamos que se pasó el extra_body con response_format json_schema
    extra_body = router.calls[0][3]
    assert "response_format" in extra_body
    assert extra_body["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_gate_valid_false() -> None:
    router = _StubRouter(response='{"valid": false}')
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("ya apagué la luz", "turn_off", "escritorio")
    assert result.valid is False


@pytest.mark.asyncio
async def test_gate_timeout_defaults_invalid() -> None:
    """Timeout debe devolver valid=False (no ejecutar si no estamos seguros)."""
    router = _StubRouter(response='{"valid": true}', sleep_s=1.0)
    gate = LLMGate(router, timeout_s=0.05)
    result = await gate.validate("apagá la luz", "turn_off", None)
    assert result.valid is False
    assert result.raw_response == "<timeout>"


@pytest.mark.asyncio
async def test_gate_exception_defaults_invalid() -> None:
    router = _StubRouter(raises=ConnectionError("vLLM down"))
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("apagá la luz", "turn_off", None)
    assert result.valid is False
    assert "ConnectionError" in result.raw_response


@pytest.mark.asyncio
async def test_gate_invalid_json_defaults_invalid() -> None:
    router = _StubRouter(response="not a json")
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("apagá la luz", "turn_off", None)
    assert result.valid is False


@pytest.mark.asyncio
async def test_gate_empty_response_defaults_invalid() -> None:
    router = _StubRouter(response="")
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("apagá la luz", "turn_off", None)
    assert result.valid is False


@pytest.mark.asyncio
async def test_gate_prompt_includes_intent_and_entity() -> None:
    """El prompt debe incluir intent + entity_hint para que el modelo razone."""
    router = _StubRouter(response='{"valid": true}')
    gate = LLMGate(router, timeout_s=2.0)
    await gate.validate("texto random", "set_brightness", "living")
    prompt = router.calls[0][0][0]
    assert "set_brightness" in prompt
    assert "living" in prompt
    assert "texto random" in prompt


@pytest.mark.asyncio
async def test_gate_prompt_handles_no_entity() -> None:
    """Cuando entity_hint es None, el prompt debe igual armarse coherente."""
    router = _StubRouter(response='{"valid": true}')
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("subí el volumen", "set_volume", None)
    assert result.valid is True
    prompt = router.calls[0][0][0]
    assert "set_volume" in prompt


@pytest.mark.asyncio
async def test_gate_elapsed_ms_is_set() -> None:
    router = _StubRouter(response='{"valid": true}')
    gate = LLMGate(router, timeout_s=2.0)
    result = await gate.validate("apagá", "turn_off", None)
    assert result.elapsed_ms >= 0
