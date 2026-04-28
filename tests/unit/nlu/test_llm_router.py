"""Tests para LLMCommandRouter — clasificador con vLLM 7B mockeado."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from src.nlu.llm_router import (
    CommandClassification,
    KNOWN_INTENTS,
    LLMCommandRouter,
)


def _make_router(llm_response: str | list[str], timeout_s: float = 5.0):
    """Helper: construir LLMCommandRouter con FastRouter mockeado."""
    fast = MagicMock()
    if isinstance(llm_response, str):
        responses = [llm_response]
    else:
        responses = llm_response
    call_count = {"n": 0}

    def fake_generate(prompts, max_tokens=256, temperature=0.3):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return [responses[idx]]

    fast.generate = fake_generate
    return LLMCommandRouter(fast_router=fast, timeout_s=timeout_s)


# ---------------- Parsing ----------------

@pytest.mark.asyncio
async def test_classify_returns_valid_command():
    response = json.dumps({
        "is_command": True,
        "confidence": 0.95,
        "rejection_reason": None,
        "intent": "turn_off",
        "entity_hint": "luz del escritorio",
        "slots": {},
    })
    router = _make_router(response)
    result = await router.classify("Nexa apagá la luz del escritorio")
    assert result.is_command is True
    assert result.intent == "turn_off"
    assert result.entity_hint == "luz del escritorio"
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_classify_returns_rejection():
    response = json.dumps({
        "is_command": False,
        "confidence": 0.9,
        "rejection_reason": "tv_replay",
        "intent": None,
        "entity_hint": None,
        "slots": {},
    })
    router = _make_router(response)
    result = await router.classify("Nexa bajá la luz al cincuenta por ciento")
    assert result.is_command is False
    assert result.rejection_reason == "tv_replay"
    assert result.intent is None


@pytest.mark.asyncio
async def test_classify_with_slots():
    response = json.dumps({
        "is_command": True,
        "confidence": 0.9,
        "rejection_reason": None,
        "intent": "set_brightness",
        "entity_hint": "luz del escritorio",
        "slots": {"brightness_pct": 50, "rgb_color": None, "value": None},
    })
    router = _make_router(response)
    result = await router.classify("Nexa luz al cincuenta por ciento")
    assert result.is_command is True
    assert result.slots == {"brightness_pct": 50}  # None values filtrados


@pytest.mark.asyncio
async def test_unknown_intent_treated_as_invalid():
    response = json.dumps({
        "is_command": True,
        "confidence": 0.8,
        "intent": "make_coffee",  # no está en KNOWN_INTENTS
        "entity_hint": "cafetera",
        "slots": {},
    })
    router = _make_router(response)
    result = await router.classify("Nexa hacé café")
    assert result.is_command is False
    assert result.rejection_reason == "unknown_intent"


@pytest.mark.asyncio
async def test_classify_handles_text_around_json():
    """Qwen a veces agrega preámbulo aunque pidamos solo JSON."""
    response = "Acá te dejo el JSON:\n" + json.dumps({
        "is_command": True,
        "intent": "turn_on",
        "entity_hint": "luz",
        "slots": {},
    }) + "\nEspero que ayude."
    router = _make_router(response)
    result = await router.classify("Nexa prendé la luz")
    assert result.is_command is True
    assert result.intent == "turn_on"


@pytest.mark.asyncio
async def test_classify_handles_invalid_json():
    router = _make_router("esto no es JSON {bla bla}")
    result = await router.classify("Nexa prendé la luz")
    assert result.is_command is False
    assert result.rejection_reason == "noise"


@pytest.mark.asyncio
async def test_classify_handles_no_json_at_all():
    router = _make_router("Lo siento, no entendí.")
    result = await router.classify("Nexa prendé la luz")
    assert result.is_command is False
    assert result.rejection_reason == "noise"


@pytest.mark.asyncio
async def test_classify_handles_empty_response():
    router = _make_router("")
    result = await router.classify("Nexa prendé la luz")
    assert result.is_command is False


# ---------------- Errors / timeouts ----------------

@pytest.mark.asyncio
async def test_classify_timeout_returns_invalid():
    """Si el LLM tarda demasiado, devolvemos rechazo (caller decide fallback)."""
    fast = MagicMock()

    def slow_generate(prompts, max_tokens=256, temperature=0.3):
        time.sleep(0.5)  # excede el timeout configurado abajo
        return [json.dumps({"is_command": True, "intent": "turn_on"})]

    fast.generate = slow_generate
    router = LLMCommandRouter(fast_router=fast, timeout_s=0.05)
    result = await router.classify("Nexa prendé")
    assert result.is_command is False
    assert "timeout" in result.raw_response


@pytest.mark.asyncio
async def test_classify_handles_llm_exception():
    fast = MagicMock()
    fast.generate = MagicMock(side_effect=ConnectionError("vLLM down"))
    router = LLMCommandRouter(fast_router=fast)
    result = await router.classify("Nexa prendé")
    assert result.is_command is False
    assert "error" in result.raw_response


# ---------------- Historial ----------------

def test_record_command_appends_history():
    router = _make_router("{}")
    router.record_command("Nexa apagá la luz del escritorio", "turn_off")
    router.record_command("Nexa bajá la temperatura", "set_temperature")
    assert len(router._history) == 2


def test_history_block_renders_with_recent_commands():
    router = _make_router("{}")
    router.record_command("Nexa apagá la luz", "turn_off")
    block = router._build_history_block()
    assert "turn_off" in block
    assert "Nexa apagá la luz" in block


def test_history_block_filters_expired():
    router = _make_router("{}")
    router.history_ttl_s = 0.05
    router.record_command("Nexa apagá la luz", "turn_off")
    time.sleep(0.10)
    block = router._build_history_block()
    assert "vacío" in block.lower()


def test_history_max_size_respected():
    router = _make_router("{}")
    router.max_history = 3
    router._history = router._history.__class__(maxlen=3)  # rebuild deque con maxlen
    for i in range(5):
        router.record_command(f"comando {i}", "turn_on")
    assert len(router._history) == 3


@pytest.mark.asyncio
async def test_history_appears_in_prompt():
    """El prompt debe incluir el historial para que el LLM detecte replays."""
    router = _make_router(json.dumps({"is_command": False}))
    router.record_command("Nexa apagá la luz del escritorio", "turn_off")

    captured_prompts = {}

    def capture_generate(prompts, max_tokens, temperature):
        captured_prompts["p"] = prompts[0]
        return [json.dumps({"is_command": False})]

    router.router.generate = capture_generate
    await router.classify("Nexa apagá la luz del escritorio")
    assert "Nexa apagá la luz del escritorio" in captured_prompts["p"]
    assert "turn_off" in captured_prompts["p"]


# ---------------- Smoke ----------------

def test_known_intents_is_nonempty():
    assert "turn_on" in KNOWN_INTENTS
    assert "turn_off" in KNOWN_INTENTS
    assert "set_brightness" in KNOWN_INTENTS


def test_command_classification_dataclass_defaults():
    c = CommandClassification(is_command=False)
    assert c.confidence == 0.0
    assert c.intent is None
    assert c.slots == {}
