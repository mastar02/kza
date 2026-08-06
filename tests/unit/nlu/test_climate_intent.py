"""Binary climate/AC intent classifier -- unit level, no network.

Everything here is deterministic: the router is mocked, so these are plain
assertEqual tests. The probabilistic part lives in benchmarks/router/, scored
against a threshold, never asserted for equality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nlu.climate_intent import (
    CLIMATE_PROMPT,
    PROMPT_FINGERPRINT,
    ClimateIntent,
    ClimateIntentClassifier,
    has_contested_vocabulary,
)


def _router(reply: str):
    r = MagicMock()
    r.complete = AsyncMock(return_value=reply)
    return r


# --- gate -----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "prendé el clima",
    "poné la temperatura en 22",
    "apagá el termostato",
    "prendé la calefacción",
    "apagá el aire",
    "subí los grados",
    "QUE TEMPERATURA HACE",          # case-insensitive
    "bajá la calefaccion",           # sin acento
])
def test_gate_fires_on_contested_vocabulary(text):
    assert has_contested_vocabulary(text) is True


@pytest.mark.parametrize("text", [
    "prendé la luz del living",
    "poné música de Spinetta",
    "agregá leche a la lista",
    "recordame sacar la basura",
    "por qué el cielo es azul",
    "",
])
def test_gate_stays_quiet_without_contested_vocabulary(text):
    assert has_contested_vocabulary(text) is False


def test_gate_does_not_match_substrings():
    # "aire" inside "airear", "grados" inside "posgrados": the gate must use
    # word boundaries or it becomes the very substring bug it replaces.
    assert has_contested_vocabulary("hay que airear la pieza") is False
    assert has_contested_vocabulary("terminé los posgrados") is False


# --- parsing --------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_action_for_action_label():
    c = ClimateIntentClassifier(_router("ACCION_AIRE"))
    assert await c.classify("prendé el clima") == ClimateIntent.ACTION


@pytest.mark.asyncio
async def test_returns_query_for_query_label():
    c = ClimateIntentClassifier(_router("PREGUNTA_TIEMPO"))
    assert await c.classify("está el clima lindo") == ClimateIntent.QUERY


@pytest.mark.asyncio
async def test_tolerates_whitespace_and_case():
    c = ClimateIntentClassifier(_router("  accion_aire\n"))
    assert await c.classify("prendé el clima") == ClimateIntent.ACTION


@pytest.mark.parametrize("reply", [
    "",                                  # empty
    "   ",                               # blank
    "OTRO",                              # off-label
    "clima",                             # the ambiguous word itself
    "clima\nRespuesta: Lo siento, no",   # contaminated output seen in probing
    "no estoy seguro",                   # hallucination
    "ACCION",                            # partial label, not the contract
])
@pytest.mark.asyncio
async def test_unrecognised_output_returns_none(reply):
    c = ClimateIntentClassifier(_router(reply))
    assert await c.classify("prendé el clima") is None


# --- failure modes --------------------------------------------------------

@pytest.mark.asyncio
async def test_router_exception_returns_none():
    r = MagicMock()
    r.complete = AsyncMock(side_effect=ConnectionError("8101 down"))
    assert await ClimateIntentClassifier(r).classify("prendé el clima") is None


@pytest.mark.asyncio
async def test_timeout_returns_none():
    async def _slow(*a, **k):
        await asyncio.sleep(1.0)
        return "ACCION_AIRE"

    r = MagicMock()
    r.complete = _slow
    c = ClimateIntentClassifier(r, timeout_s=0.05)
    assert await c.classify("prendé el clima") is None


@pytest.mark.asyncio
async def test_timeout_is_enforced_quickly():
    async def _slow(*a, **k):
        await asyncio.sleep(1.0)

    r = MagicMock()
    r.complete = _slow
    c = ClimateIntentClassifier(r, timeout_s=0.05)
    t0 = asyncio.get_running_loop().time()
    await c.classify("prendé el clima")
    assert asyncio.get_running_loop().time() - t0 < 0.5


@pytest.mark.asyncio
async def test_no_router_returns_none():
    assert await ClimateIntentClassifier(None).classify("prendé el clima") is None


@pytest.mark.asyncio
async def test_non_string_response_abstains_loudly(caplog):
    """Wiring an LLMRouter instead of a FastRouter must not fail silently.

    LLMRouter.complete() returns a RouterResult, not str. Without an explicit
    guard that lands as an AttributeError swallowed by classify(), i.e. a
    classifier that abstains on 100% of calls while looking wired up.
    """
    class _RouterResult:      # stand-in for src.llm.router.RouterResult
        text = "ACCION_AIRE"

    r = MagicMock()
    r.complete = AsyncMock(return_value=_RouterResult())

    with caplog.at_level("WARNING"):
        assert await ClimateIntentClassifier(r).classify("prendé el clima") is None

    assert any("expected str" in rec.message for rec in caplog.records), \
        "a silent abstention on every call must not be debug-level"


# --- call contract --------------------------------------------------------

@pytest.mark.asyncio
async def test_calls_router_with_deterministic_parameters():
    r = _router("ACCION_AIRE")
    await ClimateIntentClassifier(r).classify("prendé el clima")

    kwargs = r.complete.call_args.kwargs
    assert kwargs["temperature"] == 0.0, "temp must be 0: determinism is the testing contract"
    assert kwargs["max_tokens"] == 10
    assert "\n" in kwargs["stop"]
    assert "prendé el clima" in r.complete.call_args.args[0]


# --- prompt tripwire ------------------------------------------------------

def test_prompt_fingerprint_is_pinned():
    """If this fails you edited CLIMATE_PROMPT.

    The prompt is a contract, not decoration: the few-shot examples carry
    measured accuracy. Re-run the eval before updating this constant:

        .venv/bin/python3 benchmarks/router/climate_eval.py

    Then paste the new fingerprint printed by the runner.
    """
    import hashlib
    actual = hashlib.sha256(CLIMATE_PROMPT.encode()).hexdigest()[:16]
    assert actual == PROMPT_FINGERPRINT, (
        f"prompt changed (fingerprint {actual}); re-run the eval and update "
        f"PROMPT_FINGERPRINT"
    )
