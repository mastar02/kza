"""
End-to-end integration test: 7B falla → router cae a 72B → 7B vuelve.

Simula el escenario real:
1. Comando "prendé la luz" llega → router intenta 7B
2. 7B retorna 429 (mock) → cooldown 60s, rota a 72B
3. 72B responde OK → success persistido
4. Segundo comando: 7B sigue en cooldown → directo a 72B (sin latencia extra)
5. Después de avanzar el reloj +61s, 7B vuelve disponible
6. Tercer comando: 7B intenta primero → success
"""

import pytest
from unittest.mock import MagicMock

from src.llm.adapters import FastRouterAdapter, HttpReasonerAdapter
from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter, FallbackSummaryError
from src.llm.types import EndpointKind, LLMEndpoint


@pytest.fixture
def fake_clock(monkeypatch):
    """Mock time.time() for cooldown testing."""
    now = [10000.0]
    monkeypatch.setattr("src.llm.cooldown.time.time", lambda: now[0])
    return now


@pytest.fixture
def fast_router_mock():
    """Mock FastRouter client (returns list[str])."""
    fr = MagicMock(name="FastRouter")
    fr.generate = MagicMock(return_value=["respuesta-fast"])
    return fr


@pytest.fixture
def http_reasoner_mock():
    """Mock HttpReasoner client (returns OpenAI-compatible dict)."""
    hr = MagicMock(name="HttpReasoner")
    hr.return_value = {
        "choices": [{"text": "respuesta-deep"}],
        "usage": {"completion_tokens": 5},
    }
    return hr


@pytest.fixture
def router(tmp_path, fast_router_mock, http_reasoner_mock):
    """Create LLMRouter with cooldown manager and two endpoints."""
    cd_manager = CooldownManager(persistence_path=tmp_path / "cd.json")

    endpoints = [
        LLMEndpoint(
            id="fast",
            kind=EndpointKind.FAST_ROUTER,
            client=FastRouterAdapter(fast_router_mock),
            priority=0,
        ),
        LLMEndpoint(
            id="deep",
            kind=EndpointKind.HTTP_REASONER,
            client=HttpReasonerAdapter(http_reasoner_mock),
            priority=1,
        ),
    ]
    return LLMRouter(endpoints, cd_manager), cd_manager


class TestLLMFailoverE2E:
    """E2E test for failover with cooldown."""

    @pytest.mark.asyncio
    async def test_full_failover_scenario(
        self, router, fake_clock, fast_router_mock, http_reasoner_mock
    ):
        """Test full failover: 7B OK → 7B fails → 72B handles → cooldown expires."""
        rt, cd = router

        # ====== Turn 1: 7B funciona ======
        result = await rt.complete("prendé la luz", max_tokens=64)
        assert result.endpoint_id == "fast"
        assert result.text == "respuesta-fast"
        assert fast_router_mock.generate.call_count == 1
        assert http_reasoner_mock.call_count == 0

        # ====== Turn 2: 7B falla con 429 ======
        fast_router_mock.generate.side_effect = RuntimeError("429 Too Many Requests")
        result = await rt.complete("apagá el aire", max_tokens=64)
        assert result.endpoint_id == "deep"
        assert result.text == "respuesta-deep"
        assert fast_router_mock.generate.call_count == 2  # se intentó
        assert http_reasoner_mock.call_count == 1
        # 7B en cooldown
        assert cd.is_available("fast") is False

        # ====== Turn 3: 7B sigue en cooldown — directo a 72B ======
        fake_clock[0] = 10030.0  # +30s, todavía dentro del cooldown de 60s
        result = await rt.complete("subí persianas", max_tokens=64)
        assert result.endpoint_id == "deep"
        # 7B no se intentó (cooldown skip)
        assert fast_router_mock.generate.call_count == 2  # sin incremento
        assert http_reasoner_mock.call_count == 2

        # ====== Turn 4: cooldown expiró, 7B disponible y arreglado ======
        fake_clock[0] = 10100.0  # +100s, fuera del cooldown
        fast_router_mock.generate.side_effect = None  # se "arregla"
        fast_router_mock.generate.return_value = ["fast-recovered"]
        result = await rt.complete("temperatura ambiente", max_tokens=64)
        assert result.endpoint_id == "fast"
        assert result.text == "fast-recovered"
        # Después del éxito, error_count se resetea
        assert cd.get_state("fast").error_count == 0

    @pytest.mark.asyncio
    async def test_both_endpoints_down_raises(
        self, router, fake_clock, fast_router_mock, http_reasoner_mock
    ):
        """Test exception when all endpoints fail or are in cooldown."""
        rt, _ = router

        fast_router_mock.generate.side_effect = RuntimeError("429")
        http_reasoner_mock.side_effect = TimeoutError("read timeout")

        with pytest.raises(FallbackSummaryError) as exc_info:
            await rt.complete("hola", max_tokens=64)

        assert len(exc_info.value.attempts) == 2
        assert {a.endpoint_id for a in exc_info.value.attempts} == {"fast", "deep"}
