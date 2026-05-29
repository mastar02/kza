"""Tests: LLMRouter enforces ep.timeout_s and fails over on slow endpoints."""

import asyncio

import pytest

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind, LLMEndpoint


class _SlowClient:
    async def complete(self, prompt, max_tokens=256, **kwargs):
        await asyncio.sleep(2.0)
        return "tarde"


class _FastClient:
    async def complete(self, prompt, max_tokens=256, **kwargs):
        return "rápido"


@pytest.mark.asyncio
async def test_router_times_out_slow_endpoint_and_fails_over(tmp_path):
    cd = CooldownManager(persistence_path=tmp_path / "cd.json")
    eps = [
        LLMEndpoint(id="slow", kind=EndpointKind.CLOUD, client=_SlowClient(), priority=1, timeout_s=0.1),
        LLMEndpoint(id="fast", kind=EndpointKind.HTTP_REASONER, client=_FastClient(), priority=2, timeout_s=5.0),
    ]
    router = LLMRouter(endpoints=eps, cooldown_manager=cd)
    result = await router.complete("hola", max_tokens=16)
    assert result.text == "rápido"
    assert result.endpoint_id == "fast"
