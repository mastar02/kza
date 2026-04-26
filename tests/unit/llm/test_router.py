"""Tests for LLMRouter candidate chain."""

import asyncio
import pytest
from typing import Optional
from unittest.mock import AsyncMock

from src.llm.cooldown import CooldownManager
from src.llm.router import LLMRouter, FallbackSummaryError
from src.llm.types import EndpointKind, ErrorKind, LLMEndpoint


class _FakeClient:
    """Cliente fake con interface mínima: async complete(prompt, **kw) -> str."""

    def __init__(self, response: str = "ok", raises: Optional[Exception] = None):
        self.response = response
        self.raises = raises
        self.calls = 0

    async def complete(self, prompt: str, max_tokens: int = 256, **_kw) -> str:
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.response


@pytest.fixture
def cd_manager(tmp_path):
    return CooldownManager(persistence_path=tmp_path / "cd.json")


@pytest.fixture
def primary_ok():
    return LLMEndpoint(
        id="primary", kind=EndpointKind.FAST_ROUTER,
        client=_FakeClient(response="primary-ok"), priority=0,
    )


@pytest.fixture
def secondary_ok():
    return LLMEndpoint(
        id="secondary", kind=EndpointKind.HTTP_REASONER,
        client=_FakeClient(response="secondary-ok"), priority=1,
    )


class TestLLMRouterHappyPath:
    @pytest.mark.asyncio
    async def test_uses_primary_when_available(self, cd_manager, primary_ok, secondary_ok):
        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.text == "primary-ok"
        assert result.endpoint_id == "primary"
        assert result.attempts == 1
        assert primary_ok.client.calls == 1
        assert secondary_ok.client.calls == 0

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_fails(self, cd_manager, secondary_ok):
        primary = LLMEndpoint(
            id="primary", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("429 rate limit")),
            priority=0,
        )
        router = LLMRouter([primary, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.text == "secondary-ok"
        assert result.endpoint_id == "secondary"
        assert result.attempts == 2

        # primary debe quedar en cooldown
        assert cd_manager.is_available("primary") is False
        # secondary debe haber registrado success
        assert cd_manager.get_state("secondary").error_count == 0


class TestLLMRouterCooldownSkip:
    @pytest.mark.asyncio
    async def test_skips_endpoint_in_cooldown(self, cd_manager, primary_ok, secondary_ok):
        # Marcar primary en cooldown manualmente
        cd_manager.record_failure("primary", ErrorKind.RATE_LIMIT)

        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        result = await router.complete("hola", max_tokens=10)

        assert result.endpoint_id == "secondary"
        # primary nunca se invocó (cooldown skip)
        assert primary_ok.client.calls == 0


class TestLLMRouterAllFailed:
    @pytest.mark.asyncio
    async def test_all_failed_raises_fallback_summary(self, cd_manager):
        a = LLMEndpoint(
            id="a", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("429")),
            priority=0,
        )
        b = LLMEndpoint(
            id="b", kind=EndpointKind.HTTP_REASONER,
            client=_FakeClient(raises=TimeoutError("read")),
            priority=1,
        )
        router = LLMRouter([a, b], cd_manager)

        with pytest.raises(FallbackSummaryError) as exc_info:
            await router.complete("hola", max_tokens=10)

        err = exc_info.value
        assert len(err.attempts) == 2
        assert err.attempts[0].endpoint_id == "a"
        assert err.attempts[0].error_kind == ErrorKind.RATE_LIMIT
        assert err.attempts[1].endpoint_id == "b"
        assert err.attempts[1].error_kind == ErrorKind.TIMEOUT

    @pytest.mark.asyncio
    async def test_auth_error_does_not_failover(self, cd_manager, secondary_ok):
        """ErrorKind.AUTH no es failover-worthy → debe propagar la exception original."""
        primary = LLMEndpoint(
            id="primary", kind=EndpointKind.FAST_ROUTER,
            client=_FakeClient(raises=RuntimeError("401 Unauthorized")),
            priority=0,
        )
        router = LLMRouter([primary, secondary_ok], cd_manager)

        with pytest.raises(RuntimeError, match="401"):
            await router.complete("hola", max_tokens=10)

        # secondary nunca se invocó porque auth no triggered failover
        assert secondary_ok.client.calls == 0


class TestLLMRouterAllInCooldown:
    @pytest.mark.asyncio
    async def test_all_in_cooldown_raises_with_next_attempt(self, cd_manager, primary_ok, secondary_ok):
        cd_manager.record_failure("primary", ErrorKind.RATE_LIMIT)
        cd_manager.record_failure("secondary", ErrorKind.RATE_LIMIT)

        router = LLMRouter([primary_ok, secondary_ok], cd_manager)
        with pytest.raises(FallbackSummaryError) as exc_info:
            await router.complete("hola", max_tokens=10)

        err = exc_info.value
        # No se llamó a ningún cliente
        assert primary_ok.client.calls == 0
        assert secondary_ok.client.calls == 0
        # Pero el FallbackSummaryError reporta que todos están en cooldown
        assert err.soonest_retry_at is not None
        assert err.soonest_retry_at > 0
