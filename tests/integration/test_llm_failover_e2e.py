"""End-to-end test: LLMRouter rota entre endpoints con cooldown."""

import time

import pytest

from src.llm import (
    CooldownManager,
    EndpointKind,
    FallbackSummaryError,
    LLMEndpoint,
    LLMRouter,
)


class FlakyClient:
    """Cliente que falla N veces y después responde OK."""

    def __init__(self, fail_n: int, error: Exception, response: str = "ok"):
        self.fail_n = fail_n
        self.calls = 0
        self.error = error
        self.response = response

    async def complete(self, prompt, max_tokens=128, **_):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.error
        return self.response


class AlwaysOK:
    async def complete(self, prompt, max_tokens=128, **_):
        return "fallback-ok"


@pytest.fixture
def cd_path(tmp_path):
    return tmp_path / "cooldowns.json"


@pytest.mark.asyncio
class TestLLMFailoverE2E:
    async def test_primary_fails_secondary_succeeds(self, cd_path):
        """7B mock falla con timeout → router cae al 72B mock que responde OK."""
        primary = FlakyClient(fail_n=1, error=TimeoutError("vLLM timeout"))
        secondary = AlwaysOK()
        cd = CooldownManager(persistence_path=cd_path)
        router = LLMRouter(
            endpoints=[
                LLMEndpoint(id="fast", kind=EndpointKind.FAST_ROUTER,
                            client=primary, priority=1),
                LLMEndpoint(id="slow", kind=EndpointKind.HTTP_REASONER,
                            client=secondary, priority=2),
            ],
            cooldown_manager=cd,
        )
        result = await router.complete("¿qué hora es?", max_tokens=64)
        assert result.text == "fallback-ok"
        assert result.endpoint_id == "slow"
        assert result.attempts == 2
        # Primary entró en cooldown
        assert not cd.is_available("fast")
        # Secondary success limpió su contador
        assert cd.is_available("slow")

    async def test_primary_recovers_after_cooldown(self, cd_path, monkeypatch):
        """Primary entra en cooldown, expira, y el siguiente request lo usa de nuevo."""
        primary = FlakyClient(fail_n=1, error=TimeoutError("transient"))
        secondary = AlwaysOK()
        cd = CooldownManager(persistence_path=cd_path)
        router = LLMRouter(
            endpoints=[
                LLMEndpoint(id="fast", kind=EndpointKind.FAST_ROUTER,
                            client=primary, priority=1),
                LLMEndpoint(id="slow", kind=EndpointKind.HTTP_REASONER,
                            client=secondary, priority=2),
            ],
            cooldown_manager=cd,
        )
        # Primer request: primary falla, cae a secondary
        await router.complete("foo")
        # Forzar expiración del cooldown moviendo el reloj
        state = cd.get_state("fast")
        original_time = time.time
        future = original_time() + state.cooldown_until + 1
        monkeypatch.setattr(time, "time", lambda: future)
        # Segundo request: primary ya recovery (fail_n=1 agotado, ahora responde OK)
        result = await router.complete("bar")
        assert result.endpoint_id == "fast"
        assert result.text == "ok"

    async def test_all_endpoints_fail_raises_fallback_summary(self, cd_path):
        primary = FlakyClient(fail_n=99, error=TimeoutError("primary"))
        secondary = FlakyClient(fail_n=99, error=TimeoutError("secondary"))
        cd = CooldownManager(persistence_path=cd_path)
        router = LLMRouter(
            endpoints=[
                LLMEndpoint(id="fast", kind=EndpointKind.FAST_ROUTER,
                            client=primary, priority=1),
                LLMEndpoint(id="slow", kind=EndpointKind.HTTP_REASONER,
                            client=secondary, priority=2),
            ],
            cooldown_manager=cd,
        )
        with pytest.raises(FallbackSummaryError) as exc_info:
            await router.complete("foo")
        assert len(exc_info.value.attempts) == 2
        assert exc_info.value.attempts[0].endpoint_id == "fast"
        assert exc_info.value.attempts[1].endpoint_id == "slow"

    async def test_idle_timeout_is_classified_as_failover_worthy(self, cd_path):
        """IdleTimeoutError del watchdog hace que el router rote, no propague."""
        from src.llm import IdleTimeoutError

        class IdleClient:
            async def complete(self, prompt, max_tokens=128, **_):
                raise IdleTimeoutError(idle_timeout_s=8.0, chunks_received=3)

        primary = IdleClient()
        secondary = AlwaysOK()
        cd = CooldownManager(persistence_path=cd_path)
        router = LLMRouter(
            endpoints=[
                LLMEndpoint(id="fast", kind=EndpointKind.FAST_ROUTER,
                            client=primary, priority=1),
                LLMEndpoint(id="slow", kind=EndpointKind.HTTP_REASONER,
                            client=secondary, priority=2),
            ],
            cooldown_manager=cd,
        )
        result = await router.complete("foo")
        assert result.endpoint_id == "slow"
