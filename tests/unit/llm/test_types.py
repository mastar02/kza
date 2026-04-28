"""Smoke tests for LLM router type primitives."""

import pytest

from src.llm.types import (
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    CooldownState,
    RouterResult,
)


class TestEndpointKind:
    def test_known_kinds_present(self):
        assert EndpointKind.FAST_ROUTER.value == "fast_router"
        assert EndpointKind.HTTP_REASONER.value == "http_reasoner"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            EndpointKind("nonexistent")


class TestErrorKind:
    def test_failover_worthy_classification(self):
        assert ErrorKind.RATE_LIMIT.is_failover_worthy()
        assert ErrorKind.TIMEOUT.is_failover_worthy()
        assert ErrorKind.IDLE_TIMEOUT.is_failover_worthy()
        assert ErrorKind.BILLING.is_failover_worthy()
        assert ErrorKind.FORMAT.is_failover_worthy()

    def test_auth_and_permanent_not_failover_worthy(self):
        assert not ErrorKind.AUTH.is_failover_worthy()
        assert not ErrorKind.PERMANENT.is_failover_worthy()


class TestLLMEndpoint:
    def test_instantiation_with_defaults(self):
        ep = LLMEndpoint(
            id="x",
            kind=EndpointKind.FAST_ROUTER,
            client=object(),
            priority=1,
        )
        assert ep.timeout_s == 30.0
        assert ep.idle_timeout_s is None
        assert ep.max_tokens_default == 256


class TestCooldownState:
    def test_to_dict_roundtrip(self):
        s = CooldownState(
            endpoint_id="abc",
            error_count=2,
            cooldown_until=1234567890.0,
            last_used=1234567000.0,
            last_error_kind=ErrorKind.TIMEOUT,
        )
        d = s.to_dict()
        s2 = CooldownState.from_dict(d)
        assert s2 == s

    def test_from_dict_with_no_error_kind(self):
        s = CooldownState.from_dict({
            "endpoint_id": "abc",
            "error_count": 0,
            "cooldown_until": 0.0,
            "last_used": 0.0,
            "last_error_kind": None,
        })
        assert s.last_error_kind is None


class TestRouterResult:
    def test_default_metadata_is_dict(self):
        r = RouterResult(
            text="hi",
            endpoint_id="x",
            attempts=1,
            elapsed_ms=12.3,
        )
        assert r.metadata == {}
        # Mutación independiente entre instancias
        r.metadata["k"] = "v"
        r2 = RouterResult(text="", endpoint_id="y", attempts=1, elapsed_ms=0.0)
        assert r2.metadata == {}
