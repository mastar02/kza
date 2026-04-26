"""Tests for LLM router type primitives."""

import pytest
from src.llm.types import (
    EndpointKind,
    ErrorKind,
    LLMEndpoint,
    CooldownState,
    RouterResult,
)


class TestEndpointKind:
    def test_values(self):
        assert EndpointKind.FAST_ROUTER.value == "fast_router"
        assert EndpointKind.HTTP_REASONER.value == "http_reasoner"
        assert EndpointKind.LOCAL_REASONER.value == "local_reasoner"
        assert EndpointKind.CLOUD.value == "cloud"


class TestErrorKind:
    def test_values(self):
        assert ErrorKind.RATE_LIMIT.value == "rate_limit"
        assert ErrorKind.TIMEOUT.value == "timeout"
        assert ErrorKind.IDLE_TIMEOUT.value == "idle_timeout"
        assert ErrorKind.BILLING.value == "billing"
        assert ErrorKind.AUTH.value == "auth"
        assert ErrorKind.FORMAT.value == "format"
        assert ErrorKind.PERMANENT.value == "permanent"

    def test_is_failover_worthy(self):
        assert ErrorKind.RATE_LIMIT.is_failover_worthy() is True
        assert ErrorKind.TIMEOUT.is_failover_worthy() is True
        assert ErrorKind.IDLE_TIMEOUT.is_failover_worthy() is True
        assert ErrorKind.BILLING.is_failover_worthy() is True
        assert ErrorKind.FORMAT.is_failover_worthy() is True
        # Auth/permanent suben a la app — no rotación automática
        assert ErrorKind.AUTH.is_failover_worthy() is False
        assert ErrorKind.PERMANENT.is_failover_worthy() is False


class TestLLMEndpoint:
    def test_required_fields(self):
        ep = LLMEndpoint(
            id="primary",
            kind=EndpointKind.FAST_ROUTER,
            client=object(),
            priority=0,
        )
        assert ep.id == "primary"
        assert ep.kind == EndpointKind.FAST_ROUTER
        assert ep.priority == 0
        # defaults
        assert ep.timeout_s == 30.0
        assert ep.idle_timeout_s is None
        assert ep.max_tokens_default == 256

    def test_ordering_by_priority(self):
        a = LLMEndpoint(id="a", kind=EndpointKind.FAST_ROUTER, client=None, priority=0)
        b = LLMEndpoint(id="b", kind=EndpointKind.HTTP_REASONER, client=None, priority=1)
        # priority lower = tried first
        assert a.priority < b.priority


class TestCooldownState:
    def test_defaults(self):
        s = CooldownState(endpoint_id="primary")
        assert s.endpoint_id == "primary"
        assert s.error_count == 0
        assert s.cooldown_until == 0.0
        assert s.last_used == 0.0
        assert s.last_error_kind is None

    def test_to_dict_roundtrip(self):
        s = CooldownState(
            endpoint_id="primary",
            error_count=2,
            cooldown_until=1234567890.5,
            last_used=1234567880.0,
            last_error_kind=ErrorKind.RATE_LIMIT,
        )
        d = s.to_dict()
        s2 = CooldownState.from_dict(d)
        assert s2 == s


class TestRouterResult:
    def test_success(self):
        r = RouterResult(
            text="hola",
            endpoint_id="primary",
            attempts=1,
            elapsed_ms=42.0,
        )
        assert r.text == "hola"
        assert r.endpoint_id == "primary"
        assert r.attempts == 1
        assert r.elapsed_ms == 42.0

    def test_metadata_default_is_independent_dict(self):
        r1 = RouterResult(text="a", endpoint_id="x", attempts=1, elapsed_ms=1.0)
        r2 = RouterResult(text="b", endpoint_id="y", attempts=1, elapsed_ms=2.0)
        # Cada instancia tiene su propio dict (no compartido)
        assert r1.metadata == {}
        assert r2.metadata == {}
        assert r1.metadata is not r2.metadata
        r1.metadata["key"] = "value"
        assert r2.metadata == {}  # Mutación en r1 no afecta r2

    def test_metadata_explicit(self):
        r = RouterResult(
            text="ok", endpoint_id="x", attempts=1, elapsed_ms=1.0,
            metadata={"foo": "bar"},
        )
        assert r.metadata == {"foo": "bar"}
