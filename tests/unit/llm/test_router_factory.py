"""Tests for router_factory.build_llm_router."""

import pytest

from src.llm.router_factory import build_llm_router
from src.llm.router import LLMRouter
from src.llm.types import EndpointKind


class FakeClient:
    async def complete(self, prompt, max_tokens=128, **kwargs):
        return "ok"


@pytest.fixture
def cooldown_path(tmp_path):
    return tmp_path / "cd.json"


def _basic_config(cooldown_path):
    return {
        "endpoints": [
            {"id": "fast", "kind": "fast_router", "priority": 1, "timeout_s": 5.0},
            {"id": "slow", "kind": "http_reasoner", "priority": 2, "idle_timeout_s": 8.0},
        ],
        "cooldowns": {"persist_path": str(cooldown_path)},
    }


class TestBuildLLMRouter:
    def test_builds_router_with_endpoints_in_priority_order(self, cooldown_path):
        clients = {"fast": FakeClient(), "slow": FakeClient()}
        router = build_llm_router(_basic_config(cooldown_path), clients)
        assert isinstance(router, LLMRouter)
        assert [e.id for e in router._endpoints] == ["fast", "slow"]
        assert router._endpoints[0].kind == EndpointKind.FAST_ROUTER
        assert router._endpoints[1].idle_timeout_s == 8.0

    def test_raises_when_endpoints_empty(self, cooldown_path):
        with pytest.raises(ValueError, match="endpoints"):
            build_llm_router({"endpoints": [], "cooldowns": {}}, {})

    def test_raises_when_client_missing(self, cooldown_path):
        cfg = _basic_config(cooldown_path)
        with pytest.raises(ValueError, match="cliente"):
            build_llm_router(cfg, {"fast": FakeClient()})  # falta "slow"

    def test_raises_on_invalid_kind(self, cooldown_path):
        cfg = _basic_config(cooldown_path)
        cfg["endpoints"][0]["kind"] = "no_existe"
        with pytest.raises(ValueError):
            build_llm_router(cfg, {"fast": FakeClient(), "slow": FakeClient()})

    def test_default_cooldown_path_creates_parent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = {
            "endpoints": [
                {"id": "fast", "kind": "fast_router", "priority": 1},
            ],
            # cooldowns ausente → usa default ./data/llm_cooldowns.json
        }
        router = build_llm_router(cfg, {"fast": FakeClient()})
        assert (tmp_path / "data").is_dir()
        assert isinstance(router, LLMRouter)
