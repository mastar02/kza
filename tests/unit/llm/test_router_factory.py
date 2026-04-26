"""Tests for build_llm_router_from_config."""

import pytest
from unittest.mock import MagicMock

from src.llm.router import LLMRouter
from src.llm.router_factory import build_llm_router_from_config
from src.llm.types import EndpointKind


def _config(persistence_path):
    return {
        "llm": {
            "failover": {
                "cooldown_persistence_path": str(persistence_path),
                "endpoints": [
                    {
                        "id": "fast",
                        "kind": "fast_router",
                        "priority": 0,
                        "max_tokens_default": 128,
                    },
                    {
                        "id": "deep",
                        "kind": "http_reasoner",
                        "priority": 1,
                        "max_tokens_default": 512,
                    },
                ],
            }
        }
    }


class TestBuildRouterFromConfig:
    def test_builds_router_with_endpoints_in_priority_order(self, tmp_path):
        fast = MagicMock(name="FastRouter")
        deep = MagicMock(name="HttpReasoner")
        clients = {"fast": fast, "deep": deep}

        router = build_llm_router_from_config(
            _config(tmp_path / "cd.json"), clients=clients
        )

        assert isinstance(router, LLMRouter)
        # primer endpoint = fast (priority 0)
        assert router._endpoints[0].id == "fast"
        assert router._endpoints[0].kind == EndpointKind.FAST_ROUTER
        assert router._endpoints[1].id == "deep"
        assert router._endpoints[1].kind == EndpointKind.HTTP_REASONER

    def test_skips_endpoint_when_client_missing(self, tmp_path):
        # solo "fast" tiene cliente; "deep" no
        clients = {"fast": MagicMock()}

        router = build_llm_router_from_config(
            _config(tmp_path / "cd.json"), clients=clients
        )

        assert len(router._endpoints) == 1
        assert router._endpoints[0].id == "fast"

    def test_no_endpoints_raises(self, tmp_path):
        cfg = _config(tmp_path / "cd.json")
        cfg["llm"]["failover"]["endpoints"] = []
        with pytest.raises(ValueError, match="at least one"):
            build_llm_router_from_config(cfg, clients={})

    def test_unknown_kind_raises(self, tmp_path):
        cfg = _config(tmp_path / "cd.json")
        cfg["llm"]["failover"]["endpoints"][0]["kind"] = "xyz"
        with pytest.raises(ValueError, match="unknown.*kind"):
            build_llm_router_from_config(cfg, clients={"fast": MagicMock(), "deep": MagicMock()})
