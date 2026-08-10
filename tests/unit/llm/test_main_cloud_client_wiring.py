"""B3: verify HttpReasoner can be constructed with the MiniMax cloud config."""

from src.llm.hermes_reasoner import HermesCliReasoner
from src.llm.reasoner import HttpReasoner


def test_httpreasoner_cloud_construction_from_config():
    reasoner_cfg = {
        "mode": "http",
        "http_base_url": "https://api.minimax.io/v1",
        "http_model": "MiniMax-M2.7-highspeed",
        "http_timeout": 60,
        "idle_timeout_s": 25.0,
        "api_style": "chat",
        "api_key_env": "MINIMAX_API_KEY",
    }
    r = HttpReasoner(
        base_url=reasoner_cfg["http_base_url"],
        model=reasoner_cfg["http_model"],
        timeout=reasoner_cfg["http_timeout"],
        idle_timeout_s=reasoner_cfg["idle_timeout_s"],
        api_style=reasoner_cfg.get("api_style", "completions"),
        api_key_env=reasoner_cfg.get("api_key_env"),
    )
    assert r.base_url == "https://api.minimax.io/v1"
    assert r.model == "MiniMax-M2.7-highspeed"
    assert r.api_style == "chat"
    assert r.api_key_env == "MINIMAX_API_KEY"
    assert r.idle_timeout_s == 25.0


def test_hermes_cli_reasoner_construction_from_config():
    """Mismo patrón que test_httpreasoner_cloud_construction_from_config, para el modo nuevo."""
    reasoner_cfg = {
        "mode": "hermes_cli",
        "hermes_binary_path": "/opt/hermes/bin/hermes",
        "hermes_provider": "openai-codex",
        "hermes_model": None,
        "hermes_timeout_s": 90,
    }
    r = HermesCliReasoner(
        binary_path=reasoner_cfg.get("hermes_binary_path", "hermes"),
        provider=reasoner_cfg.get("hermes_provider", "openai-codex"),
        model=reasoner_cfg.get("hermes_model"),
        timeout_s=reasoner_cfg.get("hermes_timeout_s", 90.0),
    )
    assert r.binary_path == "/opt/hermes/bin/hermes"
    assert r.provider == "openai-codex"
    assert r.model is None
    assert r.timeout_s == 90


def test_hermes_cli_reasoner_construction_uses_defaults_when_keys_missing():
    reasoner_cfg = {"mode": "hermes_cli"}
    r = HermesCliReasoner(
        binary_path=reasoner_cfg.get("hermes_binary_path", "hermes"),
        provider=reasoner_cfg.get("hermes_provider", "openai-codex"),
        model=reasoner_cfg.get("hermes_model"),
        timeout_s=reasoner_cfg.get("hermes_timeout_s", 90.0),
    )
    assert r.binary_path == "hermes"
    assert r.timeout_s == 90.0
