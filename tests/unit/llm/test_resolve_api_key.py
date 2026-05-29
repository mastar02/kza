import pytest

from src.llm.reasoner import _resolve_api_key


def test_explicit_api_key_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret-123")
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    key = _resolve_api_key("https://api.minimax.io/v1", api_key_env="MINIMAX_API_KEY")
    assert key == "mm-secret-123"


def test_explicit_api_key_env_missing_raises(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        _resolve_api_key("https://api.minimax.io/v1", api_key_env="MINIMAX_API_KEY")


def test_legacy_port_heuristic_still_works(monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "llama-key")
    key = _resolve_api_key("http://127.0.0.1:8200/v1")  # sin api_key_env
    assert key == "llama-key"
