"""Tests for the cloud consent gate (privacy)."""

from src.llm.cloud_consent import cloud_reasoner_allowed, is_cloud_endpoint


def test_cloud_blocked_without_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is False


def test_cloud_allowed_with_consent():
    cfg = {"http_base_url": "https://api.minimax.io/v1", "cloud": {"consent": True}}
    assert cloud_reasoner_allowed(cfg) is True


def test_localhost_always_allowed():
    cfg = {"http_base_url": "http://127.0.0.1:8200/v1", "cloud": {"consent": False}}
    assert cloud_reasoner_allowed(cfg) is True


def test_is_cloud_endpoint():
    assert is_cloud_endpoint("https://api.minimax.io/v1") is True
    assert is_cloud_endpoint("http://127.0.0.1:8200/v1") is False
    assert is_cloud_endpoint("http://localhost:8101/v1") is False
