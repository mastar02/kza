"""Tests del selector de backend Chroma (embedded | http) del code-index."""

from unittest.mock import patch

import pytest

from src.code_index.service import build_chroma_client


def test_default_embedded_uses_persistent_client_with_chroma_path():
    with patch("chromadb.PersistentClient") as pc:
        client = build_chroma_client({"chroma_path": "/tmp/x"})
    pc.assert_called_once_with(path="/tmp/x")
    assert client is pc.return_value


def test_explicit_embedded_mode_uses_persistent_client():
    cfg = {"chroma_path": "/tmp/x", "chroma": {"mode": "embedded"}}
    with patch("chromadb.PersistentClient") as pc:
        build_chroma_client(cfg)
    pc.assert_called_once_with(path="/tmp/x")


def test_http_mode_uses_http_client_with_host_and_port():
    cfg = {
        "chroma_path": "/unused",
        "chroma": {"mode": "http", "url": "http://kza-chroma:8000"},
    }
    with patch("chromadb.HttpClient") as hc:
        client = build_chroma_client(cfg)
    hc.assert_called_once_with(host="kza-chroma", port=8000)
    assert client is hc.return_value


def test_http_mode_with_url_without_port_raises():
    cfg = {"chroma": {"mode": "http", "url": "http://kza-chroma"}}
    with pytest.raises(ValueError, match="chroma.url"):
        build_chroma_client(cfg)


def test_http_mode_without_url_raises():
    cfg = {"chroma": {"mode": "http"}}
    with pytest.raises(ValueError, match="chroma.url"):
        build_chroma_client(cfg)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="chroma.mode"):
        build_chroma_client({"chroma": {"mode": "grpc"}})
