"""Tests for the cloud consent gate (privacy)."""

from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY
from src.llm.cloud_consent import (
    cloud_reasoner_allowed,
    is_cloud_endpoint,
    resolve_http_reasoner_base_url,
)


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
    # IPv6 loopback: urlparse("http://[::1]:8200/v1").hostname == "::1"
    assert is_cloud_endpoint("http://[::1]:8200/v1") is False
    # Bind-all address also treated as local
    assert is_cloud_endpoint("http://0.0.0.0:8200/v1") is False


def test_unresolved_placeholder_is_classified_as_cloud():
    """Un "${VAR}" sin resolver tiene hostname vacío → clasificado cloud.

    Es justo lo que hace que el orden gate→fallback importe: si el gate
    viera el placeholder DESPUÉS de que el fallback local lo pisa, vería
    "127.0.0.1" (local) en vez de esto (cloud) y pasaría fail-open.
    """
    assert is_cloud_endpoint("${LLM_GATEWAY_URL}") is True


def test_gate_evaluated_before_fallback_is_applied():
    """Regresión del Critical cerrado 2026-08-02 (Task 5, "CI a verde").

    resolve_http_reasoner_base_url es la única fuente de verdad para el
    orden gate→fallback (ver su docstring en src/llm/cloud_consent.py).
    Si alguien invierte esas dos líneas — resuelve el placeholder ANTES
    de evaluar el gate — is_cloud_endpoint ve "127.0.0.1" en vez del
    placeholder y gate_allowed pasa a True aunque consent sea False. Este
    test tiene que quedar rojo si eso pasa (verificado moviendo el
    fallback antes del gate en una copia vía `git archive`, no en el
    repo real — ver informe de Task 5).
    """
    # consent=False + placeholder sin resolver → el reasoner NO se habilita,
    # aunque el fallback SÍ se aplique (para que el literal opaco no llegue
    # a ningún cliente HTTP — ver docstring de la función).
    cfg_blocked = {
        "http_base_url": "${LLM_GATEWAY_URL}",
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg_blocked, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is False
    assert resolved == DEFAULT_LOCAL_LLM_GATEWAY
    assert cfg_blocked["http_base_url"] == DEFAULT_LOCAL_LLM_GATEWAY

    # consent=True + placeholder sin resolver → SÍ se habilita, contra el
    # fallback local (loopback, nunca el literal opaco).
    cfg_allowed = {
        "http_base_url": "${LLM_GATEWAY_URL}",
        "cloud": {"consent": True},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg_allowed, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is True
    assert resolved == DEFAULT_LOCAL_LLM_GATEWAY


def test_gate_blocks_cloud_even_when_url_already_resolved():
    """Caso control: sin placeholder de por medio, consent=False bloquea."""
    cfg = {
        "http_base_url": "https://api.minimax.io/v1",
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_http_reasoner_base_url(
        cfg, DEFAULT_LOCAL_LLM_GATEWAY
    )
    assert gate_allowed is False
    # No había placeholder — el fallback no debe tocar la URL real.
    assert resolved == "https://api.minimax.io/v1"
