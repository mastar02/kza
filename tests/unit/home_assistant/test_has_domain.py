"""Tests para HomeAssistantClient.has_domain (Fix 1B).

Verifica que el helper devuelve True/False correctamente para distintos
estados del state_cache, lo que permite al request_router rechazar intents
que requieren un dominio inexistente (ej: set_temperature sin climate).
"""
from __future__ import annotations

import pytest

from src.home_assistant.ha_client import HomeAssistantClient


@pytest.fixture
def ha():
    return HomeAssistantClient(url="http://mock.local", token="fake-token")


class TestHasDomain:
    def test_empty_cache_returns_false(self, ha):
        assert ha.has_domain("light") is False
        assert ha.has_domain("climate") is False

    def test_existing_domain_returns_true(self, ha):
        ha._state_cache["light.escritorio"] = {"state": "on"}
        ha._state_cache["light.living"] = {"state": "off"}
        assert ha.has_domain("light") is True

    def test_missing_domain_returns_false(self, ha):
        ha._state_cache["light.escritorio"] = {"state": "on"}
        ha._state_cache["light.living"] = {"state": "off"}
        # No hay climate.* — aunque hay luces.
        assert ha.has_domain("climate") is False
        assert ha.has_domain("media_player") is False
        assert ha.has_domain("cover") is False

    def test_partial_match_not_counted(self, ha):
        """'lightning.x' NO califica como domain 'light'."""
        ha._state_cache["lightning.bolt"] = {"state": "on"}
        assert ha.has_domain("light") is False

    def test_exact_dot_required(self, ha):
        """Verifica que el match es por prefijo `domain.` exacto."""
        ha._state_cache["climate_zone.x"] = {"state": "on"}
        assert ha.has_domain("climate") is False

    def test_multiple_domains(self, ha):
        ha._state_cache["light.x"] = {}
        ha._state_cache["climate.y"] = {}
        ha._state_cache["media_player.z"] = {}
        assert ha.has_domain("light") is True
        assert ha.has_domain("climate") is True
        assert ha.has_domain("media_player") is True
        assert ha.has_domain("cover") is False
