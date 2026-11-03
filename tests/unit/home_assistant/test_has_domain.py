"""Tests para HomeAssistantClient.has_domain (Fix 1B).

Verifica presencia, ausencia e inventario desconocido sin convertir una
semilla cruda o una observación vencida en autoridad actual.
"""
from __future__ import annotations

import pytest

from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant.state_projection import apply_initial


@pytest.fixture
def ha():
    return HomeAssistantClient(url="http://mock.local", token="fake-token")


def _observed_inventory(ha, states):
    ha._state_sync_running = True
    ha._entity_stream_generation += 1
    ha._state_frame = apply_initial(ha._state_frame, {"a": {
        entity_id: {"s": state, "a": {}, "c": "synthetic-context",
                    "lc": 1_700_000_000., "lu": 1_700_000_000.}
        for entity_id, state in states.items()
    }}, generation=ha._entity_stream_generation, observed_at=ha._entity_now())


class TestHasDomain:
    def test_unobserved_inventory_is_unknown(self, ha):
        assert ha.has_domain("light") is None
        assert ha.has_domain("climate") is None

    def test_observed_empty_inventory_returns_false(self, ha):
        _observed_inventory(ha, {})
        assert ha.has_domain("light") is False
        assert ha.has_domain("climate") is False

    def test_existing_domain_returns_true(self, ha):
        _observed_inventory(ha, {"light.escritorio": "on", "light.living": "off"})
        assert ha.has_domain("light") is True

    def test_missing_domain_returns_false(self, ha):
        _observed_inventory(ha, {"light.escritorio": "on", "light.living": "off"})
        assert ha.has_domain("climate") is False
        assert ha.has_domain("media_player") is False
        assert ha.has_domain("cover") is False

    def test_partial_match_not_counted(self, ha):
        _observed_inventory(ha, {"lightning.bolt": "on"})
        assert ha.has_domain("light") is False

    def test_exact_dot_required(self, ha):
        _observed_inventory(ha, {"climate_zone.x": "on"})
        assert ha.has_domain("climate") is False

    def test_multiple_domains(self, ha):
        _observed_inventory(ha, {"light.x": "off", "climate.y": "off", "media_player.z": "idle"})
        assert ha.has_domain("light") is True
        assert ha.has_domain("climate") is True
        assert ha.has_domain("media_player") is True
        assert ha.has_domain("cover") is False

    def test_expired_inventory_is_unknown(self, ha, monkeypatch):
        clock = {"now": 100.}
        monkeypatch.setattr(ha, "_entity_now", lambda: clock["now"])
        _observed_inventory(ha, {"light.x": "on"})
        clock["now"] = 106.
        assert ha.has_domain("light") is None
        assert ha.has_domain("climate") is None
