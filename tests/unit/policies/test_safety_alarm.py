"""Tests for safety_alarm policy."""

from unittest.mock import patch
import pytest

from src.hooks import HaActionCall, BlockResult


def _ha_call(entity_id="alarm_control_panel.casa", service="alarm_disarm"):
    return HaActionCall(
        entity_id=entity_id, domain="alarm_control_panel",
        service=service, service_data={},
        user_id=None, user_name=None, zone_id=None, timestamp=0.0,
    )


@pytest.mark.parametrize("hour", [22, 23, 0, 6])
def test_blocks_at_night_hours(hour):
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": hour})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call())

    assert isinstance(result, BlockResult)
    assert result.rule_name == "proteger_alarma_de_noche"


@pytest.mark.parametrize("hour", [7, 8, 12, 18, 21])
def test_allows_at_day_hours(hour):
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": hour})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call())

    assert result is None


def test_only_blocks_alarm_disarm_not_other_services():
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": 23})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call(service="alarm_arm_home"))

    assert result is None


def test_only_blocks_specific_alarm_entity():
    from src.policies.safety_alarm import proteger_alarma_de_noche

    fake_now = type("dt", (), {"hour": 23})
    with patch("src.policies.safety_alarm.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        result = proteger_alarma_de_noche(_ha_call(entity_id="alarm_control_panel.otra"))

    assert result is None
