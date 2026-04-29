"""Tests for permissions policy."""

import pytest

from src.hooks import HaActionCall, BlockResult


def _ha_call(user_id, domain):
    return HaActionCall(
        entity_id=f"{domain}.x", domain=domain,
        service="turn_on", service_data={},
        user_id=user_id, user_name=None, zone_id=None, timestamp=0.0,
    )


@pytest.mark.parametrize("user", ["niño1", "niño2"])
@pytest.mark.parametrize("domain", ["climate", "lock", "alarm_control_panel"])
def test_blocks_child_in_adult_domain(user, domain):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, domain))
    assert isinstance(result, BlockResult)


@pytest.mark.parametrize("user", ["niño1", "niño2"])
@pytest.mark.parametrize("domain", ["light", "switch", "media_player"])
def test_allows_child_in_other_domains(user, domain):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, domain))
    assert result is None


@pytest.mark.parametrize("user", ["adulto1", "adulto2", None])
def test_allows_adult_or_unknown_in_adult_domains(user):
    from src.policies.permissions import chicos_sin_dominios_adultos

    result = chicos_sin_dominios_adultos(_ha_call(user, "climate"))
    assert result is None
