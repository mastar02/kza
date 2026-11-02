"""Observation age must survive failures, retries and presentation."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.observation import Observation, ObservationReason, ObservationSource, ObservationState, combine_observations

NOW = datetime(2026, 11, 2, tzinfo=UTC)


def test_failure_and_expiry_never_renew_the_original_observation():
    original = Observation.from_read(ObservationSource.HA_STATES, available=True,
        observed_at=NOW, ttl_s=6, attempted_at=NOW)
    failed = original.failed(attempted_at=NOW + timedelta(seconds=2))
    assert failed.observed_at == original.observed_at and failed.expires_at == original.expires_at
    assert not failed.current(NOW + timedelta(seconds=3))
    assert original.current(NOW + timedelta(seconds=5))
    expired = original.as_of(NOW + timedelta(seconds=6))
    assert expired.state is ObservationState.STALE and expired.reason is ObservationReason.EXPIRED
    assert expired.observed_at == NOW and expired.expires_at == NOW + timedelta(seconds=6)


def test_success_without_a_known_read_time_is_unknown():
    value = Observation.from_read(ObservationSource.HA_STATES, available=True, observed_at=None, ttl_s=6)
    assert value.state is ObservationState.UNKNOWN and not value.current(NOW)
    assert value.to_dict(NOW)['observed_at'] is None and value.to_dict(NOW)['expires_at'] is None


@pytest.mark.parametrize('kwargs', [
    {'source': 'private free text', 'state': ObservationState.UNAVAILABLE},
    {'source': ObservationSource.HA_STATES, 'state': ObservationState.FRESH},
    {'source': ObservationSource.DEMO, 'state': ObservationState.FRESH, 'observed_at': NOW, 'expires_at': NOW+timedelta(seconds=6)},
    {'source': ObservationSource.HA_STATES, 'state': ObservationState.UNAVAILABLE, 'reason': 'private exception'},
])
def test_unknown_codes_or_demo_cannot_be_certified_as_observed(kwargs):
    with pytest.raises(ValueError):
        Observation(**kwargs)


def test_partial_configuration_does_not_certify_unobserved_columns():
    value = Observation.from_read(ObservationSource.ROUTER_STATE, available=True, observed_at=NOW,
        ttl_s=6, unobserved_fields=('backend_health', 'failures_7d'))
    projected = value.to_dict(NOW)
    assert projected['state'] == 'partial'
    assert projected['unobserved_fields'] == ('backend_health', 'failures_7d')


def test_future_observation_is_not_current_and_keeps_original_dates():
    value = Observation.from_read(ObservationSource.HA_STATES, available=True,
        observed_at=NOW+timedelta(seconds=1), ttl_s=6)
    assert not value.current(NOW)
    assert value.as_of(NOW).reason is ObservationReason.CLOCK_SKEW
    assert value.as_of(NOW).observed_at == value.observed_at


@pytest.mark.parametrize('expiring', [ObservationSource.CONTROL_SELECTION, ObservationSource.HA_CONFIG])
def test_capability_lifetime_is_bounded_by_each_required_dependency(expiring):
    values = tuple(Observation.from_read(source, available=True, observed_at=NOW,
        ttl_s=1 if source is expiring else 60)
        for source in (ObservationSource.HA_STATES, ObservationSource.CONTROL_SELECTION, ObservationSource.HA_CONFIG))
    combined = combine_observations(ObservationSource.HOME_CAPABILITIES, values, now=NOW)
    assert combined.current(NOW) and not combined.current(NOW+timedelta(seconds=1))
    assert combined.expires_at == NOW+timedelta(seconds=1)
