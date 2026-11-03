"""HA transport observations stay separate from historical call counters."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.home_assistant.ha_client import HAChannelObservation, HAConnectionState, HAHealthStatus
from src.monitoring.health_aggregator import HealthAggregator, OverallStatus


class Clock:
    def __init__(self, now=100.):
        self.now = now

    def __call__(self):
        return self.now


def channel(name, *, state=HAConnectionState.CONNECTED, phase='ready',
            observed_at=90., expires_at=150., required=True, reason=None):
    return HAChannelObservation(state=state,
        source={'rest': 'ha_rest', 'calls': 'ha_ws_calls',
                'events': 'ha_ws_events', 'registry': 'ha_ws_registry'}[name],
        phase=phase, observed_at=observed_at, expires_at=expires_at,
        reason=reason, attempted_at=89., required=required, generation=7)


def health(*, rest=None, calls=None, events=None, registry=None):
    return HAHealthStatus(state=HAConnectionState.CONNECTED, last_success_ts=99.,
        success_count=42, error_count=1, avg_latency_ms=25., ws_connected=True,
        rest=rest, calls=calls, events=events, registry=registry)


def monitor(value, clock=None):
    return HealthAggregator(ha_client=SimpleNamespace(get_health_status=lambda: value),
                            health_clock=clock or Clock())


def test_current_required_channels_preserve_original_observation_fields():
    value = health(rest=channel('rest'), calls=channel('calls'), events=channel('events'))
    result = monitor(value)._check_ha()
    assert result.status is OverallStatus.HEALTHY
    assert set(result.extra['channels']) == {'rest', 'calls', 'events'}
    for name, observation in result.extra['channels'].items():
        assert observation == {
            'state': 'connected', 'source': {'rest': 'ha_rest', 'calls': 'ha_ws_calls',
                'events': 'ha_ws_events'}[name], 'phase': 'ready',
            'observed_at': 90., 'expires_at': 150., 'reason': None,
            'attempted_at': 89., 'required': True, 'generation': 7, 'current': True,
        }


def test_rest_up_requested_events_down_is_degraded_without_relabeling_rest_or_calls():
    value = health(rest=channel('rest'), calls=channel('calls'),
        events=channel('events', state=HAConnectionState.DISCONNECTED,
                       phase='retrying', reason='ha_ws_events_closed'))
    result = monitor(value)._check_ha()
    assert result.status is OverallStatus.DEGRADED
    assert 'events:disconnected (retrying)' in result.detail
    assert result.extra['channels']['rest']['state'] == 'connected'
    assert result.extra['channels']['calls']['state'] == 'connected'
    assert result.extra['channels']['events']['reason'] == 'ha_ws_events_closed'
    assert result.extra['channels']['events']['observed_at'] == 90.
    assert result.extra['channels']['events']['expires_at'] == 150.
    assert result.extra['ws_connected'] is True


def test_unsolicited_unknown_channels_do_not_degrade_rest_only_health():
    events = channel('events', state=HAConnectionState.UNKNOWN, phase='stopped',
                     required=False, observed_at=None, expires_at=None)
    value = health(rest=channel('rest'), events=events,
        registry=channel('registry', state=HAConnectionState.UNKNOWN, phase='idle',
                         required=False, observed_at=None, expires_at=None))
    consumer = monitor(value)
    assert consumer._check_ha().status is OverallStatus.HEALTHY
    value.events = replace(events, required=True)
    result = consumer._check_ha()
    assert result.status is OverallStatus.DEGRADED
    assert result.extra['channels']['events']['observed_at'] is None
    assert result.extra['channels']['events']['expires_at'] is None
    assert result.extra['channels']['registry']['current'] is False


@pytest.mark.parametrize('now, expected, current', [
    (149.999, OverallStatus.HEALTHY, True),
    (150., OverallStatus.DEGRADED, False),
    (151., OverallStatus.DEGRADED, False),
])
def test_exact_expiry_is_not_renewed_by_historical_success_or_calls_auth(now, expected, current):
    rest = channel('rest')
    value = health(rest=rest,
        calls=channel('calls', observed_at=149., expires_at=209., required=False))
    value.last_success_ts = now
    value.success_count += 1
    result = monitor(value, Clock(now))._check_ha()
    assert result.status is expected
    assert result.extra['channels']['rest']['current'] is current
    assert result.extra['channels']['rest']['observed_at'] == rest.observed_at == 90.
    assert result.extra['channels']['rest']['expires_at'] == rest.expires_at == 150.
    if not current:
        assert result.extra['channels']['rest']['state'] == 'unknown'
        assert result.extra['channels']['rest']['phase'] == 'expired'


def test_future_observation_does_not_become_current_on_an_earlier_consumer_clock():
    value = health(rest=channel('rest', observed_at=101., expires_at=161.))
    result = monitor(value)._check_ha()
    assert result.status is OverallStatus.DEGRADED
    assert result.extra['channels']['rest']['state'] == 'unknown'
    assert result.extra['channels']['rest']['phase'] == 'future'
    assert result.extra['channels']['rest']['observed_at'] == 101.
    assert result.extra['channels']['rest']['expires_at'] == 161.


@pytest.mark.parametrize('now', [50., 200.])
def test_terminal_local_close_survives_remote_evidence_expiry_or_clock_rollback(now):
    value = health(rest=channel('rest', state=HAConnectionState.DISCONNECTED,
                                phase='closed'))
    result = monitor(value, Clock(now))._check_ha()
    assert result.status is OverallStatus.UNHEALTHY
    assert result.extra['channels']['rest']['state'] == 'disconnected'
    assert result.extra['channels']['rest']['phase'] == 'closed'
    assert result.extra['channels']['rest']['current'] is False
    assert result.extra['channels']['rest']['observed_at'] == 90.
    assert result.extra['channels']['rest']['expires_at'] == 150.


@pytest.mark.parametrize('state', [HAConnectionState.AUTH_ERROR, HAConnectionState.DISCONNECTED])
def test_current_rest_failure_is_unhealthy_even_with_connected_calls(state):
    value = health(rest=channel('rest', state=state, phase='failed'), calls=channel('calls'))
    result = monitor(value)._check_ha()
    assert result.status is OverallStatus.UNHEALTHY
    assert result.extra['channels']['calls']['state'] == 'connected'


def test_historical_global_error_does_not_override_current_separate_channels():
    value = health(rest=channel('rest'), calls=channel('calls'), events=channel('events'))
    value.state = HAConnectionState.AUTH_ERROR
    value.last_error_ts = 99.
    result = monitor(value)._check_ha()
    assert result.status is OverallStatus.HEALTHY
    assert all(item['current'] for item in result.extra['channels'].values())


@pytest.mark.parametrize('field, invalid', [
    ('observed_at', True), ('observed_at', -1.), ('expires_at', float('nan')),
    ('expires_at', 90.), ('expires_at', None), ('attempted_at', float('inf')),
    ('required', 1), ('generation', True), ('generation', -1), ('source', 'ha_ws_events'),
])
def test_malformed_native_channel_never_falls_back_to_global_connected(field, invalid):
    rest = channel('rest')
    object.__setattr__(rest, field, invalid)
    result = monitor(health(rest=rest))._check_ha()
    assert result.status is OverallStatus.UNHEALTHY
    assert result.detail == 'HA transport observation unavailable'
    assert result.extra == {}


def test_partial_native_report_with_missing_rest_is_not_global_healthy():
    result = monitor(health(calls=channel('calls')))._check_ha()
    assert result.status is OverallStatus.DEGRADED
    assert result.detail.startswith('rest:unknown; ')


def test_projection_is_detached_without_mutating_held_native_observation():
    rest = channel('rest')
    consumer = monitor(health(rest=rest))
    first = consumer._check_ha()
    first.extra['channels']['rest']['observed_at'] = 120.
    first.extra['channels']['rest']['required'] = False
    second = consumer._check_ha()
    assert second.extra['channels']['rest']['observed_at'] == 90.
    assert second.extra['channels']['rest']['required'] is True
    assert rest.observed_at == 90. and rest.required is True


@pytest.mark.parametrize('state, expected', [
    (HAConnectionState.CONNECTED, OverallStatus.HEALTHY),
    (HAConnectionState.DISCONNECTED, OverallStatus.UNHEALTHY),
    (HAConnectionState.AUTH_ERROR, OverallStatus.UNHEALTHY),
    (HAConnectionState.UNKNOWN, OverallStatus.DEGRADED),
])
def test_legacy_health_without_channel_dtos_keeps_existing_mapping(state, expected):
    assert monitor(HAHealthStatus(state=state))._check_ha().status is expected


def test_provider_failure_does_not_export_private_exception_message():
    def failed():
        raise RuntimeError('synthetic private transport detail')
    consumer = HealthAggregator(ha_client=SimpleNamespace(get_health_status=failed))
    result = consumer._check_ha()
    assert result.status is OverallStatus.UNHEALTHY
    assert result.detail == 'HA health unavailable: RuntimeError'
    assert 'synthetic private' not in repr(result)
