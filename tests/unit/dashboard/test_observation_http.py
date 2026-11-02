"""Metadata cache ordering must not replace a newer real observation."""

from datetime import UTC, datetime, timedelta

from starlette.responses import Response

from src.core.observation import Observation, ObservationSource
from src.dashboard.observation_http import ObservationHeaders


def test_old_response_keeps_its_dates_without_regressing_the_next_failure():
    now = datetime(2026, 11, 2, tzinfo=UTC)
    metadata = ObservationHeaders(clock=lambda: now+timedelta(seconds=5))
    newer = Observation.from_read(ObservationSource.ZONE_MANAGER, available=True,
        observed_at=now+timedelta(seconds=2), ttl_s=6, attempted_at=now+timedelta(seconds=1))
    older = Observation.from_read(ObservationSource.ZONE_MANAGER, available=True,
        observed_at=now, ttl_s=6, attempted_at=now)
    first, delayed, failed = Response(), Response(), Response()
    metadata.apply(first, 'zones', ObservationSource.ZONE_MANAGER, observation=newer)
    metadata.apply(delayed, 'zones', ObservationSource.ZONE_MANAGER, observation=older)
    metadata.apply(failed, 'zones', ObservationSource.ZONE_MANAGER)
    assert delayed.headers['X-KZA-Observed-At'] == older.observed_at.isoformat()
    assert failed.headers['X-KZA-Observed-At'] == first.headers['X-KZA-Observed-At']
    assert failed.headers['X-KZA-Expires-At'] == first.headers['X-KZA-Expires-At']
    assert failed.headers['X-KZA-Observation-State'] == 'unavailable'


def test_demo_does_not_claim_dates_or_create_a_real_observation():
    response = Response()
    metadata = ObservationHeaders()
    metadata.apply(response, 'gpus', ObservationSource.GPU_PROBE, demo=True)
    assert response.headers['X-KZA-Source'] == 'mock'
    assert response.headers['X-KZA-Source-Id'] == 'demo'
    assert response.headers['X-KZA-Observation-State'] == 'demo'
    assert 'X-KZA-Observed-At' not in response.headers and 'X-KZA-Expires-At' not in response.headers
    failed = Response()
    metadata.apply(failed, 'gpus', ObservationSource.GPU_PROBE)
    assert failed.headers['X-KZA-Source'] == 'degraded' and 'X-KZA-Observed-At' not in failed.headers
