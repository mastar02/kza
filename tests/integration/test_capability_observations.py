"""Shared capability observations preserve age and real ownership across callers."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.core.observation import ObservationState
from src.core.task_outcome import CURRENT_REQUEST_ID
from src.home_assistant.capability_service import HomeCapabilityService
from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
from src.home_assistant.home_actions import HomeActionService
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.dashboard.home_catalog import HomeCatalogService
from src.nlu.home_command import HomeCommand
from tests.mocks.capability_source import CapabilitySource

pytestmark = pytest.mark.asyncio


def source():
    times = [datetime(2026, 11, 2, tzinfo=UTC)]
    ha = CapabilitySource(lambda: times[0])
    policy = HomeControlPolicy((EntityGrant('light.fixture', frozenset({'turn_on', 'turn_off', 'brightness'})),))
    return ha, HomeCapabilityService(ha, policy, clock=lambda: times[0]), times


async def test_cancelled_callers_and_close_observers_do_not_lose_the_shared_read():
    ha, service, _ = source()
    ha.release.clear()
    token = CURRENT_REQUEST_ID.set('synthetic-private-caller')
    try:
        waiters = [asyncio.create_task(service.snapshot()) for _ in range(2)]
    finally:
        CURRENT_REQUEST_ID.reset(token)
    await asyncio.wait_for(ha.entered.wait(), 1)
    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    assert ha.state_reads == 1 and ha.area_reads == 1
    assert not service._refresh_task.done()
    for _ in range(2):
        closer = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
    assert service._closed and not service._close_task.done()
    with pytest.raises(RuntimeError, match='capabilities_closed'):
        await service.snapshot()
    ha.release.set()
    await service.aclose()
    assert service._refresh_task.done() and service._close_task.done()
    assert ha.state_reads == 1 and ha.area_reads == 1


async def test_failure_retains_original_dates_and_does_not_claim_current_capability():
    ha, service, times = source()
    try:
        first = await service.snapshot()
        assert first.entity('light.fixture').available(times[0])
        ha.failed = True
        times[0] += timedelta(seconds=2)
        failed = await service.snapshot()
        assert failed.observation.state is ObservationState.UNAVAILABLE
        assert failed.observation.observed_at == first.observation.observed_at
        assert failed.observation.expires_at == first.observation.expires_at
        assert failed.observation.attempted_at > first.observation.attempted_at
        assert not failed.entity('light.fixture').available(times[0])
        assert failed.domain_present('light', times[0]) is None
        assert first.observation.as_of(times[0]+timedelta(seconds=5)).state is ObservationState.STALE
    finally:
        await service.aclose()


async def test_real_empty_is_distinct_from_unknown_time_or_unavailable_source():
    ha, service, times = source()
    try:
        ha.states = []
        empty = await service.snapshot()
        assert empty.observation.current(times[0]) and empty.entities == ()
        assert empty.domain_present('light', times[0]) is False
        ha.known_time = False
        unknown = await service.snapshot()
        assert unknown.observation.state is ObservationState.UNKNOWN
        assert unknown.domain_present('light', times[0]) is None
    finally:
        await service.aclose()


async def test_unexpected_child_failure_waits_for_registry_cleanup_before_publishing():
    ha, service, _ = source()
    registry_entered, cleanup_entered, cleanup_release, cleanup_done = (asyncio.Event() for _ in range(4))

    async def state():
        await registry_entered.wait()
        raise RuntimeError('synthetic read failure')

    async def registry():
        registry_entered.set()
        try:
            await asyncio.Future()
        finally:
            cleanup_entered.set()
            await cleanup_release.wait()
            cleanup_done.set()

    ha.get_state_snapshot, ha.get_area_assignments = state, registry
    waiter = asyncio.create_task(service.snapshot())
    await asyncio.wait_for(cleanup_entered.wait(), 1)
    assert not waiter.done()
    closer = asyncio.create_task(service.aclose())
    cleanup_release.set()
    result = await asyncio.wait_for(waiter, 1)
    await asyncio.wait_for(closer, 1)
    assert cleanup_done.is_set() and result.observation.state is ObservationState.UNAVAILABLE


@pytest.mark.parametrize('dependency', ['selection', 'temperature'])
async def test_casa_and_voice_retire_a_capability_when_only_its_dependency_expires(dependency):
    ha, service, times = source()
    ha.ttl_s = 60
    domain, entity_id, operation = 'light', 'light.fixture', 'brightness'
    if dependency == 'temperature':
        domain, entity_id, operation = 'climate', 'climate.fixture', 'set_temperature'
        ha.states = [{'entity_id': entity_id, 'state': 'cool', 'attributes': {
            'supported_features': 1, 'hvac_modes': ['cool'], 'min_temp': 7, 'max_temp': 35, 'target_temp_step': 1}}]
        ha.get_temperature_unit_fresh = AsyncMock(return_value='°C')
        service._policy = HomeControlPolicy((EntityGrant(entity_id, frozenset({operation}), temperature_min=7, temperature_max=35),))
        original = service._selection

        async def selection():
            grants, observation = await original()
            return grants, replace(observation, expires_at=times[0]+timedelta(seconds=60))

        service._selection = selection
    snapshot = await service.snapshot()
    assert snapshot.entity(entity_id).available(times[0])
    service.snapshot = AsyncMock(return_value=snapshot)
    actions = HomeActionService(ha, policy=service._policy)
    resolver = HomeVoiceResolver(actions, capabilities=service)
    catalog = HomeCatalogService(ha, {}, policy=service._policy, capabilities=service)
    times[0] += timedelta(seconds=7)
    try:
        view = await catalog.snapshot()
        entity = next(value for value in view.entities if value.entity_id == entity_id)
        assert snapshot.observation.current(times[0])
        assert entity.capabilities == () and not entity.available
        assert entity.observation.state is ObservationState.STALE
        target = await resolver.resolve(HomeCommand(domain, '', operation), room_id=None)
        assert target.entity_id is None and target.reason == 'capability_unavailable'
        assert target.observation.as_of(times[0]).state is ObservationState.STALE
    finally:
        await service.aclose()


async def test_policy_read_crossing_expiry_cannot_return_a_usable_voice_target():
    ha, service, times = source()
    ha.ttl_s = 60
    snapshot = await service.snapshot()
    service.snapshot = AsyncMock(return_value=snapshot)
    policy = service._policy
    original = policy.snapshot_grants
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def grants():
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await release.wait()
        return await original()

    policy.snapshot_grants = grants
    resolver = HomeVoiceResolver(HomeActionService(ha, policy=policy), capabilities=service)
    waiter = asyncio.create_task(resolver.resolve(HomeCommand('light', '', 'brightness'), room_id=None))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        times[0] += timedelta(seconds=7)
        release.set()
        target = await asyncio.wait_for(waiter, 1)
        assert target.entity_id is None and target.reason == 'capability_unavailable'
        assert target.observation.source.value == 'control-selection'
        assert target.observation.state is ObservationState.STALE
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
        await service.aclose()
