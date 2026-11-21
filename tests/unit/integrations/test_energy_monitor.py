"""Tests of the shared Home Assistant energy and power selection."""

from datetime import UTC, datetime
from decimal import Decimal
from time import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.home_assistant.control_policy import ActorKind, HomeActor
from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant.state_projection import REST_SOURCE, EntityObservation
from src.integrations.energy_monitor import (
    EnergyCostUnavailable,
    EnergyMonitor,
    EnergySensor,
    MeasurementKind,
    ReadingStatus,
)

ENERGY = MeasurementKind.ENERGY
POWER = MeasurementKind.POWER
HOUSEHOLD = HomeActor.household()
NOW = datetime(2026, 11, 21, 12, tzinfo=UTC).timestamp()


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def state(
    entity_id: str,
    value: str,
    device_class: str | None,
    unit: object,
    *,
    reported_at: float | None = NOW - 5,
    updated_at: float | None = NOW - 120,
) -> dict:
    payload = {
        "entity_id": entity_id,
        "state": value,
        "attributes": {
            "device_class": str(device_class) if device_class is not None else None,
            "unit_of_measurement": unit,
        },
    }
    if reported_at is not None:
        payload["last_reported"] = iso(reported_at)
    if updated_at is not None:
        payload["last_updated"] = iso(updated_at)
    return payload


class StateProvider:
    """Deliver recorded HA observations and expose every attempted read."""

    def __init__(self, *states: dict) -> None:
        self.observations = {
            payload["entity_id"]: EntityObservation(
                payload["entity_id"],
                payload,
                exists=True,
                source=REST_SOURCE,
                observed_at=NOW - 1,
                expires_at=NOW + 1,
                reason=None,
            )
            for payload in states
        }
        self.calls: list[str] = []

    async def read_entity_observation(self, entity_id: str) -> EntityObservation:
        self.calls.append(entity_id)
        return self.observations[entity_id]


def monitor(provider: StateProvider, *sensors: EnergySensor) -> EnergyMonitor:
    return EnergyMonitor(provider, sensors, max_staleness_s=30, clock=lambda: NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "unit", "value", "expected_unit", "expected_value"),
    [
        (POWER, "W", "-125.25", "W", "-125.25"),
        (POWER, "kW", "0.12345", "W", "123.45"),
        (ENERGY, "Wh", "123.45", "kWh", "0.12345"),
        (ENERGY, "kWh", "1.125", "kWh", "1.125"),
        (ENERGY, "MWh", "1.00001", "kWh", "1000.01"),
        (
            ENERGY,
            "MWh",
            "1.1234567890123456789012345678901",
            "kWh",
            "1123.4567890123456789012345678901",
        ),
    ],
)
async def test_point_read_converts_units_and_preserves_provenance(
    kind,
    unit,
    value,
    expected_unit,
    expected_value,
):
    sensor = EnergySensor("sensor.meter", kind, "cocina", "Medidor")
    provider = StateProvider(state(sensor.entity_id, value, kind, unit))

    reading = await monitor(provider, sensor).read_sensor(
        sensor.entity_id, actor=HOUSEHOLD
    )

    assert provider.calls == [sensor.entity_id]
    assert reading.sensor == sensor
    assert reading.status is ReadingStatus.AVAILABLE
    assert reading.value == Decimal(expected_value)
    assert reading.unit == expected_unit
    assert reading.original_unit == unit
    assert reading.source == REST_SOURCE
    assert reading.observed_at == datetime.fromtimestamp(NOW - 5, UTC)
    assert reading.stale is False


@pytest.mark.asyncio
async def test_room_and_household_totals_only_read_the_requested_kind():
    kitchen = EnergySensor("sensor.kitchen_energy", ENERGY, "cocina")
    laundry = EnergySensor("sensor.laundry_energy", ENERGY, "lavadero")
    unassigned = EnergySensor("sensor.unassigned_energy", ENERGY)
    power = EnergySensor("sensor.kitchen_power", POWER, "cocina")
    provider = StateProvider(
        state(kitchen.entity_id, "100", ENERGY, "Wh"),
        state(laundry.entity_id, "0.2", ENERGY, "kWh"),
        state(unassigned.entity_id, "0.0003", ENERGY, "MWh"),
        state(power.entity_id, "0.8", POWER, "kW"),
    )
    service = monitor(provider, kitchen, laundry, unassigned, power)

    single = await service.summarize(
        ENERGY, entity_id=laundry.entity_id, actor=HOUSEHOLD
    )
    assert (single.entity_id, single.total) == (laundry.entity_id, Decimal("0.2"))
    assert (single.available_count, single.selected_count) == (1, 1)
    assert provider.calls == [laundry.entity_id]
    provider.calls.clear()

    room = await service.summarize(ENERGY, room="cocina", actor=HOUSEHOLD)
    assert (room.room, room.unit, room.total) == ("cocina", "kWh", Decimal("0.1"))
    assert (room.available_count, room.selected_count) == (1, 1)
    assert provider.calls == [kitchen.entity_id]
    provider.calls.clear()

    energy = await service.summarize(ENERGY, actor=HOUSEHOLD)
    assert energy.as_of == datetime.fromtimestamp(NOW, UTC)
    assert (energy.room, energy.unit, energy.total) == (None, "kWh", Decimal("0.6"))
    assert (energy.available_count, energy.selected_count) == (3, 3)
    assert tuple(reading.sensor for reading in energy.readings) == (
        kitchen,
        laundry,
        unassigned,
    )
    assert set(provider.calls) == {
        kitchen.entity_id,
        laundry.entity_id,
        unassigned.entity_id,
    }
    provider.calls.clear()

    instantaneous = await service.summarize(POWER, actor=HOUSEHOLD)
    assert (instantaneous.unit, instantaneous.total) == ("W", Decimal("800"))
    assert provider.calls == [power.entity_id]


@pytest.mark.asyncio
async def test_partial_and_zero_coverage_preserve_zero_and_never_fill_energy_from_power():
    valid = EnergySensor("sensor.energy_zero", ENERGY, "cocina")
    unavailable = EnergySensor("sensor.energy_unavailable", ENERGY, "cocina")
    unknown = EnergySensor("sensor.energy_unknown", ENERGY, "lavadero")
    aged = EnergySensor("sensor.energy_aged", ENERGY, "lavadero")
    malformed = EnergySensor("sensor.energy_bad", ENERGY, "lavadero")
    power = EnergySensor("sensor.power_active", POWER, "lavadero")
    provider = StateProvider(
        state(valid.entity_id, "0", ENERGY, "kWh"),
        state(unavailable.entity_id, "unavailable", None, None),
        state(unknown.entity_id, "unknown", ENERGY, "Wh"),
        state(aged.entity_id, "900", ENERGY, "Wh", reported_at=NOW - 30),
        state(malformed.entity_id, "error", ENERGY, "Wh"),
        state(power.entity_id, "3", POWER, "kW"),
    )
    service = monitor(provider, valid, unavailable, unknown, aged, malformed, power)

    summary = await service.summarize(ENERGY, actor=HOUSEHOLD)
    assert (summary.available_count, summary.selected_count, summary.total) == (
        1,
        5,
        Decimal(0),
    )
    assert [item.status for item in summary.readings] == [
        ReadingStatus.AVAILABLE,
        ReadingStatus.UNAVAILABLE,
        ReadingStatus.UNKNOWN,
        ReadingStatus.STALE,
        ReadingStatus.NON_NUMERIC,
    ]
    assert [item.stale for item in summary.readings] == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert summary.readings[3].observed_at == datetime.fromtimestamp(NOW - 30, UTC)
    assert all(item.value is None for item in summary.readings[1:])

    zero = await service.summarize(ENERGY, room="lavadero", actor=HOUSEHOLD)
    assert (zero.available_count, zero.selected_count, zero.total) == (0, 3, None)
    assert power.entity_id not in provider.calls

    empty = await monitor(provider, power).summarize(ENERGY, actor=HOUSEHOLD)
    assert (empty.available_count, empty.selected_count, empty.total) == (0, 0, None)
    assert power.entity_id not in provider.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("device_class", "unit"),
    [(POWER, "kWh"), (ENERGY, "W"), (ENERGY, "KWh"), (None, "kWh"), (ENERGY, [])],
)
async def test_incompatible_device_class_or_unit_has_no_value(device_class, unit):
    sensor = EnergySensor("sensor.energy", ENERGY)
    provider = StateProvider(state(sensor.entity_id, "10", device_class, unit))

    summary = await monitor(provider, sensor).summarize(ENERGY, actor=HOUSEHOLD)

    assert summary.readings[0].status is ReadingStatus.INCOMPATIBLE
    assert summary.readings[0].original_unit == (
        unit if isinstance(unit, str) else None
    )
    assert (summary.available_count, summary.selected_count, summary.total) == (
        0,
        1,
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["NaN", "sNaN", "Infinity", "1e999", "1e-999"])
async def test_non_finite_states_cannot_contribute(raw):
    sensor = EnergySensor("sensor.energy", ENERGY)
    provider = StateProvider(state(sensor.entity_id, raw, ENERGY, "Wh"))

    reading = await monitor(provider, sensor).read_sensor(
        sensor.entity_id, actor=HOUSEHOLD
    )

    assert reading.status is ReadingStatus.NON_NUMERIC
    assert reading.value is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reported_at", "updated_at", "expected"),
    [
        (None, NOW - 1, ReadingStatus.AVAILABLE),
        (None, None, ReadingStatus.STALE),
        (NOW + 1, NOW - 1, ReadingStatus.STALE),
    ],
)
async def test_report_time_fallback_and_unverifiable_dates(
    reported_at, updated_at, expected
):
    sensor = EnergySensor("sensor.energy", ENERGY)
    provider = StateProvider(
        state(
            sensor.entity_id,
            "2",
            ENERGY,
            "kWh",
            reported_at=reported_at,
            updated_at=updated_at,
        )
    )

    reading = await monitor(provider, sensor).read_sensor(
        sensor.entity_id, actor=HOUSEHOLD
    )

    assert reading.status is expected
    instant = reported_at if reported_at is not None else updated_at
    assert reading.observed_at == (
        datetime.fromtimestamp(instant, UTC) if instant is not None else None
    )


@pytest.mark.asyncio
async def test_expired_source_and_404_are_excluded():
    expired = EnergySensor("sensor.expired", POWER)
    missing = EnergySensor("sensor.missing", POWER)
    provider = StateProvider()
    provider.observations[expired.entity_id] = EntityObservation(
        expired.entity_id,
        state(expired.entity_id, "350", POWER, "W"),
        exists=True,
        source=REST_SOURCE,
        observed_at=NOW - 2,
        expires_at=NOW,
        reason=None,
    )
    provider.observations[missing.entity_id] = EntityObservation(
        missing.entity_id,
        exists=False,
        source=REST_SOURCE,
        observed_at=NOW - 1,
        expires_at=NOW + 1,
        reason=None,
    )

    result = await monitor(provider, expired, missing).summarize(POWER, actor=HOUSEHOLD)

    assert (result.available_count, result.selected_count, result.total) == (0, 2, None)
    assert [item.status for item in result.readings] == [
        ReadingStatus.STALE,
        ReadingStatus.MISSING,
    ]
    assert all(item.stale for item in result.readings)


@pytest.mark.asyncio
async def test_rejects_monetary_queries_and_unselected_entities_before_io():
    selected = EnergySensor("sensor.selected", ENERGY)
    provider = StateProvider(state("sensor.private", "2", ENERGY, "kWh"))
    service = monitor(provider, selected)

    for query in ("cost", "bill"):
        with pytest.raises(EnergyCostUnavailable, match="energy_tariff_unverified"):
            await service.summarize(query, actor=HOUSEHOLD)
    for entity_id in ("sensor.private", "switch.supply"):
        with pytest.raises(ValueError, match="energy_sensor_not_selected"):
            await service.read_sensor(entity_id, actor=HOUSEHOLD)
        with pytest.raises(ValueError, match="energy_sensor_not_selected"):
            await service.summarize(ENERGY, entity_id=entity_id, actor=HOUSEHOLD)
    with pytest.raises(ValueError, match="energy_sensor_kind_mismatch"):
        await service.summarize(POWER, entity_id=selected.entity_id, actor=HOUSEHOLD)
    with pytest.raises(ValueError, match="energy_summary_scope_ambiguous"):
        await service.summarize(
            ENERGY, entity_id=selected.entity_id, room="cocina", actor=HOUSEHOLD
        )

    assert provider.calls == []


def test_selection_rejects_switches_and_duplicate_entities():
    sensor = EnergySensor("sensor.selected", ENERGY)

    with pytest.raises(ValueError, match="duplicate_energy_sensor"):
        monitor(StateProvider(), sensor, sensor)
    with pytest.raises(ValueError, match="invalid_energy_sensor_id"):
        EnergySensor("switch.supply", ENERGY)


@pytest.mark.asyncio
async def test_permissions_are_checked_before_reads_and_again_before_returning_data():
    sensor = EnergySensor("sensor.shared", ENERGY)
    provider = StateProvider(state(sensor.entity_id, "2", ENERGY, "kWh"))
    service = monitor(provider, sensor)
    reader = HomeActor(ActorKind.DASHBOARD, "dashboard", frozenset({"home:read"}))
    controller = HomeActor(
        ActorKind.DASHBOARD, "dashboard", frozenset({"home:control"})
    )
    for actor in (controller, None):
        with pytest.raises(PermissionError, match="home_read_denied"):
            await service.read_sensor(sensor.entity_id, actor=actor)
        with pytest.raises(PermissionError, match="home_read_denied"):
            await service.summarize("cost", actor=actor)
    assert provider.calls == []

    assert (await service.read_sensor(sensor.entity_id, actor=reader)).value == Decimal(
        2
    )
    assert (await service.summarize(ENERGY, actor=reader)).total == Decimal(2)

    expiring = HomeActor(
        ActorKind.DASHBOARD,
        "dashboard",
        reader.permissions,
        lambda: len(provider.calls) == 2,
    )
    with pytest.raises(PermissionError, match="home_read_denied"):
        await service.summarize(ENERGY, actor=expiring)
    assert provider.calls == [sensor.entity_id] * 3


@pytest.mark.asyncio
async def test_real_ha_client_reads_only_selected_entity_and_excludes_failed_refresh():
    requests: list[tuple[str, str]] = []
    sensor = EnergySensor("sensor.meter", ENERGY)

    async def handle(request: web.Request) -> web.Response:
        requests.append((request.method, request.path))
        if len(requests) == 1:
            return web.json_response(
                state(sensor.entity_id, "1230", ENERGY, "Wh", reported_at=time())
            )
        raise web.HTTPServiceUnavailable()

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    async with TestServer(app) as server:
        client = HomeAssistantClient(
            str(server.make_url("/")).rstrip("/"), "test-token"
        )
        try:
            service = EnergyMonitor(client, [sensor], max_staleness_s=30)
            before = await service.summarize(ENERGY, actor=HOUSEHOLD)
            after = await service.summarize(ENERGY, actor=HOUSEHOLD)
        finally:
            await client.close()

    assert (before.total, before.available_count, before.selected_count) == (
        Decimal("1.23"),
        1,
        1,
    )
    assert (after.total, after.available_count, after.selected_count) == (None, 0, 1)
    assert after.readings[0].status is ReadingStatus.SOURCE_UNAVAILABLE
    assert requests == [("GET", "/api/states/sensor.meter")] * 2
