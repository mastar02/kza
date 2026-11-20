"""Exercise the HA producer with dated observations and real alert episodes."""

import asyncio
import inspect
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import aiohttp
import pytest

import src.home_assistant.ha_client as ha_module
from src.alerts.alert_manager import AlertDelivery, AlertManager, AlertType
from src.alerts.alert_scheduler import AlertScheduler
from src.alerts.ha_alert_source import HAAlertSource
from src.conversation.command_turns import CommandTurns
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.home_assistant.ha_client import HAStatesSnapshot
from src.home_assistant.state_projection import EntityObservation
from src.pipeline.feature_manager import ALERTS_NONE, ALERTS_UNAVAILABLE, FeatureManager
from tests.integration.test_ha_entity_stream import _baseline, _delta, _hold_retry
from tests.integration.test_ha_transport_ownership import (
    WIRE_TIME, _Transport, _client, _compact_state, _reached, _rest_state,
)

pytestmark = pytest.mark.asyncio

DOOR = "binary_sensor.synthetic_contact"
WINDOW = "binary_sensor.synthetic_window"
BATTERY = "sensor.synthetic_power"
LOCK = "lock.synthetic"
ALARM = "alarm_control_panel.synthetic"


def instant(epoch):
    return datetime.fromtimestamp(epoch, UTC)


def state(entity_id, value, changed_at):
    return {"entity_id": entity_id, "state": value, "attributes": {},
            "last_changed": instant(changed_at).isoformat()}


class Reader:
    """Leaf exposing only the public HA read contract and measured invocations."""

    def __init__(self):
        self.now = 100
        self.generation = 1
        self.revision = 0
        self.callbacks = []
        self.observations = {}
        self.snapshot = HAStatesSnapshot(False, None, "ha_states_unavailable")
        self.io = []

    def register_state_callback(self, callback, *, with_observation):
        assert with_observation
        self.callbacks.append(callback)

    def unregister_state_callback(self, callback, *, with_observation):
        assert with_observation
        self.callbacks.remove(callback)

    def current_state_stream_generation(self):
        return self.generation

    def get_entity_observation(self, entity_id):
        self.io.append(("observation", entity_id))
        return self.observations.get(entity_id, EntityObservation(entity_id))

    async def get_state_snapshot(self):
        self.io.append(("snapshot", None))
        return self.snapshot

    def publish(self, entity_id, value, *, changed_at=None):
        changed_at = self.now if changed_at is None else changed_at
        payload = state(entity_id, value, changed_at) if value is not None else None
        self.revision += 1
        observation = self.observations[entity_id] = EntityObservation(
            entity_id, payload, payload is not None, "ha-ws-entities", self.now, self.now + 6,
            reason=None, generation=self.generation, revision=self.revision,
        )
        for callback in self.callbacks:
            assert callback(entity_id, observation) is None

    def rest(self, *states):
        self.snapshot = HAStatesSnapshot(True, instant(self.now), None, states, instant(self.now + 6))


@pytest.fixture
def reader():
    return Reader()


@pytest.fixture
def door_settings():
    return {"door_window": {DOOR: {"name": "La ventana", "open_seconds": 10}}}


async def test_ha_callback_only_enqueues_a_selected_delta_without_await_or_io(reader, door_settings, monkeypatch):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
    await source.start()
    try:
        callback, = reader.callbacks
        assert not inspect.iscoroutinefunction(callback)
        payload = state(DOOR, "on", 90)
        puts = []
        put = source._queue.put_nowait

        def spy(item):
            puts.append(item)
            put(item)

        monkeypatch.setattr(source._queue, "put_nowait", spy)
        reader.publish(DOOR, "on", changed_at=90)
        reader.publish("binary_sensor.another_window", "on", changed_at=90)
        assert puts == [(DOOR, reader.observations[DOOR])]
        assert puts[0][1].state == payload
        assert reader.io == [] and manager.get_history() == []
        await source._queue.join()
        assert reader.io == []
        alert, = manager.get_history()
        assert (alert.source, alert.alert_type, alert.condition) == (DOOR, AlertType.SECURITY, "door_open_unexpected")
    finally:
        await source.stop()
    assert reader.callbacks == [] and not source.running


async def test_thresholds_and_confirmed_recovery_are_independent_per_condition(reader, door_settings):
    manager = AlertManager()
    config = {**door_settings, "battery": {BATTERY: {"below_percent": 20}},
              "unavailable": {DOOR: {"grace_seconds": 20}}}
    source = HAAlertSource(reader, manager, config, clock=lambda: reader.now)
    await source.start()
    try:
        reader.rest(state(DOOR, "off", 90), state(BATTERY, "20", 90))
        await source.reconcile()
        reader.now += 1
        reader.publish(DOOR, "on")
        await source._queue.join()
        assert manager.get_history() == []
        reader.now += 10
        reader.rest(state(DOOR, "on", 101), state(BATTERY, "19", 109))
        await source.reconcile()
        door, = [alert for alert in manager.get_active_alerts() if alert.source == DOOR]
        battery, = [alert for alert in manager.get_active_alerts() if alert.source == BATTERY]
        assert "presente al iniciar" not in door.message + battery.message

        reader.now = 112
        reader.publish(DOOR, "unavailable")
        await source._queue.join()
        assert manager.get_active_alerts() == [battery, door]
        reader.now = 132
        reader.rest(state(DOOR, "unavailable", 112), state(BATTERY, "20", 132))
        await source.reconcile()
        offline, = [alert for alert in manager.get_active_alerts() if alert.condition == "offline"]
        assert door.resolved_at is None and battery.resolved_at is not None
        assert offline.source == DOOR
        reader.now = 133
        reader.publish(DOOR, "off")
        await source._queue.join()
        assert door.resolved_at is not None and offline.resolved_at is not None
        assert manager.get_active_alerts() == []
    finally:
        await source.stop()


async def test_disconnection_retained_snapshot_and_reconnection_do_not_close_an_episode(monkeypatch, door_settings):
    transport = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", transport.session_factory)
    events = transport.socket("events", initial_event={"a": {DOOR: _compact_state()}})
    transport.socket("calls")
    recovered = transport.socket("events", initial_event={"a": {DOOR: _compact_state("on")}})
    manager = AlertManager()
    async with _client(transport) as client:
        retrying, retry, _ = _hold_retry(client, transport)
        source = HAAlertSource(client, manager, door_settings)
        scheduler = AlertScheduler(manager)
        features = FeatureManager(alert_manager=manager, alert_scheduler=scheduler, ha_alert_source=source)
        await source.start()
        scheduler_task = asyncio.create_task(scheduler.start())
        try:
            await _baseline(client, events)
            transport.response([_rest_state(DOOR)])
            await source.reconcile()
            await _delta(events, {"c": {DOOR: {"+": {"s": "on", "lc": WIRE_TIME + 1}}}})
            await source._queue.join()
            original, = manager.get_active_alerts()
            assert await manager.mark_processed(original.alert_id)
            assert features.get_alert_summary() == ALERTS_NONE

            events.frames.put_nowait(aiohttp.ClientConnectionError("synthetic WS disconnect"))
            await _reached(retrying)
            assert client.current_state_stream_generation() is None
            assert transport.state_reads == 1
            assert source.get_status()["unknown"] == 1 and original.resolved_at is None
            assert features.get_alert_summary() == ALERTS_UNAVAILABLE
            transport.response(status=503)
            await source.reconcile()
            assert source.get_status()["unknown"] == 1 and original.resolved_at is None

            retry.set()
            await _reached(recovered.stream_waiting)
            transport.response(status=503)
            await source.reconcile()
            assert source.get_status()["known"] == 1
            assert manager.get_history() == [original] and original.resolved_at is None

            await _delta(recovered, {"c": {DOOR: {"+": {"s": "off", "lc": WIRE_TIME + 2}}}})
            await source._queue.join()
            assert original.resolved_at is not None
            transport.assert_no_effects_or_duplicate_receivers()
        finally:
            scheduler_task.cancel()
            await asyncio.gather(scheduler_task, return_exceptions=True)
            await source.stop()


async def test_none_stale_or_missing_states_are_unknown(reader, door_settings):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
    await source.start()
    try:
        reader.publish(DOOR, "on", changed_at=80)
        reader.now += 6  # The delta expired before the consumer picked it up.
        await source._queue.join()
        assert manager.get_history() == [] and source.get_status()["unknown"] == 1
        reader.rest(state(DOOR, "on", 80))
        await source.reconcile()
        alert, = manager.get_active_alerts()
        reader.now += 1
        reader.publish(DOOR, None)
        await source._queue.join()
        assert source.get_status()["unknown"] == 1 and alert.resolved_at is None
        reader.now += 1
        reader.rest()
        await source.reconcile()
        assert source.get_status()["unknown"] == 1 and alert.resolved_at is None
        reader.rest(state(DOOR, "off", reader.now))
        reader.now += 6
        await source.reconcile()
        assert source.get_status()["unknown"] == 1 and alert.resolved_at is None
    finally:
        await source.stop()


async def test_pending_delta_from_retired_generation_cannot_close_the_episode(reader, door_settings):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
    await source.start()
    try:
        reader.publish(DOOR, "on", changed_at=80)
        await source._queue.join()
        original, = manager.get_active_alerts()
        reader.now += 1
        reader.publish(DOOR, "off")
        reader.generation += 1
        reader.publish(DOOR, "on", changed_at=80)
        await source._queue.join()
        assert original.resolved_at is None
        assert manager.get_history() == [original]
        assert source.get_status()["known"] == 1
    finally:
        await source.stop()


async def test_rest_read_that_overlaps_a_ws_delta_cannot_close_the_new_episode(reader, door_settings, monkeypatch):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
    await source.start()
    started, release = asyncio.Event(), asyncio.Event()
    try:
        reader.rest(state(DOOR, "off", 90))
        await source.reconcile()
        read_snapshot = reader.get_state_snapshot

        async def delayed_snapshot():
            started.set()
            await release.wait()
            return await read_snapshot()

        monkeypatch.setattr(reader, "get_state_snapshot", delayed_snapshot)
        reader.now += 1
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(source.reconcile())
            await started.wait()
            reader.now += 1
            reader.publish(DOOR, "on", changed_at=80)
            await source._queue.join()
            alert, = manager.get_active_alerts()
            reader.now += 1
            reader.rest(state(DOOR, "off", 90))
            release.set()
        assert manager.get_active_alerts() == [alert]
    finally:
        await source.stop()


async def test_overflow_blocks_changes_until_a_fresh_reconciliation(reader, door_settings):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, {**door_settings, "queue_maxsize": 1}, clock=lambda: reader.now)
    await source.start()
    try:
        reader.publish(DOOR, "on", changed_at=80)
        await source._queue.join()
        first, = manager.get_active_alerts()
        reader.now += 1
        reader.publish(DOOR, "off")
        reader.publish(DOOR, "on", changed_at=80)
        assert source.get_status() == {"running": True, "uncertain": True, "overflow": 1, "known": 0, "unknown": 1}
        await source._queue.join()
        await source.reconcile()
        assert source.uncertain and first.resolved_at is None
        reader.now += 1
        reader.rest(state(DOOR, "on", 80))
        await source.reconcile()
        assert not source.uncertain and source.get_status()["overflow"] == 1
        assert manager.get_history() == [first]
        reader.now += 1
        reader.publish(DOOR, "off")
        await source._queue.join()
        assert first.resolved_at is not None
    finally:
        await source.stop()


async def test_restarted_process_notices_existing_condition_once_even_with_reannounce_enabled(reader, door_settings):
    for expected_start in (100, 150):
        reader.now = expected_start
        delivery = AsyncMock(return_value=ResponseResult("", GenerationState.COMPLETE, DeliveryState.COMPLETE))
        manager = AlertManager(tts_callback=delivery, reannounce_interval_seconds=1, clock=lambda: reader.now)
        source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
        await source.start()
        try:
            for _ in range(2):
                reader.now += 10
                reader.rest(state(DOOR, "on", 80))
                await source.reconcile()
                await asyncio.sleep(0)
            alert, = manager.get_history()
            assert "presente al iniciar" in alert.message
            delivery.assert_awaited_once_with(f"Alerta importante: {alert.message}")
            reader.now += 1
            reader.publish(DOOR, "off")
            await source._queue.join()
            reader.now += 11
            reader.publish(DOOR, "on", changed_at=reader.now - 10)
            await source._queue.join()
            reopened, = manager.get_active_alerts()
            assert "presente al iniciar" not in reopened.message and reopened is not alert
        finally:
            await source.stop()


async def test_recognized_door_closes_and_reopens_in_order_while_voice_is_busy(reader):
    notices = asyncio.Queue()

    async def speak(message):
        notices.put_nowait(message)
        return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    busy = False
    turns = CommandTurns(lambda: busy)
    manager = AlertManager(tts_callback=speak)
    config = {"door_window": {DOOR: {"name": "La ventana", "open_seconds": 0}}}
    source = HAAlertSource(reader, manager, config, clock=lambda: reader.now)
    await source.start(turns)
    try:
        reader.publish(DOOR, "on", changed_at=90)
        assert "presente al iniciar" in await asyncio.wait_for(notices.get(), 2)
        first, = manager.get_active_alerts()
        assert await manager.mark_processed(first.alert_id)

        busy = True
        reader.now += 1
        reader.publish(DOOR, "off")
        reader.now += 1
        reader.publish(DOOR, "on")
        await asyncio.wait_for(source._queue.join(), 2)
        reopened, = manager.get_active_alerts()
        assert first.resolved_at is not None and reopened is not first
        assert not reopened.processed and reopened.delivery is AlertDelivery.PENDING
        assert notices.empty()

        busy = False
        turns.settle()
        assert await asyncio.wait_for(notices.get(), 2) == "Alerta importante: La ventana sigue abierta"
        assert reopened.delivery is AlertDelivery.DELIVERED
    finally:
        await source.stop()


async def test_door_closes_while_its_notice_is_in_playback(reader, door_settings):
    entered, release = asyncio.Event(), asyncio.Event()

    async def speak(message):
        entered.set()
        await release.wait()
        return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    manager = AlertManager(tts_callback=speak)
    source = HAAlertSource(reader, manager, door_settings, clock=lambda: reader.now)
    await source.start()
    try:
        reader.publish(DOOR, "on", changed_at=80)
        await asyncio.wait_for(entered.wait(), 2)
        original, = manager.get_active_alerts()
        reader.now += 1
        reader.publish(DOOR, "off")
        await asyncio.wait_for(source._queue.join(), 2)
        assert original.resolved_at is not None and manager.get_active_alerts() == []
        assert original.delivery is AlertDelivery.PENDING
    finally:
        release.set()
        await source.stop()


async def test_history_eviction_does_not_reannounce_either_startup_contact(reader):
    notices = asyncio.Queue()

    async def speak(message):
        notices.put_nowait(message)
        return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    manager = AlertManager(max_history=1, tts_callback=speak, reannounce_interval_seconds=1,
                           clock=lambda: reader.now)
    config = {"door_window": {
        DOOR: {"name": "La puerta", "open_seconds": 0},
        WINDOW: {"name": "La ventana", "open_seconds": 0},
    }}
    source = HAAlertSource(reader, manager, config, clock=lambda: reader.now)
    await source.start()
    try:
        for _ in range(3):
            reader.now += 5
            reader.rest(state(DOOR, "on", 80), state(WINDOW, "on", 80))
            await source.reconcile()
            await asyncio.sleep(0)
        initial = [notices.get_nowait(), notices.get_nowait()]
        assert set(initial) == {
            "Alerta importante: La puerta sigue abierta (presente al iniciar)",
            "Alerta importante: La ventana sigue abierta (presente al iniciar)",
        }
        assert notices.empty() and len(manager.get_history()) == 1
        assert {alert.source for alert in manager.get_active_alerts()} == {DOOR, WINDOW}
        door = next(alert for alert in manager.get_active_alerts() if alert.source == DOOR)
        assert manager.get_alert(door.alert_id) is door
        assert await manager.mark_processed(door.alert_id)

        reader.now += 1
        reader.publish(DOOR, "off")
        reader.now += 1
        reader.publish(DOOR, "on")
        assert await asyncio.wait_for(notices.get(), 2) == "Alerta importante: La puerta sigue abierta"
        assert door.resolved_at is not None and notices.empty()
    finally:
        await source.stop()


async def test_only_explicit_read_only_rules_and_no_pattern_producer(reader):
    manager = AlertManager()
    source = HAAlertSource(reader, manager, {"unavailable": {LOCK: {}, ALARM: {}}}, clock=lambda: reader.now)
    scheduler = AlertScheduler(manager)
    scheduler.register_ha_source(source.reconcile, 15)
    await source.start()
    try:
        reader.now = 400
        reader.rest(state(LOCK, "unavailable", 99), state(ALARM, "armed_away", 99),
                    state(DOOR, "on", 1), state(BATTERY, "2", 1))
        await source.reconcile()
        alert, = manager.get_active_alerts()
        assert (alert.source, alert.condition) == (LOCK, "offline")
        assert reader.io == [("snapshot", None), ("observation", LOCK), ("observation", ALARM)]
        status = scheduler.get_status()
        assert status["registered_handlers"]["ha"] == 1
        assert status["pattern_alerts"] == "not_connected"
        assert status["registered_handlers"]["pattern"] == 0
    finally:
        await source.stop()
    with pytest.raises(ValueError, match="Invalid entity"):
        HAAlertSource(reader, manager, {"door_window": {LOCK: {}}})
