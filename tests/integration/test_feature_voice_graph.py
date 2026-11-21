"""Application feature factory reaches the shared real PCM owner."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiohttp import web
import numpy as np
import pytest

from src.alerts.alert_manager import AlertDelivery, AlertType, AlertPriority
from src.application import ApplicationFactories
from src.core.household_clock import HouseholdClock
from src.core.voice_delivery import VoiceDeliveryError
from src.intercom.intercom_system import Announcement
from src.main import build_application
from src.notifications.smart_notifications import Notification, NotificationChannel
from src.pipeline.command_event import CommandEvent
from src.pipeline.voice_services import build_voice_services
from src.reminders.reminder_store import ReminderStore, ReminderScope
from src.routines.routine_executor import RoutineExecutor
from src.tts import piper_tts
from src.vectordb import chroma_sync
from tests.integration.test_ha_transport_ownership import _compact_state, _rest_state
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, offline_config
from tests.mocks.audio_output import AudioBackend
from tests.mocks.speech import PCMEngine
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


@pytest.fixture
def daylight_clock():
    """Freeze the household clock at 12:00 in Montevideo."""
    noon = datetime(2026, 11, 21, 15, tzinfo=UTC).timestamp()
    return HouseholdClock(time_source=lambda: noon)


async def test_all_feature_callbacks_share_one_pcm_output_and_correct_origin(monkeypatch, tmp_path, daylight_clock):
    original, engine, backend, _, amplifier = voice_graph(monkeypatch)
    store = ReminderStore(str(tmp_path / 'reminders.db'))
    await store.initialize()
    ha = MagicMock()
    config = {'tts': {'streaming': {'default_output_zone': 'zone_origin'}},
              'alerts': {'scheduler': {'enabled': False}},
              'rooms': {'enabled': True, 'origin': {'name': 'Origin'}}, 'reminders': {'output_zone': 'zone_origin'}}
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        services = build_voice_services(config, tts=engine, zone_manager=original.zone_manager,
            shutdown=stack, reminder_store=store, ha_client=ha, household_clock=daylight_clock,
            alert_muted=lambda: False)
        features = services.features
        await features.intercom._load_zones()
        await features.timer_manager._announce_warning(SimpleNamespace(name='pasta', zone_id='zone_origin'), 30)
        announcement = Announcement('announcement', 'Anuncio de prueba.', ['zone_origin'])
        await features.intercom._deliver_announcement(announcement)
        assert announcement.delivered and announcement.failed_zones == []
        notice = Notification('notice', 'Aviso', 'Mensaje de prueba.', user_id='synthetic', zone_id='zone_origin')
        await features.notifications._deliver_to_channel(notice, 'synthetic', NotificationChannel.VOICE, {})
        selected_before_alert = amplifier.select_zone.call_count
        alert = await features.alert_manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Alerta de prueba.')
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.DELIVERED, None)
        assert amplifier.select_zone.call_count == selected_before_alert + 1
        amplifier.select_zone.assert_called_with(1)
        reminder = await store.create(user_id='synthetic', text='Recordatorio de prueba.', trigger_at=time.time()-1,
            scope=ReminderScope.SHARED)
        await services.reminders._fire_reminder(reminder)
        assert (await store.get_by_id(reminder.id)).state == 'fired'
        executor = RoutineExecutor(tts_engine=features.voice_delivery)
        result = await executor._execute_tts_speak({'text': 'Rutina de prueba.', 'zone': 'zone_origin'}, {})
        assert result.success
        assert len(engine.calls) == 6 and engine.peak_active == 1
        expected = np.concatenate([engine.audio(text) for text in engine.calls])
        np.testing.assert_array_equal(backend.heard()[:, 0], expected)
        amplifier.select_zone.assert_called_with(1)
        amplifier.select_all_zones.assert_not_called()
        ha.call_service.assert_not_called()
    assert services.responses.get_delivery_status()['closed']


async def test_real_ha_delta_reaches_configured_pcm_and_yields_to_an_active_command(monkeypatch, tmp_path, daylight_clock):
    door = "binary_sensor.synthetic_contact"
    subscription = asyncio.get_running_loop().create_future()
    ws_requests = []

    async def api(_request):
        return web.json_response({"message": "API running"})

    async def states(_request):
        return web.json_response([_rest_state(door)])

    async def websocket(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "auth_required"})
        assert (await socket.receive_json())["type"] == "auth"
        await socket.send_json({"type": "auth_ok"})
        async for message in socket:
            frame = message.json()
            ws_requests.append(frame["type"])
            assert frame["type"] == "subscribe_entities"
            ident = frame["id"]
            await socket.send_json({"id": ident, "type": "result", "success": True})
            await socket.send_json({"id": ident, "type": "event", "event": {"a": {door: _compact_state()}}})
            subscription.set_result((socket, ident))
        return socket

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.main.HouseholdClock", lambda _timezone: daylight_clock)
    monkeypatch.setattr(chroma_sync, "SentenceTransformer", lambda *_args, **_kwargs: OfflineEncoder())
    backend, engine = AudioBackend(), PCMEngine(frames=240)
    monkeypatch.setattr(piper_tts, "sd", backend)
    amplifier = MagicMock(audio_output_device=7)
    entered, release = threading.Event(), threading.Event()
    stt = OfflineSTT(text="¿Qué hora es?")
    transcribe = stt.transcribe_with_confidence

    def gated_stt(*args):
        entered.set()
        assert release.wait(2), "Synthetic STT was not released"
        return transcribe(*args)

    monkeypatch.setattr(stt, "transcribe_with_confidence", gated_stt)
    leaves = ApplicationFactories(stt=lambda _: stt, tts=lambda _: engine,
                                  amplifier=lambda **_: amplifier, preflight=lambda: None)
    app = web.Application()
    app.add_routes([web.get("/api/", api), web.get("/api/states", states), web.get("/api/websocket", websocket)])
    async with AsyncExitStack() as stack:
        runner = web.AppRunner(app)
        await runner.setup()
        stack.push_async_callback(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        config = offline_config(tmp_path, url)
        config["home_assistant"]["state_prefetch"]["enabled"] = True
        config["alerts"] = {"ha": {"output_zone": "zone_alert", "door_window": {
            door: {"name": "La ventana", "open_seconds": 0},
        }}}
        config["zones"] = {"enabled": True, "zone_list": [
            {"id": "zone_alert", "name": "Avisos", "mic_device_index": 0, "ma1260_zone": 2},
        ]}
        graph = await build_application(config, shutdown=stack, factories=leaves)
        stack.push_async_callback(graph._stop_services, None, None)
        features = graph.voice_services.features
        source = features.ha_alert_source
        loop = graph.pipeline.audio_loop
        waiting_for_turn = asyncio.Event()
        wait_idle = loop.command_turns.wait_idle

        async def observed_wait():
            if loop.command_turns.active:
                waiting_for_turn.set()
            await wait_idle()

        monkeypatch.setattr(loop.command_turns, "wait_idle", observed_wait)
        await graph._start_services()
        async with asyncio.timeout(2):
            socket, ident = await subscription
            while not features.alerts_available():
                await asyncio.sleep(.001)

        event = CommandEvent(np.full(16000, .1, dtype=np.float32), room_id="")
        command = loop._track(asyncio.create_task(graph.pipeline.process_command(event)), loop._dispatches)
        command.add_done_callback(loop._settle_turn)
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await socket.send_json({"id": ident, "type": "event", "event": {"c": {door: {
                "+": {"s": "on", "lc": time.time()},
            }}}})
            await asyncio.wait_for(waiting_for_turn.wait(), 2)
            assert not command.done() and engine.calls == []
            pending, = features.alert_manager.get_history()
            assert pending.delivery is AlertDelivery.PENDING
        finally:
            release.set()
        result = await asyncio.wait_for(command, 2)
        assert result["path"] == "fast_clock" and result["task_completed"] and result["latency_ms"] < 300
        await asyncio.wait_for(source._queue.join(), 2)
        alert, = features.alert_manager.get_history()
        async with asyncio.timeout(2):
            while alert.delivery is AlertDelivery.PENDING:
                await asyncio.sleep(.001)
        assert (alert.source, alert.delivery) == (door, AlertDelivery.DELIVERED)
        speech = "Alerta importante: La ventana sigue abierta"
        assert engine.calls == [result["response"], speech] and engine.peak_active == 1
        np.testing.assert_array_equal(backend.heard()[:, 0], np.concatenate([engine.audio(text) for text in engine.calls]))
        amplifier.select_zone.assert_called_once_with(2)
        amplifier.select_all_zones.assert_not_called()
        assert ws_requests == ["subscribe_entities"]
    assert not source.running and graph.ha_client._state_observers == []


async def test_feature_delivery_failure_is_not_announced_as_completed_or_retried_via_ha(monkeypatch):
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    backend.fail_start = True
    ha = MagicMock()
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'enabled': False}, 'rooms': {'origin': {'name': 'Origin'}}}, tts=engine,
            shutdown=stack, zone_manager=original.zone_manager, ha_client=ha)
        intercom = services.features.intercom
        origin = original.zone_manager.get_zone('zone_origin')
        origin.media_player_entity = 'media_player.synthetic_origin'
        origin.tts_target = 'tts.synthetic_origin'
        await intercom._load_zones()
        announcement = Announcement('announcement', 'Aviso.', ['zone_origin'])
        await intercom._deliver_announcement(announcement)
        assert not announcement.delivered and announcement.failed_zones == ['zone_origin']
        assert len(backend.outputs) == 1 and len(engine.calls) == 1
        ha.call_service.assert_not_called()
        ha.get_entities.assert_not_called()


async def test_alert_output_failure_is_recorded_on_the_alert_not_swallowed(monkeypatch, daylight_clock):
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    backend.fail_start = True
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'scheduler': {'enabled': False}, 'ha': {'output_zone': 'zone_origin'}}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager,
            household_clock=daylight_clock, alert_muted=lambda: False)
        alert = await services.features.alert_manager.create_alert(
            AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.')
        assert alert.delivery is AlertDelivery.FAILED
        assert alert.delivery_reason and alert.delivery_at is not None
        assert services.features.alert_manager.get_history() == [alert]
        assert len(engine.calls) == 1


async def test_alert_notice_that_never_started_is_retried_once_on_the_real_graph(monkeypatch, daylight_clock):
    """T036-U1-Q1: synthesis failed before any PCM reached the player, so the next check retries once."""
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    engine.failure = RuntimeError('synthetic synthesis failure')
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'scheduler': {'enabled': False}, 'ha': {'output_zone': 'zone_origin'}}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager,
            household_clock=daylight_clock, alert_muted=lambda: False)
        manager = services.features.alert_manager
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.')
        assert alert.delivery is AlertDelivery.FAILED and alert.delivery_attempts == 1
        assert backend.heard().size == 0
        engine.failure = None
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.') is None
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.DELIVERED, None)
        assert alert.delivery_attempts == 2 and len(engine.calls) == 2
        np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio(engine.calls[-1]))
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.') is None
        assert len(engine.calls) == 2


async def test_alert_notice_whose_pcm_reached_the_player_is_never_repeated(monkeypatch, daylight_clock):
    """T036-U1-Q1: PCM handed to the player counts as output started (shared convention), even if the
    device then fails to start; the notice stays failed and is never spoken again."""
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    backend.fail_start = True
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'scheduler': {'enabled': False}, 'ha': {'output_zone': 'zone_origin'}}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager,
            household_clock=daylight_clock, alert_muted=lambda: False)
        manager = services.features.alert_manager
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.')
        assert alert.delivery is AlertDelivery.FAILED
        backend.fail_start = False
        for _ in range(2):
            assert await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, 'Alerta de prueba.') is None
        assert alert.delivery is AlertDelivery.FAILED and alert.delivery_attempts == 1
        assert len(engine.calls) == 1 and len(backend.outputs) == 1


async def test_voice_services_share_the_alert_preference_and_require_a_bound_mute(monkeypatch, tmp_path, daylight_clock):
    original, engine, backend, _, amplifier = voice_graph(monkeypatch)
    store = ReminderStore(str(tmp_path / 'reminders.db'))
    config = {'alerts': {'scheduler': {'enabled': False}, 'ha': {'output_zone': 'zone_origin'}}}
    now = daylight_clock.epoch()
    clock = HouseholdClock(time_source=lambda: now)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        await store.initialize()
        services = build_voice_services(config, tts=engine, shutdown=stack, zone_manager=original.zone_manager,
                                        reminder_store=store, household_clock=clock)
        manager = services.features.alert_manager
        output = manager.voice_output
        assert output.writable is True
        assert output.configured_zones == ['zone_origin', 'zone_other']
        saved = await output.set_preference(['zone_other'], expected_revision=0, guard=lambda: True)
        assert saved == await store.get_alert_zones() and saved.zones == ('zone_other',)

        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Antes de conectar el silencio.')
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.MUTED, 'alert_mute_unavailable')
        assert engine.calls == [] and backend.heard().size == 0

        muted = False
        services.features.bind_alert_mute(lambda: muted)
        now = clock.to_epoch(datetime(2026, 11, 21, 23))
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Durante el horario silencioso.')
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.MUTED, 'alert_quiet_hours')
        assert engine.calls == []

        now = daylight_clock.epoch()
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Después de conectar el silencio.')
        assert alert.delivery is AlertDelivery.DELIVERED
        assert len(engine.calls) == 1
        amplifier.select_zone.assert_called_once_with(2)
        amplifier.select_all_zones.assert_not_called()
        np.testing.assert_array_equal(backend.heard()[:, 0], engine.audio(engine.calls[0]))

        muted = True
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Silencio global.')
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.MUTED, 'alert_global_mute')
        assert len(engine.calls) == 1


async def test_missing_background_route_does_not_follow_active_zone(monkeypatch):
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'enabled': False}}, tts=engine,
            shutdown=stack, zone_manager=original.zone_manager)
        services.responses.set_active_zone('zone_other')
        with pytest.raises(VoiceDeliveryError, match='voice_output_zone_required'):
            await services.features.timer_manager._announce_warning(SimpleNamespace(name='timer', zone_id='default'), 30)
        assert engine.calls == [] and backend.outputs == []


async def test_reminder_stop_owns_task_even_before_first_turn(monkeypatch, tmp_path):
    original, engine, _, _, _ = voice_graph(monkeypatch)
    wake = asyncio.Event()
    store = ReminderStore(str(tmp_path / 'reminders.db'), on_created=wake.set)
    await store.initialize()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        services = build_voice_services({'rooms': {'enabled': True}, 'alerts': {'enabled': False}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager, reminder_store=store,
            reminder_wake_event=wake)
        assert services.reminders._wake_event is wake
        services.start_reminders()
        task = services.reminder_task
        await services.stop_reminders()
        assert task.done() and not services.reminders._running
        assert services.reminder_task is None


async def test_reminder_commit_during_empty_query_keeps_its_wake(monkeypatch, tmp_path):
    original, engine, _, _, _ = voice_graph(monkeypatch)
    wake = asyncio.Event()
    store = ReminderStore(str(tmp_path / 'wake.db'), on_created=wake.set)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        await store.initialize()
        services = build_voice_services({'rooms': {'enabled': True}, 'alerts': {'enabled': False}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager,
            reminder_store=store, reminder_wake_event=wake)
        entered, release, selected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_query = store.get_next_pending
        first = True

        async def held_query(**kwargs):
            nonlocal first
            result = await original_query(**kwargs)
            if first:
                first = False
                assert result is None
                entered.set()
                await release.wait()
            elif result is not None:
                selected.set()
            return result

        monkeypatch.setattr(store, 'get_next_pending', held_query)
        services.start_reminders()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            reminder = await store.create('synthetic-owner', 'Synthetic future reminder', time.time() + 600,
                scope=ReminderScope.SHARED)
            assert wake.is_set()
            release.set()
            await asyncio.wait_for(selected.wait(), 3)
            assert (await original_query(shared_only=True)).id == reminder.id
            assert services.reminders._wake_event is wake
        finally:
            release.set()
            await services.stop_reminders()


async def test_missing_notification_output_is_not_counted_as_delivered(monkeypatch):
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'enabled': False}}, tts=engine,
            shutdown=stack, zone_manager=original.zone_manager)
        notifications = services.features.notifications
        monkeypatch.setattr(notifications, '_get_user_context', lambda _: {
            'is_home': True, 'dnd_active': False, 'is_quiet_hours': False, 'current_zone': None,
        })
        notice = Notification('notice', '', 'Mensaje.', user_id='synthetic')
        await notifications._deliver_notification(notice)
        assert not notice.delivered and notice.channels_used == []
        assert backend.outputs == []


async def test_legacy_sync_voice_callback_does_not_block_loop_and_failed_receipt_propagates():
    from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, invoke_voice_callback
    def sync_callback():
        time.sleep(.2)
        return ResponseResult('failed', GenerationState.COMPLETE, DeliveryState.FAILED)
    task = asyncio.create_task(invoke_voice_callback(sync_callback))
    beats = 0
    while not task.done():
        await asyncio.sleep(.005)
        beats += 1
    with pytest.raises(VoiceDeliveryError):
        await task
    assert beats >= 20
