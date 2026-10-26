"""Application feature factory reaches the shared real PCM owner."""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

import numpy as np
import pytest

from src.alerts.alert_manager import AlertType, AlertPriority
from src.core.voice_delivery import VoiceDeliveryError
from src.intercom.intercom_system import Announcement
from src.notifications.smart_notifications import Notification, NotificationChannel
from src.pipeline.voice_services import build_voice_services
from src.reminders.reminder_store import ReminderStore
from src.routines.routine_executor import RoutineExecutor
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


async def test_all_feature_callbacks_share_one_pcm_output_and_correct_origin(monkeypatch, tmp_path):
    original, engine, backend, _, amplifier = voice_graph(monkeypatch)
    store = ReminderStore(str(tmp_path / 'reminders.db'))
    await store.initialize()
    ha = MagicMock()
    config = {'tts': {'streaming': {'default_output_zone': 'zone_origin'}},
              'alerts': {'scheduler': {'enabled': False}},
              'rooms': {'enabled': True}, 'reminders': {'output_zone': 'zone_origin'}}
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        services = build_voice_services(config, tts=engine, zone_manager=original.zone_manager,
            shutdown=stack, reminder_store=store, ha_client=ha)
        features = services.features
        await features.intercom._load_zones()
        await features.timer_manager._announce_warning(SimpleNamespace(name='pasta', zone_id='zone_origin'), 30)
        announcement = Announcement('announcement', 'Anuncio de prueba.', ['zone_origin'], chime_before=False)
        await features.intercom._deliver_announcement(announcement)
        assert announcement.delivered and announcement.failed_zones == []
        notice = Notification('notice', 'Aviso', 'Mensaje de prueba.', user_id='synthetic', zone_id='zone_origin')
        await features.notifications._deliver_to_channel(notice, 'synthetic', NotificationChannel.VOICE, {})
        await features.alert_manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, 'Alerta de prueba.')
        reminder = await store.create(user_id='synthetic', text='Recordatorio de prueba.', trigger_at=time.time()-1)
        await services.reminders._fire_reminder(reminder)
        assert (await store.get_by_id(reminder.id)).state == 'fired'
        executor = RoutineExecutor(tts_engine=features.voice_delivery)
        result = await executor._execute_tts_speak({'text': 'Rutina de prueba.', 'zone': 'zone_origin'}, {})
        assert result.success
        assert len(engine.calls) == 6 and engine.peak_active == 1
        expected = np.concatenate([engine.audio(text) for text in engine.calls])
        np.testing.assert_array_equal(backend.heard()[:, 0], expected)
        amplifier.select_zone.assert_called_with(1)
        amplifier.select_all_zones.assert_called_once()
        ha.call_service.assert_not_called()
    assert services.responses.get_delivery_status()['closed']


async def test_feature_delivery_failure_is_not_announced_as_completed_or_retried_via_ha(monkeypatch):
    original, engine, backend, _, _ = voice_graph(monkeypatch)
    backend.fail_start = True
    ha = MagicMock()
    async with AsyncExitStack() as stack:
        services = build_voice_services({'alerts': {'enabled': False}}, tts=engine,
            shutdown=stack, zone_manager=original.zone_manager, ha_client=ha)
        intercom = services.features.intercom
        await intercom._load_zones()
        announcement = Announcement('announcement', 'Aviso.', ['zone_origin'], chime_before=False)
        await intercom._deliver_announcement(announcement)
        assert not announcement.delivered and announcement.failed_zones == ['zone_origin']
        assert len(backend.outputs) == 1 and len(engine.calls) == 1
        ha.call_service.assert_not_called()


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
    store = ReminderStore(str(tmp_path / 'reminders.db'))
    await store.initialize()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(store.close)
        services = build_voice_services({'rooms': {'enabled': True}, 'alerts': {'enabled': False}},
            tts=engine, shutdown=stack, zone_manager=original.zone_manager, reminder_store=store)
        services.start_reminders()
        task = services.reminder_task
        await services.stop_reminders()
        assert task.done() and not services.reminders._running
        assert services.reminder_task is None


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
