"""Legacy/private jobs cannot become household announcements through identity."""

import asyncio
from contextlib import AsyncExitStack, closing
from dataclasses import replace
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.application import ApplicationFactories
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from tests.integration.test_capability_application import configuration
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport


async def test_legacy_and_new_implicit_reminders_stay_private_after_migration_and_restart(tmp_path):
    path = tmp_path / 'reminders.db'
    with closing(sqlite3.connect(path)) as database:
        database.execute("""CREATE TABLE reminders (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
            trigger_at REAL NOT NULL, created_at REAL NOT NULL, recurrence TEXT,
            recurrence_end REAL, ha_actions TEXT, state TEXT NOT NULL DEFAULT 'active',
            last_fired_at REAL, fire_count INTEGER NOT NULL DEFAULT 0)""")
        database.executemany('INSERT INTO reminders (id,user_id,text,trigger_at,created_at) VALUES (?,?,?,?,?)',
            [('old-unknown', 'unknown', 'PRIVATE_UNKNOWN_371', 1., 0.),
             ('old-known', 'known-profile', 'PRIVATE_KNOWN_813', 2., 0.)])
        database.commit()
    async with AsyncExitStack() as stack:
        store = ReminderStore(str(path))
        stack.push_async_callback(store.close)
        await store.initialize()
        for identity, owner, text in [('old-unknown', 'unknown', 'PRIVATE_UNKNOWN_371'),
                                       ('old-known', 'known-profile', 'PRIVATE_KNOWN_813')]:
            row = await store.get_by_id(identity)
            assert row.scope is ReminderScope.PRIVATE and row.user_id == owner and row.text == text
            assert row.state is ReminderState.ACTIVE and row.fire_count == 0
        implicit = await store.create('unknown', 'PRIVATE_NEW_291', 3.)
        assert implicit.scope is ReminderScope.PRIVATE
        assert await store.get_next_pending(shared_only=True) is None
        assert await store.get_due(time.time(), shared_only=True) == []
        assert await store.count_active(shared_only=True) == 0
        with pytest.raises(ValueError):
            await store.create('unknown', 'invalid', 4., scope='public')
        await store.close()
        await store.initialize()
        assert await store.count_active() == 3
        for identity in ['old-unknown', 'old-known', implicit.id]:
            assert (await store.get_by_id(identity)).scope is ReminderScope.PRIVATE


@pytest.mark.parametrize('user_id', ['unknown', 'known-profile'])
async def test_shared_sql_filters_before_reminder_projection_fuzzy_and_mutation(tmp_path, monkeypatch, user_id):
    async with AsyncExitStack() as stack:
        store = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        private = await store.create(user_id, 'PRIVATE_XQZ_915', 1.)
        shared = await store.create('different-profile', 'Comprar tornillos', 2., scope=ReminderScope.SHARED)
        manager = ReminderManager(store)
        project = store._row_to_reminder
        selected = []

        def convert(row):
            assert row['id'] != private.id
            selected.append(row['id'])
            return project(row)

        with monkeypatch.context() as patch:
            patch.setattr(store, '_row_to_reminder', convert)
            assert await store.get_by_id(private.id, shared_only=True) is None
            assert (await store.get_next_pending(shared_only=True)).id == shared.id
            assert [r.id for r in await store.get_due(time.time(), shared_only=True)] == [shared.id]
            assert [r.id for r in await store.get_overdue_for_user(user_id, time.time(), shared_only=True)] == [shared.id]
            assert [r.id for r in await manager.get_active(user_id, shared_only=True)] == [shared.id]
            assert [r.id for r in await manager.get_today(user_id, shared_only=True)] == [shared.id]
            assert await manager.cancel_by_text(user_id, 'PRIVATE_XQZ_915', shared_only=True) is False
            assert await manager.cancel_by_id(private.id, shared_only=True) is False
            await store.mark_fired(private.id, shared_only=True)
            await store.update_trigger(private.id, 1000., shared_only=True)
        assert selected and set(selected) == {shared.id}
        current = await store.get_by_id(private.id)
        assert current.state is ReminderState.ACTIVE and current.trigger_at == 1. and current.fire_count == 0
        with pytest.raises(PermissionError, match='personal_reminder_voice_session_required'):
            manager.format_for_voice(current)
        assert 'Comprar tornillos' in manager.format_for_voice(shared)
        assert await manager.cancel_by_text(user_id, 'Comprar tornillos', shared_only=True) is True
        assert (await store.get_by_id(shared.id)).state is ReminderState.CANCELLED


async def test_private_id_cannot_be_announced_or_executed_by_a_forged_shared_dto(tmp_path):
    async with AsyncExitStack() as stack:
        store = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        private = await store.create('unknown', 'PRIVATE_ANNOUNCEMENT_629', 1., recurrence='daily',
            ha_actions=[{'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.synthetic'}])
        tts = SimpleNamespace(aspeak=AsyncMock())
        presence = MagicMock()
        actions = SimpleNamespace(execute_service=AsyncMock())
        scheduler = ReminderScheduler(store, tts, presence, None, home_actions=actions)
        for candidate in [private, replace(private, scope=ReminderScope.SHARED)]:
            await scheduler._fire_reminder(candidate)
            await scheduler._execute_home_actions(candidate)
        assert await scheduler.missed_reminder_on_arrival('unknown', 'living') == 0
        assert await scheduler.missed_reminder_on_arrival('known-profile', 'living') == 0
        tts.aspeak.assert_not_awaited()
        actions.execute_service.assert_not_awaited()
        presence.get_user_zone.assert_not_called()
        row = await store.get_by_id(private.id)
        assert row.state is ReminderState.ACTIVE and row.trigger_at == 1. and row.fire_count == 0
        assert scheduler._retry_counts == {}
        assert await store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM reminder_home_actions').fetchone()[0]) == 0


async def test_completed_shared_occurrence_cannot_be_replayed_or_advanced_again(tmp_path):
    async with AsyncExitStack() as stack:
        store = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        reminder = await store.create('household', 'Shared one-shot', time.time() - 1,
            scope=ReminderScope.SHARED)
        tts = SimpleNamespace(aspeak=AsyncMock())
        scheduler = ReminderScheduler(store, tts, None, None, config={'output_zone': 'living'})
        await scheduler._fire_reminder(reminder)
        assert (await store.get_by_id(reminder.id)).fire_count == 1
        await scheduler._fire_reminder(reminder)
        await scheduler._execute_home_actions(reminder)
        assert await scheduler.missed_reminder_on_arrival('any-profile', 'living') == 0
        assert not await store.update_trigger(reminder.id, time.time(), shared_only=True,
            expected_trigger_at=reminder.trigger_at)
        assert not await store.mark_fired(reminder.id, shared_only=True,
            expected_trigger_at=reminder.trigger_at)
        current = await store.get_by_id(reminder.id)
        assert current.state is ReminderState.FIRED and current.fire_count == 1
        tts.aspeak.assert_awaited_once()


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_actual_factory_private_jobs_stay_silent_and_shared_jobs_deliver_once(tmp_path, monkeypatch, orchestrated):
    import src.tts.piper_tts as output_module
    import src.vectordb.chroma_sync as index_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, 'SentenceTransformer', lambda *a, **kw: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(output_module, 'sd', backend)
    stt, tts, amp = OfflineSTT(), OfflineTTS(), MagicMock()
    amp.audio_output_device = 7
    factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None,
        wake_detector=OfflineWake, amplifier=lambda **kw: amp)
    transport = TaskTransport()
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['rooms'].update(enabled=True, living={'name': 'Living', 'mic_device_index': 0,
            'capture_channel': 1, 'ma1260_zone': 1})
        config['presence'] = {'enabled': False}
        config['tts']['streaming']['default_output_zone'] = 'living'
        config['reminders']['output_zone'] = 'living'
        config['orchestrator']['enabled'] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            manager = graph.reminder_manager
            store, scheduler = manager._store, graph.voice_services.reminders
            assert scheduler is graph.dashboard.reminder_scheduler
            action = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.living'}
            now = time.time()
            private = await manager.create('unknown', 'zxq privado', now - 60,
                recurrence='daily', ha_actions=[action])
            other = await manager.create('known-profile', 'pky oculto', now - 30, ha_actions=[action])
            before = list(tts.texts)
            await scheduler._fire_reminder(private)
            await scheduler._execute_home_actions(replace(other, scope=ReminderScope.SHARED))
            assert await scheduler.missed_reminder_on_arrival('unknown', 'living') == 0
            assert tts.texts == before and transport.sends == []
            future = await manager.create('different-profile', 'comprar tornillos', now + 3600,
                scope=ReminderScope.SHARED)
            due = await manager.create('different-profile', 'Anuncio del hogar.', now - 1,
                scope=ReminderScope.SHARED, ha_actions=[action])
            if orchestrated:
                await graph.pipeline.request_router._orchestrator.start()

            async def command(phrase):
                stt.text = phrase
                return await graph.pipeline.process_command(CommandEvent(
                    np.full(16000, .1, dtype=np.float32), room_id='living'))

            listed = await command('Nexa qué recordatorios tengo')
            if orchestrated:
                assert listed['success'] and 'comprar tornillos' in listed['response']
            assert not (await command('Nexa cancela el recordatorio de zxq privado'))['success']
            cancelled = await command('Nexa cancela el recordatorio de comprar tornillos')
            assert cancelled['success'] is orchestrated
            assert (await store.get_by_id(future.id)).state is (
                ReminderState.CANCELLED if orchestrated else ReminderState.ACTIVE)
            before_announcements = len(tts.texts)
            reads = AsyncMock(wraps=store.get_next_pending)
            monkeypatch.setattr(store, 'get_next_pending', reads)
            graph.voice_services.start_reminders()
            task = graph.voice_services.reminder_task
            try:
                async with asyncio.timeout(3):
                    while (await store.get_by_id(due.id)).state is not ReminderState.FIRED:
                        await asyncio.sleep(.001)
                await asyncio.sleep(.05)
                count = reads.await_count
                await asyncio.sleep(.03)
                assert reads.await_count == count and count <= 3
            finally:
                await graph.voice_services.stop_reminders()
            assert task.done()
            assert 'Anuncio del hogar.' in ' '.join(tts.texts[before_announcements:])
            assert len(backend.heard()) > 0 and len(transport.sends) == 1
            amp.select_zone.assert_called_with(1)
            assert all(marker not in ' '.join(tts.texts) for marker in ['zxq privado', 'pky oculto'])
            for identity, trigger in [(private.id, now - 60), (other.id, now - 30)]:
                row = await store.get_by_id(identity)
                assert row.scope is ReminderScope.PRIVATE and row.state is ReminderState.ACTIVE
                assert row.trigger_at == trigger and row.fire_count == 0 and row.last_fired_at is None
            assert await store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM reminder_home_actions').fetchone()[0]) == 1
