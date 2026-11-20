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
from src.users.user_manager import PermissionLevel
from tests.integration.test_capability_application import configuration
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, OfflineWake
from tests.mocks.audio_output import AudioBackend
from tests.mocks.responses import delivered
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
        assert scheduler._holds == {}
        assert await store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM reminder_home_actions').fetchone()[0]) == 0


async def test_completed_shared_occurrence_cannot_be_replayed_or_advanced_again(tmp_path):
    async with AsyncExitStack() as stack:
        store = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        reminder = await store.create('household', 'Shared one-shot', time.time() - 1,
            scope=ReminderScope.SHARED)
        tts = SimpleNamespace(aspeak=AsyncMock(return_value=delivered()))
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
            users = graph.dashboard.user_manager
            person, _ = await asyncio.to_thread(users.add_user, 'Synthetic person', PermissionLevel.ADULT)
            second, _ = await asyncio.to_thread(users.add_user, 'Synthetic other', PermissionLevel.ADULT)
            private = await manager.create(person.user_id, 'zxq privado', now - 60,
                recurrence='daily', ha_actions=[action])
            other = await manager.create(second.user_id, 'pky oculto', now - 30, ha_actions=[action])
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
            assert await store._work.read(lambda db: db.execute(
                "SELECT COUNT(*) FROM reminder_home_actions WHERE request_id LIKE 'reminder:%'").fetchone()[0]) == 1
            # The shared notice has one confirmed receipt; private jobs were never claimed.
            assert await store._work.read(lambda db: [tuple(row) for row in db.execute(
                "SELECT status, reason FROM reminder_home_actions WHERE request_id LIKE 'notice:%'")]) == [
                ('confirmed', None)]


async def test_actual_factory_voice_reminder_journey_create_list_restart_fire_once(tmp_path, monkeypatch):
    """T031-U2: a reminder created by voice is a shared household row that survives a restart and fires once."""
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
        config['orchestrator']['enabled'] = True

        async def command(graph, phrase):
            stt.text = phrase
            return await graph.pipeline.process_command(CommandEvent(
                np.full(16000, .1, dtype=np.float32), room_id='living'))

        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            await graph.pipeline.request_router._orchestrator.start()
            asked = await command(graph, 'Nexa recordame regar las plantas')
            assert asked['outcome']['status'] == 'needs_input' and not asked['task_completed']
            created = await command(graph, 'Nexa recordame en 3 segundos regar las plantas')
            assert created['task_completed'] and created['outcome']['persistence'] == 'committed'
            assert created['response'].startswith('Anotado: regar las plantas, hoy a las ')
            listed = await command(graph, 'Nexa qué recordatorios tengo')
            assert 'regar las plantas' in listed['response']
            rows = await graph.reminder_manager.get_active(None, shared_only=True)
            assert [(r.user_id, r.scope, r.text, r.ha_actions, r.recurrence) for r in rows] == [
                ('household', ReminderScope.SHARED, 'regar las plantas', None, None)]
            created_id = rows[0].id
        # Restart the real factory on the same stores: the row is there and fires exactly once.
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            store = graph.reminder_manager._store
            assert (await store.get_by_id(created_id)).state is ReminderState.ACTIVE
            before = len(tts.texts)
            graph.voice_services.start_reminders()
            try:
                async with asyncio.timeout(15):
                    while (await store.get_by_id(created_id)).state is not ReminderState.FIRED:
                        await asyncio.sleep(.01)
                await asyncio.sleep(.2)
            finally:
                await graph.voice_services.stop_reminders()
            spoken = ' '.join(tts.texts[before:])
            assert spoken.count('regar las plantas') == 1
            fired = await store.get_by_id(created_id)
            assert fired.fire_count == 1 and fired.scope is ReminderScope.SHARED
            assert transport.sends == []
            notices = await store._work.read(lambda db: [tuple(row) for row in db.execute(
                "SELECT status FROM reminder_home_actions WHERE request_id LIKE 'notice:%'").fetchall()])
            assert notices == [('confirmed',)]


async def test_actual_factory_restart_after_three_days_gives_one_late_notice_and_no_home_actions(tmp_path, monkeypatch):
    """T034-U1: a restart after an outage announces each overdue shared reminder once, never its actions."""
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
    action = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.living'}
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['rooms'].update(enabled=True, living={'name': 'Living', 'mic_device_index': 0,
            'capture_channel': 1, 'ma1260_zone': 1})
        config['presence'] = {'enabled': False}
        config['tts']['streaming']['default_output_zone'] = 'living'
        now = time.time()
        before_outage = ReminderStore(config['reminders']['db_path'])
        await before_outage.initialize()
        one_shot = await before_outage.create('household', 'Sacar la basura.', now - 3 * 86400,
            ha_actions=[action], scope=ReminderScope.SHARED)
        daily = await before_outage.create('household', 'Cerrar las persianas.', now - 3 * 86400 - 600,
            recurrence='daily', ha_actions=[action], scope=ReminderScope.SHARED)
        await before_outage.close()
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=factories)
            store = graph.reminder_manager._store
            before = list(tts.texts)
            graph.voice_services.start_reminders()
            try:
                async with asyncio.timeout(5):
                    while ((await store.get_by_id(one_shot.id)).state is not ReminderState.FIRED
                           or (await store.get_by_id(daily.id)).trigger_at < time.time()):
                        await asyncio.sleep(.01)
                await asyncio.sleep(.1)
                spoken = ' '.join(tts.texts[len(before):])
                on_time = await graph.reminder_manager.create('household', 'Regar ahora.', time.time() - 1,
                    ha_actions=[action], scope=ReminderScope.SHARED)
                async with asyncio.timeout(3):
                    while (await store.get_by_id(on_time.id)).state is not ReminderState.FIRED:
                        await asyncio.sleep(.01)
                await asyncio.sleep(.1)
            finally:
                await graph.voice_services.stop_reminders()
            for text in ('Sacar la basura', 'Cerrar las persianas'):
                assert spoken.count(text) == 1
            assert spoken.count('Este aviso llega con atraso y no ejecuté sus acciones programadas') == 2
            assert ' '.join(tts.texts).count('Regar ahora') == 1
            assert len(transport.sends) == 1  # only the on-time reminder's action
            advanced = await store.get_by_id(daily.id)
            assert advanced.state is ReminderState.ACTIVE and advanced.trigger_at > time.time()
            statuses = await store._work.read(lambda db: sorted(tuple(row) for row in db.execute(
                "SELECT status, reason FROM reminder_home_actions WHERE request_id LIKE 'reminder:%'").fetchall()))
            assert statuses == [('confirmed', None), ('failed', 'late_not_executed'), ('failed', 'late_not_executed')]
            notices = await store._work.read(lambda db: [tuple(row) for row in db.execute(
                "SELECT status, reason FROM reminder_home_actions WHERE request_id LIKE 'notice:%'").fetchall()])
            assert notices == [('confirmed', None)] * 3


# ---------------------------------------------------------------------------
# T034-U2: a living barge-in during a shared reminder announcement (real ResponseHandler)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402
import json  # noqa: E402

from src.core.voice_delivery import LocalVoiceDelivery  # noqa: E402
from src.pipeline.response_handler import ResponseHandler  # noqa: E402
from src.reminders.reminder_scheduler import DEFAULT_LATE_ACTION_GRACE_S  # noqa: E402
from tests.mocks.audio_output import GatedZones  # noqa: E402
from tests.mocks.home_control import selected_actions  # noqa: E402

LIVING = 'zone_living'
ACTION = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.living'}


@asynccontextmanager
async def living_reminders(path, zones, monkeypatch, *, clock=time.time):
    """Shared reminders announced on reminders.output_zone 'zone_living' through the real voice owner."""
    import src.tts.piper_tts as output_module

    monkeypatch.setattr(output_module, 'sd', AudioBackend())
    engine = MagicMock()
    engine.synthesize.return_value = (np.ones(1600, dtype=np.float32), 0.1)
    engine.sample_rate = 16000
    del engine.synthesize_stream
    handler = ResponseHandler(tts=engine, zone_manager=zones, streaming_enabled=False)
    ha = AsyncMock()
    async with AsyncExitStack() as stack:
        stack.push_async_callback(handler.aclose)
        store = ReminderStore(str(path))
        stack.push_async_callback(store.close)
        await store.initialize()
        scheduler = ReminderScheduler(store, LocalVoiceDelivery(handler), None, ha,
            {'output_zone': LIVING, 'notice_retry_min_gap_s': .2},
            home_actions=selected_actions(ha, ('light.living',)), clock=clock)
        stack.push_async_callback(scheduler.stop)
        yield SimpleNamespace(handler=handler, store=store, scheduler=scheduler, ha=ha, engine=engine)


async def ledger_rows(store, prefix):
    rows = await store._work.read(lambda db: [tuple(row) for row in db.execute(
        'SELECT status, reason, body FROM reminder_home_actions WHERE request_id LIKE ? ORDER BY rowid',
        (prefix + '%',)).fetchall()])
    return [(status, reason, json.loads(body).get('attempt')) for status, reason, body in rows]


@pytest.mark.parametrize('restart_after', [1, DEFAULT_LATE_ACTION_GRACE_S + 5])
async def test_living_barge_in_after_output_started_ends_the_notice_uncertain_and_it_is_never_repeated(
    tmp_path, monkeypatch, restart_after,
):
    """C02-NTC-0010: barge-in within the first sentence (spoken_segments 0, output_started) -> cortado, never re-spoken."""
    path, now = tmp_path / 'reminders.db', [time.time()]
    zones = GatedZones(block='after_first_chunk', gated={LIVING})
    async with living_reminders(path, zones, monkeypatch, clock=lambda: now[0]) as graph:
        reminder = await graph.store.create('household', 'Regar las plantas.', now[0] - 1,
            ha_actions=[ACTION], scope=ReminderScope.SHARED)
        task = asyncio.create_task(graph.scheduler._fire_reminder(reminder))
        assert await asyncio.to_thread(zones.waiting.wait, 2)
        # multi_room_audio_loop._trigger_barge_in for the living room
        assert await graph.handler.cancel(zone_id=LIVING)
        assert await asyncio.wait_for(task, 2) is False
        assert await ledger_rows(graph.store, 'notice:') == [('uncertain', 'delivery_cancelled', 1)]
        now[0] += 1
        for _ in range(3):  # nothing more in-process, also after the retry gap
            await graph.scheduler._deliver_due(now[0])
            assert await graph.scheduler._fire_reminder(reminder) is False
            now[0] += 1
        assert zones.plays == [LIVING]
        assert (await graph.store.get_by_id(reminder.id)).state is ReminderState.FIRED
    now[0] += restart_after
    zones.release.set()
    async with living_reminders(path, zones, monkeypatch, clock=lambda: now[0]) as graph:
        task = asyncio.create_task(graph.scheduler.start())
        await asyncio.sleep(.2)
        await graph.scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
        assert zones.plays == [LIVING]  # zero further TTS deliveries after the restart
        graph.ha.call_service_once.assert_not_called()
        assert await ledger_rows(graph.store, 'reminder:') == []  # actions never claimed
        [entry] = await graph.scheduler.notice_entries()
        assert (entry.status, entry.reason, entry.pending) == ('uncertain', 'delivery_cancelled', False)


async def test_living_barge_in_while_the_notice_waits_for_the_output_retries_it(tmp_path, monkeypatch):
    """C02-NTC-0011: cancelled before any PCM -> failed/not started, still pending, retried after the gap."""
    zones, now = GatedZones(block='after_first_chunk', gated={'zone_hall'}), [time.time()]
    async with living_reminders(tmp_path / 'reminders.db', zones, monkeypatch, clock=lambda: now[0]) as graph:
        busy = asyncio.create_task(graph.handler.aspeak('Respuesta en el hall.', zone_id='zone_hall'))
        assert await asyncio.to_thread(zones.waiting.wait, 2)
        reminder = await graph.store.create('household', 'Regar las plantas.', now[0] - 1,
            ha_actions=[ACTION], scope=ReminderScope.SHARED)
        task = asyncio.create_task(graph.scheduler.start())
        async with asyncio.timeout(2):
            while graph.handler.get_delivery_status()['pending'] == 0:
                await asyncio.sleep(.005)
        assert await graph.handler.cancel(zone_id=LIVING)
        async with asyncio.timeout(2):
            while not await ledger_rows(graph.store, 'notice:') or (
                    await ledger_rows(graph.store, 'notice:'))[0][0] == 'sending':
                await asyncio.sleep(.005)
        assert await ledger_rows(graph.store, 'notice:') == [('failed', 'delivery_not_started:cancelled', 1)]
        [entry] = await graph.scheduler.notice_entries()
        assert (entry.status, entry.pending, entry.failed_attempts) == ('pending', True, 1)  # not shown as delivered
        assert (await graph.store.get_by_id(reminder.id)).state is ReminderState.ACTIVE
        zones.release.set()
        assert (await asyncio.wait_for(busy, 2)).success
        now[0] += 1  # the retry gap (0.2 s on the scheduler's clock) has passed
        graph.scheduler.notify_new_reminder()
        try:
            async with asyncio.timeout(3):
                while (await graph.store.get_by_id(reminder.id)).state is not ReminderState.FIRED:
                    await asyncio.sleep(.01)
        finally:
            await graph.scheduler.stop()
            await asyncio.gather(task, return_exceptions=True)
        assert zones.plays == ['zone_hall', LIVING]
        assert await ledger_rows(graph.store, 'notice:') == [
            ('failed', 'delivery_not_started:cancelled', 1), ('confirmed', None, 2)]
        graph.ha.call_service_once.assert_awaited_once()  # on time (within late_action_grace_s)
        assert await ledger_rows(graph.store, 'reminder:') == [('confirmed', None, None)]


# ---------------------------------------------------------------------------
# T034-U2: C02-NTC seeds assigned to this unit (p3/design.json seed_entries), scheduler layer
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402

from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, ResponseStage  # noqa: E402

C02_COUNTERS = ('ha_service_calls', 'vector_searches', 'music_calls', 'list_mutations', 'reminder_rows_delta',
                'reminder_transitions', 'timer_delta', 'timer_transitions', 'queue_cancels', 'routine_runs',
                'routine_changes', 'llm_8101_calls', 'cloud_calls', 'ledger_claims', 'spoken_utterances')


@dataclass(frozen=True)
class C02NoticeSeed:
    """One C02-NTC entry (P3-F0/c02-matrix/v1 fields used by the scheduler layer)."""

    id: str
    lifecycle: str
    task_status: str
    notice: tuple[str, str | None]
    effects: dict = field(default_factory=dict)

    def expected(self) -> dict:
        return {name: self.effects.get(name, 0) for name in C02_COUNTERS}


C02_NTC_SEEDS = (
    C02NoticeSeed('C02-NTC-0003', 'crash_during_send', 'uncertain', ('uncertain', 'interrupted_send'),
                  {'reminder_transitions': 1}),
    C02NoticeSeed('C02-NTC-0007', 'partial', 'uncertain', ('uncertain', 'delivery_partial'),
                  {'reminder_transitions': 1, 'ledger_claims': 1, 'spoken_utterances': 1}),
    C02NoticeSeed('C02-NTC-0009', 'delivery_timeout', 'uncertain', ('uncertain', 'delivery_partial'),
                  {'reminder_transitions': 1, 'ledger_claims': 1, 'spoken_utterances': 1}),
    C02NoticeSeed('C02-NTC-0010', 'barge_in_during_notice', 'uncertain', ('uncertain', 'delivery_cancelled'),
                  {'reminder_transitions': 1, 'ledger_claims': 1, 'spoken_utterances': 1}),
    C02NoticeSeed('C02-NTC-0011', 'barge_in_while_queued', 'completed', ('confirmed', None),
                  {'ha_service_calls': 1, 'reminder_transitions': 1, 'ledger_claims': 3, 'spoken_utterances': 1}),
)


async def ledger_count(store):
    return await store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM reminder_home_actions').fetchone()[0])


async def reminder_rows(store):
    return await store._work.read(lambda db: [tuple(row) for row in db.execute(
        'SELECT id, state, trigger_at FROM reminders ORDER BY id').fetchall()])


def started_deliveries(zones):
    return sum(zone == LIVING for zone in zones.plays)


@pytest.mark.parametrize('seed', C02_NTC_SEEDS, ids=[seed.id for seed in C02_NTC_SEEDS])
async def test_c02_notice_seed(tmp_path, monkeypatch, seed):
    """Each seed's counters after its lifecycle point; the other C02 counters have no producer in this graph."""
    path, now = tmp_path / 'reminders.db', [time.time()]
    clock = lambda: now[0]  # noqa: E731
    gated = {LIVING} if seed.lifecycle in {'crash_during_send', 'barge_in_during_notice'} else {'zone_hall'}
    zones = GatedZones(block='after_first_chunk', gated=gated)
    effects = dict.fromkeys(C02_COUNTERS, 0)
    async with living_reminders(path, zones, monkeypatch, clock=clock) as graph:
        reminder = await graph.store.create('household', 'Regar las plantas.', now[0] - 1,
            ha_actions=[ACTION], scope=ReminderScope.SHARED)
        rows, claims, plays = await reminder_rows(graph.store), await ledger_count(graph.store), 0
        if seed.lifecycle in {'partial', 'delivery_timeout'}:
            result = (ResponseResult('x', GenerationState.COMPLETE, DeliveryState.FAILED, spoken_segments=1,
                                     output_started=True) if seed.lifecycle == 'partial' else
                      ResponseResult('x', GenerationState.COMPLETE, DeliveryState.TIMED_OUT,
                                     'response_delivery_timeout', timeout_stage=ResponseStage.DELIVERY))
            graph.scheduler._tts = SimpleNamespace(aspeak=AsyncMock(return_value=result))
            await graph.scheduler._deliver_due(now[0])
            plays = graph.scheduler._tts.aspeak.await_count
        elif seed.lifecycle in {'crash_during_send', 'barge_in_during_notice'}:
            task = asyncio.create_task(graph.scheduler._fire_reminder(reminder))
            assert await asyncio.to_thread(zones.waiting.wait, 2)
            if seed.lifecycle == 'crash_during_send':
                task.cancel()  # the process dies while the notice plays: counters start after the restart
                await asyncio.gather(task, return_exceptions=True)
            else:
                assert await graph.handler.cancel(zone_id=LIVING)
                assert await asyncio.wait_for(task, 2) is False
        else:  # barge_in_while_queued
            busy = asyncio.create_task(graph.handler.aspeak('Respuesta en el hall.', zone_id='zone_hall'))
            assert await asyncio.to_thread(zones.waiting.wait, 2)
            task = asyncio.create_task(graph.scheduler._fire_reminder(reminder))
            async with asyncio.timeout(2):
                while graph.handler.get_delivery_status()['pending'] == 0:
                    await asyncio.sleep(.005)
            assert await graph.handler.cancel(zone_id=LIVING)
            assert await asyncio.wait_for(task, 2) is False
            zones.release.set()
            await asyncio.wait_for(busy, 2)
            now[0] += 1  # notice_retry_min_gap_s (0.2 s) later, still within late_action_grace_s
            assert await graph.scheduler._fire_reminder(reminder) is True
        if seed.lifecycle != 'crash_during_send':
            effects['ledger_claims'] = await ledger_count(graph.store) - claims
            effects['spoken_utterances'] = plays or started_deliveries(zones)
            effects['ha_service_calls'] = graph.ha.call_service_once.await_count
            effects['reminder_transitions'] = int(await reminder_rows(graph.store) != rows)
            effects['reminder_rows_delta'] = len(await reminder_rows(graph.store)) - len(rows)
        zones.release.set()
    # Restart (all lifecycles): nothing more is spoken, claimed or executed for the occurrence.
    now[0] += 5
    after = GatedZones()
    async with living_reminders(path, after, monkeypatch, clock=clock) as graph:
        rows, claims = await reminder_rows(graph.store), await ledger_count(graph.store)
        await graph.scheduler._deliver_due(now[0])
        restart_effects = {'ledger_claims': await ledger_count(graph.store) - claims,
                           'spoken_utterances': started_deliveries(after),
                           'ha_service_calls': graph.ha.call_service_once.await_count,
                           'reminder_transitions': int(await reminder_rows(graph.store) != rows)}
        notices = await ledger_rows(graph.store, 'notice:')
        state = (await graph.store.get_by_id(reminder.id)).state
    if seed.lifecycle == 'crash_during_send':
        effects.update(restart_effects)
        assert notices == [('uncertain', 'interrupted_send', 1)]
    else:
        assert restart_effects == {'ledger_claims': 0, 'spoken_utterances': 0, 'ha_service_calls': 0,
                                   'reminder_transitions': 0}
        assert notices[-1][:2] == seed.notice
    assert effects == seed.expected()
    task_status = 'completed' if notices[-1][0] == 'confirmed' else notices[-1][0]
    assert (task_status, state) == (seed.task_status, ReminderState.FIRED)  # persistence: committed
