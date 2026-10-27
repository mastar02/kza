"""Real event SQL projections, raw deadlines and tracked late native writes."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import sqlite3
import threading
import time

import httpx
import pytest

from src.analytics.asr_quality import alog_asr_outcome
from src.analytics.event_logger import EventLogger
from src.analytics.services import build_event_logger
from src.core.owned_work import WorkCloseTimeout
from src.dashboard.api import DashboardAPI
from src.dashboard.live_event_bus import LiveEventBus
from src.dashboard.live_receipts import LiveReceiptStore
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(tmp_path, **options):
    clock = Clock()
    policy = FactLedger(str(tmp_path/'privacy.db'), clock=clock)
    await asyncio.to_thread(policy.initialize)
    events = EventLogger(str(tmp_path/'events.db'), source_policy=policy, clock=clock, **options)
    try:
        # Match build_event_logger: startup I/O finishes before bounded hot-path
        # logging is admitted. The 100 ms observation budget stays unchanged.
        await events.acall(events.initialize)
        yield events, policy, clock
    finally:
        await events.aclose()
        await asyncio.to_thread(policy.close)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


@asynccontextmanager
async def conversation_client(events, policy, clock):
    bus = LiveEventBus(source_policy=policy,clock=clock,receipts=LiveReceiptStore(policy))
    bus.start()
    try:
        async with ready_browser(bus) as peer:
            app = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={'X-KZA-Live-Connection':peer.connection}) as client:
                yield client
    finally:
        await bus.aclose()


async def test_event_text_expires_at_capture_boundary_while_numeric_diagnostics_remain(tmp_path):
    async with graph(tmp_path) as (events, _, clock):
        source = memory_source(now=clock.now, ttl=60)
        written = await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto privado.',
            user_name='Nombre privado.', extra_context={'transcript_copy': 'Otra copia privada.', 'rms': .3}, sources=(source,))
        clock.now += 59
        assert (await events.acall(events.get_events))[0].trigger_phrase == 'Texto privado.'
        clock.now += 1
        result = (await events.acall(events.get_events))[0]
        assert result.trigger_phrase is None and result.user_name is None
        assert result.context == {'rms': .3} and result.expires_at == written.expires_at
        with sqlite3.connect(events.db_path) as db:
            # This logger has no cleanup task: read-time filtering itself is required.
            assert db.execute('SELECT trigger_phrase FROM events_policy_v1').fetchone()[0] == 'Texto privado.'
        await events.acall(events.cleanup_old_events)
        with sqlite3.connect(events.db_path) as db:
            row = db.execute('SELECT trigger_phrase, user_name, context_json FROM events_policy_v1').fetchone()
            assert row[:2] == (None, None) and json.loads(row[2]) == {'rms': .3}
        assert events.db_path.stat().st_mode & 0o777 == 0o600


async def test_reading_and_restart_never_renew_raw_or_metadata_deadlines(tmp_path):
    async with graph(tmp_path, retention_days=3) as (events, policy, clock):
        source = memory_source(now=clock.now)
        written = await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto privado.', sources=(source,))
        clock.now += 48*3600
        assert (await events.acall(events.get_entity_history, 'light.synthetic'))[0].trigger_phrase is None
        await events.aclose()
        reopened = EventLogger(str(events.db_path), retention_days=30, source_policy=policy, clock=clock)
        try:
            item = (await reopened.acall(reopened.get_events))[0]
            assert item.trigger_phrase is None and item.expires_at == written.expires_at
            clock.now += 86400
            assert await reopened.acall(reopened.get_events) == []
            assert (await reopened.acall(reopened.get_stats))['total_events'] == 0
            assert sum((await reopened.acall(reopened.get_hourly_distribution, 'light.synthetic')).values()) == 0
            assert sum((await reopened.acall(reopened.get_weekday_distribution, 'light.synthetic')).values()) == 0
            assert await reopened.acall(reopened.get_sequences, min_occurrences=1) == []
            assert await reopened.acall(reopened.cleanup_old_events) == 1
        finally:
            await reopened.aclose()


async def test_legacy_event_rows_are_unchanged_and_excluded_from_every_active_query(tmp_path):
    path = tmp_path/'events.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, private_text TEXT)')
        db.execute('INSERT INTO events VALUES (1, ?)', ('Texto legacy sin fuentes.',))
    async with graph(tmp_path) as (events, _, _):
        assert await events.acall(events.get_events) == []
        assert (await events.acall(events.get_stats))['legacy_policy'] == 'quarantined'
        assert await events.acall(events.get_sequences, min_occurrences=1) == []
        assert sum((await events.acall(events.get_hourly_distribution, 'light.synthetic')).values()) == 0
        assert sum((await events.acall(events.get_weekday_distribution, 'light.synthetic')).values()) == 0
        await events.acall(events.cleanup_old_events)
        with sqlite3.connect(path) as db:
            assert db.execute('SELECT * FROM events').fetchall() == [(1, 'Texto legacy sin fuentes.')]


async def test_unsourced_metrics_cannot_retain_raw_text_through_any_context_field(tmp_path):
    async with graph(tmp_path) as (events, _, _):
        item = await events.alog('asr_quality:synthetic', 'accepted:ok', trigger_phrase='Texto privado.',
            user_name='Nombre privado.', extra_context={'rms': .2, 'transcript': 'Otra copia.', 'wake_score': 'Texto oculto.'})
        assert item.trigger_phrase is None and item.user_name is None and item.context == {'rms': .2}
        assert (await events.acall(events.get_events))[0].context == {'rms': .2}


async def test_revocation_filters_all_event_analytics_before_physical_cleanup(tmp_path):
    async with graph(tmp_path) as (events, policy, clock):
        first = memory_source(now=clock.now)
        await events.alog('light.first', 'turn_on', trigger_phrase='Primera.', sources=(first,))
        clock.now += 1
        second = memory_source(now=clock.now)
        await events.alog('light.second', 'turn_on', trigger_phrase='Segunda.', sources=(second,))
        assert len(await events.acall(events.get_events)) == 2
        assert await events.acall(events.get_sequences, min_occurrences=1)
        await events.acall(policy.delete_conversation, first.conversation_id, MemoryReviewer('test-admin'))
        assert [row.entity_id for row in await events.acall(events.get_events)] == ['light.second']
        assert (await events.acall(events.get_stats))['total_events'] == 1
        assert sum((await events.acall(events.get_hourly_distribution, 'light.first')).values()) == 0
        assert sum((await events.acall(events.get_weekday_distribution, 'light.first')).values()) == 0
        assert await events.acall(events.get_sequences, min_occurrences=1) == []
        assert await events.acall(events.cleanup_old_events) == 1


async def test_runtime_factory_initializes_before_first_bounded_event(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    initialize = EventLogger._init_db
    instances = []
    def slow_initialize(self):
        instances.append(self)
        entered.set()
        release.wait(3)
        initialize(self)
    monkeypatch.setattr(EventLogger, '_init_db', slow_initialize)
    async with AsyncExitStack() as stack:
        starting = asyncio.create_task(build_event_logger({'analytics':{'events_db':str(tmp_path/'events.db')}},shutdown=stack))
        try:
            await until(entered.is_set)
            await asyncio.sleep(.15)  # Deliberately beyond the unchanged 100 ms request budget.
            assert not starting.done() and not instances[0]._initialized
        finally:
            release.set()
        events = await starting
        assert events._initialized and events.observation_timeout_s == .1
        result = await events.alog('light.synthetic','turn_on',sources=(memory_source(),))
        assert result.entity_id == 'light.synthetic'


async def test_lazy_cold_event_timeout_retains_actual_initialization_and_single_write(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    initialize = EventLogger._init_db
    def slow_initialize(self):
        entered.set()
        release.wait(3)
        initialize(self)
    monkeypatch.setattr(EventLogger, '_init_db', slow_initialize)
    events = EventLogger(str(tmp_path/'events.db'))
    writing = asyncio.create_task(events.alog('light.synthetic','turn_on',sources=(memory_source(),)))
    try:
        await until(entered.is_set)
        with pytest.raises(TimeoutError):
            await writing
        assert events.observation_timeout_s == .1 and events.resource_status()['operations'] == 1
    finally:
        release.set()
        await until(lambda: events.resource_status()['operations'] == 0)
    try:
        assert len(await events.acall(events.get_events)) == 1
    finally:
        await events.aclose()


@pytest.mark.parametrize('action', ['expire', 'revoke'])
async def test_timed_out_native_metric_cannot_commit_expired_or_revoked_text(tmp_path, action):
    entered, release = threading.Event(), threading.Event()

    def context():
        entered.set()
        assert release.wait(3)
        return {'private': 'Texto que no debe reaparecer.'}

    async with graph(tmp_path, context_provider=context, observation_timeout_s=.01) as (events, policy, clock):
        source = memory_source(now=clock.now, ttl=60)
        try:
            await alog_asr_outcome(events, 'synthetic', 'accepted', 'ok', 'Texto privado.', {}, .8, .1, sources=(source,))
            assert entered.is_set() and events.resource_status()['operations'] == 1
            if action == 'expire':
                clock.now += 60
            else:
                await asyncio.to_thread(policy.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            release.set()
            await until(lambda: events.resource_status()['operations'] == 0)
            assert await events.acall(events.get_events) == []
        finally:
            release.set()


async def test_slow_metric_does_not_block_loop_and_native_scope_remains_bounded(tmp_path):
    entered, release = [], threading.Event()

    def context():
        entered.append(True)
        assert release.wait(3)
        return {}

    async with graph(tmp_path, context_provider=context, observation_timeout_s=.02) as (events, _, clock):
        source = memory_source(now=clock.now)
        calls = [asyncio.create_task(events.alog('light.synthetic', 'turn_on', sources=(source,))) for _ in range(8)]
        try:
            await until(lambda: len(entered) == 2)
            results = await asyncio.gather(*calls, return_exceptions=True)
            assert all(isinstance(result, TimeoutError) for result in results)
            assert events.resource_status()['operations'] == 8
            with pytest.raises(RuntimeError, match='capacity'):
                await events.alog('light.other', 'turn_on', sources=(source,))
            with pytest.raises(WorkCloseTimeout):
                await events.aclose()
            assert events.resource_status()['operations'] == 8
        finally:
            release.set()
            await events.aclose()
        assert events.resource_status()['operations'] == 0


async def test_factory_starts_owned_deadline_cleanup_and_removes_expired_raw_bytes(tmp_path):
    async with AsyncExitStack() as stack:
        events = await build_event_logger({'analytics': {'events_db': str(tmp_path/'events.db')}}, shutdown=stack)
        source = memory_source(now=time.time(), ttl=.2)
        await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto efímero.', sources=(source,))

        def scrubbed():
            with sqlite3.connect(events.db_path) as db:
                return db.execute('SELECT trigger_phrase FROM events_policy_v1').fetchone()[0] is None

        await until(scrubbed)
        assert (await events.acall(events.get_events))[0].trigger_phrase is None
    assert events._maintenance.done() and events.resource_status()['operations'] == 0


async def test_authenticated_dashboard_cannot_serve_expired_or_revoked_event_text(tmp_path, dashboard_auth):
    async with graph(tmp_path) as (events, policy, clock):
        source = memory_source(now=clock.now, ttl=60)
        await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto privado de prueba.', sources=(source,))
        async with conversation_client(events,policy,clock) as client:
            denied = await client.get('/api/conversations')
            assert denied.status_code == 401 and denied.headers['cache-control'] == 'no-store'
            live = await client.get('/api/conversations', headers=dashboard_auth)
            assert live.headers['x-kza-source'] == 'real' and 'Texto privado de prueba.' in live.text
            [projected] = live.json()['snapshot']['rows']
            assert projected['expires_at'] == source.expires_at
            assert projected['observed_at'] == clock.now
            assert projected['success'] is None  # No execution receipt was recorded.
            clock.now += 60
            expired = await client.get('/api/conversations', headers=dashboard_auth)
            assert expired.json()['snapshot'] is None and expired.headers['cache-control'] == 'no-store'
            await events.acall(policy.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
            assert (await client.get('/api/ha/actions', headers=dashboard_auth)).json() == []


async def test_conversation_snapshot_distinguishes_same_capture_actions_at_the_same_time(tmp_path, dashboard_auth):
    async with graph(tmp_path) as (events, policy, clock):
        source = memory_source(now=clock.now)
        for entity in ('light.first', 'light.second', 'light.first'):
            await events.alog(entity, 'turn_on', trigger_phrase='Synthetic shared capture.', sources=(source,))
        async with conversation_client(events,policy,clock) as client:
            snapshot = (await client.get('/api/conversations',headers=dashboard_auth)).json()['snapshot']['rows']
            assert len(snapshot) == len({row['id'] for row in snapshot}) == 3
            assert {row['id'] for row in (await client.get('/api/conversations',headers=dashboard_auth)).json()['snapshot']['rows']} == {row['id'] for row in snapshot}


async def test_fresh_logger_initializes_revocations_before_reading_old_policy_rows(tmp_path):
    async with graph(tmp_path) as (events, policy, clock):
        source = memory_source(now=clock.now)
        await events.alog('light.synthetic', 'turn_on', trigger_phrase='Texto revocado.', sources=(source,))
        await events.acall(policy.delete_conversation, source.conversation_id, MemoryReviewer('test-admin'))
        reopened_policy = FactLedger(str(tmp_path/'privacy.db'), clock=clock)
        reopened = EventLogger(str(events.db_path), source_policy=reopened_policy, clock=clock)
        try:
            assert await reopened.acall(reopened.get_events) == []
            assert await reopened.acall(reopened.cleanup_old_events) == 1
        finally:
            await reopened.aclose()
            await asyncio.to_thread(reopened_policy.close)


async def test_committed_event_precedes_concurrent_tombstone_and_cannot_reappear(tmp_path, monkeypatch):
    from contextlib import contextmanager
    async with graph(tmp_path) as (events, policy, clock):
        await events.acall(events.initialize)
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()
        connect = events._connect

        @contextmanager
        def blocked_connection():
            entered.set()
            assert release.wait(3)
            with connect() as db:
                yield db

        with monkeypatch.context() as patch:
            patch.setattr(events, '_connect', blocked_connection)
            writing = asyncio.create_task(events.acall(events.log, 'light.synthetic', 'turn_on',
                trigger_phrase='Texto previo al borrado.', sources=(source,)))
            deleting = None
            try:
                await until(entered.is_set)
                deleting = asyncio.create_task(asyncio.to_thread(policy.delete_conversation, source.conversation_id, MemoryReviewer('test-admin')))
                await asyncio.sleep(.02)
                assert not deleting.done()
                release.set()
                await writing
                await deleting
            finally:
                release.set()
                await asyncio.gather(writing, *([deleting] if deleting else []), return_exceptions=True)
        assert await events.acall(events.get_events) == []
        assert await events.acall(events.cleanup_old_events) == 1


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_voice_events_keep_actual_provenance_when_memory_catalogue_is_disabled(tmp_path, monkeypatch, orchestrated):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    import numpy as np
    from src.orchestrator.dispatcher import MultiUserOrchestrator
    from src.pipeline.command_event import CommandEvent
    from src.pipeline.request_router import RequestRouter
    from tests.mocks.voice_graph import voice_graph
    owner, _, _, _, _ = voice_graph(monkeypatch)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(owner.aclose)
        events = await build_event_logger({'analytics': {'events_db': str(tmp_path/'events.db')}}, shutdown=stack)
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='Nexa explicame el cielo', user=None, emotion=None, timings={}, stt_confidence=None)))
        audio = MagicMock(); audio.detect_source_zone.return_value = 'zone_origin'
        orchestrator = MultiUserOrchestrator(None, None, None, llm=owner.llm, response_handler=owner) if orchestrated else None
        router = RequestRouter(processor, owner, audio, orchestrator=orchestrator,
            orchestrator_enabled=orchestrated, llm_reasoner=owner.llm, asr_event_logger=events,
            chroma_sync=SimpleNamespace(search_command=lambda *a, **kw: None),
            routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
        worker = None
        if orchestrator:
            orchestrator._running = True
            worker = asyncio.create_task(orchestrator._process_queue())
        try:
            event = CommandEvent(np.zeros(16000, dtype=np.float32), 'origin')
            result = await router.process_command(event)
            assert result['success']
            logged = (await events.acall(events.get_events))[0]
            assert logged.sources[0].capture_id == event.capture_id
            assert logged.sources[0].captured_at == event.timestamp
            if orchestrator:
                history = orchestrator.context_manager.get('unknown').conversation_history
                assert len(history) == 2 and all(turn.sources == logged.sources for turn in history)
        finally:
            if worker:
                orchestrator._running = False
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
