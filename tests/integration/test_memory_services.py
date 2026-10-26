"""Runtime factory, source producers, both routers and authenticated review."""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest

from src.ambient.distiller import Distiller
from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.dashboard.api import DashboardAPI
from src.memory.services import build_memory_services
from src.memory.fact_ledger import MemoryReviewer
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_event import CommandEvent
from src.pipeline.request_router import RequestRouter
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.voice_graph import voice_graph

pytestmark = pytest.mark.asyncio


def configuration(tmp_path):
    return {'memory': {'privacy_db_path': str(tmp_path/'privacy.db'),
                       'preferences_path': str(tmp_path/'preferences.json')}}


async def test_factory_distiller_keeps_actual_sources_and_tombstones_inflight_derivation(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        store = AmbientStore(str(tmp_path/'capture.db'), clock=clock)
        await store.init(); stack.push_async_callback(store.close)
        utterance = AmbientUtterance('test_room', clock.now, clock.now+1, text='Un dato útil.', vad_prob=.9)
        await store.add(utterance)
        entered, release = asyncio.Event(), asyncio.Event()

        async def extract(prompt):
            entered.set(); await release.wait()
            return '[{"fact":"Un dato útil.","category":"fact","confidence":0.9}]'

        distiller = Distiller(store, extract, services.manager.long_term.astore_fact, min_batch=1)
        task = asyncio.create_task(distiller.distill_once())
        await asyncio.wait_for(entered.wait(), 1)
        await services.manager.long_term.acall(services.manager.long_term.delete_conversation, utterance.conversation_id, MemoryReviewer('test-admin'))
        release.set()
        assert await task == 0
        assert await services.manager.long_term.asearch_relevant('dato') == []
        # A different capture/recording scope can still produce usable memory.
        fresh = AmbientUtterance('another_room', clock.now, clock.now+1, text='Otro dato.', vad_prob=.9)
        await store.add(fresh)
        await store.mark_distilled([1])
        assert await distiller.distill_once() == 1
        fact = (await services.manager.long_term.asearch_relevant('dato'))[0]
        assert fact['sources'][0]['capture_id'] == fresh.capture_id
        assert fact['sources'][0]['conversation_id'] == fresh.conversation_id
        assert fact['state'] == 'automatic' and fact['expires_at'] == clock.now+30*86400


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_voice_proposal_can_only_become_permanent_through_authenticated_review(tmp_path, monkeypatch, dashboard_auth, orchestrated):
    clock = Clock()
    owner, _, _, _, _ = voice_graph(monkeypatch)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(owner.aclose)
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='Nexa recordá que prefiero cocinar temprano', user=None, emotion=None,
            timings={}, stt_confidence=None)))
        audio = MagicMock(); audio.detect_source_zone.return_value = 'zone_origin'
        orchestrator = MultiUserOrchestrator(None, None, None, llm=owner.llm,
            response_handler=owner, memory_manager=services.manager) if orchestrated else None
        router = RequestRouter(processor, owner, audio, orchestrator=orchestrator,
            orchestrator_enabled=orchestrated, memory_manager=services.manager,
            chroma_sync=SimpleNamespace(search_command=lambda *a, **kw: None),
            routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
        event = CommandEvent(np.zeros(16000, dtype=np.float32), 'origin', timestamp=clock.now, captured_at=clock.now)
        result = await router.process_command(event)
        assert result['success'] and result['intent'] == 'memory_proposal'
        proposal = result['memory_proposal']
        assert proposal['sources'][0]['capture_id'] == event.capture_id
        assert await services.manager.long_term.asearch_relevant('cocinar') == []
        app = DashboardAPI(memory_manager=services.manager).app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            denied = await client.post('/api/memory/proposals/'+proposal['id']+'/confirm')
            assert denied.status_code == 401
            assert denied.headers['cache-control'] == 'no-store'
            cross_origin = await client.post('/api/memory/proposals/'+proposal['id']+'/confirm',
                headers=dashboard_auth | {'Origin': 'https://foreign.invalid'})
            assert cross_origin.status_code == 403
            assert await services.manager.long_term.asearch_relevant('cocinar') == []
            confirmed = await client.post('/api/memory/proposals/'+proposal['id']+'/confirm', headers=dashboard_auth)
            assert confirmed.status_code == 200 and confirmed.json()['fact']['confirmed_by'] == 'dashboard-admin'
            stale_reject = await client.post('/api/memory/proposals/'+proposal['id']+'/reject', headers=dashboard_auth)
            assert stale_reject.status_code == 409
            snapshot = await client.get('/api/memory', headers=dashboard_auth)
            assert snapshot.json()['proposals'] == [] and snapshot.json()['facts'][0]['state'] == 'confirmed'
            assert snapshot.headers['cache-control'] == 'no-store'
            clock.now += 40*86400
            assert (await services.manager.long_term.asearch_relevant('cocinar'))[0]['state'] == 'confirmed'
            deleted = await client.delete('/api/memory/facts/'+proposal['id'], headers=dashboard_auth)
            assert deleted.json()['deleted'] and await services.manager.long_term.asearch_relevant('cocinar') == []
            assert (await client.get('/home/memory.js')).status_code == 200


async def test_disabled_memory_remains_unavailable_under_authenticated_dashboard(tmp_path, dashboard_auth):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=DashboardAPI().app), base_url='http://testserver') as client:
        response = await client.get('/api/memory', headers=dashboard_auth)
        assert response.status_code == 503 and response.headers['cache-control'] == 'no-store'


async def test_partial_derivation_failure_retries_without_duplicating_or_renewing_prior_fact(tmp_path):
    clock = Clock()
    async with AsyncExitStack() as stack:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        store = AmbientStore(str(tmp_path/'capture.db'), clock=clock)
        await store.init(); stack.push_async_callback(store.close)
        await store.add(AmbientUtterance('test_room', clock.now, clock.now+1, text='Dos datos útiles.', vad_prob=.9))

        async def extract(prompt):
            return '[{"fact":"Primero.","category":"fact"},{"fact":"Segundo.","category":"fact"}]'

        failed = False

        async def save(content, *args, **kwargs):
            nonlocal failed
            if content == 'Segundo.' and not failed:
                failed = True
                raise OSError('synthetic interrupted ledger write')
            return await services.manager.long_term.astore_fact(content, *args, **kwargs)

        distiller = Distiller(store, extract, save, min_batch=1)
        assert await distiller.distill_once() == 1
        initial = services.manager.long_term.ledger.list()[0]
        assert len(await store.undistilled_live(require_provenance=True)) == 1
        clock.now += 3600
        assert await distiller.distill_once() == 2
        rows = services.manager.long_term.ledger.list()
        assert len(rows) == 2 and services.manager.long_term.ledger.get(initial.id).expires_at == initial.expires_at
        assert await store.undistilled_live(require_provenance=True) == []


async def test_factory_maintenance_expires_catalog_and_index(tmp_path):
    import time
    config = configuration(tmp_path)
    config['memory']['automatic_retention_days'] = .00001
    async with AsyncExitStack() as stack:
        services = await build_memory_services(config, shutdown=stack, chroma_client=MemoryClient(tmp_path/'chroma'))
        from src.memory.fact_ledger import MemoryOrigin, MemorySource
        from uuid import uuid4
        now = time.time()
        source = MemorySource(str(uuid4()), str(uuid4()), now, now+48*3600, MemoryOrigin.VOICE)
        fact_id = await services.manager.long_term.astore_fact('Un hecho muy breve.', 'fact', sources=(source,))
        async with asyncio.timeout(3):
            while services.manager.long_term.ledger.get(fact_id) is not None or services.manager.long_term._collection.count():
                await asyncio.sleep(.01)
        assert services.manager.long_term.ledger.status()['pending_index'] == 0


async def test_short_term_reads_honor_actual_source_expiry_without_renewing_it(tmp_path):
    from uuid import uuid4
    clock = Clock()
    async with AsyncExitStack() as stack:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        source = services.manager.source_for_capture(str(uuid4()), clock.now, 'room', expires_at=clock.now+60)
        await services.manager.arecord_interaction('Texto de prueba.', 'Respuesta.', sources=(source,))
        clock.now += 59
        assert 'Texto de prueba.' in await services.manager.aformat_context_for_prompt('texto')
        clock.now += 1
        assert await services.manager.aformat_context_for_prompt('texto') == ''
        assert len(services.manager.short_term) == 0
        await services.manager.arecord_interaction('Texto nuevo.', 'Respuesta.')
        clock.now += 48*3600
        assert await services.manager.aformat_context_for_prompt('texto') == ''


async def test_legacy_explicit_preferences_stay_unchanged_and_out_of_prompts(tmp_path):
    import json
    clock = Clock()
    path = tmp_path/'preferences.json'
    original = json.dumps({'horario': {'key': 'horario', 'value': 'dato legacy privado',
        'confidence': 1, 'last_updated': clock.now, 'source': 'explicit'}})
    path.write_text(original)
    async with AsyncExitStack() as stack:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        assert await services.manager.aformat_context_for_prompt('horario') == ''
        proposal_id = await services.manager.long_term.acall(services.manager.remember_preference,
            'horario', 'propuesta nueva', explicit=True)
        assert await services.manager.aformat_context_for_prompt('horario') == ''
        await services.manager.long_term.acall(services.manager.long_term.confirm,
            proposal_id, MemoryReviewer('test-admin'))
        prompt = await services.manager.aformat_context_for_prompt('horario')
        assert 'propuesta nueva' in prompt and 'dato legacy privado' not in prompt
        assert path.read_text() == original


@pytest.mark.parametrize('orchestrated', [True, False])
async def test_both_voice_routes_retrieve_policy_memory_and_record_actual_capture_once(tmp_path, monkeypatch, orchestrated):
    from uuid import uuid4
    clock = Clock()
    owner, engine, _, endpoint, _ = voice_graph(monkeypatch)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(owner.aclose)
        services = await build_memory_services(configuration(tmp_path), shutdown=stack,
            chroma_client=MemoryClient(tmp_path/'chroma'), clock=clock)
        source = services.manager.source_for_capture(str(uuid4()), clock.now, 'origin')
        await services.manager.long_term.astore_fact('Dato sintético verificable.', 'fact', sources=(source,))
        processor = SimpleNamespace(process_command=AsyncMock(return_value=SimpleNamespace(
            text='Nexa explicame el cielo', user=None, emotion=None,
            timings={}, stt_confidence=None)))
        audio = MagicMock(); audio.detect_source_zone.return_value = 'zone_origin'
        orchestrator = MultiUserOrchestrator(None, None, None, llm=owner.llm,
            response_handler=owner, memory_manager=services.manager) if orchestrated else None
        router = RequestRouter(processor, owner, audio, orchestrator=orchestrator,
            orchestrator_enabled=orchestrated, memory_manager=services.manager, llm_reasoner=owner.llm,
            chroma_sync=SimpleNamespace(search_command=lambda *a, **kw: None),
            routine_manager=SimpleNamespace(handle=AsyncMock(return_value={'handled': False})))
        worker = None
        if orchestrator:
            orchestrator._running = True
            worker = asyncio.create_task(orchestrator._process_queue())
        try:
            event = CommandEvent(np.zeros(16000, dtype=np.float32), 'origin', timestamp=clock.now, captured_at=clock.now)
            result = await router.process_command(event)
            assert result['success'] and result['intent'] == 'conversation'
            posts = [request for request in endpoint.requests if request.method == 'POST']
            assert len(posts) == 1 and 'Dato sintético verificable.' in posts[0].content.decode('utf-8')
            assert len(engine.calls) == 1
            turns = services.manager.short_term.get_recent()
            assert len(turns) == 1 and turns[0].sources[0].capture_id == event.capture_id
            assert result['capture_id'] == event.capture_id and turns[0].sources[0].conversation_id == result['conversation_id']
            if orchestrator:
                history = orchestrator.context_manager.get('unknown').conversation_history
                assert len(history) == 2 and all(t.sources[0] == turns[0].sources[0] for t in history)
                assert all(t.timestamp == event.captured_at for t in history)
        finally:
            if worker:
                orchestrator._running = False
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
