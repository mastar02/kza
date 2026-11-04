"""Capture receipts for native NLU calls that outlive their observer."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import threading
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from src.core.owned_work import OwnedBlockingWork
from src.main import build_application
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.integration.test_capture_native_drains import submissions as _submissions_fixture
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase
from tests.mocks.capture_models import capture_factories
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio


@pytest.fixture
def submissions(monkeypatch):
    return _submissions_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def native_submissions(monkeypatch):
    from src.core.native_executor import NativeThreadPoolExecutor
    original = NativeThreadPoolExecutor.submit
    local, futures = threading.local(), {}

    def submit(executor, function, /, *args, **kwargs):
        key = object()

        def invoke():
            local.key = key
            try:
                return function(*args, **kwargs)
            finally:
                del local.key

        future = original(executor, invoke)
        futures[key] = future
        return future

    monkeypatch.setattr(NativeThreadPoolExecutor, 'submit', submit)
    return SimpleNamespace(current=lambda:local.key, future=lambda key:futures.get(key))


@asynccontextmanager
async def nlu_graph(tmp_path, monkeypatch, submissions):
    from src.llm import reasoner
    state = SimpleNamespace(entered=threading.Event(), release=threading.Event(),
        returned=threading.Event(), armed=False, keys=[], drain_entered=asyncio.Event(), nlu_work=None)
    actual_client, actual_drain = httpx.Client, OwnedBlockingWork.drain_admitted

    async def observed_drain(work):
        if work is state.nlu_work:
            state.drain_entered.set()
        return await actual_drain(work)

    monkeypatch.setattr(OwnedBlockingWork, 'drain_admitted', observed_drain)

    def handler(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'object':'list','data':[{'id':'synthetic-router','object':'model'}]})
        try:
            if state.armed:
                state.keys.append(submissions.current())
                state.entered.set()
                assert state.release.wait(8), 'synthetic NLU response not released'
            return httpx.Response(200, json={'id':'synthetic','object':'text_completion','created':0,
                'model':'synthetic-router','choices':[{'index':0,'text':'{"is_command":false,"confidence":0.9}','finish_reason':'stop'}]})
        finally:
            if state.armed:
                state.returned.set()

    class HttpxLeaf:
        @staticmethod
        def Client(*args, **kwargs):
            return actual_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        def __getattr__(self, name):
            return getattr(httpx, name)

    monkeypatch.setattr(reasoner, 'httpx', HttpxLeaf())
    monkeypatch.setattr(reasoner, '_resolve_api_key', lambda *args, **kwargs:'synthetic-token')
    leaves = capture_factories(capability_leaves.__wrapped__(monkeypatch, tmp_path))
    async with TaskTransport().serve() as url:
        config, _ = configuration(tmp_path, url)
        config['router'] = {'enabled':True,'base_url':'http://synthetic.invalid/v1','model':'synthetic-router','destination':'local'}
        config['nlu'] = {'llm_router':{'enabled':True,'timeout_s':.05}}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            request_router = graph.pipeline.request_router
            nlu = request_router.llm_command_router
            state.nlu_work = nlu._work
            try:
                yield graph, request_router, nlu, state
            finally:
                state.release.set()
                await state.nlu_work.drain_admitted()


async def test_nlu_timeout_native_call_blocks_global_mute_until_actual_return(tmp_path, monkeypatch, submissions):
    async with nlu_graph(tmp_path, monkeypatch, submissions) as (graph, request_router, nlu, state):
        source = graph.privacy.source_for_capture(str(uuid4()), graph.privacy.ledger.clock(), 'synthetic-room')
        state.armed = True
        classify = asyncio.create_task(nlu.classify('enciende luz', room_id='synthetic-room', sources=(source,)))
        try:
            await until(state.entered.is_set)
            await until(lambda: state.keys and submissions.future(state.keys[0]) is not None)
            future = submissions.future(state.keys[0])
            result = await asyncio.wait_for(asyncio.shield(classify), 3)
            assert result.is_command is None and result.rejection_reason == 'unavailable'
            assert classify.done() and not classify.cancelled()
            assert future.running() and not future.done() and not state.returned.is_set()
            assert nlu._work.status()['operations'] == 1
            assert request_router._query_work.status()['operations'] == 0
            assert graph.pipeline.command_processor._work.status()['operations'] == 0
            control = graph.capture_control
            before = control.snapshot()
            assert before['supported'] and before['phase'] == 'running'
            control.request(True, before['revision'], actor())
            await until(lambda: control.snapshot()['total_mute_confirmed']
                        or control.snapshot()['phase'] == 'failed' or state.drain_entered.is_set())
            pending = control.snapshot()
            assert not pending['total_mute_confirmed'], 'global ACK preceded actual NLU native return'
            assert pending['phase'] == 'muting' and pending['ingress_sealed']
            assert state.drain_entered.is_set()
            assert future.running() and not future.done()
            assert not nlu._work._closed and not request_router.router._adapter_sealed
            state.release.set()
            muted = await capture_phase(control, 'muted')
            assert state.returned.is_set() and future.done()
            assert muted['total_mute_confirmed']
            assert not nlu._work._closed and not request_router.router._adapter_sealed
        finally:
            state.release.set()
            await asyncio.gather(classify, return_exceptions=True)


async def test_fast_router_finished_work_is_not_the_submitted_future_return(tmp_path, monkeypatch, native_submissions):
    from src.llm.router_lifecycle import _RouterWork
    original = _RouterWork.run
    entered, release = threading.Event(), threading.Event()
    keys, works = [], []

    def native_postlude(work):
        result = original(work)
        works.append(work)
        keys.append(native_submissions.current())
        entered.set()
        assert release.wait(8), 'synthetic RouterWork postlude not released'
        return result

    monkeypatch.setattr(_RouterWork, 'run', native_postlude)
    async with nlu_graph(tmp_path, monkeypatch, native_submissions) as (graph, request_router, nlu, state):
        fast = request_router.router
        caller = asyncio.create_task(fast.complete('synthetic command'))
        try:
            await until(lambda: entered.is_set() or caller.done())
            if caller.done():
                caller.result()
            await until(lambda: keys and native_submissions.future(keys[0]) is not None)
            future = native_submissions.future(keys[0])
            assert works[0].finished and not fast._router_work
            with fast._adapter_condition:
                assert not any(lease.active for lease in fast._adapter_leases)
            assert future.running() and not future.done()
            caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
            assert caller.cancelled() and future.running() and not future.done()
            assert request_router._query_work.status()['operations'] == 0
            assert nlu._work.status()['operations'] == 0
            reads = []
            actual_done = future.done

            def observed_done():
                value = actual_done()
                if not value:
                    reads.append(False)
                return value

            monkeypatch.setattr(future, 'done', observed_done)
            control = graph.capture_control
            before = control.snapshot()
            assert before['supported'] and before['phase'] == 'running'
            control.request(True, before['revision'], actor())
            await until(lambda: control.snapshot()['total_mute_confirmed']
                        or control.snapshot()['phase'] == 'failed' or len(reads) >= 2)
            pending = control.snapshot()
            assert not pending['total_mute_confirmed'], 'global ACK preceded the actual submitted FastRouter Future return'
            assert pending['phase'] == 'muting' and pending['ingress_sealed']
            assert future.running() and not actual_done()
            assert not fast._adapter_sealed and not fast._router_executor._shutdown
            release.set()
            muted = await capture_phase(control, 'muted')
            assert actual_done() and muted['total_mute_confirmed']
            assert not fast._adapter_sealed and not fast._router_executor._shutdown
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
