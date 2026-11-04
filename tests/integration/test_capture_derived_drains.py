"""Capture ACK includes actual metric and memory work after its parent ended."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import threading
import time
from uuid import uuid4

import pytest

import src.main as application_module
from src.main import build_application
from src.memory.services import build_memory_services
from tests.integration.test_capability_application import configuration
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase
from tests.integration.test_global_capture_http import leaves as capture_leaves
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capture_leaves.__wrapped__(monkeypatch, tmp_path)


@asynccontextmanager
async def application(tmp_path, leaves, monkeypatch, kind):
    client = None
    try:
        async with TaskTransport().serve() as url:
            config, _ = configuration(tmp_path, url)
            if kind == 'events':
                config['analytics']['enabled'] = True
            else:
                config['memory']['enabled'] = True
                # The actual builder/manager use the existing synthetic embedding
                # adapter. This explicit borrowed SDK peer outlives its owners.
                client = MemoryClient(tmp_path / 'memory-index')

                async def memory_builder(settings, **kwargs):
                    return await build_memory_services(settings, chroma_client=client, **kwargs)

                monkeypatch.setattr(application_module, 'build_memory_services', memory_builder)
            async with AsyncExitStack() as stack:
                graph = await build_application(config, shutdown=stack, factories=leaves)
                yield graph
    finally:
        if client is not None:
            client.close()


async def assert_native_blocks_ack(control, returned):
    # An observation window gives an incorrectly unblocked coordinator time to
    # report success. It never substitutes for either native entry or exit.
    await asyncio.wait((control._operation,), timeout=.5)
    state = control.snapshot()
    assert not returned.is_set()
    assert not state['total_mute_confirmed']
    assert state['phase'] == 'muting' and state['ingress_sealed']
    assert not control._operation.done()


async def test_timed_out_real_event_log_prevents_global_ack_until_native_return(tmp_path, leaves, dashboard_auth, monkeypatch):
    async with application(tmp_path, leaves, monkeypatch, 'events') as graph:
        router, control = graph.pipeline.request_router, graph.capture_control
        events = router.event_logger
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        workers = []
        log = events.log

        def held(*args, **kwargs):
            value = log(*args, **kwargs)
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5), 'real event logger was not released'
            returned.set()
            return value

        monkeypatch.setattr(events, 'log', held)
        source = graph.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-derived')
        token = router._memory_source.set(source)
        try:
            parent = asyncio.create_task(router._log_domotics_event(
                'light.living', 'turn_on', 'Synthetic private metric input'))
        finally:
            router._memory_source.reset(token)
        try:
            await until(entered.is_set)
            await asyncio.wait_for(asyncio.shield(parent), 2)
            assert parent.done() and events.resource_status()['operations'] == 1
            assert len(workers) == 1 and workers[0].ident is not None
            control.request(True, control.snapshot()['revision'], actor())
            await assert_native_blocks_ack(control, returned)
        finally:
            release.set()
            await asyncio.gather(parent, return_exceptions=True)
            if control._operation is not None:
                await asyncio.shield(control._operation)
        await capture_phase(control, 'muted')
        assert returned.is_set() and events.resource_status()['operations'] == 0
        assert control.snapshot()['total_mute_confirmed']
        assert not events.resource_status()['closed']


async def test_cancelled_real_memory_record_prevents_global_ack_until_native_return(tmp_path, leaves, dashboard_auth, monkeypatch):
    async with application(tmp_path, leaves, monkeypatch, 'memory') as graph:
        router, control = graph.pipeline.request_router, graph.capture_control
        manager, memory = router.memory, router.memory.long_term
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        workers = []
        record = manager.record_interaction

        def held(*args, **kwargs):
            result = record(*args, **kwargs)
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5), 'real long-term memory call was not released'
            returned.set()
            return result

        monkeypatch.setattr(manager, 'record_interaction', held)
        source = graph.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-derived')
        parent = asyncio.create_task(router._memory_call('record_interaction',
            user_input='Synthetic private input', assistant_response='Synthetic response',
            intent='conversation', sources=(source,)))
        try:
            await until(entered.is_set)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent
            assert memory.resource_status()['operations'] == 1
            assert len(workers) == 1 and workers[0].ident is not None
            control.request(True, control.snapshot()['revision'], actor())
            await assert_native_blocks_ack(control, returned)
        finally:
            release.set()
            await asyncio.gather(parent, return_exceptions=True)
            if control._operation is not None:
                await asyncio.shield(control._operation)
        await capture_phase(control, 'muted')
        assert returned.is_set() and memory.resource_status()['operations'] == 0
        assert control.snapshot()['total_mute_confirmed']
        assert not memory.resource_status()['closed']
