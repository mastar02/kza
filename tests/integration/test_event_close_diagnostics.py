"""Actual private event workers close independently of general executor load."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from weakref import ref

import pytest

from src.analytics.event_logger import EventLogger
from src.core.logging import LogContext
from src.core.owned_work import ClosePhase,WorkCloseTimeout
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio


async def test_event_close_does_not_borrow_a_saturated_general_executor(tmp_path):
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered,release = threading.Event(),threading.Event()
    def hold():
        entered.set()
        assert release.wait(3)
    waiting = asyncio.create_task(asyncio.to_thread(hold))
    events = EventLogger(str(tmp_path/'events.db'))
    try:
        await until(entered.is_set)
        await events.acall(events.initialize)
        async with asyncio.timeout(1):
            await asyncio.shield(events.begin_close())
        assert not release.is_set()
        assert not any(thread.is_alive() for thread in events._executor._threads)
    finally:
        release.set()
        await waiting
        await events.begin_close()
        loop._default_executor = previous
        pool.shutdown(wait=True)


async def test_completed_event_cleanup_task_cannot_keep_arbitrary_caller_context(tmp_path):
    events = EventLogger(str(tmp_path/'events.db'))
    await events.acall(events.initialize)
    class Payload:
        pass
    payload = Payload()
    reference = ref(payload)
    with LogContext(private=payload):
        closing = events.begin_close()
    payload = None
    await closing
    assert reference() is None


async def test_event_close_timeout_reports_real_native_write_and_retains_private_pool(tmp_path):
    events = EventLogger(str(tmp_path/'events.db'))
    await events.acall(events.initialize)
    entered,release = threading.Event(),threading.Event()
    def held():
        entered.set()
        assert release.wait(3)
    writing = asyncio.create_task(events.acall(held))
    try:
        await until(entered.is_set)
        with pytest.raises(WorkCloseTimeout) as caught:
            await events.aclose()
        assert caught.value.phase is ClosePhase.DRAIN
        assert caught.value.diagnostics['operations']==1
        assert caught.value.diagnostics['io']['active'][0]['phase']=='native'
        assert any(thread.is_alive() for thread in events._executor._threads)
    finally:
        release.set()
        await writing
        await events.begin_close()
    assert events.resource_status()['close_phase']=='complete'
    assert not any(thread.is_alive() for thread in events._executor._threads)
