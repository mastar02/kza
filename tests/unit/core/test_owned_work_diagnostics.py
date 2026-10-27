"""Native progress remains observable without changing actual resource ownership."""

import asyncio
from contextvars import ContextVar
import json
import threading
from weakref import ref

import pytest

from src.core.owned_work import ClosePhase,OwnedBlockingWork,WorkCloseTimeout,observe_native
from src.core.logging import LogContext
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio


async def test_native_deadline_reports_running_callback_without_releasing_capacity():
    work = OwnedBlockingWork(capacity=1,close_timeout_s=.02)
    entered,release = threading.Event(),threading.Event()
    def held():
        entered.set()
        assert release.wait(3)
    observing = asyncio.create_task(work.run(held))
    try:
        await until(entered.is_set)
        observing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observing
        with pytest.raises(WorkCloseTimeout) as result:
            await work.aclose()
        assert result.value.args==('blocking_work_still_running',) and result.value.phase is ClosePhase.DRAIN
        [sample] = result.value.diagnostics['active']
        assert sample['phase']=='native' and sample['native_ms']>0 and sample['native_queue_ms']>=0
        assert work.status()['operations']==1
    finally:
        release.set()
        await work.begin_close()
    assert work.status()['operations']==0
    assert work.diagnostics()['recent'][0]['phase']=='complete'


async def test_worker_admission_keeps_function_identity_arguments_and_context():
    entered,release = asyncio.Event(),asyncio.Event()
    private = ContextVar('synthetic_native_context',default='')
    seen = []
    def function(argument):
        seen.append((argument,private.get()))
    async def worker(callback,*args,**kwargs):
        assert callback is function and args==('Synthetic private argument',) and kwargs=={}
        entered.set()
        await release.wait()
        return await asyncio.to_thread(observe_native,callback,*args,**kwargs)
    work = OwnedBlockingWork(worker=worker)
    token = private.set('Synthetic private context')
    observing = asyncio.create_task(work.run(function,'Synthetic private argument'))
    private.reset(token)
    try:
        await until(entered.is_set)
        [sample] = work.diagnostics()['active']
        assert sample['phase']=='worker' and sample['native_ms'] is None
        assert 'Synthetic private' not in json.dumps(work.diagnostics())
    finally:
        release.set()
        await observing
        await work.aclose()
    assert seen==[('Synthetic private argument','Synthetic private context')]
    assert 'Synthetic private' not in json.dumps(work.diagnostics())


async def test_after_native_wait_is_distinct_from_running_or_completed_work():
    returned,release = asyncio.Event(),asyncio.Event()
    async def worker(function,*args,**kwargs):
        result = await asyncio.to_thread(observe_native,function,*args,**kwargs)
        returned.set()
        await release.wait()
        return result
    work = OwnedBlockingWork(worker=worker)
    observing = asyncio.create_task(work.run(lambda:None))
    try:
        await until(returned.is_set)
        [sample] = work.diagnostics()['active']
        assert sample['phase']=='after_native' and sample['native_ms']>=0
        assert sample['after_native_ms']>=0 and work.status()['operations']==1
    finally:
        release.set()
        await observing
        await work.aclose()


async def test_history_is_bounded_detached_and_cannot_keep_arguments_alive():
    class Payload:
        def __repr__(self):
            raise AssertionError('Private payload formatted')
    payload = Payload()
    reference = ref(payload)
    work = OwnedBlockingWork(capacity=2)
    await work.run(lambda argument:None,payload)
    payload = None
    for _ in range(10):
        await work.run(lambda:None)
    await until(lambda:work.status()['operations']==0)
    assert reference() is None
    assert len(work.diagnostics()['recent'])==2
    detached = work.diagnostics()
    detached['recent'][0]['phase']='modified'
    assert work.diagnostics()['recent'][0]['phase']=='complete'
    json.dumps(work.diagnostics(),allow_nan=False)
    await work.aclose()


async def test_uninstrumented_async_worker_is_reported_without_claiming_native_execution():
    entered,release = asyncio.Event(),asyncio.Event()
    async def worker(function):
        entered.set()
        await release.wait()
        return function()
    work = OwnedBlockingWork(worker=worker)
    observing = asyncio.create_task(work.run(lambda:None))
    try:
        await until(entered.is_set)
        [sample] = work.diagnostics()['active']
        assert sample['phase']=='worker' and sample['native_ms'] is None
    finally:
        release.set()
        await observing
        await work.aclose()
    assert work.diagnostics()['recent'][0]['native_ms'] is None


async def test_completed_cleanup_task_cannot_keep_arbitrary_caller_context():
    class Payload:
        pass
    work = OwnedBlockingWork()
    payload = Payload()
    reference = ref(payload)
    with LogContext(private=payload):
        closing = work.begin_close()
    payload = None
    await closing
    assert reference() is None
