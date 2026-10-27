"""Timeout evidence names actual cleanup phases while native users stay owned."""

import asyncio
import json
import threading
from weakref import ref

import pytest

from src.core.owned_work import ClosePhase,WorkCloseTimeout
from src.core.logging import LogContext
from src.memory.privacy_services import PrivacyServices
from src.obs.metrics_emitter import MetricsEmitter
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('phase',[ClosePhase.LEDGER,ClosePhase.EXECUTOR])
async def test_privacy_timeout_records_actual_cleanup_phase_and_live_lease(tmp_path,monkeypatch,phase):
    privacy = PrivacyServices(str(tmp_path/'privacy.db'),close_timeout_s=.02)
    await privacy.acall(privacy.initialize)
    entered,release = threading.Event(),threading.Event()
    owner,method = (privacy.ledger,'close') if phase is ClosePhase.LEDGER else (privacy._executor,'shutdown')
    original = getattr(owner,method)
    def held(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args,**kwargs)
    monkeypatch.setattr(owner,method,held)
    observing = asyncio.create_task(privacy.aclose())
    try:
        await until(entered.is_set)
        with pytest.raises(WorkCloseTimeout) as result:
            await observing
        assert result.value.phase is phase
        status = result.value.diagnostics
        assert status['close_phase']==phase.value and status['writer_lease']
        assert status['close_io']['active'][0]['phase']=='native'
        assert not privacy.begin_close().done()
        with pytest.raises(RuntimeError,match='privacy_authority_unavailable'):
            privacy.add_borrower(privacy._work.begin_close)
        assert str(tmp_path) not in json.dumps(status)
    finally:
        release.set()
        await privacy.begin_close()
    assert privacy.resource_status()['close_phase']=='complete'
    assert not privacy.resource_status()['writer_lease']


@pytest.mark.parametrize('phase',[ClosePhase.SOCKET,ClosePhase.STORE])
async def test_metric_timeout_distinguishes_actual_socket_and_store_close(tmp_path,monkeypatch,phase):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'),close_timeout_s=.02)
    entered,release = threading.Event(),threading.Event()
    owner,method = (emitter,'_close_socket') if phase is ClosePhase.SOCKET else (emitter._store,'close')
    original = getattr(owner,method)
    def held():
        entered.set()
        assert release.wait(3)
        original()
    monkeypatch.setattr(owner,method,held)
    observing = asyncio.create_task(emitter.aclose())
    try:
        await until(entered.is_set)
        with pytest.raises(WorkCloseTimeout) as result:
            await observing
        assert result.value.phase is phase
        status = result.value.diagnostics
        assert not status['io']['active'] and not status['retention_io']['active']
        assert status['close_io']['active'][0]['phase']=='native'
        assert not emitter.begin_close().done()
        assert str(tmp_path) not in json.dumps(status)
    finally:
        release.set()
        await emitter.begin_close()
    assert emitter.resource_status()['close_phase']=='complete'


async def test_metric_start_timeout_reports_the_actual_retention_worker(tmp_path,monkeypatch):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'),close_timeout_s=.02)
    entered,release = threading.Event(),threading.Event()
    original = emitter._store.prune
    def held():
        entered.set()
        assert release.wait(3)
        original()
    monkeypatch.setattr(emitter._store,'prune',held)
    observing = asyncio.create_task(emitter.start())
    try:
        await until(entered.is_set)
        with pytest.raises(WorkCloseTimeout) as result:
            await observing
        assert result.value.args==('diagnostic_initialization_pending',)
        assert result.value.phase is ClosePhase.RETENTION
        assert result.value.diagnostics['retention_io']['active'][0]['phase']=='native'
        assert emitter._sweep is not None and not emitter._sweep.done()
    finally:
        release.set()
        await emitter.begin_close()


async def test_close_admission_does_not_wait_on_a_native_initializer_lock(tmp_path,monkeypatch):
    privacy = PrivacyServices(str(tmp_path/'privacy.db'))
    entered,release = threading.Event(),threading.Event()
    initialize = privacy.ledger.initialize
    def held():
        entered.set()
        assert release.wait(3)
        initialize()
    monkeypatch.setattr(privacy.ledger,'initialize',held)
    initializing = asyncio.create_task(privacy.acall(privacy.initialize))
    try:
        await until(entered.is_set)
        # The callback cannot run if begin_close blocks the loop on the native lock.
        asyncio.get_running_loop().call_soon(release.set)
        closing = privacy.begin_close()
        assert not closing.done() and privacy.resource_status()['writer_lease']
        await initializing
        await closing
        assert privacy.resource_status()['close_phase']=='complete'
    finally:
        release.set()
        await asyncio.gather(initializing,return_exceptions=True)
        await privacy.begin_close()


async def test_completed_privacy_close_does_not_keep_arbitrary_caller_context(tmp_path):
    class Payload:
        pass
    privacy = PrivacyServices(str(tmp_path/'privacy.db'))
    await privacy.acall(privacy.initialize)
    payload = Payload()
    reference = ref(payload)
    with LogContext(private=payload):
        closing = privacy.begin_close()
    payload = None
    await closing
    assert reference() is None
