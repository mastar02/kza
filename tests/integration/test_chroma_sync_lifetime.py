"""Installed Chroma SDK ownership through HA lookup, routine writes and close."""

import asyncio
from contextlib import AsyncExitStack
import fcntl
import os
from pathlib import Path
import json
import subprocess
import sys
import threading

import chromadb
from chromadb.config import Settings
import numpy as np
import pytest

from src.core.logging import LogContext, get_context
from src.core.owned_work import WorkCloseTimeout
from src.memory.privacy_services import PrivacyServices
from src.vectordb.chroma_sync import ChromaSync
from tests.integration.test_voice_service_lifecycle import until

pytestmark = pytest.mark.asyncio


class Encoder:
    def encode(self, text):
        return np.array([.1, .2, .3])


class Collection:
    """Retain real SDK I/O behind a controllable native call boundary."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.queried = False

    def query(self, **kwargs):
        self.entered.set()
        assert self.release.wait(5)
        result = self.sdk.query(**kwargs)
        self.queried = True
        return result

    def __getattr__(self, name):
        return getattr(self.sdk, name)


class Client:
    def __init__(self, sdk):
        self.sdk = sdk
        self.collections = []
        self.close_calls = 0
        self.open_context = get_context()
        self.close_contexts = []
        self.close_threads = []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def get_or_create_collection(self, **kwargs):
        collection = Collection(self.sdk.get_or_create_collection(embedding_function=None, **kwargs))
        self.collections.append(collection)
        return collection

    def list_collections(self):
        return self.sdk.list_collections()

    def close(self):
        self.close_calls += 1
        self.close_contexts.append(get_context())
        self.close_threads.append(threading.get_ident())
        self.entered.set()
        assert self.release.wait(5)
        self.sdk.close()


@pytest.fixture
async def graph(tmp_path, monkeypatch):
    import src.vectordb.chroma_sync as module
    clients = []
    original = chromadb.PersistentClient
    def factory(**kwargs):
        assert kwargs['settings'].anonymized_telemetry is False
        client = Client(original(**kwargs))
        clients.append(client)
        return client
    monkeypatch.setattr(chromadb, 'PersistentClient', factory)
    monkeypatch.setattr(module, 'SentenceTransformer', lambda *a, **kw: Encoder())
    owner = ChromaSync(str(tmp_path/'index'), 'synthetic', 'cpu')
    try:
        yield owner, clients, original
    finally:
        for client in clients:
            client.release.set()
            for collection in client.collections:
                collection.release.set()
        await owner.begin_close()


async def seed(owner):
    await asyncio.to_thread(owner.commands.add, ids=['synthetic'], embeddings=[[.1, .2, .3]],
        documents=['Synthetic light command'], metadatas=[{'entity_id':'light.synthetic',
            'domain':'light', 'service':'turn_on', 'friendly_name':'Synthetic'}])


async def test_initialization_is_idempotent_and_close_preserves_persisted_index(graph):
    owner, clients, original = graph
    with LogContext(private='Synthetic caller context'):
        await seed(owner)
        await asyncio.gather(asyncio.to_thread(owner.initialize), asyncio.to_thread(owner.warmup_embedder))
        assert len(clients) == 1
        await owner.aclose()
        await owner.aclose()
    assert clients[0].close_calls == 1 and clients[0].sdk._closed
    assert len(clients[0].close_threads) == 1
    assert clients[0].open_context == {} and clients[0].close_contexts == [{}]
    assert clients[0].close_threads[0] != threading.get_ident()
    peer = await asyncio.to_thread(original, path=owner.chroma_path, settings=Settings(anonymized_telemetry=False))
    try:
        assert await asyncio.to_thread(peer.get_collection('home_assistant_commands', embedding_function=None).count) == 1
    finally:
        await asyncio.to_thread(peer.close)


async def test_cancelled_async_search_retains_real_query_before_sdk_close(graph):
    owner, clients, _ = graph
    await seed(owner)
    query = clients[0].collections[0]
    query.release.clear()
    caller = asyncio.create_task(owner.asearch_command('Synthetic query'))
    await until(query.entered.is_set)
    caller.cancel()
    await asyncio.gather(caller, return_exceptions=True)
    closing = owner.begin_close()
    await asyncio.sleep(.02)
    assert not closing.done() and clients[0].close_calls == 0
    assert owner.resource_status()['operations'] == 1
    with pytest.raises(RuntimeError, match='blocking_work_closed'):
        await owner.asearch_command('After close')
    with pytest.raises(RuntimeError, match='chroma_sync_closed'):
        await asyncio.to_thread(owner.search_command, 'After close')
    query.release.set()
    await closing
    assert query.queried and clients[0].close_calls == 1


@pytest.mark.parametrize('view_name,method,kwargs', [
    ('commands', 'count', {}),
    ('routines', 'add', {'ids':['late'], 'embeddings':[[.1,.2,.3]]}),
    ('embedder', 'encode', {'text':'Synthetic query'}),
])
async def test_previously_saved_views_cannot_reopen_or_use_retired_sdk(graph, view_name, method, kwargs):
    owner, clients, _ = graph
    await seed(owner)
    view = getattr(owner, view_name)
    await owner.aclose()
    with pytest.raises(RuntimeError, match='chroma_sync_closed'):
        await asyncio.to_thread(getattr(view, method), **kwargs)
    assert len(clients) == 1 and clients[0].close_calls == 1
    assert owner._client is owner._embedder is owner._commands_collection is None


async def test_cancelled_sync_routine_write_keeps_encoder_and_client_until_actual_exit(graph, monkeypatch):
    owner, clients, _ = graph
    await seed(owner)
    entered, release = threading.Event(), threading.Event()
    def encode(text):
        entered.set()
        assert release.wait(5)
        return np.array([.1,.2,.3])
    monkeypatch.setattr(owner._embedder, 'encode', encode)
    caller = asyncio.create_task(asyncio.to_thread(owner.save_routine,
        {'id':'synthetic-routine', 'name':'Synthetic', 'description':'Synthetic routine'}))
    try:
        await until(entered.is_set)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        closing = owner.begin_close()
        await asyncio.sleep(.02)
        assert not closing.done() and clients[0].close_calls == 0
    finally:
        release.set()
    await closing
    assert clients[0].collections[1].sdk is not None and clients[0].sdk._closed


async def test_encoder_initialization_failure_closes_partial_client_once_with_safe_error(graph, monkeypatch):
    import src.vectordb.chroma_sync as module
    owner, clients, _ = graph
    def failed(*a, **kw):
        raise ValueError('Synthetic private model failure')
    monkeypatch.setattr(module, 'SentenceTransformer', failed)
    with pytest.raises(RuntimeError, match='^chroma_sync_initialization_failed$') as observed:
        await asyncio.to_thread(owner.initialize)
    assert observed.value.__context__ is None
    assert clients[0].sdk._closed and clients[0].close_calls == 1
    await owner.aclose()
    assert clients[0].close_calls == 1


async def test_close_drains_held_model_constructor_and_releases_created_sdk_handle(graph, monkeypatch):
    import src.vectordb.chroma_sync as module
    owner, clients, _ = graph
    entered, release = threading.Event(), threading.Event()
    def held(*a, **kw):
        entered.set()
        assert release.wait(5)
        return Encoder()
    monkeypatch.setattr(module, 'SentenceTransformer', held)
    caller = asyncio.create_task(asyncio.to_thread(owner.warmup_embedder))
    try:
        await until(entered.is_set)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        closing = owner.begin_close()
        await asyncio.sleep(.02)
        assert not closing.done() and clients[0].close_calls == 0
    finally:
        release.set()
    await closing
    assert clients[0].sdk._closed and owner._embedder is None


async def test_own_close_does_not_reset_or_stop_another_client_on_same_sdk_system(graph):
    owner, clients, original = graph
    await seed(owner)
    peer = await asyncio.to_thread(original, path=owner.chroma_path, settings=Settings(anonymized_telemetry=False))
    try:
        await owner.aclose()
        assert clients[0].sdk._closed and not peer._closed
        collection = peer.get_collection('home_assistant_commands', embedding_function=None)
        assert await asyncio.to_thread(collection.count) == 1
    finally:
        await asyncio.to_thread(peer.close)


async def test_native_sdk_close_timeout_and_observer_cancellation_keep_privacy_writer_lease(graph, tmp_path):
    owner, clients, _ = graph
    await seed(owner)
    privacy = PrivacyServices(str(tmp_path/'privacy.db'))
    await privacy.acall(privacy.initialize)
    privacy.add_borrower(owner.begin_close)
    clients[0].release.clear()
    observer = asyncio.create_task(owner.aclose())
    try:
        await until(clients[0].entered.is_set)
        observer.cancel()
        with pytest.raises(WorkCloseTimeout, match='chroma_sync_still_running'):
            await observer
        closing = privacy.begin_close()
        await asyncio.sleep(.02)
        assert not closing.done() and privacy.resource_status()['writer_lease']
        assert owner.resource_status()['close_phase'] == 'resources'
        descriptor = os.open(str(tmp_path/'privacy.db')+'.lock', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    finally:
        clients[0].release.set()
        await owner.begin_close()
        await privacy.begin_close()
    assert not privacy.resource_status()['writer_lease']
    assert clients[0].close_calls == 1


@pytest.mark.parametrize('stage', ['reasoner', 'privacy'])
async def test_application_registers_chroma_cleanup_before_later_construction_failure(graph, monkeypatch, tmp_path, stage):
    import src.main as module
    from unittest.mock import AsyncMock, MagicMock
    owner, clients, _ = graph
    await seed(owner)
    monkeypatch.setattr(module, 'load_dotenv', lambda: None)
    monkeypatch.setattr(module, 'load_config', lambda _: {'home_assistant':{'url':'http://synthetic.invalid', 'token':'synthetic'},
        'embeddings':{'model':'synthetic','device':'cpu'}, 'router':{'enabled':False},
        'memory':{'privacy_db_path':str(tmp_path/'privacy.db'), 'enabled':False}})
    ha = MagicMock(test_connection=AsyncMock(return_value=True), close=AsyncMock())
    monkeypatch.setattr(module, 'HomeAssistantClient', lambda **kw: ha)
    monkeypatch.setattr(module, '_preflight_vram', lambda: None)
    monkeypatch.setattr(module, 'create_stt', lambda _: None)
    monkeypatch.setattr(module, 'create_tts', lambda _: None)
    monkeypatch.setattr(module, 'ChromaSync', lambda **kw: owner)
    privacy_seen = []
    if stage == 'reasoner':
        import src.llm.cloud_consent as consent
        def failed(*a, **kw):
            raise RuntimeError('synthetic_later_factory_failure')
        monkeypatch.setattr(consent, 'resolve_reasoner_gate', failed)
    else:
        async def failed(*a, privacy, **kw):
            privacy_seen.append(privacy)
            assert privacy.resource_status()['borrowers'] == 1
            assert not owner.resource_status()['closed']
            raise RuntimeError('synthetic_later_factory_failure')
        monkeypatch.setattr(module, 'build_memory_services', failed)
    with pytest.raises(RuntimeError, match='synthetic_later_factory_failure'):
        async with AsyncExitStack() as stack:
            await module._run_main(stack)
    assert owner.resource_status()['close_phase'] == 'complete'
    assert clients[0].close_calls == 1 and clients[0].sdk._closed
    assert all(not privacy.resource_status()['writer_lease'] for privacy in privacy_seen)
    ha.close.assert_awaited_once()


SDK_FAILURE_CHILD = """
import asyncio,fcntl,json,os,sys
from pathlib import Path
import src.vectordb.chroma_sync as module
from src.memory.privacy_services import PrivacyServices
class Client:
    calls=0
    def __init__(self,sdk):self.sdk=sdk
    def get_or_create_collection(self,**kwargs):
        return self.sdk.get_or_create_collection(embedding_function=None,**kwargs)
    def close(self):
        self.calls+=1
        self.sdk.close()
        raise RuntimeError('Synthetic private SDK close payload')
async def main():
    root=Path(sys.argv[1]);real=module.chromadb.PersistentClient;clients=[]
    def factory(**kwargs):
        client=Client(real(**kwargs));clients.append(client);return client
    module.chromadb.PersistentClient=factory
    module.SentenceTransformer=lambda *a,**kw:object()
    owner=module.ChromaSync(str(root/'index'),'synthetic','cpu')
    await asyncio.to_thread(owner.initialize)
    privacy=PrivacyServices(str(root/'privacy.db'))
    await privacy.acall(privacy.initialize)
    privacy.add_borrower(owner.begin_close)
    failures=[]
    for closing in (owner.begin_close(),privacy.begin_close(),owner.begin_close()):
        try:await closing
        except RuntimeError as error:
            assert str(error)=='chroma_sync_sdk_close_failed' and error.__context__ is None
            failures.append(True)
    status=owner.resource_status()
    assert len(failures)==3 and status['client_close_failed'] and not status['client_closed']
    assert privacy.resource_status()['writer_lease'] and clients[0].calls==1 and clients[0].sdk._closed
    descriptor=os.open(str(root/'privacy.db')+'.lock',os.O_RDWR)
    rejected=False
    try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:rejected=True
    finally:os.close(descriptor)
    assert rejected
    print(json.dumps({'passed':True,'lease_retained':True,'sdk_close_called_once':True,
        'competing_writer_rejected':True,'private_exception_context_removed':True}))
asyncio.run(main())
"""


async def test_sdk_close_failure_cannot_be_retried_as_false_completion(tmp_path):
    import src.vectordb.chroma_sync as module
    root = Path(module.__file__).resolve().parents[2]
    script = 'import sys\nsys.path.insert(0,'+repr(str(root))+')\n'+SDK_FAILURE_CHILD
    result = await asyncio.to_thread(subprocess.run, [sys.executable, '-B', '-E', '-s', '-c', script, str(tmp_path)],
        cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr[-2000:]
    report = next(json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith('{'))
    assert report['passed'] and report['lease_retained'] and report['sdk_close_called_once']


async def test_close_drains_sdk_constructor_even_before_created_handle_is_assigned(graph, monkeypatch):
    owner, clients, _ = graph
    factory = chromadb.PersistentClient
    entered, release = threading.Event(), threading.Event()
    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        # The SDK reference is created only after admission has closed. The
        # owner must still wait, receive that handle and close it exactly once.
        return factory(**kwargs)
    monkeypatch.setattr(chromadb, 'PersistentClient', held)
    caller = asyncio.create_task(asyncio.to_thread(owner.initialize))
    try:
        await until(entered.is_set)
        assert owner._client is None and not clients
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        closing = owner.begin_close()
        await asyncio.sleep(.02)
        assert not closing.done() and not clients
        assert owner.resource_status()['operations'] == 1
    finally:
        release.set()
    await closing
    assert clients[0].sdk._closed and clients[0].close_calls == 1
