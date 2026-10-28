"""Real Chroma lifetimes, explicit ownership and retained native shutdown."""

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
import pytest

from src.core.logging import LogContext, get_context
from src.core.owned_work import WorkCloseTimeout
from src.memory.long_term import MemoryCloseTimeout
from src.memory.privacy_services import PrivacyServices
from src.memory.services import build_memory_services
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.memory_graph import MemoryClient, SyntheticEmbedding

pytestmark = pytest.mark.asyncio


def configuration(root):
    return {'memory': {'enabled': True, 'chroma_path': str(root/'index'),
        'privacy_db_path': str(root/'privacy.db'), 'preferences_path': str(root/'preferences.json')}}


class Client:
    """Keep the installed native SDK; replace only embeddings with CPU fixtures."""

    def __init__(self, sdk):
        self.sdk = sdk
        self.close_calls = 0
        self.close_contexts = []
        self.close_threads = []
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def get_or_create_collection(self, **kwargs):
        return self.sdk.get_or_create_collection(embedding_function=SyntheticEmbedding(), **kwargs)

    def close(self):
        self.close_calls += 1
        self.close_contexts.append(get_context())
        self.close_threads.append(threading.get_ident())
        self.entered.set()
        assert self.release.wait(4)
        self.sdk.close()


class Factory:
    def __init__(self, monkeypatch, *, hold=False):
        self.real = chromadb.PersistentClient
        self.clients, self.contexts, self.settings = [], [], []
        self.entered, self.release = threading.Event(), threading.Event()
        if not hold:
            self.release.set()
        monkeypatch.setattr(chromadb, 'PersistentClient', self)

    def __call__(self, **kwargs):
        self.contexts.append(get_context())
        self.settings.append(kwargs.get('settings'))
        kwargs.setdefault('settings', Settings(anonymized_telemetry=False))
        client = Client(self.real(**kwargs))
        self.clients.append(client)
        self.entered.set()
        assert self.release.wait(4)
        return client

    def cleanup(self):
        self.release.set()
        for client in self.clients:
            client.release.set()
            client.sdk.close()


async def test_factory_closes_owned_client_off_loop_once_and_never_resets_persisted_data(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    try:
        with LogContext(private='Synthetic startup context'):
            async with AsyncExitStack() as stack:
                services = await build_memory_services(configuration(tmp_path), shutdown=stack)
                collection = services.manager.long_term._collection
                await services.manager.long_term.acall(collection.add, ids=['synthetic-id'],
                    embeddings=[[.1, .2, .3]], documents=['Synthetic persisted index data'])
                assert services.manager is not None
            await services.aclose()
        client = factory.clients[0]
        assert client.close_calls == 1 and services.resource_status()['client_closed']
        assert factory.contexts == [{}] and client.close_contexts == [{}]
        assert client.close_threads[0] != threading.get_ident()
        assert factory.settings[0].anonymized_telemetry is False
        peer = factory.real(path=str(tmp_path/'index'), settings=Settings(anonymized_telemetry=False))
        try:
            assert peer.get_collection(name='user_memories_policy_v1', embedding_function=None).count() == 1
        finally:
            peer.close()
    finally:
        factory.cleanup()


async def test_borrowed_client_remains_open_and_readable_after_memory_services_close(tmp_path):
    client = MemoryClient(tmp_path/'index')
    async with AsyncExitStack() as stack:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack, chroma_client=client)
    assert not services.resource_status()['owns_client']
    assert not client.client._closed
    assert client.client.get_collection(name='user_memories_policy_v1', embedding_function=None).count() == 0
    client.close()


async def test_closing_owned_reference_preserves_another_real_shared_sdk_client(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    peer = factory.real(path=str(tmp_path/'index'), settings=Settings(anonymized_telemetry=False))
    try:
        async with AsyncExitStack() as stack:
            await build_memory_services(configuration(tmp_path), shutdown=stack)
        assert factory.clients[0].sdk._closed
        assert peer.get_collection(name='user_memories_policy_v1', embedding_function=None).count() == 0
    finally:
        peer.close()
        factory.cleanup()


async def test_owned_sdk_closes_only_after_actual_memory_work_and_retains_shared_authority(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    privacy = PrivacyServices(str(tmp_path/'privacy.db'), close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    stack = AsyncExitStack()
    reading = None
    try:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack, privacy=privacy)
        client = factory.clients[0]
        client.release.clear()
        def native_read():
            entered.set()
            assert release.wait(4)
            return services.manager.long_term._collection.count()
        reading = asyncio.create_task(services.manager.long_term.acall(native_read))
        await until(entered.is_set)
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)
        with pytest.raises(MemoryCloseTimeout):
            await services.aclose()
        assert not client.entered.is_set() and privacy.resource_status()['writer_lease']
        release.set()
        await until(client.entered.is_set)
        with pytest.raises(WorkCloseTimeout):
            await privacy.aclose()
        assert privacy.resource_status()['writer_lease'] and not services.resource_status()['client_closed']
    finally:
        release.set()
        for client in factory.clients:
            client.release.set()
        if reading is not None:
            await asyncio.gather(reading, return_exceptions=True)
        await privacy.begin_close()
        await stack.aclose()
        factory.cleanup()
    assert services.resource_status()['client_closed'] and not privacy.resource_status()['writer_lease']


async def test_standalone_writer_lease_outlives_the_actual_sdk_close(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    stack = AsyncExitStack()
    descriptor = None
    try:
        services = await build_memory_services(configuration(tmp_path), shutdown=stack)
        client = factory.clients[0]
        client.release.clear()
        closing = services.begin_close()
        await until(client.entered.is_set)
        descriptor = os.open(str(tmp_path/'privacy.db')+'.lock', os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not closing.done() and services.manager.long_term._lease is not None
        client.release.set()
        await closing
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for client in factory.clients:
            client.release.set()
        await stack.aclose()
        factory.cleanup()


async def test_late_constructor_remains_owned_when_build_is_cancelled_and_authority_closes(tmp_path, monkeypatch):
    factory = Factory(monkeypatch, hold=True)
    privacy = PrivacyServices(str(tmp_path/'privacy.db'), close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    stack = AsyncExitStack()
    building = asyncio.create_task(build_memory_services(configuration(tmp_path), shutdown=stack, privacy=privacy))
    closing = None
    try:
        await until(factory.entered.is_set)
        building.cancel()
        closing = privacy.begin_close()
        await asyncio.sleep(.02)
        assert not closing.done() and privacy.resource_status()['writer_lease']
    finally:
        factory.release.set()
        await asyncio.gather(building, return_exceptions=True)
        if closing is not None:
            await closing
        await stack.aclose()
        try:
            assert factory.clients[0].close_calls == 1 and factory.clients[0].sdk._closed
        finally:
            factory.cleanup()


async def test_manager_construction_failure_closes_already_created_sdk_before_propagating(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    def failed(*args, **kwargs):
        raise ValueError('Synthetic private constructor payload')
    monkeypatch.setattr('src.memory.services.MemoryManager', failed)
    try:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match='^memory_services_construction_failed$') as observed:
                await build_memory_services(configuration(tmp_path), shutdown=stack)
            assert observed.value.__context__ is None
            assert factory.clients[0].sdk._closed and factory.clients[0].close_calls == 1
    finally:
        factory.cleanup()


async def test_sdk_initialization_failure_closes_owned_client_and_removes_private_error_context(tmp_path, monkeypatch):
    factory = Factory(monkeypatch)
    def failed(*args, **kwargs):
        raise ValueError('Synthetic private collection payload')
    monkeypatch.setattr(Client, 'get_or_create_collection', failed)
    try:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match='^memory_services_initialization_failed$') as observed:
                await build_memory_services(configuration(tmp_path), shutdown=stack)
            assert observed.value.__context__ is None
            assert factory.clients[0].sdk._closed and factory.clients[0].close_calls == 1
    finally:
        factory.cleanup()


async def test_native_initialization_is_context_free_and_borrowed_sdk_is_not_closed_on_cancel(tmp_path):
    client = MemoryClient(tmp_path/'index')
    entered, release = threading.Event(), threading.Event()
    contexts = []
    original = client.get_or_create_collection
    def held(**kwargs):
        contexts.append(get_context())
        entered.set()
        assert release.wait(4)
        return original(**kwargs)
    client.get_or_create_collection = held
    stack = AsyncExitStack()
    try:
        with LogContext(private='Synthetic initialization context'):
            building = asyncio.create_task(build_memory_services(configuration(tmp_path), shutdown=stack, chroma_client=client))
        await until(entered.is_set)
        building.cancel()
        await asyncio.sleep(.02)
        assert not building.done()
    finally:
        release.set()
        await asyncio.gather(building, return_exceptions=True)
        await stack.aclose()
    assert contexts == [{}] and not client.client._closed
    client.close()


SDK_FAILURE_CHILD = '''
import asyncio,fcntl,json,os,sys
from pathlib import Path
import chromadb
from src.memory.services import build_memory_services
from src.memory.privacy_services import PrivacyServices
from contextlib import AsyncExitStack
class Client:
    calls=0
    def get_or_create_collection(self,**kwargs):return object()
    def close(self):
        self.calls+=1
        self.closed=True
        raise RuntimeError('Synthetic private SDK close payload')
async def main():
    root=Path(sys.argv[1]);client=Client()
    chromadb.PersistentClient=lambda **kwargs:client
    privacy=PrivacyServices(str(root/'privacy.db'))
    await privacy.acall(privacy.initialize)
    stack=AsyncExitStack()
    services=await build_memory_services({'memory':{'privacy_db_path':str(root/'privacy.db'),
        'chroma_path':str(root/'index'),'preferences_path':str(root/'preferences.json')}},shutdown=stack,privacy=privacy)
    failures=[]
    for closing in (services.begin_close(),privacy.begin_close()):
        try:await closing
        except RuntimeError as error:
            assert str(error)=='memory_sdk_close_failed' and error.__context__ is None
            failures.append(True)
    status=services.resource_status()
    assert len(failures)==2 and status['client_close_failed'] and not status['client_closed']
    assert privacy.resource_status()['writer_lease'] and client.calls==1
    descriptor=os.open(str(root/'privacy.db')+'.lock',os.O_RDWR)
    rejected=False
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:rejected=True
    finally:os.close(descriptor)
    assert rejected
    print(json.dumps({'passed':True,'lease_retained':True,'sdk_close_called_once':True,
        'competing_writer_rejected':True,'private_exception_context_removed':True}))
asyncio.run(main())
'''


async def test_sdk_close_failure_cannot_be_retried_as_false_physical_completion(tmp_path):
    import src.memory.services as module
    root = Path(module.__file__).resolve().parents[2]
    script = 'import sys\nsys.path.insert(0,'+repr(str(root))+')\n'+SDK_FAILURE_CHILD
    result = await asyncio.to_thread(subprocess.run, [sys.executable, '-B', '-E', '-s', '-c', script, str(tmp_path)],
                                     cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr[-2000:]
    report = next(json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith('{'))
    assert report['passed'] and report['lease_retained'] and report['sdk_close_called_once']
