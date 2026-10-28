"""Health clients coexist with the real HA index and close each SDK reference."""

import asyncio

import pytest

from src.code_index.service import build_chroma_client
from src.health.health_check import HealthChecker, HealthStatus, quick_check
from tests.integration import test_chroma_sync_lifetime as sdk_fixture

graph = sdk_fixture.graph
seed = sdk_fixture.seed

pytestmark = pytest.mark.asyncio


async def check(owner, monkeypatch, kind):
    if kind == 'full':
        checker = HealthChecker({'vector_db':{'path':owner.chroma_path}})
        result = await asyncio.to_thread(checker.check_chromadb)
        return result.status is HealthStatus.HEALTHY
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('CHROMA_PATH', owner.chroma_path)
    return await asyncio.to_thread(quick_check)


@pytest.mark.parametrize('kind', ['full', 'quick'])
async def test_repeated_health_checks_close_their_own_handle_and_keep_main_index_readable(graph, monkeypatch, kind):
    owner, clients, _ = graph
    await seed(owner)
    for _ in range(3):
        assert await check(owner, monkeypatch, kind)
        assert not clients[0].sdk._closed
        assert await asyncio.to_thread(owner.commands.count) == 1
    assert len(clients) == 4
    assert all(client.sdk._closed and client.close_calls == 1 for client in clients[1:])


async def test_failed_collection_listing_still_closes_health_client(graph, monkeypatch):
    owner, clients, _ = graph
    await seed(owner)
    from tests.integration.test_chroma_sync_lifetime import Client
    def failed(self):
        raise RuntimeError('synthetic_list_failure')
    monkeypatch.setattr(Client, 'list_collections', failed, raising=False)
    assert not await check(owner, monkeypatch, 'full')
    assert clients[1].sdk._closed and clients[1].close_calls == 1
    assert not clients[0].sdk._closed and await asyncio.to_thread(owner.commands.count) == 1


@pytest.mark.parametrize('kind', ['full', 'quick'])
async def test_failed_health_sdk_close_never_returns_healthy(graph, monkeypatch, kind):
    owner, clients, _ = graph
    await seed(owner)
    from tests.integration.test_chroma_sync_lifetime import Client
    original = Client.close
    def failed(self):
        original(self)
        raise RuntimeError('synthetic_close_failure')
    # The failure belongs to only the next health client's close. Keep the main
    # owner's real cleanup available to the fixture.
    with monkeypatch.context() as patch:
        patch.setattr(Client, 'close', failed)
        assert not await check(owner, monkeypatch, kind)
    assert clients[1].sdk._closed and clients[1].close_calls == 1
    assert not clients[0].sdk._closed


async def test_code_index_embedded_factory_can_share_private_settings_without_resetting_peer(graph):
    owner, clients, _ = graph
    await seed(owner)
    peer = await asyncio.to_thread(build_chroma_client, {'chroma_path':owner.chroma_path})
    try:
        assert len(clients) == 2 and not clients[0].sdk._closed
        assert await asyncio.to_thread(peer.sdk.get_collection('home_assistant_commands', embedding_function=None).count) == 1
    finally:
        # This factory still hands ownership to its caller. The independent code
        # index lifetime remains pending; the test cannot claim that integration.
        await asyncio.to_thread(peer.close)
    assert not clients[0].sdk._closed and await asyncio.to_thread(owner.commands.count) == 1
