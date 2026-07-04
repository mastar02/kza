"""Tests del servicio HTTP del code-index."""

import asyncio
import contextlib

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.code_index.service import INDEXER_KEY, REINDEX_STATE_KEY, create_app
from tests.unit.code_index.test_indexer import make_indexer, repo  # noqa: F401


@pytest.fixture
async def client(repo):  # noqa: F811
    idx, *_ = make_indexer(repo)
    app = create_app(idx)
    c = TestClient(TestServer(app))
    await c.start_server()
    yield c
    await c.close()


async def test_health(client):
    resp = await client.post("/reindex")
    assert resp.status == 202

    # esperar a que el reindex en background termine
    for _ in range(50):
        h = await (await client.get("/health")).json()
        if not h["reindex_running"] and h["files"] == 2:
            break
        await asyncio.sleep(0.05)
    assert h["status"] == "ok"
    assert h["files"] == 2
    assert h["last_stats"]["indexed"] == 2


async def test_search_endpoint(client):
    await client.post("/reindex")
    for _ in range(50):
        h = await (await client.get("/health")).json()
        if h["files"] == 2:
            break
        await asyncio.sleep(0.05)

    resp = await client.post("/search", json={"query": "funcion a", "top_k": 1})
    assert resp.status == 200
    data = await resp.json()
    assert len(data["results"]) == 1
    assert "path" in data["results"][0]


async def test_search_empty_query_is_400(client):
    resp = await client.post("/search", json={"query": "  "})
    assert resp.status == 400


async def test_reindex_returns_409_when_running(client, repo):  # noqa: F811
    # bloquear el lock del indexer manualmente para simular reindex en curso
    idx = client.server.app[INDEXER_KEY]
    await idx._lock.acquire()
    try:
        resp = await client.post("/reindex")
        assert resp.status == 409
    finally:
        idx._lock.release()


async def test_search_invalid_body_is_400(client):
    resp = await client.post("/search", data=b"esto no es json")
    assert resp.status == 400
    resp = await client.post("/search")
    assert resp.status == 400


async def test_search_invalid_top_k_is_400(client):
    resp = await client.post("/search", json={"query": "x", "top_k": "muchos"})
    assert resp.status == 400


async def test_reindex_second_request_409_while_first_pending(client):
    idx = client.server.app[INDEXER_KEY]
    release = asyncio.Event()

    async def slow_reindex(mode="incremental"):
        await release.wait()
        return {"indexed": 0, "deleted": 0, "cards_failed": 0, "errors": 0}

    idx.reindex = slow_reindex
    r1 = await client.post("/reindex")
    r2 = await client.post("/reindex")
    assert r1.status == 202
    assert r2.status == 409
    release.set()


async def test_reindex_failure_is_logged(client, caplog):
    import logging as _logging

    idx = client.server.app[INDEXER_KEY]

    async def boom(mode="incremental"):
        raise RuntimeError("scan explotó")

    idx.reindex = boom
    with caplog.at_level(_logging.ERROR):
        await client.post("/reindex")
        task = client.server.app[REINDEX_STATE_KEY].task
        with contextlib.suppress(RuntimeError):
            await task
        await asyncio.sleep(0)  # dejar correr el done_callback
    assert "Reindex falló" in caplog.text
