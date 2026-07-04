"""Tests del servicio HTTP del code-index."""

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.code_index.service import create_app
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
    idx = client.server.app["indexer"]
    await idx._lock.acquire()
    try:
        resp = await client.post("/reindex")
        assert resp.status == 409
    finally:
        idx._lock.release()
