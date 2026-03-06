"""Tests for list and reminder REST API endpoints."""
import time
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.dashboard.api import DashboardAPI
from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager


@pytest_asyncio.fixture
async def api(tmp_path):
    list_store = ListStore(str(tmp_path / "test.db"))
    await list_store.initialize()
    list_mgr = ListManager(store=list_store, ha_client=None, config={"default_list_name": "compras", "ha_sync_enabled": False})

    reminder_store = ReminderStore(str(tmp_path / "test_r.db"))
    await reminder_store.initialize()
    reminder_mgr = ReminderManager(store=reminder_store, config={})

    dashboard = DashboardAPI(list_manager=list_mgr, reminder_manager=reminder_mgr)
    yield dashboard
    await list_store.close()
    await reminder_store.close()


@pytest_asyncio.fixture
async def client(api):
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_create_list(client):
    resp = await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "compras"

@pytest.mark.asyncio
async def test_get_lists(client):
    await client.post("/api/lists", json={"name": "a", "user_id": "u1"})
    resp = await client.get("/api/lists", params={"user_id": "u1"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

@pytest.mark.asyncio
async def test_add_item(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    resp = await client.post(f"/api/lists/{lst['id']}/items", json={"text": "leche", "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["text"] == "leche"

@pytest.mark.asyncio
async def test_get_items(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    await client.post(f"/api/lists/{lst['id']}/items", json={"text": "pan", "user_id": "u1"})
    resp = await client.get(f"/api/lists/{lst['id']}/items")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

@pytest.mark.asyncio
async def test_delete_item(client):
    lst = (await client.post("/api/lists", json={"name": "compras", "user_id": "u1"})).json()
    item = (await client.post(f"/api/lists/{lst['id']}/items", json={"text": "pan", "user_id": "u1"})).json()
    resp = await client.delete(f"/api/lists/{lst['id']}/items/{item['id']}")
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_create_reminder(client):
    resp = await client.post("/api/reminders", json={"user_id": "u1", "text": "dentista", "trigger_at": time.time() + 3600})
    assert resp.status_code == 200
    assert resp.json()["text"] == "dentista"

@pytest.mark.asyncio
async def test_get_reminders(client):
    await client.post("/api/reminders", json={"user_id": "u1", "text": "a", "trigger_at": time.time() + 100})
    resp = await client.get("/api/reminders", params={"user_id": "u1"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

@pytest.mark.asyncio
async def test_delete_reminder(client):
    r = (await client.post("/api/reminders", json={"user_id": "u1", "text": "x", "trigger_at": time.time() + 100})).json()
    resp = await client.delete(f"/api/reminders/{r['id']}")
    assert resp.status_code == 200
