"""Tests for ReminderManager — CRUD and cancellation logic."""
import time
import pytest
import pytest_asyncio

from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest_asyncio.fixture
async def manager(store):
    config = {"default_time": "09:00", "tts_prefix": "Oye, recuerda"}
    return ReminderManager(store=store, config=config)

@pytest.mark.asyncio
async def test_create_one_shot(manager):
    r = await manager.create(user_id="u1", text="dentista", trigger_at=time.time() + 3600)
    assert r.text == "dentista"
    assert r.recurrence is None

@pytest.mark.asyncio
async def test_create_recurring(manager):
    r = await manager.create(user_id="u1", text="pastilla", trigger_at=time.time() + 3600, recurrence="daily")
    assert r.recurrence == "daily"

@pytest.mark.asyncio
async def test_create_with_ha_actions(manager):
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await manager.create(user_id="u1", text="regar", trigger_at=time.time() + 3600, ha_actions=actions)
    assert len(r.ha_actions) == 1

@pytest.mark.asyncio
async def test_cancel_by_text(manager):
    await manager.create(user_id="u1", text="sacar la basura", trigger_at=time.time() + 3600)
    result = await manager.cancel_by_text(user_id="u1", search_text="basura")
    assert result is True

@pytest.mark.asyncio
async def test_cancel_nonexistent(manager):
    result = await manager.cancel_by_text(user_id="u1", search_text="inexistente")
    assert result is False

@pytest.mark.asyncio
async def test_get_active(manager):
    await manager.create(user_id="u1", text="a", trigger_at=time.time() + 100)
    await manager.create(user_id="u1", text="b", trigger_at=time.time() + 200)
    active = await manager.get_active(user_id="u1")
    assert len(active) == 2

@pytest.mark.asyncio
async def test_get_today(manager):
    now = time.time()
    await manager.create(user_id="u1", text="hoy", trigger_at=now + 100)
    await manager.create(user_id="u1", text="manana", trigger_at=now + 90000)
    today = await manager.get_today(user_id="u1")
    assert len(today) == 1
    assert today[0].text == "hoy"

@pytest.mark.asyncio
async def test_format_for_voice(manager):
    r = await manager.create(user_id="u1", text="tomar agua", trigger_at=time.time() + 3600)
    voice = manager.format_for_voice(r)
    assert "tomar agua" in voice
