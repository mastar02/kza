"""Tests for ReminderStore — SQLite persistence for reminders."""
import time
import pytest
import pytest_asyncio

from src.reminders.reminder_store import ReminderStore, Reminder


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_reminder(store):
    r = await store.create(user_id="u1", text="sacar basura", trigger_at=time.time() + 3600)
    assert r.text == "sacar basura"
    assert r.state == "active"
    assert r.user_id == "u1"

@pytest.mark.asyncio
async def test_create_recurring_reminder(store):
    r = await store.create(user_id="u1", text="pastilla", trigger_at=time.time() + 3600, recurrence="daily")
    assert r.recurrence == "daily"

@pytest.mark.asyncio
async def test_create_reminder_with_ha_actions(store):
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await store.create(user_id="u1", text="regar", trigger_at=time.time() + 3600, ha_actions=actions)
    assert r.ha_actions == actions

@pytest.mark.asyncio
async def test_get_active_reminders(store):
    await store.create(user_id="u1", text="a", trigger_at=time.time() + 100)
    await store.create(user_id="u1", text="b", trigger_at=time.time() + 200)
    await store.create(user_id="u2", text="c", trigger_at=time.time() + 300)
    reminders = await store.get_active_for_user("u1")
    assert len(reminders) == 2

@pytest.mark.asyncio
async def test_get_next_pending(store):
    now = time.time()
    await store.create(user_id="u1", text="later", trigger_at=now + 1000)
    await store.create(user_id="u1", text="soon", trigger_at=now + 10)
    nxt = await store.get_next_pending()
    assert nxt.text == "soon"

@pytest.mark.asyncio
async def test_mark_fired(store):
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() - 10)
    await store.mark_fired(r.id)
    updated = await store.get_by_id(r.id)
    assert updated.state == "fired"
    assert updated.fire_count == 1

@pytest.mark.asyncio
async def test_cancel_reminder(store):
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() + 100)
    await store.cancel(r.id)
    updated = await store.get_by_id(r.id)
    assert updated.state == "cancelled"

@pytest.mark.asyncio
async def test_update_trigger(store):
    r = await store.create(user_id="u1", text="x", trigger_at=1000.0)
    await store.update_trigger(r.id, 2000.0)
    updated = await store.get_by_id(r.id)
    assert updated.trigger_at == 2000.0

@pytest.mark.asyncio
async def test_get_due_reminders(store):
    now = time.time()
    await store.create(user_id="u1", text="past", trigger_at=now - 60)
    await store.create(user_id="u1", text="future", trigger_at=now + 3600)
    due = await store.get_due(now)
    assert len(due) == 1
    assert due[0].text == "past"

@pytest.mark.asyncio
async def test_cleanup_old_fired(store):
    old_time = time.time() - (31 * 86400)
    r = await store.create(user_id="u1", text="old", trigger_at=old_time)
    await store.mark_fired(r.id)
    cleaned = await store.cleanup_old(days=30)
    assert cleaned >= 1
