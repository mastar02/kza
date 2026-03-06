"""Tests for ReminderScheduler — asyncio scheduling and delivery."""
import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest.fixture
def mock_deps():
    return {
        "tts": AsyncMock(),
        "presence_detector": MagicMock(),
        "ha_client": AsyncMock(),
    }

@pytest_asyncio.fixture
async def scheduler(store, mock_deps):
    config = {
        "max_retries": 3,
        "retry_interval_seconds": 1,
        "missed_reminder_on_arrival": True,
        "tts_prefix": "Oye, recuerda",
    }
    s = ReminderScheduler(
        store=store,
        tts=mock_deps["tts"],
        presence_detector=mock_deps["presence_detector"],
        ha_client=mock_deps["ha_client"],
        config=config,
    )
    yield s
    if s._running:
        await s.stop()

@pytest.mark.asyncio
async def test_fire_reminder_with_tts(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    r = await store.create(user_id="u1", text="basura", trigger_at=time.time() - 10)
    await scheduler._fire_reminder(r)
    mock_deps["tts"].speak.assert_called_once()
    updated = await store.get_by_id(r.id)
    assert updated.state == "fired"

@pytest.mark.asyncio
async def test_fire_reminder_with_ha_actions(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await store.create(user_id="u1", text="regar", trigger_at=time.time() - 10, ha_actions=actions)
    await scheduler._fire_reminder(r)
    mock_deps["ha_client"].call_service.assert_called_once_with("light", "turn_on", "light.patio", {})

@pytest.mark.asyncio
async def test_fire_recurring_reschedules(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value="living")
    r = await store.create(user_id="u1", text="pastilla", trigger_at=time.time() - 10, recurrence="daily")
    await scheduler._fire_reminder(r)
    updated = await store.get_by_id(r.id)
    assert updated.state == "active"
    assert updated.trigger_at > time.time()

@pytest.mark.asyncio
async def test_fire_user_not_present_queues_retry(scheduler, store, mock_deps):
    mock_deps["presence_detector"].get_user_zone = MagicMock(return_value=None)
    r = await store.create(user_id="u1", text="x", trigger_at=time.time() - 10)
    await scheduler._fire_reminder(r)
    updated = await store.get_by_id(r.id)
    assert updated.state == "active"  # NOT fired, retrying

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler, store):
    await store.create(user_id="u1", text="test", trigger_at=time.time() + 10000)
    task = asyncio.create_task(scheduler.start())
    await asyncio.sleep(0.1)
    assert scheduler._running is True
    await scheduler.stop()
    assert scheduler._running is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
