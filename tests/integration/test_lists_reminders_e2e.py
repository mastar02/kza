"""Integration tests for lists and reminders end-to-end flow."""
import asyncio
import time

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from src.lists.list_store import ListStore
from src.lists.list_manager import ListManager
from src.lists.ha_sync import HASyncManager
from src.reminders.reminder_store import ReminderStore
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler


@pytest_asyncio.fixture
async def list_system(tmp_path):
    """Full list system with store, manager, and HA sync."""
    store = ListStore(str(tmp_path / "lists.db"))
    await store.initialize()
    ha_client = AsyncMock()
    ha_sync = HASyncManager(ha_client=ha_client, entity_prefix="todo.kza")
    manager = ListManager(store=store, ha_client=ha_client, config={
        "default_list_name": "compras",
        "ha_sync_enabled": True,
    })
    yield {"store": store, "manager": manager, "ha_sync": ha_sync, "ha_client": ha_client}
    await store.close()


@pytest_asyncio.fixture
async def reminder_system(tmp_path):
    """Full reminder system with store, manager, and scheduler."""
    store = ReminderStore(str(tmp_path / "reminders.db"))
    await store.initialize()
    tts = AsyncMock()
    presence = MagicMock()
    presence.get_user_zone = MagicMock(return_value="living")
    ha_client = AsyncMock()
    manager = ReminderManager(store=store, config={"tts_prefix": "Oye, recuerda"})
    scheduler = ReminderScheduler(
        store=store, tts=tts, presence_detector=presence,
        ha_client=ha_client, config={"max_retries": 2, "retry_interval_seconds": 1, "tts_prefix": "Oye, recuerda"},
    )
    yield {
        "store": store,
        "manager": manager,
        "scheduler": scheduler,
        "tts": tts,
        "ha_client": ha_client,
        "presence": presence,
    }
    if scheduler._running:
        await scheduler.stop()
    await store.close()


class TestListFlow:
    """Test complete list lifecycle."""

    @pytest.mark.asyncio
    async def test_full_list_lifecycle(self, list_system):
        mgr = list_system["manager"]

        # Create list, add items
        await mgr.add_item(user_id="u1", item_text="leche")
        await mgr.add_item(user_id="u1", item_text="pan integral")
        await mgr.add_item(user_id="u1", item_text="huevos")

        # Query items
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 3

        # Remove by fuzzy match
        removed = await mgr.remove_item(user_id="u1", item_text="pan")
        assert removed is True
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 2

        # Clear list
        await mgr.clear_list(user_id="u1")
        items = await mgr.get_items(user_id="u1")
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_shared_list_between_users(self, list_system):
        mgr = list_system["manager"]

        # User 1 creates shared list
        await mgr.create_list(user_id="u1", list_name="casa", shared=True)
        await mgr.add_item(user_id="u1", item_text="jabon", list_name="casa")

        # User 2 can see and add to it
        items = await mgr.get_items(user_id="u2", list_name="casa")
        assert len(items) == 1
        await mgr.add_item(user_id="u2", item_text="detergente", list_name="casa")
        items = await mgr.get_items(user_id="u1", list_name="casa")
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_multiple_lists_per_user(self, list_system):
        mgr = list_system["manager"]

        await mgr.create_list(user_id="u1", list_name="oficina")
        await mgr.add_item(user_id="u1", item_text="leche")  # default "compras"
        await mgr.add_item(user_id="u1", item_text="lapiz", list_name="oficina")

        compras = await mgr.get_items(user_id="u1")
        oficina = await mgr.get_items(user_id="u1", list_name="oficina")
        assert len(compras) == 1
        assert len(oficina) == 1


class TestReminderFlow:
    """Test complete reminder lifecycle."""

    @pytest.mark.asyncio
    async def test_one_shot_reminder_fires(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]
        tts = reminder_system["tts"]

        # Create reminder in the past (should fire immediately)
        r = await mgr.create(user_id="u1", text="sacar basura", trigger_at=time.time() - 10)

        # Fire it through scheduler
        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        # Verify TTS was called and reminder marked fired
        tts.speak.assert_called_once()
        assert "sacar basura" in tts.speak.call_args[0][0]
        updated = await reminder_system["store"].get_by_id(r.id)
        assert updated.state == "fired"

    @pytest.mark.asyncio
    async def test_recurring_reminder_reschedules(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]

        r = await mgr.create(user_id="u1", text="pastilla", trigger_at=time.time() - 10, recurrence="daily")
        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        updated = await reminder_system["store"].get_by_id(r.id)
        assert updated.state == "active"
        assert updated.trigger_at > time.time()  # Rescheduled to tomorrow

    @pytest.mark.asyncio
    async def test_reminder_with_ha_actions(self, reminder_system):
        mgr = reminder_system["manager"]
        scheduler = reminder_system["scheduler"]
        ha = reminder_system["ha_client"]

        actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.jardin"}]
        r = await mgr.create(user_id="u1", text="regar plantas", trigger_at=time.time() - 10, ha_actions=actions)

        reminder = await reminder_system["store"].get_by_id(r.id)
        await scheduler._fire_reminder(reminder)

        ha.call_service.assert_called_once_with("light", "turn_on", "light.jardin", {})

    @pytest.mark.asyncio
    async def test_cancel_reminder_by_fuzzy_text(self, reminder_system):
        mgr = reminder_system["manager"]

        await mgr.create(user_id="u1", text="sacar la basura", trigger_at=time.time() + 3600)
        await mgr.create(user_id="u1", text="ir al dentista", trigger_at=time.time() + 7200)

        result = await mgr.cancel_by_text(user_id="u1", search_text="basura")
        assert result is True

        active = await mgr.get_active(user_id="u1")
        assert len(active) == 1
        assert active[0].text == "ir al dentista"
