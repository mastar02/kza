"""Tests for ListManager — business logic for list operations."""
import pytest
import pytest_asyncio

from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, UserList, ListItem


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ListStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def manager(store):
    config = {"default_list_name": "compras", "ha_sync_enabled": False}
    return ListManager(store=store, ha_client=None, config=config)


@pytest.mark.asyncio
async def test_add_item_creates_default_list(manager):
    result = await manager.add_item(user_id="u1", item_text="leche")
    assert result.text == "leche"

@pytest.mark.asyncio
async def test_add_item_to_named_list(manager):
    await manager.create_list(user_id="u1", list_name="oficina")
    result = await manager.add_item(user_id="u1", item_text="lapiz", list_name="oficina")
    assert result.text == "lapiz"

@pytest.mark.asyncio
async def test_remove_item_fuzzy_match(manager):
    await manager.add_item(user_id="u1", item_text="leche descremada")
    removed = await manager.remove_item(user_id="u1", item_text="leche")
    assert removed is True

@pytest.mark.asyncio
async def test_get_items_returns_list(manager):
    await manager.add_item(user_id="u1", item_text="pan")
    await manager.add_item(user_id="u1", item_text="huevos")
    items = await manager.get_items(user_id="u1")
    assert len(items) == 2

@pytest.mark.asyncio
async def test_clear_list(manager):
    await manager.add_item(user_id="u1", item_text="a")
    await manager.add_item(user_id="u1", item_text="b")
    await manager.clear_list(user_id="u1")
    items = await manager.get_items(user_id="u1")
    assert len(items) == 0

@pytest.mark.asyncio
async def test_create_shared_list(manager):
    lst = await manager.create_list(user_id="u1", list_name="casa", shared=True)
    assert lst.owner_type == "shared"

@pytest.mark.asyncio
async def test_shared_list_visible_to_other_user(manager):
    await manager.create_list(user_id="u1", list_name="casa", shared=True)
    await manager.add_item(user_id="u1", item_text="jabon", list_name="casa")
    items = await manager.get_items(user_id="u2", list_name="casa")
    assert len(items) == 1

@pytest.mark.asyncio
async def test_delete_list(manager):
    await manager.create_list(user_id="u1", list_name="temp")
    result = await manager.delete_list(user_id="u1", list_name="temp")
    assert result is True

@pytest.mark.asyncio
async def test_get_all_lists(manager):
    await manager.create_list(user_id="u1", list_name="a")
    await manager.create_list(user_id="u1", list_name="b")
    lists = await manager.get_all_lists(user_id="u1")
    assert len(lists) >= 2
