"""Tests for ListStore — SQLite persistence for lists and items."""
import pytest
import pytest_asyncio

from src.lists.list_store import ListStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ListStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_list(store):
    lst = await store.create_list("compras", "user", "user_1")
    assert lst.name == "compras"
    assert lst.owner_type == "user"
    assert lst.owner_id == "user_1"
    assert lst.id


@pytest.mark.asyncio
async def test_clearing_empty_existing_list_completes_but_absent_list_does_not(store):
    lst = await store.create_list("compras", "user", "user_1")
    assert await store.clear_list(lst.id) is True
    assert await store.delete_list(lst.id) is True
    assert await store.clear_list(lst.id) is False


@pytest.mark.asyncio
async def test_create_shared_list(store):
    lst = await store.create_list("hogar", "shared", None)
    assert lst.owner_type == "shared"
    assert lst.owner_id is None


@pytest.mark.asyncio
async def test_get_lists_by_user(store):
    await store.create_list("mia", "user", "u1")
    await store.create_list("compartida", "shared", None)
    await store.create_list("otra", "user", "u2")
    lists = await store.get_lists_for_user("u1")
    names = [l.name for l in lists]
    assert "mia" in names
    assert "compartida" in names
    assert "otra" not in names


@pytest.mark.asyncio
async def test_add_and_get_items(store):
    lst = await store.create_list("compras", "user", "u1")
    item = await store.add_item(lst.id, "leche", added_by="u1")
    assert item.text == "leche"
    items = await store.get_items(lst.id)
    assert len(items) == 1
    assert items[0].text == "leche"


@pytest.mark.asyncio
async def test_remove_item(store):
    lst = await store.create_list("compras", "user", "u1")
    item = await store.add_item(lst.id, "pan")
    await store.remove_item(item.id)
    items = await store.get_items(lst.id)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_complete_item(store):
    lst = await store.create_list("tareas", "user", "u1")
    item = await store.add_item(lst.id, "lavar")
    updated = await store.complete_item(item.id)
    assert updated.completed is True
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_clear_list(store):
    lst = await store.create_list("compras", "user", "u1")
    await store.add_item(lst.id, "a")
    await store.add_item(lst.id, "b")
    await store.clear_list(lst.id)
    items = await store.get_items(lst.id)
    assert len(items) == 0


@pytest.mark.asyncio
async def test_delete_list(store):
    lst = await store.create_list("temp", "user", "u1")
    await store.add_item(lst.id, "x")
    await store.delete_list(lst.id)
    lists = await store.get_lists_for_user("u1")
    assert len(lists) == 0


@pytest.mark.asyncio
async def test_find_list_by_name(store):
    await store.create_list("compras", "user", "u1")
    found = await store.find_list_by_name("compras", "u1")
    assert found is not None
    assert found.name == "compras"


@pytest.mark.asyncio
async def test_find_list_by_name_shared(store):
    await store.create_list("casa", "shared", None)
    found = await store.find_list_by_name("casa", "u1")
    assert found is not None


@pytest.mark.asyncio
async def test_owner_scope_admits_shared_and_own_lists_only(store):
    """T035-U1: a person's session reaches shared lists and her own, never another person's."""
    from src.lists.list_store import OwnerType
    shared = await store.create_list("compras", OwnerType.SHARED, None)
    own = await store.create_list("mia", OwnerType.USER, "ana")
    other = await store.create_list("ajena", OwnerType.USER, "beto")
    foreign = await store.add_item(other.id, "secreto", "beto")
    assert [i.text for i in await store.get_items_in_scope(own.id, owner="ana")] == []
    assert await store.get_items_in_scope(other.id, owner="ana") is None
    assert await store.get_items_in_scope(own.id, shared_only=True) is None
    assert await store.get_items_in_scope(shared.id, shared_only=True) == []
    assert await store.get_items_in_scope("missing") is None
    assert await store.add_item(other.id, "intruso", "ana", owner="ana") is None
    assert (await store.add_item(shared.id, "pan", "ana", owner="ana")).list_id == shared.id
    assert await store.remove_item(foreign.id, list_id=other.id, owner="ana") is False
    assert await store.remove_item(foreign.id, owner="ana") is False
    assert [i.text for i in await store.get_items(other.id)] == ["secreto"]
    assert await store.remove_item(foreign.id, list_id=other.id) is True
