"""Tests for HA todo platform sync."""
import pytest
from unittest.mock import AsyncMock

from src.lists.ha_sync import HASyncManager


@pytest.fixture
def ha_client():
    return AsyncMock()


@pytest.fixture
def sync_manager(ha_client):
    return HASyncManager(ha_client=ha_client, entity_prefix="todo.kza")


class TestEntityId:
    def test_simple_name(self, sync_manager):
        assert sync_manager._entity_id("compras") == "todo.kza_compras"

    def test_name_with_spaces(self, sync_manager):
        assert sync_manager._entity_id("lista del hogar") == "todo.kza_lista_del_hogar"

    def test_name_with_uppercase(self, sync_manager):
        assert sync_manager._entity_id("Oficina") == "todo.kza_oficina"


@pytest.mark.asyncio
async def test_sync_add_item(sync_manager, ha_client):
    result = await sync_manager.sync_add_item("compras", "leche")
    assert result is True
    ha_client.call_service.assert_called_once_with("todo", "add_item", "todo.kza_compras", {"item": "leche"})


@pytest.mark.asyncio
async def test_sync_remove_item(sync_manager, ha_client):
    result = await sync_manager.sync_remove_item("compras", "pan")
    assert result is True
    ha_client.call_service.assert_called_once_with("todo", "remove_item", "todo.kza_compras", {"item": "pan"})


@pytest.mark.asyncio
async def test_sync_complete_item(sync_manager, ha_client):
    result = await sync_manager.sync_complete_item("tareas", "lavar")
    assert result is True
    ha_client.call_service.assert_called_once_with("todo", "update_item", "todo.kza_tareas", {"item": "lavar", "status": "completed"})


@pytest.mark.asyncio
async def test_sync_disabled_when_no_client():
    mgr = HASyncManager(ha_client=None)
    result = await mgr.sync_add_item("compras", "leche")
    assert result is False


@pytest.mark.asyncio
async def test_sync_handles_error(sync_manager, ha_client):
    ha_client.call_service.side_effect = Exception("HA unavailable")
    result = await sync_manager.sync_add_item("compras", "leche")
    assert result is False  # Fails gracefully, doesn't raise
