"""Home Assistant todo platform sync for KZA lists."""

import logging

logger = logging.getLogger(__name__)


class HASyncManager:
    """Syncs KZA lists with Home Assistant todo platform entities.

    Each KZA list becomes a todo.kza_<name> entity in HA.
    KZA is the source of truth; HA is a mirror.
    """

    def __init__(self, ha_client, entity_prefix: str = "todo.kza"):
        self._ha = ha_client
        self._prefix = entity_prefix
        self._enabled = ha_client is not None

    def _entity_id(self, list_name: str) -> str:
        """Generate HA entity ID from list name."""
        safe_name = list_name.lower().replace(" ", "_").replace("-", "_")
        return f"{self._prefix}_{safe_name}"

    async def sync_add_item(self, list_name: str, item_text: str) -> bool:
        """Sync an added item to HA todo entity."""
        if not self._enabled:
            return False
        entity = self._entity_id(list_name)
        try:
            await self._ha.call_service("todo", "add_item", entity, {"item": item_text})
            logger.debug(f"HA sync: added '{item_text}' to {entity}")
            return True
        except Exception as e:
            logger.warning(f"HA sync failed for add_item: {e}")
            return False

    async def sync_remove_item(self, list_name: str, item_text: str) -> bool:
        """Sync a removed item to HA todo entity."""
        if not self._enabled:
            return False
        entity = self._entity_id(list_name)
        try:
            await self._ha.call_service("todo", "remove_item", entity, {"item": item_text})
            logger.debug(f"HA sync: removed '{item_text}' from {entity}")
            return True
        except Exception as e:
            logger.warning(f"HA sync failed for remove_item: {e}")
            return False

    async def sync_complete_item(self, list_name: str, item_text: str) -> bool:
        """Sync a completed item to HA todo entity."""
        if not self._enabled:
            return False
        entity = self._entity_id(list_name)
        try:
            await self._ha.call_service("todo", "update_item", entity, {"item": item_text, "status": "completed"})
            logger.debug(f"HA sync: completed '{item_text}' in {entity}")
            return True
        except Exception as e:
            logger.warning(f"HA sync failed for complete_item: {e}")
            return False
