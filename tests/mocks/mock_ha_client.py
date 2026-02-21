"""
Mock Home Assistant Client for testing.
"""

from typing import Optional
from unittest.mock import AsyncMock


class MockHomeAssistantClient:
    """Mock implementation of HomeAssistantClient for testing"""

    def __init__(self):
        self.url = "http://localhost:8123"
        self.token = "test_token"
        self._entities = self._default_entities()
        self._services_called = []
        self._ws_connected = False

    def _default_entities(self) -> list[dict]:
        return [
            {
                "entity_id": "light.living_room",
                "state": "off",
                "attributes": {
                    "friendly_name": "Luz del Living",
                    "brightness": 0,
                    "supported_features": 1
                }
            },
            {
                "entity_id": "light.bedroom",
                "state": "on",
                "attributes": {
                    "friendly_name": "Luz del Dormitorio",
                    "brightness": 255
                }
            },
            {
                "entity_id": "climate.living_room",
                "state": "cool",
                "attributes": {
                    "friendly_name": "Aire del Living",
                    "temperature": 22,
                    "current_temperature": 24,
                    "hvac_modes": ["off", "cool", "heat", "auto"]
                }
            },
            {
                "entity_id": "cover.blinds",
                "state": "closed",
                "attributes": {
                    "friendly_name": "Persianas",
                    "current_position": 0
                }
            },
            {
                "entity_id": "switch.tv",
                "state": "off",
                "attributes": {
                    "friendly_name": "Televisor"
                }
            }
        ]

    def get_all_entities(self) -> list[dict]:
        return self._entities

    def get_all_services(self) -> list[dict]:
        return [
            {
                "domain": "light",
                "services": {
                    "turn_on": {"description": "Turn on light"},
                    "turn_off": {"description": "Turn off light"},
                    "toggle": {"description": "Toggle light"}
                }
            },
            {
                "domain": "climate",
                "services": {
                    "set_temperature": {"description": "Set temperature"},
                    "turn_on": {"description": "Turn on"},
                    "turn_off": {"description": "Turn off"}
                }
            },
            {
                "domain": "cover",
                "services": {
                    "open_cover": {"description": "Open cover"},
                    "close_cover": {"description": "Close cover"},
                    "set_cover_position": {"description": "Set position"}
                }
            }
        ]

    def get_entity_state(self, entity_id: str) -> Optional[dict]:
        for entity in self._entities:
            if entity["entity_id"] == entity_id:
                return entity
        return None

    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[dict] = None
    ) -> bool:
        self._services_called.append({
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "data": data
        })

        # Simulate state change
        for entity in self._entities:
            if entity["entity_id"] == entity_id:
                if service == "turn_on":
                    entity["state"] = "on"
                elif service == "turn_off":
                    entity["state"] = "off"
                elif service == "set_temperature" and data:
                    entity["attributes"]["temperature"] = data.get("temperature")
                return True

        return False

    def get_domotics_entities(self) -> list[dict]:
        domotics_domains = ["light", "switch", "cover", "climate"]
        return [
            e for e in self._entities
            if e["entity_id"].split(".")[0] in domotics_domains
        ]

    def test_connection(self) -> bool:
        return True

    async def connect_websocket(self) -> bool:
        self._ws_connected = True
        return True

    async def call_service_ws(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: Optional[dict] = None
    ) -> bool:
        return self.call_service(domain, service, entity_id, data)

    async def close(self):
        self._ws_connected = False

    # Test helpers
    def get_services_called(self) -> list[dict]:
        return self._services_called

    def reset_services_called(self):
        self._services_called = []

    def set_entity_state(self, entity_id: str, state: str):
        for entity in self._entities:
            if entity["entity_id"] == entity_id:
                entity["state"] = state
                break
