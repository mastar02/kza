"""
Tests for Home Assistant Client.
"""

import pytest
from tests.mocks.mock_ha_client import MockHomeAssistantClient


class TestHomeAssistantClient:
    """Test suite for HA client functionality"""

    @pytest.mark.asyncio
    async def test_get_all_entities(self, mock_ha_client):
        """Test retrieving all entities"""
        entities = await mock_ha_client.get_all_entities()

        assert len(entities) > 0
        assert all("entity_id" in e for e in entities)
        assert all("state" in e for e in entities)

    @pytest.mark.asyncio
    async def test_get_domotics_entities(self, mock_ha_client):
        """Test filtering for domotics entities"""
        entities = await mock_ha_client.get_domotics_entities()

        # Should only return light, switch, cover, climate
        domains = [e["entity_id"].split(".")[0] for e in entities]
        allowed_domains = {"light", "switch", "cover", "climate"}
        assert all(d in allowed_domains for d in domains)

    @pytest.mark.asyncio
    async def test_get_entity_state(self, mock_ha_client):
        """Test getting specific entity state"""
        state = await mock_ha_client.get_entity_state("light.living_room")

        assert state is not None
        assert state["entity_id"] == "light.living_room"
        assert "state" in state

    @pytest.mark.asyncio
    async def test_get_entity_state_not_found(self, mock_ha_client):
        """Test getting non-existent entity"""
        state = await mock_ha_client.get_entity_state("light.nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_call_service_turn_on(self, mock_ha_client):
        """Test calling turn_on service"""
        # Verify initial state
        state = await mock_ha_client.get_entity_state("light.living_room")
        assert state["state"] == "off"

        # Call service
        result = await mock_ha_client.call_service(
            domain="light",
            service="turn_on",
            entity_id="light.living_room"
        )

        assert result is True

        # Verify state changed
        state = await mock_ha_client.get_entity_state("light.living_room")
        assert state["state"] == "on"

    @pytest.mark.asyncio
    async def test_call_service_with_data(self, mock_ha_client):
        """Test calling service with additional data"""
        result = await mock_ha_client.call_service(
            domain="climate",
            service="set_temperature",
            entity_id="climate.living_room",
            data={"temperature": 24}
        )

        assert result is True

        state = await mock_ha_client.get_entity_state("climate.living_room")
        assert state["attributes"]["temperature"] == 24

    @pytest.mark.asyncio
    async def test_call_service_records_history(self, mock_ha_client):
        """Test that service calls are recorded"""
        mock_ha_client.reset_services_called()

        await mock_ha_client.call_service("light", "turn_on", "light.living_room")
        await mock_ha_client.call_service("climate", "turn_off", "climate.living_room")

        history = mock_ha_client.get_services_called()
        assert len(history) == 2
        assert history[0]["domain"] == "light"
        assert history[1]["domain"] == "climate"

    @pytest.mark.asyncio
    async def test_connection(self, mock_ha_client):
        """Test connection check"""
        assert await mock_ha_client.test_connection() is True

    @pytest.mark.asyncio
    async def test_websocket_connect(self, mock_ha_client):
        """Test WebSocket connection"""
        result = await mock_ha_client.connect_websocket()
        assert result is True
        assert mock_ha_client._ws_connected is True

    @pytest.mark.asyncio
    async def test_websocket_call_service(self, mock_ha_client):
        """Test calling service via WebSocket"""
        await mock_ha_client.connect_websocket()

        result = await mock_ha_client.call_service_ws(
            domain="light",
            service="turn_on",
            entity_id="light.living_room"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_websocket_close(self, mock_ha_client):
        """Test WebSocket cleanup"""
        await mock_ha_client.connect_websocket()
        assert mock_ha_client._ws_connected is True

        await mock_ha_client.close()
        assert mock_ha_client._ws_connected is False
