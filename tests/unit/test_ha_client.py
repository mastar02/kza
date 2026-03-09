"""
Tests for Home Assistant Client.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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


# ==================== BL-007 Regression Tests ====================


class TestHAUnavailable:
    """Verify graceful degradation when HA is unreachable."""

    @pytest.mark.asyncio
    async def test_ha_unavailable_call_service_returns_false(self):
        """HAClient.call_service returns False (not an exception) when HA is down."""
        import aiohttp

        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://127.0.0.1:19999",  # nothing listening
            token="fake_token",
            timeout=0.5,
        )

        # call_service must NOT raise; it returns False
        result = await client.call_service("light", "turn_on", "light.test")
        assert result is False

        await client.close()

    @pytest.mark.asyncio
    async def test_ha_unavailable_get_entities_returns_empty(self):
        """get_all_entities returns [] when HA is unreachable."""
        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://127.0.0.1:19999",
            token="fake_token",
            timeout=0.5,
        )

        entities = await client.get_all_entities()
        assert entities == []

        await client.close()

    @pytest.mark.asyncio
    async def test_ha_unavailable_get_entity_state_returns_none(self):
        """get_entity_state returns None when HA is unreachable."""
        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://127.0.0.1:19999",
            token="fake_token",
            timeout=0.5,
        )

        state = await client.get_entity_state("light.test")
        assert state is None

        await client.close()

    @pytest.mark.asyncio
    async def test_ha_unavailable_health_shows_disconnected(self):
        """Health status reflects disconnected state after a failed call."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://127.0.0.1:19999",
            token="fake_token",
            timeout=0.5,
        )

        await client.call_service("light", "turn_on", "light.test")
        health = client.get_health_status()

        assert health.state == HAConnectionState.DISCONNECTED
        assert health.error_count >= 1
        assert health.success_count == 0
        assert health.last_error_message != ""

        await client.close()


class _TimeoutCtx:
    """Helper async context manager that raises TimeoutError on enter."""

    async def __aenter__(self):
        raise asyncio.TimeoutError()

    async def __aexit__(self, *args):
        return False


class TestHASlowResponse:
    """Verify timeout handling for slow HA responses."""

    @pytest.mark.asyncio
    async def test_ha_slow_response_call_service_timeout(self):
        """call_service returns False on timeout without raising."""
        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
            timeout=0.5,
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_TimeoutCtx())
        mock_session.closed = False

        client._session = mock_session

        result = await client.call_service("light", "turn_on", "light.test")
        assert result is False

        # Error count should have increased
        health = client.get_health_status()
        assert health.error_count >= 1

    @pytest.mark.asyncio
    async def test_ha_slow_response_get_entities_timeout(self):
        """get_all_entities returns [] on timeout."""
        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
            timeout=0.5,
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_TimeoutCtx())
        mock_session.closed = False

        client._session = mock_session

        entities = await client.get_all_entities()
        assert entities == []


class _FakeResponseCtx:
    """Helper async context manager that yields a mock HTTP response."""

    def __init__(self, status: int):
        self._response = MagicMock()
        self._response.status = status
        self._response.request_info = MagicMock()
        self._response.history = ()

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class TestHAAuthFailure:
    """Verify handling of 401/403 authentication errors."""

    @pytest.mark.asyncio
    async def test_ha_auth_failure_call_service(self):
        """call_service returns False on 401 and marks auth error in health."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="bad_token",
            timeout=1.0,
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_FakeResponseCtx(401))
        mock_session.closed = False

        client._session = mock_session

        result = await client.call_service("light", "turn_on", "light.test")
        assert result is False

        health = client.get_health_status()
        assert health.state == HAConnectionState.AUTH_ERROR
        assert health.error_count >= 1

    @pytest.mark.asyncio
    async def test_ha_auth_failure_get_entities(self):
        """get_all_entities returns [] on 403 and marks auth error."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="bad_token",
            timeout=1.0,
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_FakeResponseCtx(403))
        mock_session.closed = False

        client._session = mock_session

        entities = await client.get_all_entities()
        assert entities == []

        health = client.get_health_status()
        assert health.state == HAConnectionState.AUTH_ERROR

    @pytest.mark.asyncio
    async def test_ha_auth_failure_test_connection(self):
        """test_connection returns False on 401."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="bad_token",
            timeout=1.0,
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_FakeResponseCtx(401))
        mock_session.closed = False

        client._session = mock_session

        result = await client.test_connection()
        assert result is False

        health = client.get_health_status()
        assert health.state == HAConnectionState.AUTH_ERROR


class TestHAHealthStatus:
    """Verify get_health_status returns correct structured state."""

    @pytest.mark.asyncio
    async def test_initial_health_status(self):
        """Fresh client reports UNKNOWN state with zero counters."""
        from src.home_assistant.ha_client import HAConnectionState, HAHealthStatus, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
        )

        health = client.get_health_status()

        assert isinstance(health, HAHealthStatus)
        assert health.state == HAConnectionState.UNKNOWN
        assert health.error_count == 0
        assert health.success_count == 0
        assert health.avg_latency_ms == 0.0
        assert health.ws_connected is False
        assert health.last_error_message == ""
        assert health.last_success_ts == 0.0
        assert health.last_error_ts == 0.0

    @pytest.mark.asyncio
    async def test_health_after_success(self):
        """Health shows CONNECTED after a successful call."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
        )

        # Simulate a successful call by invoking the internal tracker
        client._record_success(latency_ms=42.0)

        health = client.get_health_status()
        assert health.state == HAConnectionState.CONNECTED
        assert health.success_count == 1
        assert health.avg_latency_ms == 42.0
        assert health.last_success_ts > 0

    @pytest.mark.asyncio
    async def test_health_after_error(self):
        """Health shows DISCONNECTED after a failed call."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
        )

        client._record_error(ConnectionError("refused"), "test")

        health = client.get_health_status()
        assert health.state == HAConnectionState.DISCONNECTED
        assert health.error_count == 1
        assert "refused" in health.last_error_message

    @pytest.mark.asyncio
    async def test_health_recovery_after_success_following_error(self):
        """Health transitions from DISCONNECTED to CONNECTED on recovery."""
        from src.home_assistant.ha_client import HAConnectionState, HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
        )

        # First fail
        client._record_error(ConnectionError("down"), "test")
        assert client.get_health_status().state == HAConnectionState.DISCONNECTED

        # Then succeed
        client._record_success(latency_ms=30.0)
        health = client.get_health_status()
        assert health.state == HAConnectionState.CONNECTED
        assert health.error_count == 1
        assert health.success_count == 1

    @pytest.mark.asyncio
    async def test_health_avg_latency_ema(self):
        """Average latency uses exponential moving average."""
        from src.home_assistant.ha_client import HomeAssistantClient

        client = HomeAssistantClient(
            url="http://localhost:8123",
            token="test_token",
        )

        client._record_success(100.0)
        assert client.get_health_status().avg_latency_ms == 100.0

        # EMA: 0.7 * 100 + 0.3 * 200 = 130
        client._record_success(200.0)
        assert client.get_health_status().avg_latency_ms == 130.0

    @pytest.mark.asyncio
    async def test_health_status_is_dataclass(self):
        """HAHealthStatus is a proper dataclass with expected fields."""
        from dataclasses import fields

        from src.home_assistant.ha_client import HAHealthStatus

        field_names = {f.name for f in fields(HAHealthStatus)}
        expected = {
            "state", "last_success_ts", "last_error_ts", "error_count",
            "success_count", "avg_latency_ms", "ws_connected", "last_error_message"
        }
        assert expected == field_names
