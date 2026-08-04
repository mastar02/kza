"""Tests for HomeAssistantClient.call_service_with_response.

Unlike call_service (which only reports success as bool), this method reads
the JSON body from services that declare SupportsResponse (weather.get_forecasts,
etc), via ?return_response=true. Mirrors the async-context-manager session
mocking pattern used for call_service in tests/unit/test_ha_client.py
(_FakeResponseCtx / _TimeoutCtx) rather than introducing a new mocking style.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.home_assistant.ha_client import HomeAssistantClient


class _FakeResponseCtx:
    """Async context manager that yields a mock HTTP response with a given status."""

    def __init__(self, status: int):
        self._response = MagicMock()
        self._response.status = status

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _RaisingCtx:
    """Async context manager that raises on entry, like a dropped connection."""

    async def __aenter__(self):
        raise RuntimeError("connection reset")

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def client():
    return HomeAssistantClient(url="http://localhost:8123", token="test_token")


class TestCallServiceWithResponse:
    @pytest.mark.asyncio
    async def test_success_returns_parsed_json_body(self, client):
        """200 with a JSON body returns the parsed dict."""
        ctx = _FakeResponseCtx(200)
        body = {"service_response": {"weather.forecast_home": {"forecast": []}}}
        ctx._response.json = AsyncMock(return_value=body)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        result = await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home", {"type": "daily"}
        )

        assert result == body

    @pytest.mark.asyncio
    async def test_non_200_returns_none_and_records_error(self, client):
        """A non-200 status (e.g. 500) returns None and records the error."""
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_FakeResponseCtx(500))
        mock_session.closed = False
        client._session = mock_session

        result = await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home"
        )

        assert result is None
        assert client.get_health_status().error_count == 1

    @pytest.mark.asyncio
    async def test_transport_exception_returns_none_and_records_error(self, client):
        """An exception raised by the session (not just a bad status) is caught."""
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_RaisingCtx())
        mock_session.closed = False
        client._session = mock_session

        result = await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home"
        )

        assert result is None
        assert client.get_health_status().error_count == 1

    @pytest.mark.asyncio
    async def test_does_not_pass_headers_kwarg(self, client):
        """_ensure_session already builds the session with self.headers."""
        ctx = _FakeResponseCtx(200)
        ctx._response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home"
        )

        _, kwargs = mock_session.post.call_args
        assert "headers" not in kwargs
        assert kwargs["json"]["entity_id"] == "weather.forecast_home"
