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

    # -----------------------------------------------------------------
    # Review 2026-08-06, bloqueante 4: este método era CIEGO a fallos de
    # auth y no logueaba nada, a diferencia de sus dos hermanos del mismo
    # archivo (`call_service`, `_get_entity_state_rest`). Escenario: se
    # vence el token de HA, cada weather.get_forecasts devuelve 401, el
    # usuario escucha "No tengo el pronóstico" —igual que si faltara el
    # sensor— y `_has_auth_error`, que existe justamente para poder decir
    # "mi token está muerto", nunca se prende.
    #
    # Mutaciones que estos tests deben atrapar: colapsar 401/403 de vuelta
    # en el `if resp.status != 200` genérico; borrar los logger.*; borrar
    # el `_record_success`.
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("status", [401, 403])
    @pytest.mark.asyncio
    async def test_auth_error_sets_the_auth_flag(self, client, status, caplog):
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_FakeResponseCtx(status))
        mock_session.closed = False
        client._session = mock_session
        assert client._has_auth_error is False

        with caplog.at_level("ERROR"):
            result = await client.call_service_with_response(
                "weather", "get_forecasts", "weather.forecast_home"
            )

        assert result is None
        assert client._has_auth_error is True
        assert any(
            r.levelname == "ERROR" and "auth error" in r.message.lower()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_non_200_is_logged_not_just_swallowed(self, client, caplog):
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_FakeResponseCtx(500))
        mock_session.closed = False
        client._session = mock_session

        with caplog.at_level("WARNING"):
            await client.call_service_with_response(
                "weather", "get_forecasts", "weather.forecast_home"
            )

        assert any("500" in r.message for r in caplog.records)
        # Un 500 no es un problema de credenciales.
        assert client._has_auth_error is False

    @pytest.mark.asyncio
    async def test_transport_exception_is_logged(self, client, caplog):
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_RaisingCtx())
        mock_session.closed = False
        client._session = mock_session

        with caplog.at_level("ERROR"):
            await client.call_service_with_response(
                "weather", "get_forecasts", "weather.forecast_home"
            )

        assert any(r.levelname == "ERROR" for r in caplog.records)

    @pytest.mark.asyncio
    async def test_timeout_is_logged_and_returns_none(self, client, caplog):
        import asyncio

        class _TimeoutCtx:
            async def __aenter__(self):
                raise asyncio.TimeoutError()

            async def __aexit__(self, *args):
                return False

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=_TimeoutCtx())
        mock_session.closed = False
        client._session = mock_session

        with caplog.at_level("ERROR"):
            result = await client.call_service_with_response(
                "weather", "get_forecasts", "weather.forecast_home"
            )

        assert result is None
        assert any("timeout" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_success_feeds_the_latency_ema(self, client):
        """El camino feliz nunca llamaba a `_record_success`: su latencia no
        entraba al EMA y el método era invisible para el health status."""
        ctx = _FakeResponseCtx(200)
        ctx._response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        assert client.get_health_status().success_count == 0
        await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home"
        )

        health = client.get_health_status()
        assert health.success_count == 1
        assert health.avg_latency_ms > 0.0

    @pytest.mark.asyncio
    async def test_caller_supplied_timeout_reaches_the_request(self, client):
        """Techo por request explícito, no el de sesión (bloqueante 5)."""
        ctx = _FakeResponseCtx(200)
        ctx._response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home", timeout=1.25
        )

        _, kwargs = mock_session.post.call_args
        assert kwargs["timeout"].total == 1.25
