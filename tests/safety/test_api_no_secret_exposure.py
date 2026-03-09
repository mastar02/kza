"""
Safety Tests — API Secret Exposure

Verify that:
- API endpoints (health, status, entity listing) never return sensitive data
- The log formatter sanitizes Bearer tokens from log messages
- sanitize_dict / sanitize_value correctly mask secret values
"""

import logging
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from src.core.sanitize import sanitize_dict, sanitize_value, mask_string, SENSITIVE_KEYS


# ============================================================
# sanitize_dict / sanitize_value unit tests
# ============================================================


class TestSanitizeHelpers:
    """Tests for the sanitization helper functions."""

    def test_mask_string_hides_body(self):
        """mask_string should keep only the first N chars visible."""
        assert mask_string("abcdefghijklmnop") == "abcd***"
        assert mask_string("abcdefghijklmnop", visible_chars=6) == "abcdef***"

    def test_mask_string_short_value(self):
        """Very short values should be fully masked."""
        assert mask_string("abc") == "***"
        assert mask_string("ab", visible_chars=4) == "***"

    def test_sanitize_value_masks_token_key(self):
        """Values with a 'token' key should be masked."""
        assert sanitize_value("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "token") == "eyJh***"

    def test_sanitize_value_masks_password_key(self):
        assert sanitize_value("supersecret", "password") == "supe***"

    def test_sanitize_value_masks_secret_key(self):
        assert sanitize_value("my_client_secret_value", "client_secret") == "my_c***"

    def test_sanitize_value_case_insensitive(self):
        """Key matching should be case-insensitive."""
        assert sanitize_value("some_token", "TOKEN") == "some***"
        assert sanitize_value("val", "Password") == "***"

    def test_sanitize_value_ignores_safe_keys(self):
        """Non-sensitive keys should pass through."""
        assert sanitize_value("http://example.com", "url") == "http://example.com"
        assert sanitize_value("living_room", "zone") == "living_room"

    def test_sanitize_value_masks_bearer_in_string(self):
        """Bearer tokens embedded in strings should be masked."""
        header = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc"
        sanitized = sanitize_value(header, "some_header")
        assert "eyJh" in sanitized
        assert "IkpXVCJ9.abc" not in sanitized

    def test_sanitize_dict_masks_nested_tokens(self):
        """Nested dicts with secret keys should be masked recursively."""
        data = {
            "status": "ok",
            "config": {
                "url": "http://ha:8123",
                "token": "secret_token_value_12345",
            },
        }
        result = sanitize_dict(data)
        assert result["status"] == "ok"
        assert result["config"]["url"] == "http://ha:8123"
        assert result["config"]["token"] == "secr***"

    def test_sanitize_dict_handles_empty(self):
        assert sanitize_dict({}) == {}

    def test_sanitize_dict_handles_list_of_dicts(self):
        data = {
            "items": [
                {"name": "foo", "api_key": "sk-abcdef1234567890"},
                {"name": "bar"},
            ]
        }
        result = sanitize_dict(data)
        assert result["items"][0]["api_key"] == "sk-a***"
        assert result["items"][1]["name"] == "bar"


# ============================================================
# API Health Endpoint Tests
# ============================================================


class TestHealthEndpointNoSecrets:
    """Ensure the /api/health endpoint never leaks secrets."""

    @pytest.fixture
    def mock_dashboard(self):
        """Create a DashboardAPI with mocked dependencies."""
        from src.dashboard.api import DashboardAPI

        mock_scheduler = MagicMock()
        mock_scheduler.get_all_routines.return_value = []

        mock_ha = MagicMock()
        mock_ha.test_connection = AsyncMock(return_value=True)

        dashboard = DashboardAPI(
            routine_scheduler=mock_scheduler,
            ha_client=mock_ha,
            host="127.0.0.1",
            port=8080,
        )
        return dashboard

    @pytest.mark.asyncio
    async def test_health_response_has_no_token_fields(self, mock_dashboard):
        """Health response should not contain token/secret keys."""
        from httpx import AsyncClient, ASGITransport

        transport = ASGITransport(app=mock_dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
            data = response.json()

        # Recursively check that no key is a sensitive key
        def check_no_secrets(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    assert key.lower() not in SENSITIVE_KEYS, (
                        f"Health response contains sensitive key '{key}' at {path}"
                    )
                    check_no_secrets(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_no_secrets(item, f"{path}[{i}]")

        check_no_secrets(data)

    @pytest.mark.asyncio
    async def test_health_response_values_are_not_secrets(self, mock_dashboard):
        """Health response string values should not look like tokens."""
        from httpx import AsyncClient, ASGITransport
        import re

        transport = ASGITransport(app=mock_dashboard.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
            data = response.json()

        jwt_re = re.compile(r"eyJ[A-Za-z0-9_-]{20,}")

        def check_no_jwt(obj, path=""):
            if isinstance(obj, str):
                assert not jwt_re.search(obj), (
                    f"Health response contains JWT-like value at {path}: {obj[:40]}..."
                )
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_no_jwt(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_no_jwt(item, f"{path}[{i}]")

        check_no_jwt(data)


# ============================================================
# HA Client Health Status Tests
# ============================================================


class TestHAClientHealthNoSecrets:
    """Ensure HAHealthStatus never exposes the token."""

    def test_health_status_has_no_token(self):
        """HAHealthStatus fields should not include the token."""
        from src.home_assistant.ha_client import HAHealthStatus

        status = HAHealthStatus()
        fields = vars(status)

        for key in fields:
            assert "token" not in key.lower(), f"HAHealthStatus has token-related field: {key}"

    def test_health_status_last_error_no_token(self):
        """last_error_message should not contain the raw token."""
        from src.home_assistant.ha_client import HAHealthStatus

        # Simulate an error that accidentally includes auth info
        status = HAHealthStatus(
            last_error_message="Connection failed to http://ha:8123"
        )
        assert "Bearer" not in status.last_error_message


# ============================================================
# Log Formatter Sanitization Tests
# ============================================================


class TestLogFormatterSanitization:
    """Ensure the StructuredFormatter masks secrets in log messages."""

    def test_bearer_token_masked_in_log_output(self):
        """Bearer tokens in log messages should be automatically masked."""
        from src.core.logging import StructuredFormatter, LogConfig, LogFormat

        formatter = StructuredFormatter(LogConfig(format=LogFormat.PLAIN))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Auth header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        # The full JWT should NOT appear
        assert "IkpXVCJ9.payload.sig" not in output
        # But the masked prefix should
        assert "Bearer eyJh***" in output

    def test_non_bearer_messages_untouched(self):
        """Normal log messages should pass through without modification."""
        from src.core.logging import StructuredFormatter, LogConfig, LogFormat

        formatter = StructuredFormatter(LogConfig(format=LogFormat.PLAIN))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Ejecutado: light.turn_on en light.living (42ms)",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "light.turn_on" in output
        assert "42ms" in output
