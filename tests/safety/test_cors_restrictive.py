"""
Safety Tests — CORS Restrictive Defaults

Verify that the DashboardAPI CORS configuration:
1. Defaults to restrictive same-origin when no config is provided
2. Respects explicit allowed_origins from settings
3. Logs a warning when allow_origins=['*'] is used
"""

import logging

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.dashboard.api import DashboardAPI


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def _minimal_deps():
    """Minimal dependencies for DashboardAPI."""
    scheduler = MagicMock()
    scheduler.get_all_routines.return_value = []
    return {"routine_scheduler": scheduler}


# ============================================================
# Default CORS Tests
# ============================================================


class TestCORSDefaults:
    """Verify that CORS defaults are restrictive."""

    def test_default_cors_does_not_allow_star(self, _minimal_deps):
        """Without explicit config, CORS should NOT allow '*'."""
        api = DashboardAPI(**_minimal_deps)

        # Inspect the middleware stack for CORSMiddleware config
        cors_mw = _get_cors_middleware(api)
        assert cors_mw is not None, "CORSMiddleware not found on app"

        # allow_all_origins should be False
        assert cors_mw.allow_all_origins is False, (
            "Default CORS should not allow all origins"
        )

    def test_default_cors_allows_localhost(self, _minimal_deps):
        """Default CORS should allow localhost origins."""
        api = DashboardAPI(**_minimal_deps)
        cors_mw = _get_cors_middleware(api)

        allowed = [o.lower() for o in cors_mw.allow_origins]
        assert any("127.0.0.1" in o or "localhost" in o for o in allowed), (
            f"Default CORS should include localhost origins, got: {allowed}"
        )

    def test_default_bind_address_is_localhost(self, _minimal_deps):
        """Default host should be 127.0.0.1 (local-only)."""
        api = DashboardAPI(**_minimal_deps)
        assert api.host == "127.0.0.1"


class TestCORSExplicitConfig:
    """Verify explicit CORS config is respected."""

    def test_custom_origins_applied(self, _minimal_deps):
        """Custom allowed_origins from config should be applied."""
        cors_config = {
            "allowed_origins": ["http://192.168.1.50:3000", "http://myhost:8080"],
        }
        api = DashboardAPI(**_minimal_deps, cors_config=cors_config)
        cors_mw = _get_cors_middleware(api)

        assert "http://192.168.1.50:3000" in cors_mw.allow_origins
        assert "http://myhost:8080" in cors_mw.allow_origins

    def test_wildcard_origin_sets_allow_all(self, _minimal_deps):
        """allow_origins=['*'] should set allow_all_origins=True."""
        cors_config = {"allowed_origins": ["*"]}
        api = DashboardAPI(**_minimal_deps, cors_config=cors_config)
        cors_mw = _get_cors_middleware(api)

        assert cors_mw.allow_all_origins is True

    def test_wildcard_origin_logs_warning(self, _minimal_deps, caplog):
        """allow_origins=['*'] should emit a log warning."""
        cors_config = {"allowed_origins": ["*"]}
        with caplog.at_level(logging.WARNING):
            DashboardAPI(**_minimal_deps, cors_config=cors_config)

        assert any("insecure" in record.message.lower() for record in caplog.records), (
            "Expected a warning about insecure CORS config"
        )


class TestAPIBindAddress:
    """Verify API bind address configuration."""

    def test_explicit_host_override(self, _minimal_deps):
        """Explicit host= parameter should override default."""
        api = DashboardAPI(**_minimal_deps, host="0.0.0.0")
        assert api.host == "0.0.0.0"

    def test_explicit_localhost(self, _minimal_deps):
        api = DashboardAPI(**_minimal_deps, host="127.0.0.1")
        assert api.host == "127.0.0.1"

    def test_custom_port(self, _minimal_deps):
        api = DashboardAPI(**_minimal_deps, port=9090)
        assert api.port == 9090


# ============================================================
# Helpers
# ============================================================


class _CORSConfig:
    """Lightweight wrapper around the CORS middleware registration kwargs."""

    def __init__(self, kwargs: dict):
        origins = kwargs.get("allow_origins", [])
        self.allow_origins = origins
        self.allow_all_origins = origins == ["*"]
        self.allow_credentials = kwargs.get("allow_credentials", False)
        self.allow_methods = kwargs.get("allow_methods", [])
        self.allow_headers = kwargs.get("allow_headers", [])


def _get_cors_middleware(api: DashboardAPI) -> _CORSConfig | None:
    """Extract CORS configuration from the FastAPI user_middleware list.

    FastAPI stores middleware registrations in ``app.user_middleware`` as
    ``Middleware(cls, **kwargs)`` objects.  The actual middleware stack is only
    built when the app starts serving, so we inspect the registration instead.
    """
    from starlette.middleware.cors import CORSMiddleware

    for mw in api.app.user_middleware:
        if mw.cls is CORSMiddleware:
            return _CORSConfig(mw.kwargs)
    return None
