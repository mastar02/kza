"""Explicit server ownership for synthetic protocol peers used by HTTP tests."""

import os

from src.dashboard.authority import DashboardAuthority


def fixture_token() -> str:
    """Protocol-only tests have their own root; configured HTTP tests use theirs."""
    return os.environ.get('KZA_DASHBOARD_TOKEN', 'synthetic-protocol-only-root')


async def make_browser_authority(stack, path):
    """Register authority before transport owners so it is released last."""
    authority = DashboardAuthority(path, expected_token=fixture_token)
    stack.push_async_callback(authority.aclose)
    await authority.initialize()
    return authority


def bind_browser_authority(bus, authority) -> None:
    """Attach test-only dependencies; product constructors still require injection."""
    bus.test_authority = authority
    bus.test_reader = authority.resolve(fixture_token()).reader_binding()
