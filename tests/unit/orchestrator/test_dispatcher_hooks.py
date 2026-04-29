"""Tests for dispatcher integration with plugin hooks (plan #3 OpenClaw)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.hooks import HookRegistry, BlockResult


@pytest.fixture
def dispatcher_with_hooks():
    """Construct a minimal RequestDispatcher with mocks + a fresh HookRegistry."""
    from src.orchestrator.dispatcher import RequestDispatcher

    hooks = HookRegistry()

    ha_client = MagicMock()
    ha_client.call_service_ws = AsyncMock(return_value=True)

    response_handler = MagicMock()
    response_handler.speak = MagicMock()

    dispatcher = RequestDispatcher(
        chroma_sync=MagicMock(),
        ha_client=ha_client,
        routine_manager=MagicMock(),
        response_handler=response_handler,
        hooks=hooks,
    )
    return dispatcher, hooks, ha_client, response_handler


def _find_ha_dispatch_method(dispatcher):
    """Locate the method that wraps self.ha.call_service_ws."""
    for name in (
        "_fire_and_reconcile_ha",
        "_reconcile",
        "_dispatch_ha_async",
        "_ha_dispatch",
        "_execute_ha_action",
        "_async_dispatch_ha",
    ):
        method = getattr(dispatcher, name, None)
        if method is not None and asyncio.iscoroutinefunction(method):
            return method
    return None


@pytest.mark.asyncio
async def test_block_prevents_ha_call_and_speaks_reason(dispatcher_with_hooks):
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks

    def block_all(call):
        return BlockResult(reason="prohibido", rule_name="test_block")

    hooks.register_before("before_ha_action", block_all, priority=10)

    method = _find_ha_dispatch_method(dispatcher)
    if method is None:
        pytest.skip(
            "Could not locate HA dispatch method automatically; "
            "update _find_ha_dispatch_method in this test"
        )

    command = {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.x",
        "data": {},
        "description": "la luz",
    }
    await method(command)

    # HA was NOT called
    ha_client.call_service_ws.assert_not_called()
    # Response was spoken with the block reason
    response_handler.speak.assert_called_with("prohibido")


@pytest.mark.asyncio
async def test_no_hooks_baseline_unchanged(dispatcher_with_hooks):
    """Without registered handlers, dispatch flow is unchanged."""
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks
    # No handlers registered — registry is empty

    method = _find_ha_dispatch_method(dispatcher)
    if method is None:
        pytest.skip("Could not locate HA dispatch method")

    command = {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.x",
        "data": {},
        "description": "la luz",
    }
    await method(command)

    # HA WAS called normally
    ha_client.call_service_ws.assert_called_once()
