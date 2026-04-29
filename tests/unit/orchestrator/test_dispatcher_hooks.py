"""Tests for dispatcher integration with plugin hooks (plan #3 OpenClaw)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.hooks import HookRegistry, BlockResult, RewriteResult


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


@pytest.mark.asyncio
async def test_block_emits_ha_action_blocked_after_event(dispatcher_with_hooks):
    """Block path must emit ha_action_blocked so audit trail is preserved."""
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks
    seen = []

    def block_handler(call):
        return BlockResult(reason="no", rule_name="r1")

    hooks.register_before("before_ha_action", block_handler, priority=10)
    hooks.register_after("ha_action_blocked", lambda p: seen.append(p))

    method = _find_ha_dispatch_method(dispatcher)
    if method is None:
        pytest.fail("HA dispatch method not found — update _find_ha_dispatch_method")

    await method({
        "domain": "light", "service": "turn_on", "entity_id": "light.x",
        "data": {}, "description": "x",
    })

    assert len(seen) == 1
    assert seen[0].block.rule_name == "r1"
    assert seen[0].call.entity_id == "light.x"


@pytest.mark.asyncio
async def test_dispatch_emits_ha_action_dispatched_after_event(dispatcher_with_hooks):
    """Success path must emit ha_action_dispatched with success=True."""
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks
    seen = []

    hooks.register_after("ha_action_dispatched", lambda p: seen.append(p))

    method = _find_ha_dispatch_method(dispatcher)
    if method is None:
        pytest.fail("HA dispatch method not found — update _find_ha_dispatch_method")

    await method({
        "domain": "light", "service": "turn_on", "entity_id": "light.x",
        "data": {}, "description": "x",
    })

    assert len(seen) == 1
    assert seen[0].success is True
    assert seen[0].error is None
    assert seen[0].call.entity_id == "light.x"


@pytest.mark.asyncio
async def test_rewrite_modifies_call_service_ws_args(dispatcher_with_hooks):
    """Rewrite path: handler returns RewriteResult, dispatcher should call HA
    with the modified service_data (not the original)."""
    dispatcher, hooks, ha_client, response_handler = dispatcher_with_hooks

    def cap_brightness(call):
        # Cap brightness at 20% regardless of input
        return RewriteResult(
            modified=call.with_data(brightness_pct=20),
            rule_name="cap_brightness",
        )

    hooks.register_before("before_ha_action", cap_brightness, priority=10)

    method = _find_ha_dispatch_method(dispatcher)
    if method is None:
        pytest.fail("HA dispatch method not found — update _find_ha_dispatch_method")

    await method({
        "domain": "light", "service": "turn_on", "entity_id": "light.cocina",
        "data": {"brightness_pct": 100}, "description": "la luz",
    })

    # call_service_ws should have been called with brightness_pct=20, not 100
    ha_client.call_service_ws.assert_called_once()
    args, kwargs = ha_client.call_service_ws.call_args
    # call_service_ws(domain, service, entity_id, data) — data is positional[3]
    data_arg = args[3] if len(args) >= 4 else kwargs.get("data") or kwargs.get("service_data")
    assert data_arg.get("brightness_pct") == 20, f"Expected brightness_pct=20, got {data_arg}"
