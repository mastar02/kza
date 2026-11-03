"""Explicit synthetic selection for tests of producers above the action service."""

from unittest.mock import AsyncMock

from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallOutcome, ServiceCallStatus
from src.home_assistant.state_projection import apply_rest, empty_frame
from src.home_assistant.home_actions import HomeActionService

ACCEPTED = ServiceCallOutcome(ServiceCallStatus.ACCEPTED, None)
REJECTED = ServiceCallOutcome(ServiceCallStatus.REJECTED, "test_rejection")


def observed_entity_client(entity_id: str, state: str) -> HomeAssistantClient:
    """Build a native cache reader with explicitly dated synthetic REST evidence."""
    client = HomeAssistantClient("http://synthetic.invalid", "synthetic-token")
    client._state_frame = apply_rest(empty_frame(), entity_id,
        {"entity_id": entity_id, "state": state, "attributes": {}},
        observed_at=client._entity_now(), expected_generation=0, expected_revision=None)
    return client


def selected_actions(ha, entities: tuple[str, ...]) -> HomeActionService:
    """Build the real boundary with explicit grants and fake HA observations."""
    async def fresh(entity_id, **kwargs):
        return {"entity_id": entity_id, "state": "off", "attributes": {"supported_color_modes": ["brightness"]}}

    ha.get_entity_state_fresh = AsyncMock(side_effect=fresh)
    ha.call_service_once = AsyncMock(return_value=ACCEPTED)
    ha.confirm_entity_state = AsyncMock(return_value=True)
    return HomeActionService(ha, policy=HomeControlPolicy(tuple(
        EntityGrant(entity_id, frozenset({"turn_on", "turn_off", "brightness"})) for entity_id in entities
    )))
