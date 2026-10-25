"""Confirmed direct controls with process-local, actor-scoped receipts."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, fields
import logging
import re

from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus, get_home_capabilities


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HomeActionRequest:
    """Immutable, allowlisted home control intent."""

    entity_id: str
    operation: str
    request_id: str
    brightness_pct: int | None = None


@dataclass(frozen=True)
class HomeActionResult:
    """Terminal confirmed/failed/uncertain; receipt reads may be in_progress.

    Confirmed describes HA state, not physical sensing.
    """

    request_id: str
    entity_id: str
    status: str
    reason: str | None = None


class HomeActionConflict(ValueError):
    """A request ID already belongs to a different immutable intent."""


def _same_action_body(previous: HomeActionRequest, current: HomeActionRequest) -> bool:
    """Preserve field types as well as values, including rejected intents."""
    for field in fields(HomeActionRequest):
        previous_value = getattr(previous, field.name)
        current_value = getattr(current, field.name)
        if type(previous_value) is not type(current_value) or previous_value != current_value:
            return False
    return True


def validate_home_action(request: HomeActionRequest) -> None:
    """Reject malformed direct controls before any HA I/O.

    Args:
        request: Requested target, operation, and idempotency identity.

    Raises:
        ValueError: An input is unsupported or cannot identify a safe intent.
    """
    if (
        not isinstance(request.entity_id, str)
        or re.fullmatch(r"(?:light|switch)\.[a-z0-9_]+", request.entity_id) is None
        or not isinstance(request.request_id, str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request.request_id) is None
        or request.operation not in ("turn_on", "turn_off", "brightness")
    ):
        raise ValueError("invalid_home_action")
    if request.operation == "brightness":
        if (
            not request.entity_id.startswith("light.")
            or type(request.brightness_pct) is not int
            or not 0 <= request.brightness_pct <= 100
        ):
            raise ValueError("invalid_brightness")
    elif request.brightness_pct is not None:
        raise ValueError("unexpected_brightness")


class HomeActionService:
    """Share active sends and serialize each entity through confirmation.

    Receipts are process-local: missing or evicted receipts are not evidence
    that a previous order was never sent. Callers must not resend automatically.
    """

    def __init__(self, ha_client: HomeAssistantClient):
        self.ha = ha_client
        self._requests: dict[tuple[str, str], tuple[HomeActionRequest, asyncio.Task]] = {}
        self._completed: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._lanes: dict[str, tuple[asyncio.Lock, int]] = {}
        self._closing = False

    async def execute(self, request: HomeActionRequest, *, actor: str) -> HomeActionResult:
        """Await one shared intent; cancellation of a waiter never resends it.

        Args:
            request: Immutable direct control intent.
            actor: Server-owned authenticated principal.

        Returns:
            Terminal confirmed, failed, or uncertain receipt.

        Raises:
            HomeActionConflict: The actor already used the ID for another body.
            ValueError: The actor is missing or malformed.
        """
        if not isinstance(actor, str) or not actor:
            raise ValueError("invalid_actor")
        if (
            not isinstance(request.request_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request.request_id) is None
        ):
            return HomeActionResult(request.request_id, request.entity_id, "failed", "invalid_request_id")
        key = (actor, request.request_id)
        existing = self._requests.get(key)
        if existing is not None:
            if not _same_action_body(existing[0], request):
                raise HomeActionConflict("request_id_body_conflict")
            return await asyncio.shield(existing[1])
        if self._closing:
            return HomeActionResult(request.request_id, request.entity_id, "failed", "home_actions_closed")
        task = asyncio.create_task(self._run(request))
        self._requests[key] = (request, task)
        task.add_done_callback(lambda done: self._finish(key, done))
        return await asyncio.shield(task)

    def get_receipt(self, *, actor: str, request_id: str) -> HomeActionResult | None:
        """Read an actor's receipt; only this method returns in_progress."""
        entry = self._requests.get((actor, request_id))
        if entry is None:
            return None
        request, task = entry
        if not task.done():
            return HomeActionResult(request_id, request.entity_id, "in_progress")
        return task.result()

    async def close(self) -> None:
        """Drain owned sends and observe exceptions without closing shared HA."""
        self._closing = True
        await asyncio.shield(asyncio.gather(
            *(entry[1] for entry in self._requests.values()), return_exceptions=True,
        ))

    def _finish(self, key: tuple[str, str], task: asyncio.Task) -> None:
        # Observe even if every HTTP waiter disconnected before completion.
        if not task.cancelled():
            task.exception()
        self._completed[key] = None
        while len(self._completed) > 256:
            expired, _ = self._completed.popitem(last=False)
            self._requests.pop(expired, None)

    async def _run(self, request: HomeActionRequest) -> HomeActionResult:
        try:
            validate_home_action(request)
        except ValueError as error:
            return HomeActionResult(request.request_id, request.entity_id, "failed", str(error))
        lock, users = self._lanes.get(request.entity_id, (asyncio.Lock(), 0))
        self._lanes[request.entity_id] = (lock, users + 1)
        attempted = False
        try:
            async with lock:
                fresh = await self.ha.get_entity_state_fresh(request.entity_id)
                if (
                    not isinstance(fresh, dict) or fresh.get("entity_id") != request.entity_id
                    or fresh.get("state") not in ("on", "off")
                    or request.operation not in get_home_capabilities(fresh)
                ):
                    return HomeActionResult(request.request_id, request.entity_id, "failed", "entity_not_controllable")
                brightness = request.brightness_pct
                service = "turn_off" if request.operation == "turn_off" or brightness == 0 else "turn_on"
                data = {"brightness_pct": brightness} if brightness else None
                attempted = True
                outcome = await self.ha.call_service_once(
                    request.entity_id.partition(".")[0], service, request.entity_id, data,
                )
                if outcome.status != ServiceCallStatus.ACCEPTED:
                    status = "failed" if outcome.status == ServiceCallStatus.REJECTED else "uncertain"
                    return HomeActionResult(request.request_id, request.entity_id, status, outcome.reason)
                confirmed = await self.ha.confirm_entity_state(
                    request.entity_id, "off" if service == "turn_off" else "on", brightness_pct=brightness,
                )
                return HomeActionResult(
                    request.request_id, request.entity_id,
                    "confirmed" if confirmed else "uncertain",
                    None if confirmed else "state_not_confirmed",
                )
        except Exception:
            logger.warning("Home action dependency failed; preserving send certainty")
            return HomeActionResult(
                request.request_id, request.entity_id,
                "uncertain" if attempted else "failed",
                "ha_attempt_uncertain" if attempted else "ha_preflight_failed",
            )
        finally:
            _, users = self._lanes[request.entity_id]
            if users == 1:
                del self._lanes[request.entity_id]
            else:
                self._lanes[request.entity_id] = (lock, users - 1)
