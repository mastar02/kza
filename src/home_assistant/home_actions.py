"""Confirmed direct controls with process-local, actor-scoped receipts."""

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields
import logging
import re
import uuid

from src.home_assistant.control_policy import ActionOrigin, HomeActor, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus, get_home_capabilities


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HomeActionRequest:
    """Immutable, allowlisted home control intent."""

    entity_id: str
    operation: str
    request_id: str
    brightness_pct: int | None = None
    origin: ActionOrigin = ActionOrigin.HTTP


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

    def __init__(self, ha_client: HomeAssistantClient, *, policy: HomeControlPolicy | None = None):
        self.ha = ha_client
        self.policy = policy if policy is not None else HomeControlPolicy()
        self._requests: dict[tuple[str, str], tuple[HomeActionRequest, asyncio.Task]] = {}
        self._completed: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._lanes: dict[str, tuple[asyncio.Lock, int]] = {}
        self._closing = False

    async def execute(self, request: HomeActionRequest, *, actor: HomeActor) -> HomeActionResult:
        """Await one shared intent; cancellation of a waiter never resends it.

        Args:
            request: Immutable direct control intent.
            actor: Server-owned authentication evidence and control scope.

        Returns:
            Terminal confirmed, failed, or uncertain receipt.

        Raises:
            HomeActionConflict: The actor already used the ID for another body.
            ValueError: The actor is missing or malformed.
        """
        if not isinstance(actor, HomeActor):
            raise ValueError("invalid_actor")
        if (
            not isinstance(request.request_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request.request_id) is None
        ):
            return HomeActionResult(request.request_id, request.entity_id, "failed", "invalid_request_id")
        key = (actor.receipt_identity, request.request_id)
        existing = self._requests.get(key)
        if existing is not None:
            if not _same_action_body(existing[0], request):
                raise HomeActionConflict("request_id_body_conflict")
            return await asyncio.shield(existing[1])
        if self._closing:
            return HomeActionResult(request.request_id, request.entity_id, "failed", "home_actions_closed")
        task = asyncio.create_task(self._run(request, actor))
        self._requests[key] = (request, task)
        task.add_done_callback(lambda done: self._finish(key, done))
        return await asyncio.shield(task)

    def get_receipt(self, *, actor: HomeActor, request_id: str) -> HomeActionResult | None:
        """Read an actor's receipt; only this method returns in_progress."""
        if not isinstance(actor, HomeActor):
            raise ValueError("invalid_actor")
        entry = self._requests.get((actor.receipt_identity, request_id))
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

    async def execute_service(
        self, domain: str, service: str, entity_id: str, data: dict | None = None,
        *, actor: HomeActor, origin: ActionOrigin, request_id: str | None = None,
    ) -> HomeActionResult:
        """Normalize a legacy producer's exact supported intent before execution.

        Unknown parameters, multiple targets and opaque macros are rejected;
        no free-form service supplied by a model reaches the HA transport.
        """
        identity = request_id if request_id is not None else uuid.uuid4().hex
        try:
            request = request_from_service(domain, service, entity_id, data, request_id=identity, origin=origin)
        except ValueError as error:
            return HomeActionResult(identity, entity_id, "failed", str(error))
        return await self.execute(request, actor=actor)

    async def validate_selection(self, request: HomeActionRequest, *, actor: HomeActor) -> None:
        """Validate admission without claiming a receipt or sending an effect."""
        validate_home_action(request)
        await self.policy.require_allowed(request, actor)

    async def _run(self, request: HomeActionRequest, actor: HomeActor) -> HomeActionResult:
        try:
            validate_home_action(request)
            await self.policy.require_allowed(request, actor)
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
                if fresh.get("attributes", {}).get("entity_id") is not None:
                    return HomeActionResult(request.request_id, request.entity_id, "failed", "group_effects_unverified")
                # A queued request or fresh-state read cannot freeze an old grant.
                try:
                    await self.policy.require_allowed(request, actor)
                except ValueError as error:
                    return HomeActionResult(request.request_id, request.entity_id, "failed", str(error))
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


def request_from_service(
    domain: str, service: str, entity_id: str, data: dict | None,
    *, request_id: str, origin: ActionOrigin,
) -> HomeActionRequest:
    """Convert supported binary light/switch calls without dropping parameters."""
    if not isinstance(entity_id, str) or domain != entity_id.partition(".")[0]:
        raise ValueError("invalid_service_target")
    data = {} if data is None else data
    if not isinstance(data, Mapping):
        raise ValueError("unsupported_action_parameters")
    data = dict(data)
    if data.keys() - {"brightness_pct", "brightness"} or len(data) > 1:
        raise ValueError("unsupported_action_parameters")
    brightness = data.get("brightness_pct")
    if "brightness" in data:
        raw = data["brightness"]
        if type(raw) is not int or not 0 <= raw <= 255:
            raise ValueError("invalid_brightness")
        brightness = max(1, round(raw * 100 / 255)) if raw else 0
    if data and (service != "turn_on" or brightness is None):
        raise ValueError("unexpected_brightness")
    operation = "brightness" if data else service
    request = HomeActionRequest(entity_id, operation, request_id, brightness, origin)
    validate_home_action(request)
    return request
