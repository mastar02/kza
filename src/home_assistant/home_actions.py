"""Confirmed controls and reviewed expansions with actor-scoped receipts."""

import asyncio
from contextvars import Context
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, replace
import logging
import re
import uuid

from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.home_assistant.control_policy import ActionOrigin, EntityGrant, HomeActionDenied, HomeActor, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus
from src.home_assistant.effect_plans import EffectIntent, EffectPlan, expand_effect_plan, validate_plan_references
from src.home_assistant.operations import (
    PARAMETER_FIELDS, action_parameters, compile_effect, validate_observation, validate_parameters,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HomeActionRequest:
    """Immutable, allowlisted home control intent."""

    entity_id: str
    operation: str
    request_id: str
    brightness_pct: int | None = None
    origin: ActionOrigin = ActionOrigin.HTTP
    color_temp_kelvin: int | None = None
    temperature: float | None = None
    hvac_mode: str | None = None
    position: int | None = None
    brightness: int | None = None


@dataclass(frozen=True)
class HomeEffectResult:
    """One expanded leaf's intent and exact transport/observation evidence."""

    intent: EffectIntent
    status: str
    reason: str | None = None
    dispatch: DispatchState = DispatchState.UNKNOWN
    effect: EffectState = EffectState.UNKNOWN
    observation_scope: str | None = None


@dataclass(frozen=True)
class HomeActionResult:
    """Preserve the public status separately from send and HA-state evidence.

    Confirmed describes HA state, not physical sensing. Missing dimensions on
    legacy/synthetic receipts stay unknown, rather than fabricating evidence.
    """

    request_id: str
    entity_id: str
    status: str
    reason: str | None = None
    effects: tuple[HomeEffectResult, ...] = ()
    dispatch: DispatchState = DispatchState.UNKNOWN
    effect: EffectState = EffectState.UNKNOWN
    observation_scope: str | None = None

    @property
    def task_outcome(self) -> TaskOutcome:
        """Project the operation while keeping immediate receipts non-durable."""
        complete = (self.status == "confirmed" and self.dispatch is DispatchState.ACCEPTED
                    and self.effect is EffectState.OBSERVED and self.observation_scope == "ha_state")
        state = TaskStatus.COMPLETED if complete else {
            "failed": TaskStatus.FAILED, "uncertain": TaskStatus.UNCERTAIN,
            "in_progress": TaskStatus.RUNNING,
        }.get(self.status, TaskStatus.UNKNOWN)
        return TaskOutcome(state, persistence=PersistenceState.VOLATILE,
                           dispatch=self.dispatch, effect=self.effect,
                           observation_scope=self.observation_scope)

    @classmethod
    def not_sent(cls, request_id: str, entity_id: str, reason: str) -> "HomeActionResult":
        """Record rejection before transport admission."""
        return cls(request_id, entity_id, "failed", reason,
                   dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED)

    @classmethod
    def composite(cls, request_id: str, entity_id: str, status: str,
                  reason: str | None, effects: tuple[HomeEffectResult, ...]) -> "HomeActionResult":
        """Aggregate known leaves without erasing partial or uncertain sends."""
        sends = {leaf.dispatch for leaf in effects}
        observed = {leaf.effect for leaf in effects}
        if sends <= {DispatchState.NOT_SENT}:
            dispatch = DispatchState.NOT_SENT
        elif sends == {DispatchState.ACCEPTED}:
            dispatch = DispatchState.ACCEPTED
        elif sends & {DispatchState.UNCERTAIN, DispatchState.UNKNOWN}:
            dispatch = DispatchState.UNCERTAIN
        elif DispatchState.ACCEPTED in sends:
            dispatch = DispatchState.PARTIAL
        else:
            dispatch = DispatchState.REJECTED
        if observed == {EffectState.OBSERVED}:
            effect = EffectState.OBSERVED
        elif EffectState.OBSERVED in observed:
            effect = EffectState.PARTIAL
        elif observed & {EffectState.UNCERTAIN, EffectState.UNKNOWN}:
            effect = EffectState.UNCERTAIN
        elif EffectState.UNCONFIRMED in observed:
            effect = EffectState.UNCONFIRMED
        else:
            effect = EffectState.NOT_OBSERVED
        scopes = {leaf.observation_scope for leaf in effects
                  if leaf.effect in {EffectState.OBSERVED, EffectState.PARTIAL}}
        scope = "ha_state" if scopes == {"ha_state"} else None
        if effect in {EffectState.OBSERVED, EffectState.PARTIAL} and scope is None:
            effect = EffectState.UNKNOWN
        return cls(request_id, entity_id, status, reason, effects, dispatch, effect, scope)


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
        or not isinstance(request.operation, str)
        or re.fullmatch(r"(?:light|switch|climate|cover|scene)\.[a-z0-9_]+", request.entity_id) is None
        or not isinstance(request.request_id, str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request.request_id) is None
    ):
        raise ValueError("invalid_home_action")
    validate_parameters(request.entity_id, request.operation, action_parameters(request))


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
            return HomeActionResult.not_sent(request.request_id, request.entity_id, "invalid_request_id")
        key = (actor.receipt_identity, request.request_id)
        existing = self._requests.get(key)
        if existing is not None:
            if not _same_action_body(existing[0], request):
                raise HomeActionConflict("request_id_body_conflict")
            return await asyncio.shield(existing[1])
        if self._closing:
            return HomeActionResult.not_sent(request.request_id, request.entity_id, "home_actions_closed")
        task = asyncio.create_task(self._run(request, actor), context=Context())
        self._requests[key] = (request, task)
        task.add_done_callback(lambda done: self._finish(key, done), context=Context())
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
            return HomeActionResult.not_sent(identity, entity_id, str(error))
        return await self.execute(request, actor=actor)

    async def validate_selection(self, request: HomeActionRequest, *, actor: HomeActor) -> None:
        """Validate all selected effects without claiming or sending an action."""
        validate_home_action(request)
        grants = await self.policy.snapshot_grants()
        self.policy.require_from_grants(request, actor, grants)
        plan = expand_effect_plan(request, grants)
        for intent in (*plan.intermediates, *plan.effects):
            self.policy.require_from_grants(self._leaf_request(request, intent), actor, grants, via_expansion=True)

    @staticmethod
    def _leaf_request(root: HomeActionRequest, intent: EffectIntent) -> HomeActionRequest:
        return replace(root, **{field.name: getattr(intent, field.name) for field in fields(intent)})

    @asynccontextmanager
    async def _entity_lanes(self, entity_ids: Iterable[str]) -> AsyncIterator[None]:
        # Register before awaiting; sorted acquisition prevents deadlocks across
        # overlapping scenes/groups and ordinary direct controls.
        registered = []
        acquired = []
        try:
            for entity_id in sorted(set(entity_ids)):
                lock, users = self._lanes.get(entity_id, (asyncio.Lock(), 0))
                self._lanes[entity_id] = (lock, users + 1)
                registered.append((entity_id, lock))
            for _, lock in registered:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            for entity_id, lock in registered:
                _, users = self._lanes[entity_id]
                if users == 1:
                    del self._lanes[entity_id]
                else:
                    self._lanes[entity_id] = (lock, users - 1)

    async def _run(self, request: HomeActionRequest, actor: HomeActor) -> HomeActionResult:
        try:
            validate_home_action(request)
            grant = await self.policy.require_allowed(request, actor)
            if grant.members or request.entity_id.startswith("scene."):
                grants = await self.policy.snapshot_grants()
                plan = expand_effect_plan(request, grants)
                return await self._run_composite(request, actor, plan)
            async with self._entity_lanes((request.entity_id,)):
                return await self._perform_leaf(request, actor)
        except ValueError as error:
            return HomeActionResult.not_sent(request.request_id, request.entity_id, str(error))
        except Exception:
            logger.warning("Home action preflight dependency failed")
            return HomeActionResult.not_sent(request.request_id, request.entity_id, "ha_preflight_failed")

    async def _require_binding(self, root: HomeActionRequest, actor: HomeActor, plan: EffectPlan, leaf: HomeActionRequest) -> EntityGrant:
        # Membership reads precede one atomic policy observation. No mutable HA
        # group or scene service is called even if its definition changes later.
        ids = [reference.entity_id for reference in plan.groups]
        if plan.scene_targets is not None:
            ids.append(root.entity_id)
        states = dict(zip(ids, await asyncio.gather(*(self.ha.get_entity_state_fresh(entity_id) for entity_id in ids))))
        validate_plan_references(root, plan, states)
        grants = await self.policy.snapshot_grants()
        if expand_effect_plan(root, grants) != plan:
            raise ValueError("effect_selection_changed")
        self.policy.require_from_grants(root, actor, grants)
        for intent in (*plan.intermediates, *plan.effects):
            self.policy.require_from_grants(self._leaf_request(root, intent), actor, grants, via_expansion=True)
        return self.policy.require_from_grants(leaf, actor, grants, via_expansion=True)

    async def _run_composite(self, root: HomeActionRequest, actor: HomeActor, plan: EffectPlan) -> HomeActionResult:
        outcomes = [HomeEffectResult(intent, "not_executed", dispatch=DispatchState.NOT_SENT,
                                     effect=EffectState.NOT_OBSERVED) for intent in plan.effects]
        ids = [root.entity_id, *(group.entity_id for group in plan.groups), *(intent.entity_id for intent in plan.effects)]
        async with self._entity_lanes(ids):
            async def binding(leaf):
                return await self._require_binding(root, actor, plan, leaf)

            try:
                await binding(root)
                for intent in plan.effects:
                    child = self._leaf_request(root, intent)
                    fresh = await self.ha.get_entity_state_fresh(child.entity_id)
                    unit = await self.ha.get_temperature_unit_fresh() if child.temperature is not None else None
                    validate_observation(child, fresh, temperature_unit=unit)
                await binding(root)
            except Exception as error:
                reason = str(error) if isinstance(error, ValueError) else "ha_preflight_failed"
                return HomeActionResult.composite(root.request_id, root.entity_id, "failed", reason, tuple(outcomes))
            for index, intent in enumerate(plan.effects):
                receipt = await self._perform_leaf(self._leaf_request(root, intent), actor, binding=binding)
                outcomes[index] = HomeEffectResult(intent, receipt.status, receipt.reason,
                    receipt.dispatch, receipt.effect, receipt.observation_scope)
                if receipt.status != "confirmed":
                    partial = index > 0
                    return HomeActionResult.composite(root.request_id, root.entity_id,
                        "uncertain" if partial or receipt.status == "uncertain" else "failed",
                        "partial_effects" if partial else receipt.reason, tuple(outcomes))
            return HomeActionResult.composite(root.request_id, root.entity_id, "confirmed", None, tuple(outcomes))

    async def _perform_leaf(self, request: HomeActionRequest, actor: HomeActor, *, binding: Callable[[HomeActionRequest], Awaitable[EntityGrant]] | None = None) -> HomeActionResult:
        dispatch = DispatchState.NOT_SENT
        def dispatched():
            nonlocal dispatch
            dispatch = DispatchState.UNCERTAIN
        try:
            actor.require_current()
            fresh = await self.ha.get_entity_state_fresh(request.entity_id)
            # Even an unsuccessful source read may outlive the original authority.
            actor.require_current()
            unit = await self.ha.get_temperature_unit_fresh() if request.temperature is not None else None
            actor.require_current()
            try:
                validate_observation(request, fresh, temperature_unit=unit)
                grant = await binding(request) if binding is not None else await self.policy.require_allowed(request, actor)
                if request.entity_id.startswith("cover.") and fresh["attributes"].get("device_class") != grant.cover_device_class:
                    raise ValueError("cover_class_changed")
            except ValueError as error:
                return HomeActionResult.not_sent(request.request_id, request.entity_id, str(error))
            effect = compile_effect(request)
            initial_state = fresh["state"]
            actor.require_current()
            if actor._current is None:
                # Trusted household and legacy adapters keep their existing
                # transport contract. HTTP actors always carry a live guard.
                dispatched()
                outcome = await self.ha.call_service_once(effect.domain, effect.service, request.entity_id, effect.data)
            else:
                outcome = await self.ha.call_service_once(effect.domain, effect.service, request.entity_id, effect.data,
                                                          guard=actor._current, on_dispatch=dispatched)
            if outcome.status == ServiceCallStatus.NOT_SENT:
                reason = {"service_authority_denied": "home_control_denied",
                          "service_authority_unavailable": "home_authority_unavailable"}.get(outcome.reason, outcome.reason)
                return HomeActionResult.not_sent(request.request_id, request.entity_id, reason or "ha_not_sent")
            if outcome.status != ServiceCallStatus.ACCEPTED:
                rejected = outcome.status == ServiceCallStatus.REJECTED
                return HomeActionResult(request.request_id, request.entity_id,
                    "failed" if rejected else "uncertain", outcome.reason,
                    dispatch=DispatchState.REJECTED if rejected else DispatchState.UNCERTAIN,
                    effect=EffectState.NOT_OBSERVED if rejected else EffectState.UNCERTAIN)
            dispatch = DispatchState.ACCEPTED
            if request.operation in {"turn_on", "turn_off", "brightness"} and request.brightness is None:
                confirmed = await self.ha.confirm_entity_state(
                    request.entity_id, "off" if effect.service == "turn_off" else "on", brightness_pct=request.brightness_pct,
                )
            else:
                confirmed = await self.ha.confirm_home_action(request, initial_state=initial_state)
            return HomeActionResult(request.request_id, request.entity_id,
                "confirmed" if confirmed else "uncertain", None if confirmed else "state_not_confirmed",
                dispatch=dispatch, effect=EffectState.OBSERVED if confirmed else EffectState.UNCONFIRMED,
                observation_scope="ha_state" if confirmed else None)
        except Exception as error:
            logger.warning("Home action dependency failed; preserving send certainty")
            attempted = dispatch is not DispatchState.NOT_SENT
            effect = (EffectState.UNCONFIRMED if dispatch is DispatchState.ACCEPTED else
                      EffectState.UNCERTAIN if attempted else EffectState.NOT_OBSERVED)
            reason = ("ha_attempt_uncertain" if attempted else
                      str(error) if isinstance(error, HomeActionDenied) else "ha_preflight_failed")
            return HomeActionResult(request.request_id, request.entity_id,
                "uncertain" if attempted else "failed", reason, dispatch=dispatch, effect=effect)


def request_from_service(
    domain: str, service: str, entity_id: str, data: dict | None,
    *, request_id: str, origin: ActionOrigin,
) -> HomeActionRequest:
    """Convert one supported service-shaped intent without dropping parameters."""
    if not isinstance(entity_id, str) or domain != entity_id.partition(".")[0]:
        raise ValueError("invalid_service_target")
    data = {} if data is None else data
    if not isinstance(data, Mapping):
        raise ValueError("unsupported_action_parameters")
    data = dict(data)
    operation = "activate_scene" if domain == "scene" and service == "turn_on" else service
    if domain == "light" and service == "turn_on":
        operation = "color_temp_kelvin" if "color_temp_kelvin" in data else ("brightness" if "brightness_pct" in data or "brightness" in data else "turn_on")
    if data.keys() - set(PARAMETER_FIELDS):
        raise ValueError("unsupported_action_parameters")
    request = HomeActionRequest(entity_id, operation, request_id, origin=origin, **data)
    validate_home_action(request)
    return request
