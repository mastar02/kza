"""
KZA Dashboard API
API REST con FastAPI para gestión de rutinas, presencia y configuración
"""

import asyncio
from contextlib import AsyncExitStack
from contextvars import Context
from dataclasses import dataclass
import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Set
from datetime import UTC, datetime
from typing import Literal, Self

import aiosqlite
from fastapi import Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.sanitize import sanitize_dict
from src.core.observation import Observation, ObservationReason, ObservationSource
from src.core.task_outcome import EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.audio.microphone_status import MicrophonesSnapshot
from src.dashboard.admin import register_admin_routes
from src.dashboard.auth import (
    COOKIE_NAME, DashboardPrincipal, require_dashboard_access, require_dashboard_session,
    routine_access_permissions, session_guard,
)
from src.dashboard.authority import DashboardAuthority, DashboardPermission
from src.home_assistant.control_policy import HomeActor
from src.lists.list_store import OwnerType
from src.dashboard.native_work import ensure_admin_work
from src.memory import output_registration_v2 as output_registration
from src.reminders.reminder_scheduler import NoticeDismissal
from src.reminders.reminder_store import ReminderScope
from src.dashboard.home_api import (
    LegacyHomeActionBody, execute_home_action, register_home_disabled_root, register_home_frontend,
    register_home_routes,
)
from src.dashboard.home_catalog import HomeCatalogService
from src.dashboard.live_event_bus import LiveEventBus
from src.dashboard.memory_api import register_memory_routes
from src.dashboard.memory_voice_api import register_memory_voice_routes
from src.dashboard.rest_delivery import DashboardApplication
from src.dashboard.voice_api import register_voice_routes
from src.dashboard.evaluation_api import register_evaluation_routes
from src.dashboard.credential_api import register_credential_routes
from src.dashboard.ambient_api import register_ambient_routes
from src.dashboard.capture_api import register_capture_routes
from src.dashboard.capture_browser_api import register_capture_browser_routes
from src.dashboard.capture_ingress import CaptureIngressMiddleware
from src.dashboard.observability import register_observability_routes
from src.dashboard.observation_http import entity_observation_metadata, observation_headers
from src.rooms.room_context import RoomConfig
from src.routines.models import (
    RoutineConflict, RoutineNotFound, RoutineSpec, RoutineValidationError, ScheduledRoutine,
    parse_routine_spec, routine_to_dict, run_to_dict,
)
from src.routines.routine_store import HOUSEHOLD_VOICE
from src.home_assistant.home_actions import HomeActionRequest, HomeActionResult, HomeActionService

logger = logging.getLogger(__name__)


# ==================== Modelos Pydantic ====================

class RoutineContractModel(BaseModel):
    """Reject unknown fields and implicit conversions at the HTTP boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RoutineScheduleModel(RoutineContractModel):
    """One local wall-clock time on selected weekdays."""

    at: str
    weekdays: list[int]


class RoutineConditionModel(RoutineContractModel):
    """A state condition evaluated freshly by the scheduler."""

    entity_id: str
    state: str


class RoutineActionModel(RoutineContractModel):
    """An allowlisted Casa action without an editable actor."""

    entity_id: str
    operation: Literal["turn_on", "turn_off", "brightness", "color_temp_kelvin", "set_temperature", "set_hvac_mode", "open_cover", "close_cover", "stop_cover", "set_cover_position", "activate_scene"]
    brightness_pct: int | None = None
    color_temp_kelvin: int | None = None
    temperature: float | None = None
    hvac_mode: str | None = None
    position: int | None = None
    brightness: int | None = None


class RoutineWriteModel(RoutineContractModel):
    """Complete canonical definition shared with persistent routines."""

    name: str
    description: str
    enabled: bool
    timezone: Literal["America/Montevideo"]
    schedule: RoutineScheduleModel | None
    conditions: list[RoutineConditionModel]
    actions: list[RoutineActionModel]

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        """Apply the same exact contract as the store and scheduler."""
        self.to_spec()
        return self

    def to_spec(self) -> RoutineSpec:
        """Return a normalized immutable definition with server metadata excluded."""
        return parse_routine_spec(self.model_dump(exclude={"expected_revision"}, exclude_unset=True))


class RoutineUpdateModel(RoutineWriteModel):
    """Complete definition and a strict optimistic concurrency revision."""

    expected_revision: int = Field(ge=1)


class RoutineRevisionModel(RoutineContractModel):
    """Revision required by enable and disable commands."""

    expected_revision: int = Field(ge=1)


class ExecuteRoutineModel(RoutineContractModel):
    """One actor-scoped idempotent manual attempt."""

    request_id: str = Field(min_length=1, max_length=128)


class PresenceUserModel(BaseModel):
    user_id: str
    name: str
    is_home: bool
    current_room: str | None = None
    last_seen: str | None = None
    devices: list[dict] = Field(default_factory=list)


class ListCreateModel(BaseModel):
    """A list for the session: shared, or personal to the session person (T035-U1).

    The owner never comes from the client: a ``user_id`` field is rejected (422).
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    shared: bool = False


class ListItemCreateModel(BaseModel):
    """One item; the session decides who added it (T035-U1)."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)


class ReminderNoticeDismissModel(BaseModel):
    """One shared occurrence, identified by reminder id and scheduled time."""

    model_config = ConfigDict(extra="forbid")
    reminder_id: str = Field(min_length=1, max_length=64)
    trigger_at: float


# Display labels by reason (T034-U2 web projection). They never select rows.
_NOTICE_LABELS = {
    ("confirmed", None): "entregado", ("confirmed", "dismissed"): "descartado",
    ("uncertain", "delivery_cancelled"): "cortado", ("uncertain", "delivery_partial"): "incompleto",
}


def _notice_label(entry) -> str:
    if entry.status == "sending":
        return "anunciando"
    if entry.status == "pending":
        return "no entregado" if entry.failed_attempts else "pendiente"
    return _NOTICE_LABELS.get((entry.status, entry.reason), "incierto")


def _notice_to_dict(entry) -> dict:
    return {"reminder_id": entry.reminder_id, "trigger_at": entry.trigger_at, "text": entry.text,
            "status": entry.status, "reason": entry.reason, "label": _notice_label(entry),
            "pending": entry.pending, "attempts": entry.attempts, "failed_attempts": entry.failed_attempts,
            "claimed_at": entry.claimed_at}


def _notice_counts(entries) -> dict:
    return {"pending": sum(entry.pending for entry in entries),
            "pending_not_delivered": sum(entry.pending and entry.failed_attempts > 0 for entry in entries),
            "confirmed": sum(entry.status == "confirmed" for entry in entries),
            "uncertain": sum(entry.status == "uncertain" for entry in entries)}


# Exclusive upper bound of a web reminder time: 9999-01-01T00:00:00Z. Any later
# epoch cannot be formatted or advanced on the household calendar (T035-U1-RV1).
REMINDER_TRIGGER_MAX = 253_370_764_800.0
# The recurrences parse_recurrence and next_trigger can walk (weekday 1..7, month day 1..31).
REMINDER_RECURRENCE_PATTERN = r"^(daily|weekdays|weekly:[1-7]|monthly:([1-9]|[12][0-9]|3[01]))$"


class ReminderCreateModel(BaseModel):
    """A shared or private reminder; the owner comes from the session (T035-U1).

    Every household session may write one, so the text is bounded and the
    recurrence is one the scheduler can walk (422 here). ``trigger_at`` is
    checked by the route (``_valid_trigger_at``): a validation error would echo
    a non-finite input, which no JSON response can carry.
    """

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)
    trigger_at: float
    recurrence: str | None = Field(default=None, max_length=16, pattern=REMINDER_RECURRENCE_PATTERN)
    ha_actions: list[dict] | None = None
    scope: ReminderScope = ReminderScope.PRIVATE


_NO_STORE = {"Cache-Control": "no-store"}


def _valid_trigger_at(value: float) -> bool:
    """A finite positive epoch before ``REMINDER_TRIGGER_MAX`` (T035-U1-RV1)."""
    return math.isfinite(value) and 0 < value < REMINDER_TRIGGER_MAX


@dataclass(frozen=True)
class TaskScope:
    """Task scope of one dashboard session, derived from its principal only (T035-U1).

    A household session bound to a person (``tasks:personal``) sees shared tasks
    and that person's own; every other session (household without a person,
    a visitor, an operator — user answer Q16) sees shared tasks only. A voice
    or persona label never selects it.

    Attributes:
        person: The session's person, or None for a shared-only session.
    """

    person: str | None

    @classmethod
    def of(cls, principal: DashboardPrincipal) -> "TaskScope":
        """Scope of a server-owned principal."""
        personal = DashboardPermission.PERSONAL_TASKS in principal.permissions and principal.user_id is not None
        return cls(principal.user_id if personal else None)

    @property
    def shared_only(self) -> bool:
        """Whether only shared tasks are in scope."""
        return self.person is None

    def permissions(self, *, write: bool) -> frozenset[str]:
        """Permissions every native step of the operation rechecks.

        A lease's permissions never change: revising, revoking, expiring or
        retiring a credential's person retires the whole lease. The
        ``tasks:personal`` recheck is therefore defense in depth; it refuses a
        lease that does not hold it even if a scope naming a person were built
        for it (T035-U1-RV5).
        """
        needed = {DashboardPermission.TASKS_WRITE if write else DashboardPermission.TASKS_READ}
        if self.person is not None:
            needed.add(DashboardPermission.PERSONAL_TASKS)
        return frozenset(permission.value for permission in needed)


def _committed() -> dict:
    return TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED,
                       effect=EffectState.OBSERVED, observation_scope="software_state").to_dict()


def _task_error(status_code: int, detail: str, *, write: bool, handled: bool = True) -> JSONResponse:
    """A task route refusal with its TaskOutcome: nothing was written."""
    outcome = TaskOutcome(TaskStatus.FAILED, handled=handled,
                          persistence=PersistenceState.NOT_WRITTEN if write else PersistenceState.NOT_REQUIRED)
    return JSONResponse(status_code=status_code, content={"detail": detail, "outcome": outcome.to_dict()},
                        headers=_NO_STORE)


def _query_rejection(request: Request, *allowed: str, write: bool = False) -> JSONResponse | None:
    """Reject a client-chosen owner and any unknown query parameter (422), on reads and writes."""
    names = set(request.query_params)
    if "user_id" in names:
        return _task_error(422, "user_id_not_accepted", write=write)
    if names - set(allowed):
        return _task_error(422, "unexpected_query_parameter", write=write)
    return None


def _list_to_dict(lst) -> dict:
    # A shared list never names a person, even one recorded by an older client.
    return {"id": lst.id, "name": lst.name, "owner_type": lst.owner_type,
            "owner_id": lst.owner_id if lst.owner_type is OwnerType.USER else None}


def _reminder_actions(reminder, rooms: Mapping[str, str]) -> list[dict] | None:
    """What a reminder will do: domain, service, entity and catalog room; never service data."""
    if not reminder.ha_actions:
        return None
    actions = []
    for action in reminder.ha_actions:
        if not isinstance(action, Mapping):
            continue
        entity_id = action.get("entity_id")
        actions.append({"domain": action.get("domain"), "service": action.get("service"), "entity_id": entity_id,
                        "room": rooms.get(entity_id) if isinstance(entity_id, str) else None})
    return actions


def _reminder_to_dict(reminder, rooms: Mapping[str, str]) -> dict:
    return {"id": reminder.id, "text": reminder.text, "trigger_at": reminder.trigger_at,
            "recurrence": reminder.recurrence, "state": reminder.state, "scope": reminder.scope,
            "ha_actions": _reminder_actions(reminder, rooms)}


# ==================== Dashboard API ====================


GRACEFUL_SHUTDOWN_SECONDS = 5
CONNECTION_DRAIN_SECONDS = 2.0
_OWNED_SIGNAL_SERVERS: dict[type, type] = {}


def _owned_signal_server(uvicorn_module) -> type:
    """A uvicorn Server whose capture_signals() is a no-op (cached per base class)."""
    base = uvicorn_module.Server
    cached = _OWNED_SIGNAL_SERVERS.get(base)
    if cached is None:
        from contextlib import contextmanager

        class OwnedSignalServer(base):
            """Leave SIGTERM/SIGINT to the process owner; never re-raise them."""

            @contextmanager
            def capture_signals(self):
                yield

        cached = _OWNED_SIGNAL_SERVERS[base] = OwnedSignalServer
    return cached


class DashboardAPI:
    """
    API REST para el dashboard de KZA.
    Gestión de rutinas, presencia, entidades y configuración.
    """

    def __init__(
        self,
        routine_scheduler=None,
        routine_executor=None,
        presence_detector=None,
        ha_client=None,
        list_manager=None,
        reminder_manager=None,
        health_aggregator=None,
        reminder_scheduler=None,
        host: str = "127.0.0.1",
        port: int = 9500,  # KZA sub-rango 9500-9599 — :8080 reservado a Kibana (Notion pág 14)
        cors_config: dict | None = None,
        event_bus: LiveEventBus | None = None,
        llm_router=None,
        user_manager=None,
        alert_manager=None,
        zone_manager=None,
        event_logger=None,
        speaker_identifier=None,
        speaker_model_lock=None,
        llm_metrics=None,
        observability_use_mocks: bool = False,
        *,
        authority: DashboardAuthority | None = None,
        home_rooms: Mapping[str, RoomConfig] | None = None,
        home_actions: HomeActionService | None = None,
        microphone_snapshot_provider: Callable[[], MicrophonesSnapshot | None] | None = None,
        home_catalog: HomeCatalogService | None = None,
        home_enabled: bool = True,
        memory_manager=None,
        voice_registry=None,
        ambient_control=None,
        capture_control=None,
        configuration_changes=None,
        consent_gate=None,
    ):
        self.scheduler = routine_scheduler
        self.authority = authority
        self.executor = routine_executor
        self.presence = presence_detector
        self.ha = ha_client
        self.list_manager = list_manager
        self.reminder_manager = reminder_manager
        self.health_aggregator = health_aggregator
        self.reminder_scheduler = reminder_scheduler
        self.host = host
        self.port = port
        self.event_bus = event_bus
        self.llm_router = llm_router
        self.user_manager = user_manager
        self.alert_manager = alert_manager
        self.zone_manager = zone_manager
        self.event_logger = event_logger
        self.speaker_identifier = speaker_identifier
        self.speaker_model_lock = speaker_model_lock
        self.llm_metrics = llm_metrics
        self.observability_use_mocks = observability_use_mocks
        self.microphone_snapshot_provider = microphone_snapshot_provider
        self.home_enabled = home_enabled
        self.memory_manager = memory_manager
        self.voice_registry = voice_registry
        self.ambient_control = ambient_control
        self.capture_control = capture_control
        # OQ-8: admin-only cloud-consent withdrawal (never voice/persona).
        self.configuration_changes = configuration_changes
        self.consent_gate = consent_gate
        self.home_actions = home_actions if home_actions is not None else (
            HomeActionService(ha_client) if ha_client is not None else None
        )
        self._owns_home_catalog = home_catalog is None
        self.home_catalog = home_catalog if home_catalog is not None else HomeCatalogService(
            ha_client, home_rooms or {}, policy=self.home_actions.policy if self.home_actions is not None else None,
            policy_provider=(lambda: self.home_actions.policy) if self.home_actions is not None else None)

        # FastAPI app
        self.app = DashboardApplication(
            title="KZA Dashboard API",
            description="API para gestión de asistente de voz KZA",
            version="1.0.0",
            dependencies=[Depends(require_dashboard_access)],
            docs_url=None, redoc_url=None, openapi_url=None,
        )
        self.app.state.dashboard_authority = authority
        self.admin_work = ensure_admin_work(self.app)
        if self.capture_control is not None:
            self.capture_control.bind_enrollment_work(self.admin_work)
            self.app.add_middleware(CaptureIngressMiddleware, control=self.capture_control)
        self.app.state.rest_delivery_owner = event_bus._retention.rest if event_bus is not None and event_bus._retention is not None else None

        # CORS — use explicit config; default to restrictive same-origin
        cors = cors_config or {}
        allowed_origins = cors.get("allowed_origins", [
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        ])
        if "*" in allowed_origins:
            logger.warning(
                "CORS allow_origins=['*'] is insecure — restrict in production "
                "via dashboard.cors.allowed_origins in settings.yaml"
            )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=cors.get("allow_credentials", True),
            allow_methods=cors.get("allow_methods", ["GET", "POST", "PUT", "DELETE"]),
            allow_headers=cors.get("allow_headers", ["*"]),
        )

        # WebSocket clients
        self._ws_clients: list[WebSocket] = []
        self._server = None
        self._serve_task = None
        self._stop_task: asyncio.Task | None = None

        # Registrar rutas
        self._register_routine_exception_handlers()
        self._register_routes()

    async def _action_rooms(self, reminders) -> dict[str, str]:
        """Room names of the entities reminder actions target, from the home catalog.

        Display only: a catalog that cannot be read leaves every room unknown
        (None) and never fails the reminder read.
        """
        if self.home_catalog is None or not any(r.ha_actions for r in reminders):
            return {}
        try:
            snapshot = await self.home_catalog.snapshot()
        except Exception as error:
            logger.warning("[Reminders] home catalog unavailable for action rooms: %s", type(error).__name__)
            return {}
        return {entity_id: room.name for room in snapshot.rooms for entity_id in room.entity_ids}

    def _register_routes(self):
        """Registrar todas las rutas de la API"""

        # ==================== Rutinas ====================

        @self.app.get("/api/routines")
        async def list_routines(
            enabled: bool | None = None,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """List KZA definitions and their latest durable receipts."""
            scheduler = self._require_routines(principal, {"home:read"})
            routines = await scheduler.get_all_routines()
            if enabled is not None:
                routines = [routine for routine in routines if routine.spec.enabled == enabled]
            return [await self._routine_to_response(routine, principal) for routine in routines]

        # Keep preview ahead of routes containing a dynamic routine identifier.
        @self.app.post("/api/routines/preview")
        async def preview_routine(
            routine: RoutineWriteModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Normalize a definition without reads, writes or actions."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            return scheduler.preview(routine.to_spec())

        @self.app.get("/api/routines/{routine_id}")
        async def get_routine(
            routine_id: str,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Read a persisted definition."""
            scheduler = self._require_routines(principal, {"home:read"})
            routine = await scheduler.get_routine(routine_id)
            if routine is None:
                raise RoutineNotFound(routine_id)
            return await self._routine_to_response(routine, principal)

        @self.app.post("/api/routines")
        async def create_routine(
            routine: RoutineWriteModel, request: Request,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Acknowledge a new definition only after its commit."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            try:
                created = await scheduler.register_routine(routine.to_spec(), actor=principal.subject,
                    guard=session_guard(request, frozenset({'routines:manage'}), principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            response = await self._routine_to_response(created, principal)
            await self._broadcast_ws({"type": "routine_created", "routine": response})
            return response

        @self.app.put("/api/routines/{routine_id}")
        async def update_routine(
            routine_id: str, update: RoutineUpdateModel, request: Request,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Persist a complete definition with compare-and-swap semantics."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            try:
                updated = await scheduler.update_routine(
                    routine_id, update.to_spec(), expected_revision=update.expected_revision,
                    actor=principal.subject,
                    guard=session_guard(request, frozenset({'routines:manage'}), principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            response = await self._routine_to_response(updated, principal)
            await self._broadcast_ws({"type": "routine_updated", "routine": response})
            return response

        @self.app.delete("/api/routines/{routine_id}")
        async def delete_routine(
            routine_id: str, request: Request,
            expected_revision: str = Query(pattern=r"^[1-9][0-9]*$"),
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Delete only the idle revision the caller has actually read."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            try:
                await scheduler.unregister_routine(routine_id, expected_revision=int(expected_revision),
                    guard=session_guard(request, frozenset({'routines:manage'}), principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            await self._broadcast_ws({"type": "routine_deleted", "routine_id": routine_id})
            return {"success": True}

        @self.app.post("/api/routines/{routine_id}/execute", status_code=202)
        async def execute_routine(
            routine_id: str, body: ExecuteRoutineModel, request: Request,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Return the durable claim while its scheduler-owned worker proceeds."""
            permissions = routine_access_permissions(principal, execute=True)
            scheduler = self._require_routines(principal, permissions)
            actor = self._require_routine_actor(principal)
            try:
                run = await scheduler.execute_by_id(
                    routine_id, actor=actor, request_id=body.request_id,
                    guard=session_guard(request, permissions, principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            return run_to_dict(run)

        @self.app.get("/api/routines/{routine_id}/runs/{run_id}")
        async def get_routine_run(
            routine_id: str, run_id: str,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Read the receipt and ordered per-action outcomes for this routine."""
            scheduler = self._require_routines(principal, routine_access_permissions(principal))
            run = await scheduler.get_run(routine_id, run_id)
            if not self._can_read_routine_run(principal, actor=run.actor, source=run.source):
                raise RoutineNotFound(run_id)
            return run_to_dict(run)

        @self.app.get("/api/routines/{routine_id}/attempts")
        async def get_manual_attempt(
            routine_id: str,
            request_id: str = Query(min_length=1, max_length=128),
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Consult a persisted actor-scoped attempt without starting execution.

            request_id is required query text of length 1..128, preserved exactly
            through browser URL normalization, including dot-only IDs.
            """
            scheduler = self._require_routines(principal, routine_access_permissions(principal))
            run = await scheduler.get_manual_attempt(
                routine_id, actor=self._require_routine_actor(principal), request_id=request_id,
            )
            return run_to_dict(run)

        @self.app.post("/api/routines/{routine_id}/enable")
        async def enable_routine(
            routine_id: str, body: RoutineRevisionModel, request: Request,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Enable an idle definition using its expected revision."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            try:
                routine = await scheduler.enable_routine(
                    routine_id, expected_revision=body.expected_revision, actor=principal.subject,
                    guard=session_guard(request, frozenset({'routines:manage'}), principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            return await self._routine_to_response(routine, principal)

        @self.app.post("/api/routines/{routine_id}/disable")
        async def disable_routine(
            routine_id: str, body: RoutineRevisionModel, request: Request,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Disable an idle definition using its expected revision."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            try:
                routine = await scheduler.disable_routine(
                    routine_id, expected_revision=body.expected_revision, actor=principal.subject,
                    guard=session_guard(request, frozenset({'routines:manage'}), principal))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="routine_write_denied") from error
            return await self._routine_to_response(routine, principal)

        # ==================== Presencia ====================

        @self.app.get("/api/presence", response_model=list[PresenceUserModel])
        async def get_presence():
            """Obtener estado de presencia de todos los usuarios"""
            if not self.presence:
                return []

            users = []
            for user_id in self.presence.get_tracked_users():
                status = self.presence.get_user_presence(user_id)
                if status:
                    users.append(PresenceUserModel(
                        user_id=user_id,
                        name=status.user_name or user_id,
                        is_home=status.is_home,
                        current_room=status.current_room,
                        last_seen=status.last_seen.isoformat() if status.last_seen else None,
                        devices=[{"mac": d.mac_address, "rssi": d.rssi} for d in status.devices]
                    ))

            return users

        @self.app.get("/api/presence/home")
        async def is_anyone_home():
            """Verificar si hay alguien en casa"""
            if not self.presence:
                return {"anyone_home": False, "users": []}

            users_home = []
            for user_id in self.presence.get_tracked_users():
                status = self.presence.get_user_presence(user_id)
                if status and status.is_home:
                    users_home.append(user_id)

            return {
                "anyone_home": len(users_home) > 0,
                "users": users_home
            }

        # ==================== Entidades HA ====================

        @self.app.get("/api/entities", dependencies=[Depends(require_dashboard_session)])
        async def list_entities(domain: str | None = None):
            """Listar entidades de Home Assistant"""
            if not self.ha:
                return []

            entities = await self.ha.get_domotics_entities()

            if domain:
                entities = [e for e in entities if e["entity_id"].startswith(f"{domain}.")]

            return [
                {
                    "entity_id": e["entity_id"],
                    "state": e["state"],
                    "name": e["attributes"].get("friendly_name", e["entity_id"]),
                    "domain": e["entity_id"].split(".")[0],
                    "attributes": e.get("attributes", {})
                }
                for e in entities
            ]

        @self.app.get("/api/entities/{entity_id}", dependencies=[Depends(require_dashboard_session)])
        async def get_entity(entity_id: str, request: Request) -> JSONResponse:
            """Read one entity under the original diagnostic session authority."""
            guard = session_guard(request, frozenset({"diagnostics:read"}))
            attempted_at = datetime.now(UTC)

            def unavailable(reason: ObservationReason) -> JSONResponse:
                observation = Observation.from_read(ObservationSource.HA_ENTITY,
                    available=False, observed_at=None, ttl_s=6,
                    attempted_at=attempted_at if reason is not ObservationReason.UNCONFIGURED else None,
                    reason=reason)
                return JSONResponse(status_code=503,
                    content={"detail": "Estado de entidad no disponible"},
                    headers=observation_headers(observation, now=datetime.now(UTC)))

            if not self.ha:
                return unavailable(ObservationReason.UNCONFIGURED)
            try:
                value = await self.ha.read_entity_observation(entity_id)
            except Exception as error:
                logger.warning("HA entity read unavailable: %s", type(error).__name__)
                return unavailable(ObservationReason.UNAVAILABLE)
            if guard() is not True:
                raise HTTPException(status_code=403, detail="dashboard_session_retired")
            now = datetime.now(UTC)
            try:
                metadata = entity_observation_metadata(value, now=now)
                headers = observation_headers(metadata, now=now)
                current = value.current(now.timestamp())
                if current and value.exists is True:
                    state = value.state
                    if isinstance(state, dict):
                        request.scope["kza.dashboard_observation"] = metadata
                        return JSONResponse(content=state, headers=headers)
                if current and value.exists is False:
                    request.scope["kza.dashboard_observation"] = metadata
                    return JSONResponse(status_code=404,
                        content={"detail": "Entidad no encontrada"}, headers=headers)
                return JSONResponse(status_code=503,
                    content={"detail": "Estado de entidad no disponible"}, headers=headers)
            except (ValueError, TypeError, OverflowError) as error:
                logger.warning("HA entity metadata unavailable: %s", type(error).__name__)
                return unavailable(ObservationReason.INVALID)

        @self.app.post("/api/entities/{entity_id}/call", response_model=HomeActionResult)
        async def call_entity_service(
            entity_id: str, service: str, request: Request, data: LegacyHomeActionBody | None = None,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ) -> HomeActionResult:
            """Map legacy controls to the shared allowlisted, confirmed service."""
            if "home:control" not in principal.permissions:
                raise HTTPException(status_code=403, detail="home_control_denied")
            if self.home_actions is None:
                raise HTTPException(status_code=503, detail="Home Assistant no disponible")
            if data is None:
                raise HTTPException(status_code=422, detail="request_id_required")
            return await execute_home_action(
                self.home_actions,
                HomeActionRequest(entity_id, service, data.request_id, data.brightness_pct), principal,
                current=session_guard(request, frozenset({'home:control'}), principal),
            )

        # ==================== Servicios disponibles ====================

        @self.app.get("/api/services", dependencies=[Depends(require_dashboard_session)])
        async def list_services():
            """Listar servicios disponibles por dominio"""
            if not self.ha:
                return {}
            return await self.ha.get_services_by_domain()

        # ==================== Templates de triggers/actions ====================

        @self.app.get("/api/templates/triggers", dependencies=[Depends(require_dashboard_session)])
        async def get_trigger_templates():
            """Obtener templates de triggers"""
            return [
                {
                    "type": "time",
                    "name": "Hora específica",
                    "description": "Ejecutar a una hora del día",
                    "config_schema": {
                        "at": {"type": "time", "required": True, "label": "Hora"}
                    }
                },
                {
                    "type": "presence_enter",
                    "name": "Al llegar",
                    "description": "Cuando alguien llega a casa",
                    "config_schema": {
                        "user_id": {"type": "select", "required": False, "label": "Usuario"},
                        "zone": {"type": "string", "default": "home", "label": "Zona"}
                    }
                },
                {
                    "type": "presence_leave",
                    "name": "Al salir",
                    "description": "Cuando alguien sale de casa",
                    "config_schema": {
                        "user_id": {"type": "select", "required": False, "label": "Usuario"},
                        "zone": {"type": "string", "default": "home", "label": "Zona"}
                    }
                },
                {
                    "type": "presence_home",
                    "name": "Primera llegada",
                    "description": "Cuando la primera persona llega a casa",
                    "config_schema": {}
                },
                {
                    "type": "presence_away",
                    "name": "Casa vacía",
                    "description": "Cuando la última persona sale",
                    "config_schema": {}
                },
                {
                    "type": "sunrise",
                    "name": "Amanecer",
                    "description": "Al salir el sol",
                    "config_schema": {
                        "offset_minutes": {"type": "number", "default": 0, "label": "Offset (minutos)"}
                    }
                },
                {
                    "type": "sunset",
                    "name": "Atardecer",
                    "description": "Al ponerse el sol",
                    "config_schema": {
                        "offset_minutes": {"type": "number", "default": 0, "label": "Offset (minutos)"}
                    }
                },
                {
                    "type": "device_state",
                    "name": "Estado de dispositivo",
                    "description": "Cuando un dispositivo cambia de estado",
                    "config_schema": {
                        "entity_id": {"type": "entity", "required": True, "label": "Entidad"},
                        "to": {"type": "string", "required": True, "label": "Nuevo estado"}
                    }
                }
            ]

        @self.app.get("/api/templates/actions", dependencies=[Depends(require_dashboard_session)])
        async def get_action_templates():
            """Obtener templates de acciones"""
            return [
                {
                    "type": "ha_service",
                    "name": "Servicio HA",
                    "description": "Ejecutar servicio de Home Assistant",
                    "config_schema": {
                        "entity_id": {"type": "entity", "required": True},
                        "domain": {"type": "string", "required": True},
                        "service": {"type": "string", "required": True},
                        "data": {"type": "object", "default": {}}
                    }
                },
                {
                    "type": "ha_scene",
                    "name": "Activar escena",
                    "description": "Activar una escena de Home Assistant",
                    "config_schema": {
                        "scene_id": {"type": "entity", "domain": "scene", "required": True}
                    }
                },
                {
                    "type": "delay",
                    "name": "Esperar",
                    "description": "Pausar ejecución",
                    "config_schema": {
                        "seconds": {"type": "number", "required": True, "label": "Segundos"}
                    }
                },
                {
                    "type": "tts_speak",
                    "name": "Hablar",
                    "description": "Reproducir mensaje de voz",
                    "config_schema": {
                        "text": {"type": "string", "required": True, "label": "Texto"},
                        "zone": {"type": "string", "required": False, "label": "Zona"}
                    }
                },
                {
                    "type": "spotify_play",
                    "name": "Reproducir música",
                    "description": "Reproducir en Spotify",
                    "config_schema": {
                        "playlist": {"type": "string", "label": "Playlist"},
                        "mood": {"type": "select", "options": ["relax", "energy", "focus", "party"]},
                        "zone": {"type": "string", "label": "Zona"}
                    }
                },
                {
                    "type": "notify",
                    "name": "Notificación",
                    "description": "Enviar notificación",
                    "config_schema": {
                        "message": {"type": "string", "required": True},
                        "title": {"type": "string", "default": "KZA"}
                    }
                }
            ]

        # ==================== Lists ====================
        # T035-U1: authority and scope come from the session, never from the client.

        @self.app.get("/api/lists")
        async def get_lists(request: Request, scope: Literal["shared", "personal"] | None = None,
                            principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Shared lists, plus the session person's own lists."""
            if (rejected := _query_rejection(request, "scope")) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.list_manager:
                return _task_error(503, "Lists not configured", write=False, handled=False)
            if scope == "personal" and tasks.person is None:
                return _task_error(403, "personal_session_required", write=False)
            try:
                lists = await self.list_manager.get_all_lists(tasks.person,
                    shared_only=tasks.shared_only or scope == "shared",
                    guard=session_guard(request, tasks.permissions(write=False), principal))
            except PermissionError:
                return _task_error(403, "list_read_denied", write=False)
            if scope is not None:
                wanted = OwnerType.SHARED if scope == "shared" else OwnerType.USER
                lists = [lst for lst in lists if lst.owner_type is wanted]
            return JSONResponse(content=[_list_to_dict(lst) for lst in lists], headers=_NO_STORE)

        @self.app.post("/api/lists")
        async def create_list(data: ListCreateModel, request: Request,
                              principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """A shared list, or a personal list of the session person."""
            if (rejected := _query_rejection(request, write=True)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.list_manager:
                return _task_error(503, "Lists not configured", write=True, handled=False)
            if not data.shared and tasks.person is None:
                return _task_error(403, "personal_session_required", write=True)
            try:
                lst = await self.list_manager.create_list(None if data.shared else tasks.person, data.name,
                    data.shared, shared_only=data.shared,
                    guard=session_guard(request, tasks.permissions(write=True), principal))
            except PermissionError:
                return _task_error(403, "list_write_denied", write=True)
            return JSONResponse(content={**_list_to_dict(lst), "outcome": _committed()}, headers=_NO_STORE)

        @self.app.get("/api/lists/{list_id}/items")
        async def get_list_items(list_id: str, request: Request,
                                 principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Items of one list in the session scope; any other list is not found."""
            if (rejected := _query_rejection(request)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.list_manager:
                return _task_error(503, "Lists not configured", write=False, handled=False)
            try:
                items = await self.list_manager.get_items_in_scope(list_id, shared_only=tasks.shared_only,
                    owner=tasks.person, guard=session_guard(request, tasks.permissions(write=False), principal))
            except PermissionError:
                return _task_error(403, "list_read_denied", write=False)
            if items is None:
                return _task_error(404, "list_not_found", write=False)
            return JSONResponse(content=[{"id": i.id, "text": i.text, "completed": i.completed} for i in items],
                                headers=_NO_STORE)

        @self.app.post("/api/lists/{list_id}/items")
        async def add_list_item(list_id: str, data: ListItemCreateModel, request: Request,
                                principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Add an item to one list in the session scope."""
            if (rejected := _query_rejection(request, write=True)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.list_manager:
                return _task_error(503, "Lists not configured", write=True, handled=False)
            try:
                item = await self.list_manager.add_item_to_list_id(list_id, data.text, tasks.person,
                    shared_only=tasks.shared_only, owner=tasks.person,
                    guard=session_guard(request, tasks.permissions(write=True), principal))
            except PermissionError:
                return _task_error(403, "list_write_denied", write=True)
            if item is None:
                return _task_error(404, "list_not_found", write=True)
            return JSONResponse(content={"id": item.id, "text": item.text, "completed": item.completed,
                                         "list_id": item.list_id, "outcome": _committed()}, headers=_NO_STORE)

        @self.app.delete("/api/lists/{list_id}/items/{item_id}")
        async def delete_list_item(list_id: str, item_id: str, request: Request,
                                   principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Remove an item of one list in the session scope."""
            if (rejected := _query_rejection(request, write=True)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.list_manager:
                return _task_error(503, "Lists not configured", write=True, handled=False)
            try:
                removed = await self.list_manager.remove_item_by_id(item_id, list_id=list_id,
                    shared_only=tasks.shared_only, owner=tasks.person,
                    guard=session_guard(request, tasks.permissions(write=True), principal))
            except PermissionError:
                return _task_error(403, "list_write_denied", write=True)
            if removed is not True:
                return _task_error(404, "List item not found", write=True)
            return JSONResponse(content={"status": "deleted", "outcome": _committed()}, headers=_NO_STORE)

        # ==================== Reminders ====================

        @self.app.get("/api/reminders")
        async def get_reminders(request: Request, scope: Literal["shared", "personal"] | None = None,
                                principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Active shared reminders with their home actions, plus the session person's own."""
            if (rejected := _query_rejection(request, "scope")) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.reminder_manager:
                return _task_error(503, "Reminders not configured", write=False, handled=False)
            if scope == "personal" and tasks.person is None:
                return _task_error(403, "personal_session_required", write=False)
            try:
                reminders = await self.reminder_manager.get_active(tasks.person,
                    shared_only=tasks.shared_only or scope == "shared", include_shared=scope is None,
                    guard=session_guard(request, tasks.permissions(write=False), principal))
            except PermissionError:
                return _task_error(403, "reminder_read_denied", write=False)
            if scope is not None:
                wanted = ReminderScope.SHARED if scope == "shared" else ReminderScope.PRIVATE
                reminders = [r for r in reminders if r.scope is wanted]
            rooms = await self._action_rooms(reminders)
            return JSONResponse(content=[_reminder_to_dict(r, rooms) for r in reminders], headers=_NO_STORE)

        @self.app.post("/api/reminders")
        async def create_reminder(data: ReminderCreateModel, request: Request,
                                  principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """A shared household reminder, or a private one of the session person."""
            if (rejected := _query_rejection(request, write=True)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.reminder_manager:
                return _task_error(503, "Reminders not configured", write=True, handled=False)
            if not _valid_trigger_at(data.trigger_at):
                return _task_error(422, "invalid_trigger_at", write=True)
            shared = data.scope is ReminderScope.SHARED
            if not shared and tasks.person is None:
                return _task_error(403, "personal_session_required", write=True)
            owner = HomeActor.household().subject if shared else tasks.person
            try:
                r = await self.reminder_manager.create(owner, data.text, data.trigger_at,
                    data.recurrence, data.ha_actions, scope=data.scope,
                    guard=session_guard(request, tasks.permissions(write=True), principal))
            except PermissionError:
                return _task_error(403, "reminder_write_denied", write=True)
            except ValueError:
                return _task_error(422, "invalid_reminder_action", write=True)
            return JSONResponse(content={"id": r.id, "text": r.text, "trigger_at": r.trigger_at,
                                         "recurrence": r.recurrence, "state": r.state, "scope": r.scope,
                                         "outcome": _committed()}, headers=_NO_STORE)

        @self.app.delete("/api/reminders/{reminder_id}")
        async def delete_reminder(reminder_id: str, request: Request,
                                  principal: DashboardPrincipal = Depends(require_dashboard_session)):
            """Cancel one active reminder in the session scope; its home actions never run."""
            if (rejected := _query_rejection(request, write=True)) is not None:
                return rejected
            tasks = TaskScope.of(principal)
            if not self.reminder_manager:
                return _task_error(503, "Reminders not configured", write=True, handled=False)
            guard = session_guard(request, tasks.permissions(write=True), principal)
            try:
                cancelled = await self.reminder_manager.cancel_by_id(reminder_id, shared_only=tasks.shared_only,
                                                                     owner=tasks.person, guard=guard)
                existing = None if cancelled is True else await self.reminder_manager.get_in_scope(
                    reminder_id, shared_only=tasks.shared_only, owner=tasks.person, guard=guard)
            except PermissionError:
                return _task_error(403, "reminder_write_denied", write=True)
            if cancelled is not True:
                if existing is not None:
                    return _task_error(409, "reminder_not_active", write=True)
                return _task_error(404, "Reminder not found", write=True)
            return JSONResponse(content={"status": "cancelled", "outcome": _committed()}, headers=_NO_STORE)

        @self.app.get("/api/reminders/notices")
        async def get_reminder_notices(request: Request):
            """Shared reminder notices for the household: pending occurrences and 48 h of history."""
            if self.reminder_scheduler is None:
                raise HTTPException(status_code=503, detail="reminder_notices_unavailable",
                                    headers={"Cache-Control": "no-store"})
            try:
                entries = await self.reminder_scheduler.notice_entries(
                    guard=session_guard(request, frozenset({'tasks:read'})))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="reminder_read_denied") from error
            now = datetime.now(UTC)
            observation = Observation.from_read(ObservationSource.REMINDER_STORE, available=True,
                                                observed_at=now, attempted_at=now, ttl_s=6)
            return JSONResponse(
                content={"source": ObservationSource.REMINDER_STORE.value, "observed_at": now.isoformat(),
                         "counts": _notice_counts(entries), "notices": [_notice_to_dict(e) for e in entries]},
                headers=observation_headers(observation, now=now))

        @self.app.post("/api/reminders/notices/dismiss")
        async def dismiss_reminder_notice(data: ReminderNoticeDismissModel, request: Request):
            """Household dismiss: a pending notice counts as acknowledged and advances without actions."""
            if self.reminder_scheduler is None:
                raise HTTPException(status_code=503, detail="reminder_notices_unavailable")
            try:
                result = await self.reminder_scheduler.dismiss_notice(
                    data.reminder_id, data.trigger_at, guard=session_guard(request, frozenset({'tasks:write'})))
            except PermissionError as error:
                raise HTTPException(status_code=403, detail="reminder_write_denied") from error
            if result is NoticeDismissal.BUSY:
                raise HTTPException(status_code=409, detail="reminder_notice_busy")
            if result is NoticeDismissal.NOT_FOUND:
                raise HTTPException(status_code=404, detail="reminder_notice_not_found")
            return JSONResponse(content={"status": result.value, "outcome": TaskOutcome(
                TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
                observation_scope="software_state").to_dict()}, headers={"Cache-Control": "no-store"})

        # ==================== WebSocket para tiempo real ====================

        @self.app.websocket("/ws")
        async def websocket_endpoint(
            websocket: WebSocket,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Subscribe an authorized reader to private dashboard events."""
            if "home:read" not in principal.permissions:
                raise WebSocketException(code=1008, reason="permission denied")
            await websocket.accept()
            self._ws_clients.append(websocket)
            logger.info(f"WebSocket client conectado ({len(self._ws_clients)} total)")

            try:
                while True:
                    # Recibir mensajes (ping/pong, comandos)
                    data = await websocket.receive_text()

                    try:
                        msg = json.loads(data)

                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})

                        elif msg.get("type") == "subscribe":
                            # Suscribirse a eventos específicos
                            pass

                    except json.JSONDecodeError:
                        pass

            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)
                logger.info("WebSocket client disconnected (%s remaining)", len(self._ws_clients))

        # ==================== Observability ====================

        @self.app.get("/api/health")
        async def health_check():
            """Overall system health with all subsystems.

            Returns only non-sensitive operational status.  Tokens and
            credentials are never included.
            """
            if self.health_aggregator is None:
                return sanitize_dict({
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                    "subsystems": [],
                })

            report = self.health_aggregator.get_system_health()
            return sanitize_dict({
                "status": str(report.status),
                "timestamp": datetime.now().isoformat(),
                "subsystems": [
                    {
                        "name": s.name,
                        "status": str(s.status),
                        "detail": s.detail,
                        "extra": s.extra,
                    }
                    for s in report.subsystems
                ],
            })

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Latency percentiles, queue depth, and command count."""
            if self.health_aggregator is None:
                return {
                    "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
                    "queue_depth": 0,
                    "command_count": 0,
                    "active_zones": 0,
                }
            return self.health_aggregator.get_metrics()

        @self.app.get("/api/subsystems")
        async def get_subsystems():
            """Per-subsystem health details."""
            if self.health_aggregator is None:
                return {"subsystems": []}

            report = self.health_aggregator.get_system_health()
            return {
                "subsystems": [
                    {
                        "name": s.name,
                        "status": str(s.status),
                        "detail": s.detail,
                        "extra": s.extra,
                    }
                    for s in report.subsystems
                ],
            }

        @self.app.get("/api/failures")
        async def get_failures(limit: int = 50):
            """Recent failures, newest first."""
            if self.health_aggregator is None:
                return {"failures": []}

            failures = self.health_aggregator.get_recent_failures(limit=limit)
            return {"failures": failures}

        @self.app.get("/api/reminders/status")
        async def get_reminders_status(request: Request):
            """Describe only the shared jobs eligible for household announcements."""
            result = {
                "pending_count": 0, "next_trigger_at": None,
                "scheduler_running": False, "output_zone_configured": False,
                "notices": {"pending": 0, "pending_not_delivered": 0, "confirmed": 0, "uncertain": 0},
            }
            if self.reminder_scheduler is not None:
                result["scheduler_running"] = getattr(self.reminder_scheduler, "_running", False)
                # Without a zone no shared reminder is ever announced (T034-U1, CORE-06).
                result["output_zone_configured"] = getattr(self.reminder_scheduler, "output_zone", None) is not None
                # T034-U2: durable notice receipts, not in-process retry counters.
                try:
                    result["notices"] = _notice_counts(await self.reminder_scheduler.notice_entries(
                        guard=session_guard(request, frozenset({'diagnostics:read'}))))
                except PermissionError as error:
                    raise HTTPException(status_code=403, detail="reminder_read_denied") from error
            if self.reminder_manager is not None:
                store = getattr(self.reminder_manager, "_store", None)
                if store is not None:
                    guard = session_guard(request, frozenset({'diagnostics:read'}))
                    try:
                        pending = await store.get_next_pending(shared_only=True, guard=guard)
                        result["pending_count"] = await store.count_active(shared_only=True, guard=guard)
                    except PermissionError as error:
                        raise HTTPException(status_code=403, detail="reminder_read_denied") from error
                    if pending is not None:
                        result["next_trigger_at"] = pending.trigger_at
            return result

        # ==================== Observability (KZA Dashboard wireframe) ====================

        register_observability_routes(
            self.app,
            event_bus=self.event_bus,
            ha_client=self.ha,
            llm_router=self.llm_router,
            user_manager=self.user_manager,
            alert_manager=self.alert_manager,
            zone_manager=self.zone_manager,
            event_logger=self.event_logger,
            llm_metrics=self.llm_metrics,
            use_mocks=self.observability_use_mocks,
            capabilities=getattr(self.home_catalog, "capabilities", None),
            admin_work=self.admin_work,
        )

        # ==================== Admin (mutativos, auth required) ============

        register_admin_routes(
            self.app,
            user_manager=self.user_manager,
            speaker_identifier=self.speaker_identifier,
            alert_manager=self.alert_manager,
            voice_registry=self.voice_registry,
            admin_work=self.admin_work,
            model_lock=self.speaker_model_lock,
            configuration_changes=self.configuration_changes,
            consent_gate=self.consent_gate,
            observability_demo=self.observability_use_mocks,
        )
        register_memory_routes(self.app, manager=self.memory_manager)
        register_memory_voice_routes(self.app, manager=self.memory_manager, room_available=lambda room_id: self._voice_room_available(room_id))
        register_voice_routes(self.app,voices=self.voice_registry)
        register_evaluation_routes(self.app,voices=self.voice_registry)
        register_credential_routes(self.app,work=self.admin_work,voices=self.voice_registry)
        register_ambient_routes(self.app,control=self.ambient_control)
        register_capture_routes(self.app, control=self.capture_control)
        register_capture_browser_routes(self.app, control=self.capture_control)

        register_home_routes(
            self.app, catalog=self.home_catalog, actions=self.home_actions,
            microphone_snapshot_provider=self.microphone_snapshot_provider,
        )

        # ==================== Static files (Frontend) ====================

        if self.home_enabled:
            register_home_frontend(self.app)
        else:
            register_home_disabled_root(self.app)

        # Servir frontend si existe: /obs/ (panel técnico por URL directa). El
        # dashboard heredado de la raíz se retiró en T040/T041 U4.
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
        if os.path.exists(frontend_path):
            self.app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

    def _voice_room_available(self, room_id: str) -> bool:
        """U3b-4a-2: a verification window needs a multi-room microphone that is capturing now.

        The room must be bound to a fresh microphone snapshot whose capture is
        waiting or receiving, and capture must not be paused. Anything unknown
        is unavailable.
        """
        from src.audio.microphone_status import CaptureState

        provider = self.microphone_snapshot_provider
        snapshot = provider() if provider is not None else None
        if snapshot is None or snapshot.stale:
            return False
        if self.capture_control is not None and self.capture_control.capture_allowed is not True:
            return False
        return any(mic.room_id == room_id and mic.capture.state in (CaptureState.WAITING, CaptureState.RECEIVING)
                   for mic in snapshot.microphones)

    def _register_routine_exception_handlers(self) -> None:
        """Translate domain and unavailable-storage failures without private details."""
        @self.app.exception_handler(RoutineNotFound)
        async def routine_not_found(request, error):
            return JSONResponse(status_code=404, content={"detail": "routine_not_found"})

        @self.app.exception_handler(RoutineValidationError)
        async def routine_invalid(request, error):
            return JSONResponse(status_code=422, content={"detail": "invalid_routine"})

        @self.app.exception_handler(RoutineConflict)
        async def routine_conflict(request, error):
            if error.code == "scheduler_stopping":
                return JSONResponse(status_code=503, content={"detail": "routines unavailable"})
            return JSONResponse(status_code=409, content={"detail": error.code})

        @self.app.exception_handler(aiosqlite.Error)
        async def routine_storage_failure(request, error):
            return JSONResponse(status_code=503, content={"detail": "routines unavailable"})

        @self.app.exception_handler(RuntimeError)
        async def routine_not_initialized(request, error):
            if str(error) != "RoutineStore must be initialized":
                raise error
            return JSONResponse(status_code=503, content={"detail": "routines unavailable"})

    def _require_routines(self, principal: DashboardPrincipal, permissions: Set[str]):
        """Check server permissions and require an initialized routines facade."""
        if not permissions <= principal.permissions:
            raise HTTPException(status_code=403, detail="permission denied")
        if self.scheduler is None:
            raise HTTPException(status_code=503, detail="routines unavailable")
        return self.scheduler

    @staticmethod
    def _require_routine_actor(principal: DashboardPrincipal) -> str:
        if principal.routine_actor is None:
            raise HTTPException(status_code=403, detail="dashboard_permission_denied")
        return principal.routine_actor

    @staticmethod
    def _can_read_routine_run(principal: DashboardPrincipal, *, actor: str, source: str) -> bool:
        return (DashboardPermission.ROUTINES_MANAGE in principal.permissions
                or (source == "manual" and actor == principal.routine_actor))

    async def _routine_to_response(self, routine: ScheduledRoutine, principal: DashboardPrincipal) -> dict:
        """Project execution and voice brake receipts independently."""
        latest_run = await self.scheduler.get_latest_run(routine.routine_id)
        if latest_run is not None and not self._can_read_routine_run(
            principal, actor=latest_run.actor, source=latest_run.source,
        ):
            latest_run = None
        brake = await self.scheduler.get_latest_brake(routine.routine_id)
        return {
            **routine_to_dict(routine),
            "latest_run": run_to_dict(latest_run) if latest_run else None,
            "latest_voice_brake": ({
                "receipt_id": brake.receipt_id, "source": HOUSEHOLD_VOICE,
                "operation": brake.operation.value, "status": brake.status.value,
            } if brake else None),
        }

    async def _broadcast_ws(self, message: dict) -> None:
        """Send routine events through each subscriber's current authority."""
        if not self._ws_clients:
            return

        for client in self._ws_clients[:]:
            try:
                # A long-lived subscription cannot retain access after the
                # configured credential is removed or rotated.
                principal = await require_dashboard_session(
                    client, client.headers.get("authorization"), client.cookies.get(COOKIE_NAME),
                )
                response = message
                if message["type"] in {"routine_created", "routine_updated"}:
                    routine = message["routine"]
                    latest_run = routine["latest_run"]
                    if latest_run is not None and not self._can_read_routine_run(
                        principal, actor=latest_run["actor"], source=latest_run["source"],
                    ):
                        response = {**message, "routine": {**routine, "latest_run": None}}
                await client.send_json(response)
            except WebSocketException:
                if client in self._ws_clients:
                    self._ws_clients.remove(client)
                try:
                    await client.close(code=1008)
                except Exception:
                    logger.debug("WebSocket unavailable during session invalidation")
            except Exception:
                if client in self._ws_clients:
                    self._ws_clients.remove(client)

    async def start(self):
        """Iniciar servidor API"""
        import uvicorn

        if self.host == "0.0.0.0":  # nosec B104 -- no es el bind, es la comparación que emite el warning de LAN antes de que uvicorn abra el socket (el bind real usa self.host, una variable, no el literal)
            logger.warning(
                "Dashboard API binding to 0.0.0.0 — accessible from LAN. "
                "Set dashboard.host to '127.0.0.1' in settings.yaml for local-only access."
            )

        logger.info(f"Dashboard API starting on {self.host}:{self.port}")

        # KZA alone owns SIGTERM (U3b-0d1): uvicorn neither replaces the
        # handler nor re-raises a captured signal, and open WebSockets are
        # cancelled after a bounded graceful period.
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
        self._server = _owned_signal_server(uvicorn)(config)
        self._serve_task = asyncio.current_task()
        try:
            await self._server.serve()
        finally:
            self._serve_task = None

    async def stop(self) -> None:
        """Retain drains through cancellation; retry only failed resource cleanup."""
        if self._server is not None:
            self._server.should_exit = True
        previous = self._stop_task
        retry = previous is not None and previous.done() and (
            previous.cancelled() or previous.exception() is not None)
        if previous is None or retry:
            native_drain = self.admin_work.begin_close()
            serving = self._serve_task
            if serving is asyncio.current_task():
                serving = None

            async def finish():
                observed = output_registration.account_publisher(self)
                if observed is not None:
                    # Publish the original _stop_task identity first, even under
                    # an eager task factory, before its completion observation.
                    await asyncio.sleep(0)
                async with AsyncExitStack() as closing:
                    closing.push_async_callback(self.app.state.observability_monitors.aclose)
                    if self._owns_home_catalog:
                        closing.push_async_callback(self.home_catalog.aclose)
                    await asyncio.shield(native_drain)
                    if serving is not None:
                        await asyncio.shield(serving)
                        await self._await_cancelled_connections()
                if observed is not None:
                    # Only after the whole original stack exited successfully.
                    await observed.account_publisher_drained(self)

            self._stop_task = asyncio.create_task(finish(), context=Context())
        stopping = self._stop_task
        cancelled = False
        while not stopping.done():
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                cancelled = True
        stopping.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _await_cancelled_connections(self) -> None:
        """Await connection tasks uvicorn cancelled but did not await (review D6)."""
        server = self._server
        state = getattr(server, "server_state", None)
        pending = [task for task in getattr(state, "tasks", ()) if not task.done()]
        if not pending:
            return
        _, remaining = await asyncio.wait(pending, timeout=CONNECTION_DRAIN_SECONDS)
        if remaining:
            logger.warning("Dashboard stop: %d connection task(s) still running after %.0fs",
                           len(remaining), CONNECTION_DRAIN_SECONDS)

    def run_sync(self):
        """Ejecutar servidor de forma síncrona (para testing)"""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
