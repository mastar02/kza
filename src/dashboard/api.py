"""
KZA Dashboard API
API REST con FastAPI para gestión de rutinas, presencia y configuración
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Literal, Self

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.sanitize import sanitize_dict
from src.audio.microphone_status import MicrophonesSnapshot
from src.dashboard.admin import register_admin_routes
from src.dashboard.auth import COOKIE_NAME, DashboardPrincipal, require_dashboard_session
from src.dashboard.home_api import (
    LegacyHomeActionBody, execute_home_action, register_home_frontend, register_home_routes,
)
from src.dashboard.home_catalog import HomeCatalogService
from src.dashboard.live_event_bus import LiveEventBus
from src.dashboard.observability import register_observability_routes
from src.rooms.room_context import RoomConfig
from src.routines.models import (
    RoutineConflict, RoutineNotFound, RoutineSpec, RoutineValidationError, ScheduledRoutine,
    parse_routine_spec, routine_to_dict, run_to_dict,
)
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
    operation: Literal["turn_on", "turn_off", "brightness"]
    brightness_pct: int | None = None


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
    name: str
    user_id: str
    shared: bool = False


class ListItemCreateModel(BaseModel):
    text: str
    user_id: str | None = None


class ReminderCreateModel(BaseModel):
    user_id: str
    text: str
    trigger_at: float
    recurrence: str | None = None
    ha_actions: list[dict] | None = None


# ==================== Dashboard API ====================

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
        llm_metrics=None,
        observability_use_mocks: bool = True,
        *,
        home_rooms: Mapping[str, RoomConfig] | None = None,
        home_actions: HomeActionService | None = None,
        microphone_snapshot_provider: Callable[[], MicrophonesSnapshot | None] | None = None,
        home_catalog: HomeCatalogService | None = None,
        home_enabled: bool = True,
    ):
        self.scheduler = routine_scheduler
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
        self.llm_metrics = llm_metrics
        self.observability_use_mocks = observability_use_mocks
        self.microphone_snapshot_provider = microphone_snapshot_provider
        self.home_enabled = home_enabled
        self.home_catalog = home_catalog if home_catalog is not None else HomeCatalogService(ha_client, home_rooms or {})
        self.home_actions = home_actions if home_actions is not None else (
            HomeActionService(ha_client) if ha_client is not None else None
        )

        # FastAPI app
        self.app = FastAPI(
            title="KZA Dashboard API",
            description="API para gestión de asistente de voz KZA",
            version="1.0.0"
        )

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

        # Registrar rutas
        self._register_routine_exception_handlers()
        self._register_routes()

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
            return [await self._routine_to_response(routine) for routine in routines]

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
            return await self._routine_to_response(routine)

        @self.app.post("/api/routines")
        async def create_routine(
            routine: RoutineWriteModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Acknowledge a new definition only after its commit."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            created = await scheduler.register_routine(routine.to_spec(), actor=principal.subject)
            response = await self._routine_to_response(created)
            await self._broadcast_ws({"type": "routine_created", "routine": response})
            return response

        @self.app.put("/api/routines/{routine_id}")
        async def update_routine(
            routine_id: str, update: RoutineUpdateModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Persist a complete definition with compare-and-swap semantics."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            updated = await scheduler.update_routine(
                routine_id, update.to_spec(), expected_revision=update.expected_revision,
                actor=principal.subject,
            )
            response = await self._routine_to_response(updated)
            await self._broadcast_ws({"type": "routine_updated", "routine": response})
            return response

        @self.app.delete("/api/routines/{routine_id}")
        async def delete_routine(
            routine_id: str,
            expected_revision: str = Query(pattern=r"^[1-9][0-9]*$"),
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Delete only the idle revision the caller has actually read."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            await scheduler.unregister_routine(routine_id, expected_revision=int(expected_revision))
            await self._broadcast_ws({"type": "routine_deleted", "routine_id": routine_id})
            return {"success": True}

        @self.app.post("/api/routines/{routine_id}/execute", status_code=202)
        async def execute_routine(
            routine_id: str, body: ExecuteRoutineModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Return the durable claim while its scheduler-owned worker proceeds."""
            scheduler = self._require_routines(principal, {"routines:manage", "home:control"})
            run = await scheduler.execute_by_id(
                routine_id, actor=principal.subject, request_id=body.request_id,
            )
            return run_to_dict(run)

        @self.app.get("/api/routines/{routine_id}/runs/{run_id}")
        async def get_routine_run(
            routine_id: str, run_id: str,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Read the receipt and ordered per-action outcomes for this routine."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            return run_to_dict(await scheduler.get_run(routine_id, run_id))

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
            scheduler = self._require_routines(principal, {"routines:manage"})
            run = await scheduler.get_manual_attempt(
                routine_id, actor=principal.subject, request_id=request_id,
            )
            return run_to_dict(run)

        @self.app.post("/api/routines/{routine_id}/enable")
        async def enable_routine(
            routine_id: str, body: RoutineRevisionModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Enable an idle definition using its expected revision."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            routine = await scheduler.enable_routine(
                routine_id, expected_revision=body.expected_revision, actor=principal.subject,
            )
            return await self._routine_to_response(routine)

        @self.app.post("/api/routines/{routine_id}/disable")
        async def disable_routine(
            routine_id: str, body: RoutineRevisionModel,
            principal: DashboardPrincipal = Depends(require_dashboard_session),
        ):
            """Disable an idle definition using its expected revision."""
            scheduler = self._require_routines(principal, {"routines:manage"})
            routine = await scheduler.disable_routine(
                routine_id, expected_revision=body.expected_revision, actor=principal.subject,
            )
            return await self._routine_to_response(routine)

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
        async def get_entity(entity_id: str):
            """Obtener estado de entidad"""
            if not self.ha:
                raise HTTPException(status_code=503, detail="Home Assistant no disponible")

            state = self.ha.get_entity_state(entity_id)
            if not state:
                raise HTTPException(status_code=404, detail="Entidad no encontrada")

            return state

        @self.app.post("/api/entities/{entity_id}/call", response_model=HomeActionResult)
        async def call_entity_service(
            entity_id: str, service: str, data: LegacyHomeActionBody | None = None,
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

        @self.app.get("/api/lists")
        async def get_lists(user_id: str):
            if not self.list_manager:
                raise HTTPException(status_code=503, detail="Lists not configured")
            lists = await self.list_manager.get_all_lists(user_id)
            return [{"id": l.id, "name": l.name, "owner_type": l.owner_type, "owner_id": l.owner_id} for l in lists]

        @self.app.post("/api/lists")
        async def create_list(data: ListCreateModel):
            if not self.list_manager:
                raise HTTPException(status_code=503, detail="Lists not configured")
            lst = await self.list_manager.create_list(data.user_id, data.name, data.shared)
            return {"id": lst.id, "name": lst.name, "owner_type": lst.owner_type, "owner_id": lst.owner_id}

        @self.app.get("/api/lists/{list_id}/items")
        async def get_list_items(list_id: str):
            if not self.list_manager:
                raise HTTPException(status_code=503, detail="Lists not configured")
            items = await self.list_manager.get_items_by_list_id(list_id)
            return [{"id": i.id, "text": i.text, "completed": i.completed} for i in items]

        @self.app.post("/api/lists/{list_id}/items")
        async def add_list_item(list_id: str, data: ListItemCreateModel):
            if not self.list_manager:
                raise HTTPException(status_code=503, detail="Lists not configured")
            item = await self.list_manager.add_item_to_list_id(list_id, data.text, data.user_id)
            return {"id": item.id, "text": item.text, "completed": item.completed, "list_id": item.list_id}

        @self.app.delete("/api/lists/{list_id}/items/{item_id}")
        async def delete_list_item(list_id: str, item_id: str):
            if not self.list_manager:
                raise HTTPException(status_code=503, detail="Lists not configured")
            await self.list_manager.remove_item_by_id(item_id)
            return {"status": "deleted"}

        # ==================== Reminders ====================

        @self.app.get("/api/reminders")
        async def get_reminders(user_id: str):
            if not self.reminder_manager:
                raise HTTPException(status_code=503, detail="Reminders not configured")
            reminders = await self.reminder_manager.get_active(user_id)
            return [{"id": r.id, "text": r.text, "trigger_at": r.trigger_at, "recurrence": r.recurrence, "state": r.state} for r in reminders]

        @self.app.post("/api/reminders")
        async def create_reminder(data: ReminderCreateModel):
            if not self.reminder_manager:
                raise HTTPException(status_code=503, detail="Reminders not configured")
            r = await self.reminder_manager.create(data.user_id, data.text, data.trigger_at, data.recurrence, data.ha_actions)
            return {"id": r.id, "text": r.text, "trigger_at": r.trigger_at, "recurrence": r.recurrence, "state": r.state}

        @self.app.delete("/api/reminders/{reminder_id}")
        async def delete_reminder(reminder_id: str):
            if not self.reminder_manager:
                raise HTTPException(status_code=503, detail="Reminders not configured")
            await self.reminder_manager.cancel_by_id(reminder_id)
            return {"status": "cancelled"}

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
        async def get_reminders_status():
            """Reminder scheduler status: pending count, next trigger, delivery failures."""
            result = {
                "pending_count": 0,
                "next_trigger_at": None,
                "scheduler_running": False,
                "delivery_failures": 0,
            }

            if self.reminder_scheduler is not None:
                result["scheduler_running"] = getattr(
                    self.reminder_scheduler, "_running", False
                )
                retry_counts = getattr(
                    self.reminder_scheduler, "_retry_counts", {}
                )
                result["delivery_failures"] = sum(retry_counts.values())

            if self.reminder_manager is not None:
                store = getattr(self.reminder_manager, "_store", None)
                if store is not None:
                    next_pending = await store.get_next_pending()
                    if next_pending is not None:
                        result["next_trigger_at"] = next_pending.trigger_at

                    # Count all active reminders across users
                    try:
                        async with store._db.execute(
                            "SELECT COUNT(*) as cnt FROM reminders WHERE state = 'active'"
                        ) as cursor:
                            row = await cursor.fetchone()
                            result["pending_count"] = row[0] if row else 0
                    except Exception:
                        logger.debug("Could not query pending reminder count")

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
        )

        # ==================== Admin (mutativos, auth required) ============

        register_admin_routes(
            self.app,
            user_manager=self.user_manager,
            speaker_identifier=self.speaker_identifier,
            alert_manager=self.alert_manager,
        )

        register_home_routes(
            self.app, catalog=self.home_catalog, actions=self.home_actions,
            microphone_snapshot_provider=self.microphone_snapshot_provider,
        )

        # ==================== Static files (Frontend) ====================

        if self.home_enabled:
            register_home_frontend(self.app)

        # Servir frontend si existe (incluye /obs/ con el dashboard wireframe)
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
        if os.path.exists(frontend_path):
            self.app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

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

    def _require_routines(self, principal: DashboardPrincipal, permissions: set[str]):
        """Check server permissions and require an initialized routines facade."""
        if not permissions.issubset(principal.permissions):
            raise HTTPException(status_code=403, detail="permission denied")
        if self.scheduler is None:
            raise HTTPException(status_code=503, detail="routines unavailable")
        return self.scheduler

    async def _routine_to_response(self, routine: ScheduledRoutine) -> dict:
        """Serialize canonical top-level fields and the latest receipt via the facade."""
        latest_run = await self.scheduler.get_latest_run(routine.routine_id)
        return {**routine_to_dict(routine), "latest_run": run_to_dict(latest_run) if latest_run else None}

    async def _broadcast_ws(self, message: dict):
        """Enviar mensaje a todos los WebSocket clients"""
        if not self._ws_clients:
            return

        for client in self._ws_clients[:]:
            try:
                # A long-lived subscription cannot retain access after the
                # configured credential is removed or rotated.
                await require_dashboard_session(
                    client, client.headers.get("authorization"), client.cookies.get(COOKIE_NAME),
                )
                await client.send_json(message)
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

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.current_task()
        try:
            await self._server.serve()
        finally:
            self._serve_task = None

    async def stop(self) -> None:
        """Stop accepting HTTP requests and drain serving before shared services.

        Uvicorn owns connection/request shutdown. Do not cancel its task while
        controls may still be running, and do not close the injected HA client.
        """
        if self._server is not None:
            self._server.should_exit = True
        task = self._serve_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.shield(task)

    def run_sync(self):
        """Ejecutar servidor de forma síncrona (para testing)"""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
