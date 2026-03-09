"""
KZA Dashboard API
API REST con FastAPI para gestión de rutinas, presencia y configuración
"""

import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import asyncio
import json
import os

logger = logging.getLogger(__name__)


# ==================== Modelos Pydantic ====================

class TriggerModel(BaseModel):
    type: str  # time, presence_enter, presence_leave, sunrise, sunset, device_state
    config: dict = Field(default_factory=dict)
    description: str | None = ""


class ConditionModel(BaseModel):
    type: str  # time_range, state, presence, weekday
    config: dict = Field(default_factory=dict)


class ActionModel(BaseModel):
    type: str = "ha_service"
    entity_id: str
    domain: str | None = None
    service: str = "turn_on"
    data: dict = Field(default_factory=dict)
    description: str | None = ""


class RoutineCreateModel(BaseModel):
    name: str
    description: str | None = ""
    triggers: list[TriggerModel]
    conditions: list[ConditionModel] = Field(default_factory=list)
    actions: list[ActionModel]
    enabled: bool = True
    cooldown_seconds: int = 60
    tags: list[str] = Field(default_factory=list)


class RoutineUpdateModel(BaseModel):
    name: str | None = None
    description: str | None = None
    triggers: list[TriggerModel] | None = None
    conditions: list[ConditionModel] | None = None
    actions: list[ActionModel] | None = None
    enabled: bool | None = None
    cooldown_seconds: int | None = None
    tags: list[str] | None = None


class RoutineResponseModel(BaseModel):
    routine_id: str
    name: str
    description: str
    triggers: list[dict]
    conditions: list[dict]
    actions: list[dict]
    enabled: bool
    cooldown_seconds: int
    created_at: str
    created_by: str
    tags: list[str]
    last_executed: str | None = None
    execution_count: int = 0


class ExecuteRoutineModel(BaseModel):
    context: dict = Field(default_factory=dict)


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
        host: str = "0.0.0.0",
        port: int = 8080
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

        # FastAPI app
        self.app = FastAPI(
            title="KZA Dashboard API",
            description="API para gestión de asistente de voz KZA",
            version="1.0.0"
        )

        # CORS para desarrollo
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # En producción, restringir
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # WebSocket clients
        self._ws_clients: list[WebSocket] = []

        # Registrar rutas
        self._register_routes()

    def _register_routes(self):
        """Registrar todas las rutas de la API"""

        # ==================== Rutinas ====================

        @self.app.get("/api/routines", response_model=list[RoutineResponseModel])
        async def list_routines(
            enabled: bool | None = None,
            tag: str | None = None
        ):
            """Listar todas las rutinas"""
            routines = self.scheduler.get_all_routines()

            # Filtrar
            if enabled is not None:
                routines = [r for r in routines if r.enabled == enabled]
            if tag:
                routines = [r for r in routines if tag in r.tags]

            return [self._routine_to_response(r) for r in routines]

        @self.app.get("/api/routines/{routine_id}", response_model=RoutineResponseModel)
        async def get_routine(routine_id: str):
            """Obtener una rutina por ID"""
            routine = self.scheduler.get_routine(routine_id)
            if not routine:
                raise HTTPException(status_code=404, detail="Rutina no encontrada")
            return self._routine_to_response(routine)

        @self.app.post("/api/routines", response_model=RoutineResponseModel)
        async def create_routine(routine: RoutineCreateModel):
            """Crear nueva rutina"""
            from ..routines.routine_scheduler import ScheduledRoutine
            import uuid

            new_routine = ScheduledRoutine(
                routine_id=f"dash_{uuid.uuid4().hex[:8]}",
                name=routine.name,
                description=routine.description or "",
                triggers=[t.dict() for t in routine.triggers],
                conditions=[c.dict() for c in routine.conditions],
                actions=[a.dict() for a in routine.actions],
                enabled=routine.enabled,
                cooldown_seconds=routine.cooldown_seconds,
                created_at=datetime.now().isoformat(),
                created_by="dashboard",
                tags=routine.tags
            )

            self.scheduler.register_routine(new_routine)

            # Notificar a WebSocket clients
            await self._broadcast_ws({
                "type": "routine_created",
                "routine": self._routine_to_response(new_routine)
            })

            return self._routine_to_response(new_routine)

        @self.app.put("/api/routines/{routine_id}", response_model=RoutineResponseModel)
        async def update_routine(routine_id: str, update: RoutineUpdateModel):
            """Actualizar rutina existente"""
            routine = self.scheduler.get_routine(routine_id)
            if not routine:
                raise HTTPException(status_code=404, detail="Rutina no encontrada")

            # Actualizar campos proporcionados
            if update.name is not None:
                routine.name = update.name
            if update.description is not None:
                routine.description = update.description
            if update.triggers is not None:
                routine.triggers = [t.dict() for t in update.triggers]
            if update.conditions is not None:
                routine.conditions = [c.dict() for c in update.conditions]
            if update.actions is not None:
                routine.actions = [a.dict() for a in update.actions]
            if update.enabled is not None:
                routine.enabled = update.enabled
            if update.cooldown_seconds is not None:
                routine.cooldown_seconds = update.cooldown_seconds
            if update.tags is not None:
                routine.tags = update.tags

            # Notificar
            await self._broadcast_ws({
                "type": "routine_updated",
                "routine": self._routine_to_response(routine)
            })

            return self._routine_to_response(routine)

        @self.app.delete("/api/routines/{routine_id}")
        async def delete_routine(routine_id: str):
            """Eliminar rutina"""
            if not self.scheduler.unregister_routine(routine_id):
                raise HTTPException(status_code=404, detail="Rutina no encontrada")

            await self._broadcast_ws({
                "type": "routine_deleted",
                "routine_id": routine_id
            })

            return {"success": True}

        @self.app.post("/api/routines/{routine_id}/execute")
        async def execute_routine(routine_id: str, body: ExecuteRoutineModel):
            """Ejecutar rutina manualmente"""
            result = await self.scheduler.execute_by_id(routine_id, body.context)
            if result is None:
                raise HTTPException(status_code=404, detail="Rutina no encontrada")
            return {"success": True, "result": result}

        @self.app.post("/api/routines/{routine_id}/enable")
        async def enable_routine(routine_id: str):
            """Habilitar rutina"""
            if not self.scheduler.enable_routine(routine_id):
                raise HTTPException(status_code=404, detail="Rutina no encontrada")
            return {"success": True}

        @self.app.post("/api/routines/{routine_id}/disable")
        async def disable_routine(routine_id: str):
            """Deshabilitar rutina"""
            if not self.scheduler.disable_routine(routine_id):
                raise HTTPException(status_code=404, detail="Rutina no encontrada")
            return {"success": True}

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

        @self.app.get("/api/entities")
        async def list_entities(domain: str | None = None):
            """Listar entidades de Home Assistant"""
            if not self.ha:
                return []

            entities = self.ha.get_domotics_entities()

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

        @self.app.get("/api/entities/{entity_id}")
        async def get_entity(entity_id: str):
            """Obtener estado de entidad"""
            if not self.ha:
                raise HTTPException(status_code=503, detail="Home Assistant no disponible")

            state = self.ha.get_entity_state(entity_id)
            if not state:
                raise HTTPException(status_code=404, detail="Entidad no encontrada")

            return state

        @self.app.post("/api/entities/{entity_id}/call")
        async def call_entity_service(entity_id: str, service: str, data: dict = None):
            """Llamar servicio en entidad"""
            if not self.ha:
                raise HTTPException(status_code=503, detail="Home Assistant no disponible")

            domain = entity_id.split(".")[0]
            success = self.ha.call_service(domain, service, entity_id, data or {})

            return {"success": success}

        # ==================== Servicios disponibles ====================

        @self.app.get("/api/services")
        async def list_services():
            """Listar servicios disponibles por dominio"""
            if not self.ha:
                return {}
            return self.ha.get_services_by_domain()

        # ==================== Templates de triggers/actions ====================

        @self.app.get("/api/templates/triggers")
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

        @self.app.get("/api/templates/actions")
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
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket para actualizaciones en tiempo real"""
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
                self._ws_clients.remove(websocket)
                logger.info(f"WebSocket client desconectado ({len(self._ws_clients)} restantes)")

        # ==================== Observability ====================

        @self.app.get("/api/health")
        async def health_check():
            """Overall system health with all subsystems."""
            if self.health_aggregator is None:
                return {
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                    "subsystems": [],
                }

            report = self.health_aggregator.get_system_health()
            return {
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
            }

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

        # ==================== Static files (Frontend) ====================

        # Servir frontend si existe
        frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
        if os.path.exists(frontend_path):
            self.app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

    def _routine_to_response(self, routine) -> dict:
        """Convertir ScheduledRoutine a modelo de respuesta"""
        return {
            "routine_id": routine.routine_id,
            "name": routine.name,
            "description": routine.description,
            "triggers": routine.triggers,
            "conditions": routine.conditions,
            "actions": routine.actions,
            "enabled": routine.enabled,
            "cooldown_seconds": routine.cooldown_seconds,
            "created_at": routine.created_at,
            "created_by": routine.created_by,
            "tags": routine.tags,
            "last_executed": routine.last_executed.isoformat() if routine.last_executed else None,
            "execution_count": routine.execution_count
        }

    async def _broadcast_ws(self, message: dict):
        """Enviar mensaje a todos los WebSocket clients"""
        if not self._ws_clients:
            return

        for client in self._ws_clients[:]:
            try:
                await client.send_json(message)
            except Exception:
                self._ws_clients.remove(client)

    async def start(self):
        """Iniciar servidor API"""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run_sync(self):
        """Ejecutar servidor de forma síncrona (para testing)"""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
