"""
Routine Scheduler
Ejecuta rutinas basadas en triggers de tiempo, presencia BLE y eventos
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Callable, Optional
import json

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    TIME = "time"                    # Hora específica
    TIME_RANGE = "time_range"        # Rango horario
    SUNRISE = "sunrise"
    SUNSET = "sunset"
    PRESENCE_ENTER = "presence_enter"  # Alguien entra a zona
    PRESENCE_LEAVE = "presence_leave"  # Alguien sale de zona
    PRESENCE_HOME = "presence_home"    # Primera persona llega a casa
    PRESENCE_AWAY = "presence_away"    # Última persona sale de casa
    PRESENCE_GUEST = "presence_guest"  # Dispositivo desconocido detectado
    DEVICE_STATE = "device_state"      # Cambio de estado de dispositivo
    VOICE_COMMAND = "voice_command"    # Ejecutar por voz
    VOICE_USER = "voice_user"          # Comando de voz de usuario específico
    VOICE_GUEST = "voice_guest"        # Comando de voz de invitado (no reconocido)
    WEBHOOK = "webhook"
    CRON = "cron"                      # Expresión cron


@dataclass
class ScheduledRoutine:
    """Rutina programada para ejecución"""
    routine_id: str
    name: str
    triggers: list[dict]
    conditions: list[dict]
    actions: list[dict]
    enabled: bool = True
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    cooldown_seconds: int = 60  # Evitar ejecuciones repetidas

    # Metadata
    created_at: str = ""
    created_by: str = "voice"  # "voice" o "dashboard"
    tags: list[str] = field(default_factory=list)

    # Permisos por usuario
    allowed_users: list[str] = field(default_factory=list)  # Vacío = todos
    owner_user_id: Optional[str] = None  # Quién creó la rutina
    guest_allowed: bool = True  # ¿Invitados pueden ejecutar?


class RoutineScheduler:
    """
    Scheduler de rutinas con múltiples tipos de triggers.
    Integra con presencia BLE y eventos de Home Assistant.
    """

    def __init__(
        self,
        routine_executor,
        presence_detector=None,
        ha_client=None
    ):
        self.executor = routine_executor
        self.presence = presence_detector
        self.ha_client = ha_client

        # Rutinas registradas
        self._routines: dict[str, ScheduledRoutine] = {}

        # Estado
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._presence_task: Optional[asyncio.Task] = None

        # Cache de presencia anterior (para detectar cambios)
        self._last_presence_state: dict[str, bool] = {}
        self._last_home_occupied: bool = False

        # Callbacks para eventos externos
        self._event_handlers: dict[str, list[Callable]] = {}

        # Configuración de sun (ajustar según ubicación)
        self._sunrise_time = time(7, 0)  # Default 7:00 AM
        self._sunset_time = time(19, 30)  # Default 7:30 PM

    # ==================== Gestión de Rutinas ====================

    def register_routine(self, routine: ScheduledRoutine) -> bool:
        """Registrar una rutina para ejecución programada"""
        if not routine.routine_id:
            logger.error("Rutina sin ID")
            return False

        self._routines[routine.routine_id] = routine
        logger.info(f"Rutina registrada: {routine.name} ({routine.routine_id})")

        # Si tiene trigger de presencia, asegurar que tenemos detector
        has_presence_trigger = any(
            t.get("type", "").startswith("presence_")
            for t in routine.triggers
        )
        if has_presence_trigger and not self.presence:
            logger.warning(f"Rutina {routine.name} tiene trigger de presencia pero no hay detector BLE")

        return True

    def unregister_routine(self, routine_id: str) -> bool:
        """Eliminar una rutina del scheduler"""
        if routine_id in self._routines:
            del self._routines[routine_id]
            logger.info(f"Rutina eliminada: {routine_id}")
            return True
        return False

    def get_routine(self, routine_id: str) -> Optional[ScheduledRoutine]:
        """Obtener una rutina por ID"""
        return self._routines.get(routine_id)

    def get_all_routines(self) -> list[ScheduledRoutine]:
        """Obtener todas las rutinas"""
        return list(self._routines.values())

    def enable_routine(self, routine_id: str) -> bool:
        """Habilitar una rutina"""
        if routine_id in self._routines:
            self._routines[routine_id].enabled = True
            return True
        return False

    def disable_routine(self, routine_id: str) -> bool:
        """Deshabilitar una rutina"""
        if routine_id in self._routines:
            self._routines[routine_id].enabled = False
            return True
        return False

    # ==================== Loop Principal ====================

    async def start(self):
        """Iniciar el scheduler"""
        if self._running:
            return

        self._running = True
        logger.info("Iniciando RoutineScheduler...")

        # Task principal de tiempo
        self._scheduler_task = asyncio.create_task(self._time_loop())

        # Task de presencia si hay detector
        if self.presence:
            self._presence_task = asyncio.create_task(self._presence_loop())

        logger.info(f"RoutineScheduler activo con {len(self._routines)} rutinas")

    async def stop(self):
        """Detener el scheduler"""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._presence_task:
            self._presence_task.cancel()
            try:
                await self._presence_task
            except asyncio.CancelledError:
                pass

        logger.info("RoutineScheduler detenido")

    async def _time_loop(self):
        """Loop principal para triggers de tiempo"""
        last_check_minute = -1

        while self._running:
            try:
                now = datetime.now()
                current_minute = now.hour * 60 + now.minute

                # Solo verificar una vez por minuto
                if current_minute != last_check_minute:
                    last_check_minute = current_minute
                    await self._check_time_triggers(now)

                # Dormir hasta el próximo segundo
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en time_loop: {e}")
                await asyncio.sleep(5)

    async def _presence_loop(self):
        """Loop para triggers de presencia BLE - MÁXIMA VELOCIDAD"""
        while self._running:
            try:
                # PRIORIDAD: Velocidad de respuesta > recursos
                # Polling cada 0.5s para detección casi instantánea
                await asyncio.sleep(0.5)

                if not self.presence:
                    continue

                # Obtener estado actual de presencia
                current_presence = {}
                home_occupied = False

                for user_id in self.presence.get_tracked_users():
                    user_status = self.presence.get_user_presence(user_id)
                    if user_status:
                        is_home = user_status.is_home
                        current_presence[user_id] = is_home
                        if is_home:
                            home_occupied = True

                # Detectar cambios de presencia
                for user_id, is_home in current_presence.items():
                    was_home = self._last_presence_state.get(user_id, False)

                    if is_home and not was_home:
                        # Usuario llegó a casa
                        await self._trigger_presence_event(
                            TriggerType.PRESENCE_ENTER,
                            user_id=user_id,
                            zone="home"
                        )
                    elif not is_home and was_home:
                        # Usuario salió de casa
                        await self._trigger_presence_event(
                            TriggerType.PRESENCE_LEAVE,
                            user_id=user_id,
                            zone="home"
                        )

                # Detectar primera llegada / última salida
                if home_occupied and not self._last_home_occupied:
                    await self._trigger_presence_event(TriggerType.PRESENCE_HOME)
                elif not home_occupied and self._last_home_occupied:
                    await self._trigger_presence_event(TriggerType.PRESENCE_AWAY)

                # Detectar dispositivos desconocidos (invitados)
                unknown_devices = self.presence.get_unknown_devices() if hasattr(self.presence, 'get_unknown_devices') else []
                for device in unknown_devices:
                    if device.get("is_new", False):
                        await self._trigger_presence_event(
                            TriggerType.PRESENCE_GUEST,
                            user_id="guest",
                            zone="home"
                        )

                # Actualizar cache
                self._last_presence_state = current_presence
                self._last_home_occupied = home_occupied

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en presence_loop: {e}")
                await asyncio.sleep(10)

    # ==================== Verificación de Triggers ====================

    async def _check_time_triggers(self, now: datetime):
        """Verificar y ejecutar triggers de tiempo"""
        current_time = now.time()

        for routine in self._routines.values():
            if not routine.enabled:
                continue

            # Verificar cooldown
            if routine.last_executed:
                elapsed = (now - routine.last_executed).total_seconds()
                if elapsed < routine.cooldown_seconds:
                    continue

            for trigger in routine.triggers:
                trigger_type = trigger.get("type", "")
                should_execute = False

                if trigger_type == "time":
                    # Trigger de hora específica
                    trigger_time = self._parse_time(trigger.get("at", ""))
                    if trigger_time and self._times_match(current_time, trigger_time):
                        should_execute = True

                elif trigger_type == "sunrise":
                    if self._times_match(current_time, self._sunrise_time):
                        offset = trigger.get("offset_minutes", 0)
                        if offset == 0 or self._check_offset(now, offset):
                            should_execute = True

                elif trigger_type == "sunset":
                    if self._times_match(current_time, self._sunset_time):
                        offset = trigger.get("offset_minutes", 0)
                        if offset == 0 or self._check_offset(now, offset):
                            should_execute = True

                elif trigger_type == "time_range":
                    # Ejecutar si estamos en el rango (solo una vez al inicio)
                    start = self._parse_time(trigger.get("start", ""))
                    end = self._parse_time(trigger.get("end", ""))
                    if start and self._times_match(current_time, start):
                        should_execute = True

                if should_execute:
                    await self._execute_routine(routine, trigger)
                    break  # Solo ejecutar una vez por ciclo

    async def _trigger_presence_event(
        self,
        trigger_type: TriggerType,
        user_id: str = None,
        zone: str = "home"
    ):
        """Disparar evento de presencia y ejecutar rutinas correspondientes"""
        logger.info(f"Evento de presencia: {trigger_type.value} user={user_id} zone={zone}")

        for routine in self._routines.values():
            if not routine.enabled:
                continue

            for trigger in routine.triggers:
                t_type = trigger.get("type", "")

                # Verificar coincidencia de trigger
                matches = False

                if t_type == trigger_type.value:
                    # Verificar filtros opcionales
                    t_user = trigger.get("user_id")
                    t_zone = trigger.get("zone", "home")

                    if t_user and user_id and t_user != user_id:
                        continue
                    if t_zone != zone:
                        continue

                    matches = True

                if matches:
                    await self._execute_routine(routine, trigger, {
                        "user_id": user_id,
                        "zone": zone,
                        "trigger_type": trigger_type.value
                    })
                    break

    # ==================== Ejecución ====================

    def _check_user_permission(
        self,
        routine: ScheduledRoutine,
        user_id: str = None,
        is_guest: bool = False
    ) -> bool:
        """Verificar si el usuario tiene permiso para ejecutar la rutina"""
        # Si es invitado
        if is_guest or user_id == "guest":
            return routine.guest_allowed

        # Si la rutina tiene usuarios permitidos específicos
        if routine.allowed_users:
            return user_id in routine.allowed_users

        # Por defecto, todos los usuarios registrados pueden ejecutar
        return True

    async def _execute_routine(
        self,
        routine: ScheduledRoutine,
        trigger: dict,
        context: dict = None
    ):
        """Ejecutar una rutina"""
        try:
            context = context or {}
            user_id = context.get("user_id")
            is_guest = context.get("is_guest", False) or user_id == "guest"

            # Verificar permisos de usuario
            if not self._check_user_permission(routine, user_id, is_guest):
                logger.info(f"Usuario {user_id} sin permiso para rutina {routine.name}")
                return None

            logger.info(f"Ejecutando rutina: {routine.name} (user={user_id})")

            # Verificar condiciones
            if routine.conditions:
                conditions_met = await self._check_conditions(routine.conditions, context)
                if not conditions_met:
                    logger.debug(f"Condiciones no cumplidas para {routine.name}")
                    return

            # Ejecutar acciones
            results = await self.executor.execute_actions(routine.actions, context)

            # Actualizar estadísticas
            routine.last_executed = datetime.now()
            routine.execution_count += 1

            logger.info(f"Rutina {routine.name} ejecutada ({routine.execution_count}x)")

            return results

        except Exception as e:
            logger.error(f"Error ejecutando rutina {routine.name}: {e}")

    async def execute_by_name(self, name: str, context: dict = None) -> Optional[dict]:
        """Ejecutar rutina por nombre (para comandos de voz)"""
        name_lower = name.lower()

        for routine in self._routines.values():
            if routine.name.lower() == name_lower:
                return await self._execute_routine(routine, {}, context)

        # Buscar coincidencia parcial
        for routine in self._routines.values():
            if name_lower in routine.name.lower():
                return await self._execute_routine(routine, {}, context)

        logger.warning(f"Rutina no encontrada: {name}")
        return None

    async def execute_by_id(self, routine_id: str, context: dict = None) -> Optional[dict]:
        """Ejecutar rutina por ID"""
        routine = self._routines.get(routine_id)
        if routine:
            return await self._execute_routine(routine, {}, context)
        return None

    async def _check_conditions(self, conditions: list[dict], context: dict = None) -> bool:
        """Verificar condiciones para ejecución"""
        for condition in conditions:
            cond_type = condition.get("type", "")

            if cond_type == "time_range":
                # Solo ejecutar en cierto rango horario
                now = datetime.now().time()
                start = self._parse_time(condition.get("start", "00:00"))
                end = self._parse_time(condition.get("end", "23:59"))

                if start and end:
                    if start <= end:
                        if not (start <= now <= end):
                            return False
                    else:  # Rango que cruza medianoche
                        if not (now >= start or now <= end):
                            return False

            elif cond_type == "state":
                # Verificar estado de entidad
                if self.ha_client:
                    entity_id = condition.get("entity_id")
                    expected_state = condition.get("state")

                    state = self.ha_client.get_entity_state(entity_id)
                    if state and state.get("state") != expected_state:
                        return False

            elif cond_type == "presence":
                # Verificar presencia de usuario
                if self.presence:
                    user_id = condition.get("user_id")
                    expected = condition.get("is_home", True)

                    user_status = self.presence.get_user_presence(user_id)
                    if user_status and user_status.is_home != expected:
                        return False

            elif cond_type == "weekday":
                # Solo ciertos días de la semana
                allowed_days = condition.get("days", [0, 1, 2, 3, 4, 5, 6])
                if datetime.now().weekday() not in allowed_days:
                    return False

        return True

    # ==================== Utilidades ====================

    def _parse_time(self, time_str: str) -> Optional[time]:
        """Parsear string de tiempo"""
        if not time_str:
            return None

        try:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            second = int(parts[2]) if len(parts) > 2 else 0
            return time(hour, minute, second)
        except (ValueError, IndexError):
            return None

    def _times_match(self, t1: time, t2: time) -> bool:
        """Verificar si dos tiempos coinciden (mismo minuto)"""
        return t1.hour == t2.hour and t1.minute == t2.minute

    def _check_offset(self, now: datetime, offset_minutes: int) -> bool:
        """Verificar offset de tiempo"""
        # TODO: Implementar offset para sunrise/sunset
        return True

    def update_sun_times(self, sunrise: time, sunset: time):
        """Actualizar tiempos de amanecer/atardecer"""
        self._sunrise_time = sunrise
        self._sunset_time = sunset
        logger.debug(f"Sun times actualizados: sunrise={sunrise}, sunset={sunset}")

    # ==================== Persistencia ====================

    def to_dict(self) -> dict:
        """Serializar scheduler a dict"""
        return {
            "routines": [
                {
                    "routine_id": r.routine_id,
                    "name": r.name,
                    "triggers": r.triggers,
                    "conditions": r.conditions,
                    "actions": r.actions,
                    "enabled": r.enabled,
                    "cooldown_seconds": r.cooldown_seconds,
                    "created_at": r.created_at,
                    "created_by": r.created_by,
                    "tags": r.tags,
                    "execution_count": r.execution_count
                }
                for r in self._routines.values()
            ]
        }

    def load_from_dict(self, data: dict):
        """Cargar rutinas desde dict"""
        for r_data in data.get("routines", []):
            routine = ScheduledRoutine(
                routine_id=r_data["routine_id"],
                name=r_data["name"],
                triggers=r_data["triggers"],
                conditions=r_data.get("conditions", []),
                actions=r_data["actions"],
                enabled=r_data.get("enabled", True),
                cooldown_seconds=r_data.get("cooldown_seconds", 60),
                created_at=r_data.get("created_at", ""),
                created_by=r_data.get("created_by", "voice"),
                tags=r_data.get("tags", []),
                execution_count=r_data.get("execution_count", 0)
            )
            self.register_routine(routine)

    def save_to_file(self, filepath: str):
        """Guardar rutinas a archivo"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Rutinas guardadas en {filepath}")

    def load_from_file(self, filepath: str):
        """Cargar rutinas desde archivo"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_dict(data)
            logger.info(f"Rutinas cargadas desde {filepath}")
        except FileNotFoundError:
            logger.debug(f"Archivo de rutinas no existe: {filepath}")
        except Exception as e:
            logger.error(f"Error cargando rutinas: {e}")
