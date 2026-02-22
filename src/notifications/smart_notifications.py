"""
Smart Notifications - Notificaciones Inteligentes
Notificaciones contextualizadas basadas en presencia, hora y preferencias.

"Solo avisarme cuando estoy despierto"
"Notificar solo si no estoy en casa"
"No molestar si estoy en reunión"
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable, Any
from enum import Enum, StrEnum
import uuid

logger = logging.getLogger(__name__)


class NotificationChannel(StrEnum):
    VOICE = "voice"           # TTS en speakers
    PUSH = "push"             # Notificación push móvil
    DISPLAY = "display"       # Mostrar en pantallas/tablets
    LED = "led"               # Indicador LED silencioso
    ALL = "all"               # Todos los canales


class NotificationPriority(Enum):
    LOW = 1           # Puede esperar, solo LED
    NORMAL = 2        # Voz si está en casa, push si no
    HIGH = 3          # Voz + push
    URGENT = 4        # Interrumpir todo, todos los canales
    EMERGENCY = 5     # Máxima prioridad, ignorar DND


@dataclass
class NotificationRule:
    """Regla de notificación contextual"""
    rule_id: str
    name: str
    enabled: bool = True

    # Condiciones
    time_start: time | None = None      # Hora inicio (ej: 08:00)
    time_end: time | None = None        # Hora fin (ej: 22:00)
    days_of_week: list[int] = None         # 0=Lunes, 6=Domingo
    user_home: bool | None = None       # True=solo en casa, False=solo fuera
    user_awake: bool | None = None      # True=solo despierto
    zone_id: str | None = None          # Solo en esta zona
    min_priority: NotificationPriority = NotificationPriority.NORMAL

    # Acciones
    channels: list[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.VOICE])
    suppress: bool = False                 # Suprimir notificación


@dataclass
class Notification:
    """Una notificación"""
    notification_id: str
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    source: str = "system"                 # Origen: "system", "home_assistant", "calendar", etc.

    # Destinatario
    user_id: str | None = None          # None = broadcast
    zone_id: str | None = None          # Zona específica

    # Metadatos
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None   # Caducidad
    data: dict = field(default_factory=dict)  # Datos adicionales

    # Estado
    delivered: bool = False
    delivered_at: datetime | None = None
    channels_used: list[NotificationChannel] = field(default_factory=list)
    suppressed: bool = False
    suppression_reason: str | None = None


class SmartNotificationManager:
    """
    Gestor de notificaciones inteligentes.

    Características:
    - Entrega contextual (presencia, hora, actividad)
    - Múltiples canales (voz, push, display, LED)
    - Reglas personalizables por usuario
    - Cola de prioridad
    - Modo No Molestar inteligente
    - Agrupación de notificaciones similares
    """

    # Horarios por defecto
    DEFAULT_QUIET_HOURS = (time(23, 0), time(7, 0))  # 23:00 - 07:00
    DEFAULT_WAKE_HOURS = (time(7, 0), time(23, 0))

    def __init__(
        self,
        tts_callback: Callable[[str, str], Any] = None,
        push_callback: Callable[[str, str, dict], Any] = None,
        display_callback: Callable[[str, dict], Any] = None,
        led_callback: Callable[[str, str], Any] = None,
        presence_detector = None,
        user_manager = None,
        ha_client = None,
        quiet_hours: tuple[time, time] = None
    ):
        self.tts = tts_callback
        self.push = push_callback
        self.display = display_callback
        self.led = led_callback
        self.presence = presence_detector
        self.users = user_manager
        self.ha = ha_client

        self.quiet_hours = quiet_hours or self.DEFAULT_QUIET_HOURS

        # Estado
        self._notification_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._notification_history: list[Notification] = []
        self._running = False
        self._process_task: asyncio.Task | None = None

        # Configuración por usuario
        self._user_rules: dict[str, list[NotificationRule]] = {}  # user_id -> reglas
        self._user_dnd: dict[str, bool] = {}  # user_id -> DND activo
        self._user_preferences: dict[str, dict] = {}  # user_id -> preferencias

        # Agrupación
        self._pending_groups: dict[str, list[Notification]] = {}  # source -> notificaciones
        self._group_timeout: float = 5.0  # Segundos para agrupar

        # Callbacks
        self._on_notification_delivered: Callable | None = None

    async def start(self):
        """Iniciar sistema de notificaciones"""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_queue())
        logger.info("Sistema de notificaciones iniciado")

    async def stop(self):
        """Detener sistema"""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

    # ==================== Notificaciones ====================

    async def notify(
        self,
        message: str,
        title: str = None,
        user_id: str = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        source: str = "system",
        data: dict = None,
        expires_in_minutes: int = None
    ) -> Notification:
        """
        Enviar una notificación.

        La notificación se procesará según:
        - Prioridad
        - Contexto del usuario (presencia, hora)
        - Reglas personalizadas
        """
        notification = Notification(
            notification_id=f"notif_{uuid.uuid4().hex[:8]}",
            title=title or source.title(),
            message=message,
            priority=priority,
            source=source,
            user_id=user_id,
            data=data or {},
            expires_at=datetime.now() + timedelta(minutes=expires_in_minutes) if expires_in_minutes else None
        )

        # Encolar con prioridad (menor número = mayor prioridad)
        await self._notification_queue.put((
            -priority.value,  # Negativo para ordenar de mayor a menor
            notification.created_at.timestamp(),
            notification
        ))

        logger.debug(f"Notificación encolada: {message[:50]}... (prioridad={priority.name})")
        return notification

    async def notify_user(
        self,
        user_id: str,
        message: str,
        title: str = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Notification:
        """Notificar a un usuario específico"""
        return await self.notify(
            message=message,
            title=title,
            user_id=user_id,
            priority=priority
        )

    async def notify_all(
        self,
        message: str,
        title: str = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Notification:
        """Notificar a todos los usuarios"""
        return await self.notify(
            message=message,
            title=title,
            user_id=None,  # Broadcast
            priority=priority
        )

    async def notify_zone(
        self,
        zone_id: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Notification:
        """Notificar en una zona específica"""
        notification = await self.notify(message=message, priority=priority)
        notification.zone_id = zone_id
        return notification

    # ==================== Procesamiento ====================

    async def _process_queue(self):
        """Procesar cola de notificaciones"""
        while self._running:
            try:
                # Obtener siguiente notificación
                priority, timestamp, notification = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0
                )

                await self._deliver_notification(notification)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error procesando notificación: {e}")

    async def _deliver_notification(self, notification: Notification):
        """Entregar notificación según contexto"""

        # Verificar expiración
        if notification.expires_at and datetime.now() > notification.expires_at:
            logger.debug(f"Notificación expirada: {notification.notification_id}")
            return

        # Determinar usuarios destino
        if notification.user_id:
            users = [notification.user_id]
        else:
            # Broadcast: obtener todos los usuarios
            users = self._get_all_user_ids()

        for user_id in users:
            # Evaluar contexto
            context = self._get_user_context(user_id)

            # Verificar reglas y determinar canales
            channels, suppress, reason = self._evaluate_rules(
                notification, user_id, context
            )

            if suppress:
                notification.suppressed = True
                notification.suppression_reason = reason
                logger.debug(f"Notificación suprimida para {user_id}: {reason}")
                continue

            # Entregar por cada canal
            for channel in channels:
                try:
                    await self._deliver_to_channel(notification, user_id, channel, context)
                    notification.channels_used.append(channel)
                except Exception as e:
                    logger.error(f"Error entregando a {channel.value}: {e}")

        notification.delivered = len(notification.channels_used) > 0
        notification.delivered_at = datetime.now()

        # Guardar en historial
        self._notification_history.append(notification)
        if len(self._notification_history) > 500:
            self._notification_history = self._notification_history[-250:]

        # Callback
        if notification.delivered and self._on_notification_delivered:
            self._on_notification_delivered(notification)

    def _get_user_context(self, user_id: str) -> dict:
        """Obtener contexto actual del usuario"""
        context = {
            "user_id": user_id,
            "is_home": True,
            "is_awake": True,
            "current_zone": None,
            "is_quiet_hours": self._is_quiet_hours(),
            "dnd_active": self._user_dnd.get(user_id, False),
            "current_time": datetime.now().time(),
            "day_of_week": datetime.now().weekday()
        }

        # Presencia
        if self.presence:
            context["is_home"] = self.presence.is_user_home(user_id)
            context["current_zone"] = self.presence.get_user_zone(user_id)

        # Inferir si está despierto
        context["is_awake"] = self._infer_user_awake(context)

        return context

    def _infer_user_awake(self, context: dict) -> bool:
        """Inferir si el usuario está despierto"""
        # En horas de sueño y en casa = probablemente dormido
        if context["is_quiet_hours"] and context["is_home"]:
            zone = context.get("current_zone", "")
            if "bedroom" in zone or "dormitorio" in zone:
                return False
        return True

    def _is_quiet_hours(self) -> bool:
        """¿Estamos en horario silencioso?"""
        now = datetime.now().time()
        start, end = self.quiet_hours

        if start <= end:
            return start <= now <= end
        else:  # Cruza medianoche
            return now >= start or now <= end

    def _evaluate_rules(
        self,
        notification: Notification,
        user_id: str,
        context: dict
    ) -> tuple[list[NotificationChannel], bool, str]:
        """
        Evaluar reglas para determinar canales y supresión.

        Returns:
            (canales, suprimir, razón)
        """
        # Canales por defecto según prioridad y contexto
        channels = self._default_channels(notification.priority, context)

        # Verificar DND
        if context["dnd_active"] and notification.priority.value < NotificationPriority.URGENT.value:
            return [], True, "DND activo"

        # Verificar horas silenciosas
        if context["is_quiet_hours"] and notification.priority.value < NotificationPriority.HIGH.value:
            # Solo LED en horas silenciosas para prioridad baja/normal
            return [NotificationChannel.LED], False, None

        # Evaluar reglas del usuario
        user_rules = self._user_rules.get(user_id, [])
        for rule in user_rules:
            if not rule.enabled:
                continue

            if self._rule_matches(rule, notification, context):
                if rule.suppress:
                    return [], True, f"Regla: {rule.name}"
                else:
                    channels = rule.channels

        # Ajustar canales según presencia
        if not context["is_home"]:
            # Fuera de casa: solo push
            channels = [ch for ch in channels if ch in [NotificationChannel.PUSH, NotificationChannel.ALL]]
            if not channels:
                channels = [NotificationChannel.PUSH]

        return channels, False, None

    def _default_channels(
        self,
        priority: NotificationPriority,
        context: dict
    ) -> list[NotificationChannel]:
        """Canales por defecto según prioridad"""
        if priority == NotificationPriority.LOW:
            return [NotificationChannel.LED]

        elif priority == NotificationPriority.NORMAL:
            if context["is_home"]:
                return [NotificationChannel.VOICE]
            else:
                return [NotificationChannel.PUSH]

        elif priority == NotificationPriority.HIGH:
            return [NotificationChannel.VOICE, NotificationChannel.PUSH]

        elif priority in [NotificationPriority.URGENT, NotificationPriority.EMERGENCY]:
            return [NotificationChannel.VOICE, NotificationChannel.PUSH, NotificationChannel.DISPLAY]

        return [NotificationChannel.VOICE]

    def _rule_matches(
        self,
        rule: NotificationRule,
        notification: Notification,
        context: dict
    ) -> bool:
        """Verificar si una regla aplica"""
        # Verificar prioridad mínima
        if notification.priority.value < rule.min_priority.value:
            return False

        # Verificar horario
        if rule.time_start and rule.time_end:
            current = context["current_time"]
            if rule.time_start <= rule.time_end:
                if not (rule.time_start <= current <= rule.time_end):
                    return False
            else:  # Cruza medianoche
                if not (current >= rule.time_start or current <= rule.time_end):
                    return False

        # Verificar día de la semana
        if rule.days_of_week is not None:
            if context["day_of_week"] not in rule.days_of_week:
                return False

        # Verificar presencia
        if rule.user_home is not None:
            if context["is_home"] != rule.user_home:
                return False

        # Verificar si está despierto
        if rule.user_awake is not None:
            if context["is_awake"] != rule.user_awake:
                return False

        # Verificar zona
        if rule.zone_id:
            if context.get("current_zone") != rule.zone_id:
                return False

        return True

    async def _deliver_to_channel(
        self,
        notification: Notification,
        user_id: str,
        channel: NotificationChannel,
        context: dict
    ):
        """Entregar notificación a un canal específico"""
        logger.info(f"🔔 Notificación [{channel.value}] para {user_id}: {notification.message[:50]}...")

        if channel == NotificationChannel.VOICE:
            if self.tts:
                zone = notification.zone_id or context.get("current_zone") or "default"
                text = notification.message
                if notification.title:
                    text = f"{notification.title}. {text}"
                self.tts(text, zone)

        elif channel == NotificationChannel.PUSH:
            if self.push:
                await self.push(
                    user_id,
                    notification.message,
                    {
                        "title": notification.title,
                        "data": notification.data,
                        "priority": notification.priority.name
                    }
                )

        elif channel == NotificationChannel.DISPLAY:
            if self.display:
                await self.display(
                    notification.zone_id or "all",
                    {
                        "title": notification.title,
                        "message": notification.message,
                        "priority": notification.priority.name,
                        "data": notification.data
                    }
                )

        elif channel == NotificationChannel.LED:
            if self.led:
                color = self._priority_to_color(notification.priority)
                await self.led(notification.zone_id or "all", color)

    def _priority_to_color(self, priority: NotificationPriority) -> str:
        """Convertir prioridad a color LED"""
        colors = {
            NotificationPriority.LOW: "blue",
            NotificationPriority.NORMAL: "green",
            NotificationPriority.HIGH: "yellow",
            NotificationPriority.URGENT: "orange",
            NotificationPriority.EMERGENCY: "red"
        }
        return colors.get(priority, "white")

    def _get_all_user_ids(self) -> list[str]:
        """Obtener todos los IDs de usuario"""
        if self.users:
            return [u.user_id for u in self.users.get_all_users()]
        return ["default"]

    # ==================== Configuración de Usuario ====================

    def set_dnd(self, user_id: str, enabled: bool):
        """Activar/desactivar No Molestar"""
        self._user_dnd[user_id] = enabled
        logger.info(f"DND {'activado' if enabled else 'desactivado'} para {user_id}")

    def is_dnd_active(self, user_id: str) -> bool:
        """¿Está DND activo para el usuario?"""
        return self._user_dnd.get(user_id, False)

    def add_rule(self, user_id: str, rule: NotificationRule):
        """Agregar regla de notificación para usuario"""
        if user_id not in self._user_rules:
            self._user_rules[user_id] = []
        self._user_rules[user_id].append(rule)

    def remove_rule(self, user_id: str, rule_id: str) -> bool:
        """Eliminar regla"""
        if user_id in self._user_rules:
            rules = self._user_rules[user_id]
            for i, rule in enumerate(rules):
                if rule.rule_id == rule_id:
                    del rules[i]
                    return True
        return False

    def get_user_rules(self, user_id: str) -> list[NotificationRule]:
        """Obtener reglas de un usuario"""
        return self._user_rules.get(user_id, [])

    def set_quiet_hours(self, start: time, end: time):
        """Configurar horas silenciosas"""
        self.quiet_hours = (start, end)

    def set_user_preference(self, user_id: str, key: str, value: Any):
        """Establecer preferencia de usuario"""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        self._user_preferences[user_id][key] = value

    # ==================== Comandos de Voz ====================

    def handle_voice_command(self, text: str, user_id: str = None) -> dict:
        """
        Manejar comandos de voz relacionados con notificaciones.

        Ejemplos:
        - "No molestar"
        - "Desactiva no molestar"
        - "Solo notifícame cosas urgentes"
        """
        text_lower = text.lower().strip()

        # DND
        if re.search(r"no\s+molest(?:ar|es)", text_lower):
            self.set_dnd(user_id, True)
            return {
                "handled": True,
                "response": "Modo no molestar activado. Solo recibirás notificaciones urgentes."
            }

        if re.search(r"desactiva(?:r)?\s+(?:el\s+)?no\s+molest", text_lower):
            self.set_dnd(user_id, False)
            return {
                "handled": True,
                "response": "Modo no molestar desactivado."
            }

        # Solo urgentes
        if re.search(r"solo\s+(?:notif|avis).+urgent", text_lower):
            rule = NotificationRule(
                rule_id=f"rule_{uuid.uuid4().hex[:6]}",
                name="Solo urgentes",
                min_priority=NotificationPriority.URGENT,
                suppress=True  # Suprimir todo lo que no sea urgente
            )
            self.add_rule(user_id, rule)
            return {
                "handled": True,
                "response": "Entendido, solo te notificaré cosas urgentes."
            }

        return {"handled": False, "response": ""}

    # ==================== Estado ====================

    def get_history(self, user_id: str = None, limit: int = 20) -> list[Notification]:
        """Obtener historial de notificaciones"""
        history = self._notification_history
        if user_id:
            history = [n for n in history if n.user_id == user_id or n.user_id is None]
        return history[-limit:]

    def get_status(self) -> dict:
        """Obtener estado del sistema"""
        return {
            "running": self._running,
            "queue_size": self._notification_queue.qsize(),
            "total_notifications": len(self._notification_history),
            "users_with_dnd": sum(1 for v in self._user_dnd.values() if v),
            "quiet_hours_active": self._is_quiet_hours(),
            "quiet_hours": f"{self.quiet_hours[0].strftime('%H:%M')}-{self.quiet_hours[1].strftime('%H:%M')}"
        }


# Importación necesaria
from datetime import timedelta
