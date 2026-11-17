"""
Alert Manager Module
Sistema central de gestión de alertas con deduplicación, historial y notificaciones de voz.

Proporciona:
- Definición de prioridades y tipos de alerta
- Deduplicación por condición (tipo, fuente, condición), nunca por el texto
- Latch por episodio: una alerta al abrirse la condición; otra sólo después de
  cerrarla (resolve_condition) o, si se configura, tras un intervalo explícito
- Recibo de entrega de voz en la alerta (delivered/failed/blocked/muted)
- Registro de handlers por tipo de alerta
- Historial de alertas con límite configurable y timestamps con zona horaria
- Resumen de alertas pendientes y activas

Uso:
    from src.alerts import AlertCondition, AlertManager, AlertPriority, AlertType

    alert_manager = AlertManager(tts_callback=response_handler.aspeak)

    # Registrar handler para alertas de seguridad
    alert_manager.register_handler(
        AlertType.SECURITY,
        lambda alert: print(f"Alert: {alert.message}")
    )

    # Crear alerta
    alert = await alert_manager.create_alert(
        alert_type=AlertType.SECURITY,
        priority=AlertPriority.CRITICAL,
        message="Puerta principal abierta",
        details={"zone": "entrada"},
        condition=AlertCondition(source="entrada", condition="door_open_unexpected"),
    )

    # Al cerrarse la puerta el episodio termina; la próxima apertura alerta de nuevo
    await alert_manager.resolve_condition(
        AlertType.SECURITY, AlertCondition(source="entrada", condition="door_open_unexpected")
    )
"""

import asyncio
import inspect
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Callable
from uuid import uuid4

from src.core.logging import get_logger
from src.core.voice_delivery import DeliveryState, ResponseResult, VoiceDeliveryError

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Niveles de prioridad de alertas"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class AlertType(StrEnum):
    """Tipos de alertas"""
    SECURITY = "security"       # Seguridad (puertas, movimiento)
    PATTERN = "pattern"         # Patrones anómalos
    DEVICE = "device"          # Estado de dispositivos
    REMINDER = "reminder"       # Recordatorios
    WELLNESS = "wellness"       # Bienestar


# Priorities spoken by voice; lower ones only stay in the history (web).
SPOKEN_PRIORITIES = frozenset({AlertPriority.CRITICAL, AlertPriority.HIGH})


class AlertDelivery(StrEnum):
    """Voice delivery receipt of an alert's notice."""

    NOT_SPOKEN = "not_spoken"   # Priority below the voice threshold; web only
    PENDING = "pending"         # Notice in flight
    DELIVERED = "delivered"     # The output confirmed a complete delivery
    FAILED = "failed"           # The output failed, raised or did not confirm
    BLOCKED = "blocked"         # The output refused the notice (policy/route)
    MUTED = "muted"             # Voice notices disabled; nothing was attempted


def _utc_now() -> datetime:
    """Return the current instant with an explicit UTC offset."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class AlertCondition:
    """Identity of the condition an alert reports; the dedup key never uses the text.

    Attributes:
        source: Entity or producer source (entity_id, device name, zone).
        condition: Stable condition name (e.g. "battery_low", "door_open_unexpected").
        latched: True for a stateful condition whose episode stays open until
            resolve_condition; False for a momentary event whose episode closes
            by itself after the manager's cooldown_seconds.
    """

    source: str
    condition: str
    latched: bool = True

    def key(self, alert_type: AlertType) -> tuple[str, str, str]:
        """Return the deduplication key of this condition for an alert type."""
        return (alert_type.value, self.source, self.condition)


@dataclass
class Alert:
    """Representa una alerta individual"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    processed: bool = False
    processed_at: datetime | None = None
    source: str | None = None
    condition: str | None = None
    last_seen_at: datetime | None = None
    resolved_at: datetime | None = None
    delivery: AlertDelivery = AlertDelivery.NOT_SPOKEN
    delivery_reason: str | None = None
    delivery_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convertir alerta a diccionario"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "priority": self.priority.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "source": self.source,
            "condition": self.condition,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "delivery": self.delivery.value,
            "delivery_reason": self.delivery_reason,
            "delivery_at": self.delivery_at.isoformat() if self.delivery_at else None,
        }


@dataclass
class _Episode:
    """Open episode of one condition key (latched until resolved or expired)."""

    alert: Alert
    latched: bool
    opened_at: float
    announced_at: float | None = None


def _delivery_outcome(result: Any) -> tuple[AlertDelivery, str | None]:
    """Map what the voice output returned to the alert's delivery receipt.

    Args:
        result: The output's return value (ResponseResult for application outputs).

    Returns:
        (delivery state, diagnostic reason).
    """
    if isinstance(result, ResponseResult):
        if result.success:
            return AlertDelivery.DELIVERED, None
        if result.delivery is DeliveryState.BLOCKED:
            return AlertDelivery.BLOCKED, result.error or "voice_delivery_blocked"
        return AlertDelivery.FAILED, result.error or "voice_delivery_" + result.delivery.value
    if result is False:
        return AlertDelivery.FAILED, "voice_delivery_failed"
    # Legacy callbacks without a receipt keep the shared acknowledgement convention
    # of src.core.voice_delivery.invoke_voice_callback.
    return AlertDelivery.DELIVERED, None


class AlertManager:
    """
    Gestor central de alertas con soporte para:
    - Deduplicación con cooldown
    - Notificaciones de voz
    - Handlers por tipo de alerta
    - Historial persistente
    """

    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        max_history: int = 1000,
        tts_callback: Callable[[str], Any] | None = None,
        reannounce_interval_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        """
        Inicializar AlertManager.

        Args:
            cooldown_seconds: Duración del episodio de un evento momentáneo
                (AlertCondition.latched=False, o sin condición explícita).
            max_history: Máximo número de alertas en historial (default: 1000)
            tts_callback: Salida de voz; recibe el texto y devuelve su ResponseResult.
                None significa avisos de voz desactivados (recibo 'muted').
            reannounce_interval_seconds: Re-anunciar un episodio que sigue abierto
                cada este intervalo. None (default) = un solo aviso por episodio.
            clock: Reloj monotónico inyectable (segundos) para episodios.
            now: Fábrica de instantes con zona horaria para los timestamps.
        """
        if reannounce_interval_seconds is not None and (
            type(reannounce_interval_seconds) not in (int, float)
            or not math.isfinite(reannounce_interval_seconds)
            or reannounce_interval_seconds <= 0
        ):
            raise ValueError("reannounce_interval_seconds must be None or finite and positive")
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.tts_callback = tts_callback
        self.reannounce_interval_seconds = reannounce_interval_seconds
        self._clock = clock or time.monotonic
        self._now = now or _utc_now

        # Handlers por tipo de alerta: {AlertType: [callback1, callback2, ...]}
        self._handlers: dict[AlertType, list[Callable[[Alert], None]]] = defaultdict(list)

        # Episodios abiertos por condición: {(type, source, condition): _Episode}
        self._episodes: dict[tuple[str, str, str], _Episode] = {}

        # Historial de alertas (FIFO)
        self._history: list[Alert] = []

        # Lock para operaciones thread-safe
        self._lock = asyncio.Lock()

        logger.info(
            "AlertManager initialized (cooldown=%ss, reannounce=%s, max_history=%s)",
            cooldown_seconds, reannounce_interval_seconds, max_history,
        )

    def register_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Registrar handler para un tipo de alerta.

        Args:
            alert_type: Tipo de alerta
            handler: Función que recibe Alert. Puede ser sync o async.

        Ejemplo:
            manager.register_handler(
                AlertType.SECURITY,
                lambda alert: print(f"Security: {alert.message}")
            )
        """
        self._handlers[alert_type].append(handler)
        logger.debug(f"Handler registered for {alert_type.value}")

    def unregister_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None],
    ) -> None:
        """Desregistrar handler"""
        if handler in self._handlers[alert_type]:
            self._handlers[alert_type].remove(handler)
            logger.debug(f"Handler unregistered for {alert_type.value}")

    async def create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        message: str,
        details: dict | None = None,
        *,
        condition: AlertCondition | None = None,
    ) -> Alert | None:
        """
        Crear una alerta con latch por episodio de su condición.

        The dedup key is (type, source, condition), never the message text. While
        the condition's episode is open, a repeated observation creates nothing; a
        more severe priority escalates the open alert in place and announces it.
        Without an explicit condition the message identifies a momentary event
        (legacy producers), whose episode closes after cooldown_seconds.

        Args:
            alert_type: Tipo de alerta
            priority: Prioridad de la alerta
            message: Mensaje descriptivo (voz y web, en español)
            details: Detalles adicionales (dict)
            condition: Identidad de la condición que reporta la alerta

        Returns:
            La alerta si se abrió un episodio o se escaló; None si el episodio ya
            estaba abierto (deduplicada).
        """
        cond = condition or AlertCondition(source="message", condition=message, latched=False)
        key = cond.key(alert_type)
        async with self._lock:
            now = self._clock()
            episode = self._episodes.get(key)
            if episode is not None and not episode.latched and now - episode.opened_at >= self.cooldown_seconds:
                del self._episodes[key]
                episode = None

            if episode is not None:
                return await self._observe_open_episode(episode, priority, message, details, now)

            alert = Alert(
                alert_id=str(uuid4())[:8],
                alert_type=alert_type,
                priority=priority,
                message=message,
                details=details or {},
                timestamp=self._now(),
                source=cond.source,
                condition=cond.condition,
            )
            episode = _Episode(alert=alert, latched=cond.latched, opened_at=now)
            self._episodes[key] = episode

            # Agregar al historial
            self._history.append(alert)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            logger.info(
                "Alert created: %s (id=%s, priority=%s, source=%s, condition=%s)",
                alert.alert_type.value, alert.alert_id, alert.priority.name, cond.source, cond.condition,
            )

            # Ejecutar handlers
            await self._execute_handlers(alert)

            # Notificación de voz para alertas críticas/altas
            if priority in SPOKEN_PRIORITIES:
                await self._announce(episode)

            return alert

    async def _observe_open_episode(
        self,
        episode: _Episode,
        priority: AlertPriority,
        message: str,
        details: dict | None,
        now: float,
    ) -> Alert | None:
        """Apply a repeated observation of an open episode (caller holds the lock)."""
        alert = episode.alert
        alert.last_seen_at = self._now()
        if priority.value < alert.priority.value:
            alert.priority = priority
            alert.message = message
            alert.details = details or {}
            # The acknowledgement covered a less severe reading: the escalated alert
            # is active again (web list and summary) until acknowledged anew.
            alert.processed = False
            alert.processed_at = None
            logger.info("Alert escalated: %s (id=%s, priority=%s)",
                        alert.alert_type.value, alert.alert_id, priority.name)
            await self._execute_handlers(alert)
            if priority in SPOKEN_PRIORITIES:
                await self._announce(episode)
            return alert
        if (
            episode.latched
            and self.reannounce_interval_seconds is not None
            and episode.announced_at is not None
            and now - episode.announced_at >= self.reannounce_interval_seconds
        ):
            logger.info("Alert re-announced: %s (id=%s)", alert.alert_type.value, alert.alert_id)
            await self._announce(episode)
            return None
        logger.debug("Alert deduped (open episode): %s (id=%s)", alert.alert_type.value, alert.alert_id)
        return None

    async def resolve_condition(self, alert_type: AlertType, condition: AlertCondition) -> Alert | None:
        """Close the condition's open episode so its next opening alerts again.

        Args:
            alert_type: Tipo de alerta de la condición.
            condition: Identidad de la condición que dejó de cumplirse.

        Returns:
            La alerta del episodio cerrado, o None si no había episodio abierto.
        """
        async with self._lock:
            episode = self._episodes.pop(condition.key(alert_type), None)
            if episode is None:
                return None
            episode.alert.resolved_at = self._now()
            logger.info("Alert condition resolved: %s (id=%s, source=%s, condition=%s)",
                        alert_type.value, episode.alert.alert_id, condition.source, condition.condition)
            return episode.alert

    async def _execute_handlers(self, alert: Alert) -> None:
        """Ejecutar todos los handlers registrados para el tipo de alerta"""
        handlers = self._handlers.get(alert.alert_type, [])

        for handler in handlers:
            try:
                # Soportar tanto handlers sync como async
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(
                    f"Error executing alert handler: {e} (type={alert.alert_type.value}, id={alert.alert_id})"
                )

    async def _announce(self, episode: _Episode) -> None:
        """Speak the episode's alert once and record its delivery receipt."""
        episode.announced_at = self._clock()
        await self._notify_voice(episode.alert)

    async def _notify_voice(self, alert: Alert) -> None:
        """Speak the alert and record the output's receipt on it, never swallowing a failure."""
        callback = self.tts_callback
        if callback is None:
            self._record_delivery(alert, AlertDelivery.MUTED, "voice_notifications_disabled")
            return
        message = self._format_voice_message(alert)
        self._record_delivery(alert, AlertDelivery.PENDING, None)
        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback(message)
            else:
                result = await asyncio.to_thread(callback, message)
                if inspect.isawaitable(result):
                    result = await result
        except asyncio.CancelledError:
            self._record_delivery(alert, AlertDelivery.FAILED, "voice_delivery_cancelled")
            raise
        except VoiceDeliveryError as error:
            self._record_delivery(alert, AlertDelivery.FAILED, str(error) or "voice_delivery_failed")
            return
        except Exception as error:
            self._record_delivery(alert, AlertDelivery.FAILED, type(error).__name__)
            return
        self._record_delivery(alert, *_delivery_outcome(result))

    def _record_delivery(self, alert: Alert, delivery: AlertDelivery, reason: str | None) -> None:
        """Store the voice receipt on the alert and log anything not delivered."""
        alert.delivery = delivery
        alert.delivery_reason = reason
        alert.delivery_at = None if delivery is AlertDelivery.PENDING else self._now()
        if delivery in (AlertDelivery.FAILED, AlertDelivery.BLOCKED):
            logger.warning("Alert voice notice %s (id=%s, reason=%s)", delivery.value, alert.alert_id, reason)
        elif delivery is AlertDelivery.MUTED:
            logger.info("Alert voice notice muted (id=%s, reason=%s)", alert.alert_id, reason)

    def _format_voice_message(self, alert: Alert) -> str:
        """Formatear mensaje para notificación de voz"""
        priority_text = {
            AlertPriority.CRITICAL: "Alerta crítica: ",
            AlertPriority.HIGH: "Alerta importante: ",
        }
        prefix = priority_text.get(alert.priority, "")
        return f"{prefix}{alert.message}"

    def get_history(self, limit: int | None = None) -> list[Alert]:
        """
        Obtener historial de alertas.

        Args:
            limit: Número máximo de alertas a retornar (más recientes primero)

        Returns:
            Lista de alertas (en orden inverso: más recientes primero)
        """
        history = list(reversed(self._history))
        if limit:
            history = history[:limit]
        return history

    def get_pending_summary(self) -> dict:
        """
        Obtener resumen de alertas pendientes (no procesadas).

        Returns:
            Dict con conteo por tipo y prioridad
        """
        pending = [a for a in self._history if not a.processed]

        summary = {
            "total_pending": len(pending),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
        }

        for alert in pending:
            summary["by_type"][alert.alert_type.value] += 1
            summary["by_priority"][alert.priority.name] += 1

        # Convertir defaultdict a dict
        return {
            "total_pending": summary["total_pending"],
            "by_type": dict(summary["by_type"]),
            "by_priority": dict(summary["by_priority"]),
        }

    def get_active_alerts(self) -> list[Alert]:
        """Return alerts still unacknowledged whose condition was not resolved, newest first."""
        return [a for a in reversed(self._history) if not a.processed and a.resolved_at is None]

    async def mark_processed(self, alert_id: str, *, before_publish: Callable[[], None] | None = None,
                             update_timestamp: bool = True,
                             timestamp_factory: Callable[[], datetime] | None = None) -> bool:
        """Recheck original authority after the lock and before publishing alert state."""
        async with self._lock:
            if before_publish is not None and before_publish() is False:
                raise PermissionError("alert_authority_retired")
            for alert in self._history:
                if alert.alert_id == alert_id:
                    timestamp = (timestamp_factory or self._now)() if update_timestamp else None
                    if before_publish is not None and before_publish() is False:
                        raise PermissionError("alert_authority_retired")
                    alert.processed = True
                    if update_timestamp:
                        alert.processed_at = timestamp
                    logger.debug("Alert marked as processed: %s", alert_id)
                    return True
            return False

    def get_alert(self, alert_id: str) -> Alert | None:
        """Obtener alerta por ID"""
        for alert in self._history:
            if alert.alert_id == alert_id:
                return alert
        return None

    def clear_cooldowns(self) -> None:
        """Olvidar todos los episodios abiertos (la próxima observación alerta de nuevo)"""
        self._episodes.clear()
        logger.debug("All alert episodes cleared")

    def clear_history(self) -> None:
        """Limpiar historial de alertas"""
        self._history.clear()
        logger.debug("Alert history cleared")

    def get_stats(self) -> dict:
        """Obtener estadísticas del sistema de alertas"""
        return {
            "total_alerts": len(self._history),
            "pending_alerts": sum(1 for a in self._history if not a.processed),
            "processed_alerts": sum(1 for a in self._history if a.processed),
            "dedup_keys": len(self._episodes),
            "by_type": {
                alert_type.value: sum(1 for a in self._history if a.alert_type == alert_type)
                for alert_type in AlertType
            },
            "by_priority": {
                priority.name: sum(1 for a in self._history if a.priority == priority)
                for priority in AlertPriority
            },
        }
