"""
Alert Manager Module
Sistema central de gestión de alertas con deduplicación, historial y notificaciones de voz.

Proporciona:
- Definición de prioridades y tipos de alerta
- Deduplicación de alertas con cooldown configurable
- Notificaciones de voz para alertas críticas/altas
- Registro de handlers por tipo de alerta
- Historial de alertas con límite configurable
- Resumen de alertas pendientes

Uso:
    from src.alerts import AlertManager, AlertPriority, AlertType

    alert_manager = AlertManager(tts_callback=lambda msg: print(msg))

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
        details={"zone": "entrada", "time": "2024-01-20 15:30"}
    )
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Callable
from uuid import uuid4

from src.core.logging import get_logger

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


@dataclass
class Alert:
    """Representa una alerta individual"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    processed_at: datetime | None = None

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
        }


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
        tts_callback: Callable[[str], None] | None = None,
    ):
        """
        Inicializar AlertManager.

        Args:
            cooldown_seconds: Tiempo mínimo entre alertas idénticas (default: 300s = 5 min)
            max_history: Máximo número de alertas en historial (default: 1000)
            tts_callback: Callback para notificaciones de voz. Recibe string con mensaje.
                         Ej: lambda msg: play_tts(msg)
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.tts_callback = tts_callback

        # Handlers por tipo de alerta: {AlertType: [callback1, callback2, ...]}
        self._handlers: dict[AlertType, list[Callable[[Alert], None]]] = defaultdict(list)

        # Deduplicación: {alert_key: last_timestamp}
        self._last_alert_time: dict[str, float] = {}

        # Historial de alertas (FIFO)
        self._history: list[Alert] = []

        # Lock para operaciones thread-safe
        self._lock = asyncio.Lock()

        logger.info(
            f"AlertManager initialized (cooldown={cooldown_seconds}s, max_history={max_history})"
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
    ) -> Alert | None:
        """
        Crear una alerta con deduplicación automática.

        Args:
            alert_type: Tipo de alerta
            priority: Prioridad de la alerta
            message: Mensaje descriptivo
            details: Detalles adicionales (dict)

        Returns:
            Alert si se creó, None si fue deduplicada por cooldown
        """
        async with self._lock:
            # Generar clave de deduplicación
            alert_key = self._make_dedup_key(alert_type, message)

            # Verificar cooldown
            last_time = self._last_alert_time.get(alert_key, 0)
            current_time = time.time()

            if current_time - last_time < self.cooldown_seconds:
                logger.debug(
                    f"Alert deduped (cooldown): {alert_type.value} - {message}"
                )
                return None

            # Crear alerta
            alert = Alert(
                alert_id=str(uuid4())[:8],
                alert_type=alert_type,
                priority=priority,
                message=message,
                details=details or {},
            )

            # Actualizar timestamp de deduplicación
            self._last_alert_time[alert_key] = current_time

            # Agregar al historial
            self._history.append(alert)
            if len(self._history) > self.max_history:
                self._history.pop(0)

            logger.info(
                f"Alert created: {alert.alert_type.value} (id={alert.alert_id}, priority={alert.priority.name})"
            )

            # Ejecutar handlers
            await self._execute_handlers(alert)

            # Notificación de voz para alertas críticas/altas
            if priority in (AlertPriority.CRITICAL, AlertPriority.HIGH):
                await self._notify_voice(alert)

            return alert

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

    async def _notify_voice(self, alert: Alert) -> None:
        """Notificar alerta por voz si hay callback disponible"""
        if not self.tts_callback:
            return

        try:
            message = self._format_voice_message(alert)
            # Soportar tanto callbacks sync como async
            if asyncio.iscoroutinefunction(self.tts_callback):
                await self.tts_callback(message)
            else:
                self.tts_callback(message)
        except Exception as e:
            logger.error(f"Error in voice notification: {e}")

    def _format_voice_message(self, alert: Alert) -> str:
        """Formatear mensaje para notificación de voz"""
        priority_text = {
            AlertPriority.CRITICAL: "Alerta crítica: ",
            AlertPriority.HIGH: "Alerta importante: ",
        }
        prefix = priority_text.get(alert.priority, "")
        return f"{prefix}{alert.message}"

    @staticmethod
    def _make_dedup_key(alert_type: AlertType, message: str) -> str:
        """Generar clave de deduplicación"""
        return f"{alert_type.value}:{message}"

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

    async def mark_processed(self, alert_id: str) -> bool:
        """
        Marcar alerta como procesada.

        Args:
            alert_id: ID de la alerta

        Returns:
            True si se marcó, False si no se encontró
        """
        async with self._lock:
            for alert in self._history:
                if alert.alert_id == alert_id:
                    alert.processed = True
                    alert.processed_at = datetime.utcnow()
                    logger.debug(f"Alert marked as processed: {alert_id}")
                    return True
            return False

    def get_alert(self, alert_id: str) -> Alert | None:
        """Obtener alerta por ID"""
        for alert in self._history:
            if alert.alert_id == alert_id:
                return alert
        return None

    def clear_cooldowns(self) -> None:
        """Limpiar todos los cooldowns de deduplicación"""
        self._last_alert_time.clear()
        logger.debug("All alert cooldowns cleared")

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
            "dedup_keys": len(self._last_alert_time),
            "by_type": {
                alert_type.value: sum(1 for a in self._history if a.alert_type == alert_type)
                for alert_type in AlertType
            },
            "by_priority": {
                priority.name: sum(1 for a in self._history if a.priority == priority)
                for priority in AlertPriority
            },
        }
