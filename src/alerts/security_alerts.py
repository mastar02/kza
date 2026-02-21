"""
Security Alerts Module
Genera alertas de seguridad basadas en estados de puertas y movimiento.

Proporciona:
- Detección de puertas abiertas
- Detección de movimiento inusual
- Alertas contextuales por zona

Uso:
    from src.alerts import SecurityAlerts

    security_alerts = SecurityAlerts(alert_manager)

    # Verificar puerta abierta
    await security_alerts.check_door_status("entrada", is_open=True)

    # Verificar movimiento inusual
    await security_alerts.check_unusual_motion(
        zone="sala",
        motion_detected=True,
        expected_occupancy=False
    )
"""

import asyncio
from typing import Optional

from src.core.logging import get_logger
from .alert_manager import Alert, AlertManager, AlertPriority, AlertType

logger = get_logger(__name__)


class SecurityAlerts:
    """Gestor de alertas de seguridad"""

    # Mapeo de zonas a nombres legibles
    ZONE_NAMES = {
        "entrada": "Puerta de entrada",
        "garaje": "Puerta del garaje",
        "cocina": "Puerta de la cocina",
        "ventana_sala": "Ventana de la sala",
        "ventana_recamara": "Ventana de la recámara",
    }

    def __init__(self, alert_manager: AlertManager):
        """
        Inicializar SecurityAlerts.

        Args:
            alert_manager: Instancia de AlertManager
        """
        self.alert_manager = alert_manager
        logger.debug("SecurityAlerts initialized")

    async def check_door_status(
        self,
        zone: str,
        is_open: bool,
        expected_open: bool = False,
    ) -> Optional[Alert]:
        """
        Verificar estado de puerta.

        Args:
            zone: Identificador de zona (ej: "entrada", "garaje")
            is_open: True si la puerta está abierta
            expected_open: True si se esperaba que esté abierta

        Returns:
            Alert si se creó, None si fue deduplicada
        """
        if not is_open:
            # Puerta cerrada, sin alerta
            return None

        if expected_open:
            # Puerta abierta pero esperada, sin alerta
            return None

        # Puerta abierta e inesperada = alerta
        zone_name = self.ZONE_NAMES.get(zone, zone)
        message = f"{zone_name} abierta inesperadamente"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message=message,
            details={
                "zone": zone,
                "type": "door_open",
                "is_open": is_open,
            },
        )

        if alert:
            logger.warning(f"Door security alert: {zone} (id={alert.alert_id})")

        return alert

    async def check_unusual_motion(
        self,
        zone: str,
        motion_detected: bool,
        expected_occupancy: bool = True,
    ) -> Optional[Alert]:
        """
        Verificar movimiento inusual.

        Args:
            zone: Identificador de zona
            motion_detected: True si se detectó movimiento
            expected_occupancy: True si se espera ocupación en la zona

        Returns:
            Alert si se creó, None si fue deduplicada
        """
        if not motion_detected:
            # Sin movimiento, sin alerta
            return None

        if expected_occupancy:
            # Movimiento esperado, sin alerta
            return None

        # Movimiento detectado en zona sin ocupación esperada = alerta
        message = f"Movimiento detectado en {zone} sin ocupación esperada"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.CRITICAL,
            message=message,
            details={
                "zone": zone,
                "type": "unusual_motion",
                "motion_detected": motion_detected,
            },
        )

        if alert:
            logger.warning(f"Motion security alert: {zone} (id={alert.alert_id})")

        return alert

    async def check_door_sequence(
        self,
        zone: str,
        is_opening: bool,
        sequence_expected: bool = False,
    ) -> Optional[Alert]:
        """
        Detectar secuencias de apertura/cierre anormales.

        Args:
            zone: Identificador de zona
            is_opening: True si se está abriendo
            sequence_expected: True si la secuencia es esperada

        Returns:
            Alert si se creó, None si fue deduplicada
        """
        if sequence_expected:
            # Secuencia esperada, sin alerta
            return None

        zone_name = self.ZONE_NAMES.get(zone, zone)
        action = "abriendo" if is_opening else "cerrando"
        message = f"Secuencia anormal detectada: {action} {zone_name}"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message=message,
            details={
                "zone": zone,
                "type": "door_sequence_anomaly",
                "is_opening": is_opening,
            },
        )

        if alert:
            logger.info(f"Door sequence alert: {action} {zone} (id={alert.alert_id})")

        return alert

    async def check_access_denied(
        self,
        zone: str,
        user: str,
        reason: str = "unauthorized",
    ) -> Optional[Alert]:
        """
        Alertar sobre acceso denegado.

        Args:
            zone: Zona donde se intentó acceso
            user: Usuario que intentó acceso
            reason: Motivo del rechazo

        Returns:
            Alert si se creó
        """
        message = f"Acceso denegado en {zone} para usuario {user}"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message=message,
            details={
                "zone": zone,
                "user": user,
                "type": "access_denied",
                "reason": reason,
            },
        )

        if alert:
            logger.warning(f"Access denied: {user} in {zone} - {reason} (id={alert.alert_id})")

        return alert
