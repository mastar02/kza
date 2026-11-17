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


from src.alerts.alert_manager import Alert, AlertCondition, AlertManager, AlertPriority, AlertType
from src.core.logging import get_logger

logger = get_logger(__name__)


# Condition names: the dedup key is (zone, condition), never the message text.
DOOR_OPEN_UNEXPECTED = "door_open_unexpected"
UNUSUAL_MOTION = "unusual_motion"
DOOR_SEQUENCE_ANOMALY = "door_sequence_anomaly"
ACCESS_DENIED = "access_denied"


class SecurityAlerts:
    """Gestor de alertas de seguridad.

    Door and motion checks are stateful episodes: one alert when the condition
    opens, closed when the check sees it no longer holds. Door sequences and
    denied accesses are momentary events whose episode lasts cooldown_seconds.
    """

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
    ) -> Alert | None:
        """
        Verificar estado de puerta.

        Args:
            zone: Identificador de zona (ej: "entrada", "garaje")
            is_open: True si la puerta está abierta
            expected_open: True si se esperaba que esté abierta

        Returns:
            Alert si se creó, None si fue deduplicada
        """
        condition = AlertCondition(source=zone, condition=DOOR_OPEN_UNEXPECTED)
        if not is_open or expected_open:
            # Puerta cerrada o abierta esperada: termina el episodio
            await self.alert_manager.resolve_condition(AlertType.SECURITY, condition)
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
            condition=condition,
        )

        if alert:
            logger.warning(f"Door security alert: {zone} (id={alert.alert_id})")

        return alert

    async def check_unusual_motion(
        self,
        zone: str,
        motion_detected: bool,
        expected_occupancy: bool = True,
    ) -> Alert | None:
        """
        Verificar movimiento inusual.

        Args:
            zone: Identificador de zona
            motion_detected: True si se detectó movimiento
            expected_occupancy: True si se espera ocupación en la zona

        Returns:
            Alert si se creó, None si fue deduplicada
        """
        condition = AlertCondition(source=zone, condition=UNUSUAL_MOTION)
        if not motion_detected or expected_occupancy:
            # Sin movimiento o movimiento esperado: termina el episodio
            await self.alert_manager.resolve_condition(AlertType.SECURITY, condition)
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
            condition=condition,
        )

        if alert:
            logger.warning(f"Motion security alert: {zone} (id={alert.alert_id})")

        return alert

    async def check_door_sequence(
        self,
        zone: str,
        is_opening: bool,
        sequence_expected: bool = False,
    ) -> Alert | None:
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
            condition=AlertCondition(source=zone, condition=f"{DOOR_SEQUENCE_ANOMALY}:{action}", latched=False),
        )

        if alert:
            logger.info(f"Door sequence alert: {action} {zone} (id={alert.alert_id})")

        return alert

    async def check_access_denied(
        self,
        zone: str,
        user: str,
        reason: str = "unauthorized",
    ) -> Alert | None:
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
            condition=AlertCondition(source=zone, condition=f"{ACCESS_DENIED}:{user}", latched=False),
        )

        if alert:
            logger.warning(f"Access denied: {user} in {zone} - {reason} (id={alert.alert_id})")

        return alert
