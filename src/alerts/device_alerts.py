"""
Device Alerts Module
Genera alertas sobre el estado de dispositivos inteligentes.

Proporciona:
- Detección de dispositivos offline
- Alertas de batería baja
- Alertas de conectividad
- Monitoreo de estado de dispositivos

Uso:
    from src.alerts import DeviceAlerts

    device_alerts = DeviceAlerts(alert_manager)

    # Verificar batería
    await device_alerts.check_battery_level(
        device_name="sensor_temp_sala",
        battery_percent=15
    )

    # Verificar conectividad
    await device_alerts.check_connectivity(
        device_name="smart_light_entrada",
        is_online=False
    )
"""


from src.alerts.alert_manager import Alert, AlertCondition, AlertManager, AlertPriority, AlertType
from src.core.logging import get_logger

logger = get_logger(__name__)


# Condition names: the dedup key is (device, condition), never the message text,
# so a battery reading 19 -> 18 -> 17 stays one episode.
BATTERY_LOW = "battery_low"
OFFLINE = "offline"
WEAK_SIGNAL = "weak_signal"
FIRMWARE_UPDATE = "firmware_update"
SLOW_RESPONSE = "slow_response"


class DeviceAlerts:
    """Gestor de alertas de estado de dispositivos.

    Each check opens its condition's episode when the condition holds and closes
    it (AlertManager.resolve_condition) when it no longer does.
    """

    # Umbrales de batería baja por tipo de dispositivo
    BATTERY_THRESHOLDS = {
        "sensor_default": 20,              # % mínimo recomendado
        "sensor_door": 15,                 # Puerta
        "sensor_motion": 20,               # Movimiento
        "sensor_temperature": 20,          # Temperatura
        "sensor_humidity": 20,             # Humedad
        "remote": 10,                      # Control remoto
        "light": 15,                       # Luz inteligente
        "switch": 15,                      # Switch inteligente
    }

    # Categorías de dispositivos
    DEVICE_TYPES = {
        "sensor": "Sensor",
        "light": "Luz inteligente",
        "switch": "Switch inteligente",
        "lock": "Cerradura inteligente",
        "camera": "Cámara",
        "hub": "Hub",
        "speaker": "Parlante",
    }

    def __init__(self, alert_manager: AlertManager):
        """
        Inicializar DeviceAlerts.

        Args:
            alert_manager: Instancia de AlertManager
        """
        self.alert_manager = alert_manager
        logger.debug("DeviceAlerts initialized")

    async def check_battery_level(
        self,
        device_name: str,
        battery_percent: float,
        device_type: str = "sensor_default",
    ) -> Alert | None:
        """
        Verificar nivel de batería de dispositivo.

        Args:
            device_name: Nombre del dispositivo
            battery_percent: Porcentaje de batería (0-100)
            device_type: Tipo de dispositivo para usar umbral correcto

        Returns:
            Alert si se abre el episodio de batería baja o escala a crítica;
            None si está ok o el episodio ya estaba abierto
        """
        threshold = self.BATTERY_THRESHOLDS.get(
            device_type,
            self.BATTERY_THRESHOLDS["sensor_default"],
        )

        condition = AlertCondition(source=device_name, condition=BATTERY_LOW)
        if battery_percent > threshold:
            # Batería ok: termina el episodio de batería baja
            await self.alert_manager.resolve_condition(AlertType.DEVICE, condition)
            return None

        # Batería baja
        priority = AlertPriority.HIGH if battery_percent < 10 else AlertPriority.MEDIUM

        message = f"Batería baja en {device_name}: {battery_percent:.0f}%"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=priority,
            message=message,
            details={
                "device_name": device_name,
                "device_type": device_type,
                "battery_percent": round(battery_percent, 1),
                "threshold_percent": threshold,
            },
            condition=condition,
        )

        if alert:
            logger.warning(
                f"Device battery low: {device_name} ({battery_percent:.0f}%) (id={alert.alert_id})"
            )

        return alert

    async def check_connectivity(
        self,
        device_name: str,
        is_online: bool,
        device_type: str = "sensor",
    ) -> Alert | None:
        """
        Verificar conectividad de dispositivo.

        Args:
            device_name: Nombre del dispositivo
            is_online: True si está online
            device_type: Tipo de dispositivo

        Returns:
            Alert si está offline, None si está online
        """
        condition = AlertCondition(source=device_name, condition=OFFLINE)
        if is_online:
            # Dispositivo online: termina el episodio offline
            await self.alert_manager.resolve_condition(AlertType.DEVICE, condition)
            return None

        # Dispositivo offline
        device_type_name = self.DEVICE_TYPES.get(device_type, device_type)
        message = f"{device_name} ({device_type_name}) está offline"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.HIGH,
            message=message,
            details={
                "device_name": device_name,
                "device_type": device_type,
                "is_online": is_online,
            },
            condition=condition,
        )

        if alert:
            logger.warning(f"Device offline: {device_name} ({device_type}) (id={alert.alert_id})")

        return alert

    async def check_signal_strength(
        self,
        device_name: str,
        signal_strength_db: float,
        min_signal_db: float = -85.0,
    ) -> Alert | None:
        """
        Verificar fortaleza de señal de dispositivo.

        Args:
            device_name: Nombre del dispositivo
            signal_strength_db: Fortaleza en dB (números negativos, ej: -60 es fuerte)
            min_signal_db: Umbral mínimo de señal aceptable

        Returns:
            Alert si señal es débil, None si está ok
        """
        condition = AlertCondition(source=device_name, condition=WEAK_SIGNAL)
        if signal_strength_db > min_signal_db:
            # Señal ok: termina el episodio de señal débil
            await self.alert_manager.resolve_condition(AlertType.DEVICE, condition)
            return None

        # Señal débil
        message = f"Señal débil en {device_name}: {signal_strength_db}dB"

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.MEDIUM,
            message=message,
            details={
                "device_name": device_name,
                "signal_strength_db": round(signal_strength_db, 1),
                "min_threshold_db": min_signal_db,
            },
            condition=condition,
        )

        if alert:
            logger.warning(f"Device weak signal: {device_name} ({signal_strength_db}dB) (id={alert.alert_id})")

        return alert

    async def check_firmware_update(
        self,
        device_name: str,
        current_version: str,
        latest_version: str,
    ) -> Alert | None:
        """
        Alertar sobre actualización de firmware disponible.

        Args:
            device_name: Nombre del dispositivo
            current_version: Versión actual
            latest_version: Versión disponible

        Returns:
            Alert si hay actualización, None si está actualizado
        """
        condition = AlertCondition(source=device_name, condition=FIRMWARE_UPDATE)
        if current_version == latest_version:
            await self.alert_manager.resolve_condition(AlertType.DEVICE, condition)
            return None

        message = (
            f"Actualización disponible para {device_name}: "
            f"{current_version} → {latest_version}"
        )

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.LOW,
            message=message,
            details={
                "device_name": device_name,
                "current_version": current_version,
                "latest_version": latest_version,
            },
            condition=condition,
        )

        if alert:
            logger.info(
                f"Device firmware update: {device_name} ({current_version} → {latest_version}) (id={alert.alert_id})"
            )

        return alert

    async def check_response_time(
        self,
        device_name: str,
        response_time_ms: float,
        max_response_time_ms: float = 5000.0,
    ) -> Alert | None:
        """
        Alertar si dispositivo responde lentamente.

        Args:
            device_name: Nombre del dispositivo
            response_time_ms: Tiempo de respuesta en ms
            max_response_time_ms: Máximo tiempo aceptable

        Returns:
            Alert si es muy lento, None si responde bien
        """
        condition = AlertCondition(source=device_name, condition=SLOW_RESPONSE)
        if response_time_ms <= max_response_time_ms:
            await self.alert_manager.resolve_condition(AlertType.DEVICE, condition)
            return None

        message = (
            f"Dispositivo {device_name} responde lentamente: "
            f"{response_time_ms:.0f}ms"
        )

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.MEDIUM,
            message=message,
            details={
                "device_name": device_name,
                "response_time_ms": round(response_time_ms, 1),
                "max_threshold_ms": max_response_time_ms,
            },
            condition=condition,
        )

        if alert:
            logger.warning(f"Device slow response: {device_name} ({response_time_ms:.0f}ms) (id={alert.alert_id})")

        return alert

    async def check_multiple_devices(
        self,
        devices: list[dict],
    ) -> list[Alert | None]:
        """
        Verificar múltiples dispositivos en una sola llamada.

        Args:
            devices: Lista de dicts con estructura:
                {
                    "name": "device_name",
                    "type": "sensor",
                    "battery_percent": 15,  # opcional
                    "is_online": True,      # opcional
                }

        Returns:
            Lista de alertas creadas (None si no hubo)
        """
        alerts = []

        for device in devices:
            name = device.get("name")
            device_type = device.get("type", "sensor")

            # Verificar batería si aplica
            if "battery_percent" in device:
                alert = await self.check_battery_level(
                    device_name=name,
                    battery_percent=device["battery_percent"],
                    device_type=device_type,
                )
                alerts.append(alert)

            # Verificar conectividad si aplica
            if "is_online" in device:
                alert = await self.check_connectivity(
                    device_name=name,
                    is_online=device["is_online"],
                    device_type=device_type,
                )
                alerts.append(alert)

        return alerts
