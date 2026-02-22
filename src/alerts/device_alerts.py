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


from src.core.logging import get_logger
from .alert_manager import Alert, AlertManager, AlertPriority, AlertType

logger = get_logger(__name__)


class DeviceAlerts:
    """Gestor de alertas de estado de dispositivos"""

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
            Alert si batería está baja, None si está ok
        """
        threshold = self.BATTERY_THRESHOLDS.get(
            device_type,
            self.BATTERY_THRESHOLDS["sensor_default"],
        )

        if battery_percent > threshold:
            # Batería ok
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
        if is_online:
            # Dispositivo online, sin alerta
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
        if signal_strength_db > min_signal_db:
            # Señal ok
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
        if current_version == latest_version:
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
        if response_time_ms <= max_response_time_ms:
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
