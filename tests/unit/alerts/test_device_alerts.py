"""
Tests for DeviceAlerts module.
"""

import pytest

from src.alerts import AlertManager, AlertPriority, AlertType, DeviceAlerts


class TestDeviceAlertsBattery:
    """Tests para alertas de batería"""

    @pytest.mark.asyncio
    async def test_battery_ok(self):
        """Test batería en nivel aceptable"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_battery_level(
            device_name="sensor_temp_sala",
            battery_percent=50.0,
            device_type="sensor_temperature",
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_battery_low(self):
        """Test batería baja"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_battery_level(
            device_name="sensor_motion_entrada",
            battery_percent=15.0,
            device_type="sensor_motion",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DEVICE
        assert alert.priority == AlertPriority.MEDIUM
        assert "batería" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_battery_critical(self):
        """Test batería crítica (< 10%)"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_battery_level(
            device_name="sensor_door",
            battery_percent=8.0,
            device_type="sensor_door",
        )

        assert alert is not None
        assert alert.priority == AlertPriority.HIGH

    @pytest.mark.asyncio
    async def test_battery_threshold_default(self):
        """Test umbral de batería por defecto"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_battery_level(
            device_name="unknown_device",
            battery_percent=15.0,
            device_type="unknown_type",
        )

        # Debería usar threshold por defecto
        assert alert is not None

    @pytest.mark.asyncio
    async def test_battery_details(self):
        """Test detalles de alerta de batería"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_battery_level(
            device_name="sensor_temp",
            battery_percent=12.0,
            device_type="sensor_temperature",
        )

        assert alert.details["device_name"] == "sensor_temp"
        assert alert.details["battery_percent"] == 12.0
        assert alert.details["device_type"] == "sensor_temperature"


class TestDeviceAlertsConnectivity:
    """Tests para alertas de conectividad"""

    @pytest.mark.asyncio
    async def test_device_online(self):
        """Test dispositivo online"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_connectivity(
            device_name="smart_light_sala",
            is_online=True,
            device_type="light",
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_device_offline(self):
        """Test dispositivo offline"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_connectivity(
            device_name="smart_light_entrada",
            is_online=False,
            device_type="light",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DEVICE
        assert alert.priority == AlertPriority.HIGH
        assert "offline" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_device_type_in_message(self):
        """Test que tipo de dispositivo aparece en mensaje"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_connectivity(
            device_name="my_camera",
            is_online=False,
            device_type="camera",
        )

        assert "cámara" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_device_offline_details(self):
        """Test detalles de dispositivo offline"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_connectivity(
            device_name="lock_puerta",
            is_online=False,
            device_type="lock",
        )

        assert alert.details["device_name"] == "lock_puerta"
        assert alert.details["is_online"] is False
        assert alert.details["device_type"] == "lock"


class TestDeviceAlertsSignal:
    """Tests para alertas de señal"""

    @pytest.mark.asyncio
    async def test_signal_good(self):
        """Test señal fuerte"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_signal_strength(
            device_name="sensor_sala",
            signal_strength_db=-60.0,
            min_signal_db=-85.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_signal_weak(self):
        """Test señal débil"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_signal_strength(
            device_name="sensor_garaje",
            signal_strength_db=-92.0,
            min_signal_db=-85.0,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DEVICE
        assert alert.priority == AlertPriority.MEDIUM
        assert "señal" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_signal_details(self):
        """Test detalles de señal débil"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_signal_strength(
            device_name="sensor_sótano",
            signal_strength_db=-90.0,
            min_signal_db=-80.0,
        )

        assert alert.details["device_name"] == "sensor_sótano"
        assert alert.details["signal_strength_db"] == -90.0


class TestDeviceAlertsFirmware:
    """Tests para alertas de firmware"""

    @pytest.mark.asyncio
    async def test_firmware_up_to_date(self):
        """Test firmware actualizado"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_firmware_update(
            device_name="hub_central",
            current_version="2.5.1",
            latest_version="2.5.1",
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_firmware_update_available(self):
        """Test actualización de firmware disponible"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_firmware_update(
            device_name="hub_central",
            current_version="2.4.0",
            latest_version="2.5.1",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DEVICE
        assert alert.priority == AlertPriority.LOW
        assert "actualización" in alert.message.lower()
        assert "2.4.0" in alert.message
        assert "2.5.1" in alert.message

    @pytest.mark.asyncio
    async def test_firmware_details(self):
        """Test detalles de actualización de firmware"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_firmware_update(
            device_name="light_controller",
            current_version="1.0.0",
            latest_version="1.1.0",
        )

        assert alert.details["device_name"] == "light_controller"
        assert alert.details["current_version"] == "1.0.0"
        assert alert.details["latest_version"] == "1.1.0"


class TestDeviceAlertsResponseTime:
    """Tests para alertas de tiempo de respuesta"""

    @pytest.mark.asyncio
    async def test_response_time_ok(self):
        """Test tiempo de respuesta aceptable"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_response_time(
            device_name="light_entrada",
            response_time_ms=100.0,
            max_response_time_ms=5000.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_response_time_slow(self):
        """Test tiempo de respuesta lento"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_response_time(
            device_name="light_remota",
            response_time_ms=8000.0,
            max_response_time_ms=5000.0,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.DEVICE
        assert alert.priority == AlertPriority.MEDIUM
        assert "lentamente" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_response_time_details(self):
        """Test detalles de tiempo de respuesta"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        alert = await device.check_response_time(
            device_name="switch_cocina",
            response_time_ms=6500.0,
            max_response_time_ms=5000.0,
        )

        assert alert.details["device_name"] == "switch_cocina"
        assert alert.details["response_time_ms"] == 6500.0


class TestDeviceAlertsMultiple:
    """Tests para verificación de múltiples dispositivos"""

    @pytest.mark.asyncio
    async def test_multiple_devices_battery_check(self):
        """Test verificar batería de múltiples dispositivos"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        devices = [
            {"name": "sensor_1", "type": "sensor", "battery_percent": 50},
            {"name": "sensor_2", "type": "sensor", "battery_percent": 10},
            {"name": "sensor_3", "type": "sensor", "battery_percent": 80},
        ]

        alerts = await device.check_multiple_devices(devices)

        # Solo sensor_2 debe generar alerta
        non_none_alerts = [a for a in alerts if a is not None]
        assert len(non_none_alerts) == 1
        assert "sensor_2" in non_none_alerts[0].message

    @pytest.mark.asyncio
    async def test_multiple_devices_connectivity_check(self):
        """Test verificar conectividad de múltiples dispositivos"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        devices = [
            {"name": "light_1", "type": "light", "is_online": True},
            {"name": "light_2", "type": "light", "is_online": False},
            {"name": "light_3", "type": "light", "is_online": False},
        ]

        alerts = await device.check_multiple_devices(devices)

        non_none_alerts = [a for a in alerts if a is not None]
        assert len(non_none_alerts) == 2

    @pytest.mark.asyncio
    async def test_multiple_devices_mixed(self):
        """Test verificar múltiples atributos de múltiples dispositivos"""
        manager = AlertManager()
        device = DeviceAlerts(manager)

        devices = [
            {"name": "dev_1", "type": "sensor", "battery_percent": 50, "is_online": True},
            {"name": "dev_2", "type": "sensor", "battery_percent": 10, "is_online": False},
        ]

        alerts = await device.check_multiple_devices(devices)

        non_none_alerts = [a for a in alerts if a is not None]
        # dev_2 genera 2 alertas (batería baja + offline)
        assert len(non_none_alerts) == 2
