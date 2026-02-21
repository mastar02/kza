"""
Tests for SecurityAlerts module.
"""

import pytest
from unittest.mock import AsyncMock

from src.alerts import AlertManager, AlertPriority, AlertType, SecurityAlerts


class TestSecurityAlertsDoors:
    """Tests para detección de puertas"""

    @pytest.mark.asyncio
    async def test_door_open_unexpected(self):
        """Test puerta abierta inesperadamente"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_status(
            zone="entrada",
            is_open=True,
            expected_open=False,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.SECURITY
        assert alert.priority == AlertPriority.HIGH
        assert "entrada" in alert.message.lower()
        assert alert.details["zone"] == "entrada"

    @pytest.mark.asyncio
    async def test_door_open_expected(self):
        """Test puerta abierta pero esperada"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_status(
            zone="entrada",
            is_open=True,
            expected_open=True,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_door_closed(self):
        """Test puerta cerrada"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_status(
            zone="entrada",
            is_open=False,
            expected_open=False,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_door_zone_names(self):
        """Test nombres de zonas se usan correctamente"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_status(
            zone="garaje",
            is_open=True,
            expected_open=False,
        )

        assert "garaje" in alert.message.lower()


class TestSecurityAlertsMotion:
    """Tests para detección de movimiento"""

    @pytest.mark.asyncio
    async def test_motion_unusual(self):
        """Test movimiento inusual detectado"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_unusual_motion(
            zone="sala",
            motion_detected=True,
            expected_occupancy=False,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.SECURITY
        assert alert.priority == AlertPriority.CRITICAL
        assert "movimiento" in alert.message.lower()
        assert "sala" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_motion_expected(self):
        """Test movimiento esperado"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_unusual_motion(
            zone="sala",
            motion_detected=True,
            expected_occupancy=True,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_no_motion(self):
        """Test sin movimiento"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_unusual_motion(
            zone="sala",
            motion_detected=False,
            expected_occupancy=False,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_motion_is_critical(self):
        """Test que movimiento inusual es CRITICAL"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_unusual_motion(
            zone="dormitorio",
            motion_detected=True,
            expected_occupancy=False,
        )

        # Movimiento inusual debe ser CRITICAL por seguridad
        assert alert.priority == AlertPriority.CRITICAL


class TestSecurityAlertsDoorSequence:
    """Tests para secuencias anormales de puertas"""

    @pytest.mark.asyncio
    async def test_door_sequence_unexpected(self):
        """Test secuencia anormal de puerta"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_sequence(
            zone="entrada",
            is_opening=True,
            sequence_expected=False,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.SECURITY
        assert alert.priority == AlertPriority.HIGH
        assert "abriendo" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_door_sequence_expected(self):
        """Test secuencia esperada de puerta"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_sequence(
            zone="entrada",
            is_opening=True,
            sequence_expected=True,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_door_closing_sequence(self):
        """Test secuencia de cierre de puerta"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_door_sequence(
            zone="garaje",
            is_opening=False,
            sequence_expected=False,
        )

        assert alert is not None
        assert "cerrando" in alert.message.lower()


class TestSecurityAlertsAccessDenied:
    """Tests para alertas de acceso denegado"""

    @pytest.mark.asyncio
    async def test_access_denied(self):
        """Test alerta de acceso denegado"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_access_denied(
            zone="entrada",
            user="desconocido",
            reason="unauthorized",
        )

        assert alert is not None
        assert alert.alert_type == AlertType.SECURITY
        assert alert.priority == AlertPriority.HIGH
        assert "acceso denegado" in alert.message.lower()
        assert "entrada" in alert.message.lower()
        assert alert.details["user"] == "desconocido"

    @pytest.mark.asyncio
    async def test_access_denied_details(self):
        """Test detalles de acceso denegado"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert = await security.check_access_denied(
            zone="sótano",
            user="juan_perez",
            reason="card_not_recognized",
        )

        assert alert.details["zone"] == "sótano"
        assert alert.details["user"] == "juan_perez"
        assert alert.details["reason"] == "card_not_recognized"


class TestSecurityAlertsDeduplication:
    """Tests para deduplicación en alertas de seguridad"""

    @pytest.mark.asyncio
    async def test_same_door_open_deduped(self):
        """Test que alertas idénticas se deduplicen"""
        manager = AlertManager(cooldown_seconds=5)
        security = SecurityAlerts(manager)

        # Primera alerta
        alert1 = await security.check_door_status(
            zone="entrada",
            is_open=True,
            expected_open=False,
        )
        assert alert1 is not None

        # Segunda alerta idéntica (deduplicada)
        alert2 = await security.check_door_status(
            zone="entrada",
            is_open=True,
            expected_open=False,
        )
        assert alert2 is None

    @pytest.mark.asyncio
    async def test_different_zones_not_deduped(self):
        """Test que diferentes zonas no se deduplican"""
        manager = AlertManager()
        security = SecurityAlerts(manager)

        alert1 = await security.check_door_status(
            zone="entrada",
            is_open=True,
            expected_open=False,
        )
        alert2 = await security.check_door_status(
            zone="garaje",
            is_open=True,
            expected_open=False,
        )

        assert alert1 is not None
        assert alert2 is not None
        assert alert1.alert_id != alert2.alert_id
