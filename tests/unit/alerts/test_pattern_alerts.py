"""
Tests for PatternAlerts module.
"""

import pytest

from src.alerts import AlertManager, AlertPriority, AlertType, PatternAlerts


class TestPatternAlertsRoutine:
    """Tests para detección de desviaciones de rutina"""

    @pytest.mark.asyncio
    async def test_routine_on_time(self):
        """Test actividad dentro de horario esperado"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_routine_deviation(
            activity="despertar",
            expected_time_utc="07:00",
            actual_time_utc="07:15",
            tolerance_minutes=30,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_routine_early(self):
        """Test actividad más temprano de lo esperado"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_routine_deviation(
            activity="despertar",
            expected_time_utc="07:00",
            actual_time_utc="05:30",
            tolerance_minutes=60,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PATTERN
        assert alert.priority == AlertPriority.MEDIUM
        assert "despertar" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_routine_late(self):
        """Test actividad más tarde de lo esperado"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_routine_deviation(
            activity="comida",
            expected_time_utc="13:00",
            actual_time_utc="14:45",
            tolerance_minutes=60,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PATTERN

    @pytest.mark.asyncio
    async def test_routine_invalid_time_format(self):
        """Test manejo de formato de tiempo inválido"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_routine_deviation(
            activity="despertar",
            expected_time_utc="invalid",
            actual_time_utc="07:00",
        )

        assert alert is None


class TestPatternAlertsActivity:
    """Tests para detección de actividad anómala"""

    @pytest.mark.asyncio
    async def test_normal_activity(self):
        """Test actividad dentro de rango normal"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_unusual_activity(
            activity_type="energy_usage",
            current_value=100.0,
            normal_baseline=100.0,
            threshold_percent=120.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_high_energy_usage(self):
        """Test uso de energía anómalo"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_unusual_activity(
            activity_type="energy_usage",
            current_value=250.0,
            normal_baseline=100.0,
            threshold_percent=120.0,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PATTERN
        assert "250" in alert.message

    @pytest.mark.asyncio
    async def test_temperature_deviation(self):
        """Test desviación de temperatura"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_unusual_activity(
            activity_type="temperature",
            current_value=28.5,
            normal_baseline=22.0,
            threshold_percent=3.0,
        )

        assert alert is not None
        assert "temperature" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_zero_baseline_handled(self):
        """Test que se maneja baseline cero"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_unusual_activity(
            activity_type="energy_usage",
            current_value=100.0,
            normal_baseline=0.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_default_threshold_used(self):
        """Test que se usa umbral por defecto"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_unusual_activity(
            activity_type="energy_usage",
            current_value=250.0,
            normal_baseline=100.0,
            # No especificar threshold, usar default
        )

        assert alert is not None


class TestPatternAlertsSleep:
    """Tests para detección de anomalías de sueño"""

    @pytest.mark.asyncio
    async def test_sleep_normal(self):
        """Test sueño dentro de rango normal"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_sleep_pattern_anomaly(
            sleep_duration_hours=8.0,
            expected_hours=8.0,
            deviation_threshold=1.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_sleep_insufficient(self):
        """Test sueño insuficiente"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_sleep_pattern_anomaly(
            sleep_duration_hours=5.0,
            expected_hours=8.0,
            deviation_threshold=2.0,
        )

        assert alert is not None
        assert alert.priority == AlertPriority.MEDIUM
        assert "insuficiente" in alert.message.lower()

    @pytest.mark.asyncio
    async def test_sleep_excessive(self):
        """Test sueño excesivo"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_sleep_pattern_anomaly(
            sleep_duration_hours=12.0,
            expected_hours=8.0,
            deviation_threshold=2.0,
        )

        assert alert is not None
        assert alert.priority == AlertPriority.LOW
        assert "excesivo" in alert.message.lower()


class TestPatternAlertsActivityGap:
    """Tests para detección de gaps de actividad"""

    @pytest.mark.asyncio
    async def test_activity_gap_normal(self):
        """Test gap de actividad dentro de normal"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_activity_gap(
            zone="sala",
            last_activity_hours_ago=2.0,
            max_gap_hours=4.0,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_activity_gap_long(self):
        """Test gap de actividad muy largo"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_activity_gap(
            zone="dormitorio",
            last_activity_hours_ago=6.0,
            max_gap_hours=4.0,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PATTERN
        assert alert.priority == AlertPriority.MEDIUM
        assert "dormitorio" in alert.message

    @pytest.mark.asyncio
    async def test_activity_gap_details(self):
        """Test detalles de gap de actividad"""
        manager = AlertManager()
        pattern = PatternAlerts(manager)

        alert = await pattern.check_activity_gap(
            zone="cocina",
            last_activity_hours_ago=5.5,
            max_gap_hours=3.0,
        )

        assert alert.details["zone"] == "cocina"
        assert alert.details["last_activity_hours"] == 5.5
        assert alert.details["max_gap_hours"] == 3.0


class TestPatternAlertsDeduplication:
    """Tests para deduplicación en alertas de patrón"""

    @pytest.mark.asyncio
    async def test_same_routine_deviation_deduped(self):
        """Test que desviaciones idénticas se deduplicen"""
        manager = AlertManager(cooldown_seconds=5)
        pattern = PatternAlerts(manager)

        alert1 = await pattern.check_routine_deviation(
            activity="despertar",
            expected_time_utc="07:00",
            actual_time_utc="09:00",
            tolerance_minutes=60,
        )
        assert alert1 is not None

        alert2 = await pattern.check_routine_deviation(
            activity="despertar",
            expected_time_utc="07:00",
            actual_time_utc="09:00",
            tolerance_minutes=60,
        )
        assert alert2 is None
