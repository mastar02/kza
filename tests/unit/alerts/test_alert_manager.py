"""
Tests for AlertManager module.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.alerts import AlertManager, AlertPriority, AlertType, Alert


class TestAlertManagerBasic:
    """Tests básicos para AlertManager"""

    def test_initialization(self):
        """Test inicialización del AlertManager"""
        manager = AlertManager(
            cooldown_seconds=300,
            max_history=1000,
            tts_callback=None,
        )
        assert manager.cooldown_seconds == 300
        assert manager.max_history == 1000
        assert manager.tts_callback is None

    def test_initialization_with_callback(self):
        """Test inicialización con callback TTS"""
        callback = Mock()
        manager = AlertManager(tts_callback=callback)
        assert manager.tts_callback == callback

    @pytest.mark.asyncio
    async def test_create_alert_basic(self):
        """Test crear alerta básica"""
        manager = AlertManager()

        alert = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test alert",
            details={"zone": "entrada"},
        )

        assert alert is not None
        assert alert.alert_type == AlertType.SECURITY
        assert alert.priority == AlertPriority.HIGH
        assert alert.message == "Test alert"
        assert alert.details["zone"] == "entrada"
        assert not alert.processed

    @pytest.mark.asyncio
    async def test_create_alert_with_deduplication(self):
        """Test deduplicación de alertas"""
        manager = AlertManager(cooldown_seconds=5)

        # Primera alerta
        alert1 = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Puerta abierta",
        )
        assert alert1 is not None

        # Segunda alerta idéntica (debe ser deduplicada)
        alert2 = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Puerta abierta",
        )
        assert alert2 is None

        # Alerta diferente (debe crearse)
        alert3 = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Ventana abierta",
        )
        assert alert3 is not None


class TestAlertManagerHandlers:
    """Tests para handlers de alertas"""

    @pytest.mark.asyncio
    async def test_register_and_execute_handler(self):
        """Test registrar y ejecutar handler"""
        manager = AlertManager()
        handler = AsyncMock()

        manager.register_handler(AlertType.SECURITY, handler)

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test",
        )

        # Verificar que el handler fue llamado
        handler.assert_called_once()
        call_args = handler.call_args[0][0]
        assert isinstance(call_args, Alert)
        assert call_args.message == "Test"

    @pytest.mark.asyncio
    async def test_register_sync_handler(self):
        """Test registrar handler síncrono"""
        manager = AlertManager()
        handler = Mock()

        manager.register_handler(AlertType.DEVICE, handler)

        await manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.MEDIUM,
            message="Battery low",
        )

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_type(self):
        """Test múltiples handlers para el mismo tipo"""
        manager = AlertManager()
        handler1 = AsyncMock()
        handler2 = Mock()

        manager.register_handler(AlertType.PATTERN, handler1)
        manager.register_handler(AlertType.PATTERN, handler2)

        await manager.create_alert(
            alert_type=AlertType.PATTERN,
            priority=AlertPriority.MEDIUM,
            message="Unusual activity",
        )

        handler1.assert_called_once()
        handler2.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_handler(self):
        """Test desregistrar handler"""
        manager = AlertManager()
        handler = AsyncMock()

        manager.register_handler(AlertType.SECURITY, handler)
        manager.unregister_handler(AlertType.SECURITY, handler)

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test",
        )

        handler.assert_not_called()


class TestAlertManagerVoiceNotification:
    """Tests para notificaciones de voz"""

    @pytest.mark.asyncio
    async def test_voice_notification_critical_alert(self):
        """Test notificación de voz para alerta crítica"""
        tts_callback = AsyncMock()
        manager = AlertManager(tts_callback=tts_callback)

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.CRITICAL,
            message="Intruso detectado",
        )

        tts_callback.assert_called_once()
        call_args = tts_callback.call_args[0][0]
        assert "Alerta crítica:" in call_args
        assert "Intruso detectado" in call_args

    @pytest.mark.asyncio
    async def test_voice_notification_high_alert(self):
        """Test notificación de voz para alerta alta"""
        tts_callback = AsyncMock()
        manager = AlertManager(tts_callback=tts_callback)

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Puerta abierta",
        )

        tts_callback.assert_called_once()
        call_args = tts_callback.call_args[0][0]
        assert "Alerta importante:" in call_args

    @pytest.mark.asyncio
    async def test_no_voice_notification_medium_alert(self):
        """Test sin notificación de voz para alerta media"""
        tts_callback = AsyncMock()
        manager = AlertManager(tts_callback=tts_callback)

        await manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.MEDIUM,
            message="Battery low",
        )

        # No debería llamarse para alertas MEDIUM o LOW
        tts_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_notification_sync_callback(self):
        """Test notificación de voz con callback síncrono"""
        tts_callback = Mock()
        manager = AlertManager(tts_callback=tts_callback)

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.CRITICAL,
            message="Alerta de seguridad",
        )

        tts_callback.assert_called_once()


class TestAlertManagerHistory:
    """Tests para historial de alertas"""

    @pytest.mark.asyncio
    async def test_history_single_alert(self):
        """Test agregar alerta al historial"""
        manager = AlertManager()

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test alert",
        )

        history = manager.get_history()
        assert len(history) == 1
        assert history[0].message == "Test alert"

    @pytest.mark.asyncio
    async def test_history_multiple_alerts(self):
        """Test múltiples alertas en historial"""
        manager = AlertManager()

        for i in range(3):
            await manager.create_alert(
                alert_type=AlertType.DEVICE,
                priority=AlertPriority.MEDIUM,
                message=f"Alert {i}",
            )

        history = manager.get_history()
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_history_limit(self):
        """Test límite de historial"""
        manager = AlertManager(max_history=5)

        for i in range(10):
            await manager.create_alert(
                alert_type=AlertType.DEVICE,
                priority=AlertPriority.LOW,
                message=f"Alert {i}",
            )

        assert len(manager._history) == 5

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        """Test obtener historial con límite"""
        manager = AlertManager()

        for i in range(5):
            await manager.create_alert(
                alert_type=AlertType.DEVICE,
                priority=AlertPriority.LOW,
                message=f"Alert {i}",
            )

        history = manager.get_history(limit=2)
        assert len(history) == 2
        # Las más recientes primero
        assert history[0].message == "Alert 4"
        assert history[1].message == "Alert 3"

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test limpiar historial"""
        manager = AlertManager()

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test",
        )

        manager.clear_history()
        assert len(manager.get_history()) == 0


class TestAlertManagerPending:
    """Tests para alertas pendientes"""

    @pytest.mark.asyncio
    async def test_pending_summary_empty(self):
        """Test resumen de pendientes vacío"""
        manager = AlertManager()
        summary = manager.get_pending_summary()

        assert summary["total_pending"] == 0
        assert summary["by_type"] == {}
        assert summary["by_priority"] == {}

    @pytest.mark.asyncio
    async def test_pending_summary_with_alerts(self):
        """Test resumen de pendientes con alertas"""
        manager = AlertManager()

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.CRITICAL,
            message="Alert 1",
        )
        await manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.MEDIUM,
            message="Alert 2",
        )
        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Alert 3",
        )

        summary = manager.get_pending_summary()
        assert summary["total_pending"] == 3
        assert summary["by_type"]["security"] == 2
        assert summary["by_type"]["device"] == 1
        assert summary["by_priority"]["CRITICAL"] == 1
        assert summary["by_priority"]["HIGH"] == 1
        assert summary["by_priority"]["MEDIUM"] == 1

    @pytest.mark.asyncio
    async def test_mark_processed(self):
        """Test marcar alerta como procesada"""
        manager = AlertManager()

        alert = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test",
        )

        assert not alert.processed

        # Marcar como procesada
        success = await manager.mark_processed(alert.alert_id)
        assert success

        # Verificar que está procesada
        summary = manager.get_pending_summary()
        assert summary["total_pending"] == 0

        # Verificar en historial
        updated_alert = manager.get_alert(alert.alert_id)
        assert updated_alert.processed
        assert updated_alert.processed_at is not None


class TestAlertManagerStats:
    """Tests para estadísticas"""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test obtener estadísticas"""
        manager = AlertManager()

        await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.CRITICAL,
            message="Alert 1",
        )
        await manager.create_alert(
            alert_type=AlertType.DEVICE,
            priority=AlertPriority.LOW,
            message="Alert 2",
        )

        stats = manager.get_stats()

        assert stats["total_alerts"] == 2
        assert stats["pending_alerts"] == 2
        assert stats["processed_alerts"] == 0
        assert stats["by_type"]["security"] == 1
        assert stats["by_type"]["device"] == 1
        assert stats["by_priority"]["CRITICAL"] == 1
        assert stats["by_priority"]["LOW"] == 1


class TestAlertDataclass:
    """Tests para la clase Alert"""

    @pytest.mark.asyncio
    async def test_alert_to_dict(self):
        """Test convertir Alert a diccionario"""
        manager = AlertManager()

        alert = await manager.create_alert(
            alert_type=AlertType.SECURITY,
            priority=AlertPriority.HIGH,
            message="Test alert",
            details={"zone": "entrada"},
        )

        alert_dict = alert.to_dict()

        assert alert_dict["alert_type"] == "security"
        assert alert_dict["priority"] == "HIGH"
        assert alert_dict["message"] == "Test alert"
        assert alert_dict["details"]["zone"] == "entrada"
        assert not alert_dict["processed"]
