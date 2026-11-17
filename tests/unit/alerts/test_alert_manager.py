"""
Tests for AlertManager module.
"""

import pytest
from unittest.mock import Mock, AsyncMock

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


# ==================== T036-U1: truthful alert core ====================

import asyncio
from datetime import UTC, datetime, timedelta

from src.alerts import AlertCondition, AlertDelivery, SecurityAlerts
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, VoiceDeliveryError


class FakeClock:
    """Monotonic clock advanced by the test."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t


def _ok(text: str = "x") -> ResponseResult:
    return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.COMPLETE)


DOOR = AlertCondition(source="entrada", condition="door_open_unexpected")


class TestConditionKeyAndEpisodeLatch:
    """The dedup key is the condition; an open episode alerts once."""

    @pytest.mark.asyncio
    async def test_same_condition_different_text_is_one_alert(self):
        manager = AlertManager()
        first = await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "Batería 19%",
                                           condition=AlertCondition("s1", "battery_low"))
        second = await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "Batería 18%",
                                            condition=AlertCondition("s1", "battery_low"))
        assert first is not None and second is None
        assert len(manager.get_history()) == 1
        assert first.source == "s1" and first.condition == "battery_low"

    @pytest.mark.asyncio
    async def test_key_includes_type_source_and_condition(self):
        manager = AlertManager()
        base = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "m",
                                          condition=AlertCondition("s1", "c1"))
        other_type = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "m",
                                                condition=AlertCondition("s1", "c1"))
        other_source = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "m",
                                                  condition=AlertCondition("s2", "c1"))
        other_condition = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "m",
                                                     condition=AlertCondition("s1", "c2"))
        assert None not in (base, other_type, other_source, other_condition)
        assert len(manager.get_history()) == 4

    @pytest.mark.asyncio
    async def test_door_open_twenty_minutes_checked_every_minute_announces_once(self):
        clock = FakeClock()
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(cooldown_seconds=300, tts_callback=tts, clock=clock)
        security = SecurityAlerts(manager)
        created = []
        for _ in range(21):
            created.append(await security.check_door_status("entrada", is_open=True))
            clock.t += 60
        assert created[0] is not None
        assert created[1:] == [None] * 20
        assert tts.await_count == 1
        assert len(manager.get_history()) == 1
        assert created[0].last_seen_at is not None

    @pytest.mark.asyncio
    async def test_close_and_reopen_gives_a_new_announcement(self):
        clock = FakeClock()
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(cooldown_seconds=300, tts_callback=tts, clock=clock)
        security = SecurityAlerts(manager)
        first = await security.check_door_status("entrada", is_open=True)
        clock.t += 1
        assert await security.check_door_status("entrada", is_open=False) is None
        clock.t += 1
        second = await security.check_door_status("entrada", is_open=True)
        assert first is not None and second is not None
        assert first.alert_id != second.alert_id
        assert first.resolved_at is not None and second.resolved_at is None
        assert tts.await_count == 2

    @pytest.mark.asyncio
    async def test_latched_episode_outlives_cooldown(self):
        clock = FakeClock()
        manager = AlertManager(cooldown_seconds=5, clock=clock)
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        clock.t += 10_000
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR) is None

    @pytest.mark.asyncio
    async def test_momentary_event_episode_closes_after_cooldown(self):
        clock = FakeClock()
        manager = AlertManager(cooldown_seconds=300, clock=clock)
        event = AlertCondition("entrada", "access_denied:ana", latched=False)
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=event)
        clock.t += 299.9
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=event) is None
        clock.t += 0.1
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=event)
        assert len(manager.get_history()) == 2

    @pytest.mark.asyncio
    async def test_legacy_message_key_keeps_cooldown_semantics(self):
        clock = FakeClock()
        manager = AlertManager(cooldown_seconds=300, clock=clock)
        first = await manager.create_alert(AlertType.PATTERN, AlertPriority.LOW, "Patrón")
        clock.t += 299
        assert await manager.create_alert(AlertType.PATTERN, AlertPriority.LOW, "Patrón") is None
        clock.t += 1
        assert await manager.create_alert(AlertType.PATTERN, AlertPriority.LOW, "Patrón") is not None
        assert first.source == "message" and first.condition == "Patrón"

    @pytest.mark.asyncio
    async def test_resolve_closes_episode_and_reports_it(self):
        manager = AlertManager(now=lambda: datetime(2026, 11, 16, 12, 0, tzinfo=UTC))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        resolved = await manager.resolve_condition(AlertType.SECURITY, DOOR)
        assert resolved is alert
        assert alert.resolved_at == datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
        assert await manager.resolve_condition(AlertType.SECURITY, DOOR) is None
        assert manager.get_stats()["dedup_keys"] == 0

    @pytest.mark.asyncio
    async def test_resolve_other_type_does_not_close(self):
        manager = AlertManager()
        await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        assert await manager.resolve_condition(AlertType.DEVICE, DOOR) is None
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR) is None

    @pytest.mark.asyncio
    async def test_clear_cooldowns_forgets_open_episodes(self):
        manager = AlertManager()
        await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        manager.clear_cooldowns()
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)

    @pytest.mark.asyncio
    async def test_clear_cooldowns_never_leaves_two_active_alerts_for_one_condition(self):
        """Review R5: the cleared episode's alert is resolved; only the reopened one is active."""
        instant = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
        manager = AlertManager(now=lambda: instant)
        first = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        manager.clear_cooldowns()
        assert first.resolved_at == instant
        assert manager.get_active_alerts() == []
        second = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        assert second is not first
        assert manager.get_active_alerts() == [second]
        assert await manager.resolve_condition(AlertType.SECURITY, DOOR) is second
        assert manager.get_active_alerts() == []


class TestEscalationAndReannounce:
    """A more severe reading escalates the open alert; re-announce only by explicit config."""

    @pytest.mark.asyncio
    async def test_escalation_updates_in_place_and_announces(self):
        tts = AsyncMock(return_value=_ok())
        handler = Mock()
        manager = AlertManager(tts_callback=tts)
        manager.register_handler(AlertType.DEVICE, handler)
        cond = AlertCondition("s1", "battery_low")
        first = await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "15%", {"p": 15}, condition=cond)
        assert tts.await_count == 0 and first.delivery is AlertDelivery.NOT_SPOKEN
        escalated = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "8%", {"p": 8}, condition=cond)
        assert escalated is first
        assert first.priority is AlertPriority.HIGH and first.message == "8%" and first.details == {"p": 8}
        assert tts.await_count == 1 and "8%" in tts.await_args.args[0]
        assert first.delivery is AlertDelivery.DELIVERED
        assert handler.call_count == 2
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "7%", condition=cond) is None
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "15%", condition=cond) is None
        assert first.priority is AlertPriority.HIGH and first.message == "8%"
        assert tts.await_count == 1
        assert len(manager.get_history()) == 1

    @pytest.mark.asyncio
    async def test_escalation_after_ack_makes_the_alert_active_again(self):
        """Review R1: battery 19% acked, then 8% is spoken and back in the active list."""
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(tts_callback=tts)
        cond = AlertCondition("sensor_puerta", "battery_low")
        alert = await manager.create_alert(
            AlertType.DEVICE, AlertPriority.MEDIUM, "Batería baja en sensor puerta: 19%", condition=cond)
        assert await manager.mark_processed(alert.alert_id)
        assert manager.get_active_alerts() == [] and alert.processed_at is not None
        escalated = await manager.create_alert(
            AlertType.DEVICE, AlertPriority.HIGH, "Batería baja en sensor puerta: 8%", condition=cond)
        assert escalated is alert
        assert tts.await_args.args[0] == "Alerta importante: Batería baja en sensor puerta: 8%"
        assert alert.processed is False and alert.processed_at is None
        assert manager.get_active_alerts() == [alert]
        assert manager.get_stats()["pending_alerts"] == 1

    @pytest.mark.asyncio
    async def test_escalation_to_unspoken_priority_reactivates_acked_alert(self):
        """Any rise in priority reopens the acknowledgement, spoken or not."""
        manager = AlertManager(tts_callback=AsyncMock(return_value=_ok()))
        cond = AlertCondition("s1", "c")
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a", condition=cond)
        await manager.mark_processed(alert.alert_id)
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "b", condition=cond) is alert
        assert manager.get_active_alerts() == [alert] and alert.processed_at is None

    @pytest.mark.asyncio
    async def test_repeat_or_lower_reading_keeps_the_ack(self):
        """Only an escalation reopens an acknowledged alert."""
        manager = AlertManager(tts_callback=AsyncMock(return_value=_ok()))
        cond = AlertCondition("s1", "c")
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "a", condition=cond)
        await manager.mark_processed(alert.alert_id)
        acked_at = alert.processed_at
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.HIGH, "a", condition=cond) is None
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a", condition=cond) is None
        assert alert.processed is True and alert.processed_at == acked_at
        assert manager.get_active_alerts() == []

    @pytest.mark.asyncio
    async def test_escalation_to_unspoken_priority_is_not_announced(self):
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(tts_callback=tts)
        cond = AlertCondition("s1", "c")
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a", condition=cond)
        assert await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "b", condition=cond) is alert
        assert tts.await_count == 0

    @pytest.mark.asyncio
    async def test_no_reannounce_by_default(self):
        clock = FakeClock()
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(tts_callback=tts, clock=clock)
        await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR)
        clock.t += 10 ** 7
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR) is None
        assert tts.await_count == 1

    @pytest.mark.asyncio
    async def test_explicit_interval_reannounces_open_episode(self):
        clock = FakeClock()
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(tts_callback=tts, clock=clock, reannounce_interval_seconds=600)
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR)
        clock.t += 599
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR) is None
        assert tts.await_count == 1
        clock.t += 1
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR) is None
        assert tts.await_count == 2
        clock.t += 599
        await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR)
        assert tts.await_count == 2
        assert len(manager.get_history()) == 1 and alert.delivery is AlertDelivery.DELIVERED

    @pytest.mark.asyncio
    async def test_interval_never_reannounces_unspoken_or_momentary(self):
        clock = FakeClock()
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(tts_callback=tts, clock=clock, cooldown_seconds=10_000,
                               reannounce_interval_seconds=60)
        await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a", condition=AlertCondition("s", "c"))
        event = AlertCondition("z", "access_denied:x", latched=False)
        await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "e", condition=event)
        clock.t += 61
        await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a", condition=AlertCondition("s", "c"))
        await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "e", condition=event)
        assert tts.await_count == 1

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "60"])
    def test_invalid_reannounce_interval_rejected(self, bad):
        with pytest.raises(ValueError):
            AlertManager(reannounce_interval_seconds=bad)

    def test_valid_reannounce_interval_kept(self):
        assert AlertManager(reannounce_interval_seconds=0.5).reannounce_interval_seconds == 0.5
        assert AlertManager().reannounce_interval_seconds is None


class TestVoiceDeliveryReceipt:
    """The alert records what the voice output actually did."""

    @pytest.mark.asyncio
    async def test_delivered_receipt(self):
        manager = AlertManager(tts_callback=AsyncMock(return_value=_ok()),
                               now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert alert.delivery is AlertDelivery.DELIVERED
        assert alert.delivery_reason is None
        assert alert.delivery_at == datetime(2026, 1, 1, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_failed_result_recorded_not_delivered(self):
        failed = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.FAILED, error="audio_device_lost")
        manager = AlertManager(tts_callback=AsyncMock(return_value=failed))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, "a")
        assert alert.delivery is AlertDelivery.FAILED
        assert alert.delivery_reason == "audio_device_lost"
        assert alert.delivery_at is not None

    @pytest.mark.asyncio
    async def test_failed_result_without_error_names_state(self):
        result = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.TIMED_OUT)
        manager = AlertManager(tts_callback=AsyncMock(return_value=result))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "voice_delivery_timed_out")

    @pytest.mark.asyncio
    async def test_complete_delivery_with_error_is_not_delivered(self):
        result = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.COMPLETE, error="late_error")
        manager = AlertManager(tts_callback=AsyncMock(return_value=result))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "late_error")

    @pytest.mark.asyncio
    async def test_blocked_result(self):
        blocked = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.BLOCKED, error="policy")
        manager = AlertManager(tts_callback=AsyncMock(return_value=blocked))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.BLOCKED, "policy")

    @pytest.mark.asyncio
    async def test_blocked_result_without_error(self):
        blocked = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.BLOCKED)
        manager = AlertManager(tts_callback=AsyncMock(return_value=blocked))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.BLOCKED, "voice_delivery_blocked")

    @pytest.mark.asyncio
    async def test_voice_delivery_error_recorded(self):
        manager = AlertManager(tts_callback=AsyncMock(side_effect=VoiceDeliveryError("voice_output_zone_required")))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "voice_output_zone_required")
        assert manager.get_history() == [alert]

    @pytest.mark.asyncio
    async def test_empty_voice_delivery_error_has_reason(self):
        manager = AlertManager(tts_callback=AsyncMock(side_effect=VoiceDeliveryError()))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "voice_delivery_failed")

    @pytest.mark.asyncio
    async def test_unexpected_exception_recorded(self):
        manager = AlertManager(tts_callback=AsyncMock(side_effect=RuntimeError("secret detail")))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "RuntimeError")

    @pytest.mark.asyncio
    async def test_false_result_is_failed(self):
        manager = AlertManager(tts_callback=Mock(return_value=False))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "voice_delivery_failed")

    @pytest.mark.asyncio
    async def test_sync_callback_failed_result(self):
        failed = ResponseResult("a", GenerationState.COMPLETE, DeliveryState.FAILED, error="e")
        manager = AlertManager(tts_callback=Mock(return_value=failed))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert alert.delivery is AlertDelivery.FAILED

    @pytest.mark.asyncio
    async def test_sync_callback_returning_awaitable(self):
        async def receipt():
            return ResponseResult("a", GenerationState.COMPLETE, DeliveryState.BLOCKED, error="b")
        manager = AlertManager(tts_callback=lambda text: receipt())
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert alert.delivery is AlertDelivery.BLOCKED

    @pytest.mark.parametrize("sync", [False, True])
    @pytest.mark.asyncio
    async def test_legacy_receiptless_return_counts_as_delivered(self, sync):
        """Review R5: a callback returning None keeps the invoke_voice_callback convention (delivered)."""
        tts = Mock(return_value=None) if sync else AsyncMock(return_value=None)
        manager = AlertManager(tts_callback=tts)
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert tts.call_count == 1
        assert alert.delivery is AlertDelivery.DELIVERED
        assert alert.delivery_reason is None and alert.delivery_at is not None

    @pytest.mark.asyncio
    async def test_muted_without_voice_output(self):
        manager = AlertManager(tts_callback=None)
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.CRITICAL, "a")
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.MUTED, "voice_notifications_disabled")
        assert alert.delivery_at is not None

    @pytest.mark.asyncio
    async def test_unspoken_priority_keeps_not_spoken(self):
        manager = AlertManager(tts_callback=None)
        alert = await manager.create_alert(AlertType.DEVICE, AlertPriority.MEDIUM, "a")
        assert alert.delivery is AlertDelivery.NOT_SPOKEN and alert.delivery_at is None

    @pytest.mark.asyncio
    async def test_pending_while_output_speaks(self):
        seen = {}
        manager = None

        async def speak(text):
            alert = manager.get_history()[0]
            seen["state"], seen["at"] = alert.delivery, alert.delivery_at
            return _ok(text)
        manager = AlertManager(tts_callback=speak)
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        assert seen == {"state": AlertDelivery.PENDING, "at": None}
        assert alert.delivery is AlertDelivery.DELIVERED

    @pytest.mark.asyncio
    async def test_cancelled_delivery_recorded_and_propagated(self):
        manager = AlertManager(tts_callback=AsyncMock(side_effect=asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a")
        [alert] = manager.get_history()
        assert (alert.delivery, alert.delivery_reason) == (AlertDelivery.FAILED, "voice_delivery_cancelled")

    @pytest.mark.asyncio
    async def test_receipt_in_to_dict(self):
        manager = AlertManager(tts_callback=None, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "a", condition=DOOR)
        await manager.resolve_condition(AlertType.SECURITY, DOOR)
        data = alert.to_dict()
        assert data["delivery"] == "muted"
        assert data["delivery_reason"] == "voice_notifications_disabled"
        assert data["delivery_at"] == "2026-01-01T00:00:00+00:00"
        assert data["source"] == "entrada" and data["condition"] == "door_open_unexpected"
        assert data["resolved_at"] == "2026-01-01T00:00:00+00:00"
        assert data["last_seen_at"] is None
        assert data["timestamp"] == "2026-01-01T00:00:00+00:00"


class TestTimezoneAwareTimestamps:
    """Every stored instant carries an explicit offset."""

    @pytest.mark.asyncio
    async def test_default_timestamps_are_utc_aware(self):
        manager = AlertManager()
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a")
        assert alert.timestamp.utcoffset() == timedelta(0)
        assert abs(datetime.now(UTC) - alert.timestamp) < timedelta(seconds=5)
        await manager.mark_processed(alert.alert_id)
        assert alert.processed_at.utcoffset() == timedelta(0)

    def test_dataclass_default_is_aware(self):
        alert = Alert(alert_id="x", alert_type=AlertType.DEVICE, priority=AlertPriority.LOW, message="m")
        assert alert.timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_injected_now_used_for_creation_seen_and_processed(self):
        instants = iter(datetime(2026, 11, 16, 10, minute, tzinfo=UTC) for minute in range(10))
        manager = AlertManager(now=lambda: next(instants))
        alert = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "a", condition=DOOR)
        await manager.mark_processed(alert.alert_id)
        assert alert.timestamp.minute == 0
        assert alert.last_seen_at.minute == 1
        assert alert.processed_at.minute == 2


class TestActiveAlerts:
    """Active = unacknowledged and condition not resolved."""

    @pytest.mark.asyncio
    async def test_active_excludes_resolved_and_processed(self):
        manager = AlertManager()
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "door", condition=DOOR)
        acked = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "acked")
        live = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "live")
        await manager.resolve_condition(AlertType.SECURITY, DOOR)
        await manager.mark_processed(acked.alert_id)
        assert manager.get_active_alerts() == [live]
        assert door.resolved_at is not None

    @pytest.mark.asyncio
    async def test_active_newest_first(self):
        manager = AlertManager()
        a = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "a")
        b = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "b")
        assert manager.get_active_alerts() == [b, a]


class TestHistoryEvictionKeepsOpenEpisodes:
    """Review R4: trimming the history never hides an open episode that still dedups."""

    @pytest.mark.asyncio
    async def test_evicted_open_episode_is_reopened_not_deduplicated(self):
        """Reviewer probe: max_history=2, door open, three unrelated open alerts, then the door again."""
        manager = AlertManager(max_history=2)
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta", condition=DOOR)
        for i in range(3):
            await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, f"otra {i}")
        assert manager.get_alert(door.alert_id) is None
        again = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta", condition=DOOR)
        assert again is not None and again is not door
        assert again in manager.get_active_alerts()
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta",
                                          condition=DOOR) is None

    @pytest.mark.asyncio
    async def test_unrelated_overflow_evicts_closed_alerts_before_the_open_episode(self):
        """With closed alerts in the history the open door is never the one evicted."""
        clock = FakeClock()
        manager = AlertManager(max_history=2, cooldown_seconds=10, clock=clock)
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta", condition=DOOR)
        for i in range(3):
            await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, f"otra {i}")
            clock.t += 10
        assert manager.get_alert(door.alert_id) is door
        assert door in manager.get_active_alerts()
        assert await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta",
                                          condition=DOOR) is None

    @pytest.mark.asyncio
    async def test_resolved_alert_is_evicted_before_an_open_one(self):
        manager = AlertManager(max_history=2)
        battery = AlertCondition("s1", "battery_low")
        old = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "vieja", condition=battery)
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "puerta", condition=DOOR)
        await manager.resolve_condition(AlertType.DEVICE, battery)
        newest = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "nueva",
                                            condition=AlertCondition("s2", "offline"))
        assert manager.get_history() == [newest, door]
        assert manager.get_alert(old.alert_id) is None

    @pytest.mark.asyncio
    async def test_expired_momentary_event_is_evicted_before_an_open_one(self):
        clock = FakeClock()
        manager = AlertManager(max_history=2, cooldown_seconds=10, clock=clock)
        event = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "evento",
                                           condition=AlertCondition("z", "access_denied:x", latched=False))
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "puerta", condition=DOOR)
        clock.t += 10
        newest = await manager.create_alert(AlertType.DEVICE, AlertPriority.LOW, "nueva")
        assert manager.get_history() == [newest, door]
        assert manager.get_alert(event.alert_id) is None
        assert manager.get_stats()["dedup_keys"] == 2

    @pytest.mark.asyncio
    async def test_history_full_of_open_episodes_drops_the_evicted_episode(self):
        """When every alert is open, the evicted one's episode goes too: the next observation reopens it."""
        tts = AsyncMock(return_value=_ok())
        manager = AlertManager(max_history=1, tts_callback=tts)
        first = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "puerta", condition=DOOR)
        window = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "ventana",
                                            condition=AlertCondition("cocina", "window_open"))
        assert manager.get_history() == [window]
        assert manager.get_stats()["dedup_keys"] == 1
        reopened = await manager.create_alert(AlertType.SECURITY, AlertPriority.HIGH, "puerta", condition=DOOR)
        assert reopened is not None and reopened is not first
        assert manager.get_active_alerts() == [reopened]
        assert tts.await_count == 3
