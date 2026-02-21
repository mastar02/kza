"""
Ejemplo de uso del sistema de alertas proactivas.

Este archivo demuestra cómo usar los diferentes componentes
del sistema de alertas del proyecto KZA.
"""

import asyncio
from datetime import datetime

from alert_manager import AlertManager, AlertPriority, AlertType
from security_alerts import SecurityAlerts
from pattern_alerts import PatternAlerts
from device_alerts import DeviceAlerts


async def tts_callback(message: str) -> None:
    """Callback simulado para notificaciones de voz"""
    print(f"[TTS] {message}")


async def security_handler(alert) -> None:
    """Handler para alertas de seguridad"""
    print(f"[SECURITY HANDLER] {alert.message}")


async def device_handler(alert) -> None:
    """Handler para alertas de dispositivos"""
    print(f"[DEVICE HANDLER] {alert.message}")


async def main():
    """Ejemplo de uso completo del sistema de alertas"""

    # Inicializar AlertManager con callback TTS
    alert_manager = AlertManager(
        cooldown_seconds=5,
        max_history=100,
        tts_callback=tts_callback,
    )

    # Registrar handlers específicos
    alert_manager.register_handler(AlertType.SECURITY, security_handler)
    alert_manager.register_handler(AlertType.DEVICE, device_handler)

    # Inicializar managers específicos
    security = SecurityAlerts(alert_manager)
    patterns = PatternAlerts(alert_manager)
    devices = DeviceAlerts(alert_manager)

    print("=" * 60)
    print("EJEMPLO DE USO: Sistema de Alertas Proactivas KZA")
    print("=" * 60)

    # ========================================
    # Ejemplo 1: Alertas de Seguridad
    # ========================================
    print("\n1. ALERTAS DE SEGURIDAD")
    print("-" * 60)

    # Puerta abierta inesperadamente
    await security.check_door_status(
        zone="entrada",
        is_open=True,
        expected_open=False,
    )

    # Movimiento inusual en zona sin ocupación esperada
    await security.check_unusual_motion(
        zone="dormitorio",
        motion_detected=True,
        expected_occupancy=False,
    )

    # Acceso denegado
    await security.check_access_denied(
        zone="sótano",
        user="desconocido",
        reason="unauthorized_access",
    )

    # ========================================
    # Ejemplo 2: Alertas de Patrones
    # ========================================
    print("\n2. ALERTAS DE PATRONES ANÓMALOS")
    print("-" * 60)

    # Desviación de rutina (despertar más tarde de lo normal)
    await patterns.check_routine_deviation(
        activity="despertar",
        expected_time_utc="07:00",
        actual_time_utc="10:30",
        tolerance_minutes=60,
    )

    # Uso de energía anómalo
    await patterns.check_unusual_activity(
        activity_type="energy_usage",
        current_value=500.0,
        normal_baseline=200.0,
        threshold_percent=120.0,
    )

    # Sueño insuficiente
    await patterns.check_sleep_pattern_anomaly(
        sleep_duration_hours=4.5,
        expected_hours=8.0,
        deviation_threshold=2.0,
    )

    # ========================================
    # Ejemplo 3: Alertas de Dispositivos
    # ========================================
    print("\n3. ALERTAS DE DISPOSITIVOS")
    print("-" * 60)

    # Batería baja
    await devices.check_battery_level(
        device_name="sensor_movimiento_sala",
        battery_percent=12,
        device_type="sensor_motion",
    )

    # Dispositivo offline
    await devices.check_connectivity(
        device_name="luz_inteligente_entrada",
        is_online=False,
        device_type="light",
    )

    # Señal débil
    await devices.check_signal_strength(
        device_name="sensor_temperatura_garaje",
        signal_strength_db=-92.0,
        min_signal_db=-85.0,
    )

    # Actualización de firmware disponible
    await devices.check_firmware_update(
        device_name="hub_central",
        current_version="2.4.0",
        latest_version="2.5.1",
    )

    # ========================================
    # Ejemplo 4: Deduplicación
    # ========================================
    print("\n4. DEDUPLICACIÓN (cooldown de 5 segundos)")
    print("-" * 60)

    # Primera alerta
    print("\nPrimer intento (debe crear alerta):")
    alert1 = await security.check_door_status(
        zone="garaje",
        is_open=True,
        expected_open=False,
    )
    print(f"Resultado: {'ALERTA CREADA' if alert1 else 'DEDUPLICADA'}")

    # Segunda alerta idéntica inmediata (debe ser deduplicada)
    print("\nSegundo intento inmediato (debe ser deduplicada):")
    alert2 = await security.check_door_status(
        zone="garaje",
        is_open=True,
        expected_open=False,
    )
    print(f"Resultado: {'ALERTA CREADA' if alert2 else 'DEDUPLICADA'}")

    # Esperar 6 segundos y reintentar
    print("\nEsperando 6 segundos...")
    await asyncio.sleep(6)

    print("Tercer intento después de esperar (debe crear alerta):")
    alert3 = await security.check_door_status(
        zone="garaje",
        is_open=True,
        expected_open=False,
    )
    print(f"Resultado: {'ALERTA CREADA' if alert3 else 'DEDUPLICADA'}")

    # ========================================
    # Ejemplo 5: Historial y Resumen
    # ========================================
    print("\n5. HISTORIAL Y RESUMEN DE ALERTAS")
    print("-" * 60)

    # Obtener estadísticas
    stats = alert_manager.get_stats()
    print(f"\nEstadísticas generales:")
    print(f"  Total de alertas: {stats['total_alerts']}")
    print(f"  Alertas pendientes: {stats['pending_alerts']}")
    print(f"  Alertas procesadas: {stats['processed_alerts']}")

    # Resumen de pendientes
    summary = alert_manager.get_pending_summary()
    print(f"\nResumen de alertas pendientes:")
    print(f"  Total: {summary['total_pending']}")
    if summary["by_type"]:
        print(f"  Por tipo: {summary['by_type']}")
    if summary["by_priority"]:
        print(f"  Por prioridad: {summary['by_priority']}")

    # Historial reciente
    print(f"\nÚltimas 5 alertas:")
    history = alert_manager.get_history(limit=5)
    for i, alert in enumerate(history, 1):
        print(f"  {i}. [{alert.priority.name}] {alert.message}")

    # ========================================
    # Ejemplo 6: Marcar alertas como procesadas
    # ========================================
    print("\n6. PROCESAR ALERTAS")
    print("-" * 60)

    if history:
        first_alert = history[0]
        print(f"\nMarcando alerta como procesada: {first_alert.alert_id}")
        await alert_manager.mark_processed(first_alert.alert_id)

        # Verificar cambio
        new_summary = alert_manager.get_pending_summary()
        print(f"Alertas pendientes después: {new_summary['total_pending']}")

    print("\n" + "=" * 60)
    print("FIN DEL EJEMPLO")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
