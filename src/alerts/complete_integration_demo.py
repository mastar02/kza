"""
Complete Integration Demo: Sistema de Alertas Full Stack

Este archivo demuestra una integración completa del sistema de alertas
incluyendo:
- Inicialización del sistema
- Simulación de eventos de Home Assistant
- Procesamiento de comandos de voz
- Manejo de alertas con handlers personalizados

Ejecutar con:
    python src/alerts/complete_integration_demo.py
"""

import asyncio
import logging
from datetime import datetime

from src.alerts import (
    AlertManager,
    AlertPriority,
    AlertType,
    SecurityAlerts,
    PatternAlerts,
    DeviceAlerts,
    AlertScheduler,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PARTE 1: Simulación de Home Assistant Client
# =============================================================================

class MockHAClient:
    """Cliente simulado de Home Assistant para demo"""

    def __init__(self):
        self.entities = {
            "binary_sensor.puerta_entrada": {"state": "off"},
            "binary_sensor.puerta_garaje": {"state": "off"},
            "binary_sensor.movimiento_dormitorio": {"state": "off"},
            "sensor.temperatura_sala": {"state": 22.5},
            "sensor.bateria_sensor_temp": {"state": 50},
            "binary_sensor.luz_entrada": {"state": "off"},
        }

    async def get_state(self, entity_id: str):
        """Obtener estado de entidad"""
        return self.entities.get(entity_id, {}).get("state")

    async def trigger_door_open(self, location: str):
        """Simular puerta abierta"""
        entity = f"binary_sensor.puerta_{location}"
        if entity in self.entities:
            self.entities[entity]["state"] = "on"
            logger.info(f"[HA] Door opened: {location}")

    async def trigger_motion(self, location: str):
        """Simular movimiento"""
        entity = f"binary_sensor.movimiento_{location}"
        if entity in self.entities:
            self.entities[entity]["state"] = "on"
            logger.info(f"[HA] Motion detected: {location}")

    async def set_battery_low(self, sensor_name: str, level: int):
        """Simular batería baja"""
        entity = f"sensor.bateria_{sensor_name}"
        if entity in self.entities:
            self.entities[entity]["state"] = level
            logger.info(f"[HA] Battery low: {sensor_name} = {level}%")


# =============================================================================
# PARTE 2: Servicio de TTS simulado
# =============================================================================

class MockTTSService:
    """Servicio TTS simulado"""

    def __init__(self):
        self.messages = []

    async def speak(self, message: str):
        """Simular reproducción de audio"""
        self.messages.append(message)
        logger.info(f"[TTS] {message}")


# =============================================================================
# PARTE 3: Sistema de Alertas Integrado
# =============================================================================

class IntegratedAlertSystem:
    """Sistema de alertas integrado con simuladores"""

    def __init__(self):
        # Servicios simulados
        self.ha_client = MockHAClient()
        self.tts_service = MockTTSService()

        # Sistema de alertas
        self.alert_manager = AlertManager(
            cooldown_seconds=5.0,  # Corto para demo
            max_history=100,
            tts_callback=self.tts_service.speak,
        )

        # Managers especializados
        self.security = SecurityAlerts(self.alert_manager)
        self.patterns = PatternAlerts(self.alert_manager)
        self.devices = DeviceAlerts(self.alert_manager)

        # Scheduler
        self.scheduler = AlertScheduler(
            alert_manager=self.alert_manager,
            security_interval=5.0,  # Cada 5 segundos para demo
            pattern_interval=10.0,
            device_interval=15.0,
        )

        # Registrar handlers
        self._register_handlers()

    def _register_handlers(self):
        """Registrar handlers para diferentes tipos de alertas"""

        async def security_handler(alert):
            logger.warning(f"[SECURITY HANDLER] {alert.message}")

        async def device_handler(alert):
            logger.warning(f"[DEVICE HANDLER] {alert.message}")

        async def pattern_handler(alert):
            logger.info(f"[PATTERN HANDLER] {alert.message}")

        self.alert_manager.register_handler(AlertType.SECURITY, security_handler)
        self.alert_manager.register_handler(AlertType.DEVICE, device_handler)
        self.alert_manager.register_handler(AlertType.PATTERN, pattern_handler)

    async def simulate_security_event(self, zone: str):
        """Simular evento de seguridad"""
        logger.info(f"\n--- Simulando evento de seguridad: {zone} ---")
        await self.ha_client.trigger_door_open(zone)
        await self.security.check_door_status(
            zone=zone,
            is_open=True,
            expected_open=False,
        )

    async def simulate_device_event(self, device_name: str, battery: int):
        """Simular evento de dispositivo"""
        logger.info(f"\n--- Simulando evento de dispositivo: {device_name} ---")
        await self.ha_client.set_battery_low(device_name, battery)
        await self.devices.check_battery_level(
            device_name=device_name,
            battery_percent=battery,
            device_type="sensor_temperature",
        )

    async def simulate_pattern_event(self):
        """Simular evento de patrón anómalo"""
        logger.info(f"\n--- Simulando evento de patrón ---")
        await self.patterns.check_unusual_activity(
            activity_type="energy_usage",
            current_value=500.0,
            normal_baseline=200.0,
            threshold_percent=120.0,
        )

    async def display_status(self):
        """Mostrar estado del sistema"""
        logger.info("\n" + "=" * 70)
        logger.info("ESTADO DEL SISTEMA DE ALERTAS")
        logger.info("=" * 70)

        stats = self.alert_manager.get_stats()
        logger.info(f"Total de alertas: {stats['total_alerts']}")
        logger.info(f"Alertas pendientes: {stats['pending_alerts']}")
        logger.info(f"Alertas procesadas: {stats['processed_alerts']}")

        summary = self.alert_manager.get_pending_summary()
        logger.info(f"\nResumen pendientes:")
        logger.info(f"  Total: {summary['total_pending']}")
        if summary["by_type"]:
            logger.info(f"  Por tipo: {summary['by_type']}")
        if summary["by_priority"]:
            logger.info(f"  Por prioridad: {summary['by_priority']}")

        history = self.alert_manager.get_history(limit=5)
        if history:
            logger.info(f"\nÚltimas {len(history)} alertas:")
            for i, alert in enumerate(history, 1):
                logger.info(
                    f"  {i}. [{alert.priority.name}] {alert.message} "
                    f"(id={alert.alert_id})"
                )

        logger.info("=" * 70 + "\n")


# =============================================================================
# PARTE 4: Demo Principal
# =============================================================================

async def demo():
    """Ejecutar demo completa"""

    logger.info("\n")
    logger.info("=" * 70)
    logger.info("DEMOSTRACIÓN COMPLETA: SISTEMA DE ALERTAS")
    logger.info("=" * 70)

    # Crear sistema integrado
    system = IntegratedAlertSystem()

    logger.info("\n[INIT] Sistema de alertas inicializado")
    logger.info(f"[INIT] TTS callback configurado: {system.alert_manager.tts_callback is not None}")
    logger.info(f"[INIT] Scheduler configurado con intervalos:")
    logger.info(f"       - Seguridad: 5s")
    logger.info(f"       - Patrones: 10s")
    logger.info(f"       - Dispositivos: 15s")

    # Mostrar estado inicial
    await system.display_status()

    # Simular eventos
    logger.info("\n[DEMO] Iniciando simulación de eventos...\n")

    # Evento 1: Puerta abierta
    await system.simulate_security_event("entrada")
    await asyncio.sleep(2)

    # Evento 2: Batería baja
    await system.simulate_device_event("sensor_temp_sala", 15)
    await asyncio.sleep(2)

    # Evento 3: Patrón anómalo
    await system.simulate_pattern_event()
    await asyncio.sleep(2)

    # Evento 4: Segunda puerta abierta
    await system.simulate_security_event("garaje")
    await asyncio.sleep(2)

    # Mostrar estado después de eventos
    await system.display_status()

    # Simular procesamiento de alertas
    logger.info("\n[DEMO] Procesando alertas...\n")
    history = system.alert_manager.get_history()
    if history:
        first_alert = history[0]
        logger.info(f"Marcando alerta como procesada: {first_alert.alert_id}")
        await system.alert_manager.mark_processed(first_alert.alert_id)

    # Mostrar estado final
    await system.display_status()

    # Mostrar mensajes de TTS
    logger.info("\n[DEMO] Mensajes de TTS generados:")
    for i, msg in enumerate(system.tts_service.messages, 1):
        logger.info(f"  {i}. {msg}")

    logger.info("\n" + "=" * 70)
    logger.info("DEMOSTRACIÓN COMPLETADA")
    logger.info("=" * 70 + "\n")


# =============================================================================
# Ejemplo de integración con scheduler activo
# =============================================================================

async def demo_with_scheduler():
    """Demo con scheduler ejecutándose en background"""

    logger.info("\n")
    logger.info("=" * 70)
    logger.info("DEMOSTRACIÓN CON SCHEDULER ACTIVO")
    logger.info("=" * 70)

    # Crear sistema
    system = IntegratedAlertSystem()

    # Iniciar scheduler
    logger.info("\n[SCHEDULER] Iniciando scheduler de alertas...")
    scheduler_task = asyncio.create_task(system.scheduler.start())

    try:
        # Esperar a que scheduler esté listo
        await asyncio.sleep(1)

        # Simular eventos durante 30 segundos
        for i in range(3):
            logger.info(f"\n[DEMO] Evento {i + 1}/3")
            await system.simulate_security_event("entrada")
            await asyncio.sleep(8)

        # Mostrar estado
        await system.display_status()

    finally:
        # Detener scheduler
        logger.info("\n[SCHEDULER] Deteniendo scheduler...")
        await system.scheduler.stop()
        await asyncio.sleep(1)

    logger.info("=" * 70 + "\n")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    # Correr demo simple
    asyncio.run(demo())

    # Uncomment para correr con scheduler:
    # asyncio.run(demo_with_scheduler())
