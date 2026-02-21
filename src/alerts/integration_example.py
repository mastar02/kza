"""
Integration Example: How to integrate the Alert System with the main application

Este archivo muestra cómo integrar el sistema de alertas con la aplicación principal.
Incluye ejemplos de:
1. Inicializar AlertManager
2. Conectar con Home Assistant client
3. Registrar handlers de alertas
4. Iniciar el scheduler
5. Comandos de voz para alertas
"""

import asyncio
import logging
from typing import Optional

from src.alerts import (
    AlertManager,
    AlertPriority,
    AlertType,
    SecurityAlerts,
    PatternAlerts,
    DeviceAlerts,
    AlertScheduler,
    CheckType,
)
from src.home_assistant.ha_client import HomeAssistantClient


logger = logging.getLogger(__name__)


# =============================================================================
# SECCIÓN 1: Inicializar el sistema de alertas
# =============================================================================

def initialize_alert_system(
    config: dict,
    tts_callback: Optional[callable] = None,
) -> tuple[AlertManager, AlertScheduler, SecurityAlerts, PatternAlerts, DeviceAlerts]:
    """
    Inicializar todo el sistema de alertas.

    Args:
        config: Configuración del YAML (config['alerts'])
        tts_callback: Callback para notificaciones de voz

    Returns:
        Tupla con (AlertManager, AlertScheduler, SecurityAlerts, PatternAlerts, DeviceAlerts)

    Ejemplo:
        alert_manager, scheduler, security, patterns, devices = initialize_alert_system(
            config=config['alerts'],
            tts_callback=tts_service.speak
        )
    """
    # Obtener configuración
    general_config = config.get("general", {})
    cooldown_seconds = general_config.get("cooldown_seconds", 300.0)
    max_history = general_config.get("max_history", 1000)
    voice_notifications = general_config.get("voice_notifications", True)

    scheduler_config = config.get("scheduler", {})
    security_interval = scheduler_config.get("security_interval_seconds", 60.0)
    pattern_interval = scheduler_config.get("pattern_interval_seconds", 300.0)
    device_interval = scheduler_config.get("device_interval_seconds", 600.0)

    # Crear AlertManager
    alert_manager = AlertManager(
        cooldown_seconds=cooldown_seconds,
        max_history=max_history,
        tts_callback=tts_callback if voice_notifications else None,
    )

    # Crear managers especializados
    security_alerts = SecurityAlerts(alert_manager)
    pattern_alerts = PatternAlerts(alert_manager)
    device_alerts = DeviceAlerts(alert_manager)

    # Crear scheduler
    alert_scheduler = AlertScheduler(
        alert_manager=alert_manager,
        security_interval=security_interval,
        pattern_interval=pattern_interval,
        device_interval=device_interval,
    )

    logger.info("Alert system initialized")
    return alert_manager, alert_scheduler, security_alerts, pattern_alerts, device_alerts


# =============================================================================
# SECCIÓN 2: Registrar handlers para diferentes tipos de alertas
# =============================================================================

def register_alert_handlers(
    alert_manager: AlertManager,
    ha_client: Optional[HomeAssistantClient] = None,
) -> None:
    """
    Registrar handlers para procesar alertas.

    Los handlers pueden:
    - Registrar alertas en una base de datos
    - Enviar notificaciones
    - Activar automatizaciones en Home Assistant
    - Ejecutar comandos específicos

    Args:
        alert_manager: Instancia de AlertManager
        ha_client: Cliente de Home Assistant (opcional)

    Ejemplo:
        register_alert_handlers(alert_manager, ha_client)
    """

    async def security_handler(alert):
        """Handler para alertas de seguridad"""
        logger.warning(f"SECURITY ALERT: {alert.message}")

        # Si hay HA client, se podría activar automáticas
        if ha_client:
            try:
                # Ejemplo: activar grabación de cámara
                if "puerta" in alert.message.lower():
                    # await ha_client.call_service("camera", "record", {"entity_id": "camera.entrada"})
                    pass
            except Exception as e:
                logger.error(f"Error calling HA service: {e}")

    async def device_handler(alert):
        """Handler para alertas de dispositivos"""
        logger.warning(f"DEVICE ALERT: {alert.message}")

        # Registrar en logs o base de datos
        if alert.priority == AlertPriority.HIGH:
            logger.error(f"High priority device alert: {alert.message}")

    async def pattern_handler(alert):
        """Handler para alertas de patrones"""
        logger.info(f"PATTERN ALERT: {alert.message}")

        # Las alertas de patrón son informativas
        # Se podrían usar para sugerencias personalizadas

    # Registrar handlers
    alert_manager.register_handler(AlertType.SECURITY, security_handler)
    alert_manager.register_handler(AlertType.DEVICE, device_handler)
    alert_manager.register_handler(AlertType.PATTERN, pattern_handler)

    logger.info("Alert handlers registered")


# =============================================================================
# SECCIÓN 3: Iniciar el scheduler en background
# =============================================================================

async def start_alert_scheduler(scheduler: AlertScheduler) -> asyncio.Task:
    """
    Iniciar el scheduler en una tarea asyncio.

    Args:
        scheduler: Instancia de AlertScheduler

    Returns:
        Task de asyncio que ejecuta el scheduler

    Ejemplo:
        scheduler_task = await start_alert_scheduler(scheduler)
        # ... más tarde ...
        await scheduler.stop()  # Detener antes de salir
    """
    logger.info("Starting alert scheduler...")
    scheduler_task = asyncio.create_task(scheduler.start())
    return scheduler_task


# =============================================================================
# SECCIÓN 4: Comandos de voz para alertas
# =============================================================================

async def handle_alert_voice_commands(
    command: str,
    alert_manager: AlertManager,
) -> Optional[str]:
    """
    Procesar comandos de voz relacionados con alertas.

    Comandos soportados:
    - "¿Tengo alertas pendientes?" -> resumen de alertas
    - "Mostrar alertas críticas" -> listar alertas críticas
    - "Silenciar alertas por 1 hora" -> snooze
    - "Borrar alertas" -> limpiar historial
    - "Estado de alertas" -> información de sistema

    Args:
        command: Comando de voz procesado
        alert_manager: Instancia de AlertManager

    Returns:
        Mensaje de respuesta para el usuario

    Ejemplo:
        response = await handle_alert_voice_commands(
            "¿Tengo alertas pendientes?",
            alert_manager
        )
        # response = "Tienes 2 alertas pendientes..."
    """
    command_lower = command.lower().strip()

    # Comando 1: Resumen de alertas pendientes
    if any(x in command_lower for x in ["tengo alertas", "alertas pendientes", "cuántas alertas"]):
        summary = alert_manager.get_pending_summary()
        total = summary["total_pending"]

        if total == 0:
            return "No tienes alertas pendientes."

        by_priority = summary.get("by_priority", {})
        critical = by_priority.get("CRITICAL", 0)
        high = by_priority.get("HIGH", 0)

        response = f"Tienes {total} alerta"
        response += "s" if total != 1 else ""
        response += " pendiente"
        response += "s." if total != 1 else "."

        if critical > 0:
            response += f" {critical} crítica"
            response += "s" if critical != 1 else ""
            response += "."

        if high > 0:
            response += f" {high} importante"
            response += "s" if high != 1 else ""
            response += "."

        return response

    # Comando 2: Mostrar alertas críticas
    elif any(x in command_lower for x in ["alertas críticas", "alertas críticas", "mostrar críticas"]):
        history = alert_manager.get_history(limit=10)
        critical_alerts = [a for a in history if a.priority == AlertPriority.CRITICAL]

        if not critical_alerts:
            return "No hay alertas críticas."

        response = f"Tienes {len(critical_alerts)} alerta"
        response += "s" if len(critical_alerts) != 1 else ""
        response += " crítica"
        response += "s:" if len(critical_alerts) != 1 else ":"

        for alert in critical_alerts[:3]:  # Máximo 3
            response += f" {alert.message}."

        return response

    # Comando 3: Silenciar alertas
    elif any(x in command_lower for x in ["silenciar", "snooze", "pausar alertas"]):
        # Aquí iría la lógica para pausar notificaciones de voz
        # Por ahora, solo limpiar cooldowns para permitir nuevas alertas
        alert_manager.clear_cooldowns()
        return "Alertas silenciadas. Los cooldowns han sido reiniciados."

    # Comando 4: Borrar alertas
    elif any(x in command_lower for x in ["borrar alertas", "limpiar alertas", "eliminar alertas"]):
        alert_manager.clear_history()
        return "Historial de alertas eliminado."

    # Comando 5: Estado del sistema de alertas
    elif any(x in command_lower for x in ["estado alertas", "información alertas", "estadísticas"]):
        stats = alert_manager.get_stats()
        return (
            f"Sistema de alertas: {stats['total_alerts']} alertas en total, "
            f"{stats['pending_alerts']} pendientes, "
            f"{stats['processed_alerts']} procesadas."
        )

    # Comando no reconocido
    return None


# =============================================================================
# SECCIÓN 5: Integración en main.py
# =============================================================================

async def main_integration_example():
    """
    Ejemplo de cómo integrar el sistema de alertas en main.py.

    Este es un ejemplo completo que muestra todos los pasos.
    """
    import os
    import sys
    from pathlib import Path
    import yaml
    from dotenv import load_dotenv

    # Cargar configuración
    load_dotenv()
    config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Configuración de alertas
    alerts_config = config.get("alerts", {})

    if not alerts_config.get("enabled"):
        logger.info("Alert system disabled in configuration")
        return

    # PASO 1: Inicializar sistema de alertas
    # Nota: tts_callback sería el servicio de TTS de tu aplicación
    alert_manager, scheduler, security, patterns, devices = initialize_alert_system(
        config=alerts_config,
        tts_callback=None,  # Aquí va tu TTS service
    )

    # PASO 2: Registrar handlers
    # ha_client sería tu cliente de Home Assistant
    register_alert_handlers(
        alert_manager=alert_manager,
        ha_client=None,  # Aquí va tu HA client
    )

    # PASO 3: Iniciar scheduler
    if alerts_config.get("scheduler", {}).get("enabled"):
        scheduler_task = await start_alert_scheduler(scheduler)
        logger.info("Alert scheduler started")

    # PASO 4: Tu aplicación puede crear alertas manualmente
    # Ejemplo: cuando un sensor detecte algo
    alert = await security.check_door_status(
        zone="entrada",
        is_open=True,
        expected_open=False,
    )

    if alert:
        logger.info(f"Alert created: {alert.message}")

    # PASO 5: Procesar comandos de voz
    response = await handle_alert_voice_commands(
        "¿Tengo alertas pendientes?",
        alert_manager,
    )
    logger.info(f"Voice response: {response}")

    # Para detener:
    # await scheduler.stop()

    logger.info("Integration example completed")


# =============================================================================
# Cómo agregar estos comandos a tu sistema de comandos de voz
# =============================================================================

"""
En tu archivo de rutas de comandos o router, agregar:

# En pipeline/voice_router.py o similar

async def route_alert_commands(self, command: str, intent: str) -> Optional[str]:
    '''Procesar comandos relacionados con alertas'''

    if intent == "system:alerts" or "alerta" in command.lower():
        response = await handle_alert_voice_commands(
            command,
            self.alert_manager
        )

        if response:
            return response

    return None

# O en tu clase principal:

class VoiceAssistant:
    def __init__(self, ...):
        # ... otros componentes ...
        self.alert_manager = alert_manager
        self.alert_scheduler = alert_scheduler

    async def process_command(self, command: str):
        # ... procesamiento normal ...

        # Antes de procesar domótica, verificar comandos de alertas
        alert_response = await handle_alert_voice_commands(
            command,
            self.alert_manager
        )

        if alert_response:
            await self.tts.speak(alert_response)
            return

        # ... continuar con procesamiento normal ...
"""


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main_integration_example())
