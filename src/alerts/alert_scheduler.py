"""
Alert Scheduler Module
Sistema de scheduler para ejecutar verificaciones periódicas de alertas.

Proporciona:
- Ejecución asincrónica periódica de diferentes tipos de alertas
- Configuración flexible de intervalos
- Manejo robusto de errores
- Integración con AlertManager

Uso:
    from src.alerts import AlertScheduler, SecurityAlerts, PatternAlerts, DeviceAlerts
    from src.alerts import AlertManager

    alert_manager = AlertManager()
    security = SecurityAlerts(alert_manager)
    patterns = PatternAlerts(alert_manager)
    devices = DeviceAlerts(alert_manager)

    scheduler = AlertScheduler(alert_manager)
    scheduler.register_security_checks(security)
    scheduler.register_pattern_checks(patterns)
    scheduler.register_device_checks(devices)

    # Iniciar scheduler (típicamente en main)
    scheduler_task = asyncio.create_task(scheduler.start())

    # Más tarde, detener
    await scheduler.stop()
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from src.core.logging import get_logger
from .alert_manager import AlertManager

# Type hints para tipos específicos (evitar importaciones circulares)
if TYPE_CHECKING:
    from .security_alerts import SecurityAlerts
    from .pattern_alerts import PatternAlerts
    from .device_alerts import DeviceAlerts

logger = get_logger(__name__)


class CheckType(Enum):
    """Tipos de verificaciones programadas"""
    SECURITY = "security"
    PATTERN = "pattern"
    DEVICE = "device"


@dataclass
class CheckConfig:
    """Configuración para una verificación periódica"""
    check_type: CheckType
    interval_seconds: float
    enabled: bool = True
    description: str = ""


class AlertScheduler:
    """
    Scheduler para ejecutar verificaciones periódicas de alertas.

    Ejecuta diferentes tipos de verificaciones en intervalos configurables:
    - Security: cada 60 segundos
    - Pattern: cada 300 segundos (5 minutos)
    - Device: cada 600 segundos (10 minutos)
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        security_interval: float = 60.0,
        pattern_interval: float = 300.0,
        device_interval: float = 600.0,
    ):
        """
        Inicializar AlertScheduler.

        Args:
            alert_manager: Instancia de AlertManager
            security_interval: Intervalo para verificaciones de seguridad (segundos)
            pattern_interval: Intervalo para verificaciones de patrones (segundos)
            device_interval: Intervalo para verificaciones de dispositivos (segundos)
        """
        self.alert_manager = alert_manager
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Configuraciones de chequeo
        self._check_configs: Dict[CheckType, CheckConfig] = {
            CheckType.SECURITY: CheckConfig(
                check_type=CheckType.SECURITY,
                interval_seconds=security_interval,
                description="Security checks (doors, motion)",
            ),
            CheckType.PATTERN: CheckConfig(
                check_type=CheckType.PATTERN,
                interval_seconds=pattern_interval,
                description="Pattern anomaly checks",
            ),
            CheckType.DEVICE: CheckConfig(
                check_type=CheckType.DEVICE,
                interval_seconds=device_interval,
                description="Device status checks",
            ),
        }

        # Handlers registrados para cada tipo de verificación
        self._security_handlers: List[Callable[[], asyncio.coroutine]] = []
        self._pattern_handlers: List[Callable[[], asyncio.coroutine]] = []
        self._device_handlers: List[Callable[[], asyncio.coroutine]] = []

        logger.info(
            f"AlertScheduler initialized "
            f"(security={security_interval}s, pattern={pattern_interval}s, device={device_interval}s)"
        )

    def register_security_handler(
        self,
        handler: Callable[[], asyncio.coroutine],
    ) -> None:
        """
        Registrar handler para verificaciones de seguridad.

        Args:
            handler: Función async que realiza la verificación de seguridad
        """
        self._security_handlers.append(handler)
        logger.debug(f"Security handler registered (total: {len(self._security_handlers)})")

    def register_pattern_handler(
        self,
        handler: Callable[[], asyncio.coroutine],
    ) -> None:
        """
        Registrar handler para verificaciones de patrones.

        Args:
            handler: Función async que realiza la verificación de patrones
        """
        self._pattern_handlers.append(handler)
        logger.debug(f"Pattern handler registered (total: {len(self._pattern_handlers)})")

    def register_device_handler(
        self,
        handler: Callable[[], asyncio.coroutine],
    ) -> None:
        """
        Registrar handler para verificaciones de dispositivos.

        Args:
            handler: Función async que realiza la verificación de dispositivos
        """
        self._device_handlers.append(handler)
        logger.debug(f"Device handler registered (total: {len(self._device_handlers)})")

    def register_security_checks(self, security_alerts: 'SecurityAlerts') -> None:
        """
        Registrar métodos estándar de verificación de seguridad.

        Args:
            security_alerts: Instancia de SecurityAlerts
        """
        # Los handlers específicos se registrarían aquí si hay datos disponibles
        # Por ahora, este es un punto de extensión para futuras integraciones
        logger.debug("Security checks registered (ready for external data)")

    def register_pattern_checks(self, pattern_alerts: 'PatternAlerts') -> None:
        """
        Registrar métodos estándar de verificación de patrones.

        Args:
            pattern_alerts: Instancia de PatternAlerts
        """
        logger.debug("Pattern checks registered (ready for external data)")

    def register_device_checks(self, device_alerts: 'DeviceAlerts') -> None:
        """
        Registrar métodos estándar de verificación de dispositivos.

        Args:
            device_alerts: Instancia de DeviceAlerts
        """
        logger.debug("Device checks registered (ready for external data)")

    async def start(self) -> None:
        """
        Iniciar el scheduler de alertas.

        Este método ejecuta indefinidamente hasta que se llame a stop().
        """
        if self._running:
            logger.warning("AlertScheduler is already running")
            return

        self._running = True
        logger.info("AlertScheduler starting...")

        try:
            # Crear tareas para cada tipo de verificación
            security_task = asyncio.create_task(
                self._run_periodic_check(
                    CheckType.SECURITY,
                    self._check_configs[CheckType.SECURITY].interval_seconds,
                )
            )
            pattern_task = asyncio.create_task(
                self._run_periodic_check(
                    CheckType.PATTERN,
                    self._check_configs[CheckType.PATTERN].interval_seconds,
                )
            )
            device_task = asyncio.create_task(
                self._run_periodic_check(
                    CheckType.DEVICE,
                    self._check_configs[CheckType.DEVICE].interval_seconds,
                )
            )

            self._tasks = [security_task, pattern_task, device_task]

            # Esperar a que todas las tareas se completen
            await asyncio.gather(*self._tasks)

        except asyncio.CancelledError:
            logger.info("AlertScheduler cancelled")
        except Exception as e:
            logger.error(f"Error in AlertScheduler: {e}")
        finally:
            self._running = False
            logger.info("AlertScheduler stopped")

    async def stop(self) -> None:
        """
        Detener el scheduler de alertas.

        Cancela todas las tareas pendientes.
        """
        if not self._running:
            logger.warning("AlertScheduler is not running")
            return

        logger.info("Stopping AlertScheduler...")
        self._running = False

        # Cancelar todas las tareas
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Esperar a que se cancelen
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Error stopping tasks: {e}")

    async def _run_periodic_check(
        self,
        check_type: CheckType,
        interval_seconds: float,
    ) -> None:
        """
        Ejecutar verificaciones periódicas de un tipo específico.

        Args:
            check_type: Tipo de verificación a ejecutar
            interval_seconds: Intervalo entre verificaciones
        """
        config = self._check_configs[check_type]
        logger.info(
            f"Starting {check_type.value} check loop (interval={interval_seconds}s)"
        )

        # Primera ejecución después de un pequeño delay
        await asyncio.sleep(1)

        while self._running:
            try:
                if not config.enabled:
                    await asyncio.sleep(interval_seconds)
                    continue

                # Ejecutar handlers según tipo
                if check_type == CheckType.SECURITY:
                    await self._execute_security_checks()
                elif check_type == CheckType.PATTERN:
                    await self._execute_pattern_checks()
                elif check_type == CheckType.DEVICE:
                    await self._execute_device_checks()

                # Esperar hasta el próximo intervalo
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in {check_type.value} check: {e}")
                # Continuar en caso de error, esperar antes de reintentar
                await asyncio.sleep(interval_seconds)

    async def _execute_security_checks(self) -> None:
        """Ejecutar todos los handlers de seguridad registrados"""
        for handler in self._security_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in security check handler: {e}")

    async def _execute_pattern_checks(self) -> None:
        """Ejecutar todos los handlers de patrones registrados"""
        for handler in self._pattern_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in pattern check handler: {e}")

    async def _execute_device_checks(self) -> None:
        """Ejecutar todos los handlers de dispositivos registrados"""
        for handler in self._device_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in device check handler: {e}")

    def set_check_enabled(self, check_type: CheckType, enabled: bool) -> None:
        """
        Habilitar o deshabilitar un tipo de verificación.

        Args:
            check_type: Tipo de verificación
            enabled: True para habilitar, False para deshabilitar
        """
        if check_type in self._check_configs:
            self._check_configs[check_type].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"{check_type.value} checks {status}")

    def set_check_interval(self, check_type: CheckType, interval_seconds: float) -> None:
        """
        Cambiar el intervalo de un tipo de verificación.

        Args:
            check_type: Tipo de verificación
            interval_seconds: Nuevo intervalo en segundos
        """
        if check_type in self._check_configs:
            self._check_configs[check_type].interval_seconds = interval_seconds
            logger.info(f"{check_type.value} check interval set to {interval_seconds}s")

    def get_status(self) -> Dict:
        """
        Obtener estado actual del scheduler.

        Returns:
            Dict con estado de cada tipo de verificación
        """
        return {
            "running": self._running,
            "checks": {
                check_type.value: {
                    "enabled": config.enabled,
                    "interval_seconds": config.interval_seconds,
                    "description": config.description,
                }
                for check_type, config in self._check_configs.items()
            },
            "registered_handlers": {
                "security": len(self._security_handlers),
                "pattern": len(self._pattern_handlers),
                "device": len(self._device_handlers),
            },
        }
