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
import math
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, TYPE_CHECKING

from src.alerts.alert_manager import AlertManager

# Type hints para tipos específicos (evitar importaciones circulares)
if TYPE_CHECKING:
    from src.alerts.security_alerts import SecurityAlerts
    from src.alerts.pattern_alerts import PatternAlerts
    from src.alerts.device_alerts import DeviceAlerts

logger = logging.getLogger(__name__)


class CheckType(StrEnum):
    """Tipos de verificaciones programadas"""
    SECURITY = "security"
    PATTERN = "pattern"
    DEVICE = "device"
    HA = "ha"


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
        self._tasks: list[asyncio.Task] = []

        # Configuraciones de chequeo
        self._check_configs: dict[CheckType, CheckConfig] = {
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
        self._security_handlers: list[Callable[[], Awaitable[None]]] = []
        self._pattern_handlers: list[Callable[[], Awaitable[None]]] = []
        self._device_handlers: list[Callable[[], Awaitable[None]]] = []
        self._ha_reconcile: Callable[[], Awaitable[None]] | None = None

        logger.info(
            f"AlertScheduler initialized "
            f"(security={security_interval}s, pattern={pattern_interval}s, device={device_interval}s)"
        )

    def register_security_handler(
        self,
        handler: Callable[[], Awaitable[None]],
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
        handler: Callable[[], Awaitable[None]],
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
        handler: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Registrar handler para verificaciones de dispositivos.

        Args:
            handler: Función async que realiza la verificación de dispositivos
        """
        self._device_handlers.append(handler)
        logger.debug(f"Device handler registered (total: {len(self._device_handlers)})")

    def register_ha_source(self, reconcile: Callable[[], Awaitable[None]], interval: float) -> None:
        """Register one bounded HA snapshot check for all selected conditions."""
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("alerts.ha.reconcile_interval_seconds must be finite and positive")
        self._ha_reconcile = reconcile
        self._check_configs[CheckType.HA] = CheckConfig(
            CheckType.HA, interval, description="Home Assistant state reconciliation",
        )

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

        tasks = []
        try:
            tasks = [
                asyncio.create_task(self._run_periodic_check(check_type, config.interval_seconds),
                                    name=f"alert-check-{check_type.value}")
                for check_type, config in self._check_configs.items()
            ]
            self._tasks = tasks

            # Esperar a que todas las tareas se completen
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("AlertScheduler cancelled")
        except Exception as e:
            logger.error(f"Error in AlertScheduler: {e}")
        finally:
            if self._tasks is tasks:
                self._running = False
                self._tasks = []
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
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

        # HA establishes its baseline at startup. Keep the legacy check delay.
        if check_type is not CheckType.HA:
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
                elif check_type == CheckType.HA and self._ha_reconcile is not None:
                    await self._ha_reconcile()

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

    def get_status(self) -> dict:
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
                "ha": int(self._ha_reconcile is not None),
            },
            "pattern_alerts": "connected" if self._pattern_handlers else "not_connected",
        }
