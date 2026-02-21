"""
Circuit Breaker para Home Assistant
Protege contra fallos en cascada cuando HA está caído o lento
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal, permitiendo requests
    OPEN = "open"          # Bloqueando requests (HA caído)
    HALF_OPEN = "half_open"  # Probando si HA se recuperó


@dataclass
class CircuitStats:
    """Estadísticas del circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Bloqueados por circuit open
    last_failure_time: float = 0
    last_success_time: float = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class HACircuitBreaker:
    """
    Circuit Breaker para llamadas a Home Assistant.

    Estados:
    - CLOSED: Funcionando normal, permite todas las llamadas
    - OPEN: HA caído, rechaza llamadas inmediatamente
    - HALF_OPEN: Probando recuperación, permite una llamada de prueba

    Configuración:
    - failure_threshold: Fallos consecutivos para abrir el circuito
    - recovery_timeout: Segundos antes de probar recuperación
    - success_threshold: Éxitos consecutivos para cerrar el circuito
    """

    def __init__(
        self,
        failure_threshold: int = 10,    # Más tolerante antes de abrir (velocidad > precaución)
        recovery_timeout: float = 10.0,  # Recuperación rápida (era 30s)
        success_threshold: int = 1,      # Un éxito basta para cerrar
        call_timeout: float = 3.0        # Timeout corto para no bloquear
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.call_timeout = call_timeout

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._lock = asyncio.Lock()

        # Callbacks opcionales
        self._on_state_change: Optional[Callable] = None
        self._on_failure: Optional[Callable] = None

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitStats:
        return self._stats

    @property
    def is_available(self) -> bool:
        """¿Está disponible para hacer llamadas?"""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.HALF_OPEN:
            return True  # Permite una llamada de prueba
        if self._state == CircuitState.OPEN:
            # Verificar si pasó el timeout de recuperación
            elapsed = time.time() - self._last_state_change
            if elapsed >= self.recovery_timeout:
                return True  # Transicionar a half-open
        return False

    async def call(
        self,
        func: Callable,
        *args,
        fallback: Any = None,
        **kwargs
    ) -> tuple[bool, Any]:
        """
        Ejecutar función protegida por circuit breaker.

        Returns:
            (success, result_or_fallback)
        """
        async with self._lock:
            self._stats.total_calls += 1

            # Verificar estado del circuito
            if not self.is_available:
                self._stats.rejected_calls += 1
                logger.warning(f"Circuit OPEN - llamada rechazada ({self._stats.rejected_calls} rechazadas)")
                return False, fallback

            # Si estamos en OPEN y pasó el timeout, transicionar a HALF_OPEN
            if self._state == CircuitState.OPEN:
                self._transition_to(CircuitState.HALF_OPEN)

        # Ejecutar la llamada (fuera del lock)
        try:
            # Aplicar timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.call_timeout
                )
            else:
                # Función síncrona - ejecutar en thread pool con timeout
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=self.call_timeout
                )

            await self._record_success()
            return True, result

        except asyncio.TimeoutError:
            logger.warning(f"HA timeout después de {self.call_timeout}s")
            await self._record_failure("timeout")
            return False, fallback

        except Exception as e:
            logger.warning(f"HA error: {e}")
            await self._record_failure(str(e))
            return False, fallback

    async def _record_success(self):
        """Registrar llamada exitosa"""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0

            # Si estamos en HALF_OPEN y suficientes éxitos, cerrar circuito
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info("Circuit CLOSED - HA recuperado")

    async def _record_failure(self, reason: str):
        """Registrar llamada fallida"""
        async with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            if self._on_failure:
                self._on_failure(reason)

            # Si suficientes fallos consecutivos, abrir circuito
            if self._stats.consecutive_failures >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"Circuit OPEN - {self._stats.consecutive_failures} fallos consecutivos. "
                        f"Recuperación en {self.recovery_timeout}s"
                    )

            # Si estamos en HALF_OPEN y falla, volver a OPEN
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning("Circuit OPEN - prueba de recuperación falló")

    def _transition_to(self, new_state: CircuitState):
        """Transicionar a nuevo estado"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

        logger.debug(f"Circuit breaker: {old_state.value} -> {new_state.value}")

    def reset(self):
        """Resetear el circuit breaker"""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        logger.info("Circuit breaker reseteado")

    def force_open(self):
        """Forzar apertura del circuito (para mantenimiento)"""
        self._transition_to(CircuitState.OPEN)

    def force_close(self):
        """Forzar cierre del circuito"""
        self._transition_to(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0

    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Registrar callback para cambios de estado"""
        self._on_state_change = callback

    def on_failure(self, callback: Callable[[str], None]):
        """Registrar callback para fallos"""
        self._on_failure = callback

    def get_status(self) -> dict:
        """Obtener estado completo"""
        return {
            "state": self._state.value,
            "is_available": self.is_available,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful": self._stats.successful_calls,
                "failed": self._stats.failed_calls,
                "rejected": self._stats.rejected_calls,
                "consecutive_failures": self._stats.consecutive_failures,
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "call_timeout": self.call_timeout,
            },
            "time_in_state": time.time() - self._last_state_change
        }


# Circuit breaker global para Home Assistant
_ha_circuit_breaker: Optional[HACircuitBreaker] = None


def get_ha_circuit_breaker() -> HACircuitBreaker:
    """Obtener instancia global del circuit breaker"""
    global _ha_circuit_breaker
    if _ha_circuit_breaker is None:
        _ha_circuit_breaker = HACircuitBreaker()
    return _ha_circuit_breaker


def reset_ha_circuit_breaker():
    """Resetear el circuit breaker global"""
    global _ha_circuit_breaker
    if _ha_circuit_breaker:
        _ha_circuit_breaker.reset()
