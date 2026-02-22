"""
Cancellation System
Permite cancelar peticiones en curso de forma segura.

Util para:
- Cancelar generacion LLM cuando el usuario hace otra pregunta
- Cancelar cuando se detecta un comando de mayor prioridad
- Timeout de peticiones largas

Ejemplo:
    # Crear token de cancelacion
    token = CancellationToken()

    # En el generador del LLM
    for chunk in llm.generate_stream(prompt):
        if token.is_cancelled:
            break
        yield chunk

    # Desde otro thread, cancelar
    token.cancel(reason="Usuario hizo nueva pregunta")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

logger = logging.getLogger(__name__)


class CancellationReason(StrEnum):
    """Razones de cancelacion"""
    USER_NEW_REQUEST = "user_new_request"      # Usuario hizo otra peticion
    HIGHER_PRIORITY = "higher_priority"        # Llego peticion de mayor prioridad
    TIMEOUT = "timeout"                        # Timeout alcanzado
    USER_EXPLICIT = "user_explicit"            # Usuario dijo "cancela" o similar
    SYSTEM_SHUTDOWN = "system_shutdown"        # Sistema apagandose
    ERROR = "error"                            # Error en procesamiento


@dataclass
class CancellationToken:
    """
    Token para cancelar operaciones de forma cooperativa.

    Thread-safe. El productor verifica periodicamente is_cancelled
    y detiene la operacion si es True.

    Atributos:
        is_cancelled: True si se solicito cancelacion
        reason: Razon de la cancelacion
        cancelled_at: Timestamp de cancelacion
    """
    _cancelled: bool = field(default=False, repr=False)
    _reason: CancellationReason | None = field(default=None, repr=False)
    _message: str | None = field(default=None, repr=False)
    _cancelled_at: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _callbacks: list[Callable] = field(default_factory=list, repr=False)

    # Metadata
    request_id: str | None = None
    user_id: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_cancelled(self) -> bool:
        """Verificar si esta cancelado (thread-safe)"""
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> CancellationReason | None:
        """Obtener razon de cancelacion"""
        with self._lock:
            return self._reason

    @property
    def message(self) -> str | None:
        """Obtener mensaje de cancelacion"""
        with self._lock:
            return self._message

    @property
    def cancelled_at(self) -> float | None:
        """Timestamp de cancelacion"""
        with self._lock:
            return self._cancelled_at

    def cancel(
        self,
        reason: CancellationReason = CancellationReason.USER_NEW_REQUEST,
        message: str = None
    ):
        """
        Solicitar cancelacion.

        Args:
            reason: Razon de la cancelacion
            message: Mensaje adicional
        """
        callbacks_to_run = []

        with self._lock:
            if self._cancelled:
                return  # Ya cancelado

            self._cancelled = True
            self._reason = reason
            self._message = message
            self._cancelled_at = time.time()
            callbacks_to_run = self._callbacks.copy()

        # Ejecutar callbacks fuera del lock
        for callback in callbacks_to_run:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error en callback de cancelacion: {e}")

        logger.debug(
            f"Cancelado: request={self.request_id}, "
            f"reason={reason.value}, msg={message}"
        )

    def on_cancel(self, callback: Callable[["CancellationToken"], None]):
        """
        Registrar callback para cuando se cancele.

        El callback recibe el token como argumento.
        """
        with self._lock:
            if self._cancelled:
                # Ya cancelado, ejecutar inmediatamente
                callback(self)
            else:
                self._callbacks.append(callback)

    def raise_if_cancelled(self):
        """Lanzar excepcion si esta cancelado"""
        if self.is_cancelled:
            raise CancelledException(self.reason, self.message)

    def reset(self):
        """Resetear el token (usar con cuidado)"""
        with self._lock:
            self._cancelled = False
            self._reason = None
            self._message = None
            self._cancelled_at = None
            self._callbacks = []

    def get_elapsed_since_cancel(self) -> float | None:
        """Tiempo transcurrido desde la cancelacion"""
        with self._lock:
            if self._cancelled_at:
                return time.time() - self._cancelled_at
            return None


class CancelledException(Exception):
    """Excepcion lanzada cuando una operacion es cancelada"""

    def __init__(
        self,
        reason: CancellationReason = None,
        message: str = None
    ):
        self.reason = reason
        self.message = message
        super().__init__(f"Operacion cancelada: {reason.value if reason else 'unknown'}")


class CancellationScope:
    """
    Contexto para manejar cancelacion con timeout automatico.

    Ejemplo:
        with CancellationScope(timeout=30.0) as token:
            for chunk in llm.generate_stream(prompt):
                token.raise_if_cancelled()
                yield chunk
    """

    def __init__(
        self,
        timeout: float = None,
        request_id: str = None,
        user_id: str = None
    ):
        """
        Args:
            timeout: Timeout en segundos (None = sin timeout)
            request_id: ID de la peticion
            user_id: ID del usuario
        """
        self.timeout = timeout
        self.token = CancellationToken(
            request_id=request_id,
            user_id=user_id
        )
        self._timer: threading.Timer | None = None

    def __enter__(self) -> CancellationToken:
        if self.timeout:
            self._timer = threading.Timer(
                self.timeout,
                self._on_timeout
            )
            self._timer.daemon = True
            self._timer.start()
        return self.token

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        return False  # No suprimir excepciones

    def _on_timeout(self):
        """Callback cuando se alcanza el timeout"""
        self.token.cancel(
            CancellationReason.TIMEOUT,
            f"Timeout de {self.timeout}s alcanzado"
        )

    def cancel(self, reason: CancellationReason = None, message: str = None):
        """Cancelar manualmente"""
        self.token.cancel(reason or CancellationReason.USER_EXPLICIT, message)


class CancellationManager:
    """
    Gestor de tokens de cancelacion por usuario.

    Permite cancelar automaticamente peticiones anteriores
    cuando llega una nueva del mismo usuario.

    Ejemplo:
        manager = CancellationManager()

        # Nueva peticion de usuario A
        token_a1 = manager.create_token("user_a", "req_1")

        # Otra peticion de A - cancela la anterior automaticamente
        token_a2 = manager.create_token("user_a", "req_2")
        # token_a1.is_cancelled == True
    """

    def __init__(self, auto_cancel_previous: bool = True):
        """
        Args:
            auto_cancel_previous: Cancelar peticion anterior del mismo usuario
        """
        self.auto_cancel = auto_cancel_previous
        self._tokens: dict[str, CancellationToken] = {}  # user_id -> token
        self._lock = threading.Lock()

    def create_token(
        self,
        user_id: str,
        request_id: str = None
    ) -> CancellationToken:
        """
        Crear token para una peticion.

        Si auto_cancel=True, cancela peticion anterior del mismo usuario.
        """
        with self._lock:
            # Cancelar token anterior si existe
            if self.auto_cancel and user_id in self._tokens:
                old_token = self._tokens[user_id]
                if not old_token.is_cancelled:
                    old_token.cancel(
                        CancellationReason.USER_NEW_REQUEST,
                        "Nueva peticion del usuario"
                    )

            # Crear nuevo token
            token = CancellationToken(
                request_id=request_id,
                user_id=user_id
            )
            self._tokens[user_id] = token

            return token

    def get_token(self, user_id: str) -> CancellationToken | None:
        """Obtener token activo de un usuario"""
        with self._lock:
            return self._tokens.get(user_id)

    def cancel_user(
        self,
        user_id: str,
        reason: CancellationReason = CancellationReason.USER_EXPLICIT
    ):
        """Cancelar peticion de un usuario"""
        with self._lock:
            token = self._tokens.get(user_id)
            if token and not token.is_cancelled:
                token.cancel(reason)

    def cancel_all(self, reason: CancellationReason = CancellationReason.SYSTEM_SHUTDOWN):
        """Cancelar todas las peticiones"""
        with self._lock:
            for token in self._tokens.values():
                if not token.is_cancelled:
                    token.cancel(reason)

    def cleanup_cancelled(self):
        """Limpiar tokens ya cancelados"""
        with self._lock:
            self._tokens = {
                uid: token for uid, token in self._tokens.items()
                if not token.is_cancelled
            }

    def get_stats(self) -> dict:
        """Estadisticas del manager"""
        with self._lock:
            active = sum(1 for t in self._tokens.values() if not t.is_cancelled)
            cancelled = sum(1 for t in self._tokens.values() if t.is_cancelled)
            return {
                "total_tokens": len(self._tokens),
                "active": active,
                "cancelled": cancelled
            }
