"""
Priority Request Queue
Cola priorizada para peticiones al LLM.

Prioridades:
- P0 (CRITICAL): Seguridad, alarmas - interrumpe todo
- P1 (HIGH): Domotica - rapido, no debe esperar
- P2 (MEDIUM): Rutinas, consultas simples
- P3 (LOW): Conversacion, razonamiento profundo

Comportamiento:
- Peticiones de mayor prioridad se procesan primero
- Domotica (P1) puede interrumpir conversacion (P3) en curso
- Peticiones del mismo usuario se cancelan automaticamente
- Timeout configurable por prioridad

Ejemplo:
    queue = PriorityRequestQueue()

    # Agregar peticiones
    req1 = queue.enqueue("user_a", "Explica relatividad", Priority.LOW)
    req2 = queue.enqueue("user_b", "Prende luz", Priority.HIGH)

    # Procesar (req2 sale primero por prioridad)
    next_req = queue.dequeue()  # -> req2
"""

import asyncio
import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from src.core.logging import get_logger
from src.orchestrator.cancellation import (
    CancellationToken,
    CancellationReason,
    CancellationManager
)

logger = get_logger(__name__)


class Priority(IntEnum):
    """
    Niveles de prioridad para peticiones.

    Menor numero = mayor prioridad.
    """
    CRITICAL = 0   # Seguridad, alarmas, emergencias
    HIGH = 1       # Domotica rapida (luces, clima)
    MEDIUM = 2     # Rutinas, consultas sobre el hogar
    LOW = 3        # Conversacion general, razonamiento


class RequestStatus(IntEnum):
    """Estado de una peticion"""
    PENDING = 0      # En cola esperando
    PROCESSING = 1   # Siendo procesada
    COMPLETED = 2    # Completada exitosamente
    CANCELLED = 3    # Cancelada
    FAILED = 4       # Fallo
    TIMEOUT = 5      # Timeout


@dataclass(order=True)
class Request:
    """
    Peticion en la cola.

    Ordenada por (priority, timestamp) para heapq.
    """
    # Campos para ordenamiento (priority first, then timestamp)
    priority: Priority = field(compare=True)
    timestamp: float = field(compare=True, default_factory=time.time)

    # Campos de datos (no afectan ordenamiento)
    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = field(compare=False, default="")
    user_name: str = field(compare=False, default="")
    zone_id: str = field(compare=False, default=None)
    text: str = field(compare=False, default="")
    intent: str = field(compare=False, default=None)

    # Estado y control
    status: RequestStatus = field(compare=False, default=RequestStatus.PENDING)
    cancellation_token: CancellationToken = field(compare=False, default=None)

    # Resultados
    result: Any = field(compare=False, default=None)
    error: str = field(compare=False, default=None)
    completed_at: float = field(compare=False, default=None)

    # Callbacks
    on_complete: Callable = field(compare=False, default=None, repr=False)
    on_cancel: Callable = field(compare=False, default=None, repr=False)

    def __post_init__(self):
        if self.cancellation_token is None:
            self.cancellation_token = CancellationToken(
                request_id=self.request_id,
                user_id=self.user_id
            )

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation_token.is_cancelled

    @property
    def wait_time(self) -> float:
        """Tiempo en cola (segundos)"""
        if self.status == RequestStatus.PENDING:
            return time.time() - self.timestamp
        return 0

    @property
    def processing_time(self) -> float | None:
        """Tiempo de procesamiento (segundos)"""
        if self.completed_at:
            return self.completed_at - self.timestamp
        return None

    def cancel(self, reason: CancellationReason = CancellationReason.USER_NEW_REQUEST):
        """Cancelar esta peticion"""
        self.cancellation_token.cancel(reason)
        self.status = RequestStatus.CANCELLED
        if self.on_cancel:
            try:
                self.on_cancel(self)
            except Exception as e:
                logger.error(f"Error en on_cancel callback: {e}")

    def complete(self, result: Any):
        """Marcar como completada"""
        self.result = result
        self.status = RequestStatus.COMPLETED
        self.completed_at = time.time()
        if self.on_complete:
            try:
                self.on_complete(self)
            except Exception as e:
                logger.error(f"Error en on_complete callback: {e}")

    def fail(self, error: str):
        """Marcar como fallida"""
        self.error = error
        self.status = RequestStatus.FAILED
        self.completed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "zone_id": self.zone_id,
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "priority": self.priority.name,
            "status": self.status.name,
            "wait_time": self.wait_time,
            "processing_time": self.processing_time,
            "is_cancelled": self.is_cancelled
        }


class PriorityRequestQueue:
    """
    Cola priorizada thread-safe para peticiones al LLM.

    Caracteristicas:
    - Ordenamiento por prioridad y timestamp
    - Cancelacion automatica de peticiones anteriores del mismo usuario
    - Timeouts configurables por prioridad
    - Estadisticas de rendimiento

    Ejemplo:
        queue = PriorityRequestQueue()

        # Encolar peticion
        request = queue.enqueue(
            user_id="user_123",
            text="¿Qué es Python?",
            priority=Priority.LOW
        )

        # Esperar peticion en worker
        next_request = await queue.dequeue_async()

        # Procesar y completar
        result = process(next_request)
        next_request.complete(result)
    """

    # Timeouts por prioridad (segundos)
    DEFAULT_TIMEOUTS = {
        Priority.CRITICAL: 60,
        Priority.HIGH: 30,
        Priority.MEDIUM: 60,
        Priority.LOW: 120
    }

    def __init__(
        self,
        auto_cancel_previous: bool = True,
        timeouts: dict[Priority, float] = None,
        max_queue_size: int = 100
    ):
        """
        Args:
            auto_cancel_previous: Cancelar peticion anterior del mismo usuario
            timeouts: Timeouts por prioridad
            max_queue_size: Tamaño maximo de la cola
        """
        self.auto_cancel = auto_cancel_previous
        self.timeouts = timeouts or self.DEFAULT_TIMEOUTS
        self.max_queue_size = max_queue_size

        # Cola priorizada (heap)
        self._queue: list[Request] = []
        self._lock = threading.Lock()

        # Event para notificar nuevas peticiones
        self._not_empty = threading.Condition(self._lock)
        self._async_event = asyncio.Event()

        # Tracking por usuario
        self._user_requests: dict[str, Request] = {}  # user_id -> request activo

        # Peticion actualmente en proceso
        self._current: Request | None = None

        # Estadisticas
        self._stats = {
            "total_enqueued": 0,
            "total_processed": 0,
            "total_cancelled": 0,
            "total_timeout": 0,
            "by_priority": {p: 0 for p in Priority}
        }

    def enqueue(
        self,
        user_id: str,
        text: str,
        priority: Priority = Priority.MEDIUM,
        user_name: str = None,
        zone_id: str = None,
        intent: str = None,
        on_complete: Callable = None,
        on_cancel: Callable = None
    ) -> Request:
        """
        Agregar peticion a la cola.

        Args:
            user_id: ID del usuario
            text: Texto de la peticion
            priority: Prioridad
            user_name: Nombre del usuario
            zone_id: Zona de origen
            intent: Intent detectado
            on_complete: Callback al completar
            on_cancel: Callback al cancelar

        Returns:
            Request creado
        """
        request = Request(
            priority=priority,
            user_id=user_id,
            user_name=user_name or user_id,
            zone_id=zone_id,
            text=text,
            intent=intent,
            on_complete=on_complete,
            on_cancel=on_cancel
        )

        with self._lock:
            # Verificar tamaño maximo
            if len(self._queue) >= self.max_queue_size:
                # Rechazar peticiones de baja prioridad si la cola esta llena
                if priority == Priority.LOW:
                    raise QueueFullError("Cola llena, intente más tarde")

            # Cancelar peticion anterior del mismo usuario
            if self.auto_cancel and user_id in self._user_requests:
                old_request = self._user_requests[user_id]
                if old_request.status == RequestStatus.PENDING:
                    old_request.cancel(CancellationReason.USER_NEW_REQUEST)
                    self._stats["total_cancelled"] += 1
                    logger.debug(f"Peticion anterior cancelada: {old_request.request_id}")

            # Agregar a la cola
            heapq.heappush(self._queue, request)
            self._user_requests[user_id] = request

            # Estadisticas
            self._stats["total_enqueued"] += 1
            self._stats["by_priority"][priority] += 1

            # Notificar
            self._not_empty.notify()

        # Notificar async
        try:
            self._async_event.set()
        except RuntimeError:
            pass  # No hay event loop

        logger.debug(
            f"Enqueued: {request.request_id} "
            f"(user={user_id}, priority={priority.name}, queue_size={len(self._queue)})"
        )

        return request

    def dequeue(self, timeout: float = None) -> Request | None:
        """
        Obtener siguiente peticion (bloqueante).

        Args:
            timeout: Timeout en segundos (None = esperar indefinidamente)

        Returns:
            Request o None si timeout
        """
        with self._not_empty:
            # Esperar si la cola esta vacia
            while not self._queue:
                if not self._not_empty.wait(timeout):
                    return None  # Timeout

            # Obtener peticion de mayor prioridad
            request = self._get_next_valid()

            if request:
                request.status = RequestStatus.PROCESSING
                self._current = request
                self._stats["total_processed"] += 1

            return request

    async def dequeue_async(self, timeout: float = None) -> Request | None:
        """
        Obtener siguiente peticion (async).

        Args:
            timeout: Timeout en segundos

        Returns:
            Request o None si timeout
        """
        start = time.time()

        while True:
            # Intentar obtener de la cola
            with self._lock:
                request = self._get_next_valid()
                if request:
                    request.status = RequestStatus.PROCESSING
                    self._current = request
                    self._stats["total_processed"] += 1
                    return request

            # Verificar timeout
            if timeout:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed
            else:
                remaining = 1.0

            # Esperar notificacion o timeout
            self._async_event.clear()
            try:
                await asyncio.wait_for(
                    self._async_event.wait(),
                    timeout=min(remaining, 1.0)
                )
            except asyncio.TimeoutError:
                if timeout and (time.time() - start) >= timeout:
                    return None

    def _get_next_valid(self) -> Request | None:
        """
        Obtener siguiente peticion valida (no cancelada).

        Debe llamarse con _lock adquirido.
        """
        while self._queue:
            request = heapq.heappop(self._queue)

            # Saltar peticiones canceladas
            if request.is_cancelled:
                continue

            # Verificar timeout
            timeout = self.timeouts.get(request.priority, 60)
            if request.wait_time > timeout:
                request.status = RequestStatus.TIMEOUT
                self._stats["total_timeout"] += 1
                logger.warning(f"Request timeout: {request.request_id}")
                continue

            return request

        return None

    def interrupt_for_priority(self, min_priority: Priority) -> bool:
        """
        Verificar si hay peticion de mayor prioridad que deberia interrumpir.

        Args:
            min_priority: Prioridad minima para interrumpir

        Returns:
            True si hay peticion que deberia interrumpir
        """
        with self._lock:
            for request in self._queue:
                if not request.is_cancelled and request.priority < min_priority:
                    return True
            return False

    def cancel_user_request(self, user_id: str) -> bool:
        """
        Cancelar peticion de un usuario.

        Returns:
            True si se cancelo alguna peticion
        """
        with self._lock:
            request = self._user_requests.get(user_id)
            if request and not request.is_cancelled:
                request.cancel(CancellationReason.USER_EXPLICIT)
                self._stats["total_cancelled"] += 1
                return True
            return False

    def cancel_all(self):
        """Cancelar todas las peticiones"""
        with self._lock:
            for request in self._queue:
                if not request.is_cancelled:
                    request.cancel(CancellationReason.SYSTEM_SHUTDOWN)
            self._queue.clear()

    def get_current(self) -> Request | None:
        """Obtener peticion actualmente en proceso"""
        return self._current

    def clear_current(self):
        """Limpiar peticion actual (llamar al terminar procesamiento)"""
        self._current = None

    def get_queue_status(self) -> list[dict]:
        """Obtener estado de la cola"""
        with self._lock:
            return [r.to_dict() for r in sorted(self._queue) if not r.is_cancelled]

    def get_position(self, request_id: str) -> int | None:
        """Obtener posicion en la cola de una peticion"""
        with self._lock:
            sorted_queue = sorted(
                [r for r in self._queue if not r.is_cancelled]
            )
            for i, req in enumerate(sorted_queue):
                if req.request_id == request_id:
                    return i + 1
            return None

    def get_estimated_wait(self, request_id: str, avg_processing_time: float = 10.0) -> float | None:
        """
        Estimar tiempo de espera para una peticion.

        Args:
            request_id: ID de la peticion
            avg_processing_time: Tiempo promedio de procesamiento (segundos)

        Returns:
            Tiempo estimado en segundos o None si no encontrada
        """
        position = self.get_position(request_id)
        if position is None:
            return None
        return position * avg_processing_time

    def get_stats(self) -> dict:
        """Obtener estadisticas de la cola"""
        with self._lock:
            pending = len([r for r in self._queue if not r.is_cancelled])

            return {
                "queue_size": pending,
                "max_queue_size": self.max_queue_size,
                "current_processing": self._current.request_id if self._current else None,
                **self._stats
            }

    def __len__(self) -> int:
        with self._lock:
            return len([r for r in self._queue if not r.is_cancelled])


class QueueFullError(Exception):
    """Error cuando la cola esta llena"""
    pass


class RequestProcessor:
    """
    Procesador de peticiones de la cola.

    Worker que consume peticiones y las procesa con el LLM.

    Ejemplo:
        processor = RequestProcessor(queue, llm, context_manager)
        await processor.start()  # Inicia procesamiento en background
    """

    def __init__(
        self,
        queue: PriorityRequestQueue,
        llm,
        context_manager,
        tts=None,
        check_interrupt_interval: float = 0.5
    ):
        """
        Args:
            queue: Cola de peticiones
            llm: LLM para generar respuestas
            context_manager: Gestor de contextos
            tts: Motor TTS (opcional)
            check_interrupt_interval: Intervalo para verificar interrupciones
        """
        self.queue = queue
        self.llm = llm
        self.context_manager = context_manager
        self.tts = tts
        self.check_interval = check_interrupt_interval

        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Iniciar procesamiento en background"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("RequestProcessor iniciado")

    async def stop(self):
        """Detener procesamiento"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RequestProcessor detenido")

    async def _process_loop(self):
        """Loop principal de procesamiento"""
        while self._running:
            try:
                # Obtener siguiente peticion
                request = await self.queue.dequeue_async(timeout=1.0)

                if request is None:
                    continue

                # Procesar peticion
                await self._process_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en proceso de peticion: {e}")

    async def _process_request(self, request: Request):
        """Procesar una peticion individual"""
        logger.info(f"Procesando: {request.request_id} ({request.user_name})")

        try:
            # Construir prompt con contexto
            prompt = self.context_manager.build_prompt(
                request.user_id,
                request.text
            )

            # Generar respuesta con verificacion de cancelacion
            response = await self._generate_with_cancellation(
                prompt,
                request.cancellation_token,
                request.priority
            )

            if request.is_cancelled:
                logger.info(f"Peticion cancelada durante generacion: {request.request_id}")
                return

            # Agregar al contexto
            self.context_manager.add_turn(
                request.user_id, "user", request.text
            )
            self.context_manager.add_turn(
                request.user_id, "assistant", response
            )

            # Completar peticion
            request.complete(response)

            logger.info(
                f"Completado: {request.request_id} "
                f"({request.processing_time:.1f}s)"
            )

        except Exception as e:
            logger.error(f"Error procesando {request.request_id}: {e}")
            request.fail(str(e))

        finally:
            self.queue.clear_current()

    async def _generate_with_cancellation(
        self,
        prompt: str,
        token: CancellationToken,
        priority: Priority
    ) -> str:
        """
        Generar respuesta verificando cancelacion e interrupciones.
        """
        # Si el LLM soporta streaming
        if hasattr(self.llm, 'generate_stream'):
            chunks = []
            for chunk in self.llm.generate_stream(prompt):
                # Verificar cancelacion
                if token.is_cancelled:
                    raise GenerationInterruptedError("Generacion cancelada")

                # Verificar si hay peticion de mayor prioridad
                if self.queue.interrupt_for_priority(priority):
                    token.cancel(CancellationReason.HIGHER_PRIORITY)
                    raise GenerationInterruptedError("Interrumpido por mayor prioridad")

                chunks.append(chunk.get("token", ""))

                # Pequena pausa para permitir interrupciones
                await asyncio.sleep(0)

            return "".join(chunks)

        else:
            # Generacion no-streaming
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(prompt)
            )
            return response


class GenerationInterruptedError(Exception):
    """Error cuando una generacion es interrumpida"""
    pass
