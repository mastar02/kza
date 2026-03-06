"""
Request Dispatcher
Enruta peticiones al path correcto (Fast Path vs Slow Path).

Fast Path (paralelo):
- Domotica via vector search
- Consultas simples via Router 7B
- Rutinas predefinidas
- Respuestas < 1 segundo

Slow Path (serializado):
- Razonamiento profundo con LLM 32B/70B
- Conversaciones multi-turno
- Peticiones complejas
- Cola priorizada

Ejemplo:
    dispatcher = RequestDispatcher(
        chroma_sync=chroma,
        router=router_7b,
        llm=llm_32b,
        context_manager=context_manager,
        priority_queue=queue
    )

    # Procesar peticion - automaticamente va al path correcto
    result = await dispatcher.dispatch(
        user_id="user_123",
        text="Prende la luz",
        zone_id="living"
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from src.core.logging import get_logger, LogContext, generate_request_id
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.priority_queue import (
    Priority,
    Request,
    PriorityRequestQueue
)
from src.orchestrator.cancellation import CancellationToken

logger = get_logger(__name__)


class PathType(StrEnum):
    """Tipo de path para procesar la peticion"""
    FAST_DOMOTICS = "fast_domotics"       # Vector search + HA
    FAST_ROUTINE = "fast_routine"          # Rutinas predefinidas
    FAST_ROUTER = "fast_router"            # Router 7B para respuestas simples
    FAST_MUSIC = "fast_music"              # Spotify - búsqueda directa
    SLOW_MUSIC = "slow_music"              # Spotify - interpretación con LLM
    SLOW_LLM = "slow_llm"                  # LLM grande para razonamiento
    SYNC = "sync"                          # Comandos de sincronizacion
    ENROLLMENT = "enrollment"              # Registro de usuarios
    FEEDBACK = "feedback"                  # Feedback sobre respuestas
    FAST_LIST = "fast_list"                # List CRUD
    FAST_REMINDER = "fast_reminder"        # Reminder CRUD


@dataclass
class DispatchResult:
    """Resultado del dispatch"""
    path: PathType
    priority: Priority
    success: bool
    response: str
    intent: str = None
    action: dict = None
    timings: dict = field(default_factory=dict)
    error: str = None
    was_queued: bool = False
    queue_position: int = None
    user_id: str = None
    zone_id: str = None

    def to_dict(self) -> dict:
        return {
            "path": self.path.value,
            "priority": self.priority.name,
            "success": self.success,
            "response": self.response,
            "intent": self.intent,
            "timings": self.timings,
            "was_queued": self.was_queued,
            "queue_position": self.queue_position
        }


class RequestDispatcher:
    """
    Dispatcher que enruta peticiones al path optimo.

    Arquitectura:
                        ┌─────────────────┐
                        │    DISPATCH     │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌───────────┐    ┌───────────────┐   ┌───────────┐
        │ FAST PATH │    │  SLOW PATH    │   │  SPECIAL  │
        │ (paralelo)│    │ (serializado) │   │  COMMANDS │
        ├───────────┤    ├───────────────┤   ├───────────┤
        │• Domotica │    │• LLM Queue    │   │• Sync     │
        │• Router 7B│    │• Context/user │   │• Enroll   │
        │• Rutinas  │    │• Buffering    │   │• Feedback │
        └───────────┘    └───────────────┘   └───────────┘
    """

    # Palabras clave para detectar intents rapidos
    DOMOTICS_KEYWORDS = [
        "prende", "enciende", "apaga", "sube", "baja",
        "abre", "cierra", "pon", "cambia", "activa", "desactiva"
    ]

    SYNC_KEYWORDS = [
        "sincroniza", "actualiza", "refresca", "sync"
    ]

    ENROLLMENT_KEYWORDS = [
        "agregar persona", "agregar usuario", "nueva persona",
        "registrar", "add user"
    ]

    CANCEL_KEYWORDS = [
        "cancela", "olvida", "para", "detente", "cancel", "stop"
    ]

    LIST_KEYWORDS = [
        "lista de", "agrega", "agregale", "quita", "quitale",
        "qué hay en la lista", "vacía la lista", "vaciala",
        "crea una lista", "borra la lista", "lista compartida",
    ]

    REMINDER_KEYWORDS = [
        "recuérdame", "recuerdame", "recordatorio",
        "avísame", "avisame", "qué tengo pendiente",
        "que tengo pendiente", "qué recordatorios",
        "que recordatorios", "todos los lunes",
        "todos los días", "todos los dias",
        "cada día", "cada dia", "cada lunes",
        "cada martes", "de lunes a viernes",
        "cancela el recordatorio",
    ]

    # Música - Fast path (búsqueda directa)
    MUSIC_DIRECT_KEYWORDS = [
        "pon música de", "música de", "canciones de", "reproduce",
        "playlist", "pausa", "siguiente canción", "anterior",
        "qué suena", "qué está sonando"
    ]

    # Música - Slow path (requiere interpretación)
    MUSIC_CONTEXT_KEYWORDS = [
        "música para", "algo para", "algo tranquilo", "algo alegre",
        "música relajante", "música mientras", "ambiente"
    ]

    def __init__(
        self,
        chroma_sync,
        ha_client,
        routine_manager,
        router=None,
        llm=None,
        tts=None,
        context_manager: ContextManager = None,
        priority_queue: PriorityRequestQueue = None,
        buffered_streamer=None,
        music_dispatcher=None,
        list_manager=None,
        reminder_manager=None,
        vector_threshold: float = 0.65,
        use_router_for_simple: bool = True
    ):
        """
        Args:
            chroma_sync: Sincronizador de ChromaDB
            ha_client: Cliente de Home Assistant
            routine_manager: Gestor de rutinas
            router: Router 7B (opcional, para respuestas simples)
            llm: LLM grande para razonamiento
            tts: Motor TTS
            context_manager: Gestor de contextos por usuario
            priority_queue: Cola priorizada para slow path
            buffered_streamer: Streamer con buffering para TTS
            music_dispatcher: Dispatcher de música/Spotify
            vector_threshold: Umbral de similitud para vector search
            use_router_for_simple: Usar router 7B para preguntas simples
        """
        self.chroma = chroma_sync
        self.ha = ha_client
        self.routines = routine_manager
        self.router = router
        self.llm = llm
        self.tts = tts
        self.context_manager = context_manager or ContextManager()
        self.queue = priority_queue or PriorityRequestQueue()
        self.streamer = buffered_streamer
        self.music = music_dispatcher
        self.vector_threshold = vector_threshold
        self.use_router = use_router_for_simple
        self.list_manager = list_manager
        self.reminder_manager = reminder_manager

        # Estadisticas
        self._stats = {
            "total_requests": 0,
            "fast_path": 0,
            "slow_path": 0,
            "music_requests": 0,
            "by_path": {p: 0 for p in PathType}
        }

        # Callback para respuestas del slow path
        self._slow_path_callbacks: dict[str, Callable] = {}

    async def dispatch(
        self,
        user_id: str,
        text: str,
        user_name: str = None,
        zone_id: str = None,
        permission_level: int = 3,
        on_response: Callable[[DispatchResult], None] = None,
        timeout: float = 60.0
    ) -> DispatchResult:
        """
        Procesar una peticion, enrutando al path correcto.

        Args:
            user_id: ID del usuario
            text: Texto de la peticion
            user_name: Nombre del usuario
            zone_id: Zona de origen
            permission_level: Nivel de permisos
            on_response: Callback cuando hay respuesta (para slow path)
            timeout: Timeout maximo

        Returns:
            DispatchResult con la respuesta
        """
        start_time = time.perf_counter()
        self._stats["total_requests"] += 1

        # Normalizar texto
        text = text.strip()
        text_lower = text.lower()

        # Obtener/crear contexto del usuario
        ctx = self.context_manager.get_or_create(
            user_id=user_id,
            user_name=user_name,
            zone_id=zone_id,
            permission_level=permission_level
        )

        # 1. Detectar comandos especiales
        special_result = await self._check_special_commands(text_lower, user_id, ctx)
        if special_result:
            special_result.timings["total"] = (time.perf_counter() - start_time) * 1000
            return special_result

        # 2. Detectar intent y prioridad
        path, priority = self._classify_request(text_lower)

        # 3. Enrutar al path correcto
        if path == PathType.FAST_MUSIC:
            # Música - búsqueda directa
            result = await self._fast_music_path(text, user_id)
            self._stats["fast_path"] += 1
            self._stats["music_requests"] += 1

        elif path == PathType.SLOW_MUSIC:
            # Música - requiere interpretación con LLM
            result = await self._slow_music_path(text, user_id)
            self._stats["slow_path"] += 1
            self._stats["music_requests"] += 1

        elif path in [PathType.FAST_DOMOTICS, PathType.FAST_ROUTINE, PathType.FAST_ROUTER]:
            # Fast path - procesar inmediatamente
            result = await self._fast_path(
                text=text,
                path=path,
                user_id=user_id,
                zone_id=zone_id,
                permission_level=permission_level
            )
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_LIST:
            result = await self._fast_list_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_REMINDER:
            result = await self._fast_reminder_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1

        else:
            # Slow path - encolar para LLM
            result = await self._slow_path(
                text=text,
                user_id=user_id,
                user_name=user_name or ctx.user_name,
                zone_id=zone_id,
                priority=priority,
                on_response=on_response,
                timeout=timeout
            )
            self._stats["slow_path"] += 1

        # Actualizar estadisticas
        self._stats["by_path"][path] += 1

        # Agregar timings
        result.timings["total"] = (time.perf_counter() - start_time) * 1000
        result.user_id = user_id
        result.zone_id = zone_id

        return result

    def _classify_request(self, text_lower: str) -> tuple[PathType, Priority]:
        """
        Clasificar peticion para determinar path y prioridad.

        Returns:
            (PathType, Priority)
        """
        # Detectar música - contexto complejo (slow path)
        if self.music:
            for keyword in self.MUSIC_CONTEXT_KEYWORDS:
                if keyword in text_lower:
                    return PathType.SLOW_MUSIC, Priority.MEDIUM

            # Detectar música - búsqueda directa (fast path)
            for keyword in self.MUSIC_DIRECT_KEYWORDS:
                if keyword in text_lower:
                    return PathType.FAST_MUSIC, Priority.HIGH

        # Detect lists
        for keyword in self.LIST_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_LIST, Priority.HIGH

        # Detect reminders
        for keyword in self.REMINDER_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_REMINDER, Priority.HIGH

        # Detectar domotica por keywords
        for keyword in self.DOMOTICS_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_DOMOTICS, Priority.HIGH

        # Detectar rutinas
        if any(word in text_lower for word in ["rutina", "automatiza", "cuando"]):
            return PathType.FAST_ROUTINE, Priority.MEDIUM

        # Si tenemos router, preguntas simples van por fast path
        if self.router and self.use_router:
            # Preguntas muy simples
            if self._is_simple_query(text_lower):
                return PathType.FAST_ROUTER, Priority.MEDIUM

        # Todo lo demas va al slow path
        return PathType.SLOW_LLM, Priority.LOW

    def _is_simple_query(self, text_lower: str) -> bool:
        """Detectar si es una pregunta simple que el router puede manejar"""
        simple_patterns = [
            "que hora es",
            "que dia es",
            "como esta el clima",
            "que temperatura",
            "hola", "buenos dias", "buenas tardes", "buenas noches"
        ]
        return any(pattern in text_lower for pattern in simple_patterns)

    async def _check_special_commands(
        self,
        text_lower: str,
        user_id: str,
        ctx
    ) -> DispatchResult | None:
        """Verificar comandos especiales"""

        # Comando de cancelacion
        for keyword in self.CANCEL_KEYWORDS:
            if keyword in text_lower:
                cancelled = self.queue.cancel_user_request(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTER,
                    priority=Priority.HIGH,
                    success=True,
                    response="Cancelado" if cancelled else "No hay nada que cancelar",
                    intent="cancel"
                )

        # Comando de sincronizacion
        for keyword in self.SYNC_KEYWORDS:
            if keyword in text_lower:
                return DispatchResult(
                    path=PathType.SYNC,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Sincronizando comandos...",
                    intent="sync"
                )

        # Comando de enrollment
        for keyword in self.ENROLLMENT_KEYWORDS:
            if keyword in text_lower:
                return DispatchResult(
                    path=PathType.ENROLLMENT,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Iniciando registro de usuario...",
                    intent="enrollment"
                )

        # Confirmacion pendiente
        if ctx.pending_confirmation:
            if any(word in text_lower for word in ["si", "confirma", "acepto", "ok"]):
                # Procesar confirmacion
                confirmation = ctx.pending_confirmation
                self.context_manager.clear_pending_confirmation(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTINE,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Confirmado",
                    intent="confirmation",
                    action=confirmation
                )
            elif any(word in text_lower for word in ["no", "cancela", "rechaza"]):
                self.context_manager.clear_pending_confirmation(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTINE,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Cancelado",
                    intent="rejection"
                )

        return None

    async def _fast_path(
        self,
        text: str,
        path: PathType,
        user_id: str,
        zone_id: str,
        permission_level: int
    ) -> DispatchResult:
        """
        Procesar peticion por fast path (paralelo, sin cola).
        """
        timings = {}

        if path == PathType.FAST_DOMOTICS:
            # Buscar comando en vector DB
            t0 = time.perf_counter()
            command = self.chroma.search_command(text, self.vector_threshold)
            timings["vector_search"] = (time.perf_counter() - t0) * 1000

            if command:
                # Ejecutar en Home Assistant
                t1 = time.perf_counter()
                success = self.ha.call_service(
                    command["domain"],
                    command["service"],
                    command["entity_id"],
                    command.get("data")
                )
                timings["home_assistant"] = (time.perf_counter() - t1) * 1000

                return DispatchResult(
                    path=path,
                    priority=Priority.HIGH,
                    success=success,
                    response=command["description"] if success else "No pude hacerlo",
                    intent="domotics",
                    action=command,
                    timings=timings
                )

            # No encontrado en vector DB, intentar con router
            if self.router:
                path = PathType.FAST_ROUTER

        if path == PathType.FAST_ROUTER and self.router:
            # Usar router para respuesta rapida
            t0 = time.perf_counter()
            try:
                response = self.router.generate([text], max_tokens=128)[0]
                timings["router"] = (time.perf_counter() - t0) * 1000

                return DispatchResult(
                    path=path,
                    priority=Priority.MEDIUM,
                    success=True,
                    response=response.strip(),
                    intent="simple_query",
                    timings=timings
                )
            except Exception as e:
                logger.warning(f"Router fallo, pasando a slow path: {e}")

        if path == PathType.FAST_ROUTINE:
            # Procesar rutina
            t0 = time.perf_counter()
            routine_result = await self.routines.handle(text)
            timings["routine"] = (time.perf_counter() - t0) * 1000

            if routine_result["handled"]:
                return DispatchResult(
                    path=path,
                    priority=Priority.MEDIUM,
                    success=routine_result["success"],
                    response=routine_result["response"],
                    intent="routine",
                    timings=timings
                )

        # Fallback a slow path
        return await self._slow_path(
            text=text,
            user_id=user_id,
            user_name=None,
            zone_id=zone_id,
            priority=Priority.LOW,
            on_response=None,
            timeout=60.0
        )

    async def _slow_path(
        self,
        text: str,
        user_id: str,
        user_name: str,
        zone_id: str,
        priority: Priority,
        on_response: Callable,
        timeout: float
    ) -> DispatchResult:
        """
        Procesar peticion por slow path (cola priorizada).
        """
        timings = {}

        # Crear evento para esperar respuesta
        response_event = asyncio.Event()
        result_holder = {"result": None}

        def on_complete(request: Request):
            result_holder["result"] = request.result
            try:
                response_event.set()
            except RuntimeError:
                pass

        def on_cancel(request: Request):
            result_holder["result"] = "Cancelado"
            try:
                response_event.set()
            except RuntimeError:
                pass

        # Encolar peticion
        t0 = time.perf_counter()
        request = self.queue.enqueue(
            user_id=user_id,
            text=text,
            priority=priority,
            user_name=user_name,
            zone_id=zone_id,
            on_complete=on_complete,
            on_cancel=on_cancel
        )
        timings["queue"] = (time.perf_counter() - t0) * 1000

        position = self.queue.get_position(request.request_id)

        # Si hay callback, notificar que esta en cola
        if on_response and position and position > 1:
            on_response(DispatchResult(
                path=PathType.SLOW_LLM,
                priority=priority,
                success=True,
                response=f"Un momento, hay {position - 1} peticion(es) antes",
                intent="queued",
                was_queued=True,
                queue_position=position,
                timings=timings
            ))

        # Esperar respuesta o timeout
        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            request.cancel()
            return DispatchResult(
                path=PathType.SLOW_LLM,
                priority=priority,
                success=False,
                response="Lo siento, tarde demasiado. Intenta de nuevo.",
                intent="timeout",
                error="timeout",
                was_queued=True,
                timings=timings
            )

        response = result_holder["result"]
        timings["llm"] = request.processing_time * 1000 if request.processing_time else 0

        return DispatchResult(
            path=PathType.SLOW_LLM,
            priority=priority,
            success=request.status.name == "COMPLETED",
            response=response or "Sin respuesta",
            intent="conversation",
            was_queued=True,
            queue_position=position,
            timings=timings
        )

    async def _fast_music_path(self, text: str, user_id: str) -> DispatchResult:
        """
        Procesar comando de música por fast path (búsqueda directa).
        """
        timings = {}

        if not self.music:
            return DispatchResult(
                path=PathType.FAST_MUSIC,
                priority=Priority.HIGH,
                success=False,
                response="Spotify no está configurado",
                intent="music_error",
                timings=timings
            )

        t0 = time.perf_counter()

        # Obtener preferencias del usuario si existen
        user_prefs = None
        ctx = self.context_manager.get(user_id)
        if ctx and hasattr(ctx, 'music_preferences'):
            user_prefs = ctx.music_preferences

        # Procesar comando de música
        result = await self.music.process(text, user_preferences=user_prefs)
        timings["spotify"] = (time.perf_counter() - t0) * 1000

        return DispatchResult(
            path=PathType.FAST_MUSIC,
            priority=Priority.HIGH,
            success=result.success,
            response=result.response,
            intent=f"music_{result.intent.value}",
            action=result.details,
            timings=timings
        )

    async def _slow_music_path(self, text: str, user_id: str) -> DispatchResult:
        """
        Procesar comando de música por slow path (requiere LLM para interpretar).
        Ejemplo: "Pon música para una cena romántica a la luz de las velas"
        """
        timings = {}

        if not self.music:
            return DispatchResult(
                path=PathType.SLOW_MUSIC,
                priority=Priority.MEDIUM,
                success=False,
                response="Spotify no está configurado",
                intent="music_error",
                timings=timings
            )

        t0 = time.perf_counter()

        # Obtener preferencias del usuario
        user_prefs = None
        ctx = self.context_manager.get(user_id)
        if ctx and hasattr(ctx, 'music_preferences'):
            user_prefs = ctx.music_preferences

        # Procesar - el MusicDispatcher usará LLM internamente si es necesario
        result = await self.music.process(text, user_preferences=user_prefs)
        timings["spotify_with_llm"] = (time.perf_counter() - t0) * 1000

        return DispatchResult(
            path=PathType.SLOW_MUSIC,
            priority=Priority.MEDIUM,
            success=result.success,
            response=result.response,
            intent=f"music_{result.intent.value}",
            action={"interpreted_mood": result.details.get("interpreted_as")},
            timings=timings
        )

    async def _fast_list_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle list commands via ListManager."""
        if not self.list_manager:
            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=False, response="Listas no configuradas",
            )

        text_lower = text.lower()
        try:
            if any(w in text_lower for w in ["qué hay", "que hay", "dime la lista", "lee la lista"]):
                list_name = self._extract_list_name(text_lower)
                items = await self.list_manager.get_items(user_id, list_name)
                if not items:
                    response = "La lista está vacía"
                else:
                    item_texts = ", ".join(i.text for i in items)
                    response = f"En la lista tienes: {item_texts}"
            elif any(w in text_lower for w in ["vacía", "vacia", "limpia"]):
                list_name = self._extract_list_name(text_lower)
                await self.list_manager.clear_list(user_id, list_name)
                response = "Listo, vacié la lista"
            elif any(w in text_lower for w in ["borra la lista", "elimina la lista"]):
                list_name = self._extract_list_name(text_lower)
                if list_name and await self.list_manager.delete_list(user_id, list_name):
                    response = f"Borré la lista {list_name}"
                else:
                    response = "No encontré esa lista"
            elif any(w in text_lower for w in ["crea una lista", "nueva lista"]):
                shared = "compartida" in text_lower
                list_name = self._extract_list_name(text_lower)
                if list_name:
                    await self.list_manager.create_list(user_id, list_name, shared=shared)
                    response = f"Creé la lista {list_name}"
                else:
                    response = "No entendí el nombre de la lista"
            elif any(w in text_lower for w in ["quita", "quitale", "elimina", "tacha"]):
                item_text = self._extract_item_text(text_lower, removing=True)
                list_name = self._extract_list_name(text_lower)
                if item_text and await self.list_manager.remove_item(user_id, item_text, list_name):
                    response = f"Quité {item_text}"
                else:
                    response = "No encontré ese artículo en la lista"
            elif any(w in text_lower for w in ["agrega", "agregale", "añade", "pon"]):
                item_text = self._extract_item_text(text_lower, removing=False)
                list_name = self._extract_list_name(text_lower)
                if item_text:
                    await self.list_manager.add_item(user_id, item_text, list_name)
                    response = f"Agregué {item_text}"
                else:
                    response = "No entendí qué agregar"
            else:
                # Fallback: list all lists
                lists = await self.list_manager.get_all_lists(user_id)
                if lists:
                    names = ", ".join(lst.name for lst in lists)
                    response = f"Tienes estas listas: {names}"
                else:
                    response = "No tienes listas creadas"

            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=True, response=response,
            )
        except Exception as e:
            logger.error("List command error: %s", e)
            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=False, response="Hubo un error con la lista",
            )

    async def _fast_reminder_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle reminder commands via ReminderManager."""
        if not self.reminder_manager:
            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=False, response="Recordatorios no configurados",
            )

        text_lower = text.lower()
        try:
            if any(w in text_lower for w in ["qué recordatorios", "que recordatorios", "mis recordatorios"]):
                active = await self.reminder_manager.get_active(user_id)
                if not active:
                    response = "No tienes recordatorios activos"
                else:
                    lines = [self.reminder_manager.format_for_voice(r) for r in active[:5]]
                    response = "Tus recordatorios: " + ". ".join(lines)
            elif any(w in text_lower for w in ["qué tengo pendiente", "que tengo pendiente"]):
                today = await self.reminder_manager.get_today(user_id)
                if not today:
                    response = "No tienes nada pendiente hoy"
                else:
                    lines = [self.reminder_manager.format_for_voice(r) for r in today]
                    response = "Pendiente hoy: " + ". ".join(lines)
            elif "cancela" in text_lower and "recordatorio" in text_lower:
                import re
                match = re.search(r'recordatorio\s+(?:de\s+)?(.+)', text_lower)
                search_text = match.group(1).strip() if match else text_lower
                if await self.reminder_manager.cancel_by_text(user_id, search_text):
                    response = "Recordatorio cancelado"
                else:
                    response = "No encontré ese recordatorio"
            else:
                response = "Entendido, pero necesito el Router para interpretar la hora. Usa la API por ahora."

            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=True, response=response,
            )
        except Exception as e:
            logger.error("Reminder command error: %s", e)
            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=False, response="Hubo un error con el recordatorio",
            )

    def _extract_list_name(self, text: str) -> str | None:
        """Extract list name from text like 'la lista de compras' or 'la lista del hogar'."""
        import re
        # "la lista de X" / "la lista del X" / "a la lista X"
        match = re.search(r'(?:la lista (?:de(?:l)?|)\s+)(\w[\w\s]*?)(?:\s*$|[,.])', text)
        if match:
            return match.group(1).strip()
        # "lista compartida X"
        match = re.search(r'lista compartida\s+(?:de(?:l)?\s+)?(\w[\w\s]*?)(?:\s*$|[,.])', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_item_text(self, text: str, removing: bool = False) -> str | None:
        """Extract item text from commands like 'agrega leche a la lista'."""
        import re
        if removing:
            # "quita X de la lista"
            match = re.search(r'(?:quita|quitale|elimina|tacha)\s+(?:el |la |los |las )?(.+?)(?:\s+de la lista|\s*$)', text)
        else:
            # "agrega X a la lista" or just "agrega X"
            match = re.search(r'(?:agrega|agregale|añade|pon)\s+(.+?)(?:\s+a la lista|\s+en la lista|\s*$)', text)
        if match:
            item = match.group(1).strip()
            # Remove trailing list name reference
            item = re.sub(r'\s+(?:de|a|en)\s+la\s+lista.*$', '', item)
            return item if item else None
        return None

    async def dispatch_batch(
        self,
        requests: list[dict]
    ) -> list[DispatchResult]:
        """
        Procesar multiples peticiones en paralelo.

        Args:
            requests: Lista de {"user_id", "text", ...}

        Returns:
            Lista de resultados
        """
        tasks = [
            self.dispatch(**req)
            for req in requests
        ]
        return await asyncio.gather(*tasks)

    def get_queue_status(self) -> dict:
        """Obtener estado de la cola"""
        return self.queue.get_stats()

    def get_stats(self) -> dict:
        """Obtener estadisticas del dispatcher"""
        return {
            **self._stats,
            "queue": self.queue.get_stats(),
            "contexts": self.context_manager.get_stats()
        }

    def notify_user_waiting(
        self,
        user_id: str,
        zone_id: str,
        other_user_name: str
    ):
        """
        Notificar a un usuario que debe esperar.

        Usado cuando llega una peticion y hay otra en proceso.
        """
        message = f"Un momento, estoy respondiendo a {other_user_name}"
        if self.tts:
            # TODO: Enviar a zona especifica
            self.tts.speak(message)
        return message


class MultiUserOrchestrator:
    """
    Orquestador completo para multiples usuarios.

    Coordina todos los componentes:
    - Context Manager
    - Priority Queue
    - Request Dispatcher
    - Request Processor

    Ejemplo:
        orchestrator = MultiUserOrchestrator(
            chroma_sync=chroma,
            ha_client=ha,
            routine_manager=routines,
            router=router_7b,
            llm=llm_32b,
            tts=tts
        )

        await orchestrator.start()

        # Procesar peticion
        result = await orchestrator.process(
            user_id="juan",
            text="Explícame la relatividad",
            zone_id="living"
        )
    """

    def __init__(
        self,
        chroma_sync,
        ha_client,
        routine_manager,
        router=None,
        llm=None,
        tts=None,
        speaker_identifier=None,
        user_manager=None,
        music_dispatcher=None,
        list_manager=None,
        reminder_manager=None,
        max_context_history: int = 10,
        context_timeout: float = 300,
        auto_cancel_previous: bool = True
    ):
        # Componentes principales
        self.chroma = chroma_sync
        self.ha = ha_client
        self.routines = routine_manager
        self.router = router
        self.llm = llm
        self.tts = tts
        self.speaker_id = speaker_identifier
        self.user_manager = user_manager
        self.music = music_dispatcher

        # Inicializar subsistemas
        self._context_manager = ContextManager(
            max_history=max_context_history,
            inactive_timeout=context_timeout
        )

        self._queue = PriorityRequestQueue(
            auto_cancel_previous=auto_cancel_previous
        )

        self._cancel_manager = self._queue  # Para acceso desde VoicePipeline

        self.dispatcher = RequestDispatcher(
            chroma_sync=chroma_sync,
            ha_client=ha_client,
            routine_manager=routine_manager,
            router=router,
            llm=llm,
            tts=tts,
            context_manager=self._context_manager,
            priority_queue=self._queue,
            music_dispatcher=music_dispatcher,
            list_manager=list_manager,
            reminder_manager=reminder_manager,
        )

        self._running = False
        self._processor_task = None

    async def start(self):
        """Iniciar el orquestador"""
        if self._running:
            return

        self._running = True

        # Iniciar limpieza de contextos
        self.context_manager.start_cleanup_thread()

        # Iniciar procesador de cola
        self._processor_task = asyncio.create_task(self._process_queue())

        logger.info("MultiUserOrchestrator iniciado")

    async def stop(self):
        """Detener el orquestador"""
        self._running = False

        # Cancelar peticiones pendientes
        self.queue.cancel_all()

        # Detener cleanup
        self.context_manager.stop_cleanup_thread()

        # Detener procesador
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("MultiUserOrchestrator detenido")

    async def process(
        self,
        user_id: str,
        text: str,
        audio: any = None,
        zone_id: str = None,
        on_response: Callable = None
    ) -> DispatchResult:
        """
        Procesar una peticion de usuario.

        Args:
            user_id: ID del usuario (o None para identificar por voz)
            text: Texto transcrito
            audio: Audio original (para speaker ID si user_id es None)
            zone_id: Zona de origen
            on_response: Callback para respuestas

        Returns:
            DispatchResult
        """
        # Identificar usuario si no se proporciono
        if user_id is None and audio is not None and self.speaker_id:
            user_id, user_name = await self._identify_speaker(audio)
        else:
            user_name = None
            if self.user_manager and user_id:
                user = self.user_manager.get_user(user_id)
                if user:
                    user_name = user.name

        user_id = user_id or "unknown"
        permission_level = 0

        if self.user_manager and user_id != "unknown":
            user = self.user_manager.get_user(user_id)
            if user:
                permission_level = user.permission_level.value

        # Dispatch
        return await self.dispatcher.dispatch(
            user_id=user_id,
            text=text,
            user_name=user_name,
            zone_id=zone_id,
            permission_level=permission_level,
            on_response=on_response
        )

    async def _identify_speaker(self, audio) -> tuple[str, str]:
        """Identificar usuario por voz"""
        if not self.speaker_id or not self.user_manager:
            return None, None

        embeddings = self.user_manager.get_all_embeddings()
        if not embeddings:
            return None, None

        match = self.speaker_id.identify(audio, embeddings)
        if match.is_known and match.user_id:
            user = self.user_manager.get_user(match.user_id)
            if user:
                return user.user_id, user.name

        return None, None

    async def _process_queue(self):
        """Procesar peticiones de la cola"""
        while self._running:
            try:
                request = await self.queue.dequeue_async(timeout=1.0)
                if request is None:
                    continue

                # Notificar a otros usuarios en espera
                await self._notify_waiting_users(request)

                # Procesar con LLM
                await self._process_llm_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error en proceso de cola", data={"error": str(e)})

    async def _notify_waiting_users(self, current_request: Request):
        """Notificar a usuarios en espera que deben esperar"""
        queue_status = self.queue.get_queue_status()

        for queued in queue_status:
            if queued["user_id"] != current_request.user_id:
                # Notificar que debe esperar
                ctx = self.context_manager.get(queued["user_id"])
                if ctx and ctx.zone_id and self.tts:
                    message = f"Un momento, estoy con {current_request.user_name}"
                    # TODO: Enviar a zona especifica
                    logger.debug(f"Notificando a {queued['user_id']}: {message}")

    async def _process_llm_request(self, request: Request):
        """Procesar peticion con el LLM"""
        try:
            # Construir prompt con contexto
            prompt = self.context_manager.build_prompt(
                request.user_id,
                request.text
            )

            # Generar respuesta
            if hasattr(self.llm, 'generate_stream'):
                # Streaming con verificacion de cancelacion
                response_parts = []
                for chunk in self.llm.generate_stream(prompt):
                    if request.is_cancelled:
                        logger.info(
                            "Request cancelado",
                            request_id=request.request_id,
                            user_id=request.user_id
                        )
                        return

                    # Verificar interrupciones de mayor prioridad
                    if self.queue.interrupt_for_priority(request.priority):
                        request.cancel()
                        logger.info(
                            "Request interrumpido por mayor prioridad",
                            request_id=request.request_id,
                            user_id=request.user_id
                        )
                        return

                    response_parts.append(chunk.get("token", ""))

                response = "".join(response_parts)
            else:
                response = self.llm.generate(prompt)

            # Agregar al contexto
            self.context_manager.add_turn(
                request.user_id, "user", request.text
            )
            self.context_manager.add_turn(
                request.user_id, "assistant", response
            )

            # Completar
            request.complete(response)

        except Exception as e:
            logger.error(
                "Error procesando request",
                request_id=request.request_id,
                user_id=request.user_id,
                data={"error": str(e)}
            )
            request.fail(str(e))

        finally:
            self.queue.clear_current()

    def get_stats(self) -> dict:
        """Obtener estadisticas completas"""
        return self.dispatcher.get_stats()
