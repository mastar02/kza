"""
Context Manager
Maneja el contexto de conversacion por usuario.

Cada usuario tiene su propio historial de conversacion, permitiendo
conversaciones multi-turno independientes y paralelas.

Ejemplo:
    manager = ContextManager(max_history=10)

    # Usuario A pregunta
    ctx_a = manager.get_or_create("user_a", "Juan", "zone_living")
    prompt_a = manager.build_prompt("user_a", "¿Qué es Python?")

    # Usuario B pregunta (contexto separado)
    ctx_b = manager.get_or_create("user_b", "María", "zone_kitchen")
    prompt_b = manager.build_prompt("user_b", "Prende la luz")

    # Agregar respuestas
    manager.add_turn("user_a", "assistant", "Python es un lenguaje...")
    manager.add_turn("user_b", "assistant", "Luz encendida")

    # Siguiente turno de A mantiene contexto
    prompt_a2 = manager.build_prompt("user_a", "¿Y para qué sirve?")
    # Incluye el historial previo de A, no de B

Plan #2 OpenClaw: si se inyecta un Compactor, los turnos viejos se compactan
en background al alcanzar `compaction_threshold` (default 6) — el Compactor
llama al 30B vía kza-llm-ik :8200 y devuelve un summary que reemplaza el
prefijo. Si además se inyecta un ContextPersister, el contexto sobrevive a
reinicios — `cleanup_inactive_async` snapshotea a `data/contexts/<user_id>.json`
al expirar, y `get_or_create` hidrata desde disco al volver el usuario.
Conversation history NO se restaura: los turnos literales mueren con la sesión;
sólo el summary + preserved_ids cruzan sesiones.

Spec: docs/superpowers/specs/2026-04-28-openclaw-context-compaction-design.md
"""

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from src.core.logging import get_logger
from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemorySource, sources_current

logger = get_logger(__name__)


MAX_SUMMARY_CHARS = 4000  # cap on compacted_summary to prevent prompt blowout


def _truncate_summary_head(text: str, cap: int | None = None) -> str:
    """If text exceeds cap, drop the head and prefix with elision marker.

    The tail is preserved (more recent compactions are more relevant).
    `cap` defaults to the module-level MAX_SUMMARY_CHARS at call time
    (so tests can monkeypatch it).
    """
    if cap is None:
        cap = MAX_SUMMARY_CHARS
    if len(text) <= cap:
        return text
    keep = cap - 32  # leave room for the marker
    if keep <= 0:
        return text[-cap:]  # cap is too tight; just take tail
    return "[…anteriores omitidos] " + text[-keep:]


@dataclass
class ConversationTurn:
    """Un turno en la conversacion"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: str | None = None  # "domotics", "conversation", "routine", etc.
    entities: list[str] = field(default_factory=list)  # Entidades mencionadas
    sources: tuple[MemorySource, ...] = ()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "entities": self.entities,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        data = dict(data)
        data["sources"] = tuple(MemorySource.from_dict(s) for s in data.get("sources", []))
        return cls(**data)


@dataclass
class MusicPreferences:
    """Preferencias de música del usuario"""
    favorite_genres: list[str] = field(default_factory=list)
    favorite_artists: list[str] = field(default_factory=list)
    disliked_genres: list[str] = field(default_factory=list)
    default_energy: float | None = None  # 0.0-1.0
    default_valence: float | None = None  # 0.0-1.0 (positividad)

    def to_dict(self) -> dict:
        return {
            "favorite_genres": self.favorite_genres,
            "favorite_artists": self.favorite_artists,
            "disliked_genres": self.disliked_genres,
            "default_energy": self.default_energy,
            "default_valence": self.default_valence
        }


@dataclass
class UserContext:
    """
    Contexto de un usuario individual.

    Mantiene:
    - Historial de conversacion (ultimos N turnos)
    - Zona activa (de donde habla)
    - Preferencias del usuario
    - Estado de la conversacion actual
    """
    user_id: str
    user_name: str
    zone_id: str | None = None

    # Historial de conversacion
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    max_history: int = 10  # Maximo turnos a mantener

    # Preferencias (cargadas del user manager)
    preferences: dict = field(default_factory=dict)
    permission_level: int = 0

    # Preferencias de música
    music_preferences: MusicPreferences = field(default_factory=MusicPreferences)

    # Estado temporal
    last_active: float = field(default_factory=time.time)
    last_intent: str | None = None
    pending_confirmation: dict | None = None  # Para rutinas que requieren confirmacion

    # Estadisticas de sesion
    session_start: float = field(default_factory=time.time)
    turns_count: int = 0

    # Compaction state (plan #2 OpenClaw)
    compacted_summary: str | None = None
    preserved_ids: list[str] = field(default_factory=list)
    compaction_inflight: bool = False  # transient, no se serializa
    session_count: int = 1
    summary_sources: tuple[MemorySource, ...] = ()
    privacy_revision: int = 0
    activity_revision: int = 0
    _compaction_token: object | None = field(default=None, repr=False)
    _clock: Callable[[], float] | None = field(default=None, repr=False)
    _source_policy: FactLedger | None = field(default=None, repr=False)

    def sources_available(self, sources: tuple[MemorySource, ...]) -> bool:
        """Apply raw expiry and, when connected, committed conversation revocations."""
        now = time.time() if self._clock is None else self._clock()
        return sources_current(sources, now) and (self._source_policy is None or self._source_policy.sources_available(sources))

    def prune(self) -> None:
        """Remove expired/revoked text and summaries before any projection."""
        self.conversation_history = [t for t in self.conversation_history if self.sources_available(t.sources)]
        if not self.sources_available(self.summary_sources):
            self.compacted_summary = None
            self.preserved_ids = []
            self.summary_sources = ()

    def add_turn(self, role: str, content: str, intent: str = None, entities: list = None,
                 *, sources: tuple[MemorySource, ...] | None = None):
        """Agregar turno al historial"""
        now = time.time() if self._clock is None else self._clock()
        if sources is None:
            sources = (MemorySource(str(uuid4()), str(uuid4()), now, now+48*3600, MemoryOrigin.INTERACTION),)
        self.prune()
        if not self.sources_available(sources):
            return
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            entities=entities or [], sources=sources,
            timestamp=min(s.captured_at for s in sources),
        )
        self.conversation_history.append(turn)
        self.turns_count += 1
        self.last_active = time.time()
        self.activity_revision += 1

        if intent:
            self.last_intent = intent

        # Mantener limite de historial
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_history_for_prompt(self, include_system: bool = False) -> list[dict]:
        """Obtener historial formateado para prompt"""
        self.prune()
        history = []
        for turn in self.conversation_history:
            if turn.role == "system" and not include_system:
                continue
            history.append({
                "role": turn.role,
                "content": turn.content
            })
        return history

    def get_recent_entities(self, n_turns: int = 3) -> list[str]:
        """Obtener entidades mencionadas recientemente"""
        self.prune()
        entities = []
        for turn in self.conversation_history[-n_turns:]:
            entities.extend(turn.entities)
        return list(set(entities))

    def clear_history(self):
        """Limpiar historial manteniendo preferencias"""
        self.conversation_history = []
        self.turns_count = 0
        self.pending_confirmation = None
        self.compacted_summary = None
        self.summary_sources = ()
        self.preserved_ids = []
        self.privacy_revision += 1
        self.compaction_inflight = False
        self._compaction_token = None

    def is_active(self, timeout_seconds: float = 300) -> bool:
        """Verificar si el contexto esta activo (ultimo uso < timeout)"""
        return (time.time() - self.last_active) < timeout_seconds

    def get_session_duration(self) -> float:
        """Duracion de la sesion en segundos"""
        return time.time() - self.session_start

    def to_dict(self) -> dict:
        """Serializar contexto.

        NOTA: compaction_inflight es transient (mutex flag) y NO se incluye.
        El persister bypassea to_dict() y construye su propio payload con
        version + last_seen, pero los demás callsites (debug dump, etc.)
        usan esto.
        """
        self.prune()
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "zone_id": self.zone_id,
            "conversation_history": [t.to_dict() for t in self.conversation_history],
            "preferences": self.preferences,
            "permission_level": self.permission_level,
            "last_active": self.last_active,
            "session_start": self.session_start,
            "turns_count": self.turns_count,
            # Plan #2 OpenClaw
            "compacted_summary": self.compacted_summary,
            "preserved_ids": list(self.preserved_ids),
            "session_count": self.session_count,
            "summary_sources": [s.to_dict() for s in self.summary_sources],
        }


class ContextManager:
    """
    Gestor de contextos de conversacion para multiples usuarios.

    Thread-safe para acceso concurrente desde multiples peticiones.

    Caracteristicas:
    - Contexto separado por usuario
    - Limpieza automatica de contextos inactivos
    - Construccion de prompts con historial
    - Soporte para preferencias de usuario

    Ejemplo:
        manager = ContextManager(max_history=10, inactive_timeout=300)

        # Obtener/crear contexto
        ctx = manager.get_or_create("user123", "Juan", "living_room")

        # Construir prompt con historial
        prompt = manager.build_prompt("user123", "¿Cómo está el clima?")

        # Agregar respuesta
        manager.add_turn("user123", "assistant", "El clima está soleado...")
    """

    def __init__(
        self,
        max_history: int = 10,
        inactive_timeout: float = 300,
        cleanup_interval: float = 60,
        system_prompt: str = None,
        compactor=None,  # Compactor | None — plan #2 OpenClaw
        persister=None,  # ContextPersister | None — plan #2 OpenClaw
        compaction_threshold: int = 6,
        keep_recent_turns: int = 3,
        source_policy: FactLedger | None = None,
        clock: Callable[[], float] | None = None,
        io_worker: Callable | None = None,
    ):
        """
        Args:
            max_history: Maximo de turnos por usuario
            inactive_timeout: Segundos sin actividad antes de limpiar contexto
            cleanup_interval: Intervalo de limpieza automatica
            system_prompt: Prompt de sistema predeterminado
            compactor: Compactor instance (plan #2 OpenClaw); si None, sin compactación
            persister: ContextPersister instance; si None, sin persistencia cross-sesión
            compaction_threshold: turnos al alcanzar los cuales dispara compactación
            keep_recent_turns: turnos literal preservados al final tras compactar
        """
        self.max_history = max_history
        self.inactive_timeout = inactive_timeout
        self.cleanup_interval = cleanup_interval
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Plan #2 OpenClaw — compaction + persistence
        self.compactor = compactor
        self.persister = persister
        self.compaction_threshold = compaction_threshold
        self.keep_recent_turns = keep_recent_turns
        self.source_policy = source_policy
        self._clock = clock
        self._io_worker = io_worker
        self._io_tasks: set[asyncio.Task] = set()
        self._io_closed = False

        self._contexts: dict[str, UserContext] = {}
        self._lock = threading.RLock()

        self._cleanup_running = False
        self._cleanup_thread: threading.Thread | None = None

        self._total_contexts_created = 0
        self._total_contexts_cleaned = 0

        # Plan #2 OpenClaw — observability + task tracking
        self._compaction_tasks: set[asyncio.Task] = set()
        self._compaction_attempts: int = 0
        self._compaction_failures: int = 0
        self._compaction_last_error: str | None = None
        self._persist_failures: int = 0
        self._persist_load_failures: int = 0  # reserved; persister.load swallows errors
        self._no_loop_warned: bool = False
        self._cleanup_iteration_count: int = 0
        self._stats_log_every_n: int = 60  # log stats every N cleanup ticks

    def _default_system_prompt(self) -> str:
        """Prompt de sistema predeterminado"""
        return """Eres un asistente de hogar inteligente. Responde de forma concisa y natural en español.

Puedes ayudar con:
- Control de dispositivos del hogar (luces, clima, persianas, etc.)
- Creación de rutinas y automatizaciones
- Preguntas generales y conversación
- Consultas sobre el estado del hogar

Mantén respuestas breves pero informativas. Usa el contexto de la conversación para dar respuestas relevantes."""

    def start_cleanup_thread(self):
        """Iniciar thread de limpieza automatica"""
        if self._cleanup_running:
            return

        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="ContextCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Context cleanup thread iniciado")

    def stop_cleanup_thread(self):
        """Detener thread de limpieza"""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def _cleanup_loop(self):
        """Loop de limpieza de contextos inactivos"""
        while self._cleanup_running:
            time.sleep(self.cleanup_interval)
            try:
                self.cleanup_inactive()
            except Exception as e:
                logger.error(f"Error en cleanup: {e}")

    def get_or_create(
        self,
        user_id: str,
        user_name: str = None,
        zone_id: str = None,
        preferences: dict = None,
        permission_level: int = 0,
    ) -> UserContext:
        """
        Obtener contexto existente o crear uno nuevo.

        Si hay un ContextPersister inyectado y existe snapshot en disco para
        este user_id, se hidrata el nuevo contexto con compacted_summary,
        preserved_ids y session_count incrementado. Conversation history NO
        se restaura (los turnos viejos murieron).

        Args:
            user_id: ID unico del usuario
            user_name: Nombre para mostrar
            zone_id: Zona desde donde habla
            preferences: Preferencias del usuario
            permission_level: Nivel de permisos

        Returns:
            UserContext del usuario
        """
        with self._lock:
            if user_id in self._contexts:
                ctx = self._contexts[user_id]
                if zone_id:
                    ctx.zone_id = zone_id
                ctx.last_active = time.time()
                ctx.activity_revision += 1
                ctx.prune()
                return ctx

        hydrated = self.persister.load(user_id) if self.persister is not None else None
        with self._lock:
            if user_id in self._contexts:
                return self.get_or_create(user_id, user_name, zone_id, preferences, permission_level)

            ctx = UserContext(
                user_id=user_id,
                user_name=user_name or (hydrated and hydrated.get("user_name")) or f"Usuario_{user_id[:8]}",
                zone_id=zone_id,
                max_history=self.max_history,
                preferences=preferences or {},
                permission_level=permission_level,
                _clock=self._clock, _source_policy=self.source_policy,
            )
            if hydrated:
                ctx.compacted_summary = hydrated.get("compacted_summary")
                ctx.preserved_ids = list(hydrated.get("preserved_ids") or [])
                ctx.session_count = (hydrated.get("session_count") or 1) + 1
                ctx.summary_sources = tuple(MemorySource.from_dict(s) for s in hydrated.get("summary_sources", []))
                ctx.prune()
                logger.info(
                    f"[ContextManager] hydrated user={user_id} session_count={ctx.session_count}"
                )

            self._contexts[user_id] = ctx
            self._total_contexts_created += 1

            logger.debug(f"Contexto creado: {user_id} ({ctx.user_name})")
            return ctx

    async def _offload(self, function, *args, **kwargs):
        if self._io_closed:
            raise RuntimeError("context_io_closed")
        if self._io_worker is not None:
            return await self._io_worker(function, *args, **kwargs)
        if len(self._io_tasks) >= 8:
            raise RuntimeError("context_io_capacity")
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._io_tasks.add(task)
        def finished(done):
            self._io_tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def aget_or_create(self, user_id: str, *args, **kwargs) -> UserContext:
        """Hydrate private disk state through an owned off-loop operation."""
        if self._io_closed:
            raise RuntimeError("context_io_closed")
        with self._lock:
            if user_id in self._contexts:
                return self.get_or_create(user_id, *args, **kwargs)
        return await self._offload(self.get_or_create, user_id, *args, **kwargs)

    def interaction_sources(self, captured_at: float | None = None) -> tuple[MemorySource, ...]:
        """Mark a server-received text exchange once at admission, not completion."""
        now = time.time() if self._clock is None else self._clock()
        captured_at = now if captured_at is None else captured_at
        return (MemorySource(str(uuid4()), str(uuid4()), captured_at, captured_at+48*3600, MemoryOrigin.INTERACTION),)

    async def aclose(self) -> None:
        """Stop compaction and wait for actual owned disk operations to finish."""
        self._io_closed = True
        tasks = list(self._compaction_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._io_tasks:
            async with asyncio.timeout(1):
                await asyncio.shield(asyncio.gather(*list(self._io_tasks), return_exceptions=True))

    def get(self, user_id: str) -> UserContext | None:
        """Obtener contexto si existe"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.prune()
            return ctx

    def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        intent: str = None,
        entities: list = None,
        *, sources: tuple[MemorySource, ...] | None = None,
    ):
        """Agregar turno al historial. Si hay compactor y se alcanza el
        umbral, lanza compactación en background fire-and-forget.

        Args:
            user_id: ID del usuario
            role: "user" o "assistant"
            content: Contenido del mensaje
            intent: Tipo de intent detectado
            entities: Entidades mencionadas
        """
        should_compact = False
        with self._lock:
            ctx = self._contexts.get(user_id)
            if not ctx:
                logger.warning(f"Contexto no encontrado: {user_id}")
                return
            ctx.add_turn(role, content, intent, entities, sources=sources)

            if (
                self.compactor is not None
                and not ctx.compaction_inflight
                and len(ctx.conversation_history) >= self.compaction_threshold
            ):
                ctx.compaction_inflight = True
                should_compact = True

        if should_compact:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop (uso desde código sync) — desactivar trigger silenciosamente
                if not self._no_loop_warned:
                    logger.warning(
                        "[ContextManager] no event loop running — compaction trigger disabled "
                        "for this manager. (Won't log this again until restart.)"
                    )
                    self._no_loop_warned = True
                with self._lock:
                    ctx = self._contexts.get(user_id)
                    if ctx:
                        ctx.compaction_inflight = False
                return
            # Strong reference to fire-and-forget task: Python 3.11+ event loops
            # only hold weak refs to running tasks. Without this set, mid-flight
            # GC can produce "RuntimeWarning: Task was destroyed".
            task = loop.create_task(self._compact_background(user_id))
            self._compaction_tasks.add(task)
            task.add_done_callback(self._compaction_tasks.discard)

    async def _compact_background(self, user_id: str, *, keep_recent: int | None = None) -> None:
        """Publish a summary only while its original context and sources survive."""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return
            ctx.prune()
            count = max(0, len(ctx.conversation_history) - (self.keep_recent_turns if keep_recent is None else keep_recent))
            if count == 0:
                ctx.compaction_inflight = False
                return
            original = ctx
            revision = ctx.privacy_revision
            token = object()
            ctx._compaction_token = token
            ctx.compaction_inflight = True
            turns = list(ctx.conversation_history[:count])
            turn_ids = {id(t) for t in turns}
            sources = tuple({s.capture_id: s for s in (*ctx.summary_sources, *(s for t in turns for s in t.sources))}.values())
            preserved = list(ctx.preserved_ids)
            for turn in turns:
                preserved.extend(turn.entities)

        self._compaction_attempts += 1
        try:
            result = await self.compactor.compact(turns, preserved_entities=preserved)
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx is not original or ctx.privacy_revision != revision or not ctx.sources_available(sources):
                    return
                ctx.prune()
                existing = (ctx.compacted_summary + " ") if ctx.compacted_summary else ""
                ctx.compacted_summary = _truncate_summary_head((existing + result.summary).strip())
                ctx.summary_sources = sources
                ctx.preserved_ids = sorted(set(ctx.preserved_ids) | set(result.preserved_ids))
                ctx.conversation_history = [t for t in ctx.conversation_history if id(t) not in turn_ids]
                ctx.privacy_revision += 1
        except Exception as error:
            self._compaction_failures += 1
            from src.orchestrator.compactor import CompactionError
            self._compaction_last_error = error.kind.name if isinstance(error, CompactionError) else type(error).__name__
            logger.warning("[ContextManager] compaction failed: %s", type(error).__name__)
        finally:
            with self._lock:
                if original._compaction_token is token:
                    original.compaction_inflight = False
                    original._compaction_token = None

    def build_prompt(
        self,
        user_id: str,
        user_input: str,
        include_system: bool = True,
        include_home_state: str = None,
        custom_system_prompt: str = None
    ) -> str:
        """
        Construir prompt completo con historial de conversacion.

        Args:
            user_id: ID del usuario
            user_input: Nuevo input del usuario
            include_system: Incluir prompt de sistema
            include_home_state: Estado del hogar para incluir
            custom_system_prompt: Override del system prompt

        Returns:
            Prompt formateado listo para el LLM
        """
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.prune()
            if not ctx:
                # Sin contexto, prompt simple
                return self._simple_prompt(user_input, custom_system_prompt)

            parts = []

            # System prompt
            if include_system:
                system = custom_system_prompt or self.system_prompt
                # Personalizar con nombre
                system = system.replace("{user_name}", ctx.user_name)
                parts.append(system)

            # Estado del hogar (opcional)
            if include_home_state:
                parts.append(f"\nEstado actual del hogar:\n{include_home_state}")

            # Contexto temporal
            now = datetime.now()
            parts.append(f"\nFecha y hora: {now.strftime('%A %d de %B, %H:%M')}")
            parts.append(f"Usuario: {ctx.user_name}")
            if ctx.zone_id:
                parts.append(f"Ubicación: {ctx.zone_id}")

            # Plan #2 OpenClaw — inject compacted summary + preserved entity hints.
            # Without this, enabling compaction REMOVES memory rather than compresses it.
            if ctx.compacted_summary:
                parts.append(f"\nResumen de conversación previa:\n{ctx.compacted_summary}")
            if ctx.preserved_ids:
                parts.append(
                    "Entidades del hogar referenciadas previamente: "
                    + ", ".join(ctx.preserved_ids)
                )

            # Historial de conversacion
            history = ctx.get_history_for_prompt()
            if history:
                parts.append("\nConversación anterior:")
                for turn in history:
                    role_label = "Usuario" if turn["role"] == "user" else "Asistente"
                    parts.append(f"{role_label}: {turn['content']}")

            # Input actual
            parts.append(f"\nUsuario: {user_input}")
            parts.append("\nAsistente:")

            return "\n".join(parts)

    def _simple_prompt(self, user_input: str, custom_system: str = None) -> str:
        """Prompt simple sin contexto"""
        system = custom_system or self.system_prompt
        return f"{system}\n\nUsuario: {user_input}\n\nAsistente:"

    def build_chat_messages(
        self,
        user_id: str,
        user_input: str,
        custom_system_prompt: str = None
    ) -> list[dict]:
        """
        Construir mensajes en formato chat (para modelos que lo soporten).

        Returns:
            Lista de mensajes [{role, content}, ...]
        """
        messages = []

        # System message
        system = custom_system_prompt or self.system_prompt
        messages.append({"role": "system", "content": system})

        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.prune()
                # Plan #2 OpenClaw — surface compacted summary as system-side context
                # before conversation_history. Without this, compaction removes memory
                # instead of compressing it.
                if ctx.compacted_summary:
                    summary_msg = f"Resumen de conversación previa:\n{ctx.compacted_summary}"
                    if ctx.preserved_ids:
                        summary_msg += (
                            "\nEntidades del hogar referenciadas previamente: "
                            + ", ".join(ctx.preserved_ids)
                        )
                    messages.append({"role": "system", "content": summary_msg})
                # Agregar historial
                for turn in ctx.conversation_history:
                    messages.append({
                        "role": turn.role,
                        "content": turn.content
                    })

        # Agregar input actual
        messages.append({"role": "user", "content": user_input})

        return messages

    def set_pending_confirmation(self, user_id: str, confirmation_data: dict):
        """Establecer confirmacion pendiente (ej: para rutinas)"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.pending_confirmation = confirmation_data

    def get_pending_confirmation(self, user_id: str) -> dict | None:
        """Obtener confirmacion pendiente"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            return ctx.pending_confirmation if ctx else None

    def clear_pending_confirmation(self, user_id: str):
        """Limpiar confirmacion pendiente"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.pending_confirmation = None

    def cleanup_inactive(self) -> int:
        """
        Limpiar contextos inactivos.

        Returns:
            Numero de contextos limpiados
        """
        cleaned = 0
        now = time.time()

        with self._lock:
            for ctx in self._contexts.values():
                ctx.prune()
            inactive_ids = [
                uid for uid, ctx in self._contexts.items()
                if (now - ctx.last_active) > self.inactive_timeout
            ]

            for uid in inactive_ids:
                del self._contexts[uid]
                cleaned += 1
                self._total_contexts_cleaned += 1

        if cleaned > 0:
            logger.info(f"Contextos limpiados: {cleaned}")

        return cleaned

    async def start_cleanup_loop_async(self) -> None:
        """Loop async de cleanup. Reemplaza al thread daemon cuando hay event loop.

        Llamar desde main.py: asyncio.create_task(mgr.start_cleanup_loop_async()).
        Detener con stop_cleanup_loop_async().

        ATENCIÓN: no llamar simultáneamente con start_cleanup_thread() — ambos
        intentarán limpiar el mismo dict y pueden borrar contextos dos veces.
        MultiUserOrchestrator.start() ya elige uno u otro en función de
        self._persister; este método sólo debe invocarse manualmente en tests
        o admin tools.
        """
        self._cleanup_running = True
        logger.info("[ContextManager] async cleanup loop started")
        while self._cleanup_running:
            try:
                await self._cleanup_inactive_async()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ContextManager] cleanup loop error: {e}")
        logger.info("[ContextManager] async cleanup loop stopped")

    def stop_cleanup_loop_async(self) -> None:
        """Pedirle al loop async que termine en la próxima iteración.

        No bloquea — el caller debe await la task externa (ver
        MultiUserOrchestrator.stop()).
        """
        self._cleanup_running = False

    async def _cleanup_inactive_async(self) -> int:
        """Async equivalente de cleanup_inactive: si hay persister, snapshot
        antes de eliminar el contexto."""
        now = time.time()
        if self.persister is not None:
            await self._offload(self.persister.purge_expired)
        with self._lock:
            for ctx in self._contexts.values():
                ctx.prune()
            inactive_ids = [
                uid for uid, ctx in self._contexts.items()
                if (now - ctx.last_active) > self.inactive_timeout
            ]

        cleaned = 0
        for uid in inactive_ids:
            cleaned += int(await self._snapshot_and_remove(uid))

        if cleaned > 0:
            logger.info(f"[ContextManager] async cleanup removed {cleaned} contexts")

        self._cleanup_iteration_count += 1
        if self._cleanup_iteration_count % self._stats_log_every_n == 0:
            stats = self.get_stats()
            # Only log Plan #2 counters (the rest are noisy / unchanged most of the time)
            logger.info(
                f"[ContextManager] stats tick: "
                f"compaction_attempts={stats['compaction_attempts']} "
                f"compaction_failures={stats['compaction_failures']} "
                f"persist_failures={stats['persist_failures']} "
                f"in_flight={stats['compaction_tasks_in_flight']} "
                f"last_error={stats['compaction_last_error']!r}"
            )
        return cleaned

    async def _snapshot_and_remove(self, user_id: str) -> bool:
        """Compactar pendiente + persistir snapshot atómico + remover de memoria.

        - Si hay compactor + persister + turnos pendientes: compacta el
          resto antes de persistir (best-effort, errores logueados +
          contados en _compaction_failures).
        - Construye el payload bajo lock para evitar torn reads (otro
          thread podría mutar UserContext durante el dump JSON).
        - Persiste outside-of-lock vía asyncio.to_thread sobre el payload
          inmutable (no el UserContext mutable).
        - Si persister.save_payload falla, incrementa _persist_failures y
          deja el contexto en memoria — el próximo cleanup tick reintenta.
        """
        with self._lock:
            original = self._contexts.get(user_id)
            if original is None or original.compaction_inflight:
                return False
            original.prune()
            revision = original.privacy_revision
            last_active = original.last_active
            admission_last_active = original.last_active
            activity_revision = original.activity_revision
        if self.compactor is not None and self.persister is not None:
            await self._compact_background(user_id, keep_recent=0)

        # Step 2: build immutable payload under lock (or skip persist)
        payload = None
        if self.persister is not None:
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx is not original or ctx.last_active != admission_last_active or ctx.activity_revision != activity_revision:
                    return False
                ctx.prune()
                revision = ctx.privacy_revision
                last_active = ctx.last_active
                if ctx.compacted_summary:
                    payload = self.persister.build_payload(ctx)

        # Step 3: persist outside lock; on failure keep ctx in memory
        if payload is not None:
            try:
                await self._offload(self.persister.save_payload, payload)
            except Exception as e:
                self._persist_failures += 1
                logger.warning(
                    f"[ContextManager] snapshot save failed for {user_id}: {e} "
                    "(keeping context in memory; will retry on next cleanup tick)"
                )
                return False  # Keep context for a later retry.

        # Step 4: remove from memory
        with self._lock:
            current = self._contexts.get(user_id)
            if current is original and current.last_active == last_active and current.privacy_revision == revision and current.activity_revision == activity_revision:
                del self._contexts[user_id]
                self._total_contexts_cleaned += 1
                return True
        return False

    def clear_user_history(self, user_id: str):
        """Limpiar historial de un usuario especifico"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx:
                ctx.clear_history()
                logger.debug(f"Historial limpiado: {user_id}")

    def remove_context(self, user_id: str):
        """Eliminar contexto de un usuario"""
        with self._lock:
            if user_id in self._contexts:
                del self._contexts[user_id]
                logger.debug(f"Contexto eliminado: {user_id}")

    def get_active_users(self) -> list[str]:
        """Obtener lista de usuarios activos"""
        with self._lock:
            return [
                uid for uid, ctx in self._contexts.items()
                if ctx.is_active(self.inactive_timeout)
            ]

    def get_stats(self) -> dict:
        """Obtener estadisticas del manager"""
        with self._lock:
            active = sum(1 for ctx in self._contexts.values()
                        if ctx.is_active(self.inactive_timeout))

            total_turns = sum(ctx.turns_count for ctx in self._contexts.values())

            return {
                "total_contexts": len(self._contexts),
                "active_contexts": active,
                "total_contexts_created": self._total_contexts_created,
                "total_contexts_cleaned": self._total_contexts_cleaned,
                "total_turns": total_turns,
                "max_history_per_user": self.max_history,
                "inactive_timeout": self.inactive_timeout,
                # Plan #2 OpenClaw — observability
                "compaction_attempts": self._compaction_attempts,
                "compaction_failures": self._compaction_failures,
                "compaction_last_error": self._compaction_last_error,
                "compaction_tasks_in_flight": len(self._compaction_tasks),
                "persist_failures": self._persist_failures,
                "persist_load_failures": self._persist_load_failures,
            }

    def get_user_stats(self, user_id: str) -> dict | None:
        """Obtener estadisticas de un usuario"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if not ctx:
                return None

            return {
                "user_id": ctx.user_id,
                "user_name": ctx.user_name,
                "zone_id": ctx.zone_id,
                "turns_count": ctx.turns_count,
                "history_length": len(ctx.conversation_history),
                "session_duration": ctx.get_session_duration(),
                "last_active": ctx.last_active,
                "is_active": ctx.is_active(self.inactive_timeout),
                "last_intent": ctx.last_intent,
                "has_pending_confirmation": ctx.pending_confirmation is not None
            }
