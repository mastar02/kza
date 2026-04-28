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
from dataclasses import dataclass, field
from datetime import datetime

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """Un turno en la conversacion"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: str | None = None  # "domotics", "conversation", "routine", etc.
    entities: list[str] = field(default_factory=list)  # Entidades mencionadas

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "entities": self.entities
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
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

    def add_turn(self, role: str, content: str, intent: str = None, entities: list = None):
        """Agregar turno al historial"""
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            entities=entities or []
        )
        self.conversation_history.append(turn)
        self.turns_count += 1
        self.last_active = time.time()

        if intent:
            self.last_intent = intent

        # Mantener limite de historial
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_history_for_prompt(self, include_system: bool = False) -> list[dict]:
        """Obtener historial formateado para prompt"""
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
        entities = []
        for turn in self.conversation_history[-n_turns:]:
            entities.extend(turn.entities)
        return list(set(entities))

    def clear_history(self):
        """Limpiar historial manteniendo preferencias"""
        self.conversation_history = []
        self.turns_count = 0
        self.pending_confirmation = None

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
                return ctx

            # Plan #2 OpenClaw — hidratar desde disco si hay persister
            hydrated = None
            if self.persister is not None:
                data = self.persister.load(user_id)
                if data is not None:
                    hydrated = data

            ctx = UserContext(
                user_id=user_id,
                user_name=user_name or (hydrated and hydrated.get("user_name")) or f"Usuario_{user_id[:8]}",
                zone_id=zone_id,
                max_history=self.max_history,
                preferences=preferences or {},
                permission_level=permission_level,
            )
            if hydrated:
                ctx.compacted_summary = hydrated.get("compacted_summary")
                ctx.preserved_ids = list(hydrated.get("preserved_ids") or [])
                ctx.session_count = (hydrated.get("session_count") or 1) + 1
                logger.info(
                    f"[ContextManager] hydrated user={user_id} session_count={ctx.session_count}"
                )

            self._contexts[user_id] = ctx
            self._total_contexts_created += 1

            logger.debug(f"Contexto creado: {user_id} ({ctx.user_name})")
            return ctx

    def get(self, user_id: str) -> UserContext | None:
        """Obtener contexto si existe"""
        with self._lock:
            return self._contexts.get(user_id)

    def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        intent: str = None,
        entities: list = None,
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
            ctx.add_turn(role, content, intent, entities)

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
                        f"[ContextManager] no event loop running — compaction trigger disabled "
                        f"for this manager. (Won't log this again until restart.)"
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

    async def _compact_background(self, user_id: str) -> None:
        """Compactación fire-and-forget en background.

        Snapshotea los turnos a compactar bajo lock, libera el lock para
        la llamada al LLM (que tarda ~seg), y re-adquiere para mutar el
        contexto. Errores se loguean (incrementando _compaction_failures)
        y limpian compaction_inflight; nunca propagan.

        Race-safe: identifica los turnos compactados por object identity
        en lugar de slice posicional, así turnos agregados durante el
        await no se evicean por error si max_history truncó la cabeza.

        Side effects:
            - ctx.compacted_summary: append nuevo resumen
            - ctx.preserved_ids: union con result.preserved_ids
            - ctx.conversation_history: drop sólo los turnos compactados
            - ctx.compaction_inflight: siempre False al salir
        """
        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return
            n_compact = max(0, len(ctx.conversation_history) - self.keep_recent_turns)
            if n_compact <= 0:
                ctx.compaction_inflight = False
                return
            turns_snapshot = list(ctx.conversation_history[:n_compact])
            compacted_turn_ids = {id(t) for t in turns_snapshot}
            preserved = []
            for t in turns_snapshot:
                preserved.extend(t.entities or [])
            preserved.extend(ctx.preserved_ids)

        self._compaction_attempts += 1
        try:
            result = await self.compactor.compact(turns_snapshot, preserved_entities=preserved)
        except Exception as e:
            self._compaction_failures += 1
            self._compaction_last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"[ContextManager] compaction failed for {user_id}: {e}")
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx:
                    ctx.compaction_inflight = False
            return

        with self._lock:
            ctx = self._contexts.get(user_id)
            if ctx is None:
                return
            existing = (ctx.compacted_summary + " ") if ctx.compacted_summary else ""
            ctx.compacted_summary = (existing + result.summary).strip()
            ctx.preserved_ids = sorted(set(ctx.preserved_ids) | set(result.preserved_ids))
            # Race-safe drop: keep only turns NOT in the compacted snapshot
            # (positional slice would evict recent turns if max_history trimmed head during await)
            ctx.conversation_history = [
                t for t in ctx.conversation_history if id(t) not in compacted_turn_ids
            ]
            ctx.compaction_inflight = False
            logger.info(
                f"[ContextManager] user={user_id} compacted_turns={n_compact} "
                f"summary_chars={len(ctx.compacted_summary)} preserved={len(ctx.preserved_ids)}"
            )

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
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive_async()
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
        with self._lock:
            inactive_ids = [
                uid for uid, ctx in self._contexts.items()
                if (now - ctx.last_active) > self.inactive_timeout
            ]

        cleaned = 0
        for uid in inactive_ids:
            await self._snapshot_and_remove(uid)
            cleaned += 1

        if cleaned > 0:
            logger.info(f"[ContextManager] async cleanup removed {cleaned} contexts")
        return cleaned

    async def _snapshot_and_remove(self, user_id: str) -> None:
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
        # Step 1: optionally compact remaining turns
        if self.compactor is not None and self.persister is not None:
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx is None:
                    return
                pending = list(ctx.conversation_history)
                preserved_seed = list(ctx.preserved_ids)
            if pending:
                try:
                    extra = []
                    for t in pending:
                        extra.extend(t.entities or [])
                    result = await self.compactor.compact(
                        pending, preserved_entities=preserved_seed + extra
                    )
                    with self._lock:
                        ctx = self._contexts.get(user_id)
                        if ctx:
                            existing = (ctx.compacted_summary + " ") if ctx.compacted_summary else ""
                            ctx.compacted_summary = (existing + result.summary).strip()
                            ctx.preserved_ids = sorted(
                                set(ctx.preserved_ids) | set(result.preserved_ids)
                            )
                            ctx.conversation_history = []
                except Exception as e:
                    self._compaction_failures += 1
                    self._compaction_last_error = f"{type(e).__name__}: {e}"
                    logger.warning(
                        f"[ContextManager] final compaction failed for {user_id}: {e}"
                    )

        # Step 2: build immutable payload under lock (or skip persist)
        payload = None
        if self.persister is not None:
            with self._lock:
                ctx = self._contexts.get(user_id)
                if ctx is not None and (ctx.compacted_summary or ctx.conversation_history):
                    payload = self.persister._build_payload(ctx)

        # Step 3: persist outside lock; on failure keep ctx in memory
        if payload is not None:
            try:
                await asyncio.to_thread(self.persister.save_payload, payload)
            except Exception as e:
                self._persist_failures += 1
                logger.warning(
                    f"[ContextManager] snapshot save failed for {user_id}: {e} "
                    "(keeping context in memory; will retry on next cleanup tick)"
                )
                return  # IMPORTANT: do NOT delete from in-memory dict

        # Step 4: remove from memory
        with self._lock:
            if user_id in self._contexts:
                del self._contexts[user_id]
                self._total_contexts_cleaned += 1

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
