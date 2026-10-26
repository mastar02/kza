"""
Memory Manager Module
Gestiona memoria de corto y largo plazo para contexto conversacional.

Este es el diferenciador clave vs Alexa: recordar preferencias,
patrones de uso y conversaciones previas.
"""

import json
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from src.memory.fact_ledger import FactDependency, MemoryOrigin, MemorySource
from src.memory.long_term import LongTermMemory
from src.memory.prompt_context import PromptContext, merge_sources
from src.memory.sources import CaptureScopes

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Un turno de conversación"""
    timestamp: float
    user_input: str
    assistant_response: str
    intent: str | None = None
    entities_used: list[str] = field(default_factory=list)
    sources: tuple[MemorySource, ...] = ()


@dataclass
class UserPreference:
    """Preferencia del usuario"""
    key: str
    value: str
    confidence: float
    last_updated: float
    source: str  # "explicit" o "inferred"


@dataclass
class MemoryFact:
    """Un hecho extraído de conversaciones"""
    content: str
    category: str  # personal, preference, pattern, fact
    confidence: float
    created_at: float
    last_accessed: float
    access_count: int = 0


class ShortTermMemory:
    """
    Memoria de corto plazo: últimos N turnos de conversación.
    Permite mantener contexto conversacional inmediato.
    """

    def __init__(self, max_turns: int = 10, *, clock=None, source_policy=None):
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._clock = time.time if clock is None else clock
        self._lock = threading.RLock()
        self._source_policy = source_policy

    def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        intent: str | None = None,
        entities_used: list[str] | None = None,
        sources: tuple[MemorySource, ...] = (),
    ):
        """Agregar un turno de conversación"""
        turn = ConversationTurn(
            timestamp=min((s.captured_at for s in sources), default=self._clock()),
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_used=entities_used or [], sources=sources,
        )
        with self._lock:
            self._turns.append(turn)
        logger.debug(f"Short-term memory: {len(self._turns)} turns")

    def get_recent(self, n: int = 3) -> list[ConversationTurn]:
        """Obtener los últimos N turnos"""
        with self._lock:
            now = self._clock()
            self._turns = deque((t for t in self._turns
                if t.timestamp+48*3600 > now and all(s.expires_at > now for s in t.sources)
                and (not any(s.fact_dependencies for s in t.sources) if self._source_policy is None else self._source_policy.sources_available(t.sources))),
                maxlen=self.max_turns)
            return list(self._turns)[-n:]

    def format_for_prompt(self, n: int = 3) -> str:
        """Formatear últimos turnos para incluir en prompt"""
        return self.format_turns(self.get_recent(n))

    @staticmethod
    def format_turns(recent: list[ConversationTurn]) -> str:
        """Format the same selected snapshot whose source receipts are retained."""
        if not recent:
            return ""

        lines = ["Conversación reciente:"]
        for turn in recent:
            lines.append(f"Usuario: {turn.user_input}")
            lines.append(f"Asistente: {turn.assistant_response}")

        return "\n".join(lines)

    def clear(self):
        """Limpiar memoria de corto plazo"""
        with self._lock:
            self._turns.clear()

    def __len__(self) -> int:
        return len(self.get_recent(self.max_turns))


class PreferencesStore:
    """
    Almacén de preferencias del usuario.
    JSON file para acceso rápido a preferencias frecuentes.
    """

    def __init__(self, file_path: str = "./data/preferences.json"):
        self.file_path = Path(file_path)
        self._preferences: dict[str, UserPreference] = {}
        self._load()

    def _load(self):
        """Cargar preferencias desde archivo"""
        if self.file_path.exists():
            try:
                with open(self.file_path) as f:
                    data = json.load(f)
                for key, pref_data in data.items():
                    self._preferences[key] = UserPreference(**pref_data)
                logger.info(f"Loaded {len(self._preferences)} preferences")
            except Exception as e:
                logger.warning(f"Error loading preferences: {e}")

    def _save(self):
        """Guardar preferencias a archivo"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: {
                "key": pref.key,
                "value": pref.value,
                "confidence": pref.confidence,
                "last_updated": pref.last_updated,
                "source": pref.source
            }
            for key, pref in self._preferences.items()
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def set(
        self,
        key: str,
        value: str,
        confidence: float = 0.8,
        source: str = "inferred"
    ):
        """Establecer una preferencia"""
        self._preferences[key] = UserPreference(
            key=key,
            value=value,
            confidence=confidence,
            last_updated=time.time(),
            source=source
        )
        self._save()
        logger.info(f"Preference set: {key}={value} ({source})")

    def get(self, key: str) -> str | None:
        """Obtener valor de preferencia"""
        pref = self._preferences.get(key)
        return pref.value if pref else None

    def get_all(self) -> dict[str, str]:
        """Obtener todas las preferencias"""
        return {k: v.value for k, v in self._preferences.items()}

    def delete(self, key: str) -> bool:
        """Eliminar una preferencia"""
        if key in self._preferences:
            del self._preferences[key]
            self._save()
            return True
        return False


class MemoryManager:
    """
    Gestor principal de memoria que coordina todas las capas.

    Arquitectura:
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Short-term      │  │ Long-term       │  │ Preferences     │
    │ (últimos 10     │  │ (ChromaDB)      │  │ (JSON)          │
    │  turnos)        │  │ - hechos        │  │ - horarios      │
    └────────┬────────┘  │ - patrones      │  │ - temps         │
             │           └────────┬────────┘  └──────┬──────────┘
             └──────────────┬─────┴──────────────────┘
                            ▼
                     Memory Manager
                            ▼
                     Prompt con contexto
    """

    def __init__(
        self,
        chroma_client,
        preferences_path: str = "./data/preferences.json",
        max_short_term_turns: int = 10,
        *, ledger_path: str = ":memory:", collection_name: str = "user_memories",
        clock=None, automatic_days: float = 30, privacy=None,
    ):
        self._clock = privacy.ledger.clock if privacy is not None else (time.time if clock is None else clock)
        self.long_term = LongTermMemory(chroma_client, collection_name,
                                       ledger_path=ledger_path, clock=self._clock, automatic_days=automatic_days, privacy=privacy)
        self.short_term = ShortTermMemory(max_turns=max_short_term_turns, clock=self._clock,
                                          source_policy=self.long_term.ledger)
        self.preferences = PreferencesStore(preferences_path)
        self._capture_scopes = privacy.scopes if privacy is not None else CaptureScopes(self.long_term.ledger)
        self._record_lock = threading.Lock()

        # Fact extractor se inyecta después para evitar dependencia circular
        self._fact_extractor = None

    def set_fact_extractor(self, extractor):
        """Configurar el extractor de hechos"""
        self._fact_extractor = extractor

    def initialize(self):
        """Inicializar todas las capas de memoria"""
        self.long_term.initialize()
        logger.info("Memory manager initialized")

    def source_for_capture(self, capture_id: str, captured_at: float, room_id: str | None,
                           *, conversation_id: str | None = None,
                           expires_at: float | None = None,
                           origin: MemoryOrigin = MemoryOrigin.VOICE) -> MemorySource:
        """Assign an interaction scope by producer/room and a 120s idle gap.

        Scopes describe recorded exchanges, not speaker identity or ownership.
        An ambient textual wake preserves its existing capture/conversation IDs.
        """
        return self._capture_scopes.source_for_capture(capture_id, captured_at, room_id,
            conversation_id=conversation_id, expires_at=expires_at, origin=origin)

    async def ainitialize(self) -> None:
        """Initialize policy memory through its owned worker scope."""
        await self.long_term.acall(self.initialize)

    async def arecord_interaction(self, *args, **kwargs) -> None:
        """Record/derive through owned workers, preserving cancellation tracking."""
        await self.long_term.acall(self.record_interaction, *args, **kwargs)

    async def aformat_context_for_prompt(self, query: str) -> str:
        """Build memory context without performing disk or embeddings on-loop."""
        return await self.long_term.acall(self.format_context_for_prompt, query)

    def recent_prompt_context(self, limit: int = 3) -> PromptContext:
        """Select short-term router context with the same raw dependency snapshot."""
        recent = self.short_term.get_recent(limit)
        return PromptContext(self.short_term.format_turns(recent), merge_sources(*(t.sources for t in recent)))

    async def aprompt_context(self, query: str) -> PromptContext:
        """Build text and provenance in one owned memory operation."""
        return await self.long_term.acall(self.prompt_context, query)

    def record_interaction(
        self,
        user_input: str,
        assistant_response: str,
        intent: str | None = None,
        entities_used: list[str] | None = None,
        *, sources: tuple[MemorySource, ...] = (),
    ):
        """
        Registrar una interacción en memoria.
        Extrae hechos automáticamente si hay un extractor configurado.
        """
        if not sources:
            sources = (self.source_for_capture(str(uuid4()), self._clock(), None, origin=MemoryOrigin.INTERACTION),)
        with self._record_lock:
            if not self.long_term.ledger.sources_admitted(sources):
                return
            self.short_term.add_turn(
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                entities_used=entities_used, sources=sources,
            )

        # Extraer hechos si el extractor está disponible
        if self._fact_extractor:
            try:
                facts = self._fact_extractor.extract(user_input, assistant_response)
                for fact in facts:
                    self.long_term.store_fact(
                        fact=fact["content"],
                        category=fact["category"],
                        confidence=fact.get("confidence", 0.7), sources=sources,
                    )
            except Exception as e:
                logger.debug(f"Fact extraction skipped: {e}")

    def build_context(self, current_query: str) -> dict:
        """
        Construir contexto completo para un prompt.

        Returns:
            {
                "short_term": str,       # Conversación reciente
                "relevant_facts": list,  # Hechos relevantes
                "preferences": dict      # Preferencias del usuario
            }
        """
        # Conversación reciente
        recent = self.short_term.get_recent(n=3)
        short_term_context = self.short_term.format_turns(recent)

        # Hechos relevantes
        relevant_facts = self.long_term.search_relevant(
            query=current_query,
            n_results=5,
            min_confidence=0.6
        )

        # A legacy JSON "explicit" flag is not authenticated confirmation.
        # Preserve that file for review; only policy-confirmed preferences
        # enter this context channel. Automatic preference facts use search.
        preferences = {}
        preference_facts = {}
        for fact in self.long_term.get_by_category("preference", limit=100):
            if fact["state"] == "confirmed":
                key, separator, value = fact["content"].partition(": ")
                preferences[key if separator else fact["id"]] = value if separator else key
                preference_facts[key if separator else fact["id"]] = fact

        return {
            "short_term": short_term_context,
            "relevant_facts": relevant_facts,
            "preferences": preferences,
            "sources": merge_sources(*(turn.sources for turn in recent)),
            "fact_dependencies": tuple(FactDependency(f["id"], f["revision"], f["expires_at"])
                for f in {f["id"]: f for f in [*preference_facts.values(), *relevant_facts[:3]]}.values()),
        }

    def format_context_for_prompt(self, current_query: str) -> str:
        """Formatear contexto como string para incluir en prompt"""
        return self.prompt_context(current_query).text

    def prompt_context(self, current_query: str) -> PromptContext:
        """Select prompt text and exactly its short-term/fact dependencies."""
        context = self.build_context(current_query)

        sections = []

        # Preferencias
        if context["preferences"]:
            prefs_lines = [f"- {k}: {v}" for k, v in context["preferences"].items()]
            sections.append("Preferencias del usuario:\n" + "\n".join(prefs_lines))

        # Hechos relevantes
        if context["relevant_facts"]:
            facts_lines = [f"- {f['content']}" for f in context["relevant_facts"][:3]]
            sections.append("Información relevante:\n" + "\n".join(facts_lines))

        # Conversación reciente
        if context["short_term"]:
            sections.append(context["short_term"])

        return PromptContext("\n\n".join(sections), context["sources"], context["fact_dependencies"])

    def remember_preference(self, key: str, value: str, explicit: bool = False):
        """Propose an explicit preference; inference gets only an automatic TTL."""
        source = self.source_for_capture(str(uuid4()), self._clock(), None, origin=MemoryOrigin.INTERACTION)
        if explicit:
            return self.long_term.propose(f"{key}: {value}", (source,), category="preference")["id"]
        return self.long_term.store_fact(f"{key}: {value}", "preference", confidence=.7, sources=(source,))

    def get_stats(self) -> dict:
        """Obtener estadísticas de todas las capas de memoria"""
        return {
            "short_term_turns": len(self.short_term),
            "long_term": self.long_term.get_stats(),
            "preferences_count": len(self.long_term.ledger.eligible_ids(category="preference")),
            "legacy_preferences_policy": "quarantined",
        }
