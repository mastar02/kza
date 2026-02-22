"""
Memory Manager Module
Gestiona memoria de corto y largo plazo para contexto conversacional.

Este es el diferenciador clave vs Alexa: recordar preferencias,
patrones de uso y conversaciones previas.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Un turno de conversación"""
    timestamp: float
    user_input: str
    assistant_response: str
    intent: str | None = None
    entities_used: list[str] = field(default_factory=list)


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

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        intent: str | None = None,
        entities_used: list[str] | None = None
    ):
        """Agregar un turno de conversación"""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_used=entities_used or []
        )
        self._turns.append(turn)
        logger.debug(f"Short-term memory: {len(self._turns)} turns")

    def get_recent(self, n: int = 3) -> list[ConversationTurn]:
        """Obtener los últimos N turnos"""
        return list(self._turns)[-n:]

    def format_for_prompt(self, n: int = 3) -> str:
        """Formatear últimos turnos para incluir en prompt"""
        recent = self.get_recent(n)
        if not recent:
            return ""

        lines = ["Conversación reciente:"]
        for turn in recent:
            lines.append(f"Usuario: {turn.user_input}")
            lines.append(f"Asistente: {turn.assistant_response}")

        return "\n".join(lines)

    def clear(self):
        """Limpiar memoria de corto plazo"""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


class LongTermMemory:
    """
    Memoria de largo plazo usando ChromaDB.
    Almacena hechos, preferencias y patrones extraídos.
    """

    def __init__(self, chroma_client, collection_name: str = "user_memories"):
        self._client = chroma_client
        self._collection_name = collection_name
        self._collection = None

    def initialize(self):
        """Inicializar colección en ChromaDB"""
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "User memories and facts"}
        )
        logger.info(f"Long-term memory initialized: {self._collection_name}")

    def store_fact(
        self,
        fact: str,
        category: str,
        confidence: float = 0.8,
        metadata: dict | None = None
    ) -> str:
        """
        Almacenar un hecho en memoria de largo plazo.

        Args:
            fact: El hecho a almacenar
            category: Categoría (personal, preference, pattern, fact)
            confidence: Nivel de confianza (0-1)
            metadata: Metadatos adicionales

        Returns:
            ID del documento creado
        """
        if self._collection is None:
            self.initialize()

        doc_id = f"fact_{int(time.time() * 1000)}"

        doc_metadata = {
            "category": category,
            "confidence": confidence,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0
        }
        if metadata:
            doc_metadata.update(metadata)

        self._collection.add(
            documents=[fact],
            ids=[doc_id],
            metadatas=[doc_metadata]
        )

        logger.info(f"Stored fact [{category}]: {fact[:50]}...")
        return doc_id

    def search_relevant(
        self,
        query: str,
        n_results: int = 5,
        min_confidence: float = 0.5
    ) -> list[dict]:
        """
        Buscar hechos relevantes para una consulta.

        Args:
            query: Consulta de búsqueda
            n_results: Número máximo de resultados
            min_confidence: Confianza mínima

        Returns:
            Lista de hechos relevantes con metadatos
        """
        if self._collection is None:
            self.initialize()

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"confidence": {"$gte": min_confidence}}
        )

        facts = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # Actualizar último acceso
                doc_id = results["ids"][0][i]
                self._update_access(doc_id)

                facts.append({
                    "content": doc,
                    "similarity": 1 - (distance / 2),  # Convert to similarity
                    **metadata
                })

        return facts

    def _update_access(self, doc_id: str):
        """Actualizar metadatos de acceso"""
        try:
            current = self._collection.get(ids=[doc_id])
            if current["metadatas"]:
                metadata = current["metadatas"][0]
                metadata["last_accessed"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                self._collection.update(ids=[doc_id], metadatas=[metadata])
        except Exception as e:
            logger.debug(f"Error updating access: {e}")

    def get_by_category(self, category: str, limit: int = 10) -> list[dict]:
        """Obtener hechos por categoría"""
        if self._collection is None:
            self.initialize()

        results = self._collection.get(
            where={"category": category},
            limit=limit
        )

        facts = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                facts.append({"content": doc, **metadata})

        return facts

    def delete_fact(self, doc_id: str) -> bool:
        """Eliminar un hecho"""
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting fact: {e}")
            return False

    def get_stats(self) -> dict:
        """Obtener estadísticas de la memoria"""
        if self._collection is None:
            self.initialize()

        return {
            "total_facts": self._collection.count(),
            "collection": self._collection_name
        }


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
        max_short_term_turns: int = 10
    ):
        self.short_term = ShortTermMemory(max_turns=max_short_term_turns)
        self.long_term = LongTermMemory(chroma_client)
        self.preferences = PreferencesStore(preferences_path)

        # Fact extractor se inyecta después para evitar dependencia circular
        self._fact_extractor = None

    def set_fact_extractor(self, extractor):
        """Configurar el extractor de hechos"""
        self._fact_extractor = extractor

    def initialize(self):
        """Inicializar todas las capas de memoria"""
        self.long_term.initialize()
        logger.info("Memory manager initialized")

    def record_interaction(
        self,
        user_input: str,
        assistant_response: str,
        intent: str | None = None,
        entities_used: list[str] | None = None
    ):
        """
        Registrar una interacción en memoria.
        Extrae hechos automáticamente si hay un extractor configurado.
        """
        # Guardar en memoria de corto plazo
        self.short_term.add_turn(
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_used=entities_used
        )

        # Extraer hechos si el extractor está disponible
        if self._fact_extractor:
            try:
                facts = self._fact_extractor.extract(user_input, assistant_response)
                for fact in facts:
                    self.long_term.store_fact(
                        fact=fact["content"],
                        category=fact["category"],
                        confidence=fact.get("confidence", 0.7)
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
        short_term_context = self.short_term.format_for_prompt(n=3)

        # Hechos relevantes
        relevant_facts = self.long_term.search_relevant(
            query=current_query,
            n_results=5,
            min_confidence=0.6
        )

        # Preferencias
        preferences = self.preferences.get_all()

        return {
            "short_term": short_term_context,
            "relevant_facts": relevant_facts,
            "preferences": preferences
        }

    def format_context_for_prompt(self, current_query: str) -> str:
        """Formatear contexto como string para incluir en prompt"""
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

        return "\n\n".join(sections) if sections else ""

    def remember_preference(self, key: str, value: str, explicit: bool = False):
        """Recordar una preferencia del usuario"""
        self.preferences.set(
            key=key,
            value=value,
            confidence=0.95 if explicit else 0.7,
            source="explicit" if explicit else "inferred"
        )

    def get_stats(self) -> dict:
        """Obtener estadísticas de todas las capas de memoria"""
        return {
            "short_term_turns": len(self.short_term),
            "long_term": self.long_term.get_stats(),
            "preferences_count": len(self.preferences.get_all())
        }
