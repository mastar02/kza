"""
Suggestion Engine Module
Genera sugerencias de automatizacion basadas en patrones detectados.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.analytics.pattern_analyzer import Pattern, PatternType, PatternAnalyzer
from src.analytics.event_logger import EventLogger
from src.analytics.pattern_projection import pattern_key
from src.analytics.suggestion_store import SuggestionRecord, SuggestionStatus, SuggestionStore
from src.memory.fact_ledger import MemorySource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Suggestion:
    """Una sugerencia de automatizacion"""
    id: str
    pattern: Pattern
    message: str                    # Mensaje para el usuario
    automation_yaml: str            # YAML de Home Assistant
    created_at: float
    status: SuggestionStatus = SuggestionStatus.PENDING
    user_response: str | None = None
    snooze_until: float | None = None
    expires_at: float | None = None
    sources: tuple[MemorySource, ...] = ()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern.to_dict(),
            "message": self.message,
            "automation_yaml": self.automation_yaml,
            "created_at": self.created_at,
            "status": self.status.value,
            "user_response": self.user_response,
            "snooze_until": self.snooze_until,
            "expires_at": self.expires_at,
        }


class SuggestionEngine:
    """
    Genera sugerencias de automatizacion inteligentes.
    
    Flujo:
    1. Analiza patrones
    2. Genera sugerencia en lenguaje natural
    3. Crea YAML de automatizacion para Home Assistant
    4. Presenta al usuario
    5. Registra la respuesta y prepara el borrador para revisión
    """
    
    def __init__(
        self,
        event_logger: EventLogger,
        pattern_analyzer: PatternAnalyzer = None,
        suggestions_path: str = "./data/suggestions.json",
        llm_reasoner = None,
        ha_client = None
    ):
        self.event_logger = event_logger
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer(event_logger)
        self.suggestions_path = Path(suggestions_path)
        self.llm = llm_reasoner
        self.ha = ha_client
        
        self.store = SuggestionStore(event_logger, suggestions_path)
        self._clock = self.store.policy.clock

    def _project(self, record: SuggestionRecord) -> Suggestion:
        status = record.status
        if status is SuggestionStatus.SNOOZED and record.snooze_until <= self._clock():
            status = SuggestionStatus.PENDING
        return Suggestion(
            record.id, record.pattern, record.message, self._generate_automation_yaml(record.pattern),
            record.created_at, status=status, snooze_until=record.snooze_until,
            expires_at=record.expires_at, sources=record.sources,
        )

    @property
    def _suggestions(self) -> dict[str, Suggestion]:
        """Compatibility projection for the CLI; no private in-memory store exists."""
        return {record.id: self._project(record) for record in self.store.records()}

    def available_now(self, suggestion: Suggestion) -> bool:
        """Cheap expiry/revocation fence for an already resolved voice projection."""
        policy = self.store.policy
        return (
            isinstance(suggestion, Suggestion) and suggestion.expires_at is not None
            and not self.event_logger.resource_status()["closed"]
            and suggestion.created_at <= self._clock() < suggestion.expires_at
            and policy.activity_sources_available(suggestion.sources)
        )

    def generate_suggestions(self, min_confidence: float = .7, max_suggestions: int = 5) -> list[Suggestion]:
        """Generate bounded proposals, revalidating actual events at final commit."""
        if type(min_confidence) not in (int, float) or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("invalid_suggestion_confidence")
        if type(max_suggestions) is not int or not 0 < max_suggestions <= 50:
            raise ValueError("invalid_suggestion_limit")
        patterns = self.pattern_analyzer.get_actionable_patterns(min_confidence)
        result = []
        for pattern in patterns[:max_suggestions * 2]:
            try:
                # A supplied description/confidence is not event evidence.
                normalized = self.store.prepare(pattern)
                if normalized.confidence < min_confidence:
                    continue
                if self._find_existing_suggestion(normalized):
                    continue
                suggestion = self._create_suggestion(normalized)
                if suggestion is not None and self.available_now(suggestion):
                    result.append(suggestion)
            except (PermissionError, ValueError) as error:
                logger.debug("Suggestion generation discarded: %s", type(error).__name__)
            if len(result) >= max_suggestions:
                break
        return [suggestion for suggestion in result if self.available_now(suggestion)]

    def _find_existing_suggestion(self, pattern: Pattern) -> Suggestion | None:
        records = self.store.records(selection_key=pattern_key(pattern))
        return self._project(records[0]) if records else None

    def _create_suggestion(self, pattern: Pattern) -> Suggestion | None:
        created_at = self._clock()
        message = self._generate_message(pattern)
        if not message or not self._generate_automation_yaml(pattern):
            return None
        record = self.store.create(pattern, message, created_at)
        return self._project(record) if record is not None else None

    def _generate_message(self, pattern: Pattern) -> str:
        """Generar mensaje en lenguaje natural para el usuario"""
        
        # Si hay LLM, usarlo para mensaje mas natural
        if self.llm:
            return self._generate_message_with_llm(pattern)
        
        # Fallback: mensaje basado en plantillas
        return self._generate_message_template(pattern)
    
    def _generate_message_template(self, pattern: Pattern) -> str:
        """Generar mensaje usando plantillas"""
        
        if pattern.pattern_type == PatternType.DAILY_TIME:
            return (
                f"He notado que {pattern.description.lower()}. "
                f"Esto ha ocurrido {pattern.occurrences} veces con {int(pattern.confidence * 100)}% de consistencia. "
                f"¿Quieres que lo haga automaticamente?"
            )
        
        elif pattern.pattern_type == PatternType.WEEKDAY_TIME:
            return (
                f"Durante los dias de semana, sueles {pattern.description.lower()}. "
                f"¿Te gustaria que cree una automatizacion para esto?"
            )
        
        elif pattern.pattern_type == PatternType.WEEKEND_TIME:
            return (
                f"Los fines de semana, generalmente {pattern.description.lower()}. "
                f"¿Quieres que lo automatice?"
            )
        
        elif pattern.pattern_type == PatternType.SEQUENCE:
            trigger = self._friendly_entity(pattern.trigger_entity)
            target = self._friendly_entity(pattern.entity_id)
            return (
                f"Cuando activas {trigger}, casi siempre despues activas {target}. "
                f"¿Quieres que lo haga automaticamente?"
            )
        
        return pattern.description
    
    def _generate_message_with_llm(self, pattern: Pattern) -> str:
        """Generar mensaje usando el LLM para mayor naturalidad"""
        prompt = f"""Genera un mensaje corto y amigable para sugerir una automatizacion al usuario.

Patron detectado:
- Tipo: {pattern.pattern_type.value}
- Descripcion: {pattern.description}
- Confianza: {int(pattern.confidence * 100)}%
- Ocurrencias: {pattern.occurrences} veces

El mensaje debe:
1. Ser conversacional y amigable
2. Mencionar el patron observado
3. Preguntar si quiere automatizarlo
4. Ser breve (2-3 oraciones)

Responde SOLO con el mensaje, sin explicaciones:"""

        try:
            response = self.llm.generate(prompt, max_tokens=150, temperature=0.7)
            return response.strip() if isinstance(response, str) and 0 < len(response.strip()) <= 4096 else self._generate_message_template(pattern)
        except Exception as error:
            logger.warning("Suggestion wording unavailable: %s", type(error).__name__)
            return self._generate_message_template(pattern)
    
    def _generate_automation_yaml(self, pattern: Pattern) -> str:
        """Generar YAML de automatizacion para Home Assistant"""
        
        automation_id = f"auto_{pattern.entity_id.replace('.', '_')}_{pattern.action}"
        
        if pattern.pattern_type in [PatternType.DAILY_TIME, PatternType.WEEKDAY_TIME, PatternType.WEEKEND_TIME]:
            return self._generate_time_automation_yaml(pattern, automation_id)
        
        elif pattern.pattern_type == PatternType.SEQUENCE:
            return self._generate_sequence_automation_yaml(pattern, automation_id)
        
        return ""
    
    def _generate_time_automation_yaml(self, pattern: Pattern, automation_id: str) -> str:
        """Generar YAML para automatizacion basada en tiempo"""
        
        # Determinar dias
        if pattern.pattern_type == PatternType.WEEKDAY_TIME:
            days = "mon,tue,wed,thu,fri"
            alias_suffix = "dias de semana"
        elif pattern.pattern_type == PatternType.WEEKEND_TIME:
            days = "sat,sun"
            alias_suffix = "fines de semana"
        else:
            days = "mon,tue,wed,thu,fri,sat,sun"
            alias_suffix = "diario"
        
        # Determinar servicio
        domain = pattern.entity_id.split(".")[0]
        service = f"{domain}.{pattern.action}"
        
        hour = pattern.hour if pattern.hour is not None else 12
        minute = pattern.minute or 0
        
        yaml = f"""# Automatizacion generada por KZA Voice
# Patron detectado: {pattern.description}
# Confianza: {int(pattern.confidence * 100)}%

alias: "{self._friendly_action(pattern.action)} {self._friendly_entity(pattern.entity_id)} - {alias_suffix}"
description: "Generado automaticamente basado en {pattern.occurrences} eventos"
trigger:
  - platform: time
    at: "{hour:02d}:{minute:02d}:00"
condition:
  - condition: time
    weekday:
      - {days.replace(',', chr(10) + '      - ')}
action:
  - service: {service}
    target:
      entity_id: {pattern.entity_id}
mode: single
"""
        return yaml
    
    def _generate_sequence_automation_yaml(self, pattern: Pattern, automation_id: str) -> str:
        """Generar YAML para automatizacion basada en secuencia"""
        
        trigger_domain = pattern.trigger_entity.split(".")[0]
        target_domain = pattern.entity_id.split(".")[0]
        
        # Determinar estado trigger
        if pattern.trigger_action == "turn_on":
            trigger_state = "on"
        elif pattern.trigger_action == "turn_off":
            trigger_state = "off"
        else:
            trigger_state = "on"
        
        service = f"{target_domain}.{pattern.action}"
        
        yaml = f"""# Automatizacion generada por KZA Voice
# Patron detectado: {pattern.description}
# Confianza: {int(pattern.confidence * 100)}%

alias: "{self._friendly_entity(pattern.entity_id)} cuando {self._friendly_entity(pattern.trigger_entity)}"
description: "Generado automaticamente basado en {pattern.occurrences} secuencias"
trigger:
  - platform: state
    entity_id: {pattern.trigger_entity}
    to: "{trigger_state}"
condition: []
action:
  - delay:
      seconds: 2
  - service: {service}
    target:
      entity_id: {pattern.entity_id}
mode: single
"""
        return yaml
    
    def _friendly_entity(self, entity_id: str) -> str:
        """Convertir entity_id a nombre amigable"""
        parts = entity_id.split(".")
        if len(parts) == 2:
            domain, name = parts
            return name.replace("_", " ")
        return entity_id
    
    def _friendly_action(self, action: str) -> str:
        """Convertir accion a verbo amigable"""
        action_names = {
            "turn_on": "Encender",
            "turn_off": "Apagar",
            "toggle": "Alternar",
            "open_cover": "Abrir",
            "close_cover": "Cerrar"
        }
        return action_names.get(action, action.replace("_", " ").title())
    
    def get_pending_suggestions(self) -> list[Suggestion]:
        """Return only current proposals; snoozing never renews their expiry."""
        pending = [self._project(record) for record in self.store.records(pending=True)]
        return sorted((s for s in pending if self.available_now(s)), key=lambda s: s.pattern.confidence, reverse=True)

    def respond_to_suggestion(self, suggestion_id: str, accept: bool, snooze_hours: int = 0) -> dict:
        """Record a response to a live proposal, without claiming HA execution."""
        try:
            record = self.store.respond(suggestion_id, accept=accept, snooze_hours=snooze_hours)
        except PermissionError:
            record = None
        if record is None:
            return {"success": False, "error": "La sugerencia ya no está disponible."}
        if snooze_hours:
            return {"success": True, "message": "Sugerencia pospuesta mientras siga vigente."}
        if accept:
            return {
                "success": True,
                "message": "Sugerencia aceptada para revisión. El borrador está disponible.",
                "yaml": self._generate_automation_yaml(record.pattern),
            }
        return {"success": True, "message": "Sugerencia descartada."}

    def get_suggestion_to_present(self) -> Suggestion | None:
        """Resolve the current proposal again before handing it to a voice consumer."""
        pending = self.get_pending_suggestions() or self.generate_suggestions(max_suggestions=3)
        if not pending:
            return None
        records = self.store.records(suggestion_id=pending[0].id)
        selected = self._project(records[0]) if records else None
        return selected if selected is not None and self.available_now(selected) else None

    def get_stats(self) -> dict:
        """Count valid proposals only; legacy, expired and revoked rows are excluded."""
        suggestions = self._suggestions
        counts = {}
        for suggestion in suggestions.values():
            key = suggestion.status.value
            counts[key] = counts.get(key, 0) + 1
        accepted, rejected = counts.get("accepted", 0), counts.get("rejected", 0)
        return {"total_suggestions": len(suggestions), "by_status": counts,
                "acceptance_rate": accepted / (accepted + rejected) if accepted + rejected else 0}
