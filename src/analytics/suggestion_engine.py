"""
Suggestion Engine Module
Genera sugerencias de automatizacion basadas en patrones detectados.
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from .pattern_analyzer import Pattern, PatternType, PatternAnalyzer
from .event_logger import EventLogger

logger = logging.getLogger(__name__)


class SuggestionStatus(StrEnum):
    """Estado de una sugerencia"""
    PENDING = "pending"       # Esperando respuesta del usuario
    ACCEPTED = "accepted"     # Usuario acepto
    REJECTED = "rejected"     # Usuario rechazo
    SNOOZED = "snoozed"       # Usuario pospuso
    IMPLEMENTED = "implemented"  # Automatizacion creada


@dataclass
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
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern.to_dict(),
            "message": self.message,
            "automation_yaml": self.automation_yaml,
            "created_at": self.created_at,
            "status": self.status.value,
            "user_response": self.user_response
        }


class SuggestionEngine:
    """
    Genera sugerencias de automatizacion inteligentes.
    
    Flujo:
    1. Analiza patrones
    2. Genera sugerencia en lenguaje natural
    3. Crea YAML de automatizacion para Home Assistant
    4. Presenta al usuario
    5. Si acepta, crea la automatizacion
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
        
        self._suggestions: dict[str, Suggestion] = {}
        self._load_suggestions()
    
    def _load_suggestions(self):
        """Cargar sugerencias guardadas"""
        if self.suggestions_path.exists():
            try:
                with open(self.suggestions_path) as f:
                    data = json.load(f)
                
                for s_data in data.get("suggestions", []):
                    # Reconstruir Pattern
                    p_data = s_data["pattern"]
                    pattern = Pattern(
                        pattern_type=PatternType(p_data["type"]),
                        entity_id=p_data["entity_id"],
                        action=p_data["action"],
                        confidence=p_data["confidence"],
                        occurrences=p_data["occurrences"],
                        description=p_data["description"],
                        hour=p_data.get("hour"),
                        minute=p_data.get("minute"),
                        weekdays=p_data.get("weekdays", []),
                        trigger_entity=p_data.get("trigger_entity"),
                        trigger_action=p_data.get("trigger_action")
                    )
                    
                    suggestion = Suggestion(
                        id=s_data["id"],
                        pattern=pattern,
                        message=s_data["message"],
                        automation_yaml=s_data["automation_yaml"],
                        created_at=s_data["created_at"],
                        status=SuggestionStatus(s_data["status"]),
                        user_response=s_data.get("user_response"),
                        snooze_until=s_data.get("snooze_until")
                    )
                    self._suggestions[suggestion.id] = suggestion
                
                logger.info(f"Cargadas {len(self._suggestions)} sugerencias")
            except Exception as e:
                logger.error(f"Error cargando sugerencias: {e}")
    
    def _save_suggestions(self):
        """Guardar sugerencias"""
        self.suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "suggestions": [s.to_dict() for s in self._suggestions.values()]
        }
        
        with open(self.suggestions_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def generate_suggestions(
        self,
        min_confidence: float = 0.7,
        max_suggestions: int = 5
    ) -> list[Suggestion]:
        """
        Generar nuevas sugerencias basadas en patrones.
        
        Args:
            min_confidence: Confianza minima para sugerir
            max_suggestions: Maximo de sugerencias a generar
        
        Returns:
            Lista de nuevas sugerencias
        """
        # Obtener patrones con alta confianza
        patterns = self.pattern_analyzer.get_actionable_patterns(min_confidence)
        
        new_suggestions = []
        
        for pattern in patterns[:max_suggestions * 2]:
            # Verificar si ya existe sugerencia para este patron
            existing = self._find_existing_suggestion(pattern)
            if existing:
                continue
            
            # Generar sugerencia
            suggestion = self._create_suggestion(pattern)
            if suggestion:
                self._suggestions[suggestion.id] = suggestion
                new_suggestions.append(suggestion)
                
                if len(new_suggestions) >= max_suggestions:
                    break
        
        if new_suggestions:
            self._save_suggestions()
            logger.info(f"Generadas {len(new_suggestions)} nuevas sugerencias")
        
        return new_suggestions
    
    def _find_existing_suggestion(self, pattern: Pattern) -> Suggestion | None:
        """Buscar si ya existe una sugerencia similar"""
        for suggestion in self._suggestions.values():
            p = suggestion.pattern
            if (p.entity_id == pattern.entity_id and
                p.action == pattern.action and
                p.pattern_type == pattern.pattern_type and
                p.hour == pattern.hour):
                return suggestion
        return None
    
    def _create_suggestion(self, pattern: Pattern) -> Suggestion | None:
        """Crear una sugerencia a partir de un patron"""
        import time
        
        suggestion_id = f"sug_{int(time.time() * 1000)}"
        
        # Generar mensaje para el usuario
        message = self._generate_message(pattern)
        
        # Generar YAML de automatizacion
        automation_yaml = self._generate_automation_yaml(pattern)
        
        if not message or not automation_yaml:
            return None
        
        return Suggestion(
            id=suggestion_id,
            pattern=pattern,
            message=message,
            automation_yaml=automation_yaml,
            created_at=time.time(),
            status=SuggestionStatus.PENDING
        )
    
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
            return response.strip()
        except Exception as e:
            logger.warning(f"Error generando mensaje con LLM: {e}")
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
        
        hour = pattern.hour or 12
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
        """Obtener sugerencias pendientes de respuesta"""
        import time
        now = time.time()
        
        pending = []
        for s in self._suggestions.values():
            if s.status == SuggestionStatus.PENDING:
                pending.append(s)
            elif s.status == SuggestionStatus.SNOOZED and s.snooze_until and s.snooze_until <= now:
                s.status = SuggestionStatus.PENDING
                pending.append(s)
        
        return sorted(pending, key=lambda s: s.pattern.confidence, reverse=True)
    
    def respond_to_suggestion(
        self,
        suggestion_id: str,
        accept: bool,
        snooze_hours: int = 0
    ) -> dict:
        """
        Responder a una sugerencia.
        
        Args:
            suggestion_id: ID de la sugerencia
            accept: True para aceptar, False para rechazar
            snooze_hours: Horas para posponer (si > 0)
        
        Returns:
            Resultado de la operacion
        """
        import time
        
        if suggestion_id not in self._suggestions:
            return {"success": False, "error": "Sugerencia no encontrada"}
        
        suggestion = self._suggestions[suggestion_id]
        
        if snooze_hours > 0:
            suggestion.status = SuggestionStatus.SNOOZED
            suggestion.snooze_until = time.time() + (snooze_hours * 3600)
            suggestion.user_response = f"Pospuesto por {snooze_hours} horas"
            self._save_suggestions()
            return {"success": True, "message": f"Sugerencia pospuesta por {snooze_hours} horas"}
        
        if accept:
            suggestion.status = SuggestionStatus.ACCEPTED
            suggestion.user_response = "Aceptado"
            
            # Intentar crear la automatizacion en Home Assistant
            if self.ha:
                result = self._create_automation_in_ha(suggestion)
                if result["success"]:
                    suggestion.status = SuggestionStatus.IMPLEMENTED
                self._save_suggestions()
                return result
            
            self._save_suggestions()
            return {
                "success": True,
                "message": "Sugerencia aceptada. Copia el YAML a tu Home Assistant.",
                "yaml": suggestion.automation_yaml
            }
        else:
            suggestion.status = SuggestionStatus.REJECTED
            suggestion.user_response = "Rechazado"
            self._save_suggestions()
            return {"success": True, "message": "Sugerencia descartada"}
    
    def _create_automation_in_ha(self, suggestion: Suggestion) -> dict:
        """Crear automatizacion directamente en Home Assistant"""
        # TODO: Implementar cuando HA client soporte crear automatizaciones
        # Por ahora, solo retornar el YAML
        return {
            "success": True,
            "message": "Automatizacion lista. Copia este YAML a tu automations.yaml:",
            "yaml": suggestion.automation_yaml
        }
    
    def get_suggestion_to_present(self) -> Suggestion | None:
        """
        Obtener la sugerencia mas relevante para presentar al usuario.
        Usado por el pipeline de voz.
        """
        pending = self.get_pending_suggestions()
        
        if not pending:
            # Generar nuevas si no hay pendientes
            new_suggestions = self.generate_suggestions(max_suggestions=3)
            if new_suggestions:
                return new_suggestions[0]
            return None
        
        return pending[0]
    
    def get_stats(self) -> dict:
        """Obtener estadisticas de sugerencias"""
        by_status = {}
        for s in self._suggestions.values():
            status = s.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_suggestions": len(self._suggestions),
            "by_status": by_status,
            "acceptance_rate": self._calculate_acceptance_rate()
        }
    
    def _calculate_acceptance_rate(self) -> float:
        """Calcular tasa de aceptacion"""
        accepted = sum(1 for s in self._suggestions.values() 
                      if s.status in [SuggestionStatus.ACCEPTED, SuggestionStatus.IMPLEMENTED])
        responded = sum(1 for s in self._suggestions.values()
                       if s.status in [SuggestionStatus.ACCEPTED, SuggestionStatus.IMPLEMENTED, 
                                       SuggestionStatus.REJECTED])
        
        if responded == 0:
            return 0.0
        return accepted / responded
