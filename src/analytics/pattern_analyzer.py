"""
Pattern Analyzer Module
Detecta patrones de uso para sugerir automatizaciones.
"""

from collections import defaultdict
import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import StrEnum

from src.analytics.event_evidence import EventDependency, PatternEvent
from src.analytics.event_logger import EventLogger

logger = logging.getLogger(__name__)


class PatternType(StrEnum):
    """Tipos de patrones detectados"""
    DAILY_TIME = "daily_time"           # Mismo horario todos los dias
    WEEKDAY_TIME = "weekday_time"       # Mismo horario dias de semana
    WEEKEND_TIME = "weekend_time"       # Mismo horario fines de semana
    SEQUENCE = "sequence"               # A siempre seguido de B
    USER_PREFERENCE = "user_preference" # Usuario especifico siempre hace X
    CONDITIONAL = "conditional"         # Si condicion entonces accion


@dataclass
class Pattern:
    """Un patron detectado"""
    pattern_type: PatternType
    entity_id: str
    action: str
    confidence: float              # 0.0 a 1.0
    occurrences: int               # Veces que ocurrio
    description: str               # Descripcion en lenguaje natural
    
    # Detalles especificos del patron
    hour: int | None = None     # Hora del patron (0-23)
    minute: int | None = None   # Minuto aproximado
    hour_std: float | None = None  # Desviacion estandar de hora
    weekdays: list[int] = field(default_factory=list)  # Dias aplicables
    
    # Para patrones de secuencia
    trigger_entity: str | None = None
    trigger_action: str | None = None
    delay_seconds: float | None = None
    
    # Para patrones de usuario
    user_id: str | None = None
    user_name: str | None = None
    
    evidence: tuple[EventDependency, ...] = field(default=(), repr=False)

    @property
    def expires_at(self) -> float | None:
        """Keep the earliest actual dependency deadline; reading never renews it."""
        return min((item.expires_at for item in self.evidence), default=None)

    def to_dict(self) -> dict:
        return {
            "type": self.pattern_type.value,
            "entity_id": self.entity_id,
            "action": self.action,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "description": self.description,
            "hour": self.hour,
            "minute": self.minute,
            "weekdays": self.weekdays,
            "trigger_entity": self.trigger_entity,
            "trigger_action": self.trigger_action,
            "user_id": self.user_id,
            "event_dependencies": [item.to_dict() for item in self.evidence],
            "expires_at": self.expires_at,
        }


class PatternAnalyzer:
    """
    Analiza eventos para detectar patrones de uso.
    
    Tipos de patrones:
    1. Temporales: "Prende luz a las 7am todos los dias"
    2. Secuencias: "Despues de prender la cafetera, prende la luz"
    3. Por usuario: "Juan siempre apaga todo a las 11pm"
    4. Condicionales: "Si es fin de semana, despierta mas tarde"
    """
    
    # Umbrales de deteccion
    MIN_OCCURRENCES = 5           # Minimo de eventos para considerar patron
    MIN_CONFIDENCE = 0.6          # Confianza minima
    TIME_TOLERANCE_MINUTES = 30   # Tolerancia para "misma hora"
    SEQUENCE_WINDOW_MINUTES = 10  # Ventana para secuencias
    MAX_SEQUENCE_COMPARISONS = 100_000
    
    def __init__(self, event_logger: EventLogger):
        self.event_logger = event_logger
    
    def analyze_all(self, days: int = 30) -> list[Pattern]:
        """
        Analizar todos los eventos y detectar patrones.
        
        Args:
            days: Dias de historial a analizar
        
        Returns:
            Lista de patrones detectados ordenados por confianza
        """
        history = self.event_logger.get_pattern_history(days)
        if not history.complete:
            logger.warning("Pattern analysis deferred: incomplete activity window")
            return []
        # Diagnostic counters cannot become home automation suggestions. Text,
        # names and arbitrary event context are absent from this projection.
        events = tuple(event for event in history.events if
            re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", event.entity_id)
            and re.fullmatch(r"[a-z_]+", event.action))
        by_entity = defaultdict(list)
        for event in events:
            by_entity[event.entity_id].append(event)
        patterns = []
        for entity_id, entity_events in by_entity.items():
            patterns.extend(self._analyze_time_patterns(entity_id, entity_events))
        patterns.extend(self._analyze_sequences(events))
        patterns = [p for p in patterns if p.confidence >= self.MIN_CONFIDENCE]
        # Analysis can be slow; a marker or expiry during calculation makes its
        # result unusable even before physical cleanup removes the source row.
        evidence = tuple(dict.fromkeys(d for p in patterns for d in p.evidence))
        if patterns and not self.event_logger.pattern_evidence_current(evidence):
            logger.info("Pattern analysis discarded: activity evidence unavailable")
            return []
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info("Detected %d sourced activity patterns", len(patterns))
        return patterns

    def _analyze_time_patterns(self, entity_id: str, events: list[PatternEvent]) -> list[Pattern]:
        """Detect temporal patterns from one complete, shared metadata window."""
        patterns = []
        by_action = defaultdict(list)
        for event in events:
            by_action[event.action].append(event)
        for action, action_events in by_action.items():
            if len(action_events) < self.MIN_OCCURRENCES:
                continue
            # Weekday/weekend comparisons use negative as well as positive
            # evidence. Keep both groups, not just the matching occurrences.
            evidence = tuple(event.dependency for event in action_events)
            for detector in (self._detect_daily_pattern, self._detect_weekday_pattern,
                             self._detect_weekend_pattern):
                pattern = detector(entity_id, action, action_events)
                if pattern is not None:
                    pattern.evidence = evidence
                    patterns.append(pattern)
        return patterns

    def _detect_daily_pattern(
        self,
        entity_id: str,
        action: str,
        events: list[PatternEvent]
    ) -> Pattern | None:
        """Detectar si hay un patron de hora consistente"""
        if len(events) < self.MIN_OCCURRENCES:
            return None
        
        # Extraer horas (en minutos desde medianoche para mas precision)
        times_minutes = [e.hour * 60 + e.minute for e in events]
        
        # Calcular estadisticas
        mean_minutes = statistics.mean(times_minutes)
        
        if len(times_minutes) >= 2:
            std_minutes = statistics.stdev(times_minutes)
        else:
            std_minutes = 60  # Default alto
        
        # Si la desviacion es menor que la tolerancia, hay patron
        if std_minutes <= self.TIME_TOLERANCE_MINUTES:
            mean_hour = int(mean_minutes // 60)
            mean_minute = int(mean_minutes % 60)
            
            # Calcular confianza basada en consistencia y cantidad
            consistency_score = 1 - (std_minutes / self.TIME_TOLERANCE_MINUTES)
            quantity_score = min(1.0, len(events) / 20)  # Normalizar a 20 eventos
            confidence = (consistency_score * 0.7) + (quantity_score * 0.3)
            
            # Verificar que ocurre en multiples dias
            unique_days = len(set(e.datetime.date() for e in events))
            if unique_days < 3:
                confidence *= 0.5  # Penalizar si son pocos dias
            
            if confidence >= self.MIN_CONFIDENCE:
                description = self._format_time_description(
                    entity_id, action, mean_hour, mean_minute, "todos los dias"
                )
                
                return Pattern(
                    pattern_type=PatternType.DAILY_TIME,
                    entity_id=entity_id,
                    action=action,
                    confidence=round(confidence, 2),
                    occurrences=len(events),
                    description=description,
                    hour=mean_hour,
                    minute=mean_minute,
                    hour_std=round(std_minutes / 60, 2),
                    weekdays=list(range(7))
                )
        
        return None
    
    def _detect_weekday_pattern(
        self,
        entity_id: str,
        action: str,
        events: list[PatternEvent]
    ) -> Pattern | None:
        """Detectar patron solo en dias de semana"""
        weekday_events = [e for e in events if not e.is_weekend]
        
        if len(weekday_events) < self.MIN_OCCURRENCES:
            return None
        
        times_minutes = [e.hour * 60 + e.minute for e in weekday_events]
        mean_minutes = statistics.mean(times_minutes)
        
        if len(times_minutes) >= 2:
            std_minutes = statistics.stdev(times_minutes)
        else:
            return None
        
        if std_minutes <= self.TIME_TOLERANCE_MINUTES:
            mean_hour = int(mean_minutes // 60)
            mean_minute = int(mean_minutes % 60)
            
            consistency_score = 1 - (std_minutes / self.TIME_TOLERANCE_MINUTES)
            quantity_score = min(1.0, len(weekday_events) / 15)
            confidence = (consistency_score * 0.7) + (quantity_score * 0.3)
            
            # Verificar que hay menos eventos en fines de semana
            weekend_events = [e for e in events if e.is_weekend]
            if len(weekend_events) > len(weekday_events) * 0.5:
                return None  # No es especifico de dias de semana
            
            if confidence >= self.MIN_CONFIDENCE:
                description = self._format_time_description(
                    entity_id, action, mean_hour, mean_minute, "dias de semana"
                )
                
                return Pattern(
                    pattern_type=PatternType.WEEKDAY_TIME,
                    entity_id=entity_id,
                    action=action,
                    confidence=round(confidence, 2),
                    occurrences=len(weekday_events),
                    description=description,
                    hour=mean_hour,
                    minute=mean_minute,
                    hour_std=round(std_minutes / 60, 2),
                    weekdays=[0, 1, 2, 3, 4]  # Lunes a viernes
                )
        
        return None
    
    def _detect_weekend_pattern(
        self,
        entity_id: str,
        action: str,
        events: list[PatternEvent]
    ) -> Pattern | None:
        """Detectar patron solo en fines de semana"""
        weekend_events = [e for e in events if e.is_weekend]
        
        if len(weekend_events) < 3:  # Menos eventos en fines de semana
            return None
        
        times_minutes = [e.hour * 60 + e.minute for e in weekend_events]
        mean_minutes = statistics.mean(times_minutes)
        
        if len(times_minutes) >= 2:
            std_minutes = statistics.stdev(times_minutes)
        else:
            return None
        
        if std_minutes <= self.TIME_TOLERANCE_MINUTES * 1.5:  # Mas tolerancia
            mean_hour = int(mean_minutes // 60)
            mean_minute = int(mean_minutes % 60)
            
            # Verificar que la hora es diferente a dias de semana
            weekday_events = [e for e in events if not e.is_weekend]
            if weekday_events:
                weekday_mean = statistics.mean([e.hour * 60 + e.minute for e in weekday_events])
                hour_diff = abs(mean_minutes - weekday_mean) / 60
                if hour_diff < 1:  # Menos de 1 hora de diferencia
                    return None  # No es un patron diferente
            
            consistency_score = 1 - (std_minutes / (self.TIME_TOLERANCE_MINUTES * 1.5))
            quantity_score = min(1.0, len(weekend_events) / 8)
            confidence = (consistency_score * 0.7) + (quantity_score * 0.3)
            
            if confidence >= self.MIN_CONFIDENCE:
                description = self._format_time_description(
                    entity_id, action, mean_hour, mean_minute, "fines de semana"
                )
                
                return Pattern(
                    pattern_type=PatternType.WEEKEND_TIME,
                    entity_id=entity_id,
                    action=action,
                    confidence=round(confidence, 2),
                    occurrences=len(weekend_events),
                    description=description,
                    hour=mean_hour,
                    minute=mean_minute,
                    hour_std=round(std_minutes / 60, 2),
                    weekdays=[5, 6]  # Sabado y domingo
                )
        
        return None
    
    def _analyze_sequences(self, events: tuple[PatternEvent, ...]) -> list[Pattern]:
        """Count distinct triggers, using the same bounded observation as time patterns."""
        by_trigger = defaultdict(list)
        for event in events:
            by_trigger[event.entity_id, event.action].append(event.dependency)
        # Each trigger counts at most once for a target action, even if that
        # action repeats in its window. All matching targets remain dependencies.
        pairs = {}
        comparisons = 0
        for index, first in enumerate(events):
            later = index + 1
            while later < len(events):
                second = events[later]
                delay = second.timestamp - first.timestamp
                if delay > self.SEQUENCE_WINDOW_MINUTES * 60:
                    break
                comparisons += 1
                if comparisons > self.MAX_SEQUENCE_COMPARISONS:
                    logger.warning("Sequence analysis deferred: activity window capacity")
                    return []
                if delay > 0 and first.entity_id != second.entity_id:
                    key = (first.entity_id, first.action, second.entity_id, second.action)
                    triggers, targets = pairs.setdefault(key, (set(), {}))
                    triggers.add(first.dependency.event_id)
                    targets[second.dependency.event_id] = second.dependency
                later += 1
        patterns = []
        candidates = sorted(pairs.items(), key=lambda item: (-len(item[1][0]), item[0]))
        for key, (triggers, targets) in candidates:
            trigger_entity, trigger_action, target_entity, target_action = key
            count = len(triggers)
            if count < self.MIN_OCCURRENCES:
                continue
            trigger_evidence = by_trigger[trigger_entity, trigger_action]
            confidence = count / len(trigger_evidence)
            if confidence < self.MIN_CONFIDENCE:
                continue
            patterns.append(Pattern(
                pattern_type=PatternType.SEQUENCE,
                entity_id=target_entity,
                action=target_action,
                confidence=round(confidence, 2),
                occurrences=count,
                description=(
                    f"Cuando se ejecuta {trigger_action} en {self._friendly_entity(trigger_entity)}, "
                    f"generalmente se ejecuta {target_action} en {self._friendly_entity(target_entity)}"
                ),
                trigger_entity=trigger_entity,
                trigger_action=trigger_action,
                delay_seconds=self.SEQUENCE_WINDOW_MINUTES * 60 / 2,
                evidence=tuple(dict.fromkeys((*trigger_evidence, *targets.values()))),
            ))
            if len(patterns) == 50:
                break
        return patterns

    def _format_time_description(
        self,
        entity_id: str,
        action: str,
        hour: int,
        minute: int,
        when: str
    ) -> str:
        """Formatear descripcion de patron temporal"""
        entity_name = self._friendly_entity(entity_id)
        action_name = self._friendly_action(action)
        time_str = f"{hour:02d}:{minute:02d}"
        
        return f"{action_name} {entity_name} aproximadamente a las {time_str} {when}"
    
    def _friendly_entity(self, entity_id: str) -> str:
        """Convertir entity_id a nombre amigable"""
        # light.living_room -> luz del living room
        parts = entity_id.split(".")
        if len(parts) == 2:
            domain, name = parts
            name = name.replace("_", " ")
            
            domain_names = {
                "light": "luz",
                "switch": "interruptor",
                "climate": "aire acondicionado",
                "cover": "persiana",
                "fan": "ventilador",
                "media_player": "reproductor"
            }
            
            domain_friendly = domain_names.get(domain, domain)
            return f"{domain_friendly} de {name}"
        
        return entity_id
    
    def _friendly_action(self, action: str) -> str:
        """Convertir accion a verbo amigable"""
        action_names = {
            "turn_on": "Encender",
            "turn_off": "Apagar",
            "toggle": "Alternar",
            "open_cover": "Abrir",
            "close_cover": "Cerrar",
            "set_temperature": "Ajustar temperatura de",
            "set_hvac_mode": "Cambiar modo de"
        }
        return action_names.get(action, action.replace("_", " ").title())
    
    def get_patterns_for_entity(self, entity_id: str, days: int = 30) -> list[Pattern]:
        """Obtener patrones especificos de una entidad"""
        all_patterns = self.analyze_all(days)
        return [p for p in all_patterns if p.entity_id == entity_id]
    
    def get_actionable_patterns(self, min_confidence: float = 0.7) -> list[Pattern]:
        """Obtener patrones con alta confianza listos para sugerir"""
        all_patterns = self.analyze_all()
        return [p for p in all_patterns if p.confidence >= min_confidence]
