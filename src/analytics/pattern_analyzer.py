"""
Pattern Analyzer Module
Detecta patrones de uso para sugerir automatizaciones.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
import math

from .event_logger import EventLogger, Event

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
            "user_id": self.user_id
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
        patterns = []
        
        # Obtener entidades unicas
        stats = self.event_logger.get_stats()
        entities = [e["entity"] for e in stats.get("top_entities", [])]
        
        # Agregar mas entidades si hay pocas
        if len(entities) < 10:
            all_events = self.event_logger.get_events(limit=10000)
            entities = list(set(e.entity_id for e in all_events))
        
        logger.info(f"Analizando patrones para {len(entities)} entidades...")
        
        # Analizar cada entidad
        for entity_id in entities:
            # Patrones temporales
            time_patterns = self._analyze_time_patterns(entity_id, days)
            patterns.extend(time_patterns)
        
        # Patrones de secuencia (global)
        sequence_patterns = self._analyze_sequences(days)
        patterns.extend(sequence_patterns)
        
        # Filtrar por confianza minima y ordenar
        patterns = [p for p in patterns if p.confidence >= self.MIN_CONFIDENCE]
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        logger.info(f"Detectados {len(patterns)} patrones")
        return patterns
    
    def _analyze_time_patterns(self, entity_id: str, days: int) -> list[Pattern]:
        """Analizar patrones temporales para una entidad"""
        patterns = []
        
        events = self.event_logger.get_entity_history(entity_id, days)
        if len(events) < self.MIN_OCCURRENCES:
            return patterns
        
        # Agrupar por accion
        by_action = {}
        for event in events:
            if event.action not in by_action:
                by_action[event.action] = []
            by_action[event.action].append(event)
        
        for action, action_events in by_action.items():
            if len(action_events) < self.MIN_OCCURRENCES:
                continue
            
            # Analizar patron diario
            daily_pattern = self._detect_daily_pattern(entity_id, action, action_events)
            if daily_pattern:
                patterns.append(daily_pattern)
            
            # Analizar patron dias de semana vs fines de semana
            weekday_pattern = self._detect_weekday_pattern(entity_id, action, action_events)
            if weekday_pattern:
                patterns.append(weekday_pattern)
            
            weekend_pattern = self._detect_weekend_pattern(entity_id, action, action_events)
            if weekend_pattern:
                patterns.append(weekend_pattern)
        
        return patterns
    
    def _detect_daily_pattern(
        self,
        entity_id: str,
        action: str,
        events: list[Event]
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
        events: list[Event]
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
        events: list[Event]
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
    
    def _analyze_sequences(self, days: int) -> list[Pattern]:
        """Detectar patrones de secuencia (A -> B)"""
        patterns = []
        
        sequences = self.event_logger.get_sequences(
            window_minutes=self.SEQUENCE_WINDOW_MINUTES,
            min_occurrences=self.MIN_OCCURRENCES,
            days=days
        )
        
        for first_event, second_event, count in sequences:
            # Parsear eventos
            trigger_entity, trigger_action = first_event.rsplit(":", 1)
            target_entity, target_action = second_event.rsplit(":", 1)
            
            # Ignorar si es la misma entidad
            if trigger_entity == target_entity:
                continue
            
            # Calcular confianza
            # Obtener total de veces que ocurrio el trigger
            trigger_events = self.event_logger.get_events(
                entity_id=trigger_entity,
                action=trigger_action,
                limit=10000
            )
            
            if len(trigger_events) == 0:
                continue
            
            # Confianza = veces que ocurrio secuencia / veces que ocurrio trigger
            confidence = count / len(trigger_events)
            
            if confidence >= self.MIN_CONFIDENCE:
                description = (
                    f"Cuando se ejecuta {trigger_action} en {self._friendly_entity(trigger_entity)}, "
                    f"generalmente se ejecuta {target_action} en {self._friendly_entity(target_entity)}"
                )
                
                pattern = Pattern(
                    pattern_type=PatternType.SEQUENCE,
                    entity_id=target_entity,
                    action=target_action,
                    confidence=round(confidence, 2),
                    occurrences=count,
                    description=description,
                    trigger_entity=trigger_entity,
                    trigger_action=trigger_action,
                    delay_seconds=self.SEQUENCE_WINDOW_MINUTES * 60 / 2  # Estimado
                )
                patterns.append(pattern)
        
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
