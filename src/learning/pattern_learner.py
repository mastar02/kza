"""
Pattern Learner - Automatizaciones Aprendidas
Observa patrones de comportamiento y sugiere rutinas automáticamente.

"Noto que apagas las luces del salón a las 11pm cada noche.
¿Quieres que cree una rutina para hacerlo automáticamente?"
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional, Callable, Any
from collections import defaultdict
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Registro de una acción del usuario"""
    action_type: str           # "light_on", "light_off", "climate_set", etc.
    entity_id: str             # "light.sala", "climate.casa"
    timestamp: datetime
    user_id: Optional[str] = None
    day_of_week: int = 0       # 0=lunes, 6=domingo
    hour: int = 0
    minute: int = 0
    data: dict = field(default_factory=dict)  # Datos adicionales (temperatura, brillo, etc.)
    trigger: str = "voice"     # "voice", "app", "physical", "routine"


@dataclass
class DetectedPattern:
    """Patrón detectado en el comportamiento"""
    pattern_id: str
    action_type: str
    entity_id: str
    confidence: float          # 0-1
    occurrences: int           # Número de veces observado

    # Patrón temporal
    typical_time: dtime        # Hora típica
    time_variance_minutes: float  # Varianza en minutos
    days_of_week: list[int]    # Días en que ocurre

    # Contexto
    user_id: Optional[str] = None
    typical_data: dict = field(default_factory=dict)

    # Estado
    suggested: bool = False    # Ya se sugirió al usuario
    accepted: bool = False     # Usuario aceptó crear rutina
    dismissed: bool = False    # Usuario rechazó la sugerencia

    def __str__(self):
        days = ["L", "M", "X", "J", "V", "S", "D"]
        day_str = "".join(days[d] for d in sorted(self.days_of_week))
        return (
            f"{self.action_type} {self.entity_id} "
            f"a las {self.typical_time.strftime('%H:%M')} ({day_str}) "
            f"[{self.confidence:.0%}]"
        )


@dataclass
class RoutineSuggestion:
    """Sugerencia de rutina para el usuario"""
    suggestion_id: str
    pattern: DetectedPattern
    routine_name: str
    description: str
    trigger: dict              # Trigger propuesto
    actions: list[dict]        # Acciones propuestas
    created_at: datetime = None


class PatternLearner:
    """
    Sistema de aprendizaje de patrones para sugerir automatizaciones.

    Características:
    - Observa acciones del usuario en el tiempo
    - Detecta patrones repetitivos (misma hora, mismos días)
    - Genera sugerencias de rutinas
    - Aprende de aceptaciones/rechazos
    - Respeta privacidad (todo local)
    """

    # Configuración de detección
    MIN_OCCURRENCES = 5         # Mínimo de ocurrencias para detectar patrón
    MAX_TIME_VARIANCE = 30      # Máxima varianza en minutos para considerar patrón
    MIN_CONFIDENCE = 0.7        # Confianza mínima para sugerir
    HISTORY_DAYS = 30           # Días de historial a mantener

    def __init__(
        self,
        data_dir: str = None,
        min_occurrences: int = None,
        max_time_variance: int = None
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("data/patterns")
        self.min_occurrences = min_occurrences or self.MIN_OCCURRENCES
        self.max_time_variance = max_time_variance or self.MAX_TIME_VARIANCE

        # Historial de acciones
        self._action_history: list[ActionRecord] = []

        # Patrones detectados
        self._patterns: dict[str, DetectedPattern] = {}

        # Sugerencias pendientes
        self._pending_suggestions: dict[str, RoutineSuggestion] = {}

        # Patrones rechazados (para no volver a sugerir)
        self._dismissed_patterns: set[str] = set()

        # Callbacks
        self._on_pattern_detected: Optional[Callable] = None
        self._on_suggestion_ready: Optional[Callable] = None

        # Cargar datos si existen
        self._load_data()

    def _load_data(self):
        """Cargar datos persistidos"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        history_file = self.data_dir / "action_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self._action_history = [
                        ActionRecord(**{**r, "timestamp": datetime.fromisoformat(r["timestamp"])})
                        for r in data
                    ]
                logger.info(f"Cargadas {len(self._action_history)} acciones del historial")
            except Exception as e:
                logger.warning(f"Error cargando historial: {e}")

        dismissed_file = self.data_dir / "dismissed_patterns.json"
        if dismissed_file.exists():
            try:
                with open(dismissed_file) as f:
                    self._dismissed_patterns = set(json.load(f))
            except Exception as e:
                logger.warning(f"Error cargando patrones rechazados: {e}")

    def _save_data(self):
        """Persistir datos"""
        try:
            # Guardar historial (solo últimos N días)
            cutoff = datetime.now() - timedelta(days=self.HISTORY_DAYS)
            recent_history = [
                r for r in self._action_history
                if r.timestamp > cutoff
            ]

            history_file = self.data_dir / "action_history.json"
            with open(history_file, 'w') as f:
                json.dump([
                    {**r.__dict__, "timestamp": r.timestamp.isoformat()}
                    for r in recent_history
                ], f, indent=2)

            # Guardar patrones rechazados
            dismissed_file = self.data_dir / "dismissed_patterns.json"
            with open(dismissed_file, 'w') as f:
                json.dump(list(self._dismissed_patterns), f)

        except Exception as e:
            logger.error(f"Error guardando datos: {e}")

    def record_action(
        self,
        action_type: str,
        entity_id: str,
        user_id: str = None,
        data: dict = None,
        trigger: str = "voice"
    ):
        """
        Registrar una acción del usuario.

        Args:
            action_type: Tipo de acción ("light_on", "climate_set", etc.)
            entity_id: Entidad afectada
            user_id: Usuario que ejecutó la acción
            data: Datos adicionales (temperatura, brillo, etc.)
            trigger: Origen de la acción ("voice", "app", "physical")
        """
        now = datetime.now()

        record = ActionRecord(
            action_type=action_type,
            entity_id=entity_id,
            timestamp=now,
            user_id=user_id,
            day_of_week=now.weekday(),
            hour=now.hour,
            minute=now.minute,
            data=data or {},
            trigger=trigger
        )

        self._action_history.append(record)

        logger.debug(f"Acción registrada: {action_type} {entity_id} por {user_id}")

        # Guardar periódicamente
        if len(self._action_history) % 10 == 0:
            self._save_data()

    async def analyze_patterns(self) -> list[DetectedPattern]:
        """
        Analizar historial y detectar patrones.

        Returns:
            Lista de nuevos patrones detectados
        """
        new_patterns = []

        # Agrupar acciones por (tipo, entidad, usuario)
        grouped = defaultdict(list)
        for record in self._action_history:
            key = (record.action_type, record.entity_id, record.user_id)
            grouped[key].append(record)

        for (action_type, entity_id, user_id), records in grouped.items():
            # Necesitamos suficientes ocurrencias
            if len(records) < self.min_occurrences:
                continue

            # Analizar distribución temporal
            pattern = self._analyze_temporal_pattern(
                action_type, entity_id, user_id, records
            )

            if pattern and pattern.confidence >= self.MIN_CONFIDENCE:
                # Verificar si es nuevo
                if pattern.pattern_id not in self._patterns:
                    if pattern.pattern_id not in self._dismissed_patterns:
                        self._patterns[pattern.pattern_id] = pattern
                        new_patterns.append(pattern)

                        if self._on_pattern_detected:
                            self._on_pattern_detected(pattern)

                        logger.info(f"Nuevo patrón detectado: {pattern}")

        return new_patterns

    def _analyze_temporal_pattern(
        self,
        action_type: str,
        entity_id: str,
        user_id: str,
        records: list[ActionRecord]
    ) -> Optional[DetectedPattern]:
        """Analizar patrón temporal en las acciones"""

        # Convertir horas a minutos desde medianoche
        times_minutes = [r.hour * 60 + r.minute for r in records]

        # Calcular estadísticas
        mean_time = statistics.mean(times_minutes)
        try:
            stdev_time = statistics.stdev(times_minutes)
        except statistics.StatisticsError:
            stdev_time = 0

        # Si la varianza es muy alta, no es un patrón claro
        if stdev_time > self.max_time_variance:
            return None

        # Calcular días de la semana más comunes
        day_counts = defaultdict(int)
        for r in records:
            day_counts[r.day_of_week] += 1

        # Días con al menos 20% de las ocurrencias
        threshold = len(records) * 0.2
        common_days = [day for day, count in day_counts.items() if count >= threshold]

        if not common_days:
            return None

        # Calcular confianza basada en consistencia
        # Mayor consistencia = mayor confianza
        time_consistency = 1 - (stdev_time / self.max_time_variance)
        day_consistency = len(common_days) / 7  # Más días = menos específico
        occurrence_factor = min(len(records) / 20, 1.0)  # Más ocurrencias = más confianza

        confidence = (time_consistency * 0.5 + (1 - day_consistency) * 0.3 + occurrence_factor * 0.2)

        # Crear patrón
        typical_hour = int(mean_time // 60)
        typical_minute = int(mean_time % 60)

        pattern_id = f"{action_type}_{entity_id}_{typical_hour:02d}{typical_minute:02d}"
        if user_id:
            pattern_id += f"_{user_id}"

        # Datos típicos (más común)
        typical_data = {}
        if records[0].data:
            # Encontrar valores más comunes
            for key in records[0].data.keys():
                values = [r.data.get(key) for r in records if r.data.get(key) is not None]
                if values:
                    # Usar el más común o promedio si es numérico
                    if all(isinstance(v, (int, float)) for v in values):
                        typical_data[key] = round(statistics.mean(values), 1)
                    else:
                        typical_data[key] = max(set(values), key=values.count)

        return DetectedPattern(
            pattern_id=pattern_id,
            action_type=action_type,
            entity_id=entity_id,
            confidence=confidence,
            occurrences=len(records),
            typical_time=dtime(typical_hour, typical_minute),
            time_variance_minutes=stdev_time,
            days_of_week=sorted(common_days),
            user_id=user_id,
            typical_data=typical_data
        )

    def generate_suggestion(self, pattern: DetectedPattern) -> RoutineSuggestion:
        """Generar sugerencia de rutina a partir de un patrón"""
        import uuid

        # Generar nombre amigable
        entity_name = pattern.entity_id.split(".")[-1].replace("_", " ")
        action_name = self._get_action_name(pattern.action_type)
        time_str = pattern.typical_time.strftime("%H:%M")

        routine_name = f"{action_name} {entity_name}"

        # Descripción
        days_str = self._get_days_description(pattern.days_of_week)
        description = f"{action_name} {entity_name} a las {time_str} {days_str}"

        # Trigger
        trigger = {
            "type": "time",
            "config": {
                "at": pattern.typical_time.strftime("%H:%M:00"),
                "days": self._days_to_names(pattern.days_of_week)
            }
        }

        # Acción
        action = self._pattern_to_action(pattern)

        suggestion = RoutineSuggestion(
            suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
            pattern=pattern,
            routine_name=routine_name,
            description=description,
            trigger=trigger,
            actions=[action],
            created_at=datetime.now()
        )

        self._pending_suggestions[suggestion.suggestion_id] = suggestion

        return suggestion

    def _get_action_name(self, action_type: str) -> str:
        """Obtener nombre legible de la acción"""
        names = {
            "light_on": "Encender",
            "light_off": "Apagar",
            "climate_set": "Ajustar clima",
            "scene_activate": "Activar escena",
            "switch_on": "Encender",
            "switch_off": "Apagar",
            "media_play": "Reproducir",
            "media_pause": "Pausar",
        }
        return names.get(action_type, action_type.replace("_", " ").title())

    def _get_days_description(self, days: list[int]) -> str:
        """Obtener descripción de los días"""
        if len(days) == 7:
            return "todos los días"
        if days == [0, 1, 2, 3, 4]:
            return "de lunes a viernes"
        if days == [5, 6]:
            return "los fines de semana"

        day_names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        return "los " + ", ".join(day_names[d] for d in days)

    def _days_to_names(self, days: list[int]) -> list[str]:
        """Convertir días a nombres cortos"""
        names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        return [names[d] for d in days]

    def _pattern_to_action(self, pattern: DetectedPattern) -> dict:
        """Convertir patrón a acción de rutina"""
        domain = pattern.entity_id.split(".")[0]

        action = {
            "type": "ha_service",
            "domain": domain,
            "entity_id": pattern.entity_id
        }

        if "on" in pattern.action_type:
            action["service"] = "turn_on"
        elif "off" in pattern.action_type:
            action["service"] = "turn_off"
        elif "set" in pattern.action_type:
            if domain == "climate":
                action["service"] = "set_temperature"
            else:
                action["service"] = "set"

        if pattern.typical_data:
            action["data"] = pattern.typical_data

        return action

    def get_suggestion_text(self, suggestion: RoutineSuggestion) -> str:
        """Generar texto para preguntar al usuario"""
        pattern = suggestion.pattern

        entity_name = pattern.entity_id.split(".")[-1].replace("_", " ")
        action_name = self._get_action_name(pattern.action_type).lower()
        time_str = pattern.typical_time.strftime("%H:%M")
        days_str = self._get_days_description(pattern.days_of_week)

        text = (
            f"He notado que sueles {action_name} {entity_name} "
            f"alrededor de las {time_str} {days_str}. "
            f"¿Quieres que cree una rutina para hacerlo automáticamente?"
        )

        return text

    def accept_suggestion(self, suggestion_id: str) -> Optional[dict]:
        """
        Usuario acepta la sugerencia.

        Returns:
            Diccionario de rutina para crear
        """
        suggestion = self._pending_suggestions.get(suggestion_id)
        if not suggestion:
            return None

        suggestion.pattern.accepted = True
        suggestion.pattern.suggested = True

        # Remover de pendientes
        del self._pending_suggestions[suggestion_id]

        # Retornar rutina para crear
        return {
            "name": suggestion.routine_name,
            "triggers": [suggestion.trigger],
            "actions": suggestion.actions,
            "created_by": "pattern_learner",
            "source_pattern": suggestion.pattern.pattern_id
        }

    def dismiss_suggestion(self, suggestion_id: str):
        """Usuario rechaza la sugerencia"""
        suggestion = self._pending_suggestions.get(suggestion_id)
        if suggestion:
            suggestion.pattern.dismissed = True
            suggestion.pattern.suggested = True

            # Agregar a rechazados para no volver a sugerir
            self._dismissed_patterns.add(suggestion.pattern.pattern_id)

            del self._pending_suggestions[suggestion_id]

            self._save_data()

    def get_pending_suggestions(self) -> list[RoutineSuggestion]:
        """Obtener sugerencias pendientes de respuesta"""
        return list(self._pending_suggestions.values())

    async def run_analysis_loop(self, interval_hours: float = 24):
        """Loop de análisis periódico"""
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)
                patterns = await self.analyze_patterns()

                for pattern in patterns:
                    if not pattern.suggested:
                        suggestion = self.generate_suggestion(pattern)

                        if self._on_suggestion_ready:
                            self._on_suggestion_ready(suggestion)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en análisis de patrones: {e}")

    # ==================== Callbacks ====================

    def on_pattern_detected(self, callback: Callable[[DetectedPattern], None]):
        """Registrar callback para nuevos patrones"""
        self._on_pattern_detected = callback

    def on_suggestion_ready(self, callback: Callable[[RoutineSuggestion], None]):
        """Registrar callback para sugerencias"""
        self._on_suggestion_ready = callback

    # ==================== Estado ====================

    def get_status(self) -> dict:
        """Obtener estado del sistema"""
        return {
            "total_actions_recorded": len(self._action_history),
            "patterns_detected": len(self._patterns),
            "pending_suggestions": len(self._pending_suggestions),
            "dismissed_patterns": len(self._dismissed_patterns)
        }

    def get_patterns(self) -> list[dict]:
        """Obtener patrones detectados"""
        return [
            {
                "pattern_id": p.pattern_id,
                "description": str(p),
                "confidence": p.confidence,
                "occurrences": p.occurrences,
                "suggested": p.suggested,
                "accepted": p.accepted
            }
            for p in self._patterns.values()
        ]
