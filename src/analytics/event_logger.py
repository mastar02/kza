"""
Event Logger Module
Registra todos los eventos de domotica para analisis de patrones.
"""

import sqlite3
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Tipos de eventos registrados"""
    COMMAND = "command"           # Comando de voz ejecutado
    AUTOMATION = "automation"     # Automatizacion ejecutada
    SCHEDULE = "schedule"         # Evento programado
    SENSOR = "sensor"             # Cambio de sensor
    STATE_CHANGE = "state_change" # Cambio de estado manual (desde HA)


@dataclass
class Event:
    """Un evento de domotica"""
    timestamp: float
    event_type: EventType
    entity_id: str
    action: str                   # turn_on, turn_off, set_temperature, etc.
    user_id: str | None = None
    user_name: str | None = None
    trigger_phrase: str | None = None  # Frase original del usuario
    context: dict = field(default_factory=dict)  # hora, dia, clima, etc.
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)
    
    @property
    def hour(self) -> int:
        return self.datetime.hour
    
    @property
    def minute(self) -> int:
        return self.datetime.minute
    
    @property
    def weekday(self) -> int:
        """0=Lunes, 6=Domingo"""
        return self.datetime.weekday()
    
    @property
    def weekday_name(self) -> str:
        days = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
        return days[self.weekday]
    
    @property
    def is_weekend(self) -> bool:
        return self.weekday >= 5
    
    @property
    def time_of_day(self) -> str:
        """Clasificar momento del dia"""
        hour = self.hour
        if 5 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "noon"
        elif 14 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 21:
            return "evening"
        else:
            return "night"


class EventLogger:
    """
    Registra eventos de domotica en SQLite para analisis posterior.
    
    Features:
    - Almacenamiento persistente en SQLite
    - Contexto automatico (hora, dia, etc.)
    - Queries eficientes por entidad/tiempo
    - Limpieza automatica de eventos antiguos
    """
    
    def __init__(
        self,
        db_path: str = "./data/events.db",
        retention_days: int = 90,
        context_provider=None
    ):
        """
        Args:
            db_path: Ruta a la base de datos SQLite
            retention_days: Dias a retener eventos (default 90)
            context_provider: Callback para obtener contexto adicional
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.context_provider = context_provider
        
        self._init_db()
    
    def _init_db(self):
        """Inicializar esquema de base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    user_name TEXT,
                    trigger_phrase TEXT,
                    context_json TEXT,
                    -- Campos pre-calculados para queries rapidas
                    hour INTEGER,
                    minute INTEGER,
                    weekday INTEGER,
                    is_weekend INTEGER,
                    time_of_day TEXT
                )
            """)
            
            # Indices para queries comunes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON events(entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hour ON events(hour)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_weekday ON events(weekday)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON events(entity_id, action)")
            
            conn.commit()
        
        logger.info(f"EventLogger inicializado: {self.db_path}")
    
    def log(
        self,
        entity_id: str,
        action: str,
        event_type: EventType = EventType.COMMAND,
        user_id: str = None,
        user_name: str = None,
        trigger_phrase: str = None,
        extra_context: dict = None
    ) -> Event:
        """
        Registrar un evento.
        
        Args:
            entity_id: ID de la entidad (light.living_room)
            action: Accion ejecutada (turn_on, turn_off, etc.)
            event_type: Tipo de evento
            user_id: ID del usuario (opcional)
            user_name: Nombre del usuario (opcional)
            trigger_phrase: Frase que disparo el comando (opcional)
            extra_context: Contexto adicional (opcional)
        
        Returns:
            Event registrado
        """
        timestamp = time.time()
        
        # Construir contexto
        context = self._build_context(extra_context)
        
        event = Event(
            timestamp=timestamp,
            event_type=event_type,
            entity_id=entity_id,
            action=action,
            user_id=user_id,
            user_name=user_name,
            trigger_phrase=trigger_phrase,
            context=context
        )
        
        # Guardar en DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events (
                    timestamp, event_type, entity_id, action,
                    user_id, user_name, trigger_phrase, context_json,
                    hour, minute, weekday, is_weekend, time_of_day
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.event_type.value,
                event.entity_id,
                event.action,
                event.user_id,
                event.user_name,
                event.trigger_phrase,
                json.dumps(context),
                event.hour,
                event.minute,
                event.weekday,
                1 if event.is_weekend else 0,
                event.time_of_day
            ))
            conn.commit()
        
        logger.debug(f"Event logged: {entity_id} {action} at {event.datetime}")
        return event
    
    def _build_context(self, extra: dict = None) -> dict:
        """Construir contexto del evento"""
        context = {
            "logged_at": datetime.now().isoformat()
        }
        
        # Obtener contexto del provider si existe
        if self.context_provider:
            try:
                provider_context = self.context_provider()
                if provider_context:
                    context.update(provider_context)
            except Exception as e:
                logger.warning(f"Error obteniendo contexto: {e}")
        
        # Agregar contexto extra
        if extra:
            context.update(extra)
        
        return context
    
    def get_events(
        self,
        entity_id: str = None,
        action: str = None,
        user_id: str = None,
        start_time: float = None,
        end_time: float = None,
        hour: int = None,
        weekday: int = None,
        limit: int = 1000
    ) -> list[Event]:
        """
        Obtener eventos con filtros.
        
        Args:
            entity_id: Filtrar por entidad
            action: Filtrar por accion
            user_id: Filtrar por usuario
            start_time: Desde timestamp
            end_time: Hasta timestamp
            hour: Filtrar por hora del dia
            weekday: Filtrar por dia de semana (0=Lunes)
            limit: Maximo de resultados
        
        Returns:
            Lista de eventos
        """
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if hour is not None:
            query += " AND hour = ?"
            params.append(hour)
        
        if weekday is not None:
            query += " AND weekday = ?"
            params.append(weekday)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                event = Event(
                    timestamp=row["timestamp"],
                    event_type=EventType(row["event_type"]),
                    entity_id=row["entity_id"],
                    action=row["action"],
                    user_id=row["user_id"],
                    user_name=row["user_name"],
                    trigger_phrase=row["trigger_phrase"],
                    context=json.loads(row["context_json"]) if row["context_json"] else {}
                )
                events.append(event)
        
        return events
    
    def get_entity_history(
        self,
        entity_id: str,
        days: int = 30
    ) -> list[Event]:
        """Obtener historial de una entidad"""
        start_time = time.time() - (days * 24 * 3600)
        return self.get_events(entity_id=entity_id, start_time=start_time)
    
    def get_hourly_distribution(
        self,
        entity_id: str,
        action: str = None,
        days: int = 30
    ) -> dict[int, int]:
        """
        Obtener distribucion de eventos por hora.
        
        Returns:
            {hora: cantidad_eventos}
        """
        start_time = time.time() - (days * 24 * 3600)
        
        query = """
            SELECT hour, COUNT(*) as count
            FROM events
            WHERE entity_id = ? AND timestamp >= ?
        """
        params = [entity_id, start_time]
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        query += " GROUP BY hour ORDER BY hour"
        
        distribution = {h: 0 for h in range(24)}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor:
                distribution[row[0]] = row[1]
        
        return distribution
    
    def get_weekday_distribution(
        self,
        entity_id: str,
        action: str = None,
        days: int = 30
    ) -> dict[int, int]:
        """
        Obtener distribucion de eventos por dia de semana.
        
        Returns:
            {weekday: cantidad_eventos} donde 0=Lunes
        """
        start_time = time.time() - (days * 24 * 3600)
        
        query = """
            SELECT weekday, COUNT(*) as count
            FROM events
            WHERE entity_id = ? AND timestamp >= ?
        """
        params = [entity_id, start_time]
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        query += " GROUP BY weekday ORDER BY weekday"
        
        distribution = {d: 0 for d in range(7)}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor:
                distribution[row[0]] = row[1]
        
        return distribution
    
    def get_sequences(
        self,
        window_minutes: int = 10,
        min_occurrences: int = 3,
        days: int = 30
    ) -> list[tuple[str, str, int]]:
        """
        Encontrar secuencias frecuentes de eventos (A -> B).
        
        Args:
            window_minutes: Ventana de tiempo para considerar secuencia
            min_occurrences: Minimo de veces que debe ocurrir
            days: Dias a analizar
        
        Returns:
            Lista de (entity_a, entity_b, count)
        """
        start_time = time.time() - (days * 24 * 3600)
        window_seconds = window_minutes * 60
        
        # Query para encontrar pares de eventos cercanos
        query = """
            SELECT 
                e1.entity_id || ':' || e1.action as first_event,
                e2.entity_id || ':' || e2.action as second_event,
                COUNT(*) as occurrences
            FROM events e1
            JOIN events e2 ON 
                e2.timestamp > e1.timestamp 
                AND e2.timestamp <= e1.timestamp + ?
                AND e1.id != e2.id
            WHERE e1.timestamp >= ?
            GROUP BY first_event, second_event
            HAVING occurrences >= ?
            ORDER BY occurrences DESC
            LIMIT 50
        """
        
        sequences = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (window_seconds, start_time, min_occurrences))
            for row in cursor:
                sequences.append((row[0], row[1], row[2]))
        
        return sequences
    
    def get_stats(self) -> dict:
        """Obtener estadisticas generales"""
        with sqlite3.connect(self.db_path) as conn:
            # Total eventos
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            
            # Eventos ultimos 7 dias
            week_ago = time.time() - (7 * 24 * 3600)
            last_week = conn.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp >= ?",
                (week_ago,)
            ).fetchone()[0]
            
            # Entidades unicas
            entities = conn.execute(
                "SELECT COUNT(DISTINCT entity_id) FROM events"
            ).fetchone()[0]
            
            # Top 5 entidades
            top_entities = conn.execute("""
                SELECT entity_id, COUNT(*) as count
                FROM events
                GROUP BY entity_id
                ORDER BY count DESC
                LIMIT 5
            """).fetchall()
            
            # Rango de fechas
            date_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM events
            """).fetchone()
        
        return {
            "total_events": total,
            "events_last_week": last_week,
            "unique_entities": entities,
            "top_entities": [{"entity": e[0], "count": e[1]} for e in top_entities],
            "first_event": datetime.fromtimestamp(date_range[0]).isoformat() if date_range[0] else None,
            "last_event": datetime.fromtimestamp(date_range[1]).isoformat() if date_range[1] else None,
            "db_path": str(self.db_path)
        }
    
    def cleanup_old_events(self):
        """Eliminar eventos mas antiguos que retention_days"""
        cutoff = time.time() - (self.retention_days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff,)
            )
            deleted = result.rowcount
            conn.commit()
        
        if deleted > 0:
            logger.info(f"Eliminados {deleted} eventos antiguos")
        
        return deleted
