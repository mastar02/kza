"""
Event Logger Module
Registra todos los eventos de domotica para analisis de patrones.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
import math
import os
import re
import threading
import sqlite3
import logging
import time
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
import json

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.analytics.event_storage import event_descriptor, event_sources
from src.analytics.event_evidence import (
    EventDependency, PatternHistory, PatternInputs, event_target_id, project_pattern_event,
)
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactLedger, MemorySource, sources_current

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
    
    sources: tuple[MemorySource, ...] = ()
    expires_at: float | None = None
    text_expires_at: float | None = None

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
        context_provider=None,
        *, source_policy: FactLedger | None = None, clock=None, io_worker=None,
        observation_timeout_s: float = .1,
        retention_notify=None,
    ):
        """
        Args:
            db_path: Ruta a la base de datos SQLite
            retention_days: Dias a retener eventos (default 90)
            context_provider: Callback para obtener contexto adicional
        """
        if type(retention_days) not in (int, float) or not math.isfinite(retention_days) or retention_days <= 0:
            raise ValueError("invalid_event_retention")
        if type(observation_timeout_s) not in (int, float) or not math.isfinite(observation_timeout_s) or observation_timeout_s <= 0:
            raise ValueError("invalid_event_observation_timeout")
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.context_provider = context_provider
        self.source_policy = source_policy
        self._clock = clock if clock is not None else (source_policy.clock if source_policy is not None else time.time)
        self.observation_timeout_s = observation_timeout_s
        self._retention_notify = retention_notify
        self._initializing = threading.RLock()
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='event-writer') if io_worker is None else None
        self._work = OwnedBlockingWork(worker=io_worker or self._offload)
        self._closing: asyncio.Task | None = None
        self._maintenance: asyncio.Task | None = None
        self._maintenance_loop = None
        self._changed = asyncio.Event()

    def initialize(self) -> None:
        """Create only policy tables, preserving the legacy events table."""
        with self._initializing:
            if self._work.status()["closed"]:
                raise RuntimeError("event_logger_closed")
            if self._initialized:
                return
            if self.source_policy is not None:
                self.source_policy.initialize()
            descriptor = event_descriptor(self.db_path)
            if self.source_policy is not None and self.source_policy.path != ":memory:" and descriptor["path"] == str(Path(self.source_policy.path).resolve()):
                raise ValueError("event_store_authority_location_conflict")
            self.db_path = Path(descriptor['path'])
            if self.source_policy is not None:
                self.source_policy.erasures.register_target(ErasureParticipant.EVENTS, 1, descriptor)
            self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(self.db_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            os.chmod(self.db_path, 0o600)
            self._init_db()
            self._initialized = True

    def _init_db(self):
        """Inicializar esquema de base de datos"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events_policy_v1 (
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
                    time_of_day TEXT,
                    sources_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    text_expires_at REAL
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_event_timestamp ON events_policy_v1(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_event_expiry ON events_policy_v1(expires_at)")
            conn.commit()

    def _unrevoked(self, encoded: str, raw: bool = False, *, committed: bool = False) -> bool:
        try:
            sources = event_sources(encoded, raw)
            if self.source_policy is None:
                return committed or not any(s.fact_dependencies for s in sources)
            if not self.source_policy.authority_available:
                raise RuntimeError("privacy_authority_unavailable")
            if committed:
                return not self.source_policy.sources_committed_unavailable(sources)
            return all(not self.source_policy.conversation_revoked(s.conversation_id)
                       and self.source_policy.facts_available(s.fact_dependencies) for s in sources)
        except (TypeError, ValueError, KeyError, AttributeError):
            if committed:
                raise ValueError('unknown_event_sources') from None
            return False

    def _text_available(self, encoded: str, expires_at) -> bool:
        """A stored raw deadline cannot override actual capture source expiry."""
        if type(expires_at) not in (int,float) or not math.isfinite(expires_at) or self._clock() >= expires_at:
            return False
        try:
            return sources_current(event_sources(encoded, True), self._clock())
        except (TypeError, ValueError, KeyError, AttributeError):
            return False

    def _metadata_current(self, captured_at, expires_at) -> bool:
        """Unknown metadata deadlines cannot authorize a policy event projection."""
        return all(type(v) in (int,float) and math.isfinite(v) for v in (captured_at,expires_at)) and captured_at <= self._clock() < expires_at

    @contextmanager
    def _connect(self):
        self.initialize()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.create_function("policy_now", 0, self._clock)
            db.create_function("event_unrevoked", 2, self._unrevoked)
            db.create_function("event_text_available", 2, self._text_available)
            db.create_function("event_metadata_current", 2, self._metadata_current)
            db.create_function("event_committed_unrevoked", 1, lambda encoded: self._unrevoked(encoded, committed=True))
            # All existing analytics SQL resolves this temporary view. Legacy
            # rows cannot enter retrieval, distributions, sequences or stats.
            db.execute("""CREATE TEMP VIEW events AS SELECT
                id, timestamp, event_type, entity_id, action, user_id,
                CASE WHEN event_text_available(sources_json,text_expires_at) THEN user_name END AS user_name,
                CASE WHEN event_text_available(sources_json,text_expires_at) THEN trigger_phrase END AS trigger_phrase,
                CASE WHEN event_text_available(sources_json,text_expires_at) THEN context_json ELSE metrics_json END AS context_json,
                hour, minute, weekday, is_weekend, time_of_day, sources_json, expires_at, text_expires_at
                FROM main.events_policy_v1
                WHERE event_metadata_current(timestamp,expires_at) AND event_unrevoked(sources_json,
                    trigger_phrase IS NOT NULL OR user_name IS NOT NULL OR context_json IS NULL OR context_json!=metrics_json)""")
            yield db

    @staticmethod
    def _metrics(context: dict) -> dict:
        allowed = {"wake_score", "rms", "compression_ratio", "no_speech_prob", "avg_logprob", "confidence", "latency_ms", "success"}
        return {key: value for key, value in context.items() if key in allowed
                and type(value) in (bool, int, float) and math.isfinite(value)}

    def log(
        self,
        entity_id: str,
        action: str,
        event_type: EventType = EventType.COMMAND,
        user_id: str = None,
        user_name: str = None,
        trigger_phrase: str = None,
        extra_context: dict = None,
        *, sources: tuple[MemorySource, ...] = (),
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
        if self._work.status()["closed"]:
            raise RuntimeError("event_logger_closed")
        timestamp = self._clock()
        if not all(isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,256}", value) for value in (entity_id, action)):
            raise ValueError("invalid_event_identifier")
        if user_id is not None and (not isinstance(user_id, str) or not re.fullmatch(r"[\w.:-]{1,256}", user_id)):
            raise ValueError("invalid_event_user_id")
        if not isinstance(event_type, EventType):
            raise ValueError("invalid_event_type")
        if not isinstance(sources, tuple) or len(sources) > 256 or any(not isinstance(source, MemorySource) for source in sources):
            raise ValueError("invalid_event_sources")
        if sources and (not sources_current(sources, timestamp) or self.source_policy is not None and not self.source_policy.sources_available(sources)):
            raise PermissionError("event_source_expired_or_deleted")
        expires_at = timestamp + self.retention_days*86400
        if not math.isfinite(expires_at) or expires_at <= timestamp:
            raise ValueError("invalid_event_retention")
        text_expiry = min((source.expires_at for source in sources), default=None)
        
        # Construir contexto
        context = self._build_context(extra_context)
        
        metrics = self._metrics(context)
        context = context if sources else metrics

        event = Event(
            timestamp=timestamp,
            event_type=event_type,
            entity_id=entity_id,
            action=action,
            user_id=user_id,
            user_name=user_name if sources else None,
            trigger_phrase=trigger_phrase if sources else None,
            context=context, sources=sources, expires_at=expires_at, text_expires_at=text_expiry
        )
        
        self.initialize()
        guard = self.source_policy.admit_source_write(sources) if self.source_policy is not None and sources else nullcontext()
        with guard, self._connect() as conn:
            conn.execute("""INSERT INTO events_policy_v1 (
                timestamp, event_type, entity_id, action, user_id, user_name,
                trigger_phrase, context_json, hour, minute, weekday, is_weekend,
                time_of_day, sources_json, metrics_json, expires_at, text_expires_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                event.timestamp, event.event_type.value, entity_id, action, user_id,
                event.user_name, event.trigger_phrase, json.dumps(context, allow_nan=False),
                event.hour, event.minute, event.weekday, int(event.is_weekend), event.time_of_day,
                json.dumps([source.to_dict() for source in sources]), json.dumps(metrics), expires_at, text_expiry))
        if self._maintenance_loop is not None:
            self._maintenance_loop.call_soon_threadsafe(self._changed.set)
        if self._retention_notify is not None:
            self._retention_notify(str(self.db_path))
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
            except Exception as error:
                logger.warning("Event context provider unavailable: %s", type(error).__name__)
        
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
        with self._connect() as conn:
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
                    context=json.loads(row["context_json"]) if row["context_json"] else {},
                    sources=tuple(MemorySource.from_dict(item) for item in json.loads(row["sources_json"])),
                    expires_at=row["expires_at"], text_expires_at=row["text_expires_at"]
                )
                events.append(event)
        
        return events
    
    def get_entity_history(
        self,
        entity_id: str,
        days: int = 30
    ) -> list[Event]:
        """Obtener historial de una entidad"""
        start_time = self._clock() - (days * 24 * 3600)
        return self.get_events(entity_id=entity_id, start_time=start_time)

    def get_pattern_history(self, days: float = 30, *, limit: int = 10_000) -> PatternHistory:
        """Read bounded metadata in one SQL observation, without loading raw text.

        An incomplete history cannot support a confidence denominator. Callers
        must abstain rather than treating a truncated sample as a full window.
        This blocking primitive belongs in the logger's owned native scope.
        """
        if type(days) not in (int, float) or not math.isfinite(days) or days <= 0:
            raise ValueError("invalid_pattern_history_window")
        if type(limit) is not int or not 0 < limit <= 10_000:
            raise ValueError("invalid_pattern_history_limit")
        with self._connect() as db:
            target = event_target_id(event_descriptor(self.db_path))
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """SELECT id,timestamp,event_type,entity_id,action,user_id,sources_json,expires_at
                FROM events WHERE timestamp>=? ORDER BY timestamp,id LIMIT ?""",
                (self._clock() - days * 86400, limit + 1),
            ).fetchall()
            if len(rows) > limit:
                return PatternHistory((), False)
            return PatternHistory(tuple(project_pattern_event(row, target) for row in rows), True)

    def _resolve_pattern_evidence(self, db, evidence: tuple[EventDependency, ...]) -> PatternInputs | None:
        if (
            type(evidence) is not tuple or not 0 < len(evidence) <= 10_000
            or any(not isinstance(item, EventDependency) for item in evidence)
        ):
            return None
        target = event_target_id(event_descriptor(self.db_path))
        if any(item.target_id != target or self._clock() >= item.expires_at for item in evidence):
            return None
        expected = {item.event_id: item for item in evidence}
        if len(expected) != len(evidence):
            return None
        db.row_factory = sqlite3.Row
        ids = tuple(expected)
        events, sources = [], set()
        for offset in range(0, len(ids), 128):
            batch = ids[offset:offset + 128]
            rows = db.execute(
                """SELECT id,timestamp,event_type,entity_id,action,user_id,sources_json,expires_at
                FROM events WHERE id IN (SELECT value FROM json_each(?))""",
                (json.dumps(batch),),
            ).fetchall()
            if len(rows) != len(batch):
                return None
            for row in rows:
                event = project_pattern_event(row, target)
                if event.dependency != expected[row["id"]]:
                    return None
                events.append(event)
                sources.update(event_sources(row["sources_json"], False))
                if len(sources) > 10_000:
                    raise ValueError("pattern_lineage_capacity")
        if any(self._clock() >= item.expires_at for item in evidence) or (
            self.source_policy is not None and not self.source_policy.activity_sources_available(tuple(sources))
        ):
            return None
        return PatternInputs(
            tuple(sorted(events, key=lambda e: (e.timestamp, e.dependency.event_id))),
            tuple(sorted(sources, key=lambda source: json.dumps(source.to_dict(), sort_keys=True))),
        )

    def resolve_pattern_evidence(self, evidence: tuple[EventDependency, ...]) -> PatternInputs | None:
        """Read actual receipts and lineage; this observation cannot admit writes."""
        with self._connect() as db:
            db.execute("BEGIN")
            return self._resolve_pattern_evidence(db, evidence)

    def pattern_evidence_current(self, evidence: tuple[EventDependency, ...]) -> bool:
        """Check original metadata deadlines and authority without renewing them."""
        return self.resolve_pattern_evidence(evidence) is not None

    @contextmanager
    def admit_pattern_write(self, evidence: tuple[EventDependency, ...]):
        """Fence a metadata derivative commit against markers and event mutation.

        Lock order is the leased authority, event database, then the derivative
        store. Resolution is repeated inside both guards; a prior read result
        or a raw capture's expired TTL never authorizes this metadata channel.
        """
        if self.source_policy is None:
            raise PermissionError("activity_write_requires_privacy_authority")
        self.initialize()
        with self.source_policy.serialize_activity_write(), self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            inputs = self._resolve_pattern_evidence(db, evidence)
            if inputs is None:
                raise PermissionError("activity_evidence_unavailable")
            yield inputs

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
        start_time = self._clock() - (days * 24 * 3600)
        
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
        
        with self._connect() as conn:
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
        start_time = self._clock() - (days * 24 * 3600)
        
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
        
        with self._connect() as conn:
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
        start_time = self._clock() - (days * 24 * 3600)
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
        with self._connect() as conn:
            cursor = conn.execute(query, (window_seconds, start_time, min_occurrences))
            for row in cursor:
                sequences.append((row[0], row[1], row[2]))
        
        return sequences
    
    def get_stats(self) -> dict:
        """Obtener estadisticas generales"""
        with self._connect() as conn:
            # Total eventos
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            
            # Eventos ultimos 7 dias
            week_ago = self._clock() - (7 * 24 * 3600)
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
            "db_path": str(self.db_path),
            "legacy_policy": "quarantined", "policy_version": 1,
        }
    
    def cleanup_old_events(self) -> int:
        """Delete expired/revoked policy events and scrub expired raw text."""
        with self._connect() as conn:
            removed = conn.execute("DELETE FROM events_policy_v1 WHERE expires_at<=? OR NOT event_committed_unrevoked(sources_json)", (self._clock(),)).rowcount
            conn.execute("""UPDATE events_policy_v1 SET trigger_phrase=NULL, user_name=NULL,
                context_json=metrics_json WHERE text_expires_at<=?""", (self._clock(),))
        return removed

    def next_maintenance_delay(self) -> float:
        """Wake for the next raw or metadata deadline, capped for revocations."""
        with self._connect() as db:
            deadline = db.execute("""SELECT MIN(deadline) FROM (
                SELECT MIN(expires_at) AS deadline FROM events_policy_v1
                UNION ALL SELECT MIN(text_expires_at) FROM events_policy_v1
                WHERE trigger_phrase IS NOT NULL OR context_json!=metrics_json OR user_name IS NOT NULL)""").fetchone()[0]
        return 60 if deadline is None else min(60, max(.01, deadline-self._clock()))

    async def acall(self, function, *args, **kwargs):
        """Run actual disk work in the tracked native scope."""
        return await self._work.run(function, *args, **kwargs)

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._executor,
            copy_context().run, partial(function, *args, **kwargs))

    async def alog(self, *args, **kwargs) -> Event:
        """Bound the caller's wait; a slow write retains native ownership."""
        async with asyncio.timeout(self.observation_timeout_s):
            return await self.acall(self.log, *args, **kwargs)

    def start(self) -> None:
        """Start exactly one deadline-aware owned cleanup loop."""
        if self._work.status()["closed"]:
            raise RuntimeError("event_logger_closed")
        if self._maintenance is None:
            self._maintenance = asyncio.create_task(self._maintain())

    async def _maintain(self) -> None:
        self._maintenance_loop = asyncio.get_running_loop()
        while True:
            self._changed.clear()
            try:
                async with asyncio.timeout(1):
                    await self.acall(self.cleanup_old_events)
                delay = await self.acall(self.next_maintenance_delay)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Event maintenance deferred: %s", type(error).__name__)
                delay = 60
            try:
                async with asyncio.timeout(delay):
                    await self._changed.wait()
            except TimeoutError:
                pass

    def resource_status(self) -> dict:
        """Expose retained native work even after caller timeout/cancellation."""
        return self._work.status()

    def begin_close(self) -> asyncio.Task:
        """Close event admission and retain writes/maintenance until actual exit."""
        if self._closing is None:
            drain = self._work.begin_close()
            if self._maintenance is not None:
                self._maintenance.cancel()
            async def finish():
                if self._maintenance is not None:
                    await asyncio.gather(self._maintenance, return_exceptions=True)
                await drain
                if self._executor is not None:
                    await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while the shared privacy lease retains actual writes."""
        closing = self.begin_close()
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(closing)
        except TimeoutError as error:
            raise WorkCloseTimeout('event_resources_still_running') from error
