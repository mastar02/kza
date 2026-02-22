"""
Latency Monitor Module
Monitorea y reporta latencias del pipeline de voz en tiempo real.
"""

import json
import logging
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Un registro de latencia de un comando procesado"""
    timestamp: float
    total_ms: float
    components: dict[str, float]  # {component_name: latency_ms}
    target_ms: float
    met_target: bool
    user: str | None = None
    intent: str | None = None


@dataclass
class ComponentStats:
    """Estadísticas de un componente"""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_vals = sorted(self.recent)
        return sorted_vals[len(sorted_vals) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_vals = sorted(self.recent)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_vals = sorted(self.recent)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def record(self, latency_ms: float):
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.recent.append(latency_ms)


class LatencyMonitor:
    """
    Monitor de latencia para el pipeline de voz.

    Funcionalidades:
    - Tracking de latencia por componente
    - Estadísticas en tiempo real (avg, p50, p95, p99)
    - Persistencia en SQLite
    - Alertas cuando se excede el target
    - Dashboard CLI
    """

    # Componentes estándar del pipeline
    COMPONENTS = [
        "parallel_stt_speaker",  # STT + Speaker ID paralelo
        "stt",                   # Speech-to-Text
        "speaker_id",            # Identificación de speaker
        "vector_search",         # Búsqueda en ChromaDB
        "home_assistant",        # Llamada a HA
        "tts",                   # Text-to-Speech
        "router",                # Fast Router
        "llm",                   # LLM 70B
        "routine_check",         # Verificación de rutinas
    ]

    def __init__(
        self,
        db_path: str = "./data/latency.db",
        target_ms: float = 300.0,
        alert_callback: Callable[[LatencyRecord], None] | None = None,
        window_size: int = 100
    ):
        self.db_path = Path(db_path)
        self.target_ms = target_ms
        self.alert_callback = alert_callback
        self.window_size = window_size

        self._lock = Lock()
        self._stats: dict[str, ComponentStats] = {}
        self._total_stats = ComponentStats(name="total")
        self._records: deque[LatencyRecord] = deque(maxlen=1000)

        # Contadores
        self._total_commands = 0
        self._commands_under_target = 0
        self._session_start = time.time()

        # Inicializar stats para componentes conocidos
        for comp in self.COMPONENTS:
            self._stats[comp] = ComponentStats(name=comp)

        self._init_db()

    def _init_db(self):
        """Inicializar base de datos SQLite"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    total_ms REAL,
                    target_ms REAL,
                    met_target INTEGER,
                    user TEXT,
                    intent TEXT,
                    components_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON latency_records(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_met_target
                ON latency_records(met_target)
            """)

    def record(self, timings: dict[str, float], user: str = None, intent: str = None):
        """
        Registrar latencias de un comando procesado.

        Args:
            timings: Dict de {componente: latencia_ms}
            user: Usuario identificado (opcional)
            intent: Intent detectado (opcional)
        """
        with self._lock:
            # Calcular total
            total_ms = sum(timings.values())
            met_target = total_ms <= self.target_ms

            # Crear registro
            record = LatencyRecord(
                timestamp=time.time(),
                total_ms=total_ms,
                components=timings.copy(),
                target_ms=self.target_ms,
                met_target=met_target,
                user=user,
                intent=intent
            )

            # Actualizar estadísticas por componente
            for comp, latency in timings.items():
                if comp not in self._stats:
                    self._stats[comp] = ComponentStats(name=comp)
                self._stats[comp].record(latency)

            # Actualizar total
            self._total_stats.record(total_ms)
            self._total_commands += 1
            if met_target:
                self._commands_under_target += 1

            # Guardar en memoria
            self._records.append(record)

            # Persistir en DB
            self._save_record(record)

            # Alertar si excede target
            if not met_target and self.alert_callback:
                self.alert_callback(record)

            return record

    def _save_record(self, record: LatencyRecord):
        """Guardar registro en SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO latency_records
                    (timestamp, total_ms, target_ms, met_target, user, intent, components_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp,
                    record.total_ms,
                    record.target_ms,
                    1 if record.met_target else 0,
                    record.user,
                    record.intent,
                    json.dumps(record.components)
                ))
        except Exception as e:
            logger.error(f"Error saving latency record: {e}")

    def get_stats(self) -> dict:
        """Obtener estadísticas actuales"""
        with self._lock:
            success_rate = (
                self._commands_under_target / self._total_commands * 100
                if self._total_commands > 0 else 0.0
            )

            session_duration = time.time() - self._session_start

            stats = {
                "session": {
                    "duration_seconds": session_duration,
                    "total_commands": self._total_commands,
                    "commands_under_target": self._commands_under_target,
                    "success_rate_percent": round(success_rate, 1),
                    "target_ms": self.target_ms
                },
                "total": {
                    "avg_ms": round(self._total_stats.avg_ms, 1),
                    "min_ms": round(self._total_stats.min_ms, 1) if self._total_stats.min_ms != float('inf') else 0,
                    "max_ms": round(self._total_stats.max_ms, 1),
                    "p50_ms": round(self._total_stats.p50_ms, 1),
                    "p95_ms": round(self._total_stats.p95_ms, 1),
                    "p99_ms": round(self._total_stats.p99_ms, 1)
                },
                "components": {}
            }

            for name, comp_stats in self._stats.items():
                if comp_stats.count > 0:
                    stats["components"][name] = {
                        "count": comp_stats.count,
                        "avg_ms": round(comp_stats.avg_ms, 1),
                        "min_ms": round(comp_stats.min_ms, 1) if comp_stats.min_ms != float('inf') else 0,
                        "max_ms": round(comp_stats.max_ms, 1),
                        "p50_ms": round(comp_stats.p50_ms, 1),
                        "p95_ms": round(comp_stats.p95_ms, 1)
                    }

            return stats

    def get_recent_records(self, limit: int = 10) -> list[dict]:
        """Obtener registros recientes"""
        with self._lock:
            records = list(self._records)[-limit:]
            return [
                {
                    "timestamp": r.timestamp,
                    "total_ms": round(r.total_ms, 1),
                    "met_target": r.met_target,
                    "user": r.user,
                    "intent": r.intent,
                    "components": {k: round(v, 1) for k, v in r.components.items()}
                }
                for r in records
            ]

    def format_dashboard(self) -> str:
        """Generar dashboard de texto para CLI"""
        stats = self.get_stats()
        session = stats["session"]
        total = stats["total"]
        components = stats["components"]

        lines = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║              KZA Voice - Latency Dashboard                   ║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")

        # Session info
        duration_min = session["duration_seconds"] / 60
        lines.append(f"║  Session: {duration_min:.1f} min | Commands: {session['total_commands']} | Target: {session['target_ms']:.0f}ms")

        # Success rate con color
        rate = session["success_rate_percent"]
        rate_indicator = "✅" if rate >= 95 else "⚠️" if rate >= 80 else "❌"
        lines.append(f"║  Success Rate: {rate:.1f}% {rate_indicator} ({session['commands_under_target']}/{session['total_commands']} under target)")

        lines.append("║")
        lines.append("║  Total Latency:")
        lines.append(f"║    avg: {total['avg_ms']:6.1f}ms | p50: {total['p50_ms']:6.1f}ms | p95: {total['p95_ms']:6.1f}ms")
        lines.append(f"║    min: {total['min_ms']:6.1f}ms | max: {total['max_ms']:6.1f}ms | p99: {total['p99_ms']:6.1f}ms")

        lines.append("║")
        lines.append("║  Components:")
        lines.append("║  ┌────────────────────┬────────┬────────┬────────┬────────┐")
        lines.append("║  │ Component          │ Avg    │ P50    │ P95    │ Count  │")
        lines.append("║  ├────────────────────┼────────┼────────┼────────┼────────┤")

        # Ordenar por latencia promedio descendente
        sorted_comps = sorted(
            components.items(),
            key=lambda x: x[1]["avg_ms"],
            reverse=True
        )

        for name, comp in sorted_comps:
            name_display = name[:18].ljust(18)
            lines.append(
                f"║  │ {name_display} │ {comp['avg_ms']:6.1f} │ {comp['p50_ms']:6.1f} │ {comp['p95_ms']:6.1f} │ {comp['count']:6d} │"
            )

        lines.append("║  └────────────────────┴────────┴────────┴────────┴────────┘")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")

        return "\n".join(lines)

    def get_historical_stats(self, hours: int = 24) -> dict:
        """Obtener estadísticas históricas de la DB"""
        cutoff = time.time() - (hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total y success rate
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN met_target = 1 THEN 1 ELSE 0 END) as under_target,
                    AVG(total_ms) as avg_ms,
                    MIN(total_ms) as min_ms,
                    MAX(total_ms) as max_ms
                FROM latency_records
                WHERE timestamp > ?
            """, (cutoff,)).fetchone()

            # Por hora
            hourly = conn.execute("""
                SELECT
                    strftime('%H', datetime(timestamp, 'unixepoch', 'localtime')) as hour,
                    COUNT(*) as count,
                    AVG(total_ms) as avg_ms
                FROM latency_records
                WHERE timestamp > ?
                GROUP BY hour
                ORDER BY hour
            """, (cutoff,)).fetchall()

            # Por intent
            by_intent = conn.execute("""
                SELECT
                    intent,
                    COUNT(*) as count,
                    AVG(total_ms) as avg_ms
                FROM latency_records
                WHERE timestamp > ? AND intent IS NOT NULL
                GROUP BY intent
                ORDER BY count DESC
            """, (cutoff,)).fetchall()

            return {
                "period_hours": hours,
                "total_commands": row["total"],
                "success_rate": row["under_target"] / row["total"] * 100 if row["total"] > 0 else 0,
                "avg_ms": row["avg_ms"] or 0,
                "min_ms": row["min_ms"] or 0,
                "max_ms": row["max_ms"] or 0,
                "by_hour": [{"hour": h["hour"], "count": h["count"], "avg_ms": h["avg_ms"]} for h in hourly],
                "by_intent": [{"intent": i["intent"], "count": i["count"], "avg_ms": i["avg_ms"]} for i in by_intent]
            }

    def reset_session(self):
        """Reiniciar estadísticas de sesión (no borra DB)"""
        with self._lock:
            for stats in self._stats.values():
                stats.count = 0
                stats.total_ms = 0.0
                stats.min_ms = float('inf')
                stats.max_ms = 0.0
                stats.recent.clear()

            self._total_stats = ComponentStats(name="total")
            self._total_commands = 0
            self._commands_under_target = 0
            self._session_start = time.time()
            self._records.clear()

            logger.info("Latency monitor session reset")
