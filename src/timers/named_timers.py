"""
Named Timers - Sistema de Timers con Nombre
Múltiples timers simultáneos con nombres descriptivos para cocina y más.

"Timer pasta 8 minutos"
"Timer salsa 15 minutos"
"¿Cuánto falta para la pasta?"
"Cancela el timer de la salsa"
"¿Qué timers tengo?"
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Any
from enum import StrEnum
import uuid

logger = logging.getLogger(__name__)


class TimerState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class NamedTimer:
    """Un timer con nombre"""
    timer_id: str
    name: str                          # "pasta", "salsa", "huevos"
    duration_seconds: int              # Duración total
    created_at: datetime
    zone_id: str = "default"           # Zona donde se creó
    user_id: str | None = None      # Usuario que lo creó

    # Estado
    state: TimerState = TimerState.RUNNING
    remaining_seconds: float = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Configuración
    announce_at: list[int] = field(default_factory=lambda: [60, 30, 10])  # Anunciar a X segundos
    repeat: bool = False               # ¿Repetir automáticamente?

    def __post_init__(self):
        if self.remaining_seconds == 0:
            self.remaining_seconds = self.duration_seconds
        if self.started_at is None:
            self.started_at = datetime.now()

    @property
    def elapsed_seconds(self) -> float:
        return self.duration_seconds - self.remaining_seconds

    @property
    def progress_percent(self) -> float:
        if self.duration_seconds == 0:
            return 100
        return (self.elapsed_seconds / self.duration_seconds) * 100

    def format_remaining(self) -> str:
        """Formatear tiempo restante para voz"""
        remaining = int(self.remaining_seconds)

        if remaining <= 0:
            return "completado"
        elif remaining < 60:
            return f"{remaining} segundos"
        elif remaining < 3600:
            minutes = remaining // 60
            seconds = remaining % 60
            if seconds == 0:
                return f"{minutes} minutos" if minutes > 1 else "1 minuto"
            return f"{minutes} minutos y {seconds} segundos"
        else:
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            if minutes == 0:
                return f"{hours} horas" if hours > 1 else "1 hora"
            return f"{hours} horas y {minutes} minutos"


class NamedTimerManager:
    """
    Gestor de timers con nombre.

    Características:
    - Múltiples timers simultáneos
    - Nombres descriptivos (no "Timer 1", sino "pasta")
    - Anuncios configurables
    - Búsqueda fuzzy por nombre
    - Pausar/reanudar
    - Preguntas de estado por voz
    """

    # Patrones para parsear comandos de voz
    DURATION_PATTERNS = [
        (r"(\d+)\s*(?:hora|horas|h)", 3600),
        (r"(\d+)\s*(?:minuto|minutos|min|m)", 60),
        (r"(\d+)\s*(?:segundo|segundos|seg|s)", 1),
    ]

    def __init__(
        self,
        tts_callback: Callable[[str, str], None] = None,  # (text, zone_id)
        max_timers: int = 20,
        default_announce_intervals: list[int] = None
    ):
        self.tts = tts_callback
        self.max_timers = max_timers
        self.default_announce = default_announce_intervals or [60, 30, 10, 5]

        # Timers activos
        self._timers: dict[str, NamedTimer] = {}

        # Task de actualización
        self._update_task: asyncio.Task | None = None
        self._running = False

        # Callbacks
        self._on_timer_complete: Callable[[NamedTimer], None] | None = None
        self._on_timer_warning: Callable[[NamedTimer, int], None] | None = None

    async def start(self):
        """Iniciar el manager de timers"""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Timer manager iniciado")

    async def stop(self):
        """Detener el manager"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Timer manager detenido")

    async def _update_loop(self):
        """Loop de actualización de timers"""
        last_update = time.time()

        while self._running:
            try:
                await asyncio.sleep(0.5)  # Actualizar cada 500ms

                now = time.time()
                elapsed = now - last_update
                last_update = now

                # Actualizar timers activos
                for timer_id, timer in list(self._timers.items()):
                    if timer.state != TimerState.RUNNING:
                        continue

                    old_remaining = timer.remaining_seconds
                    timer.remaining_seconds -= elapsed

                    # Verificar anuncios
                    for threshold in timer.announce_at:
                        if old_remaining > threshold >= timer.remaining_seconds:
                            await self._announce_warning(timer, threshold)

                    # Verificar completado
                    if timer.remaining_seconds <= 0:
                        timer.remaining_seconds = 0
                        timer.state = TimerState.COMPLETED
                        timer.completed_at = datetime.now()
                        await self._announce_complete(timer)

                        if self._on_timer_complete:
                            self._on_timer_complete(timer)

                        # Repetir si está configurado
                        if timer.repeat:
                            timer.remaining_seconds = timer.duration_seconds
                            timer.state = TimerState.RUNNING
                            timer.started_at = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en timer loop: {e}")

    # ==================== Comandos de Voz ====================

    def handle_voice_command(self, text: str, user_id: str = None, zone_id: str = "default") -> dict:
        """
        Procesar comando de voz relacionado con timers.

        Returns:
            {
                "handled": bool,
                "response": str,
                "timer": NamedTimer o None
            }
        """
        text_lower = text.lower().strip()

        # Crear timer
        if self._is_create_command(text_lower):
            return self._handle_create(text_lower, user_id, zone_id)

        # Consultar tiempo restante
        if self._is_query_command(text_lower):
            return self._handle_query(text_lower)

        # Cancelar timer
        if self._is_cancel_command(text_lower):
            return self._handle_cancel(text_lower)

        # Pausar timer
        if self._is_pause_command(text_lower):
            return self._handle_pause(text_lower)

        # Reanudar timer
        if self._is_resume_command(text_lower):
            return self._handle_resume(text_lower)

        # Listar timers
        if self._is_list_command(text_lower):
            return self._handle_list()

        return {"handled": False, "response": "", "timer": None}

    def _is_create_command(self, text: str) -> bool:
        patterns = [
            r"timer\s+.+\s+\d+",
            r"pon(?:me|er)?\s+(?:un\s+)?timer",
            r"crea(?:r)?\s+(?:un\s+)?timer",
            r"activa(?:r)?\s+(?:un\s+)?timer",
            r"\d+\s+(?:minuto|segundo|hora).+timer",
        ]
        return any(re.search(p, text) for p in patterns)

    def _is_query_command(self, text: str) -> bool:
        patterns = [
            r"cu[aá]nto\s+(?:falta|queda)",
            r"tiempo\s+(?:del|de la|para)",
            r"c[oó]mo\s+va\s+(?:el|la)",
            r"estado\s+(?:del|de la)\s+timer",
        ]
        return any(re.search(p, text) for p in patterns)

    def _is_cancel_command(self, text: str) -> bool:
        patterns = [
            r"cancela(?:r)?\s+(?:el\s+)?timer",
            r"elimina(?:r)?\s+(?:el\s+)?timer",
            r"quita(?:r)?\s+(?:el\s+)?timer",
            r"para(?:r)?\s+(?:el\s+)?timer",
        ]
        return any(re.search(p, text) for p in patterns)

    def _is_pause_command(self, text: str) -> bool:
        return "pausa" in text and "timer" in text

    def _is_resume_command(self, text: str) -> bool:
        patterns = ["continua", "reanuda", "sigue", "resume"]
        return any(p in text for p in patterns) and "timer" in text

    def _is_list_command(self, text: str) -> bool:
        patterns = [
            r"qu[eé]\s+timers",
            r"cu[aá]les\s+timers",
            r"lista(?:r)?\s+timers",
            r"mis\s+timers",
            r"timers\s+activos",
        ]
        return any(re.search(p, text) for p in patterns)

    def _handle_create(self, text: str, user_id: str, zone_id: str) -> dict:
        """Crear nuevo timer"""
        # Extraer nombre y duración
        name, duration = self._parse_timer_command(text)

        if not name:
            name = f"timer_{len(self._timers) + 1}"

        if duration <= 0:
            return {
                "handled": True,
                "response": "No entendí la duración. Di por ejemplo: timer pasta 10 minutos",
                "timer": None
            }

        # Verificar límite
        active_count = sum(1 for t in self._timers.values() if t.state == TimerState.RUNNING)
        if active_count >= self.max_timers:
            return {
                "handled": True,
                "response": f"Ya tienes {active_count} timers activos. Cancela alguno primero.",
                "timer": None
            }

        # Verificar nombre duplicado
        existing = self._find_timer_by_name(name)
        if existing and existing.state == TimerState.RUNNING:
            return {
                "handled": True,
                "response": f"Ya tienes un timer llamado {name}. ¿Quieres reemplazarlo?",
                "timer": None
            }

        # Crear timer
        timer = NamedTimer(
            timer_id=f"timer_{uuid.uuid4().hex[:8]}",
            name=name,
            duration_seconds=duration,
            created_at=datetime.now(),
            zone_id=zone_id,
            user_id=user_id,
            announce_at=self.default_announce
        )

        self._timers[timer.timer_id] = timer

        response = f"Timer {name} configurado para {timer.format_remaining()}"
        logger.info(f"Timer creado: {name} = {duration}s")

        return {"handled": True, "response": response, "timer": timer}

    def _handle_query(self, text: str) -> dict:
        """Consultar estado de timer(s)"""
        # Buscar nombre específico
        name = self._extract_timer_name(text)

        if name:
            timer = self._find_timer_by_name(name)
            if timer and timer.state == TimerState.RUNNING:
                response = f"Faltan {timer.format_remaining()} para {timer.name}"
                return {"handled": True, "response": response, "timer": timer}
            elif timer:
                response = f"El timer {timer.name} está {timer.state.value}"
                return {"handled": True, "response": response, "timer": timer}
            else:
                return {
                    "handled": True,
                    "response": f"No encontré un timer llamado {name}",
                    "timer": None
                }

        # Sin nombre específico: reportar todos
        running = [t for t in self._timers.values() if t.state == TimerState.RUNNING]

        if not running:
            return {"handled": True, "response": "No tienes timers activos", "timer": None}

        if len(running) == 1:
            t = running[0]
            response = f"Timer {t.name}: faltan {t.format_remaining()}"
        else:
            parts = [f"{t.name} {t.format_remaining()}" for t in sorted(running, key=lambda x: x.remaining_seconds)]
            response = "Timers: " + ", ".join(parts)

        return {"handled": True, "response": response, "timer": None}

    def _handle_cancel(self, text: str) -> dict:
        """Cancelar timer"""
        name = self._extract_timer_name(text)

        # Cancelar todos
        if "todos" in text or "all" in text:
            count = 0
            for timer in self._timers.values():
                if timer.state == TimerState.RUNNING:
                    timer.state = TimerState.CANCELLED
                    count += 1
            return {
                "handled": True,
                "response": f"Cancelé {count} timers" if count else "No hay timers activos",
                "timer": None
            }

        if name:
            timer = self._find_timer_by_name(name)
            if timer and timer.state == TimerState.RUNNING:
                timer.state = TimerState.CANCELLED
                return {
                    "handled": True,
                    "response": f"Timer {timer.name} cancelado",
                    "timer": timer
                }
            return {
                "handled": True,
                "response": f"No encontré un timer activo llamado {name}",
                "timer": None
            }

        # Sin nombre: cancelar el más reciente
        running = [t for t in self._timers.values() if t.state == TimerState.RUNNING]
        if running:
            timer = max(running, key=lambda t: t.created_at)
            timer.state = TimerState.CANCELLED
            return {
                "handled": True,
                "response": f"Timer {timer.name} cancelado",
                "timer": timer
            }

        return {"handled": True, "response": "No hay timers activos", "timer": None}

    def _handle_pause(self, text: str) -> dict:
        """Pausar timer"""
        name = self._extract_timer_name(text)
        timer = self._find_timer_by_name(name) if name else None

        if not timer:
            running = [t for t in self._timers.values() if t.state == TimerState.RUNNING]
            timer = running[0] if running else None

        if timer and timer.state == TimerState.RUNNING:
            timer.state = TimerState.PAUSED
            return {
                "handled": True,
                "response": f"Timer {timer.name} pausado con {timer.format_remaining()} restantes",
                "timer": timer
            }

        return {"handled": True, "response": "No hay timer para pausar", "timer": None}

    def _handle_resume(self, text: str) -> dict:
        """Reanudar timer"""
        name = self._extract_timer_name(text)
        timer = self._find_timer_by_name(name) if name else None

        if not timer:
            paused = [t for t in self._timers.values() if t.state == TimerState.PAUSED]
            timer = paused[0] if paused else None

        if timer and timer.state == TimerState.PAUSED:
            timer.state = TimerState.RUNNING
            timer.started_at = datetime.now()
            return {
                "handled": True,
                "response": f"Timer {timer.name} reanudado, faltan {timer.format_remaining()}",
                "timer": timer
            }

        return {"handled": True, "response": "No hay timer pausado", "timer": None}

    def _handle_list(self) -> dict:
        """Listar todos los timers"""
        running = [t for t in self._timers.values() if t.state == TimerState.RUNNING]
        paused = [t for t in self._timers.values() if t.state == TimerState.PAUSED]

        if not running and not paused:
            return {"handled": True, "response": "No tienes timers activos", "timer": None}

        parts = []
        if running:
            for t in sorted(running, key=lambda x: x.remaining_seconds):
                parts.append(f"{t.name}: {t.format_remaining()}")
        if paused:
            for t in paused:
                parts.append(f"{t.name}: pausado")

        response = "Timers: " + ", ".join(parts)
        return {"handled": True, "response": response, "timer": None}

    # ==================== Parsing ====================

    def _parse_timer_command(self, text: str) -> tuple[str, int]:
        """
        Parsear comando de timer.

        Returns:
            (nombre, duración_segundos)
        """
        # Extraer duración
        duration = 0
        for pattern, multiplier in self.DURATION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                duration += int(match.group(1)) * multiplier

        # Extraer nombre (lo que viene después de "timer" y antes del número)
        name = None

        # Patrón: "timer NOMBRE X minutos"
        match = re.search(r"timer\s+(?:de\s+)?(?:la\s+|el\s+)?(\w+)", text)
        if match:
            potential_name = match.group(1)
            # Verificar que no sea un número o palabra de duración
            if not potential_name.isdigit() and potential_name not in ["minuto", "minutos", "segundo", "segundos", "hora", "horas", "para", "de"]:
                name = potential_name

        # Si no encontramos nombre, buscar después del número
        if not name:
            match = re.search(r"\d+\s*(?:minuto|segundo|hora)s?\s+(?:para\s+)?(?:la\s+|el\s+)?(\w+)", text)
            if match:
                name = match.group(1)

        return name, duration

    def _extract_timer_name(self, text: str) -> str | None:
        """Extraer nombre de timer del texto"""
        # Patrones para extraer nombre
        patterns = [
            r"timer\s+(?:de\s+)?(?:la\s+|el\s+)?(\w+)",
            r"(?:del|de la|para(?:\s+la)?)\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1)
                if name not in ["timer", "timers", "todos", "all"]:
                    return name

        return None

    def _find_timer_by_name(self, name: str) -> NamedTimer | None:
        """Buscar timer por nombre (fuzzy)"""
        name_lower = name.lower()

        # Búsqueda exacta
        for timer in self._timers.values():
            if timer.name.lower() == name_lower:
                return timer

        # Búsqueda parcial
        for timer in self._timers.values():
            if name_lower in timer.name.lower() or timer.name.lower() in name_lower:
                return timer

        return None

    # ==================== Anuncios ====================

    async def _announce_warning(self, timer: NamedTimer, seconds_left: int):
        """Anunciar advertencia de tiempo"""
        if seconds_left >= 60:
            time_str = f"{seconds_left // 60} minutos"
        else:
            time_str = f"{seconds_left} segundos"

        message = f"Quedan {time_str} para {timer.name}"
        logger.info(f"Timer warning: {message}")

        if self._on_timer_warning:
            self._on_timer_warning(timer, seconds_left)

        if self.tts:
            self.tts(message, timer.zone_id)

    async def _announce_complete(self, timer: NamedTimer):
        """Anunciar timer completado"""
        message = f"¡Timer {timer.name} completado!"
        logger.info(f"Timer complete: {timer.name}")

        if self.tts:
            # Anunciar 3 veces para asegurar que se escuche
            for _ in range(3):
                self.tts(message, timer.zone_id)
                await asyncio.sleep(2)

    # ==================== API ====================

    def create_timer(
        self,
        name: str,
        duration_seconds: int,
        user_id: str = None,
        zone_id: str = "default"
    ) -> NamedTimer:
        """Crear timer programáticamente"""
        timer = NamedTimer(
            timer_id=f"timer_{uuid.uuid4().hex[:8]}",
            name=name,
            duration_seconds=duration_seconds,
            created_at=datetime.now(),
            zone_id=zone_id,
            user_id=user_id,
            announce_at=self.default_announce
        )
        self._timers[timer.timer_id] = timer
        return timer

    def get_timer(self, timer_id: str) -> NamedTimer | None:
        """Obtener timer por ID"""
        return self._timers.get(timer_id)

    def get_all_timers(self) -> list[NamedTimer]:
        """Obtener todos los timers"""
        return list(self._timers.values())

    def get_active_timers(self) -> list[NamedTimer]:
        """Obtener timers activos"""
        return [t for t in self._timers.values() if t.state == TimerState.RUNNING]

    def cancel_timer(self, timer_id: str) -> bool:
        """Cancelar timer por ID"""
        if timer_id in self._timers:
            self._timers[timer_id].state = TimerState.CANCELLED
            return True
        return False

    def cleanup_completed(self, max_age_hours: int = 24):
        """Limpiar timers completados/cancelados antiguos"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            tid for tid, t in self._timers.items()
            if t.state in [TimerState.COMPLETED, TimerState.CANCELLED]
            and t.created_at < cutoff
        ]
        for tid in to_remove:
            del self._timers[tid]

    # ==================== Callbacks ====================

    def on_timer_complete(self, callback: Callable[[NamedTimer], None]):
        """Registrar callback para timer completado"""
        self._on_timer_complete = callback

    def on_timer_warning(self, callback: Callable[[NamedTimer, int], None]):
        """Registrar callback para advertencias"""
        self._on_timer_warning = callback

    # ==================== Estado ====================

    def get_status(self) -> dict:
        """Obtener estado del manager"""
        return {
            "running": self._running,
            "total_timers": len(self._timers),
            "active": sum(1 for t in self._timers.values() if t.state == TimerState.RUNNING),
            "paused": sum(1 for t in self._timers.values() if t.state == TimerState.PAUSED),
            "completed": sum(1 for t in self._timers.values() if t.state == TimerState.COMPLETED),
        }
