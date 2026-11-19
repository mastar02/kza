"""
Named Timers - Sistema de Timers con Nombre
Múltiples timers simultáneos con nombres descriptivos para cocina y más.

"Timer pasta 8 minutos"
"Timer salsa 15 minutos"
"¿Cuánto falta para la pasta?"
"Cancela el timer de la salsa"
"¿Qué timers tengo?"

Runtime (T030-U2): a running timer keeps a deadline on a monotonic clock (the
default event loop's clock) plus its wall-clock ``ends_at``; the remaining time
is always computed from the deadline, never decremented per tick. Warnings and
the ring run in tasks owned by the manager, off the tick loop, so a ring that
speaks for seconds never delays another timer. A ring stops when the household
silences it (voice stop or barge-in), and its outcome is a delivery receipt
(``RingDelivery``) beside the unchanged ``TimerState`` (design schema_v2_draft:
ring delivery is not a timer state). ``stop()`` cancels and awaits every
delivery task. Timers stay volatile until SCHEMA-V2 (T030-U3).
"""

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from enum import StrEnum
import uuid

from src.core.voice_delivery import DeliveryState, ResponseResult, ResponseStage, VoiceDeliveryError, invoke_voice_callback
from src.core.task_outcome import EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.timers.voice_request import format_duration

logger = logging.getLogger(__name__)

# T030-U1-Q4 (user answer 2026-11-17): one "le queda 1 minuto" warning, and none
# for timers of 2 minutes or less (they ring soon enough on their own).
TIME_LEFT_WARNING_S = 60
NO_WARNING_UP_TO_S = 120

# Longest sleep of the tick loop; it wakes earlier for the next warning or deadline.
TICK_S = 0.5
# The ring: the same sentence up to three times, two seconds apart (T030-U1 behaviour).
RING_REPEATS = 3
RING_GAP_S = 2.0
# A warning found this late (the loop was stalled) would name a wrong time left: it is skipped.
WARNING_STALE_S = 5.0
# A stop command this soon after a ring ended (barge-in already cut it) still names that ring,
# so it never falls through to cancelling another running timer.
RING_FOLLOW_UP_S = 30.0


def warning_thresholds(
    duration_seconds: float, intervals: list[int] | tuple[int, ...] = (TIME_LEFT_WARNING_S,)
) -> list[int]:
    """Seconds-left marks at which a timer of this duration speaks a warning.

    Args:
        duration_seconds: Total duration of the timer.
        intervals: Warning marks for a timer long enough to get any.

    Returns:
        A new list: empty for 2 minutes or less, else ``intervals``.
    """
    if duration_seconds <= NO_WARNING_UP_TO_S:
        return []
    return list(intervals)


class TimerState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RingDelivery(StrEnum):
    """Delivery receipt of a timer's ring (not a timer state; design schema_v2_draft).

    PENDING: not due yet. RINGING: the ring task is speaking or waiting between
    repetitions. DELIVERED: at least one repetition played completely.
    STOPPED: the household silenced it (voice stop, or a barge-in cut a
    repetition whose audio had started). RING_FAILED: the output broke after
    its audio may have started, or the delivery could not be verified.
    UNDELIVERED: no repetition produced any audio.
    """

    PENDING = "pending"
    RINGING = "ringing"
    DELIVERED = "delivered"
    STOPPED = "stopped"
    RING_FAILED = "ring_failed"
    UNDELIVERED = "undelivered"


RING_NOT_HEARD = frozenset({RingDelivery.RING_FAILED, RingDelivery.UNDELIVERED})


@dataclass
class NamedTimer:
    """Un timer con nombre.

    While RUNNING the timer owns ``deadline`` (seconds on ``clock``) and
    ``ends_at`` (epoch seconds on ``wall_clock``); ``remaining_seconds`` is
    derived from the deadline. While paused, cancelled or completed it keeps
    the seconds that were left and has no deadline.
    """
    timer_id: str
    name: str                          # "pasta", "salsa", "huevos"
    duration_seconds: int              # Duración total
    created_at: datetime
    zone_id: str = "default"           # Zona donde se creó
    user_id: str | None = None      # Usuario que lo creó

    # Estado
    state: TimerState = TimerState.RUNNING
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Configuración
    announce_at: list[int] | None = None  # Seconds-left warnings; None -> warning_thresholds(duration)
    repeat: bool = False               # ¿Repetir automáticamente?

    # Runtime (T030-U2)
    ring: RingDelivery = RingDelivery.PENDING
    ring_reported: bool = False        # A ring that was not heard was already said in a query
    deadline: float | None = field(default=None, repr=False)
    ends_at: float | None = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    wall_clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)
    warned: set[int] = field(default_factory=set, repr=False, compare=False)
    ring_ended_at: float | None = field(default=None, repr=False, compare=False)
    _left: float = field(default=0.0, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()
        if self.announce_at is None:
            self.announce_at = warning_thresholds(self.duration_seconds)
        self._left = float(self.duration_seconds)
        # Marks at or above the whole duration were never crossed going down: never spoken.
        self.warned = {mark for mark in self.announce_at if mark >= self.duration_seconds}
        if self.state is TimerState.RUNNING and self.deadline is None:
            self._arm(self._left)

    def _arm(self, seconds: float) -> None:
        """Run for ``seconds`` from now: deadline on the monotonic clock, ends_at on the wall clock."""
        self.deadline = self.clock() + seconds
        self.ends_at = self.wall_clock() + seconds

    def _freeze(self, *, keep_ends_at: bool = False) -> None:
        """Keep what is left and drop the deadline (pause, cancel, completion)."""
        self._left = self.remaining_seconds
        self.deadline = None
        if not keep_ends_at:
            self.ends_at = None

    @property
    def remaining_seconds(self) -> float:
        """Seconds left: from the deadline while running, else what was left when it stopped."""
        if self.state is TimerState.RUNNING and self.deadline is not None:
            return max(0.0, self.deadline - self.clock())
        return self._left

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


def _spoken_name(timer) -> str:
    """Label of a timer for voice, or its duration when it has none."""
    name = getattr(timer, "name", "")
    return name if name else format_duration(timer.duration_seconds)


def _describe(timer) -> str:
    """'de pasta' / 'de 5 minutos' (same wording as the voice commands)."""
    return f"de {_spoken_name(timer)}"


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
        default_announce_intervals: list[int] = None,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        """Create the manager.

        Args:
            tts_callback: Voice output ``(text, zone_id)``; the application passes
                ``LocalVoiceDelivery.aspeak``, whose ``ResponseResult`` is the ring receipt.
            max_timers: Most running timers at once.
            default_announce_intervals: Seconds-left warning marks for long timers.
            clock: Monotonic seconds for deadlines (default ``time.monotonic``, the
                default event loop's clock).
            wall_clock: Epoch seconds for ``ends_at`` (default ``time.time``).
            sleep: Awaitable sleep for the tick loop and the gap between ring
                repetitions (default ``asyncio.sleep``); tests pass a virtual clock.
        """
        self.tts = tts_callback
        self.max_timers = max_timers
        self.default_announce = list(default_announce_intervals or [TIME_LEFT_WARNING_S])
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._sleep = sleep or asyncio.sleep

        # Timers activos
        self._timers: dict[str, NamedTimer] = {}

        # Task de actualización
        self._update_task: asyncio.Task | None = None
        self._running = False

        # Owned delivery tasks: warnings and rings, by timer id (T030-U2).
        self._deliveries: dict[asyncio.Task, str] = {}
        self._rings: dict[str, asyncio.Task] = {}

        # Callbacks
        self._on_timer_complete: Callable[[NamedTimer], None] | None = None
        self._on_timer_warning: Callable[[NamedTimer, int], None] | None = None

    @property
    def running(self) -> bool:
        """Whether timers are ticking: started, not stopped, and the update task alive.

        Voice commands confirm nothing while this is False (a FeatureManager
        start failure stops the manager, T030-U1).
        """
        return self._running and self._update_task is not None and not self._update_task.done()

    async def start(self):
        """Iniciar el manager de timers"""
        if self._running:
            return
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Timer manager iniciado")

    async def stop(self):
        """Stop ticking, then cancel and await every warning and ring task.

        After it returns no task of this manager is pending and nothing more
        is spoken.
        """
        self._running = False
        task = self._update_task  # kept, done, after stop (lifecycle checks read it)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        deliveries = list(self._deliveries)
        for delivery in deliveries:
            delivery.cancel()
        if deliveries:
            await asyncio.gather(*deliveries, return_exceptions=True)
        logger.info("Timer manager detenido")

    async def _update_loop(self):
        """Tick on deadlines; sleep until the next warning or deadline, at most ``TICK_S``."""
        while self._running:
            wait = TICK_S
            try:
                wait = self._tick()
            except Exception as error:
                logger.error("Timer tick failed: %s", type(error).__name__)
            await self._sleep(wait)

    def _tick(self) -> float:
        """Fire due warnings and completions; return the seconds until the next mark (<= TICK_S).

        Nothing here awaits: warnings and rings start as owned tasks, so a
        ring never delays the deadlines of the other timers.
        """
        now = self._clock()
        for timer in list(self._timers.values()):
            if timer.state is not TimerState.RUNNING or timer.deadline is None:
                continue
            left = timer.deadline - now
            for mark in sorted(timer.announce_at, reverse=True):
                if mark in timer.warned or left > mark:
                    continue
                timer.warned.add(mark)
                if left <= 0 or mark - left > WARNING_STALE_S:
                    logger.debug("Timer warning skipped id=%s seconds_left=%d", timer.timer_id, mark)
                    continue
                self._spawn(timer, self._warn(timer, mark))
            if left <= 0:
                self._complete(timer)
        return self._next_wait(self._clock())

    def _next_wait(self, now: float) -> float:
        wait = TICK_S
        for timer in self._timers.values():
            if timer.state is not TimerState.RUNNING or timer.deadline is None:
                continue
            left = timer.deadline - now
            wait = min(wait, left)
            for mark in timer.announce_at:
                if mark not in timer.warned and left > mark:
                    wait = min(wait, left - mark)
        return max(0.0, wait)

    def _complete(self, timer: NamedTimer) -> None:
        """Mark a due timer COMPLETED and start its ring task (re-arm a repeating one)."""
        due = timer.deadline
        timer.state = TimerState.COMPLETED
        timer.completed_at = datetime.now()
        timer._freeze(keep_ends_at=True)
        timer._left = 0.0
        logger.info("Timer due id=%s", timer.timer_id)
        previous = self._rings.pop(timer.timer_id, None)
        if previous is not None:
            previous.cancel()
        self._rings[timer.timer_id] = self._spawn(timer, self._announce_complete(timer))
        if self._on_timer_complete:
            try:
                self._on_timer_complete(timer)
            except Exception as error:
                logger.error("Timer completion callback failed: %s", type(error).__name__)
        if timer.repeat:
            # Next round from the old deadline, so a repeating timer never drifts.
            timer.state = TimerState.RUNNING
            timer.started_at = datetime.now()
            timer.deadline = due + timer.duration_seconds
            timer.ends_at = (timer.ends_at or self._wall_clock()) + timer.duration_seconds
            timer.warned = {mark for mark in timer.announce_at if mark >= timer.duration_seconds}

    def _spawn(self, timer: NamedTimer, coroutine) -> asyncio.Task:
        """Start an owned delivery task for ``timer``; ``stop()`` cancels and awaits it."""
        task = asyncio.create_task(coroutine)
        self._deliveries[task] = timer.timer_id

        def finished(done: asyncio.Task) -> None:
            self._deliveries.pop(done, None)
            if self._rings.get(timer.timer_id) is done:
                del self._rings[timer.timer_id]
            if not done.cancelled() and done.exception() is not None:
                logger.error("Timer delivery task failed: %s", type(done.exception()).__name__)

        task.add_done_callback(finished)
        return task

    def _cancel_deliveries(self, timer_id: str) -> None:
        """Cancel this timer's warning and ring tasks (no audio after a cancellation)."""
        for task, owner in list(self._deliveries.items()):
            if owner == timer_id:
                task.cancel()

    async def _warn(self, timer: NamedTimer, seconds_left: int) -> None:
        """One owned warning; a failed delivery is logged, never retried (the ring follows)."""
        try:
            await self._announce_warning(timer, seconds_left)
        except VoiceDeliveryError as error:
            logger.warning("Timer warning undelivered id=%s reason=%s", timer.timer_id, error)
        except Exception as error:
            logger.warning("Timer warning failed id=%s error=%s", timer.timer_id, type(error).__name__)

    # ==================== Comandos de Voz ====================

    def handle_voice_command(self, text: str, user_id: str = None, zone_id: str = "default") -> dict:
        """Distinguish a recognized timer request from an observed mutation."""
        text_lower = text.lower().strip()
        if self._is_create_command(text_lower):
            result = self._handle_create(text_lower, user_id, zone_id)
            done = result.get("timer") is not None
            outcome = TaskOutcome(TaskStatus.COMPLETED if done else TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.VOLATILE if done else PersistenceState.NOT_WRITTEN,
                effect=EffectState.OBSERVED if done else EffectState.NOT_OBSERVED,
                observation_scope="software_state" if done else None)
        elif self._is_query_command(text_lower) or self._is_list_command(text_lower):
            listing = self._is_list_command(text_lower)
            result = self._handle_list() if listing else self._handle_query(text_lower)
            name = self._extract_timer_name(text_lower) if not listing else None
            found = listing or not name or self._find_timer_by_name(name) is not None
            outcome = TaskOutcome(TaskStatus.COMPLETED if found else TaskStatus.NEEDS_INPUT)
        else:
            before = {key: timer.state for key, timer in self._timers.items()}
            if self._is_cancel_command(text_lower):
                result = self._handle_cancel(text_lower)
            elif self._is_pause_command(text_lower):
                result = self._handle_pause(text_lower)
            elif self._is_resume_command(text_lower):
                result = self._handle_resume(text_lower)
            else:
                return {"handled": False, "response": "", "timer": None}
            changed = before != {key: timer.state for key, timer in self._timers.items()}
            outcome = TaskOutcome(TaskStatus.COMPLETED if changed else TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.VOLATILE if changed else PersistenceState.NOT_WRITTEN,
                effect=EffectState.OBSERVED if changed else EffectState.NOT_OBSERVED,
                observation_scope="software_state" if changed else None)
        return {**result, "outcome": outcome}

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
            announce_at=warning_thresholds(duration, self.default_announce),
            clock=self._clock,
            wall_clock=self._wall_clock,
        )

        self._timers[timer.timer_id] = timer

        response = f"Timer {name} configurado para {timer.format_remaining()}"
        # Labels are household content: never above DEBUG (design household_conventions.logging).
        logger.info("Timer created id=%s duration_s=%d", timer.timer_id, duration)

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
            for timer in list(self._timers.values()):
                if timer.state == TimerState.RUNNING:
                    count += self.cancel_timer(timer.timer_id)
            return {
                "handled": True,
                "response": f"Cancelé {count} timers" if count else "No hay timers activos",
                "timer": None
            }

        if name:
            timer = self._find_timer_by_name(name)
            if timer and timer.state == TimerState.RUNNING:
                self.cancel_timer(timer.timer_id)
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
            self.cancel_timer(timer.timer_id)
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
            self.pause_timer(timer.timer_id)
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
            self.resume_timer(timer.timer_id)
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
        """Buscar timer por nombre (fuzzy); un timer sin nombre nunca coincide."""
        name_lower = name.lower()
        if not name_lower:
            return None

        # Búsqueda exacta
        for timer in self._timers.values():
            if timer.name and timer.name.lower() == name_lower:
                return timer

        # Búsqueda parcial
        for timer in self._timers.values():
            if timer.name and (name_lower in timer.name.lower() or timer.name.lower() in name_lower):
                return timer

        return None

    # ==================== Anuncios ====================

    async def _announce_warning(self, timer: NamedTimer, seconds_left: int):
        """Announce the time left, naming the timer like its ring does.

        "Al temporizador de pasta le queda 1 minuto." / "Al temporizador de
        5 minutos le quedan 30 segundos." (T030-U1 R4).
        """
        verb = "queda" if seconds_left in (1, 60, 3600) else "quedan"
        message = f"Al temporizador {_describe(timer)} le {verb} {format_duration(seconds_left)}."
        logger.info("Timer warning id=%s seconds_left=%d", getattr(timer, "timer_id", None), seconds_left)

        if self._on_timer_warning:
            self._on_timer_warning(timer, seconds_left)

        if self.tts:
            await invoke_voice_callback(self.tts, message, timer.zone_id)

    async def _announce_complete(self, timer: NamedTimer) -> RingDelivery:
        """Ring: the sentence up to ``RING_REPEATS`` times, ``RING_GAP_S`` apart; keep its receipt.

        Each repetition's ``ResponseResult`` is inspected (not hidden behind
        ``invoke_voice_callback``). A repetition cut after its audio started
        (barge-in) ends the ring as STOPPED; one cancelled before any audio
        (swept from the output queue) is not heard and the ring goes on; a
        failure ends it. Cancelling the task (voice stop, a cancelled repeating
        timer, ``stop()``) cuts the audio of the repetition in progress.

        Returns:
            The final receipt, also left on ``timer.ring``.
        """
        message = f"¡Terminó el temporizador {_describe(timer)}!"
        logger.info("Timer complete id=%s", timer.timer_id)
        current = asyncio.current_task()
        timer.ring = RingDelivery.RINGING
        heard = cut = broke = False
        try:
            for repetition in range(RING_REPEATS):
                if repetition:
                    await self._sleep(RING_GAP_S)
                played = await self._ring_once(message, timer.zone_id)
                if played == "heard":
                    heard = True
                elif played == "cut":
                    cut = True
                    break
                elif played == "swept":
                    continue
                else:
                    broke = played == "partial"
                    break
            receipt = (RingDelivery.STOPPED if cut else RingDelivery.DELIVERED if heard
                       else RingDelivery.RING_FAILED if broke else RingDelivery.UNDELIVERED)
        except asyncio.CancelledError:
            if self._rings.get(timer.timer_id, current) is current and timer.ring is RingDelivery.RINGING:
                timer.ring = RingDelivery.DELIVERED if heard else RingDelivery.UNDELIVERED
            timer.ring_ended_at = self._clock()
            logger.info("Timer ring cancelled id=%s receipt=%s", timer.timer_id, timer.ring.value)
            raise
        if self._rings.get(timer.timer_id, current) is current and timer.ring is RingDelivery.RINGING:
            timer.ring = receipt
        timer.ring_ended_at = self._clock()
        if timer.ring in RING_NOT_HEARD:
            logger.warning("Timer ring not heard id=%s receipt=%s", timer.timer_id, timer.ring.value)
        else:
            logger.info("Timer ring finished id=%s receipt=%s", timer.timer_id, timer.ring.value)
        return timer.ring

    async def _ring_once(self, message: str, zone_id: str) -> str:
        """Speak one repetition and classify it: heard, cut, swept, partial or not_started.

        Mapping (design notice_semantics.delivery_mapping, applied to a ring):
        success -> heard; CANCELLED after output may have started -> cut (the
        household silenced it); CANCELLED before any output -> swept; any other
        failure or a post-queue timeout after output may have started -> partial;
        before any output -> not_started; an output that refused the route
        (VoiceDeliveryError) -> not_started; another exception -> partial
        (unverified). A legacy callback without a ResponseResult keeps the
        ``invoke_voice_callback`` convention: only ``False`` is a failure.
        """
        callback = self.tts
        if callback is None:
            logger.warning("Timer ring has no voice output")
            return "not_started"
        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback(message, zone_id)
            else:
                result = await asyncio.to_thread(callback, message, zone_id)
                if inspect.isawaitable(result):
                    result = await result
        except VoiceDeliveryError as error:
            logger.warning("Timer ring refused by the output: %s", error)
            return "not_started"
        except Exception as error:
            logger.warning("Timer ring delivery unverified: %s", type(error).__name__)
            return "partial"
        if not isinstance(result, ResponseResult):
            return "not_started" if result is False else "heard"
        if result.success:
            return "heard"
        started = result.spoken_segments > 0 or result.output_started
        if result.delivery is DeliveryState.CANCELLED:
            return "cut" if started else "swept"
        if result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is not ResponseStage.QUEUE:
            started = True
        logger.warning("Timer ring repetition failed: delivery=%s", result.delivery.value)
        return "partial" if started else "not_started"

    # ==================== Ring control ====================

    def ringing_timers(self, within_s: float = 0.0) -> list[NamedTimer]:
        """Timers whose ring is sounding, or ended less than ``within_s`` seconds ago, oldest first."""
        now = self._clock()
        found = [t for t in self._timers.values()
                 if t.timer_id in self._rings
                 or (within_s > 0 and t.ring_ended_at is not None and now - t.ring_ended_at < within_s)]
        return sorted(found, key=lambda t: t.created_at)

    async def stop_ring(self, timer_id: str) -> bool:
        """Silence a timer's ring now: no further repetition, the one playing is cut.

        Returns:
            True when a ring was sounding (its receipt is then STOPPED).
        """
        task = self._rings.get(timer_id)
        timer = self._timers.get(timer_id)
        if task is None or task.done() or timer is None:
            return False
        timer.ring = RingDelivery.STOPPED
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        logger.info("Timer ring stopped id=%s", timer_id)
        return True

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
            announce_at=warning_thresholds(duration_seconds, self.default_announce),
            clock=self._clock,
            wall_clock=self._wall_clock,
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
        """Cancel a running or paused timer by id.

        Returns:
            True only when the timer existed and changed to CANCELLED.
        """
        timer = self._timers.get(timer_id)
        if timer is None or timer.state not in (TimerState.RUNNING, TimerState.PAUSED):
            return False
        timer._freeze()
        timer.state = TimerState.CANCELLED
        self._cancel_deliveries(timer_id)
        return True

    def pause_timer(self, timer_id: str) -> bool:
        """Pause a running timer by id; True only when it changed to PAUSED."""
        timer = self._timers.get(timer_id)
        if timer is None or timer.state is not TimerState.RUNNING:
            return False
        timer._freeze()
        timer.state = TimerState.PAUSED
        return True

    def resume_timer(self, timer_id: str) -> bool:
        """Resume a paused timer by id; True only when it changed to RUNNING."""
        timer = self._timers.get(timer_id)
        if timer is None or timer.state is not TimerState.PAUSED:
            return False
        timer._arm(timer._left)
        timer.state = TimerState.RUNNING
        timer.started_at = datetime.now()
        return True

    def cleanup_completed(self, max_age_hours: int = 24):
        """Limpiar timers completados/cancelados antiguos"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            tid for tid, t in self._timers.items()
            if t.state in [TimerState.COMPLETED, TimerState.CANCELLED]
            and t.created_at < cutoff and tid not in self._rings
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
            "running": self.running,
            # Timers do not survive a restart until SCHEMA-V2 (T030-U3).
            "persistence": PersistenceState.VOLATILE.value,
            "total_timers": len(self._timers),
            "active": sum(1 for t in self._timers.values() if t.state == TimerState.RUNNING),
            "paused": sum(1 for t in self._timers.values() if t.state == TimerState.PAUSED),
            "completed": sum(1 for t in self._timers.values() if t.state == TimerState.COMPLETED),
            "ringing": len(self._rings),
            # Rings nobody heard (T030-U2): never counted as a delivered completion.
            "ring_not_heard": sum(1 for t in self._timers.values() if t.ring in RING_NOT_HEARD),
            "delivery_tasks": len(self._deliveries),
        }
