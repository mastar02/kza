"""
Alert Manager Module
Sistema central de gestión de alertas con deduplicación, historial y notificaciones de voz.

Proporciona:
- Definición de prioridades y tipos de alerta
- Deduplicación por condición (tipo, fuente, condición), nunca por el texto
- Latch por episodio: una alerta al abrirse la condición; otra sólo después de
  cerrarla (resolve_condition) o, si se configura, tras un intervalo explícito
- Recibo de entrega de voz en la alerta (delivered/failed/blocked/muted)
- Un único reintento del aviso en el siguiente chequeo si falló sin sonar nada
- Registro de handlers por tipo de alerta
- Historial de alertas con límite configurable y timestamps con zona horaria
- Resumen de alertas pendientes y activas

Uso:
    from src.alerts import AlertCondition, AlertManager, AlertPriority, AlertType

    alert_manager = AlertManager(tts_callback=response_handler.aspeak)

    # Registrar handler para alertas de seguridad
    alert_manager.register_handler(
        AlertType.SECURITY,
        lambda alert: print(f"Alert: {alert.message}")
    )

    # Crear alerta
    alert = await alert_manager.create_alert(
        alert_type=AlertType.SECURITY,
        priority=AlertPriority.CRITICAL,
        message="Puerta principal abierta",
        details={"zone": "entrada"},
        condition=AlertCondition(source="entrada", condition="door_open_unexpected"),
    )

    # Al cerrarse la puerta el episodio termina; la próxima apertura alerta de nuevo
    await alert_manager.resolve_condition(
        AlertType.SECURITY, AlertCondition(source="entrada", condition="door_open_unexpected")
    )
"""

import asyncio
import inspect
import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Hashable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Callable
from uuid import uuid4

from src.core.household_clock import HouseholdClock, RoomNamespace
from src.core.logging import get_logger
from src.core.observation import Observation
from src.core.sqlite_work import AuthorityGuard
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, ResponseStage, VoiceDeliveryError
from src.reminders.reminder_store import AlertZonesPreference, ReminderStore

logger = get_logger(__name__)


class AlertPriority(Enum):
    """Niveles de prioridad de alertas"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class AlertType(StrEnum):
    """Tipos de alertas"""
    SECURITY = "security"       # Seguridad (puertas, movimiento)
    PATTERN = "pattern"         # Patrones anómalos
    DEVICE = "device"          # Estado de dispositivos
    REMINDER = "reminder"       # Recordatorios
    WELLNESS = "wellness"       # Bienestar


# Priorities spoken by voice; lower ones only stay in the history (web).
SPOKEN_PRIORITIES = frozenset({AlertPriority.CRITICAL, AlertPriority.HIGH})
QUIET_START_HOUR = 23
QUIET_END_HOUR = 7
_POLICY_MUTE_REASONS = frozenset({
    "alert_quiet_hours", "alert_global_mute", "alert_mute_unavailable", "alert_no_output_zones",
})


@dataclass(frozen=True)
class AlertProducerStatus:
    """Current coverage of a shared producer, without entity IDs or payloads."""

    name: str
    observation: Observation
    connected: bool
    known: int
    unknown: int
    overflow: int
    uncertain: bool


class AlertVoiceOutput:
    """Use the household's durable zone selection and local clock for alerts."""

    def __init__(
        self, speak: Callable[..., Awaitable[ResponseResult]], rooms: RoomNamespace, fallback: str | None,
        clock: HouseholdClock, store: ReminderStore | None,
        *, enabled: bool, output_key: Callable[[str], Hashable | None] | None,
    ) -> None:
        self._speak = speak
        self._rooms = rooms
        self._fallback = self._rooms.to_zone(fallback)
        self._clock = clock
        self._store = store
        self._muted: Callable[[], bool] | None = None
        self._output_key = output_key
        self.enabled = enabled

    def bind_mute(self, muted: Callable[[], bool]) -> None:
        """Attach the application's current mute policy before producer startup."""
        self._muted = muted

    @property
    def writable(self) -> bool:
        """Whether a durable store is wired for this installation."""
        return self._store is not None

    @property
    def configured_zones(self) -> list[str]:
        """List the configured household rooms in output-zone form."""
        return [zone for room in sorted(self._rooms.rooms) if (zone := self._rooms.to_zone(room))]

    def selected_zones(self, preference: AlertZonesPreference | None) -> list[str]:
        """Resolve a missing or reset preference, and skip retired rooms."""
        if preference is None or preference.zones is None:
            return [self._fallback] if self._fallback is not None else []
        return list(dict.fromkeys(zone for value in preference.zones if (zone := self._rooms.to_zone(value))))

    async def get_preference(self, *, guard: AuthorityGuard | None = None) -> AlertZonesPreference | None:
        """Load the current preference through the shared SQLite owner."""
        return None if self._store is None else await self._store.get_alert_zones(guard=guard)

    async def set_preference(
        self, zones: list[str] | None, *, expected_revision: int | None, guard: AuthorityGuard,
    ) -> AlertZonesPreference:
        """Validate room IDs and persist an ordered zone selection or reset."""
        if self._store is None:
            raise RuntimeError("alert_zones_store_unavailable")
        selected = None
        if zones is not None:
            selected = []
            for value in zones:
                zone = self._rooms.to_zone(value)
                if zone is None:
                    raise ValueError("invalid_alert_zone")
                if zone not in selected:
                    selected.append(zone)
        return await self._store.set_alert_zones(selected, expected_revision=expected_revision, guard=guard)

    def mute_reason(self, now: datetime | None = None) -> str | None:
        """Check the current mute state before queueing an episode or any output."""
        if not self.enabled:
            return "voice_notifications_disabled"
        if self._muted is None:
            return "alert_mute_unavailable"
        if self._muted():
            return "alert_global_mute"
        if not QUIET_END_HOUR <= (now or self._clock.now_local()).hour < QUIET_START_HOUR:
            return "alert_quiet_hours"
        return None

    async def speak(self, text: str, *, is_current: Callable[[], bool] | None = None) -> ResponseResult:
        """Deliver once per selected physical output and combine delivery receipts."""
        reason = self.mute_reason()
        if reason is not None:
            return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.BLOCKED, reason)
        zones = self.selected_zones(await self.get_preference())
        if not zones:
            return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.BLOCKED, "alert_no_output_zones")
        results: list[ResponseResult] = []
        seen = set()
        for zone in zones:
            now = self._clock.now_local()
            reason = self.mute_reason(now)
            if is_current is not None and not is_current():
                reason = "alert_evidence_unavailable"
            if reason is not None:
                results.append(ResponseResult(text, GenerationState.COMPLETE, DeliveryState.BLOCKED, reason))
                break
            key = self._output_key(zone) if self._output_key is not None else None
            if key is None:
                results.append(ResponseResult(text, GenerationState.COMPLETE, DeliveryState.BLOCKED,
                                              "voice_output_zone_unavailable"))
            elif key not in seen:
                seen.add(key)
                cutoff = now.replace(hour=QUIET_START_HOUR, minute=0, second=0, microsecond=0)
                results.append(await self._speak(text, zone_id=zone, timeout_s=(cutoff - now).total_seconds()))
        # A whole-episode retry is safe only if every route proved no audio began.
        receipt = next((result for result in results if not result.success), results[0])
        if all(_failed_without_output(result) for result in results):
            return receipt
        if any(result.success or result.output_started or result.spoken_segments for result in results):
            return replace(receipt, output_started=True)
        if _failed_without_output(receipt):
            return next(result for result in results if not _failed_without_output(result))
        return receipt


class AlertDelivery(StrEnum):
    """Voice delivery receipt of an alert's notice."""

    NOT_SPOKEN = "not_spoken"   # Priority below the voice threshold; web only
    PENDING = "pending"         # Notice in flight
    DELIVERED = "delivered"     # The output confirmed a complete delivery
    FAILED = "failed"           # The output failed, raised or did not confirm
    BLOCKED = "blocked"         # The output refused the notice (policy/route)
    MUTED = "muted"             # Voice notices disabled; nothing was attempted


def _utc_now() -> datetime:
    """Return the current instant with an explicit UTC offset."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class AlertCondition:
    """Identity of the condition an alert reports; the dedup key never uses the text.

    Attributes:
        source: Entity or producer source (entity_id, device name, zone).
        condition: Stable condition name (e.g. "battery_low", "door_open_unexpected").
        latched: True for a stateful condition whose episode stays open until
            resolve_condition; False for a momentary event whose episode closes
            by itself after the manager's cooldown_seconds.
        reannounce: Whether interval-based announcements may repeat this episode.
        retain_if_history_full: Keep the active episode visible until it resolves.
    """

    source: str
    condition: str
    latched: bool = True
    reannounce: bool = True
    retain_if_history_full: bool = False

    def key(self, alert_type: AlertType) -> tuple[str, str, str]:
        """Return the deduplication key of this condition for an alert type."""
        return (alert_type.value, self.source, self.condition)


@dataclass
class Alert:
    """Representa una alerta individual"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    processed: bool = False
    processed_at: datetime | None = None
    source: str | None = None
    condition: str | None = None
    last_seen_at: datetime | None = None
    resolved_at: datetime | None = None
    delivery: AlertDelivery = AlertDelivery.NOT_SPOKEN
    delivery_reason: str | None = None
    delivery_at: datetime | None = None
    delivery_attempts: int = 0

    def to_dict(self) -> dict:
        """Convertir alerta a diccionario"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "priority": self.priority.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "source": self.source,
            "condition": self.condition,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "delivery": self.delivery.value,
            "delivery_reason": self.delivery_reason,
            "delivery_at": self.delivery_at.isoformat() if self.delivery_at else None,
            "delivery_attempts": self.delivery_attempts,
        }


@dataclass
class _Episode:
    """Open episode of one condition key (latched until resolved or expired)."""

    alert: Alert
    latched: bool
    opened_at: float
    reannounce: bool = True
    announced_at: float | None = None
    # The last notice failed without any audio and is retried once at the next check.
    retry_due: bool = False
    retain_if_history_full: bool = False
    defer_voice: bool = False
    pending_notice: bool | None = None  # None: no job; bool: whether to retry
    voice_in_flight: bool = False
    voice_dispatched: bool = False


# Failed delivery states that can prove no audio was produced (with output_started
# False and no spoken segment). BLOCKED is a refusal by policy and is never retried.
_OUTPUT_FREE_STATES = frozenset({DeliveryState.FAILED, DeliveryState.EMPTY, DeliveryState.CANCELLED})


def _failed_without_output(result: Any) -> bool:
    """Whether a voice result proves the notice failed before any audio started.

    Mirrors the reminder notice mapping (T034): only a typed ResponseResult can
    prove it. Output started (a PCM chunk or a spoken segment), a blocked notice,
    a timeout after the queue stage, an exception or a receipt-less return are
    never treated as output-free, so they are never spoken again.

    Args:
        result: What the voice output returned.

    Returns:
        True when the notice may be retried without risking a repeated sound.
    """
    if not isinstance(result, ResponseResult) or result.success:
        return False
    if result.output_started or result.spoken_segments > 0:
        return False
    if result.delivery is DeliveryState.TIMED_OUT:
        return result.timeout_stage is ResponseStage.QUEUE
    return result.delivery in _OUTPUT_FREE_STATES


def _delivery_outcome(result: Any) -> tuple[AlertDelivery, str | None]:
    """Map what the voice output returned to the alert's delivery receipt.

    Args:
        result: The output's return value (ResponseResult for application outputs).

    Returns:
        (delivery state, diagnostic reason).
    """
    if isinstance(result, ResponseResult):
        if result.success:
            return AlertDelivery.DELIVERED, None
        if result.delivery is DeliveryState.BLOCKED:
            if result.error in _POLICY_MUTE_REASONS and not result.output_started:
                return AlertDelivery.MUTED, result.error
            return AlertDelivery.BLOCKED, result.error or "voice_delivery_blocked"
        return AlertDelivery.FAILED, result.error or "voice_delivery_" + result.delivery.value
    if result is False:
        return AlertDelivery.FAILED, "voice_delivery_failed"
    # Legacy callbacks without a receipt keep the shared acknowledgement convention
    # of src.core.voice_delivery.invoke_voice_callback.
    return AlertDelivery.DELIVERED, None


class AlertManager:
    """
    Gestor central de alertas con soporte para:
    - Deduplicación con cooldown
    - Notificaciones de voz
    - Handlers por tipo de alerta
    - Historial persistente
    """

    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        max_history: int = 1000,
        tts_callback: Callable[[str], Any] | None = None,
        reannounce_interval_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], datetime] | None = None,
        voice_output: AlertVoiceOutput | None = None,
    ):
        """
        Inicializar AlertManager.

        Args:
            cooldown_seconds: Duración del episodio de un evento momentáneo
                (AlertCondition.latched=False, o sin condición explícita).
            max_history: Máximo número de alertas en historial (default: 1000)
            tts_callback: Salida de voz; recibe el texto y devuelve su ResponseResult.
                None significa avisos de voz desactivados (recibo 'muted').
            reannounce_interval_seconds: Re-anunciar un episodio que sigue abierto
                y sin reconocer cada este intervalo. None (default) = un solo aviso
                por episodio. Una alerta reconocida no se vuelve a decir salvo que
                empeore (T036-U1-Q5).
            clock: Reloj monotónico inyectable (segundos) para episodios.
            now: Fábrica de instantes con zona horaria para los timestamps.
            voice_output: Salida de hogar con preferencia de zonas y política Q20.
        """
        if reannounce_interval_seconds is not None and (
            type(reannounce_interval_seconds) not in (int, float)
            or not math.isfinite(reannounce_interval_seconds)
            or reannounce_interval_seconds <= 0
        ):
            raise ValueError("reannounce_interval_seconds must be None or finite and positive")
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.tts_callback = tts_callback
        self.voice_output = voice_output
        self.producer_status: Callable[[], tuple[AlertProducerStatus, ...]] | None = None
        self.reannounce_interval_seconds = reannounce_interval_seconds
        self._clock = clock or time.monotonic
        self._now = now or _utc_now

        # Handlers por tipo de alerta: {AlertType: [callback1, callback2, ...]}
        self._handlers: dict[AlertType, list[Callable[[Alert], None]]] = defaultdict(list)

        # Episodios abiertos por condición: {(type, source, condition): _Episode}
        self._episodes: dict[tuple[str, str, str], _Episode] = {}

        # Historial de alertas (FIFO)
        self._history: list[Alert] = []

        # Lock para operaciones thread-safe
        self._lock = asyncio.Lock()

        logger.info(
            "AlertManager initialized (cooldown=%ss, reannounce=%s, max_history=%s)",
            cooldown_seconds, reannounce_interval_seconds, max_history,
        )

    def register_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None],
    ) -> None:
        """
        Registrar handler para un tipo de alerta.

        Args:
            alert_type: Tipo de alerta
            handler: Función que recibe Alert. Puede ser sync o async.

        Ejemplo:
            manager.register_handler(
                AlertType.SECURITY,
                lambda alert: print(f"Security: {alert.message}")
            )
        """
        self._handlers[alert_type].append(handler)
        logger.debug(f"Handler registered for {alert_type.value}")

    def unregister_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None],
    ) -> None:
        """Desregistrar handler"""
        if handler in self._handlers[alert_type]:
            self._handlers[alert_type].remove(handler)
            logger.debug(f"Handler unregistered for {alert_type.value}")

    async def create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        message: str,
        details: dict | None = None,
        *,
        condition: AlertCondition | None = None,
        defer_voice: bool = False,
    ) -> Alert | None:
        """
        Crear una alerta con latch por episodio de su condición.

        The dedup key is (type, source, condition), never the message text. While
        the condition's episode is open, a repeated observation creates nothing; a
        more severe priority escalates the open alert in place and announces it.
        Without an explicit condition the message identifies a momentary event
        (legacy producers), whose episode closes after cooldown_seconds.

        Args:
            alert_type: Tipo de alerta
            priority: Prioridad de la alerta
            message: Mensaje descriptivo (voz y web, en español)
            details: Detalles adicionales (dict)
            condition: Identidad de la condición que reporta la alerta
            defer_voice: Let the owning producer deliver notices outside state evaluation.

        Returns:
            La alerta si se abrió un episodio o se escaló; None si el episodio ya
            estaba abierto (deduplicada, aunque reintente o re-anuncie su aviso).
        """
        cond = condition or AlertCondition(source="message", condition=message, latched=False)
        key = cond.key(alert_type)
        async with self._lock:
            now = self._clock()
            episode = self._episodes.get(key)
            if episode is not None and not self._is_live(episode, now):
                del self._episodes[key]
                episode = None

            if episode is not None:
                return await self._observe_open_episode(episode, priority, message, details, now)

            alert = Alert(
                alert_id=str(uuid4())[:8],
                alert_type=alert_type,
                priority=priority,
                message=message,
                details=details or {},
                timestamp=self._now(),
                source=cond.source,
                condition=cond.condition,
            )
            episode = _Episode(alert=alert, latched=cond.latched, opened_at=now,
                               reannounce=cond.reannounce,
                               retain_if_history_full=cond.latched and cond.retain_if_history_full,
                               defer_voice=defer_voice)
            self._episodes[key] = episode

            # Agregar al historial
            self._history.append(alert)
            self._evict_overflow(now)

            logger.info(
                "Alert created: %s (id=%s, priority=%s, source=%s, condition=%s)",
                alert.alert_type.value, alert.alert_id, alert.priority.name, cond.source, cond.condition,
            )

            # Ejecutar handlers
            await self._execute_handlers(alert)

            # Notificación de voz para alertas críticas/altas
            if priority in SPOKEN_PRIORITIES:
                await self._announce(episode)

            return alert

    def _is_live(self, episode: _Episode, now: float) -> bool:
        """Whether an episode is still open: latched, or momentary within cooldown."""
        return episode.latched or now - episode.opened_at < self.cooldown_seconds

    def _evict_overflow(self, now: float) -> None:
        """Trim the history to max_history without hiding an open episode (caller holds the lock).

        The oldest alert not owned by a live episode (resolved, expired momentary
        event, or legacy) goes first. Active episodes with fixed producer keys
        can remain visible independently of the bounded history.
        """
        while len(self._history) > self.max_history:
            owners = {id(ep.alert): key for key, ep in self._episodes.items()}
            live = {id(ep.alert) for ep in self._episodes.values() if self._is_live(ep, now)}
            index = next((i for i, a in enumerate(self._history) if id(a) not in live), 0)
            evicted = self._history.pop(index)
            key = owners.get(id(evicted))
            if key is None:
                continue
            if self._episodes[key].retain_if_history_full:
                continue
            del self._episodes[key]
            if id(evicted) in live:
                logger.warning("Alert history full of open episodes; evicted open alert %s (id=%s)",
                               evicted.alert_type.value, evicted.alert_id)

    async def _observe_open_episode(
        self,
        episode: _Episode,
        priority: AlertPriority,
        message: str,
        details: dict | None,
        now: float,
    ) -> Alert | None:
        """Apply a repeated observation of an open episode (caller holds the lock)."""
        alert = episode.alert
        alert.last_seen_at = self._now()
        if priority.value < alert.priority.value:
            alert.priority = priority
            alert.message = message
            alert.details = details or {}
            # The acknowledgement covered a less severe reading: the escalated alert
            # is active again (web list and summary) until acknowledged anew.
            alert.processed = False
            alert.processed_at = None
            logger.info("Alert escalated: %s (id=%s, priority=%s)",
                        alert.alert_type.value, alert.alert_id, priority.name)
            await self._execute_handlers(alert)
            if priority in SPOKEN_PRIORITIES:
                await self._announce(episode)
            return alert
        if alert.processed:
            # User answer T036-U1-Q5 (option A): an acknowledged alert is never spoken
            # again, neither by the retry nor by the re-announce. Only an escalation
            # (above) reopens the acknowledgement and speaks.
            if episode.retry_due:
                episode.retry_due = False
                logger.info("Alert voice retry dropped, alert acknowledged (id=%s)", alert.alert_id)
            logger.debug("Alert not spoken again, acknowledged (id=%s)", alert.alert_id)
            return None
        if episode.pending_notice is not None or episode.voice_in_flight:
            return None
        if episode.retry_due:
            episode.retry_due = False
            logger.info("Alert voice notice retried: %s (id=%s)", alert.alert_type.value, alert.alert_id)
            await self._announce(episode, retry=True)
            return None
        if (
            self.voice_output is None
            and episode.latched
            and episode.reannounce
            and self.reannounce_interval_seconds is not None
            and episode.announced_at is not None
            and now - episode.announced_at >= self.reannounce_interval_seconds
        ):
            logger.info("Alert re-announced: %s (id=%s)", alert.alert_type.value, alert.alert_id)
            await self._announce(episode)
            return None
        logger.debug("Alert deduped (open episode): %s (id=%s)", alert.alert_type.value, alert.alert_id)
        return None

    async def resolve_condition(self, alert_type: AlertType, condition: AlertCondition) -> Alert | None:
        """Close the condition's open episode so its next opening alerts again.

        Args:
            alert_type: Tipo de alerta de la condición.
            condition: Identidad de la condición que dejó de cumplirse.

        Returns:
            La alerta del episodio cerrado, o None si no había episodio abierto.
        """
        async with self._lock:
            episode = self._episodes.pop(condition.key(alert_type), None)
            if episode is None:
                return None
            episode.alert.resolved_at = self._now()
            self._drop_pending_notice(episode, "alert_condition_resolved")
            logger.info("Alert condition resolved: %s (id=%s, source=%s, condition=%s)",
                        alert_type.value, episode.alert.alert_id, condition.source, condition.condition)
            return episode.alert

    async def _execute_handlers(self, alert: Alert) -> None:
        """Ejecutar todos los handlers registrados para el tipo de alerta"""
        handlers = self._handlers.get(alert.alert_type, [])

        for handler in handlers:
            try:
                # Soportar tanto handlers sync como async
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(
                    f"Error executing alert handler: {e} (type={alert.alert_type.value}, id={alert.alert_id})"
                )

    async def _announce(self, episode: _Episode, *, retry: bool = False) -> None:
        """Speak the episode's alert and record its delivery receipt.

        A notice that failed before any audio started is retried once, at the
        episode's next observation; a retry is never retried again (user answer
        T036-U1-Q1). Output that started, a blocked or muted notice, and every
        other failure are never repeated. Neither the retry nor the re-announce
        speaks an acknowledged alert; only an escalation does (T036-U1-Q5).

        Args:
            episode: The open episode whose alert is spoken.
            retry: This notice is the single retry of an output-free failure.
        """
        if self.voice_output is not None:
            if episode.voice_dispatched and not retry:
                return
            reason = self.voice_output.mute_reason()
            if reason is not None:
                episode.voice_dispatched = True
                episode.pending_notice = None
                self._record_delivery(episode.alert, AlertDelivery.MUTED, reason)
                return
        episode.retry_due = False
        if episode.defer_voice:
            episode.pending_notice = retry
            if episode.alert.delivery_attempts == 0:
                self._record_delivery(episode.alert, AlertDelivery.PENDING, None)
            return
        episode.announced_at = self._clock()
        episode.voice_dispatched = True
        output_free = await self._notify_voice(episode.alert)
        episode.retry_due = output_free and not retry
        if episode.retry_due:
            logger.info("Alert voice notice failed before any audio; one retry at the next check (id=%s)",
                        episode.alert.alert_id)

    async def deliver_deferred_notice(self, alert_type: AlertType, condition: AlertCondition,
                                      *, is_current: Callable[[], bool]) -> None:
        """Deliver one pending notice while leaving the episode lock free during playback.

        Args:
            alert_type: Type of the episode that owns the notice.
            condition: Stable identity of the episode that owns the notice.
            is_current: Synchronous admission check of the producer's evidence and turn state.
        """
        async with self._lock:
            episode = self._episodes.get(condition.key(alert_type))
            if (episode is None or episode.pending_notice is None or episode.voice_in_flight
                    or episode.alert.processed or not is_current()):
                return
            retry, episode.pending_notice = episode.pending_notice, None
            episode.voice_in_flight = True
            episode.voice_dispatched = True
            episode.announced_at = self._clock()
        output_free = False
        try:
            output_free = await self._notify_voice(
                episode.alert, is_current=lambda: not episode.alert.processed and is_current(),
            )
        finally:
            episode.voice_in_flight = False
            episode.retry_due = (output_free and not retry and not episode.alert.processed
                                 and episode.pending_notice is None)

    def _drop_pending_notice(self, episode: _Episode, reason: str) -> None:
        if episode.pending_notice is not None:
            episode.pending_notice = None
            if not episode.voice_in_flight and episode.alert.delivery_attempts == 0:
                self._record_delivery(episode.alert, AlertDelivery.BLOCKED, reason)

    async def _notify_voice(self, alert: Alert, *, is_current: Callable[[], bool] | None = None) -> bool:
        """Speak the alert and record the output's receipt on it, never swallowing a failure.

        Returns:
            True when the notice failed before any audio started (retryable).
        """
        callback = self.tts_callback
        if callback is None and self.voice_output is None:
            self._record_delivery(alert, AlertDelivery.MUTED, "voice_notifications_disabled")
            return False
        message = self._format_voice_message(alert)
        alert.delivery_attempts += 1
        self._record_delivery(alert, AlertDelivery.PENDING, None)
        try:
            if self.voice_output is not None:
                result = await self.voice_output.speak(message, is_current=is_current)
            elif inspect.iscoroutinefunction(callback):
                result = await callback(message)
            else:
                result = await asyncio.to_thread(callback, message)
                if inspect.isawaitable(result):
                    result = await result
        except asyncio.CancelledError:
            self._record_delivery(alert, AlertDelivery.FAILED, "voice_delivery_cancelled")
            raise
        except VoiceDeliveryError as error:
            self._record_delivery(alert, AlertDelivery.FAILED, str(error) or "voice_delivery_failed")
            return False
        except Exception as error:
            self._record_delivery(alert, AlertDelivery.FAILED, type(error).__name__)
            return False
        self._record_delivery(alert, *_delivery_outcome(result))
        return _failed_without_output(result)

    def _record_delivery(self, alert: Alert, delivery: AlertDelivery, reason: str | None) -> None:
        """Store the voice receipt on the alert and log anything not delivered."""
        alert.delivery = delivery
        alert.delivery_reason = reason
        alert.delivery_at = None if delivery is AlertDelivery.PENDING else self._now()
        if delivery in (AlertDelivery.FAILED, AlertDelivery.BLOCKED):
            logger.warning("Alert voice notice %s (id=%s, reason=%s)", delivery.value, alert.alert_id, reason)
        elif delivery is AlertDelivery.MUTED:
            logger.info("Alert voice notice muted (id=%s, reason=%s)", alert.alert_id, reason)

    def _format_voice_message(self, alert: Alert) -> str:
        """Formatear mensaje para notificación de voz"""
        priority_text = {
            AlertPriority.CRITICAL: "Alerta crítica: ",
            AlertPriority.HIGH: "Alerta importante: ",
        }
        prefix = priority_text.get(alert.priority, "")
        return f"{prefix}{alert.message}"

    def get_history(self, limit: int | None = None) -> list[Alert]:
        """
        Obtener historial de alertas.

        Args:
            limit: Número máximo de alertas a retornar (más recientes primero)

        Returns:
            Lista de alertas (en orden inverso: más recientes primero)
        """
        history = list(reversed(self._history))
        if limit:
            history = history[:limit]
        return history

    def get_pending_summary(self) -> dict:
        """
        Obtener resumen de alertas pendientes (no procesadas).

        Returns:
            Dict con conteo por tipo y prioridad
        """
        pending = [a for a in self._history if not a.processed]

        summary = {
            "total_pending": len(pending),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int),
        }

        for alert in pending:
            summary["by_type"][alert.alert_type.value] += 1
            summary["by_priority"][alert.priority.name] += 1

        # Convertir defaultdict a dict
        return {
            "total_pending": summary["total_pending"],
            "by_type": dict(summary["by_type"]),
            "by_priority": dict(summary["by_priority"]),
        }

    def get_active_alerts(self) -> list[Alert]:
        """Return alerts still unacknowledged whose condition was not resolved, newest first."""
        return [a for a in self._alerts_with_retained_episodes() if not a.processed and a.resolved_at is None]

    def _alerts_with_retained_episodes(self) -> list[Alert]:
        history_ids = {id(alert) for alert in self._history}
        retained = [episode.alert for episode in reversed(self._episodes.values())
                    if episode.retain_if_history_full and id(episode.alert) not in history_ids]
        return list(reversed(self._history)) + retained

    async def mark_processed(self, alert_id: str, *, before_publish: Callable[[], None] | None = None,
                             update_timestamp: bool = True,
                             timestamp_factory: Callable[[], datetime] | None = None) -> bool:
        """Recheck original authority after the lock and before publishing alert state."""
        async with self._lock:
            if before_publish is not None and before_publish() is False:
                raise PermissionError("alert_authority_retired")
            for alert in self._alerts_with_retained_episodes():
                if alert.alert_id == alert_id:
                    timestamp = (timestamp_factory or self._now)() if update_timestamp else None
                    if before_publish is not None and before_publish() is False:
                        raise PermissionError("alert_authority_retired")
                    alert.processed = True
                    if update_timestamp:
                        alert.processed_at = timestamp
                    if alert.source is not None and alert.condition is not None:
                        episode = self._episodes.get((alert.alert_type.value, alert.source, alert.condition))
                        if episode is not None and episode.alert is alert:
                            self._drop_pending_notice(episode, "alert_acknowledged")
                    logger.debug("Alert marked as processed: %s", alert_id)
                    return True
            return False

    def get_alert(self, alert_id: str) -> Alert | None:
        """Obtener alerta por ID"""
        for alert in self._alerts_with_retained_episodes():
            if alert.alert_id == alert_id:
                return alert
        return None

    def clear_cooldowns(self) -> None:
        """Close every open episode; the next observation of a condition alerts again.

        Each cleared episode's alert is marked resolved, so a condition never has
        two active alerts (the cleared one and the one its next observation opens).
        """
        resolved_at = self._now()
        for episode in self._episodes.values():
            episode.alert.resolved_at = resolved_at
            self._drop_pending_notice(episode, "alert_condition_resolved")
        self._episodes.clear()
        logger.debug("All alert episodes cleared")

    def clear_history(self) -> None:
        """Limpiar historial de alertas"""
        self._history.clear()
        logger.debug("Alert history cleared")

    def get_stats(self) -> dict:
        """Obtener estadísticas del sistema de alertas"""
        return {
            "total_alerts": len(self._history),
            "pending_alerts": sum(1 for a in self._history if not a.processed),
            "processed_alerts": sum(1 for a in self._history if a.processed),
            "dedup_keys": len(self._episodes),
            "by_type": {
                alert_type.value: sum(1 for a in self._history if a.alert_type == alert_type)
                for alert_type in AlertType
            },
            "by_priority": {
                priority.name: sum(1 for a in self._history if a.priority == priority)
                for priority in AlertPriority
            },
        }
