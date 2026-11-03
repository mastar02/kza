"""
Home Assistant Client
Comunicacion con Home Assistant via REST API y WebSocket.

Graceful degradation: all methods return safe defaults when HA is
unavailable so the voice server keeps running.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from contextvars import Context
from enum import StrEnum
from time import monotonic
from typing import Callable

import aiohttp

from src.core.owned_work import OwnedBlockingWork
from src.home_assistant.effect_plans import MAX_EFFECTS
from src.home_assistant.operations import ActionIntent, DOMAIN_OPERATIONS, action_capabilities, finite_number, matches_observation
from src.home_assistant.state_projection import (
    MAX_FRAME_BYTES, EntityObservation, apply_delta, apply_initial, apply_rest, apply_rest_failure,
    apply_rest_snapshot, decode_message, decode_states_message, empty_frame, entity_observation,
    legacy_frame, observation_view,
)


def _raw_state_json(raw):
    """Transfer raw JSON to the bounded native decoder without parsing here."""
    return raw








def _state_projection_cpu(kind, base, raw, options, callback_count=0):
    """Compute a bounded immutable replacement; never publish or call observers."""
    replacement = result = notices = decoded = value = None
    try:
        if type(raw) is tuple:
            raw = b''.join(raw)
        if kind == 'decode':
            return None, decode_message(raw), ()
        if kind == 'snapshot':
            return None, decode_states_message(raw), ()
        if kind in ('initial', 'delta'):
            decoded = decode_message(raw) if type(raw) in (str, bytes) else raw
            subscribe_id = options.get('subscribe_id')
            if subscribe_id is not None:
                decoded = HomeAssistantClient._entities_payload(decoded, subscribe_id)
            if kind == 'initial':
                replacement = apply_initial(base, decoded, generation=options['stream_generation'],
                    observed_at=options['observed_at'], ttl_s=STATE_OBSERVATION_TTL_S,
                    admitted_revision=options.get('admitted_revision'))
            else:
                replacement = apply_delta(base, decoded, observed_at=options['observed_at'],
                    ttl_s=STATE_OBSERVATION_TTL_S)
        elif kind in ('rest', 'failure'):
            entity_id = options['entity_id']
            expected = dict(expected_generation=options['expected_generation'],
                expected_revision=options['expected_revision'], attempted_at=options['attempted_at'])
            if kind == 'rest':
                decoded = None if options.get('missing') else decode_message(raw)
                if decoded is None and not options.get('missing'):
                    raise ValueError('invalid_entity_state')
                replacement = apply_rest(base, entity_id, decoded,
                    observed_at=options['observed_at'], ttl_s=STATE_OBSERVATION_TTL_S, **expected)
            else:
                replacement = apply_rest_failure(base, entity_id, reason=options['reason'], **expected)
            result = entity_observation(replacement, entity_id, now=options['now'])
            if kind == 'failure' and replacement is not base and entity_id not in replacement.entries:
                result = observation_view(entity_observation(base, entity_id, now=options['now']),
                    exists=None, reason=options['reason'], attempted_at=options['attempted_at'])
            if replacement is base:
                result = observation_view(result, exists=None, reason='source_changed',
                    attempted_at=options['attempted_at'])
            if result.source is None and result._value is None:
                result = replace(result, source='ha-rest-entity')
        elif kind == 'bulk':
            expected = {value['entity_id']: options['base'].entry_revision(value['entity_id'])
                for value in raw.states}
            replacement = apply_rest_snapshot(base, raw.states,
                observed_at=raw.observed_at.timestamp(), expires_at=raw.expires_at.timestamp(),
                attempted_at=None if raw.attempted_at is None else raw.attempted_at.timestamp(),
                expected_generation=options['expected_generation'], expected_revisions=expected)
        else:
            raise ValueError('invalid_state_projection_kind')
        notices = []
        if callback_count and kind == 'delta':
            for identity in replacement.changed_ids + replacement.removed_ids:
                value = entity_observation(replacement, identity, now=options['now'])
                admitted = value.current(options['now']) and value.exists is True
                notices.append((identity, tuple(value.state if admitted else None
                    for _ in range(callback_count))))
        return replacement, result, tuple(notices)
    except Exception:
        pass
    finally:
        base = raw = options = replacement = result = notices = decoded = value = None
        expected = identity = admitted = None
    raise ValueError('invalid_ha_state_projection')


class ServiceCallStatus(StrEnum):
    """Certainty of a single REST service attempt, independent of state."""

    NOT_SENT = "not_sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ServiceCallOutcome:
    """Transport outcome; acceptance does not confirm entity state."""

    status: ServiceCallStatus
    reason: str | None


@dataclass(frozen=True)
class HAStatesSnapshot:
    """Fresh REST observation, retaining the last successful copy on failure."""

    available: bool
    observed_at: datetime | None
    reason: str | None
    states: tuple[dict, ...] = ()
    expires_at: datetime | None = None
    attempted_at: datetime | None = None


@dataclass(frozen=True)
class HAAreaAssignments:
    """Reduced registry observation with entity and device areas reconciled."""

    available: bool
    observed_at: datetime | None
    reason: str | None
    area_names: dict[str, str] = field(default_factory=dict)
    entity_area_ids: dict[str, str] = field(default_factory=dict)
    expires_at: datetime | None = None
    attempted_at: datetime | None = None


def get_home_capabilities(entity: dict) -> tuple[str, ...]:
    """Return the dashboard's supported actions from explicit HA evidence.

    Args:
        entity: HA state object, including the reduced capability attributes.

    Returns:
        Operations evidenced by light, switch, climate or blind capabilities.
    """
    return action_capabilities(entity)


def _optional_registry_id(value: object) -> str | None:
    """Validate optional registry references before resolving any fallback."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid optional registry ID")
    return value


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all timeouts in one place for easy tuning
# ---------------------------------------------------------------------------
REST_TIMEOUT_DEFAULT = 5.0      # Standard REST call timeout (seconds)
REST_TIMEOUT_BULK = 10.0        # Bulk endpoints (/api/states, /api/services)
REST_TIMEOUT_HEALTH = 5.0       # Health-check / test_connection
STATE_OBSERVATION_TTL_S = 6.0
REGISTRY_OBSERVATION_TTL_S = 60.0
WS_AUTH_TIMEOUT = 5.0           # WebSocket auth handshake timeout
WS_CALL_TIMEOUT = 2.0           # WebSocket service-call timeout
STATE_SYNC_ATTEMPT_TIMEOUT_S = 15.0
STATE_SYNC_BACKOFF_INITIAL_S = 5.0
STATE_SYNC_BACKOFF_MAX_S = 30.0
TRANSPORT_OBSERVATION_TTL_S = 60.0
STATE_STREAM_MAX_MESSAGE_BYTES = MAX_FRAME_BYTES
STATE_REST_ADMISSION_CAPACITY = MAX_EFFECTS * 4 + 1
STATE_REST_HTTP_CAPACITY = MAX_EFFECTS
STATE_RAW_BUFFER_CAPACITY = MAX_FRAME_BYTES * 8
AUTOMATION_TIMEOUT = 10.0       # Automation CRUD operations
RELOAD_TIMEOUT = 5.0            # Automation reload

# User-facing fallback messages (Spanish, per project convention)
FALLBACK_MSG_UNAVAILABLE = "Home Assistant no está disponible en este momento. Intenta de nuevo en unos segundos."
FALLBACK_MSG_AUTH = "Error de autenticación con Home Assistant. Verifica el token de acceso."
FALLBACK_MSG_TIMEOUT = "Home Assistant tardó demasiado en responder."


# ---------------------------------------------------------------------------
# Health status DTO
# ---------------------------------------------------------------------------
class HAConnectionState(StrEnum):
    """Possible connection states for the HA client."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HAChannelObservation:
    """Transport evidence only; it does not renew an entity observation."""
    state: HAConnectionState = HAConnectionState.UNKNOWN
    source: str = ''
    phase: str = 'unobserved'
    observed_at: float | None = None
    expires_at: float | None = None
    reason: str | None = None
    attempted_at: float | None = None
    required: bool = False
    generation: int = 0


@dataclass
class HAHealthStatus:
    """Structured health information for the HA integration.

    Attributes:
        state: Current connection state.
        last_success_ts: Epoch timestamp of the last successful call (0 if never).
        last_error_ts: Epoch timestamp of the last error (0 if never).
        error_count: Total errors since client creation.
        success_count: Total successful calls since client creation.
        avg_latency_ms: Exponential moving average of REST call latency.
        ws_connected: Whether the WebSocket transport is connected.
        last_error_message: Human-readable description of the most recent error.
    """
    state: HAConnectionState = HAConnectionState.UNKNOWN
    last_success_ts: float = 0.0
    last_error_ts: float = 0.0
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    ws_connected: bool = False
    last_error_message: str = ""
    rest: HAChannelObservation | None = None
    calls: HAChannelObservation | None = None
    events: HAChannelObservation | None = None
    registry: HAChannelObservation | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class HomeAssistantClient:
    """Cliente para comunicacion con Home Assistant via REST API y WebSocket.

    All public methods degrade gracefully: connection errors, timeouts, and
    auth failures return safe fallback values (empty list, None, False) and
    log the incident rather than raising exceptions.
    """

    def __init__(self, url: str, token: str, timeout: float = REST_TIMEOUT_DEFAULT,
                 state_confirmation_timeout_s: float = 0.3, operation_confirmation_timeouts: dict[str, float] | None = None,
                 *, session: aiohttp.ClientSession | None = None):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.state_confirmation_timeout_s = state_confirmation_timeout_s
        if operation_confirmation_timeouts is not None and type(operation_confirmation_timeouts) is not dict:
            raise ValueError("invalid_operation_confirmation_timeout")
        timeouts = dict(operation_confirmation_timeouts or {})
        supported = set().union(*DOMAIN_OPERATIONS.values()) - {"turn_on", "turn_off", "brightness"}
        if timeouts.keys() - supported or any(not finite_number(value) or not 0 < value <= 120 for value in timeouts.values()):
            raise ValueError("invalid_operation_confirmation_timeout")
        self._operation_confirmation_timeouts = timeouts
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._session: aiohttp.ClientSession | None = session
        self._session_owned = session is None
        self._session_lock = asyncio.Lock()
        self._closed = False
        self._transport_generation = 0
        self._state_generation = 0
        self._events_attempt_generation = 0
        self._close_task = None
        self._state_stop_task = None
        self._state_stop_owner = None
        self._state_start_ready = None
        self._calls_connect_task = None
        self._registry_read_task = None
        self._transport_tasks: dict[asyncio.Task, str] = {}
        self._ws_open_tasks: dict[str, asyncio.Task] = {}
        self._ws_handshake_locks = {name: asyncio.Lock() for name in ('calls', 'events', 'registry')}
        self._owned_sockets: dict[int, tuple[object, str]] = {}
        self._socket_close_tasks: dict[int, asyncio.Task] = {}
        self._ws_registry = None
        self._channel_observations = {
            name: HAChannelObservation(source='ha_rest' if name == 'rest' else 'ha_ws_' + name,
                required=name == 'rest') for name in ('rest', 'calls', 'events', 'registry')
        }
        # Dual WebSocket connections (fix 2026-04-24): aiohttp forbids concurrent
        # receive() on the same WS. _ws_calls is used by call_service_ws (short
        # bounded receives), _ws_events is used by the state subscribe loop
        # (long-lived receive). HA accepts multiple WS with the same token and
        # scopes message IDs per connection — counters must NOT be shared.
        self._ws_calls: aiohttp.ClientWebSocketResponse | None = None
        self._ws_events: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connected = False  # reflects _ws_calls readiness
        self._ws_msg_id_calls: int = 1
        self._ws_msg_id_events: int = 1
        self._ws_reconnect_attempts = 0
        self._ws_max_reconnect_attempts = 3
        # Serializa el (re)connect del canal [calls]. Sin esto, dos comandos
        # concurrentes que encuentran el socket muerto hacen dos handshakes,
        # los dos asignan self._ws_calls y el perdedor se filtra sin cerrarse
        # (incidente 2026-07-25).
        self._ws_connect_lock = asyncio.Lock()
        # One service call owns the calls socket through ACK or REST fallback.
        # This also preserves on/off ordering after an uncertain WS timeout.
        self._ws_call_lock = asyncio.Lock()

        # Health tracking
        self._error_count = 0
        self._success_count = 0
        self._last_success_ts: float = 0.0
        self._last_error_ts: float = 0.0
        self._last_error_message: str = ""
        self._avg_latency_ms: float = 0.0
        self._has_auth_error: bool = False

        # State prefetch cache (S6): entity_id -> state dict
        # Populated via REST snapshot + updated via WebSocket state_changed events.
        # Single-writer (_state_sync_loop) / multi-reader model: GIL + atomic dict
        # operations make read-during-write safe for a single key.
        self._state_frame = empty_frame()
        self._state_projection_work = OwnedBlockingWork(capacity=1, isolate_context=True)
        self._state_projection_close: asyncio.Task | None = None
        self._state_projection_lock = asyncio.Lock()
        self._state_projection_tasks: dict[str, set[asyncio.Task]] = {'events': set(), 'rest': set()}
        self._state_read_slots = asyncio.Semaphore(STATE_REST_HTTP_CAPACITY)
        self._state_raw_condition = asyncio.Condition()
        self._state_raw_bytes = 0
        self._entity_stream_generation = 0
        self._state_subscribe_task: asyncio.Task | None = None
        self._state_callbacks: list = []  # observers opcionales
        self._state_sync_running = False
        self._state_rebuild_requested = False
        self._state_last_full_refresh: float = 0.0
        self._state_full_refresh_interval_s: float = 300.0

        # Dashboard reads have independent provenance; the voice cache is not
        # evidence of a successful current REST observation.
        self._home_states = HAStatesSnapshot(False, None, "ha_states_unavailable")
        self._state_snapshot_lock = asyncio.Lock()
        self._area_assignments: HAAreaAssignments | None = None
        self._registry_lock = asyncio.Lock()
        self._registry_refresh_at = 0.0

    # ------------------------------------------------------------------
    # Backward-compat alias
    # ------------------------------------------------------------------
    @property
    def _ws_connection(self) -> aiohttp.ClientWebSocketResponse | None:
        """Backward-compat alias for the calls-channel WebSocket.

        Deprecated: new code should reference self._ws_calls (service calls)
        or self._ws_events (state subscribe) explicitly.
        """
        return self._ws_calls

    @_ws_connection.setter
    def _ws_connection(self, value: aiohttp.ClientWebSocketResponse | None) -> None:
        # Legacy setter kept so tests / callers that assigned directly keep
        # working. Writes go to the calls channel.
        self._ws_calls = value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _state_cache(self) -> dict[str, dict]:
        """Detached historical projection for legacy trusted diagnostics only."""
        result = {}
        for entity_id, observation in self._state_frame.entries.items():
            state = observation.state
            if state is not None and observation.exists is not False:
                result[entity_id] = state
        return result

    @_state_cache.setter
    def _state_cache(self, states: dict[str, dict]) -> None:
        if any(not task.done() for tasks in self._state_projection_tasks.values() for task in tasks):
            raise RuntimeError('ha_state_projection_busy')
        previous = self._state_frame
        self._state_frame = legacy_frame(states, generation=previous.generation,
                                         revision=previous.revision + 1)

    def _entity_now(self) -> float:
        return time.time()

    def get_entity_observation(self, entity_id: str) -> EntityObservation:
        """Read original per-entity evidence without I/O or freshness renewal."""
        value = entity_observation(self._state_frame, entity_id, now=self._entity_now())
        if self._closed:
            return observation_view(value, exists=None, reason='source_closed')
        if (value.source == 'ha-ws-entities' and
                (not self._state_sync_running or self._state_frame.generation != self._entity_stream_generation)):
            return observation_view(value, exists=None, reason=value.reason or 'source_changed')
        return value

    def _state_failure(self, entity_id: str, reason: str, attempted_at: float) -> EntityObservation:
        """Return metadata-only local rejection without copying historical payloads."""
        frame = self._state_frame
        previous = frame.entries.get(entity_id)
        if previous is None:
            source = 'ha-ws-entities' if frame.initialized else 'ha-rest-entity'
            observed, expiry = (frame.observed_at, frame.expires_at) if frame.initialized else (None, None)
        else:
            source = previous.source or 'ha-rest-entity'
            observed, expiry = previous.observed_at, previous.expires_at
        return EntityObservation(entity_id, exists=None, source=source, observed_at=observed,
            expires_at=expiry, attempted_at=attempted_at, reason=reason,
            generation=frame.generation, revision=frame.entry_revision(entity_id))

    async def _state_operation(self, scope: str, function, *args, **kwargs):
        """Retain one actor per scope, including I/O, CPU exit and publication."""
        purpose = scope
        current = self._state_projection_tasks[purpose]
        if asyncio.current_task() in current:
            return await function(*args, **kwargs)
        capacity = 1 if purpose == 'events' else STATE_REST_ADMISSION_CAPACITY
        if self._closed or sum(not task.done() for task in current) >= capacity:
            raise aiohttp.ClientConnectionError('ha_state_projection_unavailable')
        slot = {'waiting': True}
        async def finish(function=function, args=args, kwargs=kwargs, slot=slot):
            result = None
            try:
                result = await function(*args, **kwargs)
                if slot.get('waiting'):
                    slot['value'] = result
            finally:
                result = function = None
                args, kwargs = (), {}
        operation = finish()
        try:
            task = self._retain_transport(operation, purpose)
        except BaseException:
            operation.close()
            raise
        self._state_projection_tasks[purpose].add(task)
        def finished(done):
            self._state_projection_tasks[purpose].discard(done)
        task.add_done_callback(finished, context=Context())
        try:
            await asyncio.shield(task)
            return slot.pop('value', None)
        finally:
            slot.clear()
            function, args, kwargs = None, (), {}
            task = operation = None

    async def _state_native(self, kind, base, raw, options, callback_count=0):
        """Keep the serial lock owned until the actual CPU worker has exited."""
        task = result = None
        cancelled = False
        try:
            task = asyncio.create_task(self._state_projection_work.run(
                _state_projection_cpu, kind, base, raw, options, callback_count), context=Context())
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                except RuntimeError:
                    break
            try:
                result = task.result()
            except RuntimeError:
                raise aiohttp.ClientConnectionError('ha_state_projection_unavailable') from None
            if cancelled:
                raise asyncio.CancelledError
            return result
        finally:
            task = result = kind = base = raw = options = None

    def _state_scope_current(self, purpose, options):
        if self._closed or options['transport_generation'] != self._transport_generation:
            return False
        if not options.get('withdraw_only') and monotonic() >= options['deadline']:
            return False
        if purpose == 'events':
            if options['state_generation'] != self._state_generation:
                return False
            if options.get('require_running') and not self._state_sync_running:
                return False
            if ('stream_generation' in options
                    and options['stream_generation'] != self._entity_stream_generation):
                return False
        return True

    async def _project_state(self, kind, raw, options, *, purpose):
        """Serialize decode/reduction and publish the exact computed frame on-loop."""
        result = replacement = base = notices = callbacks = None
        try:
            async with self._state_projection_lock:
                if not self._state_scope_current(purpose, options):
                    raise aiohttp.ClientConnectionError('ha_state_projection_retired')
                base = self._state_frame
                if kind == 'bulk' and base.initialized:
                    return None
                callbacks = tuple(self._state_callbacks) if kind == 'delta' else ()
                options = {**options, 'now': self._entity_now()}
                replacement, result, notices = await self._state_native(
                    kind, base, raw, options, len(callbacks))
                if not self._state_scope_current(purpose, options):
                    raise aiohttp.ClientConnectionError('ha_state_projection_retired')
                if self._state_frame is not base:
                    raise aiohttp.ClientConnectionError('ha_state_projection_superseded')
                if replacement is not None:
                    self._state_frame = replacement
                if kind in ('initial', 'delta', 'rest') and self._entity_now() >= options['observed_at'] + STATE_OBSERVATION_TTL_S:
                    raise aiohttp.ClientConnectionError('ha_state_projection_expired')
                for identity, payloads in notices:
                    for callback, payload in zip(callbacks, payloads):
                        value = self._state_frame.entries.get(identity)
                        admitted = (self._state_scope_current(purpose, options)
                            and self._state_frame is replacement and value is not None
                            and value.generation == self._state_frame.generation
                            and value.exists is True and value.current(self._entity_now()))
                        try:
                            callback(identity, payload if admitted else None)
                        except Exception as error:
                            logger.debug('HA state callback failed: %s', type(error).__name__)
                return result
        finally:
            raw = options = result = replacement = base = notices = callbacks = None
            identity = payloads = payload = value = None

    async def _decode_state_message(self, raw, *, deadline, state_generation=None):
        options = dict(transport_generation=self._transport_generation, deadline=deadline,
            state_generation=self._state_generation if state_generation is None else state_generation)
        return await self._state_operation('events', self._project_state,
            'decode', raw, options, purpose='events')

    async def _drain_state_projections(self, purpose=None):
        tasks = tuple(task for scope, jobs in self._state_projection_tasks.items() for task in jobs
            if (purpose is None or scope == purpose) and not task.done() and task is not asyncio.current_task())
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    async def _release_state_bytes(self, size):
        if size:
            async with self._state_raw_condition:
                self._state_raw_bytes -= size
                self._state_raw_condition.notify_all()

    async def _read_state_body(self, response):
        """Reserve one entire bounded body so partial readers cannot deadlock."""
        parts, size, charged = [], 0, 0
        credit = MAX_FRAME_BYTES + 1
        try:
            async with self._state_raw_condition:
                await self._state_raw_condition.wait_for(lambda:
                    self._closed or self._state_raw_bytes + credit <= STATE_RAW_BUFFER_CAPACITY)
                if self._closed:
                    raise aiohttp.ClientConnectionError('ha_client_closed')
                self._state_raw_bytes += credit
                charged = credit
            while size <= MAX_FRAME_BYTES:
                part = await response.content.read(min(65536, credit - size))
                if not part:
                    await self._release_state_bytes(credit - size)
                    charged = size
                    return tuple(parts), charged
                parts.append(part)
                size += len(part)
            raise ValueError('ha_state_body_too_large')
        except BaseException:
            await self._release_state_bytes(charged)
            raise
        finally:
            parts = part = None


    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure an aiohttp ClientSession exists, creating one lazily if needed.

        Returns:
            Active aiohttp.ClientSession instance.
        """
        async with self._session_lock:
            if self._closed:
                raise aiohttp.ClientConnectionError('ha_client_closed')
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
                self._session_owned = True
        return self._session

    def _transport_current(self, generation: int, purpose: str, state_generation: int,
                           attempt_generation: int) -> bool:
        return (not self._closed and generation == self._transport_generation
            and (purpose != 'events' or state_generation == self._state_generation
                 and attempt_generation == self._events_attempt_generation))

    def _observe_channel(self, channel: str, phase: str, state: HAConnectionState,
                         reason: str | None = None, *, evidence: bool = False,
                         attempted_at: float | None = None, required: bool | None = None,
                         observed_at: float | None = None) -> None:
        previous = self._channel_observations[channel]
        observed = (time.time() if observed_at is None else observed_at) if evidence else previous.observed_at
        self._channel_observations[channel] = replace(previous, phase=phase, state=state,
            reason=reason, observed_at=observed,
            expires_at=observed + TRANSPORT_OBSERVATION_TTL_S if evidence else previous.expires_at,
            attempted_at=attempted_at if attempted_at is not None else previous.attempted_at,
            required=previous.required if required is None else required,
            generation=self._state_generation if channel == 'events' else self._transport_generation)

    def _channel_status(self, channel: str) -> HAChannelObservation:
        value = self._channel_observations[channel]
        if (value.expires_at is not None and time.time() >= value.expires_at
                and value.phase not in ('stopped', 'closed')):
            return replace(value, state=HAConnectionState.UNKNOWN, phase='expired',
                           reason='ha_transport_observation_expired')
        return value

    def _retain_transport(self, operation, purpose: str) -> asyncio.Task:
        task = asyncio.create_task(operation, context=Context())
        self._transport_tasks[task] = purpose
        def finished(done):
            self._transport_tasks.pop(done, None)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished, context=Context())
        return task

    async def _close_socket(self, ws, purpose: str) -> None:
        if ws is None:
            return
        key = id(ws)
        if ws.closed is True:
            self._owned_sockets.pop(key, None)
            return
        task = self._socket_close_tasks.get(key)
        if task is None:
            async def finish():
                for attempt in range(3):
                    try:
                        await ws.close()
                        if ws.closed is not True:
                            raise aiohttp.ClientConnectionError('ha_socket_close_unconfirmed')
                        self._owned_sockets.pop(key, None)
                        return
                    except Exception:
                        self._observe_channel(purpose, 'retiring', HAConnectionState.UNKNOWN,
                                              'ha_socket_close_pending')
                        if attempt < 2:
                            await asyncio.sleep(.05 * (2 ** attempt))
                logger.warning('HA socket retirement unresolved [%s]; review required', purpose)
                raise aiohttp.ClientConnectionError('ha_socket_retirement_unresolved')
            task = self._socket_close_tasks[key] = self._retain_transport(finish(), purpose)
            def finished(done):
                if (self._socket_close_tasks.get(key) is done and not done.cancelled()
                        and done.exception() is None):
                    self._socket_close_tasks.pop(key, None)
            task.add_done_callback(finished, context=Context())
        await asyncio.shield(task)

    async def _retire_socket(self, purpose: str, ws=None) -> None:
        attr = '_ws_' + purpose
        if ws is None:
            ws = getattr(self, attr, None)
        if getattr(self, attr, None) is ws:
            setattr(self, attr, None)
        if purpose == 'calls':
            self._ws_connected = False
        if ws is not None:
            self._observe_channel(purpose, 'retiring', HAConnectionState.UNKNOWN,
                                  'ha_socket_retiring')
            await self._close_socket(ws, purpose)
        else:
            for candidate, kind in tuple(self._owned_sockets.values()):
                if kind == purpose:
                    await self._close_socket(candidate, purpose)

    async def _drain_transports(self, purpose: str | None = None) -> None:
        while True:
            tasks = tuple(task for task, kind in self._transport_tasks.items()
                          if (purpose is None or purpose == kind) and not task.done()
                          and task is not asyncio.current_task())
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    def _record_success(self, latency_ms: float, *, channel: str = 'rest', observed_at: float | None = None) -> None:
        """Record a successful HA call for health tracking."""
        if self._closed:
            return
        self._success_count += 1
        self._last_success_ts = time.time() if observed_at is None else observed_at
        self._has_auth_error = False
        self._observe_channel(channel, 'ready', HAConnectionState.CONNECTED, evidence=True,
            observed_at=observed_at)
        # Exponential moving average (alpha=0.3)
        if self._avg_latency_ms == 0.0:
            self._avg_latency_ms = latency_ms
        else:
            self._avg_latency_ms = 0.7 * self._avg_latency_ms + 0.3 * latency_ms

    def _record_error(self, error: Exception, context: str, *, channel: str = 'rest') -> None:
        """Record a failed HA call for health tracking."""
        if self._closed:
            return
        self._error_count += 1
        self._last_error_ts = time.time()
        self._last_error_message = f"{context}: {error}"
        if isinstance(error, aiohttp.ClientResponseError) and error.status in (401, 403):
            self._has_auth_error = True
        denied = isinstance(error, aiohttp.ClientResponseError) and error.status in (401, 403)
        self._observe_channel(channel, 'denied' if denied else 'failed',
            HAConnectionState.AUTH_ERROR if denied else HAConnectionState.DISCONNECTED,
            'ha_auth_denied' if denied else 'ha_transport_failure', evidence=True)

    # ==================== REST API ====================

    async def get_state_snapshot(self, *, _deadline: float | None = None) -> HAStatesSnapshot:
        """Read one bounded retained REST snapshot, preserving original dates."""
        attempted = datetime.fromtimestamp(self._entity_now(), UTC)
        deadline = monotonic() + REST_TIMEOUT_BULK if _deadline is None else _deadline
        if asyncio.current_task() in self._state_projection_tasks['rest']:
            return await self._get_state_snapshot_owned(deadline, attempted)
        try:
            return await asyncio.wait_for(self._state_operation('rest',
                self._get_state_snapshot_owned, deadline, attempted), timeout=max(0., deadline - monotonic()))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return replace(self._home_states, available=False, reason='ha_states_unavailable',
                attempted_at=attempted)


    async def _get_state_snapshot_owned(self, deadline, attempted_at) -> HAStatesSnapshot:
        raw = states = None
        charged = 0
        acquired = False
        generation = self._transport_generation
        started = time.perf_counter()
        reason = 'ha_states_unavailable'
        try:
            async with asyncio.timeout_at(deadline):
                await self._state_snapshot_lock.acquire()
                acquired = True
                async with self._state_read_slots:
                    if self._closed or monotonic() >= deadline:
                        raise aiohttp.ClientConnectionError('ha_state_read_retired')
                    session = await self._ensure_session()
                    async with session.get(f'{self.url}/api/states',
                            timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_BULK)) as response:
                        if response.status in (401, 403):
                            reason = 'ha_states_denied'
                        response.raise_for_status()
                        raw, charged = await self._read_state_body(response)
                        observed_at = datetime.fromtimestamp(self._entity_now(), UTC)
            options = dict(transport_generation=generation, deadline=deadline)
            states = await self._project_state('snapshot', raw, options, purpose='rest')
            if self._closed or generation != self._transport_generation:
                raise aiohttp.ClientConnectionError('ha_state_observation_retired')
            self._home_states = HAStatesSnapshot(True, observed_at, None, states,
                observed_at + timedelta(seconds=STATE_OBSERVATION_TTL_S), attempted_at)
            self._record_success((time.perf_counter() - started) * 1000,
                observed_at=observed_at.timestamp())
        except (ValueError, TypeError, KeyError):
            reason = 'ha_states_invalid'
            self._record_error(ValueError('invalid_ha_states_body'), 'get_state_snapshot')
            self._home_states = replace(self._home_states, available=False, reason=reason, attempted_at=attempted_at)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            self._record_error(error, 'get_state_snapshot')
            self._home_states = replace(self._home_states, available=False, reason=reason, attempted_at=attempted_at)
        finally:
            raw = states = None
            if acquired:
                self._state_snapshot_lock.release()
            await self._release_state_bytes(charged)
        return self._home_states



    async def get_area_assignments(self) -> HAAreaAssignments:
        """Read registries on an owned socket, caching one refresh for 60 seconds.

        Returns:
            Reduced registry data, retaining previous content and time on failure.
        """
        async with self._registry_lock:
            generation = self._transport_generation
            if self._closed:
                previous = self._area_assignments or HAAreaAssignments(False, None, None)
                return replace(previous, available=False, reason='ha_client_closed')
            if self._area_assignments is not None and monotonic() < self._registry_refresh_at:
                return self._area_assignments
            previous = self._area_assignments or HAAreaAssignments(False, None, None)
            attempted_at = datetime.now(UTC)
            try:
                areas, entities, devices = await self._read_home_registries()
                names = {area["area_id"]: area["name"] for area in areas}
                if any(not isinstance(device.get("id"), str) for device in devices):
                    raise ValueError("Invalid registry device ID")
                device_areas = {
                    device["id"]: _optional_registry_id(device.get("area_id"))
                    for device in devices
                }
                assignments = {}
                for entity in entities:
                    if not isinstance(entity.get("entity_id"), str) or "." not in entity["entity_id"]:
                        raise ValueError("Invalid registry entity ID")
                    area_id = _optional_registry_id(entity.get("area_id"))
                    device_id = _optional_registry_id(entity.get("device_id"))
                    if area_id is None:
                        area_id = device_areas.get(device_id)
                    if area_id is not None:
                        assignments[entity["entity_id"]] = area_id
                if not all(isinstance(key, str) and isinstance(value, str) for key, value in names.items()):
                    raise ValueError("Invalid area registry")
                if self._closed or generation != self._transport_generation:
                    raise aiohttp.ClientConnectionError('ha_registry_observation_retired')
                observed_at = datetime.now(UTC)
                self._area_assignments = HAAreaAssignments(
                    True, observed_at, None, names, assignments,
                    observed_at + timedelta(seconds=REGISTRY_OBSERVATION_TTL_S), attempted_at,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as error:
                self._record_error(error, "get_area_assignments", channel='registry')
                self._area_assignments = replace(
                    previous, available=False, reason="ha_registries_unavailable", attempted_at=attempted_at,
                )
                logger.warning("HA dashboard registries unavailable")
            self._registry_refresh_at = monotonic() + REGISTRY_OBSERVATION_TTL_S
            return self._area_assignments

    async def _read_home_registries(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Retain the temporary registry reader before its first transport await."""
        if self._closed:
            raise aiohttp.ClientConnectionError('ha_client_closed')
        task = self._registry_read_task
        if task is None or task.done():
            task = self._registry_read_task = self._retain_transport(
                self._read_home_registries_owned(self._transport_generation), 'registry')
            def finished(done):
                if self._registry_read_task is done:
                    self._registry_read_task = None
            task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def _read_home_registries_owned(self, generation):
        ws = None
        try:
            async with asyncio.timeout(REST_TIMEOUT_BULK):
                ws = await self._open_ws_authenticated('registry')
                if ws is None:
                    raise aiohttp.ClientConnectionError('ha_registry_unavailable')
                results = []
                for message_id, registry in enumerate(('area', 'entity', 'device'), start=1):
                    if self._closed or generation != self._transport_generation:
                        raise aiohttp.ClientConnectionError('ha_registry_retired')
                    await ws.send_json({'id': message_id, 'type': f'config/{registry}_registry/list'})
                    message = await ws.receive_json()
                    if (not isinstance(message, dict) or message.get('id') != message_id
                            or message.get('type') != 'result' or message.get('success') is not True):
                        raise ValueError('ha_registry_result_invalid')
                    result = message.get('result')
                    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
                        raise ValueError('ha_registry_entries_invalid')
                    results.append(result)
                if self._closed or generation != self._transport_generation:
                    raise aiohttp.ClientConnectionError('ha_registry_retired')
                self._observe_channel('registry', 'ready', HAConnectionState.CONNECTED, evidence=True)
                return results[0], results[1], results[2]
        finally:
            opening = self._ws_open_tasks.get('registry')
            if opening is not None:
                await asyncio.gather(asyncio.shield(opening), return_exceptions=True)
            await self._retire_socket('registry', ws)


    async def get_all_entities(self) -> list[dict]:
        """Obtener todas las entidades de Home Assistant."""
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/states",
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_BULK)
            ) as response:
                if response.status in (401, 403):
                    self._has_auth_error = True
                    logger.error(f"HA auth error {response.status} getting entities")
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        ),
                        "get_all_entities",
                    )
                    return []
                response.raise_for_status()
                result = await response.json()
                elapsed = (time.perf_counter() - t_start) * 1000
                self._record_success(elapsed)
                return result
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA timeout getting entities after {elapsed:.0f}ms (limit {REST_TIMEOUT_BULK}s)")
            self._record_error(asyncio.TimeoutError(), "get_all_entities")
            return []
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (get_all_entities): {e}")
            self._record_error(e, "get_all_entities")
            return []
        except Exception as e:
            logger.error(f"Error obteniendo entidades: {e}")
            self._record_error(e, "get_all_entities")
            return []

    async def get_all_services(self) -> list[dict]:
        """Obtener todos los servicios disponibles."""
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/services",
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_BULK)
            ) as response:
                if response.status in (401, 403):
                    self._has_auth_error = True
                    logger.error(f"HA auth error {response.status} getting services")
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        ),
                        "get_all_services",
                    )
                    return []
                response.raise_for_status()
                result = await response.json()
                elapsed = (time.perf_counter() - t_start) * 1000
                self._record_success(elapsed)
                return result
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA timeout getting services after {elapsed:.0f}ms (limit {REST_TIMEOUT_BULK}s)")
            self._record_error(asyncio.TimeoutError(), "get_all_services")
            return []
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (get_all_services): {e}")
            self._record_error(e, "get_all_services")
            return []
        except Exception as e:
            logger.error(f"Error obteniendo servicios: {e}")
            self._record_error(e, "get_all_services")
            return []

    async def get_entity_state(self, entity_id: str) -> dict | None:
        value = self.get_entity_observation(entity_id)
        if value.current(self._entity_now()):
            return value.state if value.exists is True else None
        return await self._get_entity_state_rest(entity_id)


    async def _get_entity_state_rest(self, entity_id: str) -> dict | None:
        value = await self.read_entity_observation(entity_id)
        return value.state if value.current(self._entity_now()) and value.exists is True else None

    async def read_entity_observation(self, entity_id: str, timeout: float = REST_TIMEOUT_DEFAULT) -> EntityObservation:
        """Force one bounded read; retain its actual I/O/projection after observer loss."""
        if not finite_number(timeout) or timeout <= 0:
            raise ValueError('invalid_entity_read_timeout')
        deadline = monotonic() + timeout
        attempted = self._entity_now()
        frame = self._state_frame
        options = dict(entity_id=entity_id, attempted_at=attempted,
            expected_generation=frame.generation, expected_revision=frame.entry_revision(entity_id),
            transport_generation=self._transport_generation, deadline=deadline)
        frame = None
        try:
            return await asyncio.wait_for(self._state_operation('rest',
                self._read_entity_observation_owned, entity_id, timeout, options),
                timeout=max(0., deadline - monotonic()))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            reason = 'source_closed' if self._closed else 'source_unavailable'
            return self._state_failure(entity_id, reason, attempted)
        finally:
            options = None

    async def _read_entity_observation_owned(self, entity_id, timeout, options):
        raw = result = None
        charged = 0
        started = time.perf_counter()
        reason = 'source_unavailable'
        try:
            try:
                async with asyncio.timeout_at(options['deadline']):
                    async with self._state_read_slots:
                        if self._closed or monotonic() >= options['deadline']:
                            raise aiohttp.ClientConnectionError('ha_state_read_retired')
                        session = await self._ensure_session()
                        async with session.get(f'{self.url}/api/states/{entity_id}',
                                timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                            if response.status in (401, 403):
                                reason = 'source_denied'
                            missing = response.status == 404
                            if not missing:
                                response.raise_for_status()
                                raw, charged = await self._read_state_body(response)
                            observed = self._entity_now()
                options = {**options, 'missing': missing, 'observed_at': observed}
                result = await self._project_state('rest', raw, options, purpose='rest')
                self._record_success((time.perf_counter() - started) * 1000, observed_at=observed)
                return result
            except (ValueError, TypeError, KeyError):
                reason = 'source_invalid'
                self._record_error(ValueError('invalid_entity_state'), 'read_entity_observation')
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                self._record_error(error, 'read_entity_observation')
            if self._closed or options['transport_generation'] != self._transport_generation:
                return self._state_failure(entity_id, 'source_closed', options['attempted_at'])
            options = {**options, 'reason': reason, 'withdraw_only': True}
            return await self._project_state('failure', None, options, purpose='rest')
        finally:
            raw = result = options = None
            response = session = None
            await self._release_state_bytes(charged)



    async def get_entity_state_fresh(self, entity_id: str, timeout: float = 0.15) -> dict | None:
        value = await self.read_entity_observation(entity_id, timeout=timeout)
        return value.state if value.current(self._entity_now()) and value.exists is True else None


    async def confirm_entity_state(
        self, entity_id: str, expected_state: str, timeout: float | None = None,
        *, brightness_pct: int | None = None,
    ) -> bool:
        """Confirm a requested state using fresh reads, never an old cache entry.

        Args:
            entity_id: Target Home Assistant entity.
            expected_state: Requested binary state (on or off).
            timeout: Maximum confirmation wait after the service response.
            brightness_pct: Requested light percentage; confirm its fresh
                0-255 attribute too, allowing one step for device rounding.

        Returns:
            Whether HA reported the requested state before the deadline.
            This is an integration state confirmation, not physical sensing.
        """
        duration = self.state_confirmation_timeout_s if timeout is None else timeout
        deadline = asyncio.get_running_loop().time() + duration
        while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            state = await self.get_entity_state_fresh(entity_id, timeout=remaining)
            if state is None:
                return False
            if state.get("state") == expected_state:
                if brightness_pct is None or brightness_pct == 0:
                    return True
                actual = (state.get("attributes") or {}).get("brightness")
                if isinstance(actual, (int, float)) and abs(actual - brightness_pct * 255 / 100) <= 1:
                    return True
            if state.get("state") in ("unknown", "unavailable"):
                return False
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.sleep(min(0.025, remaining))
        return False

    async def get_temperature_unit_fresh(self, timeout: float = 0.15) -> str | None:
        """Read the installation unit used by climate REST state and service data."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/config", timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=False,
            ) as response:
                if response.status in (401, 403):
                    self._has_auth_error = True
                if response.status != 200:
                    return None
                config = await response.json()
                unit = config.get("unit_system", {}).get("temperature")
                return unit if unit in {"°C", "°F"} else None
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, AttributeError):
            return None

    async def confirm_home_action(
        self, request: ActionIntent, *, initial_state: str | None = None, timeout: float | None = None,
    ) -> bool:
        """Poll operation-specific state under one deadline, without repeating a send.

        Temperature confirms the setpoint and unchanged HVAC mode, never room
        temperature. Stop-cover requires two nonmoving observations with stable
        position when reported. All confirmations describe HA, not physical sensing.
        """
        duration = getattr(self, "_operation_confirmation_timeouts", {}).get(request.operation, self.state_confirmation_timeout_s) if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration
        previous_stop = None
        while (remaining := deadline - loop.time()) > 0:
            state = await self.get_entity_state_fresh(request.entity_id, timeout=remaining)
            if state is None or state.get("state") in {"unknown", "unavailable"}:
                return False
            if request.temperature is not None:
                remaining = deadline - loop.time()
                if remaining <= 0 or await self.get_temperature_unit_fresh(timeout=remaining) != "°C":
                    return False
                if state.get("attributes", {}).get("temperature_unit") not in (None, "°C"):
                    return False
            if matches_observation(request, state, initial_state=initial_state):
                if request.operation != "stop_cover":
                    return loop.time() <= deadline
                observed = (state["state"], state.get("attributes", {}).get("current_position"))
                if previous_stop == observed:
                    return loop.time() <= deadline
                previous_stop = observed
            else:
                previous_stop = None
            remaining = deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(min(0.025, remaining))
        return False

    def get_entity_state_cached(self, entity_id: str) -> dict | None:
        value = self.get_entity_observation(entity_id)
        return value.state if value.current(self._entity_now()) and value.exists is True else None


    def is_entity_available(self, entity_id: str) -> bool | None:
        """¿La entidad puede actuar, según el último estado conocido?

        HA acepta un service_call contra una entidad `unavailable` y lo filtra
        en silencio (helpers/service.py), devolviendo success=true. Este chequeo
        existe para no confundir "HA aceptó el mensaje" con "el dispositivo hizo
        algo".

        Returns:
            True si está disponible, False si está `unavailable`,
            None si no hay entry en cache. Los controles deben pasar por el
            servicio de acciones, que consulta estado fresco y política antes
            del envío; el cache no autoriza una acción.
        """
        entry = self.get_entity_state_cached(entity_id)
        if entry is None:
            return None
        # ⚠️ SOLO `unavailable`. `unknown` NO va acá, por más que suene igual
        # de roto (el error es intuitivo y ya se cometió una vez):
        #
        #   - `Entity._stringify_state()` devuelve `unavailable` si y solo si
        #     `available == False`. `unknown` solo es alcanzable con
        #     `available == True`.
        #   - El filtro silencioso que motiva este precheck es
        #     `[e for e in entity_candidates if e.available]`
        #     (helpers/service.py). O sea: `unavailable` ⟹ HA descarta la
        #     llamada en silencio; `unknown` ⟹ HA la ejecuta normalmente.
        #
        # Bloquear `unknown` sería una regresión funcional del camino de voz.
        # Víctima concreta: `Scene.state` es el timestamp de la última
        # activación, o `None` → `unknown`. Una escena nunca activada queda
        # `unknown` para siempre, y el bloqueo es AUTOBLOQUEANTE: la única vía
        # por la que saldría de `unknown` es exactamente la que se estaría
        # bloqueando. Hay ~25 frases de escenas indexadas en Chroma.
        #
        # Este criterio está atado por test al de `smoke_check.entity_problem`
        # (tests/unit/monitoring/test_smoke_check.py): si cambia uno solo, el
        # diagnóstico y el dispatcher vuelven a contradecirse.
        # NO confundir con `select_syncable` del sync a Chroma, que sí excluye
        # `unknown` a propósito — ver el comentario allá.
        return entry.get("state") != "unavailable"

    def has_domain(self, domain: str) -> bool | None:
        """Current presence, or absence only from a current complete baseline."""
        now = self._entity_now()
        unknown = False
        for entity_id in self._state_frame.entries:
            if entity_id.startswith(domain + '.'):
                value = self.get_entity_observation(entity_id)
                if value.current(now):
                    if value.exists is True:
                        return True
                else:
                    unknown = True
        if unknown:
            return None
        if (not self._closed and self._state_sync_running
                and self._state_frame.generation == self._entity_stream_generation
                and self._state_frame.baseline_current(now)):
            return False
        return None


    async def _fetch_all_states_rest(self) -> list[dict]:
        """Wrapper sobre get_all_entities para claridad en el sync loop."""
        snapshot = await self.get_state_snapshot()
        if not snapshot.available:
            raise RuntimeError("ha_state_snapshot_unavailable")
        return list(snapshot.states)

    async def call_service_once(
        self, domain: str, service: str, entity_id: str, data: dict | None = None,
        *, guard: Callable[[], bool] | None = None, on_dispatch: Callable[[], None] | None = None,
    ) -> ServiceCallOutcome:
        """Send one REST POST without redirect, fallback, or automatic retry.

        Args:
            domain: Home Assistant domain.
            service: Service name in that domain.
            entity_id: Exact target; data cannot replace it.
            data: Additional service fields, excluding target selectors.
            guard: Original authority, rechecked after waiting for the session.
            on_dispatch: Synchronous notification immediately before POST begins.

        Returns:
            Local non-dispatch, explicit rejection, acceptance or uncertainty.
        """
        if (
            not isinstance(domain, str) or re.fullmatch(r"[a-z_]+", domain) is None
            or not isinstance(service, str) or re.fullmatch(r"[a-z_]+", service) is None
            or not isinstance(entity_id, str)
            or re.fullmatch(re.escape(domain) + r"\.[a-z0-9_]+", entity_id) is None
            or (data is not None and (
                not isinstance(data, dict) or {"entity_id", "device_id", "area_id", "target"}.intersection(data)
            ))
        ):
            return ServiceCallOutcome(ServiceCallStatus.REJECTED, "invalid_service_target")
        payload = {**(data or {}), "entity_id": entity_id}
        started, attempted = time.perf_counter(), False
        try:
            session = await self._ensure_session()
            if guard is not None:
                try:
                    allowed = guard()
                except Exception:
                    return ServiceCallOutcome(ServiceCallStatus.NOT_SENT, "service_authority_unavailable")
                if allowed is not True:
                    return ServiceCallOutcome(ServiceCallStatus.NOT_SENT, "service_authority_denied")
            if on_dispatch is not None:
                try:
                    on_dispatch()
                except Exception:
                    return ServiceCallOutcome(ServiceCallStatus.NOT_SENT, "service_dispatch_unavailable")
            attempted = True
            async with session.post(
                f"{self.url}/api/services/{domain}/{service}", json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout), allow_redirects=False,
            ) as response:
                if response.status == 200:
                    self._record_success((time.perf_counter() - started) * 1000)
                    return ServiceCallOutcome(ServiceCallStatus.ACCEPTED, None)
                if response.status in (401, 403):
                    self._has_auth_error = True
                status = ServiceCallStatus.REJECTED if 400 <= response.status < 500 else ServiceCallStatus.UNCERTAIN
                return ServiceCallOutcome(status, f"ha_http_{response.status}")
        except (asyncio.TimeoutError, aiohttp.ClientError, OSError):
            return ServiceCallOutcome(ServiceCallStatus.UNCERTAIN if attempted else ServiceCallStatus.NOT_SENT,
                                      "ha_transport_uncertain" if attempted else "ha_session_unavailable")

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: dict | None = None
    ) -> bool:
        """Ejecutar un servicio de Home Assistant.

        Args:
            domain: Dominio (light, climate, cover, etc.)
            service: Servicio (turn_on, turn_off, set_temperature, etc.)
            entity_id: ID de la entidad
            data: Datos adicionales (temperature, brightness, etc.)

        Returns:
            True si la llamada fue exitosa.
        """
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.url}/api/services/{domain}/{service}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_DEFAULT)
            ) as response:
                elapsed = (time.perf_counter() - t_start) * 1000
                if response.status in (401, 403):
                    self._has_auth_error = True
                    logger.error(f"HA auth error {response.status}: {domain}.{service} on {entity_id}")
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        ),
                        f"call_service({domain}.{service})",
                    )
                    return False
                success = response.status == 200
                if success:
                    self._record_success(elapsed)
                    logger.info(f"Ejecutado: {domain}.{service} en {entity_id} ({elapsed:.0f}ms)")
                else:
                    logger.warning(f"Error {response.status}: {domain}.{service} ({elapsed:.0f}ms)")
                return success
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(
                f"HA timeout calling {domain}.{service} on {entity_id} "
                f"after {elapsed:.0f}ms (limit {REST_TIMEOUT_DEFAULT}s)"
            )
            self._record_error(asyncio.TimeoutError(), f"call_service({domain}.{service})")
            return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (call_service {domain}.{service}): {e}")
            self._record_error(e, f"call_service({domain}.{service})")
            return False
        except Exception as e:
            logger.error(f"Error llamando servicio: {e}")
            self._record_error(e, f"call_service({domain}.{service})")
            return False

    async def call_service_with_response(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: dict | None = None,
        timeout: float = REST_TIMEOUT_DEFAULT,
    ) -> dict | None:
        """Call a service that returns a body (weather.get_forecasts, etc).

        `call_service` only reports success as a bool. Services declaring
        SupportsResponse need `?return_response=true` and their body read.

        Args:
            domain: Dominio del servicio (weather, climate, ...).
            service: Servicio a invocar (get_forecasts, ...).
            entity_id: Entidad objetivo.
            data: Datos extra del payload.
            timeout: Techo por request, en segundos. Explícito para que el
                caller lo elija según su presupuesto de latencia en vez de
                heredar el timeout de sesión.

        Returns:
            El body decodificado, o None ante cualquier fallo (siempre
            logueado). Un 401/403 además prende `_has_auth_error`.
        """
        payload = {"entity_id": entity_id, **(data or {})}
        url = f"{self.url}/api/services/{domain}/{service}?return_response=true"
        label = f"call_service_with_response({domain}.{service})"
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            # No headers= here: _ensure_session already builds the session with
            # self.headers (verified 2026-08-04, see _ensure_session).
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                elapsed = (time.perf_counter() - t_start) * 1000
                # Mismo trato que call_service / _get_entity_state_rest: sin
                # esto un token de HA vencido es indistinguible de un sensor
                # caído, y `_has_auth_error` —que existe justamente para poder
                # decir "mi token está muerto"— nunca se prende.
                if resp.status in (401, 403):
                    self._has_auth_error = True
                    logger.error(
                        f"HA auth error {resp.status}: {domain}.{service} on {entity_id}"
                    )
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                        ),
                        label,
                    )
                    return None
                if resp.status != 200:
                    try:
                        body_snippet = (await resp.text())[:300]
                    except Exception:
                        body_snippet = "<unreadable body>"
                    logger.warning(
                        f"Error {resp.status}: {domain}.{service} on {entity_id} "
                        f"({elapsed:.0f}ms): {body_snippet}"
                    )
                    self._record_error(RuntimeError(f"HTTP {resp.status}"), label)
                    return None
                body = await resp.json()
                self._record_success(elapsed)
                return body
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(
                f"HA timeout calling {domain}.{service} on {entity_id} "
                f"after {elapsed:.0f}ms (limit {timeout}s)"
            )
            self._record_error(asyncio.TimeoutError(), label)
            return None
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable ({domain}.{service} with response): {e}")
            self._record_error(e, label)
            return None
        except Exception as exc:  # noqa: BLE001 - logged and surfaced as None
            logger.error(
                f"Unexpected error in {label} on {entity_id}: {exc}",
                exc_info=True,
            )
            self._record_error(exc, label)
            return None

    # ==================== Automatizaciones ====================

    async def create_automation(self, automation_id: str, config: dict) -> tuple[bool, str]:
        """Crear una nueva automatizacion.

        Args:
            automation_id: ID unico para la automatizacion
            config: Configuracion de la automatizacion en formato HA

        Returns:
            (success, error_message)
        """
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.url}/api/config/automation/config/{automation_id}",
                json=config,
                timeout=aiohttp.ClientTimeout(total=AUTOMATION_TIMEOUT)
            ) as response:
                elapsed = (time.perf_counter() - t_start) * 1000
                if response.status in (401, 403):
                    self._has_auth_error = True
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        ),
                        "create_automation",
                    )
                    return False, FALLBACK_MSG_AUTH
                if response.status in [200, 201]:
                    # Recargar automatizaciones
                    await self._reload_automations()
                    self._record_success(elapsed)
                    logger.info(f"Automatizacion creada: {automation_id} ({elapsed:.0f}ms)")
                    return True, "OK"
                else:
                    error = await response.text()
                    logger.error(f"Error creando automatizacion: {error}")
                    return False, error
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA timeout creating automation {automation_id} after {elapsed:.0f}ms")
            self._record_error(asyncio.TimeoutError(), "create_automation")
            return False, FALLBACK_MSG_TIMEOUT
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (create_automation): {e}")
            self._record_error(e, "create_automation")
            return False, FALLBACK_MSG_UNAVAILABLE
        except Exception as e:
            logger.error(f"Error creando automatizacion: {e}")
            self._record_error(e, "create_automation")
            return False, str(e)

    async def delete_automation(self, automation_id: str) -> bool:
        """Eliminar una automatizacion."""
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.delete(
                f"{self.url}/api/config/automation/config/{automation_id}",
                timeout=aiohttp.ClientTimeout(total=AUTOMATION_TIMEOUT)
            ) as response:
                elapsed = (time.perf_counter() - t_start) * 1000
                if response.status in [200, 204]:
                    await self._reload_automations()
                    self._record_success(elapsed)
                    logger.info(f"Automatizacion eliminada: {automation_id} ({elapsed:.0f}ms)")
                    return True
                logger.warning(f"Delete automation {automation_id} returned status {response.status}")
                return False
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA timeout deleting automation {automation_id} after {elapsed:.0f}ms")
            self._record_error(asyncio.TimeoutError(), "delete_automation")
            return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (delete_automation): {e}")
            self._record_error(e, "delete_automation")
            return False
        except Exception as e:
            logger.error(f"Error eliminando automatizacion: {e}")
            self._record_error(e, "delete_automation")
            return False

    async def get_automations(self) -> list[dict]:
        """Obtener lista de automatizaciones."""
        entities = await self.get_all_entities()
        return [e for e in entities if e["entity_id"].startswith("automation.")]

    async def _reload_automations(self) -> None:
        """Recargar automatizaciones en Home Assistant."""
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.url}/api/services/automation/reload",
                timeout=aiohttp.ClientTimeout(total=RELOAD_TIMEOUT)
            ) as response:
                if response.status not in (200, 204):
                    logger.warning(f"Automation reload returned status {response.status}")
        except Exception as e:
            logger.error(f"Error reloading automations: {e}")

    # ==================== WebSocket (menor latencia) ====================

    async def _open_ws_authenticated(self, purpose: str):
        """Join a retained, generation-bound handshake for one channel."""
        if purpose not in self._ws_handshake_locks or self._closed:
            return None
        task = self._ws_open_tasks.get(purpose)
        if task is None or task.done():
            task = self._retain_transport(self._open_ws_owned(purpose,
                self._transport_generation, self._state_generation,
                self._events_attempt_generation), purpose)
            self._ws_open_tasks[purpose] = task
            def finished(done):
                if self._ws_open_tasks.get(purpose) is done:
                    self._ws_open_tasks.pop(purpose, None)
            task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def _open_ws_owned(self, purpose, generation, state_generation, attempt_generation):
        async with self._ws_handshake_locks[purpose]:
            def current():
                return self._transport_current(generation, purpose, state_generation, attempt_generation)
            if not current():
                return None
            await self._retire_socket(purpose)
            for previous, kind in tuple(self._owned_sockets.values()):
                if kind == purpose:
                    await self._close_socket(previous, purpose)
            if purpose == 'events':
                await self._drain_state_projections('events')
            if not current():
                return None
            ws = None
            published = False
            deadline = monotonic() + STATE_SYNC_ATTEMPT_TIMEOUT_S
            self._observe_channel(purpose, 'connecting', HAConnectionState.UNKNOWN,
                'ha_connecting', attempted_at=time.time())
            try:
                async with asyncio.timeout(STATE_SYNC_ATTEMPT_TIMEOUT_S):
                    session = await self._ensure_session()
                    if not current():
                        return None
                    options = {'heartbeat': 30.0}
                    if purpose == 'events':
                        options['max_msg_size'] = STATE_STREAM_MAX_MESSAGE_BYTES
                    ws = await session.ws_connect(self.url.replace('http', 'ws') + '/api/websocket', **options)
                    self._owned_sockets[id(ws)] = (ws, purpose)
                    if not current():
                        return None
                    json_options = {'loads': _raw_state_json} if purpose == 'events' else {}
                    first = await asyncio.wait_for(ws.receive_json(**json_options), timeout=WS_AUTH_TIMEOUT)
                    if purpose == 'events':
                        first = await self._decode_state_message(first, deadline=deadline,
                            state_generation=state_generation)
                    if not isinstance(first, dict) or first.get('type') != 'auth_required':
                        raise ValueError('ha_ws_protocol_invalid')
                    if not current():
                        return None
                    await ws.send_json({'type': 'auth', 'access_token': self.token})
                    result = await asyncio.wait_for(ws.receive_json(**json_options), timeout=WS_AUTH_TIMEOUT)
                    auth_observed_at = time.time()
                    if purpose == 'events':
                        result = await self._decode_state_message(result, deadline=deadline,
                            state_generation=state_generation)
                    if not isinstance(result, dict) or result.get('type') != 'auth_ok':
                        if isinstance(result, dict) and result.get('type') == 'auth_invalid':
                            if current():
                                self._has_auth_error = True
                                self._observe_channel(purpose, 'denied', HAConnectionState.AUTH_ERROR,
                                    'ha_auth_denied', evidence=True)
                            logger.warning('HA WebSocket authentication denied [%s]', purpose)
                            return None
                        raise ValueError('ha_ws_protocol_invalid')
                    if not current() or monotonic() >= deadline:
                        return None
                    setattr(self, '_ws_' + purpose, ws)
                    if purpose == 'calls':
                        self._ws_connected = True
                    self._observe_channel(purpose, 'authenticated',
                        HAConnectionState.UNKNOWN if purpose == 'events' else HAConnectionState.CONNECTED,
                        evidence=True, observed_at=auth_observed_at)
                    published = True
                    return ws
            except (asyncio.TimeoutError, aiohttp.ClientError, ValueError, OSError) as error:
                if current():
                    self._record_error(error, 'open_ws(' + purpose + ')', channel=purpose)
                logger.warning('HA WebSocket unavailable [%s]: %s', purpose, type(error).__name__)
                return None
            finally:
                if ws is not None and not published:
                    await self._close_socket(ws, purpose)


    async def connect_websocket(self) -> bool:
        """Join one calls connection owner; observer cancellation cannot lose it."""
        if self._closed:
            return False
        self._observe_channel('calls', self._channel_observations['calls'].phase,
            self._channel_observations['calls'].state, required=True)
        if self._ws_connected and self._ws_calls is not None and not self._ws_calls.closed:
            return True
        task = self._calls_connect_task
        if task is None or task.done():
            task = self._calls_connect_task = self._retain_transport(
                self._connect_calls_owned(self._transport_generation), 'calls')
            def finished(done):
                if self._calls_connect_task is done:
                    self._calls_connect_task = None
            task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def _connect_calls_owned(self, generation) -> bool:
        async with self._ws_connect_lock:
            if self._closed or generation != self._transport_generation:
                return False
            if self._ws_connected and self._ws_calls is not None and not self._ws_calls.closed:
                return True
            await self._retire_socket('calls')
            if self._closed or generation != self._transport_generation:
                return False
            try:
                ws = await self._open_ws_authenticated('calls')
            except aiohttp.ClientError:
                return False
            if ws is None:
                self._ws_connected = False
                return False
            if self._closed or generation != self._transport_generation:
                await self._retire_socket('calls', ws)
                return False
            self._ws_calls = ws
            self._ws_connected = True
            self._ws_reconnect_attempts = 0
            return True


    async def ensure_websocket_connected(self) -> bool:
        """Asegurar que el canal de calls esta conectado, reconectar si es necesario."""
        if self._ws_connected and self._ws_calls:
            # Verificar que la conexion sigue viva
            if self._ws_calls.closed:
                self._ws_connected = False

        if not self._ws_connected:
            if self._ws_reconnect_attempts < self._ws_max_reconnect_attempts:
                self._ws_reconnect_attempts += 1
                logger.debug(f"Reconectando WebSocket HA (intento {self._ws_reconnect_attempts})")
                return await self.connect_websocket()

        return self._ws_connected

    async def call_service_ws(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: dict | None = None
    ) -> bool:
        """Llamar servicio via WebSocket (mas rapido que REST).

        Latencia tipica: 10-20ms vs 50-100ms de REST.
        Falls back to REST automatically on WS failure.

        Si la conexión tuvo que abrirse dentro de esta misma llamada (primer
        comando después de un hueco idle, incidente 2026-07-25), un único
        fallo no es evidencia confiable: se reintenta una vez sobre el socket
        nuevo antes de darlo por perdido. Sobre una conexión ya establecida NO
        se reintenta — ahí el fallo es real y un retry dispararía el servicio
        dos veces.
        """
        async with self._ws_call_lock:
            # ¿La conexión ya estaba viva antes de esta llamada?
            was_established = (
                self._ws_connected
                and self._ws_calls is not None
                and not self._ws_calls.closed
            )

            # Asegurar conexion WebSocket
            if not await self.ensure_websocket_connected():
                logger.debug("WebSocket no disponible, usando REST")
                return await self.call_service(domain, service, entity_id, data)

            result = await self._send_ws_call(domain, service, entity_id, data)

            if result is True:
                return result
            if result is False:
                if was_established:
                    return False
                logger.info(
                    f"[WS-RETRY] {domain}.{service}@{entity_id} falló sobre una "
                    f"conexión recién abierta — un reintento"
                )
                retry = await self._send_ws_call(domain, service, entity_id, data)
                if retry is not None:
                    return retry

            # result is None (o el retry también fue None) → WS inutilizable.
            return await self.call_service(domain, service, entity_id, data)

    async def _receive_ws_result(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        msg_id: int,
    ) -> dict:
        """Leer frames de `ws` hasta encontrar el result de `msg_id`.

        HA scopea los ids por conexión y puede intercalar frames que no son
        nuestros: la respuesta tardía de una llamada previa que agotó
        WS_CALL_TIMEOUT y cayó a REST, o un event si alguien suscribió en este
        canal. Aceptar el primer frame a ciegas corre el stream un lugar de
        forma PERMANENTE — cada llamada leería el resultado de la anterior.

        WS_CALL_TIMEOUT acota el total de la espera, no cada frame individual.

        Args:
            ws: Socket del canal de calls sobre el que se mandó el request.
            msg_id: Id que mandamos y cuyo result esperamos.

        Returns:
            El frame cuyo "id" coincide con msg_id.

        Raises:
            asyncio.TimeoutError: si se agota WS_CALL_TIMEOUT sin encontrarlo.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WS_CALL_TIMEOUT
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            frame = await asyncio.wait_for(ws.receive_json(), timeout=remaining)
            if frame.get("id") == msg_id:
                return frame
            logger.warning(
                f"[WS-DESYNC] descarto frame ajeno id={frame.get('id')} "
                f"type={frame.get('type')} (esperando id={msg_id})"
            )

    async def _send_ws_call(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: dict | None = None,
    ) -> bool | None:
        """Un intento de call_service por el canal WS de calls.

        Returns:
            True/False según el `success` del result de HA, o None si el WS
            quedó inutilizable y el caller debe caer a REST.
        """
        ws = self._ws_calls
        if ws is None:
            return None

        t_start = time.perf_counter()
        try:
            # ID unico para este mensaje — counter propio del canal de calls.
            self._ws_msg_id_calls += 1
            msg_id = self._ws_msg_id_calls

            service_data = {"entity_id": entity_id}
            if data:
                service_data.update(data)

            await ws.send_json({
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "service_data": service_data
            })

            response = await self._receive_ws_result(ws, msg_id)

            elapsed = (time.perf_counter() - t_start) * 1000
            success = response.get("success", False)
            if success:
                self._record_success(elapsed, channel="calls")
                logger.debug(f"WS: {domain}.{service} -> {entity_id} ({elapsed:.0f}ms)")
            else:
                # El body es la única evidencia de POR QUÉ falló. Sin esto, un
                # "success=False took=8ms" es inatribuible (incidente 07-25).
                logger.warning(
                    f"[WS-FAIL] {domain}.{service}@{entity_id} success=False "
                    f"({elapsed:.0f}ms) response={response}"
                )
            return success

        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.warning(
                f"WebSocket timeout for {domain}.{service} on {entity_id} "
                f"after {elapsed:.0f}ms (limit {WS_CALL_TIMEOUT}s), fallback to REST"
            )
            self._record_error(asyncio.TimeoutError(), f"call_service_ws({domain}.{service})", channel="calls")
            return None

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False  # Marcar para reconexion
            self._record_error(e, f"call_service_ws({domain}.{service})", channel="calls")
            return None


    # ==================== State Prefetch Cache (S6) ====================

    async def start_state_sync(self, full_refresh_interval_s: float = 300.0) -> None:
        """Publish the retained warmup/loop owner before any startup await."""
        if not finite_number(full_refresh_interval_s) or full_refresh_interval_s <= 0:
            raise ValueError('invalid_ha_full_refresh_interval')
        if self._closed:
            return
        stopping = self._state_stop_task
        if (self._state_sync_running and self._state_subscribe_task is not None
                and not self._state_subscribe_task.done()):
            ready = self._state_start_ready
            if ready is not None:
                await asyncio.shield(ready)
            return
        self._state_generation += 1
        generation = self._state_generation
        self._state_sync_running = True
        self._state_full_refresh_interval_s = full_refresh_interval_s
        ready = self._state_start_ready = asyncio.get_running_loop().create_future()
        self._observe_channel('events', 'starting', HAConnectionState.UNKNOWN,
            'ha_state_sync_starting', required=True)
        self._state_subscribe_task = asyncio.create_task(
            self._state_sync_owned(generation, stopping, ready), context=Context())
        self._state_subscribe_task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(ready)

    async def _state_sync_owned(self, generation, stopping, ready):
        warm = False
        try:
            if stopping is not None:
                await asyncio.shield(stopping)
            if not self._state_current(generation):
                return
            try:
                async with asyncio.timeout(STATE_SYNC_ATTEMPT_TIMEOUT_S):
                    if not self._has_auth_error:
                        await self._wait_for_ws_ready(max_attempts=1, backoff_s=0.)
                        warm = True
                        if self._state_current(generation):
                            await self.connect_websocket()
            except (RuntimeError, aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning('HA warmup deferred to bounded state recovery')
            finally:
                if not ready.done():
                    ready.set_result(warm and self._state_current(generation))
            if not self._state_current(generation):
                return
            if self._state_current(generation):
                await self._state_sync_loop()
        except asyncio.CancelledError:
            return
        finally:
            if not ready.done():
                ready.set_result(False)
            if generation == self._state_generation:
                self._state_sync_running = False

    def _state_current(self, generation):
        return not self._closed and self._state_sync_running and generation == self._state_generation

    async def _state_retry_wait(self, delay: float) -> None:
        await asyncio.sleep(delay)


    async def _wait_for_ws_ready(self, max_attempts: int = 3, backoff_s: float = 2.0) -> None:
        generation = self._state_generation
        for attempt in range(max_attempts):
            if self._closed or generation != self._state_generation:
                return
            try:
                ws = await self._open_ws_authenticated('events')
            except aiohttp.ClientError:
                raise
            except ConnectionError:
                ws = None
            if self._closed or generation != self._state_generation:
                if ws is not None:
                    await self._retire_socket('events', ws)
                return
            if ws is not None:
                self._ws_events = ws
                return
            if attempt + 1 < max_attempts:
                await self._state_retry_wait(backoff_s)
        raise RuntimeError('ha_events_warmup_unavailable')


    async def stop_state_sync(self) -> None:
        """Seal events synchronously; retain actual retirement across observer cancel."""
        task = self._state_stop_task
        owner = self._state_subscribe_task
        if task is None or task.done() or self._state_stop_owner is not owner:
            previous = task
            self._state_generation += 1
            self._events_attempt_generation += 1
            self._entity_stream_generation += 1
            generation = self._state_generation
            self._state_sync_running = False
            self._state_stop_owner = owner
            if self._state_start_ready is not None and not self._state_start_ready.done():
                self._state_start_ready.set_result(False)
            self._observe_channel('events', 'stopping', HAConnectionState.UNKNOWN,
                'ha_state_sync_stopping', required=False)
            task = self._state_stop_task = asyncio.create_task(
                self._stop_state_owned(owner, generation, previous), context=Context())
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(task)

    async def _stop_state_owned(self, owner, generation, previous=None):
        if owner is not None and not owner.done():
            owner.cancel()
        if previous is not None:
            await asyncio.shield(previous)
        await self._retire_socket('events')
        if owner is not None:
            await asyncio.gather(asyncio.shield(owner), return_exceptions=True)
        await self._drain_transports('events')
        for ws, purpose in tuple(self._owned_sockets.values()):
            if purpose == 'events':
                await self._close_socket(ws, purpose)
        if generation == self._state_generation:
            if self._state_subscribe_task is owner:
                self._state_subscribe_task = None
            self._observe_channel('events', 'stopped', HAConnectionState.DISCONNECTED,
                'ha_state_sync_stopped', required=False)


    async def _state_sync_loop(self) -> None:
        """One event receiver with capped retry delay; no service/action replay."""
        generation = self._state_generation
        backoff = STATE_SYNC_BACKOFF_INITIAL_S
        while self._state_current(generation):
            try:
                await self._subscribe_and_sync()
                backoff = STATE_SYNC_BACKOFF_INITIAL_S
                if self._state_rebuild_requested and self._state_current(generation):
                    self._state_rebuild_requested = False
                    continue
            except asyncio.CancelledError:
                return
            except Exception as error:
                if not self._state_current(generation):
                    return
                logger.warning('HA events attempt failed: %s', type(error).__name__)
                self._observe_channel('events', 'retrying',
                    self._channel_observations['events'].state, 'ha_events_retry_pending')
            if self._state_current(generation):
                await self._state_retry_wait(backoff)
                backoff = min(backoff * 2, STATE_SYNC_BACKOFF_MAX_S)


    async def _subscribe_and_sync(self) -> None:
        """One coherent initial frame followed by serial compact deltas."""
        self._state_rebuild_requested = False
        generation = self._state_generation
        self._entity_stream_generation += 1
        stream_generation = self._entity_stream_generation
        ws = None
        started = time.perf_counter()
        deadline = monotonic() + STATE_SYNC_ATTEMPT_TIMEOUT_S
        try:
            async with asyncio.timeout(STATE_SYNC_ATTEMPT_TIMEOUT_S):
                if not self._state_current(generation):
                    return
                ws = self._ws_events
                if ws is None or ws.closed:
                    ws = await self._open_ws_authenticated('events')
                if ws is None or not self._state_current(generation):
                    raise aiohttp.ClientConnectionError('ha_entities_unavailable')
                self._ws_events = ws
                self._ws_msg_id_events += 1
                subscribe_id = self._ws_msg_id_events
                admitted_revision = self._state_frame.revision
                await ws.send_json({'id': subscribe_id, 'type': 'subscribe_entities'})
                confirm = await asyncio.wait_for(ws.receive_json(loads=_raw_state_json), timeout=WS_AUTH_TIMEOUT)
                confirm = await self._decode_state_message(confirm, deadline=deadline,
                    state_generation=generation)
                if (not isinstance(confirm, dict) or type(confirm.get('id')) is not int
                        or confirm['id'] != subscribe_id or confirm.get('type') != 'result'
                        or confirm.get('success') is not True):
                    raise ValueError('ha_entities_confirmation_invalid')
                first = await asyncio.wait_for(ws.receive_json(loads=_raw_state_json), timeout=WS_AUTH_TIMEOUT)
                observed_at = self._entity_now()
                if (not self._state_current(generation) or monotonic() >= deadline
                        or stream_generation != self._entity_stream_generation):
                    return
                await self._publish_entities_initial(first, stream_generation=stream_generation,
                    admitted_revision=admitted_revision, subscribe_id=subscribe_id,
                    observed_at=observed_at, deadline=deadline)
                if not self._state_current(generation) or stream_generation != self._entity_stream_generation:
                    return
                self._record_success((time.perf_counter() - started) * 1000, channel='events',
                    observed_at=observed_at)
                self._observe_channel('events', 'subscribed', HAConnectionState.CONNECTED)
            full_deadline = monotonic() + self._state_full_refresh_interval_s
            while (self._state_current(generation) and self._ws_events is ws
                    and stream_generation == self._entity_stream_generation):
                remaining = full_deadline - monotonic()
                if remaining <= 0:
                    self._state_rebuild_requested = True
                    return
                try:
                    message = await asyncio.wait_for(ws.receive_json(loads=_raw_state_json), timeout=min(30., remaining))
                except asyncio.TimeoutError:
                    if ws.closed:
                        raise aiohttp.ClientConnectionError('ha_entities_disconnected')
                    continue
                observed_at = self._entity_now()
                if (not self._state_current(generation)
                        or stream_generation != self._entity_stream_generation):
                    return
                await self._publish_entities_delta(message, subscribe_id=subscribe_id,
                    observed_at=observed_at, deadline=monotonic() + STATE_OBSERVATION_TTL_S)
                if not self._state_current(generation) or stream_generation != self._entity_stream_generation:
                    return
                self._observe_channel('events', 'subscribed', HAConnectionState.CONNECTED,
                    evidence=True, observed_at=observed_at)
        except Exception as error:
            if self._state_current(generation) and self._channel_observations['events'].state is not HAConnectionState.AUTH_ERROR:
                self._record_error(error, 'entity_stream', channel='events')
            raise
        finally:
            await self._retire_event_attempt(ws)

    @staticmethod
    def _entities_payload(message, subscribe_id):
        if (not isinstance(message, dict) or set(message) != {'id', 'type', 'event'}
                or type(message.get('id')) is not int or message['id'] != subscribe_id
                or message.get('type') != 'event' or not isinstance(message.get('event'), dict)):
            raise ValueError('ha_entities_event_invalid')
        return message['event']

    async def _publish_entities_initial(self, payload, *, stream_generation: int,
            admitted_revision: int | None = None, subscribe_id=None, observed_at=None, deadline=None) -> None:
        observed = self._entity_now() if observed_at is None else observed_at
        options = dict(transport_generation=self._transport_generation,
            state_generation=self._state_generation, stream_generation=stream_generation,
            admitted_revision=admitted_revision, subscribe_id=subscribe_id, require_running=True,
            observed_at=observed, deadline=monotonic() + STATE_OBSERVATION_TTL_S if deadline is None else deadline)
        await self._state_operation('events', self._project_state,
            'initial', payload, options, purpose='events')
        self._state_last_full_refresh = observed


    async def _publish_entities_delta(self, payload, *, subscribe_id=None, observed_at=None, deadline=None) -> None:
        options = dict(transport_generation=self._transport_generation,
            state_generation=self._state_generation, stream_generation=self._entity_stream_generation,
            subscribe_id=subscribe_id, require_running=True,
            observed_at=self._entity_now() if observed_at is None else observed_at,
            deadline=monotonic() + STATE_OBSERVATION_TTL_S if deadline is None else deadline)
        await self._state_operation('events', self._project_state,
            'delta', payload, options, purpose='events')


    def _notify_entity_change(self, entity_id, *, historical=False):
        for callback in tuple(self._state_callbacks):
            try:
                value = self.get_entity_observation(entity_id)
                admitted = (value.exists is True and value.current(self._entity_now())
                    or historical and value.exists is not False)
                callback(entity_id, value.state if admitted else None)
            except Exception as error:
                logger.debug('HA state callback failed: %s', type(error).__name__)


    async def _retire_event_attempt(self, ws=None):
        self._events_attempt_generation += 1
        self._entity_stream_generation += 1
        opening = self._ws_open_tasks.get('events')
        await self._retire_socket('events', ws)
        if opening is not None:
            await asyncio.gather(asyncio.shield(opening), return_exceptions=True)
        await self._retire_socket('events')
        await self._drain_state_projections('events')


    async def _refresh_full_state_snapshot(self) -> None:
        """Explicit legacy bulk publication, sharing the retained REST discipline."""
        if self._closed or self._state_frame.initialized:
            return
        deadline = monotonic() + REST_TIMEOUT_BULK
        try:
            await self._state_operation('rest', self._refresh_full_state_snapshot_owned, deadline)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, KeyError):
            return

    async def _refresh_full_state_snapshot_owned(self, deadline) -> None:
        previous = self._state_frame
        transport = self._transport_generation
        snapshot = await self.get_state_snapshot(_deadline=deadline)
        if (self._closed or transport != self._transport_generation or self._state_frame.initialized
                or self._state_frame.generation != previous.generation):
            return
        if (not snapshot.available or not snapshot.states or snapshot.observed_at is None
                or snapshot.expires_at is None):
            return
        options = dict(transport_generation=transport, deadline=deadline,
            expected_generation=previous.generation, base=previous)
        operation = self._project_state('bulk', snapshot, options, purpose='rest')
        previous = options = None
        await operation
        self._state_last_full_refresh = snapshot.observed_at.timestamp()




    def _handle_state_changed(self, event: dict) -> None:
        """Legacy manual history adapter; runtime uses the compact publisher."""
        if (self._closed or self._state_frame.initialized or not isinstance(event, dict)
                or any(not task.done() for jobs in self._state_projection_tasks.values() for task in jobs)):
            return
        data = event.get('data', {})
        if not isinstance(data, dict):
            return
        entity_id = data.get('entity_id')
        state = data.get('new_state')
        if not isinstance(entity_id, str) or state is not None and not isinstance(state, dict):
            return
        history = self._state_cache
        if state is None:
            history.pop(entity_id, None)
        else:
            history[entity_id] = {**state, 'entity_id': entity_id}
        self._state_cache = history
        self._notify_entity_change(entity_id, historical=True)


    def register_state_callback(self, callback) -> None:
        """Registrar un callback `(entity_id, new_state) -> None` para cambios.

        Useful para integraciones que quieren reaccionar proactivamente a
        cambios de state (ej: alerts, analytics).
        """
        self._state_callbacks.append(callback)

    async def close(self) -> None:
        """Terminal admission seal with retained socket/session retirement."""
        if self._close_task is None:
            self._closed = True
            self._state_projection_close = self._state_projection_work.begin_close()
            self._transport_generation += 1
            self._state_generation += 1
            self._events_attempt_generation += 1
            self._state_sync_running = False
            if self._state_start_ready is not None and not self._state_start_ready.done():
                self._state_start_ready.set_result(False)
            for channel in self._channel_observations:
                self._observe_channel(channel, 'closing', HAConnectionState.UNKNOWN,
                    'ha_client_closing', required=False)
            self._close_task = asyncio.create_task(self._close_owned(), context=Context())
            self._close_task.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())
        await asyncio.shield(self._close_task)

    async def _close_owned(self):
        await self.stop_state_sync()
        for purpose in ('events', 'calls', 'registry'):
            await self._retire_socket(purpose)
        for ws, purpose in tuple(self._owned_sockets.values()):
            await self._close_socket(ws, purpose)
        await self._drain_transports()
        if self._state_projection_close is not None:
            await asyncio.shield(self._state_projection_close)
        for ws, purpose in tuple(self._owned_sockets.values()):
            await self._close_socket(ws, purpose)
        async with self._session_lock:
            session = self._session
            if self._session_owned and session is not None and not session.closed:
                await session.close()
                if session.closed is not True:
                    raise aiohttp.ClientConnectionError('ha_session_close_unconfirmed')
        self._ws_calls = self._ws_events = self._ws_registry = None
        self._ws_connected = False
        for channel in self._channel_observations:
            self._observe_channel(channel, 'closed', HAConnectionState.DISCONNECTED,
                'ha_client_closed', required=False)


    # ==================== Health Status ====================

    def get_health_status(self) -> HAHealthStatus:
        """Return structured health information about the HA connection.

        Returns:
            HAHealthStatus with current metrics.
        """
        if self._has_auth_error:
            state = HAConnectionState.AUTH_ERROR
        elif self._last_success_ts > 0 and (
            self._last_error_ts == 0 or self._last_success_ts > self._last_error_ts
        ):
            state = HAConnectionState.CONNECTED
        elif self._error_count > 0 and (
            self._last_success_ts == 0 or self._last_error_ts > self._last_success_ts
        ):
            state = HAConnectionState.DISCONNECTED
        else:
            state = HAConnectionState.UNKNOWN

        return HAHealthStatus(
            state=state,
            last_success_ts=self._last_success_ts,
            last_error_ts=self._last_error_ts,
            error_count=self._error_count,
            success_count=self._success_count,
            avg_latency_ms=round(self._avg_latency_ms, 2),
            ws_connected=self._ws_connected,
            last_error_message=self._last_error_message,
            rest=self._channel_status('rest'),
            calls=self._channel_status('calls'),
            events=self._channel_status('events'),
            registry=self._channel_status('registry'),
        )

    # ==================== Utilidades ====================

    async def get_domotics_entities(self) -> list[dict]:
        """Obtener solo entidades relevantes para domotica."""
        domotics_domains = [
            "light", "switch", "cover", "climate", "fan",
            "media_player", "vacuum", "lock", "scene",
            "script", "automation", "input_boolean"
        ]

        entities = await self.get_all_entities()
        return [
            e for e in entities
            if e["entity_id"].split(".")[0] in domotics_domains
        ]

    async def get_services_by_domain(self) -> dict[str, list[str]]:
        """Obtener servicios organizados por dominio."""
        services = await self.get_all_services()
        return {
            s["domain"]: list(s["services"].keys())
            for s in services
        }

    async def test_connection(self) -> bool:
        """Verificar conexion con Home Assistant."""
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/",
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_HEALTH)
            ) as response:
                elapsed = (time.perf_counter() - t_start) * 1000
                if response.status == 200:
                    self._record_success(elapsed)
                    return True
                self._record_error(
                    aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                    ),
                    "test_connection",
                )
                return False
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA connection test timeout after {elapsed:.0f}ms")
            self._record_error(asyncio.TimeoutError(), "test_connection")
            return False
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (test_connection): {e}")
            self._record_error(e, "test_connection")
            return False
        except Exception as e:
            logger.error(f"Error testing HA connection: {e}")
            self._record_error(e, "test_connection")
            return False
