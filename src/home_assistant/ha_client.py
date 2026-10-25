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
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic

import aiohttp

from src.home_assistant.operations import ActionIntent, DOMAIN_OPERATIONS, action_capabilities, finite_number, matches_observation


class ServiceCallStatus(StrEnum):
    """Certainty of a single REST service attempt, independent of state."""

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


@dataclass(frozen=True)
class HAAreaAssignments:
    """Reduced registry observation with entity and device areas reconciled."""

    available: bool
    observed_at: datetime | None
    reason: str | None
    area_names: dict[str, str] = field(default_factory=dict)
    entity_area_ids: dict[str, str] = field(default_factory=dict)


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
WS_AUTH_TIMEOUT = 5.0           # WebSocket auth handshake timeout
WS_CALL_TIMEOUT = 2.0           # WebSocket service-call timeout
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
                 state_confirmation_timeout_s: float = 0.3, operation_confirmation_timeouts: dict[str, float] | None = None):
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
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
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
        self._state_cache: dict[str, dict] = {}
        self._state_subscribe_task: asyncio.Task | None = None
        self._state_callbacks: list = []  # observers opcionales
        self._state_sync_running = False
        self._state_last_full_refresh: float = 0.0
        self._state_full_refresh_interval_s: float = 300.0

        # Dashboard reads have independent provenance; the voice cache is not
        # evidence of a successful current REST observation.
        self._home_states = HAStatesSnapshot(False, None, "ha_states_unavailable")
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

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure an aiohttp ClientSession exists, creating one lazily if needed.

        Returns:
            Active aiohttp.ClientSession instance.
        """
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
        return self._session

    def _record_success(self, latency_ms: float) -> None:
        """Record a successful HA call for health tracking."""
        self._success_count += 1
        self._last_success_ts = time.time()
        self._has_auth_error = False
        # Exponential moving average (alpha=0.3)
        if self._avg_latency_ms == 0.0:
            self._avg_latency_ms = latency_ms
        else:
            self._avg_latency_ms = 0.7 * self._avg_latency_ms + 0.3 * latency_ms

    def _record_error(self, error: Exception, context: str) -> None:
        """Record a failed HA call for health tracking."""
        self._error_count += 1
        self._last_error_ts = time.time()
        self._last_error_message = f"{context}: {error}"
        if isinstance(error, aiohttp.ClientResponseError) and error.status in (401, 403):
            self._has_auth_error = True

    # ==================== REST API ====================

    async def get_state_snapshot(self) -> HAStatesSnapshot:
        """Read bounded REST states and distinguish failure from an empty home.

        Returns:
            Typed observation with its successful read time and failure reason.
        """
        started = time.perf_counter()
        reason = "ha_states_unavailable"
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/states",
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_BULK),
            ) as response:
                response.raise_for_status()
                states = await response.json()
            if not isinstance(states, list) or any(
                not isinstance(state, dict)
                or not isinstance(state.get("entity_id"), str)
                or "." not in state["entity_id"]
                or not isinstance(state.get("state"), str)
                or not isinstance(state.get("attributes", {}), dict)
                for state in states
            ):
                reason = "ha_states_invalid"
                raise ValueError("Invalid HA states response")
            self._home_states = HAStatesSnapshot(True, datetime.now(UTC), None, tuple(states))
            self._record_success((time.perf_counter() - started) * 1000)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            self._record_error(error, "get_state_snapshot")
            self._home_states = replace(self._home_states, available=False, reason=reason)
            logger.warning("HA dashboard state snapshot unavailable")
        return self._home_states

    async def get_area_assignments(self) -> HAAreaAssignments:
        """Read registries on an owned socket, caching one refresh for 60 seconds.

        Returns:
            Reduced registry data, retaining previous content and time on failure.
        """
        async with self._registry_lock:
            if self._area_assignments is not None and monotonic() < self._registry_refresh_at:
                return self._area_assignments
            previous = self._area_assignments or HAAreaAssignments(False, None, None)
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
                self._area_assignments = HAAreaAssignments(
                    True, datetime.now(UTC), None, names, assignments,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as error:
                self._record_error(error, "get_area_assignments")
                self._area_assignments = replace(
                    previous, available=False, reason="ha_registries_unavailable",
                )
                logger.warning("HA dashboard registries unavailable")
            self._registry_refresh_at = monotonic() + 60.0
            return self._area_assignments

    async def _read_home_registries(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Read only registry list commands with a bounded temporary connection."""
        ws = None
        try:
            async with asyncio.timeout(REST_TIMEOUT_BULK):
                ws = await self._open_ws_authenticated("registry")
                if ws is None:
                    raise ValueError("Registry websocket unavailable")
                results = []
                for message_id, registry in enumerate(("area", "entity", "device"), start=1):
                    await ws.send_json({"id": message_id, "type": f"config/{registry}_registry/list"})
                    message = await ws.receive_json()
                    if (
                        not isinstance(message, dict)
                        or message.get("id") != message_id
                        or message.get("type") != "result"
                        or message.get("success") is not True
                    ):
                        raise ValueError("Invalid registry result")
                    result = message.get("result")
                    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
                        raise ValueError("Invalid registry entries")
                    results.append(result)
                return results[0], results[1], results[2]
        finally:
            if ws is not None:
                await ws.close()

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
        """Obtener estado de una entidad.

        Cache-first: si el prefetch cache tiene la entidad la devuelve sin I/O.
        Si no, fallback al REST endpoint.
        """
        cached = self.get_entity_state_cached(entity_id)
        if cached is not None:
            return cached
        return await self._get_entity_state_rest(entity_id)

    async def _get_entity_state_rest(self, entity_id: str) -> dict | None:
        """Obtener estado via REST (sin tocar cache). Usado como fallback."""
        t_start = time.perf_counter()
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/states/{entity_id}",
                timeout=aiohttp.ClientTimeout(total=REST_TIMEOUT_DEFAULT)
            ) as response:
                if response.status in (401, 403):
                    self._has_auth_error = True
                    logger.error(f"HA auth error {response.status} getting state for {entity_id}")
                    self._record_error(
                        aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        ),
                        f"get_entity_state({entity_id})",
                    )
                    return None
                response.raise_for_status()
                result = await response.json()
                elapsed = (time.perf_counter() - t_start) * 1000
                self._record_success(elapsed)
                return result
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.error(f"HA timeout getting state for {entity_id} after {elapsed:.0f}ms")
            self._record_error(asyncio.TimeoutError(), f"get_entity_state({entity_id})")
            return None
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (get_entity_state {entity_id}): {e}")
            self._record_error(e, f"get_entity_state({entity_id})")
            return None
        except Exception as e:
            logger.error(f"Error obteniendo estado de {entity_id}: {e}")
            self._record_error(e, f"get_entity_state({entity_id})")
            return None

    async def get_entity_state_fresh(self, entity_id: str, timeout: float = 0.15) -> dict | None:
        """Read a bounded REST state and refresh the local entity cache."""
        try:
            state = await asyncio.wait_for(self._get_entity_state_rest(entity_id), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if state is not None:
            self._state_cache[entity_id] = state
        return state

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
        """Lookup sin I/O contra el prefetch cache.

        Returns:
            Dict con el state más reciente conocido via WS push o REST snapshot,
            o None si no hay entry en cache (típicamente porque el sync aún no
            corrió o la entidad no existe).
        """
        return self._state_cache.get(entity_id)

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
        entry = self._state_cache.get(entity_id)
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

    def has_domain(self, domain: str) -> bool:
        """¿Hay al menos una entidad del dominio en el cache?

        Permite al pipeline rechazar intents que requieren un dominio inexistente
        (ej: `set_temperature` sin entidad `climate.*`) en vez de hacer fallback
        a un dominio similar por embedding.

        Args:
            domain: dominio HA (sin punto). Ej: "climate", "media_player", "light".

        Returns:
            True si hay ≥1 entidad cacheada con `entity_id` que empieza con `domain.`
        """
        prefix = f"{domain}."
        return any(eid.startswith(prefix) for eid in self._state_cache)

    async def _fetch_all_states_rest(self) -> list[dict]:
        """Wrapper sobre get_all_entities para claridad en el sync loop."""
        return await self.get_all_entities()

    async def call_service_once(
        self, domain: str, service: str, entity_id: str, data: dict | None = None,
    ) -> ServiceCallOutcome:
        """Send one REST POST without redirect, fallback, or automatic retry.

        Args:
            domain: Home Assistant domain.
            service: Service name in that domain.
            entity_id: Exact target; data cannot replace it.
            data: Additional service fields, excluding target selectors.

        Returns:
            Explicit rejection, HTTP acceptance, or transport uncertainty.
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
        started = time.perf_counter()
        try:
            session = await self._ensure_session()
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
            return ServiceCallOutcome(ServiceCallStatus.UNCERTAIN, "ha_transport_uncertain")

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

    async def _open_ws_authenticated(
        self, purpose: str
    ) -> aiohttp.ClientWebSocketResponse | None:
        """Open a new authenticated WebSocket connection to HA.

        Each HA WebSocket requires an auth handshake even with a long-lived
        token:
            1. Server sends {"type": "auth_required"}.
            2. Client sends {"type": "auth", "access_token": TOKEN}.
            3. Server sends {"type": "auth_ok"} on success.

        Args:
            purpose: "calls", "events" or "registry". Used only for log messages.

        Returns:
            The authenticated aiohttp.ClientWebSocketResponse, or None on
            failure. The caller decides which attribute to store the result on
            (self._ws_calls or self._ws_events).
        """
        ws_url = self.url.replace("http", "ws") + "/api/websocket"
        ws = None
        authenticated = False
        try:
            session = await self._ensure_session()

            ws = await session.ws_connect(
                ws_url,
                heartbeat=30.0,  # Ping cada 30s para mantener conexion
            )

            # Auth handshake
            msg = await asyncio.wait_for(
                ws.receive_json(),
                timeout=WS_AUTH_TIMEOUT,
            )

            if msg.get("type") != "auth_required":
                logger.error(
                    f"WS[{purpose}] unexpected first frame (expected auth_required): {msg}"
                )
                return None

            await ws.send_json({
                "type": "auth",
                "access_token": self.token,
            })

            auth_result = await asyncio.wait_for(
                ws.receive_json(),
                timeout=WS_AUTH_TIMEOUT,
            )

            if auth_result.get("type") == "auth_ok":
                logger.info(f"WebSocket HA[{purpose}] conectado y autenticado")
                authenticated = True
                return ws

            self._has_auth_error = True
            logger.error(f"Auth WebSocket[{purpose}] fallo: {auth_result}")
            return None

        except asyncio.TimeoutError:
            logger.error(f"WebSocket[{purpose}] auth timeout after {WS_AUTH_TIMEOUT}s")
            self._record_error(asyncio.TimeoutError(), f"open_ws({purpose})")
            return None
        except aiohttp.ClientConnectorError as e:
            logger.error(f"HA unavailable (open_ws[{purpose}]): {e}")
            self._record_error(e, f"open_ws({purpose})")
            return None
        except Exception as e:
            logger.error(f"Error conectando WebSocket HA[{purpose}]: {e}")
            self._record_error(e, f"open_ws({purpose})")
            return None
        finally:
            if ws is not None and not authenticated:
                await ws.close()

    async def connect_websocket(self) -> bool:
        """Conectar el canal de calls via WebSocket para menor latencia.

        Incluye auto-reconexion en caso de fallo. Usa _open_ws_authenticated
        para el handshake. Solo afecta al canal self._ws_calls.
        """
        if self._ws_connected and self._ws_calls and not self._ws_calls.closed:
            return True

        async with self._ws_connect_lock:
            # Re-chequear adentro del lock: otro caller pudo conectar mientras
            # esperábamos, y un segundo handshake filtraría ese socket.
            if self._ws_connected and self._ws_calls and not self._ws_calls.closed:
                return True

            previous = self._ws_calls
            ws = await self._open_ws_authenticated("calls")
            if ws is None:
                self._ws_connected = False
                return False

            self._ws_calls = ws
            self._ws_connected = True
            self._ws_reconnect_attempts = 0

            # Cerrar el socket que reemplazamos. aiohttp no libera el fd solo.
            if previous is not None and previous is not ws:
                try:
                    await previous.close()
                except Exception as e:
                    logger.debug(f"Error cerrando _ws_calls reemplazado: {e}")

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
                self._record_success(elapsed)
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
            self._record_error(asyncio.TimeoutError(), f"call_service_ws({domain}.{service})")
            return None

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False  # Marcar para reconexion
            self._record_error(e, f"call_service_ws({domain}.{service})")
            return None

    # ==================== State Prefetch Cache (S6) ====================

    async def start_state_sync(
        self,
        full_refresh_interval_s: float = 300.0,
    ) -> None:
        """Arrancar el loop de prefetch de state cache.

        1. Snapshot inicial REST (/api/states) para poblar el cache.
        2. Subscribe WS state_changed para updates incrementales.
        3. Refresh periódico como fallback si el WS se cae en silencio.

        Idempotente: si ya está corriendo, no-op.

        Args:
            full_refresh_interval_s: Cada cuántos segundos rehacer el snapshot
                REST completo para atrapar drift si se perdieron events.
        """
        if self._state_sync_running and self._state_subscribe_task and not self._state_subscribe_task.done():
            logger.debug("State sync ya está corriendo")
            return

        # Pre-condición: confirmar que el canal WS events handshake-ea OK
        # antes de marcar el servicio como ready. Mata el ruido de "auth
        # timeout" en logs cuando el primer ws_connect es lento (HA digiriendo
        # el snapshot REST recién hecho). Si todos los reintentos fallan,
        # propaga RuntimeError para que main.py decida (sigue/fail).
        await self._wait_for_ws_ready(max_attempts=3, backoff_s=2.0)

        # Pre-conectar canal [calls] al startup. Sin esto, el primer comando
        # del usuario paga ~90ms de SSL+auth handshake por el lazy connect.
        # Heartbeat=30s mantiene viva la conexión.
        try:
            await self.connect_websocket()
        except Exception as e:
            logger.warning(f"WS calls warmup falló (seguirá lazy): {e}")

        self._state_full_refresh_interval_s = full_refresh_interval_s
        self._state_sync_running = True
        self._state_subscribe_task = asyncio.create_task(self._state_sync_loop())
        logger.info("HA state prefetch cache: sync loop arrancado")

    async def _wait_for_ws_ready(
        self,
        max_attempts: int = 3,
        backoff_s: float = 2.0,
    ) -> None:
        """Pre-conectar el canal events con retries antes del background loop.

        En LAN, el primer ws_connect post-snapshot REST grande (~329 entities)
        puede demorar 5-7s mientras HA termina de servir el payload. El
        timeout default de 5s del handshake auth lo dispara aunque HA esté
        sano. Reintentar 3 veces con backoff resuelve el race sin ocultar
        fallos reales (un HA caído sigue fallando los 3 intentos en ~10s).

        Args:
            max_attempts: Cuántos intentos antes de propagar.
            backoff_s: Espera entre intentos (lineal, no exponencial — el
                problema es transient ms, no sostenido).

        Raises:
            RuntimeError: si tras `max_attempts` no se pudo abrir el WS.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                ws = await self._open_ws_authenticated("events")
                if ws is not None:
                    self._ws_events = ws
                    if attempt > 1:
                        logger.info(
                            f"WS events ready en intento {attempt}/{max_attempts}"
                        )
                    return
            except Exception as e:
                logger.warning(
                    f"WS warmup intento {attempt}/{max_attempts} excepción: {e}"
                )
            if attempt < max_attempts:
                await asyncio.sleep(backoff_s)
        raise RuntimeError(
            f"WS HA events no disponible tras {max_attempts} intentos"
        )

    async def stop_state_sync(self) -> None:
        """Parar el loop de sync (para shutdown graceful)."""
        self._state_sync_running = False
        if self._state_subscribe_task:
            self._state_subscribe_task.cancel()
            try:
                await self._state_subscribe_task
            except (asyncio.CancelledError, Exception):
                pass
            self._state_subscribe_task = None
        # Cerrar la conexión dedicada de events. Se reabrirá cuando start_state_sync
        # vuelva a correr.
        if self._ws_events is not None and not self._ws_events.closed:
            try:
                await self._ws_events.close()
            except Exception as e:
                logger.debug(f"Error cerrando _ws_events: {e}")
        self._ws_events = None

    async def _state_sync_loop(self) -> None:
        """Loop resiliente: reconnect con backoff exponencial si el WS falla.

        Nunca propaga excepciones — si algo sale mal, loguea y reintenta. El
        servicio de voz debe seguir funcionando aunque HA esté caído.
        """
        backoff = 5.0
        max_backoff = 60.0
        while self._state_sync_running:
            try:
                await self._subscribe_and_sync()
                # Si salimos limpiamente (WS cerrado por HA), resetear backoff
                backoff = 5.0
            except asyncio.CancelledError:
                logger.info("HA state sync cancelado")
                return
            except Exception as e:
                logger.warning(
                    f"HA state sync error: {e}, reconnectando en {backoff:.0f}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * 2, max_backoff)

    async def _subscribe_and_sync(self) -> None:
        """Ciclo de vida: snapshot REST → subscribe WS → consumir events.

        Cada iteración del outer loop (_state_sync_loop) rehace esto para
        garantizar consistencia después de una reconexión.

        Usa el canal dedicado self._ws_events — independiente del canal de
        calls — para evitar el race de aiohttp 'Concurrent call to receive()'.
        """
        # 1. Snapshot inicial REST para poblar cache completo
        await self._refresh_full_state_snapshot()

        # 2. Abrir (o reutilizar) la conexión dedicada de events.
        if self._ws_events is None or self._ws_events.closed:
            ws = await self._open_ws_authenticated("events")
            if ws is None:
                raise RuntimeError("No se pudo conectar al WebSocket de HA (events)")
            self._ws_events = ws

        # 3. Subscribe al event state_changed (id counter propio del canal).
        self._ws_msg_id_events += 1
        subscribe_id = self._ws_msg_id_events
        await self._ws_events.send_json({
            "id": subscribe_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
        })

        # Esperar confirmación del subscribe (result message)
        # HA responde con {"id": N, "type": "result", "success": true}
        try:
            confirm = await asyncio.wait_for(
                self._ws_events.receive_json(),
                timeout=WS_AUTH_TIMEOUT,
            )
        except asyncio.TimeoutError as e:
            raise RuntimeError("Timeout esperando confirmación de subscribe") from e

        if confirm.get("type") != "result" or not confirm.get("success", False):
            raise RuntimeError(f"Subscribe no confirmado: {confirm}")

        logger.info("HA state prefetch: suscripto a state_changed (canal events)")

        # 4. Consumir events hasta que el WS se cierre o nos cancelen
        last_full = time.time()
        while self._state_sync_running:
            try:
                msg = await asyncio.wait_for(
                    self._ws_events.receive_json(),
                    timeout=30.0,  # timeout largo, solo para no colgar para siempre
                )
            except asyncio.TimeoutError:
                # Sin events por 30s es normal (HA idle). Chequear si toca refresh.
                if time.time() - last_full > self._state_full_refresh_interval_s:
                    await self._refresh_full_state_snapshot()
                    last_full = time.time()
                continue

            if msg.get("type") == "event":
                self._handle_state_changed(msg.get("event", {}))

            # Refresh periódico para atrapar events perdidos (por si acaso)
            if time.time() - last_full > self._state_full_refresh_interval_s:
                await self._refresh_full_state_snapshot()
                last_full = time.time()

    async def _refresh_full_state_snapshot(self) -> None:
        """Repoblar cache completo via REST /api/states."""
        states = await self._fetch_all_states_rest()
        if not states:
            logger.warning("HA state snapshot vacío — posible error de conexión")
            return
        # Replace atomically: construimos dict nuevo y lo asignamos
        new_cache = {st["entity_id"]: st for st in states if "entity_id" in st}
        self._state_cache = new_cache
        self._state_last_full_refresh = time.time()
        logger.info(f"HA state cache snapshot: {len(new_cache)} entities")

    def _handle_state_changed(self, event: dict) -> None:
        """Aplicar un event state_changed al cache.

        Event payload shape:
            {
              "event_type": "state_changed",
              "data": {
                "entity_id": "light.escritorio",
                "new_state": {"state": "on", ...} | None,
                "old_state": {...} | None,
              }
            }
        """
        data = event.get("data", {})
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")

        if not entity_id:
            return

        if new_state is None:
            # Entity removed/unavailable
            self._state_cache.pop(entity_id, None)
        else:
            self._state_cache[entity_id] = new_state

        # Notificar observers sin dejar que un callback roto rompa el loop
        for cb in self._state_callbacks:
            try:
                cb(entity_id, new_state)
            except Exception as e:
                logger.debug(f"State callback error: {e}")

    def register_state_callback(self, callback) -> None:
        """Registrar un callback `(entity_id, new_state) -> None` para cambios.

        Useful para integraciones que quieren reaccionar proactivamente a
        cambios de state (ej: alerts, analytics).
        """
        self._state_callbacks.append(callback)

    async def close(self) -> None:
        """Cerrar conexiones (ambos canales WS + session HTTP)."""
        await self.stop_state_sync()
        for attr_name in ("_ws_events", "_ws_calls"):
            ws = getattr(self, attr_name, None)
            if ws is not None and not ws.closed:
                try:
                    await ws.close()
                except Exception as e:
                    logger.debug(f"Error cerrando {attr_name}: {e}")
        self._ws_events = None
        self._ws_calls = None
        self._ws_connected = False
        if self._session and not self._session.closed:
            await self._session.close()

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
                if response.status in (401, 403):
                    self._has_auth_error = True
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
