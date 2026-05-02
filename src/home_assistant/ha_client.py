"""
Home Assistant Client
Comunicacion con Home Assistant via REST API y WebSocket.

Graceful degradation: all methods return safe defaults when HA is
unavailable so the voice server keeps running.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

import aiohttp

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

    def __init__(self, url: str, token: str, timeout: float = REST_TIMEOUT_DEFAULT):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
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

    def get_entity_state_cached(self, entity_id: str) -> dict | None:
        """Lookup sin I/O contra el prefetch cache.

        Returns:
            Dict con el state más reciente conocido via WS push o REST snapshot,
            o None si no hay entry en cache (típicamente porque el sync aún no
            corrió o la entidad no existe).
        """
        return self._state_cache.get(entity_id)

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
            purpose: "calls" or "events". Used only for log messages.

        Returns:
            The authenticated aiohttp.ClientWebSocketResponse, or None on
            failure. The caller decides which attribute to store the result on
            (self._ws_calls or self._ws_events).
        """
        ws_url = self.url.replace("http", "ws") + "/api/websocket"

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
                await ws.close()
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
                return ws

            self._has_auth_error = True
            logger.error(f"Auth WebSocket[{purpose}] fallo: {auth_result}")
            await ws.close()
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

    async def connect_websocket(self) -> bool:
        """Conectar el canal de calls via WebSocket para menor latencia.

        Incluye auto-reconexion en caso de fallo. Usa _open_ws_authenticated
        para el handshake. Solo afecta al canal self._ws_calls.
        """
        if self._ws_connected and self._ws_calls and not self._ws_calls.closed:
            return True

        ws = await self._open_ws_authenticated("calls")
        if ws is None:
            self._ws_connected = False
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
        """
        # Asegurar conexion WebSocket
        if not await self.ensure_websocket_connected():
            logger.debug("WebSocket no disponible, usando REST")
            return await self.call_service(domain, service, entity_id, data)

        t_start = time.perf_counter()
        try:
            # ID unico para este mensaje — counter propio del canal de calls.
            self._ws_msg_id_calls += 1
            msg_id = self._ws_msg_id_calls

            service_data = {"entity_id": entity_id}
            if data:
                service_data.update(data)

            await self._ws_calls.send_json({
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "service_data": service_data
            })

            # Esperar respuesta (con timeout corto para baja latencia)
            response = await asyncio.wait_for(
                self._ws_calls.receive_json(),
                timeout=WS_CALL_TIMEOUT
            )

            elapsed = (time.perf_counter() - t_start) * 1000
            success = response.get("success", False)
            if success:
                self._record_success(elapsed)
                logger.debug(f"WS: {domain}.{service} -> {entity_id} ({elapsed:.0f}ms)")
            return success

        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.warning(
                f"WebSocket timeout for {domain}.{service} on {entity_id} "
                f"after {elapsed:.0f}ms (limit {WS_CALL_TIMEOUT}s), fallback to REST"
            )
            self._record_error(asyncio.TimeoutError(), f"call_service_ws({domain}.{service})")
            return await self.call_service(domain, service, entity_id, data)

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False  # Marcar para reconexion
            self._record_error(e, f"call_service_ws({domain}.{service})")
            return await self.call_service(domain, service, entity_id, data)

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
