"""
Home Assistant Client
Comunicacion con Home Assistant via REST API y WebSocket
"""

import asyncio
import logging
import aiohttp

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Cliente para comunicacion con Home Assistant via REST API y WebSocket."""

    def __init__(self, url: str, token: str, timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._ws_connection = None
        self._ws_connected = False
        self._ws_msg_id = 1  # Contador de mensajes WebSocket
        self._ws_reconnect_attempts = 0
        self._ws_max_reconnect_attempts = 3

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

    # ==================== REST API ====================

    async def get_all_entities(self) -> list[dict]:
        """Obtener todas las entidades de Home Assistant."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/states",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo entidades: {e}")
            return []

    async def get_all_services(self) -> list[dict]:
        """Obtener todos los servicios disponibles."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/services",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo servicios: {e}")
            return []

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """Obtener estado de una entidad especifica."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/states/{entity_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error obteniendo estado de {entity_id}: {e}")
            return None

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

        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.url}/api/services/{domain}/{service}",
                json=payload
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Ejecutado: {domain}.{service} en {entity_id}")
                else:
                    logger.warning(f"Error {response.status}: {domain}.{service}")
                return success
        except Exception as e:
            logger.error(f"Error llamando servicio: {e}")
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
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.url}/api/config/automation/config/{automation_id}",
                json=config,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 201]:
                    # Recargar automatizaciones
                    await self._reload_automations()
                    logger.info(f"Automatizacion creada: {automation_id}")
                    return True, "OK"
                else:
                    error = await response.text()
                    logger.error(f"Error creando automatizacion: {error}")
                    return False, error
        except Exception as e:
            logger.error(f"Error creando automatizacion: {e}")
            return False, str(e)

    async def delete_automation(self, automation_id: str) -> bool:
        """Eliminar una automatizacion."""
        try:
            session = await self._ensure_session()
            async with session.delete(
                f"{self.url}/api/config/automation/config/{automation_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 204]:
                    await self._reload_automations()
                    logger.info(f"Automatizacion eliminada: {automation_id}")
                    return True
                logger.warning(f"Delete automation {automation_id} returned status {response.status}")
                return False
        except Exception as e:
            logger.error(f"Error eliminando automatizacion: {e}")
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
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status not in (200, 204):
                    logger.warning(f"Automation reload returned status {response.status}")
        except Exception as e:
            logger.error(f"Error reloading automations: {e}")

    # ==================== WebSocket (menor latencia) ====================

    async def connect_websocket(self) -> bool:
        """Conectar via WebSocket para menor latencia (~10-20ms vs ~50-100ms REST).

        Incluye auto-reconexion en caso de fallo.
        """
        if self._ws_connected:
            return True

        ws_url = self.url.replace("http", "ws") + "/api/websocket"

        try:
            session = await self._ensure_session()

            self._ws_connection = await session.ws_connect(
                ws_url,
                heartbeat=30.0  # Ping cada 30s para mantener conexion
            )

            # Autenticar
            msg = await asyncio.wait_for(
                self._ws_connection.receive_json(),
                timeout=5.0
            )

            if msg.get("type") == "auth_required":
                await self._ws_connection.send_json({
                    "type": "auth",
                    "access_token": self.token
                })

                auth_result = await asyncio.wait_for(
                    self._ws_connection.receive_json(),
                    timeout=5.0
                )

                if auth_result.get("type") == "auth_ok":
                    self._ws_connected = True
                    self._ws_reconnect_attempts = 0
                    logger.info("WebSocket HA conectado y autenticado")
                    return True
                else:
                    logger.error(f"Auth WebSocket fallo: {auth_result}")

            return False

        except Exception as e:
            logger.error(f"Error conectando WebSocket HA: {e}")
            self._ws_connected = False
            return False

    async def ensure_websocket_connected(self) -> bool:
        """Asegurar que WebSocket esta conectado, reconectar si es necesario."""
        if self._ws_connected and self._ws_connection:
            # Verificar que la conexion sigue viva
            if self._ws_connection.closed:
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
        """
        # Asegurar conexion WebSocket
        if not await self.ensure_websocket_connected():
            logger.debug("WebSocket no disponible, usando REST")
            return await self.call_service(domain, service, entity_id, data)

        try:
            # ID unico para este mensaje
            self._ws_msg_id += 1
            msg_id = self._ws_msg_id

            service_data = {"entity_id": entity_id}
            if data:
                service_data.update(data)

            await self._ws_connection.send_json({
                "id": msg_id,
                "type": "call_service",
                "domain": domain,
                "service": service,
                "service_data": service_data
            })

            # Esperar respuesta (con timeout corto para baja latencia)
            response = await asyncio.wait_for(
                self._ws_connection.receive_json(),
                timeout=self.timeout
            )

            success = response.get("success", False)
            if success:
                logger.debug(f"WS: {domain}.{service} -> {entity_id}")
            return success

        except asyncio.TimeoutError:
            logger.warning("Timeout WebSocket, fallback a REST")
            return await self.call_service(domain, service, entity_id, data)

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False  # Marcar para reconexion
            return await self.call_service(domain, service, entity_id, data)

    async def close(self) -> None:
        """Cerrar conexiones."""
        if self._ws_connection:
            await self._ws_connection.close()
        if self._session and not self._session.closed:
            await self._session.close()

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
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.url}/api/",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error testing HA connection: {e}")
            return False
