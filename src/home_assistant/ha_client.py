"""
Home Assistant Client
Comunicación con Home Assistant via REST API y WebSocket
"""

import asyncio
import json
import logging
from typing import Optional
import aiohttp
import requests

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Cliente para comunicación con Home Assistant"""

    def __init__(self, url: str, token: str, timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self._ws_connection = None
        self._ws_session = None
        self._ws_connected = False
        self._ws_msg_id = 1  # Contador de mensajes WebSocket
        self._ws_reconnect_attempts = 0
        self._ws_max_reconnect_attempts = 3
    
    # ==================== REST API ====================
    
    def get_all_entities(self) -> list[dict]:
        """Obtener todas las entidades de Home Assistant"""
        try:
            response = requests.get(
                f"{self.url}/api/states",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo entidades: {e}")
            return []
    
    def get_all_services(self) -> list[dict]:
        """Obtener todos los servicios disponibles"""
        try:
            response = requests.get(
                f"{self.url}/api/services",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo servicios: {e}")
            return []
    
    def get_entity_state(self, entity_id: str) -> Optional[dict]:
        """Obtener estado de una entidad específica"""
        try:
            response = requests.get(
                f"{self.url}/api/states/{entity_id}",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo estado de {entity_id}: {e}")
            return None
    
    def call_service(
        self, 
        domain: str, 
        service: str, 
        entity_id: str,
        data: Optional[dict] = None
    ) -> bool:
        """
        Ejecutar un servicio de Home Assistant
        
        Args:
            domain: Dominio (light, climate, cover, etc.)
            service: Servicio (turn_on, turn_off, set_temperature, etc.)
            entity_id: ID de la entidad
            data: Datos adicionales (temperature, brightness, etc.)
        
        Returns:
            True si la llamada fue exitosa
        """
        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)
        
        try:
            response = requests.post(
                f"{self.url}/api/services/{domain}/{service}",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            success = response.status_code == 200
            if success:
                logger.info(f"Ejecutado: {domain}.{service} en {entity_id}")
            else:
                logger.warning(f"Error {response.status_code}: {domain}.{service}")
            return success
        except Exception as e:
            logger.error(f"Error llamando servicio: {e}")
            return False
    
    # ==================== Automatizaciones ====================
    
    def create_automation(self, automation_id: str, config: dict) -> tuple[bool, str]:
        """
        Crear una nueva automatización
        
        Args:
            automation_id: ID único para la automatización
            config: Configuración de la automatización en formato HA
        
        Returns:
            (success, error_message)
        """
        try:
            response = requests.post(
                f"{self.url}/api/config/automation/config/{automation_id}",
                headers=self.headers,
                json=config,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                # Recargar automatizaciones
                self._reload_automations()
                logger.info(f"Automatización creada: {automation_id}")
                return True, "OK"
            else:
                error = response.text
                logger.error(f"Error creando automatización: {error}")
                return False, error
        except Exception as e:
            logger.error(f"Error creando automatización: {e}")
            return False, str(e)
    
    def delete_automation(self, automation_id: str) -> bool:
        """Eliminar una automatización"""
        try:
            response = requests.delete(
                f"{self.url}/api/config/automation/config/{automation_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                self._reload_automations()
                logger.info(f"Automatización eliminada: {automation_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando automatización: {e}")
            return False
    
    def get_automations(self) -> list[dict]:
        """Obtener lista de automatizaciones"""
        entities = self.get_all_entities()
        return [e for e in entities if e["entity_id"].startswith("automation.")]
    
    def _reload_automations(self):
        """Recargar automatizaciones en Home Assistant"""
        try:
            requests.post(
                f"{self.url}/api/services/automation/reload",
                headers=self.headers,
                timeout=5
            )
        except:
            pass
    
    # ==================== WebSocket (menor latencia) ====================

    async def connect_websocket(self) -> bool:
        """
        Conectar via WebSocket para menor latencia (~10-20ms vs ~50-100ms REST).
        Incluye auto-reconexión en caso de fallo.
        """
        if self._ws_connected:
            return True

        ws_url = self.url.replace("http", "ws") + "/api/websocket"

        try:
            if self._ws_session is None:
                self._ws_session = aiohttp.ClientSession()

            self._ws_connection = await self._ws_session.ws_connect(
                ws_url,
                heartbeat=30.0  # Ping cada 30s para mantener conexión
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
                    logger.error(f"Auth WebSocket falló: {auth_result}")

            return False

        except Exception as e:
            logger.error(f"Error conectando WebSocket HA: {e}")
            self._ws_connected = False
            return False

    async def ensure_websocket_connected(self) -> bool:
        """Asegurar que WebSocket está conectado, reconectar si es necesario."""
        if self._ws_connected and self._ws_connection:
            # Verificar que la conexión sigue viva
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
        data: Optional[dict] = None
    ) -> bool:
        """
        Llamar servicio via WebSocket (más rápido que REST).
        Latencia típica: 10-20ms vs 50-100ms de REST.
        """
        # Asegurar conexión WebSocket
        if not await self.ensure_websocket_connected():
            logger.debug("WebSocket no disponible, usando REST")
            return self.call_service(domain, service, entity_id, data)

        try:
            # ID único para este mensaje
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
            return self.call_service(domain, service, entity_id, data)

        except Exception as e:
            logger.error(f"Error WebSocket: {e}")
            self._ws_connected = False  # Marcar para reconexión
            return self.call_service(domain, service, entity_id, data)
    
    async def close(self):
        """Cerrar conexiones"""
        if self._ws_connection:
            await self._ws_connection.close()
        if hasattr(self, '_ws_session'):
            await self._ws_session.close()
    
    # ==================== Utilidades ====================
    
    def get_domotics_entities(self) -> list[dict]:
        """Obtener solo entidades relevantes para domótica"""
        domotics_domains = [
            "light", "switch", "cover", "climate", "fan",
            "media_player", "vacuum", "lock", "scene",
            "script", "automation", "input_boolean"
        ]
        
        entities = self.get_all_entities()
        return [
            e for e in entities
            if e["entity_id"].split(".")[0] in domotics_domains
        ]
    
    def get_services_by_domain(self) -> dict[str, list[str]]:
        """Obtener servicios organizados por dominio"""
        services = self.get_all_services()
        return {
            s["domain"]: list(s["services"].keys())
            for s in services
        }
    
    def test_connection(self) -> bool:
        """Verificar conexión con Home Assistant"""
        try:
            response = requests.get(
                f"{self.url}/api/",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
