"""
Home Assistant Integration para KZA
Expone sensores y servicios de KZA en Home Assistant.

Sensores:
- sensor.kza_state: Estado del pipeline (listening, processing, speaking)
- sensor.kza_latency: Latencia promedio
- sensor.kza_echo_suppressor: Estado del supresor de eco
- sensor.kza_active_timers: Timers activos
- sensor.kza_presence: Usuarios en casa
- sensor.kza_alerts: Alertas activas
- binary_sensor.kza_listening: ¿Está escuchando?
- binary_sensor.kza_speaking: ¿Está hablando?

Servicios:
- kza.announce: Hacer anuncio
- kza.set_timer: Crear timer
- kza.cancel_timer: Cancelar timer
- kza.set_dnd: Activar/desactivar No Molestar
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


class KZAHomeAssistantIntegration:
    """
    Integración de KZA con Home Assistant.

    Publica estados de KZA como sensores en HA y
    expone servicios para control desde HA.
    """

    # Prefijo para entidades
    ENTITY_PREFIX = "kza"

    def __init__(self, ha_client, pipeline):
        """
        Args:
            ha_client: Cliente de Home Assistant
            pipeline: VoicePipeline de KZA
        """
        self.ha = ha_client
        self.pipeline = pipeline
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._update_interval = 5  # Segundos entre updates

        # Estado actual de sensores
        self._sensor_states = {}

    async def start(self):
        """Iniciar integración"""
        if self._running:
            return

        self._running = True

        # Registrar servicios en HA
        await self._register_services()

        # Crear sensores iniciales
        await self._create_sensors()

        # Iniciar loop de actualización
        self._update_task = asyncio.create_task(self._update_loop())

        logger.info("🏠 Integración con Home Assistant iniciada")

    async def stop(self):
        """Detener integración"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

    async def _register_services(self):
        """Registrar servicios de KZA en Home Assistant"""
        services = [
            {
                "domain": "kza",
                "service": "announce",
                "description": "Hacer un anuncio por voz",
                "fields": {
                    "message": {"description": "Mensaje a anunciar", "required": True},
                    "zones": {"description": "Zonas (vacío = todas)"},
                    "priority": {"description": "normal, high, emergency"}
                }
            },
            {
                "domain": "kza",
                "service": "set_timer",
                "description": "Crear un timer con nombre",
                "fields": {
                    "name": {"description": "Nombre del timer", "required": True},
                    "minutes": {"description": "Duración en minutos", "required": True},
                    "zone": {"description": "Zona para anuncio"}
                }
            },
            {
                "domain": "kza",
                "service": "cancel_timer",
                "description": "Cancelar un timer",
                "fields": {
                    "name": {"description": "Nombre del timer", "required": True}
                }
            },
            {
                "domain": "kza",
                "service": "set_dnd",
                "description": "Activar/desactivar No Molestar",
                "fields": {
                    "user_id": {"description": "ID del usuario"},
                    "enabled": {"description": "true/false", "required": True}
                }
            },
            {
                "domain": "kza",
                "service": "speak",
                "description": "Hacer que KZA diga algo",
                "fields": {
                    "message": {"description": "Texto a decir", "required": True},
                    "zone": {"description": "Zona específica"}
                }
            }
        ]

        # Nota: En una integración real de HA, esto se hace via el registro de servicios
        # Aquí simulamos guardando la config para que el handler de servicios la use
        self._registered_services = {s["service"]: s for s in services}
        logger.info(f"Servicios registrados: {list(self._registered_services.keys())}")

    async def handle_service_call(self, service: str, data: dict) -> dict:
        """
        Manejar llamada a servicio desde Home Assistant.

        Args:
            service: Nombre del servicio (announce, set_timer, etc.)
            data: Datos del servicio

        Returns:
            Resultado de la operación
        """
        try:
            if service == "announce":
                return await self._service_announce(data)
            elif service == "set_timer":
                return await self._service_set_timer(data)
            elif service == "cancel_timer":
                return await self._service_cancel_timer(data)
            elif service == "set_dnd":
                return await self._service_set_dnd(data)
            elif service == "speak":
                return await self._service_speak(data)
            else:
                return {"success": False, "error": f"Servicio desconocido: {service}"}
        except Exception as e:
            logger.error(f"Error en servicio {service}: {e}")
            return {"success": False, "error": str(e)}

    async def _service_announce(self, data: dict) -> dict:
        message = data.get("message", "")
        zones = data.get("zones", "").split(",") if data.get("zones") else None
        priority = data.get("priority", "normal")

        if zones:
            zones = [z.strip() for z in zones if z.strip()]

        return await self.pipeline.announce(message, zones, priority)

    async def _service_set_timer(self, data: dict) -> dict:
        name = data.get("name", "timer")
        minutes = int(data.get("minutes", 5))
        zone = data.get("zone", "default")

        timer = self.pipeline.create_timer(name, minutes * 60, zone_id=zone)
        return {"success": True, "timer_id": timer.timer_id}

    async def _service_cancel_timer(self, data: dict) -> dict:
        name = data.get("name", "")
        success = self.pipeline.cancel_timer(name)
        return {"success": success}

    async def _service_set_dnd(self, data: dict) -> dict:
        user_id = data.get("user_id", "default")
        enabled = str(data.get("enabled", "true")).lower() == "true"
        self.pipeline.set_do_not_disturb(user_id, enabled)
        return {"success": True, "dnd": enabled}

    async def _service_speak(self, data: dict) -> dict:
        message = data.get("message", "")
        zone = data.get("zone")
        self.pipeline.response_handler.speak(message, zone_id=zone)
        return {"success": True}

    async def _create_sensors(self):
        """Crear sensores iniciales en HA"""
        # Los sensores se crean/actualizan en el loop
        pass

    async def _update_loop(self):
        """Loop de actualización de sensores"""
        while self._running:
            try:
                await self._update_sensors()
                await asyncio.sleep(self._update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error actualizando sensores: {e}")
                await asyncio.sleep(self._update_interval)

    async def _update_sensors(self):
        """Actualizar todos los sensores en HA"""
        sensors = self._collect_sensor_data()

        for sensor_id, sensor_data in sensors.items():
            await self._update_sensor(sensor_id, sensor_data)

    def _collect_sensor_data(self) -> dict:
        """Recolectar datos de todos los sensores"""
        sensors = {}

        # Estado general del pipeline
        sensors["sensor.kza_state"] = {
            "state": self.pipeline.echo_suppressor.state.value,
            "attributes": {
                "friendly_name": "KZA Estado",
                "icon": "mdi:robot",
                "running": self.pipeline._running
            }
        }

        # Echo Suppressor
        echo_stats = self.pipeline.echo_suppressor.get_stats()
        sensors["sensor.kza_echo_suppressor"] = {
            "state": echo_stats.get("state", "unknown"),
            "attributes": {
                "friendly_name": "KZA Echo Suppressor",
                "icon": "mdi:microphone-off",
                "total_suppressed_ms": echo_stats.get("total_suppressed_ms", 0),
                "echo_detections": echo_stats.get("echo_detections", 0),
                "false_triggers_prevented": echo_stats.get("false_triggers_prevented", 0)
            }
        }

        # Binary sensor: hablando
        is_speaking = self.pipeline.echo_suppressor.state.value == "speaking"
        sensors["binary_sensor.kza_speaking"] = {
            "state": "on" if is_speaking else "off",
            "attributes": {
                "friendly_name": "KZA Hablando",
                "icon": "mdi:account-voice",
                "device_class": "sound"
            }
        }

        # Binary sensor: escuchando (no hablando y no en cooldown)
        is_listening = self.pipeline.echo_suppressor.is_safe_to_listen
        sensors["binary_sensor.kza_listening"] = {
            "state": "on" if is_listening else "off",
            "attributes": {
                "friendly_name": "KZA Escuchando",
                "icon": "mdi:ear-hearing",
                "device_class": "sound"
            }
        }

        # Timers activos
        active_timers = self.pipeline.get_active_timers()
        timer_list = [
            {
                "name": t.name,
                "remaining": t.format_remaining(),
                "remaining_seconds": int(t.remaining_seconds)
            }
            for t in active_timers
        ]
        sensors["sensor.kza_active_timers"] = {
            "state": len(active_timers),
            "attributes": {
                "friendly_name": "KZA Timers Activos",
                "icon": "mdi:timer-outline",
                "timers": timer_list,
                "unit_of_measurement": "timers"
            }
        }

        # Presencia (si está disponible)
        if self.pipeline.presence:
            who_home = self.pipeline.who_is_home()
            sensors["sensor.kza_presence"] = {
                "state": len(who_home),
                "attributes": {
                    "friendly_name": "KZA Presencia",
                    "icon": "mdi:home-account",
                    "users": who_home,
                    "anyone_home": len(who_home) > 0,
                    "unit_of_measurement": "personas"
                }
            }

        # Alertas activas
        active_alerts = self.pipeline.get_active_alerts()
        sensors["sensor.kza_alerts"] = {
            "state": len(active_alerts),
            "attributes": {
                "friendly_name": "KZA Alertas",
                "icon": "mdi:alert-circle",
                "alerts": [
                    {"message": a.message if hasattr(a, 'message') else str(a)}
                    for a in active_alerts[:5]  # Máximo 5 para no sobrecargar
                ],
                "unit_of_measurement": "alertas"
            }
        }

        # Notificaciones - DND status
        sensors["binary_sensor.kza_dnd"] = {
            "state": "on" if self.pipeline.is_do_not_disturb("default") else "off",
            "attributes": {
                "friendly_name": "KZA No Molestar",
                "icon": "mdi:bell-off"
            }
        }

        # Follow-up mode
        sensors["binary_sensor.kza_conversation_active"] = {
            "state": "on" if self.pipeline.follow_up.is_active else "off",
            "attributes": {
                "friendly_name": "KZA Conversación Activa",
                "icon": "mdi:chat",
                "state": self.pipeline.follow_up.state.value if hasattr(self.pipeline.follow_up, 'state') else "unknown"
            }
        }

        # Intercom zonas
        intercom_status = self.pipeline.intercom.get_status()
        sensors["sensor.kza_intercom"] = {
            "state": intercom_status.get("zones_count", 0),
            "attributes": {
                "friendly_name": "KZA Intercom",
                "icon": "mdi:bullhorn",
                "zones": intercom_status.get("zones", []),
                "queue_size": intercom_status.get("queue_size", 0),
                "unit_of_measurement": "zonas"
            }
        }

        # Estadísticas generales
        try:
            features_status = self.pipeline.get_new_features_status()
            sensors["sensor.kza_features"] = {
                "state": "active",
                "attributes": {
                    "friendly_name": "KZA Features",
                    "icon": "mdi:feature-search",
                    "follow_up": features_status.get("follow_up_mode", {}).get("active", False),
                    "ambient_detection": features_status.get("ambient_detection", {}).get("enabled", False),
                    "pattern_learning": features_status.get("pattern_learning", {}).get("enabled", False),
                    "briefings": features_status.get("briefings", {}).get("enabled", False)
                }
            }
        except Exception as e:
            logger.debug(f"Failed to get features status for HA sensor: {e}")

        return sensors

    async def _update_sensor(self, entity_id: str, data: dict):
        """Actualizar un sensor en Home Assistant"""
        state = data.get("state", "unknown")
        attributes = data.get("attributes", {})

        # Guardar estado local
        self._sensor_states[entity_id] = {
            "state": state,
            "attributes": attributes,
            "last_updated": datetime.now().isoformat()
        }

        # Actualizar en HA vía API
        try:
            await self.ha.set_state(
                entity_id=entity_id,
                state=state,
                attributes=attributes
            )
        except Exception as e:
            # set_state puede no estar implementado, usar alternativa
            logger.debug(f"No se pudo actualizar {entity_id}: {e}")

    def get_sensor_state(self, entity_id: str) -> dict:
        """Obtener estado de un sensor"""
        return self._sensor_states.get(entity_id, {})

    def get_all_sensors(self) -> dict:
        """Obtener todos los estados de sensores"""
        return self._sensor_states.copy()


class KZALovelaceCards:
    """
    Generador de tarjetas Lovelace para el dashboard de KZA.
    """

    @staticmethod
    def generate_dashboard_yaml() -> str:
        """Generar YAML para dashboard de KZA en Lovelace"""
        return """
# KZA Dashboard - Agregar a configuration.yaml o ui-lovelace.yaml
# Copiar este contenido en tu dashboard de Home Assistant

title: KZA Voice Assistant
views:
  - title: KZA
    icon: mdi:robot
    cards:
      # Estado Principal
      - type: entities
        title: Estado de KZA
        show_header_toggle: false
        entities:
          - entity: sensor.kza_state
            name: Estado
          - entity: binary_sensor.kza_listening
            name: Escuchando
          - entity: binary_sensor.kza_speaking
            name: Hablando
          - entity: binary_sensor.kza_conversation_active
            name: Conversación Activa
          - entity: binary_sensor.kza_dnd
            name: No Molestar

      # Timers
      - type: entities
        title: Timers Activos
        show_header_toggle: false
        entities:
          - entity: sensor.kza_active_timers
            name: Cantidad
        footer:
          type: buttons
          entities:
            - entity: script.kza_cancel_all_timers
              name: Cancelar Todos
              icon: mdi:timer-off

      # Tarjeta de Timers detallada (custom)
      - type: markdown
        title: Detalle Timers
        content: |
          {% set timers = state_attr('sensor.kza_active_timers', 'timers') %}
          {% if timers and timers | length > 0 %}
          | Timer | Restante |
          |-------|----------|
          {% for t in timers %}
          | {{ t.name }} | {{ t.remaining }} |
          {% endfor %}
          {% else %}
          No hay timers activos
          {% endif %}

      # Echo Suppressor
      - type: entities
        title: Echo Suppressor
        entities:
          - entity: sensor.kza_echo_suppressor
            name: Estado
        footer:
          type: graph
          entity: sensor.kza_echo_suppressor
          hours_to_show: 1

      # Presencia
      - type: entities
        title: Presencia
        entities:
          - entity: sensor.kza_presence
            name: Personas en Casa
        footer:
          type: markdown
          content: |
            {% set users = state_attr('sensor.kza_presence', 'users') %}
            {% if users %}{{ users | join(', ') }}{% else %}Nadie{% endif %}

      # Alertas
      - type: conditional
        conditions:
          - entity: sensor.kza_alerts
            state_not: "0"
        card:
          type: entities
          title: ⚠️ Alertas Activas
          entities:
            - entity: sensor.kza_alerts
              name: Cantidad

      # Intercom
      - type: entities
        title: Intercom
        entities:
          - entity: sensor.kza_intercom
            name: Zonas

      # Acciones Rápidas
      - type: button
        name: Anunciar
        icon: mdi:bullhorn
        tap_action:
          action: call-service
          service: kza.announce
          service_data:
            message: "Prueba de anuncio"

      - type: horizontal-stack
        cards:
          - type: button
            name: Timer 5min
            icon: mdi:timer
            tap_action:
              action: call-service
              service: kza.set_timer
              service_data:
                name: "rapido"
                minutes: 5

          - type: button
            name: Timer 10min
            icon: mdi:timer
            tap_action:
              action: call-service
              service: kza.set_timer
              service_data:
                name: "medio"
                minutes: 10

      # Features Status
      - type: glance
        title: Features
        entities:
          - entity: sensor.kza_features
            name: Estado
"""

    @staticmethod
    def generate_scripts_yaml() -> str:
        """Generar scripts de HA para KZA"""
        return """
# Scripts para KZA - Agregar a scripts.yaml

kza_cancel_all_timers:
  alias: "KZA: Cancelar Todos los Timers"
  sequence:
    - service: kza.cancel_timer
      data:
        name: "todos"

kza_morning_briefing:
  alias: "KZA: Briefing Matutino"
  sequence:
    - service: kza.speak
      data:
        message: "Buenos días. Dame un momento para preparar tu briefing."

kza_emergency_announce:
  alias: "KZA: Anuncio de Emergencia"
  sequence:
    - service: kza.announce
      data:
        message: "{{ message }}"
        priority: "emergency"
  fields:
    message:
      description: "Mensaje de emergencia"
      required: true
"""

    @staticmethod
    def generate_automations_yaml() -> str:
        """Generar automatizaciones de ejemplo"""
        return """
# Automatizaciones de ejemplo para KZA - Agregar a automations.yaml

# Anunciar cuando alguien llega a casa
- alias: "KZA: Anunciar llegada"
  trigger:
    - platform: state
      entity_id: sensor.kza_presence
  condition:
    - condition: template
      value_template: "{{ trigger.to_state.state | int > trigger.from_state.state | int }}"
  action:
    - service: kza.announce
      data:
        message: "Alguien acaba de llegar a casa"

# Activar DND en la noche
- alias: "KZA: DND nocturno"
  trigger:
    - platform: time
      at: "23:00:00"
  action:
    - service: kza.set_dnd
      data:
        enabled: true

# Desactivar DND en la mañana
- alias: "KZA: Desactivar DND"
  trigger:
    - platform: time
      at: "07:00:00"
  action:
    - service: kza.set_dnd
      data:
        enabled: false
"""
