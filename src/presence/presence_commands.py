"""
Presence Voice Commands
Comandos de voz para gestión de presencia y dispositivos BLE.
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PresenceIntent(Enum):
    """Intents de comandos de presencia"""
    # Consultas
    WHO_IS_HOME = "who_is_home"
    WHO_IS_IN_ZONE = "who_is_in_zone"
    IS_USER_HOME = "is_user_home"
    WHERE_IS_USER = "where_is_user"
    ZONE_STATUS = "zone_status"

    # Registro de dispositivos
    REGISTER_MY_PHONE = "register_my_phone"
    REGISTER_DEVICE = "register_device"
    FORGET_DEVICE = "forget_device"
    LIST_DEVICES = "list_devices"

    # Escaneo
    SCAN_DEVICES = "scan_devices"
    LIST_NEARBY = "list_nearby"

    UNKNOWN = "unknown"


@dataclass
class PresenceCommand:
    """Comando de presencia parseado"""
    intent: PresenceIntent
    user_name: Optional[str] = None
    zone_name: Optional[str] = None
    device_address: Optional[str] = None
    device_name: Optional[str] = None


# Patrones de reconocimiento
PATTERNS = {
    # Consultas de presencia
    PresenceIntent.WHO_IS_HOME: [
        r"(?:quién|quien|quienes)\s+(?:está|estan|hay)\s+(?:en\s+)?casa",
        r"(?:hay\s+)?alguien\s+(?:en\s+)?casa",
        r"(?:who(?:'s| is)?\s+)?(?:home|at home)",
        r"cuántas?\s+personas?\s+(?:hay\s+)?(?:en\s+)?casa",
    ],
    PresenceIntent.WHO_IS_IN_ZONE: [
        r"(?:quién|quien)\s+(?:está|hay)\s+(?:en\s+)?(?:el\s+|la\s+)?(.+)",
        r"hay\s+alguien\s+(?:en\s+)?(?:el\s+|la\s+)?(.+)",
        r"who(?:'s| is)\s+in\s+(?:the\s+)?(.+)",
    ],
    PresenceIntent.IS_USER_HOME: [
        r"(?:está|esta)\s+(.+?)\s+(?:en\s+)?casa",
        r"(.+?)\s+(?:está|esta)\s+(?:en\s+)?casa",
        r"is\s+(.+?)\s+(?:home|at home)",
    ],
    PresenceIntent.WHERE_IS_USER: [
        r"(?:dónde|donde)\s+(?:está|esta)\s+(.+)",
        r"(?:en\s+)?(?:qué|que)\s+(?:zona|habitación|cuarto)\s+(?:está|esta)\s+(.+)",
        r"where(?:'s| is)\s+(.+)",
    ],
    PresenceIntent.ZONE_STATUS: [
        r"(?:estado|status)\s+(?:de\s+)?(?:la\s+|el\s+)?zona\s+(.+)",
        r"(?:cómo|como)\s+(?:está|esta)\s+(?:la\s+|el\s+)?(.+)",
    ],

    # Registro de dispositivos
    PresenceIntent.REGISTER_MY_PHONE: [
        r"(?:registra|asocia|vincula)\s+(?:mi\s+)?(?:teléfono|celular|móvil|phone)",
        r"(?:conecta|enlaza)\s+(?:mi\s+)?(?:teléfono|celular|móvil|phone)",
        r"(?:register|link)\s+(?:my\s+)?phone",
    ],
    PresenceIntent.REGISTER_DEVICE: [
        r"(?:registra|asocia)\s+(?:el\s+)?dispositivo\s+(.+?)\s+(?:como|a)\s+(.+)",
        r"(?:vincula|conecta)\s+(.+?)\s+(?:con|a)\s+(?:el\s+usuario\s+)?(.+)",
    ],
    PresenceIntent.FORGET_DEVICE: [
        r"(?:olvida|elimina|borra)\s+(?:el\s+)?dispositivo\s+(.+)",
        r"(?:desvincula|desconecta)\s+(.+)",
        r"(?:forget|remove)\s+device\s+(.+)",
    ],
    PresenceIntent.LIST_DEVICES: [
        r"(?:lista|muestra|ver)\s+(?:los\s+)?dispositivos\s+(?:registrados)?",
        r"(?:qué|que|cuales)\s+dispositivos\s+(?:hay|tengo)\s+(?:registrados)?",
        r"(?:list|show)\s+(?:registered\s+)?devices",
    ],

    # Escaneo
    PresenceIntent.SCAN_DEVICES: [
        r"(?:escanea|busca)\s+(?:los\s+)?dispositivos\s+(?:bluetooth|ble|cercanos)?",
        r"(?:scan|search)\s+(?:for\s+)?(?:ble\s+)?devices",
    ],
    PresenceIntent.LIST_NEARBY: [
        r"(?:qué|que)\s+(?:dispositivos|aparatos)\s+(?:hay\s+)?cerca",
        r"(?:dispositivos|aparatos)\s+cercanos",
        r"(?:what\s+)?devices?\s+(?:are\s+)?nearby",
    ],
}


def detect_presence_intent(text: str) -> PresenceCommand:
    """
    Detectar intent de comando de presencia.

    Args:
        text: Texto del comando

    Returns:
        PresenceCommand con intent y parámetros extraídos
    """
    text_lower = text.lower().strip()

    for intent, patterns in PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                command = PresenceCommand(intent=intent)

                # Extraer parámetros según el intent
                if intent == PresenceIntent.WHO_IS_IN_ZONE and match.groups():
                    command.zone_name = match.group(1).strip()

                elif intent == PresenceIntent.IS_USER_HOME and match.groups():
                    command.user_name = match.group(1).strip()

                elif intent == PresenceIntent.WHERE_IS_USER and match.groups():
                    command.user_name = match.group(1).strip()

                elif intent == PresenceIntent.ZONE_STATUS and match.groups():
                    command.zone_name = match.group(1).strip()

                elif intent == PresenceIntent.REGISTER_DEVICE and len(match.groups()) >= 2:
                    command.device_name = match.group(1).strip()
                    command.user_name = match.group(2).strip()

                elif intent == PresenceIntent.FORGET_DEVICE and match.groups():
                    command.device_name = match.group(1).strip()

                return command

    return PresenceCommand(intent=PresenceIntent.UNKNOWN)


class PresenceCommandHandler:
    """
    Manejador de comandos de presencia.

    Procesa comandos de voz relacionados con presencia y genera respuestas.
    """

    def __init__(self, presence_detector, user_manager=None):
        """
        Args:
            presence_detector: PresenceDetector instance
            user_manager: UserManager para resolver nombres de usuario
        """
        self.detector = presence_detector
        self.user_manager = user_manager

    async def handle(self, text: str, speaker_user=None) -> str:
        """
        Procesar comando de presencia y generar respuesta.

        Args:
            text: Texto del comando
            speaker_user: Usuario que habla (para comandos como "mi teléfono")

        Returns:
            Respuesta en texto natural
        """
        command = detect_presence_intent(text)

        handlers = {
            PresenceIntent.WHO_IS_HOME: self._handle_who_is_home,
            PresenceIntent.WHO_IS_IN_ZONE: self._handle_who_is_in_zone,
            PresenceIntent.IS_USER_HOME: self._handle_is_user_home,
            PresenceIntent.WHERE_IS_USER: self._handle_where_is_user,
            PresenceIntent.ZONE_STATUS: self._handle_zone_status,
            PresenceIntent.REGISTER_MY_PHONE: self._handle_register_my_phone,
            PresenceIntent.LIST_DEVICES: self._handle_list_devices,
            PresenceIntent.LIST_NEARBY: self._handle_list_nearby,
            PresenceIntent.SCAN_DEVICES: self._handle_scan_devices,
        }

        handler = handlers.get(command.intent)
        if handler:
            return await handler(command, speaker_user)

        return None  # No es un comando de presencia

    async def _handle_who_is_home(self, command: PresenceCommand, speaker_user) -> str:
        """¿Quién está en casa?"""
        summary = self.detector.get_summary()
        total = summary["total_people"]
        users = summary["users_home"]

        if total == 0:
            return "No hay nadie en casa."

        if users:
            user_names = self._get_user_names(users)
            if len(user_names) == 1:
                return f"Solo está {user_names[0]} en casa."
            return f"Están en casa: {', '.join(user_names[:-1])} y {user_names[-1]}."

        if total == 1:
            return "Hay una persona en casa, pero no la reconozco."

        return f"Hay {total} personas en casa, pero no las reconozco."

    async def _handle_who_is_in_zone(self, command: PresenceCommand, speaker_user) -> str:
        """¿Quién está en una zona?"""
        zone_name = command.zone_name
        zone_id = self._resolve_zone(zone_name)

        if not zone_id:
            return f"No conozco la zona '{zone_name}'."

        occupancy = self.detector.get_zone_occupancy(zone_id)
        if not occupancy:
            return f"No tengo información sobre {zone_name}."

        if not occupancy.is_occupied:
            return f"No hay nadie en {occupancy.zone_name}."

        users = self._get_user_names(occupancy.known_users)
        unknown = occupancy.unknown_devices

        parts = []
        if users:
            if len(users) == 1:
                parts.append(f"está {users[0]}")
            else:
                parts.append(f"están {', '.join(users[:-1])} y {users[-1]}")

        if unknown:
            if unknown == 1:
                parts.append("hay una persona que no reconozco")
            else:
                parts.append(f"hay {unknown} personas que no reconozco")

        return f"En {occupancy.zone_name} {' y '.join(parts)}."

    async def _handle_is_user_home(self, command: PresenceCommand, speaker_user) -> str:
        """¿Está X en casa?"""
        user_name = command.user_name
        user_id = self._resolve_user(user_name)

        if not user_id:
            return f"No conozco a ningún usuario llamado '{user_name}'."

        presence = self.detector.get_user_presence(user_id)
        if not presence:
            return f"No tengo información de presencia de {user_name}."

        real_name = self._get_user_name(user_id)

        if presence.is_home:
            zone = presence.current_zone
            if zone:
                zone_name = self._get_zone_name(zone)
                return f"Sí, {real_name} está en casa, en {zone_name}."
            return f"Sí, {real_name} está en casa."

        if presence.state == presence.state.JUST_LEFT:
            return f"{real_name} se acaba de ir hace poco."

        return f"No, {real_name} no está en casa."

    async def _handle_where_is_user(self, command: PresenceCommand, speaker_user) -> str:
        """¿Dónde está X?"""
        user_name = command.user_name
        user_id = self._resolve_user(user_name)

        if not user_id:
            return f"No conozco a ningún usuario llamado '{user_name}'."

        presence = self.detector.get_user_presence(user_id)
        if not presence:
            return f"No tengo información de {user_name}."

        real_name = self._get_user_name(user_id)

        if not presence.is_home:
            return f"{real_name} no está en casa."

        zone = presence.current_zone
        if zone:
            zone_name = self._get_zone_name(zone)
            return f"{real_name} está en {zone_name}."

        return f"{real_name} está en casa, pero no sé en qué zona."

    async def _handle_zone_status(self, command: PresenceCommand, speaker_user) -> str:
        """Estado de una zona"""
        zone_name = command.zone_name
        zone_id = self._resolve_zone(zone_name)

        if not zone_id:
            return f"No conozco la zona '{zone_name}'."

        occupancy = self.detector.get_zone_occupancy(zone_id)
        if not occupancy:
            return f"No tengo información sobre {zone_name}."

        parts = [f"{occupancy.zone_name}:"]

        if occupancy.is_occupied:
            parts.append(f"{occupancy.estimated_people} persona(s)")
            if occupancy.known_users:
                names = self._get_user_names(occupancy.known_users)
                parts.append(f"({', '.join(names)})")
        else:
            parts.append("vacío")

        if occupancy.minutes_since_motion < 5:
            parts.append(", movimiento reciente")

        return " ".join(parts)

    async def _handle_register_my_phone(self, command: PresenceCommand, speaker_user) -> str:
        """Registrar teléfono del usuario que habla"""
        if not speaker_user:
            return "No sé quién eres. ¿Podrías identificarte primero?"

        # TODO: Escanear dispositivos cercanos y encontrar el más probable
        # Por ahora, instrucciones manuales
        return (
            f"Para registrar tu teléfono, {speaker_user.name}, "
            "necesito que hagas visibles tus dispositivos Bluetooth. "
            "Di 'escanear dispositivos' y luego 'vincular dispositivo [nombre] conmigo'."
        )

    async def _handle_list_devices(self, command: PresenceCommand, speaker_user) -> str:
        """Listar dispositivos registrados"""
        devices = []
        for scanner in self.detector._scanners.values():
            for addr, info in scanner.get_registered_devices().items():
                user_name = self._get_user_name(info.get("user_id"))
                devices.append(f"{info.get('friendly_name', addr)} → {user_name}")

        if not devices:
            return "No hay dispositivos registrados."

        return f"Dispositivos registrados: {', '.join(devices)}"

    async def _handle_list_nearby(self, command: PresenceCommand, speaker_user) -> str:
        """Listar dispositivos cercanos"""
        devices = []
        for zone_id, scanner in self.detector._scanners.items():
            for device in scanner.get_active_devices():
                if device.is_nearby:
                    name = device.friendly_name or device.name or device.address[-8:]
                    dist = f"{device.estimated_distance_m:.1f}m" if device.estimated_distance_m else "?"
                    devices.append(f"{name} ({dist})")

        if not devices:
            return "No detecto dispositivos cercanos."

        return f"Dispositivos cercanos: {', '.join(devices[:5])}"

    async def _handle_scan_devices(self, command: PresenceCommand, speaker_user) -> str:
        """Escanear dispositivos BLE"""
        all_devices = []

        for zone_id, scanner in self.detector._scanners.items():
            devices = await scanner.scan_once(timeout=5.0)
            for d in devices:
                if not d.user_id:  # Solo no registrados
                    name = d.name or d.address[-8:]
                    all_devices.append(f"{name} ({d.device_type.value})")

        if not all_devices:
            return "No encontré dispositivos nuevos."

        return f"Dispositivos encontrados: {', '.join(all_devices[:8])}"

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_user(self, name: str) -> Optional[str]:
        """Resolver nombre de usuario a user_id"""
        if not self.user_manager:
            return None

        name_lower = name.lower()
        for user in self.user_manager.get_all_users():
            if user.name.lower() == name_lower:
                return user.id
            if any(alias.lower() == name_lower for alias in getattr(user, 'aliases', [])):
                return user.id

        return None

    def _resolve_zone(self, name: str) -> Optional[str]:
        """Resolver nombre de zona a zone_id"""
        name_lower = name.lower()

        for zone_id, config in self.detector._zones.items():
            if zone_id.lower() == name_lower:
                return zone_id
            if config.get("name", "").lower() == name_lower:
                return zone_id

        return None

    def _get_user_name(self, user_id: str) -> str:
        """Obtener nombre de usuario"""
        if not self.user_manager or not user_id:
            return user_id or "desconocido"

        user = self.user_manager.get_user(user_id)
        return user.name if user else user_id

    def _get_user_names(self, user_ids: list) -> list[str]:
        """Obtener lista de nombres de usuarios"""
        return [self._get_user_name(uid) for uid in user_ids]

    def _get_zone_name(self, zone_id: str) -> str:
        """Obtener nombre de zona"""
        if zone_id in self.detector._zones:
            return self.detector._zones[zone_id].get("name", zone_id)
        return zone_id
