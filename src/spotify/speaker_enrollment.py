"""
Speaker Enrollment - Aprendizaje de bocinas por voz

Permite enseñarle a KZA sobre las bocinas mientras las instalas:
- "KZA, esta bocina se llama cocina"
- "KZA, la cocina está en la planta baja"
- "KZA, crea un grupo llamado área social con cocina y sala"
- "KZA, detecté una bocina nueva?"

Auto-descubre dispositivos Spotify y guía al usuario para configurarlos.
"""

import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum
from pathlib import Path

from .client import SpotifyClient, SpotifyDevice
from .speaker_groups import SpeakerGroupManager, Speaker, SpeakerGroup, GroupType

logger = logging.getLogger(__name__)


class EnrollmentIntent(Enum):
    """Intents para comandos de enrollment"""
    # Naming
    NAME_SPEAKER = "name_speaker"           # "esta bocina se llama cocina"
    RENAME_SPEAKER = "rename_speaker"       # "renombra la cocina a cocina principal"

    # Location
    SET_ROOM = "set_room"                   # "la cocina está en la cocina" (habitación)
    SET_FLOOR = "set_floor"                 # "la cocina está en la planta baja"

    # Groups
    CREATE_GROUP = "create_group"           # "crea un grupo llamado X con Y y Z"
    ADD_TO_GROUP = "add_to_group"           # "agrega la cocina al grupo planta baja"
    REMOVE_FROM_GROUP = "remove_from_group" # "quita la cocina del grupo X"
    DELETE_GROUP = "delete_group"           # "elimina el grupo X"

    # Discovery
    DISCOVER_DEVICES = "discover_devices"   # "busca bocinas nuevas"
    LIST_SPEAKERS = "list_speakers"         # "qué bocinas tengo?"
    LIST_GROUPS = "list_groups"             # "qué grupos tengo?"
    LIST_PENDING = "list_pending"           # "hay bocinas sin configurar?"

    # Aliases
    ADD_ALIAS = "add_alias"                 # "la cocina también se llama kitchen"

    # Default
    SET_DEFAULT = "set_default"             # "la sala es la bocina principal"

    # Removal
    FORGET_SPEAKER = "forget_speaker"       # "olvida la bocina del garage"

    # Confirmation
    CONFIRM = "confirm"                     # "sí", "confirmo"
    CANCEL = "cancel"                       # "no", "cancela"

    UNKNOWN = "unknown"


@dataclass
class EnrollmentCommand:
    """Comando de enrollment parseado"""
    intent: EnrollmentIntent
    raw_text: str
    speaker_name: Optional[str] = None
    new_name: Optional[str] = None
    room: Optional[str] = None
    floor: Optional[str] = None
    group_name: Optional[str] = None
    speaker_list: List[str] = field(default_factory=list)
    alias: Optional[str] = None


@dataclass
class PendingDevice:
    """Dispositivo Spotify descubierto pero no configurado"""
    spotify_device: SpotifyDevice
    discovered_at: float
    suggested_name: Optional[str] = None


class SpeakerEnrollment:
    """
    Sistema de enrollment de bocinas por voz.

    Uso típico:
        enrollment = SpeakerEnrollment(spotify_client, speaker_manager)

        # El usuario dice: "KZA, esta bocina se llama cocina"
        result = await enrollment.process("esta bocina se llama cocina")
        # KZA responde: "Entendido. La bocina 'Echo Dot' ahora se llama 'cocina'"

        # Después: "la cocina está en la planta baja"
        result = await enrollment.process("la cocina está en la planta baja")
        # KZA responde: "Listo. La cocina está en la planta baja"
    """

    def __init__(
        self,
        spotify_client: SpotifyClient,
        speaker_manager: SpeakerGroupManager,
        auto_save: bool = True,
        config_path: Optional[Path] = None
    ):
        self.spotify = spotify_client
        self.speakers = speaker_manager
        self.auto_save = auto_save
        self.config_path = config_path or Path("./data/speaker_groups.json")

        # Dispositivos pendientes de configurar
        self._pending_devices: Dict[str, PendingDevice] = {}

        # Estado de conversación (para confirmaciones)
        self._pending_action: Optional[Dict] = None
        self._last_mentioned_speaker: Optional[str] = None

        # Callback para notificar al usuario
        self._notify_callback: Optional[Callable[[str], None]] = None

        # Keywords para detección
        self._floor_keywords = {
            "planta baja": "planta_baja",
            "primer piso": "planta_baja",
            "abajo": "planta_baja",
            "planta alta": "planta_alta",
            "segundo piso": "planta_alta",
            "arriba": "planta_alta",
            "tercer piso": "tercer_piso",
            "sótano": "sotano",
            "basement": "sotano",
        }

    # =========================================================================
    # Detección de Intent
    # =========================================================================

    def detect_intent(self, text: str) -> EnrollmentCommand:
        """Detectar intent de un comando de enrollment"""
        text_lower = text.lower().strip()

        # Confirmación/Cancelación (prioridad alta)
        if self._pending_action:
            if any(w in text_lower for w in ["sí", "si", "confirmo", "ok", "dale", "correcto"]):
                return EnrollmentCommand(intent=EnrollmentIntent.CONFIRM, raw_text=text)
            if any(w in text_lower for w in ["no", "cancela", "cancelar", "olvídalo"]):
                return EnrollmentCommand(intent=EnrollmentIntent.CANCEL, raw_text=text)

        # Add alias (antes de NAME_SPEAKER porque es más específico)
        alias_match = re.search(r"(?:la\s+)?(\w+)\s+también\s+se\s+llama\s+(.+)", text_lower)
        if alias_match:
            return EnrollmentCommand(
                intent=EnrollmentIntent.ADD_ALIAS,
                raw_text=text,
                speaker_name=alias_match.group(1).strip(),
                alias=alias_match.group(2).strip()
            )

        # Naming: "esta bocina se llama X" / "llama a esta bocina X"
        name_patterns = [
            r"(?:esta\s+)?bocina\s+se\s+llama\s+(.+)",
            r"llama(?:la|le)?\s+(?:a\s+esta\s+bocina\s+)?(.+)",
            r"ponle\s+(?:de\s+nombre\s+)?(.+?)(?:\s+a\s+esta\s+bocina)?",
            r"nombre\s+(?:es\s+)?(.+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return EnrollmentCommand(
                    intent=EnrollmentIntent.NAME_SPEAKER,
                    raw_text=text,
                    new_name=match.group(1).strip()
                )

        # Renombrar: "renombra X a Y"
        rename_match = re.search(r"renombra\s+(?:la\s+)?(.+?)\s+a\s+(.+)", text_lower)
        if rename_match:
            return EnrollmentCommand(
                intent=EnrollmentIntent.RENAME_SPEAKER,
                raw_text=text,
                speaker_name=rename_match.group(1).strip(),
                new_name=rename_match.group(2).strip()
            )

        # Set room: "la X está en la Y" (habitación)
        room_match = re.search(r"(?:la\s+)?(\w+)\s+está\s+en\s+(?:la\s+|el\s+)?(?:habitación\s+|cuarto\s+)?(\w+)", text_lower)
        if room_match and not any(f in text_lower for f in self._floor_keywords.keys()):
            return EnrollmentCommand(
                intent=EnrollmentIntent.SET_ROOM,
                raw_text=text,
                speaker_name=room_match.group(1).strip(),
                room=room_match.group(2).strip()
            )

        # Set floor: "la X está en la planta baja"
        for floor_phrase, floor_id in self._floor_keywords.items():
            if floor_phrase in text_lower:
                speaker_match = re.search(r"(?:la\s+)?(\w+)\s+está", text_lower)
                speaker_name = speaker_match.group(1) if speaker_match else self._last_mentioned_speaker
                return EnrollmentCommand(
                    intent=EnrollmentIntent.SET_FLOOR,
                    raw_text=text,
                    speaker_name=speaker_name,
                    floor=floor_id
                )

        # Create group: "crea un grupo llamado X con Y y Z"
        group_create = re.search(
            r"crea\s+(?:un\s+)?grupo\s+(?:llamado\s+)?(.+?)\s+con\s+(.+)",
            text_lower
        )
        if group_create:
            group_name = group_create.group(1).strip()
            speakers_text = group_create.group(2).strip()
            # Parse speaker list: "cocina, sala y dormitorio" or "cocina y sala"
            speaker_list = self._parse_speaker_list(speakers_text)
            return EnrollmentCommand(
                intent=EnrollmentIntent.CREATE_GROUP,
                raw_text=text,
                group_name=group_name,
                speaker_list=speaker_list
            )

        # Add to group: "agrega X al grupo Y"
        add_group = re.search(r"agrega\s+(?:la\s+)?(.+?)\s+al?\s+grupo\s+(.+)", text_lower)
        if add_group:
            return EnrollmentCommand(
                intent=EnrollmentIntent.ADD_TO_GROUP,
                raw_text=text,
                speaker_name=add_group.group(1).strip(),
                group_name=add_group.group(2).strip()
            )

        # Remove from group
        remove_group = re.search(r"(?:quita|elimina|saca)\s+(?:la\s+)?(.+?)\s+del?\s+grupo\s+(.+)", text_lower)
        if remove_group:
            return EnrollmentCommand(
                intent=EnrollmentIntent.REMOVE_FROM_GROUP,
                raw_text=text,
                speaker_name=remove_group.group(1).strip(),
                group_name=remove_group.group(2).strip()
            )

        # Delete group
        if re.search(r"(?:elimina|borra|quita)\s+(?:el\s+)?grupo\s+(.+)", text_lower):
            match = re.search(r"grupo\s+(.+)", text_lower)
            return EnrollmentCommand(
                intent=EnrollmentIntent.DELETE_GROUP,
                raw_text=text,
                group_name=match.group(1).strip() if match else None
            )

        # Discovery
        if any(w in text_lower for w in ["busca bocinas", "detecta bocinas", "bocinas nuevas", "descubre"]):
            return EnrollmentCommand(intent=EnrollmentIntent.DISCOVER_DEVICES, raw_text=text)

        # List speakers
        if any(w in text_lower for w in ["qué bocinas", "que bocinas", "lista de bocinas", "mis bocinas"]):
            return EnrollmentCommand(intent=EnrollmentIntent.LIST_SPEAKERS, raw_text=text)

        # List groups
        if any(w in text_lower for w in ["qué grupos", "que grupos", "lista de grupos", "mis grupos"]):
            return EnrollmentCommand(intent=EnrollmentIntent.LIST_GROUPS, raw_text=text)

        # Pending devices
        if any(w in text_lower for w in ["sin configurar", "pendientes", "falta configurar"]):
            return EnrollmentCommand(intent=EnrollmentIntent.LIST_PENDING, raw_text=text)

        # Set default
        if any(w in text_lower for w in ["bocina principal", "por defecto", "default"]):
            speaker_match = re.search(r"(?:la\s+)?(\w+)\s+(?:es|será)", text_lower)
            return EnrollmentCommand(
                intent=EnrollmentIntent.SET_DEFAULT,
                raw_text=text,
                speaker_name=speaker_match.group(1) if speaker_match else None
            )

        # Forget speaker
        forget_match = re.search(r"(?:olvida|elimina|borra)\s+(?:la\s+)?bocina\s+(?:del?\s+)?(.+)", text_lower)
        if forget_match:
            return EnrollmentCommand(
                intent=EnrollmentIntent.FORGET_SPEAKER,
                raw_text=text,
                speaker_name=forget_match.group(1).strip()
            )

        return EnrollmentCommand(intent=EnrollmentIntent.UNKNOWN, raw_text=text)

    def _parse_speaker_list(self, text: str) -> List[str]:
        """Parsear lista de speakers: 'cocina, sala y dormitorio'"""
        # Reemplazar "y" por coma
        text = text.replace(" y ", ", ")
        # Split por coma
        speakers = [s.strip() for s in text.split(",")]
        return [s for s in speakers if s]

    # =========================================================================
    # Procesamiento de Comandos
    # =========================================================================

    async def process(self, text: str) -> Dict[str, Any]:
        """
        Procesar comando de enrollment.

        Returns:
            Dict con:
            - success: bool
            - response: str (para que KZA diga)
            - action: str (acción realizada)
            - needs_confirmation: bool (si espera confirmación)
        """
        command = self.detect_intent(text)

        try:
            if command.intent == EnrollmentIntent.CONFIRM:
                return await self._handle_confirm()

            if command.intent == EnrollmentIntent.CANCEL:
                return self._handle_cancel()

            if command.intent == EnrollmentIntent.NAME_SPEAKER:
                return await self._handle_name_speaker(command)

            if command.intent == EnrollmentIntent.RENAME_SPEAKER:
                return await self._handle_rename_speaker(command)

            if command.intent == EnrollmentIntent.SET_ROOM:
                return self._handle_set_room(command)

            if command.intent == EnrollmentIntent.SET_FLOOR:
                return self._handle_set_floor(command)

            if command.intent == EnrollmentIntent.CREATE_GROUP:
                return self._handle_create_group(command)

            if command.intent == EnrollmentIntent.ADD_TO_GROUP:
                return self._handle_add_to_group(command)

            if command.intent == EnrollmentIntent.REMOVE_FROM_GROUP:
                return self._handle_remove_from_group(command)

            if command.intent == EnrollmentIntent.DELETE_GROUP:
                return self._handle_delete_group(command)

            if command.intent == EnrollmentIntent.DISCOVER_DEVICES:
                return await self._handle_discover()

            if command.intent == EnrollmentIntent.LIST_SPEAKERS:
                return self._handle_list_speakers()

            if command.intent == EnrollmentIntent.LIST_GROUPS:
                return self._handle_list_groups()

            if command.intent == EnrollmentIntent.LIST_PENDING:
                return self._handle_list_pending()

            if command.intent == EnrollmentIntent.ADD_ALIAS:
                return self._handle_add_alias(command)

            if command.intent == EnrollmentIntent.SET_DEFAULT:
                return self._handle_set_default(command)

            if command.intent == EnrollmentIntent.FORGET_SPEAKER:
                return self._handle_forget_speaker(command)

            return {
                "success": False,
                "response": "No entendí qué quieres hacer con las bocinas",
                "action": "unknown"
            }

        except Exception as e:
            logger.error(f"Error processing enrollment: {e}")
            return {
                "success": False,
                "response": f"Hubo un error: {str(e)}",
                "action": "error"
            }

    # =========================================================================
    # Handlers
    # =========================================================================

    async def _handle_name_speaker(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Nombrar la bocina activa actual"""
        # Buscar dispositivo activo en Spotify
        devices = await self.spotify.get_devices()
        active_device = next((d for d in devices if d.is_active), None)

        if not active_device:
            # Buscar en pendientes
            if self._pending_devices:
                # Usar el más reciente
                pending = list(self._pending_devices.values())[-1]
                active_device = pending.spotify_device
            else:
                return {
                    "success": False,
                    "response": "No encontré ninguna bocina activa. Reproduce algo primero para que pueda detectarla.",
                    "action": "name_failed"
                }

        new_name = command.new_name
        speaker_id = new_name.lower().replace(" ", "_")

        # Crear speaker
        speaker = Speaker(
            id=speaker_id,
            name=new_name.title(),
            spotify_device_id=active_device.id,
            aliases=[new_name.lower()]
        )

        self.speakers.add_speaker(speaker)
        self._last_mentioned_speaker = speaker_id

        # Remover de pendientes si estaba
        if active_device.id in self._pending_devices:
            del self._pending_devices[active_device.id]

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Entendido. La bocina '{active_device.name}' ahora se llama '{new_name}'.",
            "action": "speaker_named",
            "speaker_id": speaker_id
        }

    async def _handle_rename_speaker(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Renombrar una bocina existente"""
        # Buscar speaker por nombre
        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {
                "success": False,
                "response": f"No encontré la bocina '{command.speaker_name}'",
                "action": "rename_failed"
            }

        old_speaker = target["speaker"]
        old_name = old_speaker.name

        # Actualizar nombre
        old_speaker.name = command.new_name.title()
        old_speaker.aliases.append(command.new_name.lower())

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. '{old_name}' ahora se llama '{command.new_name}'.",
            "action": "speaker_renamed"
        }

    def _handle_set_room(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Asignar habitación a una bocina"""
        speaker_name = command.speaker_name or self._last_mentioned_speaker
        if not speaker_name:
            return {
                "success": False,
                "response": "¿De qué bocina me hablas?",
                "action": "set_room_failed"
            }

        target = self.speakers.resolve_target(speaker_name)
        if not target or target["type"] != "speaker":
            return {
                "success": False,
                "response": f"No encontré la bocina '{speaker_name}'",
                "action": "set_room_failed"
            }

        speaker = target["speaker"]
        speaker.room = command.room
        self._last_mentioned_speaker = speaker.id

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. La {speaker.name} está en {command.room}.",
            "action": "room_set"
        }

    def _handle_set_floor(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Asignar piso a una bocina"""
        speaker_name = command.speaker_name or self._last_mentioned_speaker
        if not speaker_name:
            return {
                "success": False,
                "response": "¿De qué bocina me hablas?",
                "action": "set_floor_failed"
            }

        target = self.speakers.resolve_target(speaker_name)
        if not target or target["type"] != "speaker":
            return {
                "success": False,
                "response": f"No encontré la bocina '{speaker_name}'",
                "action": "set_floor_failed"
            }

        speaker = target["speaker"]
        speaker.floor = command.floor
        self._last_mentioned_speaker = speaker.id

        # Nombre legible del piso
        floor_names = {
            "planta_baja": "planta baja",
            "planta_alta": "planta alta",
            "tercer_piso": "tercer piso",
            "sotano": "sótano"
        }
        floor_display = floor_names.get(command.floor, command.floor)

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. La {speaker.name} está en la {floor_display}.",
            "action": "floor_set"
        }

    def _handle_create_group(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Crear un grupo de bocinas"""
        if not command.group_name:
            return {"success": False, "response": "¿Cómo quieres que se llame el grupo?", "action": "create_group_failed"}

        if not command.speaker_list:
            return {"success": False, "response": "¿Qué bocinas quieres en el grupo?", "action": "create_group_failed"}

        # Resolver speakers
        speaker_ids = []
        not_found = []

        for name in command.speaker_list:
            target = self.speakers.resolve_target(name)
            if target and target["type"] == "speaker":
                speaker_ids.append(target["speaker"].id)
            else:
                not_found.append(name)

        if not speaker_ids:
            return {
                "success": False,
                "response": f"No encontré ninguna de esas bocinas: {', '.join(command.speaker_list)}",
                "action": "create_group_failed"
            }

        group_id = command.group_name.lower().replace(" ", "_")

        try:
            self.speakers.create_group(
                id=group_id,
                name=command.group_name.title(),
                group_type=GroupType.ZONE,
                speaker_ids=speaker_ids,
                aliases=[command.group_name.lower()]
            )
        except ValueError as e:
            return {"success": False, "response": str(e), "action": "create_group_failed"}

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        response = f"Grupo '{command.group_name}' creado con {len(speaker_ids)} bocinas."
        if not_found:
            response += f" No encontré: {', '.join(not_found)}."

        return {
            "success": True,
            "response": response,
            "action": "group_created",
            "group_id": group_id
        }

    def _handle_add_to_group(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Agregar bocina a un grupo"""
        # Buscar speaker
        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {"success": False, "response": f"No encontré la bocina '{command.speaker_name}'", "action": "add_to_group_failed"}

        # Buscar grupo
        group = None
        for g in self.speakers.groups.values():
            if g.matches_name(command.group_name):
                group = g
                break

        if not group:
            return {"success": False, "response": f"No encontré el grupo '{command.group_name}'", "action": "add_to_group_failed"}

        speaker = target["speaker"]
        self.speakers.add_speaker_to_group(speaker.id, group.id)

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. {speaker.name} está ahora en el grupo {group.name}.",
            "action": "added_to_group"
        }

    def _handle_remove_from_group(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Quitar bocina de un grupo"""
        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {"success": False, "response": f"No encontré la bocina '{command.speaker_name}'", "action": "remove_from_group_failed"}

        group = None
        for g in self.speakers.groups.values():
            if g.matches_name(command.group_name):
                group = g
                break

        if not group:
            return {"success": False, "response": f"No encontré el grupo '{command.group_name}'", "action": "remove_from_group_failed"}

        speaker = target["speaker"]
        self.speakers.remove_speaker_from_group(speaker.id, group.id)

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. {speaker.name} ya no está en el grupo {group.name}.",
            "action": "removed_from_group"
        }

    def _handle_delete_group(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Eliminar un grupo"""
        if command.group_name == "everywhere" or command.group_name == "toda la casa":
            return {"success": False, "response": "No puedo eliminar el grupo 'toda la casa'.", "action": "delete_group_failed"}

        group = None
        for g in self.speakers.groups.values():
            if g.matches_name(command.group_name):
                group = g
                break

        if not group:
            return {"success": False, "response": f"No encontré el grupo '{command.group_name}'", "action": "delete_group_failed"}

        # Pedir confirmación
        self._pending_action = {
            "type": "delete_group",
            "group_id": group.id,
            "group_name": group.name
        }

        return {
            "success": True,
            "response": f"¿Seguro que quieres eliminar el grupo '{group.name}'?",
            "action": "confirm_delete_group",
            "needs_confirmation": True
        }

    async def _handle_discover(self) -> Dict[str, Any]:
        """Buscar dispositivos Spotify nuevos"""
        import time

        devices = await self.spotify.get_devices()

        # Encontrar dispositivos no configurados
        configured_ids = {s.spotify_device_id for s in self.speakers.speakers.values() if s.spotify_device_id}
        new_devices = [d for d in devices if d.id not in configured_ids]

        # Guardar como pendientes
        for device in new_devices:
            self._pending_devices[device.id] = PendingDevice(
                spotify_device=device,
                discovered_at=time.time(),
                suggested_name=self._suggest_name(device.name)
            )

        if not new_devices:
            if devices:
                return {
                    "success": True,
                    "response": f"Encontré {len(devices)} bocinas, pero todas ya están configuradas.",
                    "action": "discover_none_new"
                }
            return {
                "success": False,
                "response": "No encontré ninguna bocina de Spotify. Asegúrate de que estén encendidas.",
                "action": "discover_none"
            }

        # Construir respuesta
        if len(new_devices) == 1:
            device = new_devices[0]
            suggested = self._pending_devices[device.id].suggested_name
            response = f"Encontré una bocina nueva: '{device.name}'."
            if suggested:
                response += f" ¿Quieres que la llame '{suggested}'?"
                self._pending_action = {
                    "type": "name_suggested",
                    "device_id": device.id,
                    "suggested_name": suggested
                }
                return {
                    "success": True,
                    "response": response,
                    "action": "discover_one",
                    "needs_confirmation": True
                }
        else:
            names = [d.name for d in new_devices]
            response = f"Encontré {len(new_devices)} bocinas nuevas: {', '.join(names)}. "
            response += "Dime cómo quieres llamar a cada una."

        return {
            "success": True,
            "response": response,
            "action": "discover_found",
            "devices": [{"id": d.id, "name": d.name} for d in new_devices]
        }

    def _suggest_name(self, device_name: str) -> Optional[str]:
        """Sugerir nombre basado en el nombre del dispositivo"""
        name_lower = device_name.lower()

        # Patrones comunes
        room_patterns = {
            "kitchen": "cocina",
            "cocina": "cocina",
            "living": "sala",
            "sala": "sala",
            "bedroom": "dormitorio",
            "dormitorio": "dormitorio",
            "bathroom": "baño",
            "baño": "baño",
            "office": "oficina",
            "oficina": "oficina",
        }

        for pattern, suggestion in room_patterns.items():
            if pattern in name_lower:
                return suggestion

        return None

    def _handle_list_speakers(self) -> Dict[str, Any]:
        """Listar todas las bocinas"""
        speakers = list(self.speakers.speakers.values())

        if not speakers:
            return {
                "success": True,
                "response": "No tienes bocinas configuradas todavía.",
                "action": "list_speakers_empty"
            }

        lines = []
        for s in speakers:
            line = s.name
            if s.room:
                line += f" (en {s.room})"
            if s.is_default:
                line += " - principal"
            lines.append(line)

        response = f"Tienes {len(speakers)} bocinas: " + ", ".join(lines) + "."

        return {
            "success": True,
            "response": response,
            "action": "list_speakers",
            "count": len(speakers)
        }

    def _handle_list_groups(self) -> Dict[str, Any]:
        """Listar todos los grupos"""
        groups = [g for g in self.speakers.groups.values() if g.id != "everywhere"]

        if not groups:
            return {
                "success": True,
                "response": "No tienes grupos personalizados. Solo está el grupo automático 'toda la casa'.",
                "action": "list_groups_empty"
            }

        lines = []
        for g in groups:
            count = len(g.speaker_ids)
            lines.append(f"{g.name} ({count} bocinas)")

        response = f"Tienes {len(groups)} grupos: " + ", ".join(lines) + "."

        return {
            "success": True,
            "response": response,
            "action": "list_groups",
            "count": len(groups)
        }

    def _handle_list_pending(self) -> Dict[str, Any]:
        """Listar dispositivos pendientes de configurar"""
        if not self._pending_devices:
            return {
                "success": True,
                "response": "No hay bocinas pendientes de configurar.",
                "action": "list_pending_empty"
            }

        names = [p.spotify_device.name for p in self._pending_devices.values()]
        response = f"Hay {len(names)} bocinas sin configurar: {', '.join(names)}."

        return {
            "success": True,
            "response": response,
            "action": "list_pending",
            "count": len(names)
        }

    def _handle_add_alias(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Agregar alias a una bocina"""
        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {"success": False, "response": f"No encontré la bocina '{command.speaker_name}'", "action": "add_alias_failed"}

        speaker = target["speaker"]
        if command.alias not in speaker.aliases:
            speaker.aliases.append(command.alias.lower())

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. Ahora puedes llamar a {speaker.name} también como '{command.alias}'.",
            "action": "alias_added"
        }

    def _handle_set_default(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Establecer bocina por defecto"""
        if not command.speaker_name:
            return {"success": False, "response": "¿Cuál bocina quieres como principal?", "action": "set_default_failed"}

        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {"success": False, "response": f"No encontré la bocina '{command.speaker_name}'", "action": "set_default_failed"}

        # Quitar default de otras
        for s in self.speakers.speakers.values():
            s.is_default = False

        speaker = target["speaker"]
        speaker.is_default = True

        if self.auto_save:
            self.speakers.save_config(self.config_path)

        return {
            "success": True,
            "response": f"Listo. {speaker.name} es ahora la bocina principal.",
            "action": "default_set"
        }

    def _handle_forget_speaker(self, command: EnrollmentCommand) -> Dict[str, Any]:
        """Olvidar una bocina"""
        target = self.speakers.resolve_target(command.speaker_name)
        if not target or target["type"] != "speaker":
            return {"success": False, "response": f"No encontré la bocina '{command.speaker_name}'", "action": "forget_failed"}

        speaker = target["speaker"]

        # Pedir confirmación
        self._pending_action = {
            "type": "forget_speaker",
            "speaker_id": speaker.id,
            "speaker_name": speaker.name
        }

        return {
            "success": True,
            "response": f"¿Seguro que quieres olvidar la bocina '{speaker.name}'?",
            "action": "confirm_forget",
            "needs_confirmation": True
        }

    async def _handle_confirm(self) -> Dict[str, Any]:
        """Manejar confirmación"""
        if not self._pending_action:
            return {"success": False, "response": "No hay nada pendiente de confirmar.", "action": "confirm_nothing"}

        action = self._pending_action
        self._pending_action = None

        if action["type"] == "delete_group":
            self.speakers.delete_group(action["group_id"])
            if self.auto_save:
                self.speakers.save_config(self.config_path)
            return {
                "success": True,
                "response": f"Grupo '{action['group_name']}' eliminado.",
                "action": "group_deleted"
            }

        if action["type"] == "forget_speaker":
            self.speakers.remove_speaker(action["speaker_id"])
            if self.auto_save:
                self.speakers.save_config(self.config_path)
            return {
                "success": True,
                "response": f"Bocina '{action['speaker_name']}' olvidada.",
                "action": "speaker_forgotten"
            }

        if action["type"] == "name_suggested":
            # Nombrar con el nombre sugerido
            device_id = action["device_id"]
            suggested_name = action["suggested_name"]

            if device_id in self._pending_devices:
                pending = self._pending_devices[device_id]
                speaker_id = suggested_name.lower().replace(" ", "_")

                speaker = Speaker(
                    id=speaker_id,
                    name=suggested_name.title(),
                    spotify_device_id=device_id,
                    aliases=[suggested_name.lower()]
                )

                self.speakers.add_speaker(speaker)
                del self._pending_devices[device_id]

                if self.auto_save:
                    self.speakers.save_config(self.config_path)

                return {
                    "success": True,
                    "response": f"Listo. La bocina ahora se llama '{suggested_name}'.",
                    "action": "speaker_named"
                }

        return {"success": False, "response": "No pude procesar la confirmación.", "action": "confirm_error"}

    def _handle_cancel(self) -> Dict[str, Any]:
        """Cancelar acción pendiente"""
        self._pending_action = None
        return {
            "success": True,
            "response": "Cancelado.",
            "action": "cancelled"
        }

    # =========================================================================
    # Auto-Discovery en Background
    # =========================================================================

    async def check_new_devices(self) -> Optional[str]:
        """
        Verificar si hay dispositivos nuevos (llamar periódicamente).

        Returns:
            Mensaje para el usuario si hay dispositivos nuevos, None si no hay
        """
        import time

        try:
            devices = await self.spotify.get_devices()
        except Exception:
            return None

        configured_ids = {s.spotify_device_id for s in self.speakers.speakers.values() if s.spotify_device_id}

        new_devices = []
        for device in devices:
            if device.id not in configured_ids and device.id not in self._pending_devices:
                self._pending_devices[device.id] = PendingDevice(
                    spotify_device=device,
                    discovered_at=time.time(),
                    suggested_name=self._suggest_name(device.name)
                )
                new_devices.append(device)

        if new_devices:
            if len(new_devices) == 1:
                device = new_devices[0]
                return f"Detecté una bocina nueva: '{device.name}'. ¿Cómo quieres llamarla?"
            else:
                names = [d.name for d in new_devices]
                return f"Detecté {len(new_devices)} bocinas nuevas: {', '.join(names)}. Dime cómo llamarlas."

        return None

    def on_notify(self, callback: Callable[[str], None]):
        """Registrar callback para notificaciones"""
        self._notify_callback = callback
