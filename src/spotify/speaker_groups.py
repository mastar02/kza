"""
Speaker Groups - Gestión de grupos de altavoces estilo Alexa

Permite:
- Definir grupos de altavoces (toda la casa, planta baja, dormitorios)
- Mapear dispositivos Spotify a zonas de la casa
- Comandos de voz: "pon música en la cocina", "pon música en toda la casa"
- Seguimiento automático: la música sigue al usuario
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class GroupType(Enum):
    """Tipos de grupos de altavoces"""
    ROOM = "room"           # Habitación individual (cocina, sala)
    FLOOR = "floor"         # Piso completo (planta baja, planta alta)
    ZONE = "zone"           # Zona personalizada (área social, dormitorios)
    EVERYWHERE = "everywhere"  # Toda la casa


@dataclass
class Speaker:
    """Un altavoz/dispositivo de audio"""
    id: str                     # ID único interno
    name: str                   # Nombre amigable ("Bocina Cocina")
    spotify_device_id: Optional[str] = None  # ID de Spotify Connect
    zone_id: Optional[str] = None            # ID de zona MA1260
    room: Optional[str] = None               # Habitación ("cocina", "sala")
    floor: Optional[str] = None              # Piso ("planta_baja", "planta_alta")
    is_default: bool = False                 # Altavoz por defecto
    supports_spotify: bool = True            # Si soporta Spotify Connect
    volume_offset: int = 0                   # Ajuste de volumen relativo (-20 a +20)

    # Aliases para reconocimiento de voz
    aliases: List[str] = field(default_factory=list)

    def matches_name(self, query: str) -> bool:
        """Verificar si el query coincide con este speaker"""
        query_lower = query.lower().strip()

        # Coincidencia directa
        if query_lower in self.name.lower():
            return True

        # Coincidencia por habitación
        if self.room and query_lower in self.room.lower():
            return True

        # Coincidencia por alias
        for alias in self.aliases:
            if query_lower in alias.lower() or alias.lower() in query_lower:
                return True

        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "spotify_device_id": self.spotify_device_id,
            "zone_id": self.zone_id,
            "room": self.room,
            "floor": self.floor,
            "is_default": self.is_default,
            "supports_spotify": self.supports_spotify,
            "volume_offset": self.volume_offset,
            "aliases": self.aliases
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Speaker":
        return cls(**data)


@dataclass
class SpeakerGroup:
    """Grupo de altavoces"""
    id: str                     # "toda_la_casa", "planta_baja"
    name: str                   # Nombre amigable
    group_type: GroupType       # Tipo de grupo
    speaker_ids: List[str]      # IDs de speakers en el grupo

    # Aliases para reconocimiento de voz
    aliases: List[str] = field(default_factory=list)

    # Configuración de grupo
    sync_playback: bool = True  # Sincronizar reproducción entre speakers
    master_speaker_id: Optional[str] = None  # Speaker principal del grupo

    def matches_name(self, query: str) -> bool:
        """Verificar si el query coincide con este grupo"""
        query_lower = query.lower().strip()

        if query_lower in self.name.lower():
            return True

        for alias in self.aliases:
            if query_lower in alias.lower() or alias.lower() in query_lower:
                return True

        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "group_type": self.group_type.value,
            "speaker_ids": self.speaker_ids,
            "aliases": self.aliases,
            "sync_playback": self.sync_playback,
            "master_speaker_id": self.master_speaker_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerGroup":
        data = data.copy()
        data["group_type"] = GroupType(data["group_type"])
        return cls(**data)


class SpeakerGroupManager:
    """
    Gestor de grupos de altavoces.

    Uso:
        manager = SpeakerGroupManager()

        # Agregar speakers
        manager.add_speaker(Speaker(
            id="kitchen",
            name="Bocina Cocina",
            spotify_device_id="abc123",
            room="cocina",
            floor="planta_baja"
        ))

        # Crear grupo
        manager.create_group(
            id="downstairs",
            name="Planta Baja",
            group_type=GroupType.FLOOR,
            speaker_ids=["kitchen", "living"]
        )

        # Resolver destino de "pon música en la cocina"
        target = manager.resolve_target("cocina")
        # target = {"type": "speaker", "speaker": Speaker(...)}

        # O "pon música en toda la casa"
        target = manager.resolve_target("toda la casa")
        # target = {"type": "group", "group": SpeakerGroup(...), "speakers": [...]}
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        auto_discover: bool = True
    ):
        """
        Args:
            config_path: Ruta al archivo de configuración
            auto_discover: Intentar descubrir dispositivos automáticamente
        """
        self.speakers: Dict[str, Speaker] = {}
        self.groups: Dict[str, SpeakerGroup] = {}
        self.config_path = config_path

        # Aliases comunes para "toda la casa"
        self._everywhere_aliases = [
            "toda la casa", "everywhere", "all", "todas partes",
            "todos los cuartos", "todas las habitaciones", "en todos lados"
        ]

        # Cargar configuración si existe
        if config_path and config_path.exists():
            self.load_config(config_path)

        # Crear grupo "toda la casa" automáticamente
        self._ensure_everywhere_group()

    # =========================================================================
    # Gestión de Speakers
    # =========================================================================

    def add_speaker(self, speaker: Speaker):
        """Agregar un altavoz"""
        self.speakers[speaker.id] = speaker
        logger.info(f"Speaker agregado: {speaker.name} ({speaker.id})")

        # Actualizar grupo "toda la casa"
        self._ensure_everywhere_group()

    def remove_speaker(self, speaker_id: str):
        """Eliminar un altavoz"""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]

            # Eliminar de grupos
            for group in self.groups.values():
                if speaker_id in group.speaker_ids:
                    group.speaker_ids.remove(speaker_id)

    def get_speaker(self, speaker_id: str) -> Optional[Speaker]:
        """Obtener speaker por ID"""
        return self.speakers.get(speaker_id)

    def get_speaker_by_spotify_id(self, spotify_id: str) -> Optional[Speaker]:
        """Obtener speaker por ID de Spotify"""
        for speaker in self.speakers.values():
            if speaker.spotify_device_id == spotify_id:
                return speaker
        return None

    def get_speakers_by_room(self, room: str) -> List[Speaker]:
        """Obtener speakers de una habitación"""
        return [s for s in self.speakers.values() if s.room and s.room.lower() == room.lower()]

    def get_speakers_by_floor(self, floor: str) -> List[Speaker]:
        """Obtener speakers de un piso"""
        return [s for s in self.speakers.values() if s.floor and s.floor.lower() == floor.lower()]

    def get_default_speaker(self) -> Optional[Speaker]:
        """Obtener speaker por defecto"""
        for speaker in self.speakers.values():
            if speaker.is_default:
                return speaker
        # Fallback al primero
        return next(iter(self.speakers.values()), None)

    def update_spotify_device_id(self, speaker_id: str, spotify_device_id: str):
        """Actualizar ID de Spotify para un speaker"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id].spotify_device_id = spotify_device_id
            logger.info(f"Speaker {speaker_id} vinculado a Spotify device {spotify_device_id}")

    # =========================================================================
    # Gestión de Grupos
    # =========================================================================

    def create_group(
        self,
        id: str,
        name: str,
        group_type: GroupType,
        speaker_ids: List[str],
        aliases: List[str] = None
    ) -> SpeakerGroup:
        """Crear un nuevo grupo de altavoces"""
        # Validar que los speakers existan
        valid_ids = [sid for sid in speaker_ids if sid in self.speakers]

        if not valid_ids:
            raise ValueError(f"No hay speakers válidos para el grupo {name}")

        group = SpeakerGroup(
            id=id,
            name=name,
            group_type=group_type,
            speaker_ids=valid_ids,
            aliases=aliases or [],
            master_speaker_id=valid_ids[0]  # Primer speaker como master
        )

        self.groups[id] = group
        logger.info(f"Grupo creado: {name} con {len(valid_ids)} speakers")

        return group

    def delete_group(self, group_id: str):
        """Eliminar un grupo"""
        if group_id in self.groups and group_id != "everywhere":
            del self.groups[group_id]

    def get_group(self, group_id: str) -> Optional[SpeakerGroup]:
        """Obtener grupo por ID"""
        return self.groups.get(group_id)

    def get_group_speakers(self, group_id: str) -> List[Speaker]:
        """Obtener speakers de un grupo"""
        group = self.groups.get(group_id)
        if not group:
            return []
        return [self.speakers[sid] for sid in group.speaker_ids if sid in self.speakers]

    def add_speaker_to_group(self, speaker_id: str, group_id: str):
        """Agregar speaker a un grupo"""
        if speaker_id in self.speakers and group_id in self.groups:
            if speaker_id not in self.groups[group_id].speaker_ids:
                self.groups[group_id].speaker_ids.append(speaker_id)

    def remove_speaker_from_group(self, speaker_id: str, group_id: str):
        """Quitar speaker de un grupo"""
        if group_id in self.groups:
            if speaker_id in self.groups[group_id].speaker_ids:
                self.groups[group_id].speaker_ids.remove(speaker_id)

    def _ensure_everywhere_group(self):
        """Asegurar que existe el grupo 'toda la casa'"""
        all_speaker_ids = list(self.speakers.keys())

        if "everywhere" not in self.groups:
            self.groups["everywhere"] = SpeakerGroup(
                id="everywhere",
                name="Toda la Casa",
                group_type=GroupType.EVERYWHERE,
                speaker_ids=all_speaker_ids,
                aliases=self._everywhere_aliases
            )
        else:
            # Actualizar speakers
            self.groups["everywhere"].speaker_ids = all_speaker_ids

    # =========================================================================
    # Resolución de Destino (para comandos de voz)
    # =========================================================================

    def resolve_target(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Resolver el destino de un comando de voz.

        Args:
            query: Texto del usuario ("cocina", "toda la casa", "dormitorio principal")

        Returns:
            Dict con información del destino:
            - type: "speaker" | "group"
            - speaker: Speaker (si type es speaker)
            - group: SpeakerGroup (si type es group)
            - speakers: List[Speaker] (siempre)
            - spotify_device_ids: List[str] (IDs de Spotify)
        """
        query_lower = query.lower().strip()

        # Primero verificar si es "toda la casa" o similar
        if self._is_everywhere_query(query_lower):
            return self._resolve_everywhere()

        # Buscar grupo que coincida
        for group in self.groups.values():
            if group.matches_name(query_lower):
                speakers = self.get_group_speakers(group.id)
                return {
                    "type": "group",
                    "group": group,
                    "speakers": speakers,
                    "spotify_device_ids": [s.spotify_device_id for s in speakers if s.spotify_device_id]
                }

        # Buscar speaker individual
        for speaker in self.speakers.values():
            if speaker.matches_name(query_lower):
                return {
                    "type": "speaker",
                    "speaker": speaker,
                    "speakers": [speaker],
                    "spotify_device_ids": [speaker.spotify_device_id] if speaker.spotify_device_id else []
                }

        # Buscar por habitación
        room_speakers = self.get_speakers_by_room(query_lower)
        if room_speakers:
            return {
                "type": "room",
                "room": query_lower,
                "speakers": room_speakers,
                "spotify_device_ids": [s.spotify_device_id for s in room_speakers if s.spotify_device_id]
            }

        # Buscar por piso
        floor_speakers = self.get_speakers_by_floor(query_lower)
        if floor_speakers:
            return {
                "type": "floor",
                "floor": query_lower,
                "speakers": floor_speakers,
                "spotify_device_ids": [s.spotify_device_id for s in floor_speakers if s.spotify_device_id]
            }

        return None

    def _is_everywhere_query(self, query: str) -> bool:
        """Verificar si el query se refiere a 'toda la casa'"""
        for alias in self._everywhere_aliases:
            if alias in query or query in alias:
                return True
        return False

    def _resolve_everywhere(self) -> Dict[str, Any]:
        """Resolver destino 'toda la casa'"""
        everywhere = self.groups.get("everywhere")
        speakers = list(self.speakers.values())

        return {
            "type": "group",
            "group": everywhere,
            "speakers": speakers,
            "spotify_device_ids": [s.spotify_device_id for s in speakers if s.spotify_device_id]
        }

    def resolve_default(self) -> Optional[Dict[str, Any]]:
        """Resolver destino por defecto (speaker principal)"""
        default = self.get_default_speaker()
        if default:
            return {
                "type": "speaker",
                "speaker": default,
                "speakers": [default],
                "spotify_device_ids": [default.spotify_device_id] if default.spotify_device_id else []
            }
        return None

    # =========================================================================
    # Parsing de Comandos de Voz
    # =========================================================================

    def parse_zone_from_command(self, text: str) -> tuple[Optional[Dict], str]:
        """
        Extraer zona/grupo de un comando de voz.

        Args:
            text: "pon música de Bad Bunny en la cocina"

        Returns:
            (target, cleaned_text):
            - target: Resultado de resolve_target() o None
            - cleaned_text: Texto sin la parte de la zona ("pon música de Bad Bunny")
        """
        text_lower = text.lower()

        # Patrones para detectar ubicación
        location_patterns = [
            " en la ", " en el ", " en ",
            " por toda ", " por todo ",
            " en todas ", " en todos "
        ]

        for pattern in location_patterns:
            if pattern in text_lower:
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    location_part = parts[1].strip()

                    # Limpiar palabras finales comunes
                    for suffix in [" por favor", " porfa", " gracias"]:
                        if location_part.endswith(suffix):
                            location_part = location_part[:-len(suffix)].strip()

                    target = self.resolve_target(location_part)
                    if target:
                        cleaned = parts[0].strip()
                        return target, cleaned

        # No se encontró ubicación
        return None, text

    # =========================================================================
    # Persistencia
    # =========================================================================

    def save_config(self, path: Optional[Path] = None):
        """Guardar configuración a archivo"""
        path = path or self.config_path
        if not path:
            return

        config = {
            "speakers": {sid: s.to_dict() for sid, s in self.speakers.items()},
            "groups": {gid: g.to_dict() for gid, g in self.groups.items() if gid != "everywhere"}
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuración guardada en {path}")

    def load_config(self, path: Path):
        """Cargar configuración desde archivo"""
        if not path.exists():
            return

        try:
            with open(path) as f:
                config = json.load(f)

            # Cargar speakers
            for sid, sdata in config.get("speakers", {}).items():
                self.speakers[sid] = Speaker.from_dict(sdata)

            # Cargar grupos
            for gid, gdata in config.get("groups", {}).items():
                self.groups[gid] = SpeakerGroup.from_dict(gdata)

            logger.info(f"Configuración cargada: {len(self.speakers)} speakers, {len(self.groups)} grupos")

        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")

    # =========================================================================
    # Estado
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual"""
        return {
            "speakers": {
                sid: {
                    "name": s.name,
                    "room": s.room,
                    "floor": s.floor,
                    "has_spotify": bool(s.spotify_device_id)
                }
                for sid, s in self.speakers.items()
            },
            "groups": {
                gid: {
                    "name": g.name,
                    "type": g.group_type.value,
                    "speaker_count": len(g.speaker_ids)
                }
                for gid, g in self.groups.items()
            },
            "default_speaker": self.get_default_speaker().id if self.get_default_speaker() else None
        }

    def list_available_targets(self) -> List[str]:
        """Listar todos los destinos disponibles para comandos de voz"""
        targets = []

        # Speakers individuales
        for speaker in self.speakers.values():
            targets.append(speaker.name)
            if speaker.room:
                targets.append(speaker.room)
            targets.extend(speaker.aliases)

        # Grupos
        for group in self.groups.values():
            targets.append(group.name)
            targets.extend(group.aliases)

        # Deduplicate y ordenar
        return sorted(list(set(targets)))
