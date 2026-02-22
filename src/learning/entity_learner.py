"""
Entity Learner - Aprendizaje Dinámico de Entidades
Aprende usuarios, dispositivos, habitaciones y alias de forma dinámica.

NO hay nada hardcodeado - todo se descubre o aprende:
- Usuarios: de voice enrollment y conversación
- Dispositivos: de Home Assistant
- Habitaciones: de Home Assistant + alias del usuario
- Alias: de la conversación natural
"""

import asyncio
import logging
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LearnedUser:
    """Usuario aprendido del sistema"""
    user_id: str
    name: str                          # Nombre como lo dice el usuario
    voice_profile_id: str | None = None
    ble_fingerprints: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # "papá", "mi esposo"
    preferences: dict = field(default_factory=dict)
    first_seen: datetime = None
    last_seen: datetime = None


@dataclass
class LearnedEntity:
    """Entidad de Home Assistant con alias aprendidos"""
    entity_id: str                     # "light.sala"
    domain: str                        # "light"
    friendly_name: str                 # Nombre de HA: "Luz Sala"
    area_id: str | None = None      # Área en HA
    area_name: str | None = None    # "Sala"
    aliases: list[str] = field(default_factory=list)  # ["luz del living", "la luz principal"]
    user_aliases: dict[str, list[str]] = field(default_factory=dict)  # Por usuario


@dataclass
class LearnedArea:
    """Área/habitación aprendida"""
    area_id: str
    name: str                          # Nombre oficial de HA
    aliases: list[str] = field(default_factory=list)  # ["el cuarto de los niños", "arriba"]
    entities: list[str] = field(default_factory=list)  # Entidades en esta área


class EntityLearner:
    """
    Sistema de aprendizaje dinámico de entidades.

    Aprende de múltiples fuentes:
    1. Home Assistant - descubrimiento automático
    2. Voice Enrollment - usuarios registrados
    3. Conversación - alias naturales del usuario
    4. Correcciones - cuando el usuario corrige a KZA
    """

    def __init__(
        self,
        ha_client=None,
        user_manager=None,
        data_dir: str = None
    ):
        self.ha = ha_client
        self.user_manager = user_manager
        self.data_dir = Path(data_dir) if data_dir else Path("data/learned")

        # Conocimiento aprendido
        self._users: dict[str, LearnedUser] = {}
        self._entities: dict[str, LearnedEntity] = {}
        self._areas: dict[str, LearnedArea] = {}

        # Índices para búsqueda rápida
        self._alias_to_entity: dict[str, str] = {}  # "luz del baño" -> "light.bathroom"
        self._alias_to_area: dict[str, str] = {}    # "arriba" -> "area.second_floor"
        self._alias_to_user: dict[str, str] = {}    # "papá" -> "user_gabriel"

        # Patrones de aprendizaje de conversación
        self._pending_learnings: list[dict] = []

        # Cargar datos
        self._load_data()

    def _load_data(self):
        """Cargar conocimiento persistido"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for data_type in ["users", "entities", "areas", "aliases"]:
            file_path = self.data_dir / f"{data_type}.json"
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        if data_type == "users":
                            self._users = {
                                k: LearnedUser(**v) for k, v in data.items()
                            }
                        elif data_type == "entities":
                            self._entities = {
                                k: LearnedEntity(**v) for k, v in data.items()
                            }
                        elif data_type == "areas":
                            self._areas = {
                                k: LearnedArea(**v) for k, v in data.items()
                            }
                        elif data_type == "aliases":
                            self._alias_to_entity = data.get("entities", {})
                            self._alias_to_area = data.get("areas", {})
                            self._alias_to_user = data.get("users", {})
                except Exception as e:
                    logger.warning(f"Error cargando {data_type}: {e}")

    def _save_data(self):
        """Persistir conocimiento"""
        try:
            # Usuarios
            with open(self.data_dir / "users.json", 'w') as f:
                json.dump({k: v.__dict__ for k, v in self._users.items()}, f, indent=2, default=str)

            # Entidades
            with open(self.data_dir / "entities.json", 'w') as f:
                json.dump({k: v.__dict__ for k, v in self._entities.items()}, f, indent=2)

            # Áreas
            with open(self.data_dir / "areas.json", 'w') as f:
                json.dump({k: v.__dict__ for k, v in self._areas.items()}, f, indent=2)

            # Alias
            with open(self.data_dir / "aliases.json", 'w') as f:
                json.dump({
                    "entities": self._alias_to_entity,
                    "areas": self._alias_to_area,
                    "users": self._alias_to_user
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Error guardando datos: {e}")

    # ==================== Sincronización con Home Assistant ====================

    async def sync_from_home_assistant(self):
        """
        Sincronizar entidades y áreas desde Home Assistant.
        Esta es la fuente principal de verdad para dispositivos.
        """
        if not self.ha:
            logger.warning("No hay cliente de Home Assistant configurado")
            return

        logger.info("Sincronizando con Home Assistant...")

        try:
            # Obtener todas las entidades
            states = await self.ha.get_states()

            for state in states:
                entity_id = state.get("entity_id", "")
                attributes = state.get("attributes", {})

                # Extraer información
                domain = entity_id.split(".")[0]
                friendly_name = attributes.get("friendly_name", entity_id)

                # Crear o actualizar entidad
                if entity_id not in self._entities:
                    self._entities[entity_id] = LearnedEntity(
                        entity_id=entity_id,
                        domain=domain,
                        friendly_name=friendly_name
                    )
                else:
                    self._entities[entity_id].friendly_name = friendly_name

                # Generar alias automáticos del nombre
                self._generate_auto_aliases(entity_id, friendly_name)

            # Obtener áreas
            if hasattr(self.ha, 'get_areas'):
                areas = await self.ha.get_areas()
                for area in areas:
                    area_id = area.get("area_id")
                    name = area.get("name")

                    if area_id not in self._areas:
                        self._areas[area_id] = LearnedArea(
                            area_id=area_id,
                            name=name
                        )

                    # Alias automáticos del área
                    self._add_area_alias(area_id, name.lower())

            self._save_data()
            logger.info(f"Sincronizadas {len(self._entities)} entidades y {len(self._areas)} áreas")

        except Exception as e:
            logger.error(f"Error sincronizando con HA: {e}")

    def _generate_auto_aliases(self, entity_id: str, friendly_name: str):
        """Generar alias automáticos de un nombre"""
        name_lower = friendly_name.lower()
        domain = entity_id.split(".")[0]

        # Alias directo del nombre
        self._add_entity_alias(entity_id, name_lower)

        # Sin el dominio ("luz sala" -> "sala" si es luz)
        domain_words = {
            "light": ["luz", "lámpara", "foco", "iluminación"],
            "switch": ["interruptor", "switch", "enchufe"],
            "climate": ["clima", "aire", "ac", "calefacción", "termostato"],
            "media_player": ["tv", "televisor", "bocina", "speaker", "sonos"],
            "cover": ["persiana", "cortina", "toldo"],
            "fan": ["ventilador", "abanico"],
            "lock": ["cerradura", "chapa"],
            "sensor": ["sensor"],
            "binary_sensor": ["sensor"],
        }

        # Remover palabras del dominio para crear alias cortos
        for word in domain_words.get(domain, []):
            if word in name_lower:
                short_name = name_lower.replace(word, "").strip()
                if short_name and len(short_name) > 2:
                    self._add_entity_alias(entity_id, short_name)

        # "luz de la sala" -> también matchea "luz sala" y "sala"
        clean_name = re.sub(r'\b(de la|del|de|la|el|los|las)\b', '', name_lower)
        clean_name = ' '.join(clean_name.split())  # Normalizar espacios
        if clean_name != name_lower:
            self._add_entity_alias(entity_id, clean_name)

    # ==================== Aprendizaje de Usuarios ====================

    def register_user(
        self,
        user_id: str,
        name: str,
        voice_profile_id: str = None,
        ble_fingerprint: str = None
    ):
        """
        Registrar un nuevo usuario.
        Llamado desde voice enrollment o manualmente.
        """
        if user_id not in self._users:
            self._users[user_id] = LearnedUser(
                user_id=user_id,
                name=name,
                first_seen=datetime.now()
            )

        user = self._users[user_id]
        user.name = name
        user.last_seen = datetime.now()

        if voice_profile_id:
            user.voice_profile_id = voice_profile_id

        if ble_fingerprint and ble_fingerprint not in user.ble_fingerprints:
            user.ble_fingerprints.append(ble_fingerprint)

        # Crear alias del nombre
        self._add_user_alias(user_id, name.lower())

        # Variantes comunes
        if " " in name:
            first_name = name.split()[0].lower()
            self._add_user_alias(user_id, first_name)

        self._save_data()
        logger.info(f"Usuario registrado: {name} ({user_id})")

    def add_user_alias(self, user_id: str, alias: str, context: str = None):
        """
        Agregar alias para un usuario.
        Ej: "papá", "mi esposo", "el jefe"
        """
        if user_id not in self._users:
            return

        alias_lower = alias.lower()
        self._users[user_id].aliases.append(alias_lower)
        self._add_user_alias(user_id, alias_lower)

        logger.info(f"Alias de usuario aprendido: '{alias}' -> {user_id}")
        self._save_data()

    # ==================== Aprendizaje de Conversación ====================

    def learn_from_utterance(
        self,
        text: str,
        user_id: str = None,
        resolved_entity: str = None,
        was_correction: bool = False
    ):
        """
        Aprender de lo que el usuario dice.

        Ejemplos:
        - "Enciende la luz del cuarto de mi hijo" + entity=light.kids_room
          → Aprende: "cuarto de mi hijo" es alias de "light.kids_room" o su área

        - "No, me refiero a la otra luz" (corrección)
          → Registra que el alias anterior estaba mal
        """
        text_lower = text.lower()

        # Extraer posibles alias del texto
        potential_aliases = self._extract_potential_aliases(text_lower)

        if resolved_entity and potential_aliases:
            # El usuario usó estos alias para referirse a esta entidad
            for alias in potential_aliases:
                if alias not in self._alias_to_entity:
                    # Nuevo alias aprendido
                    if user_id:
                        # Alias específico de este usuario
                        if resolved_entity in self._entities:
                            entity = self._entities[resolved_entity]
                            if user_id not in entity.user_aliases:
                                entity.user_aliases[user_id] = []
                            if alias not in entity.user_aliases[user_id]:
                                entity.user_aliases[user_id].append(alias)
                    else:
                        # Alias global
                        self._add_entity_alias(resolved_entity, alias)

                    logger.info(f"Alias aprendido de conversación: '{alias}' -> {resolved_entity}")

        if was_correction:
            # El usuario corrigió a KZA - registrar para no repetir el error
            self._pending_learnings.append({
                "text": text_lower,
                "user_id": user_id,
                "type": "correction",
                "timestamp": datetime.now().isoformat()
            })

        self._save_data()

    def _extract_potential_aliases(self, text: str) -> list[str]:
        """Extraer posibles alias del texto"""
        aliases = []

        # Patrones como "la luz de X", "el X del Y"
        patterns = [
            r"(?:la |el |las |los )?(?:luz|lámpara|aire|clima|tv|tele) (?:de(?:l)? )?(.+?)(?:\s*$|\s+(?:por favor|please))",
            r"(?:de(?:l)? |en (?:el |la )?)?(.+?)(?:\s*$)",
            r"cuarto de (?:mi |los |las )?(.+)",
            r"habitación de (?:mi |los |las )?(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                alias = match.group(1).strip()
                if alias and len(alias) > 2 and len(alias) < 30:
                    # Limpiar
                    alias = re.sub(r'\b(por favor|please|gracias)\b', '', alias).strip()
                    if alias:
                        aliases.append(alias)

        return aliases

    # ==================== Resolución de Entidades ====================

    def resolve_entity(
        self,
        text: str,
        user_id: str = None,
        domain: str = None
    ) -> str | None:
        """
        Resolver texto a entity_id.

        Args:
            text: Lo que dijo el usuario ("la luz del baño")
            user_id: Usuario actual (para alias personalizados)
            domain: Filtrar por dominio ("light", "climate", etc.)

        Returns:
            entity_id o None
        """
        text_lower = text.lower()

        # 1. Buscar en alias exactos
        if text_lower in self._alias_to_entity:
            entity_id = self._alias_to_entity[text_lower]
            if not domain or entity_id.startswith(f"{domain}."):
                return entity_id

        # 2. Buscar alias específicos del usuario
        if user_id:
            for entity_id, entity in self._entities.items():
                if domain and not entity_id.startswith(f"{domain}."):
                    continue
                if user_id in entity.user_aliases:
                    if text_lower in entity.user_aliases[user_id]:
                        return entity_id

        # 3. Búsqueda parcial en alias
        for alias, entity_id in self._alias_to_entity.items():
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            if text_lower in alias or alias in text_lower:
                return entity_id

        # 4. Buscar en friendly_name
        for entity_id, entity in self._entities.items():
            if domain and not entity_id.startswith(f"{domain}."):
                continue
            if text_lower in entity.friendly_name.lower():
                return entity_id

        return None

    def resolve_area(self, text: str) -> str | None:
        """Resolver texto a area_id"""
        text_lower = text.lower()

        # Buscar en alias
        if text_lower in self._alias_to_area:
            return self._alias_to_area[text_lower]

        # Búsqueda parcial
        for alias, area_id in self._alias_to_area.items():
            if text_lower in alias or alias in text_lower:
                return area_id

        return None

    def resolve_user(self, text: str) -> str | None:
        """Resolver texto a user_id"""
        text_lower = text.lower()

        if text_lower in self._alias_to_user:
            return self._alias_to_user[text_lower]

        # Buscar parcial
        for alias, user_id in self._alias_to_user.items():
            if text_lower in alias or alias in text_lower:
                return user_id

        return None

    def get_user_name(self, user_id: str) -> str | None:
        """Obtener nombre de usuario"""
        if user_id in self._users:
            return self._users[user_id].name
        return None

    def get_entity_name(self, entity_id: str) -> str:
        """Obtener nombre amigable de entidad"""
        if entity_id in self._entities:
            return self._entities[entity_id].friendly_name
        return entity_id.split(".")[-1].replace("_", " ")

    # ==================== Helpers ====================

    def _add_entity_alias(self, entity_id: str, alias: str):
        """Agregar alias para entidad"""
        alias_lower = alias.lower().strip()
        if alias_lower and len(alias_lower) > 1:
            self._alias_to_entity[alias_lower] = entity_id
            if entity_id in self._entities:
                if alias_lower not in self._entities[entity_id].aliases:
                    self._entities[entity_id].aliases.append(alias_lower)

    def _add_area_alias(self, area_id: str, alias: str):
        """Agregar alias para área"""
        alias_lower = alias.lower().strip()
        if alias_lower:
            self._alias_to_area[alias_lower] = area_id
            if area_id in self._areas:
                if alias_lower not in self._areas[area_id].aliases:
                    self._areas[area_id].aliases.append(alias_lower)

    def _add_user_alias(self, user_id: str, alias: str):
        """Agregar alias para usuario"""
        alias_lower = alias.lower().strip()
        if alias_lower:
            self._alias_to_user[alias_lower] = user_id

    # ==================== API ====================

    def get_all_users(self) -> list[dict]:
        """Obtener todos los usuarios"""
        return [
            {
                "user_id": u.user_id,
                "name": u.name,
                "aliases": u.aliases,
                "has_voice_profile": bool(u.voice_profile_id),
                "ble_devices": len(u.ble_fingerprints)
            }
            for u in self._users.values()
        ]

    def get_all_entities(self, domain: str = None) -> list[dict]:
        """Obtener todas las entidades"""
        entities = []
        for e in self._entities.values():
            if domain and e.domain != domain:
                continue
            entities.append({
                "entity_id": e.entity_id,
                "name": e.friendly_name,
                "domain": e.domain,
                "aliases": e.aliases,
                "area": e.area_name
            })
        return entities

    def get_all_areas(self) -> list[dict]:
        """Obtener todas las áreas"""
        return [
            {
                "area_id": a.area_id,
                "name": a.name,
                "aliases": a.aliases,
                "entity_count": len(a.entities)
            }
            for a in self._areas.values()
        ]

    def get_status(self) -> dict:
        """Obtener estado del sistema"""
        return {
            "users": len(self._users),
            "entities": len(self._entities),
            "areas": len(self._areas),
            "total_aliases": len(self._alias_to_entity) + len(self._alias_to_area) + len(self._alias_to_user)
        }
