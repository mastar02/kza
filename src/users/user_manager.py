"""
User Manager Module
Gestiona usuarios, sus perfiles de voz y permisos.

Jerarquía de permisos:
- ADMIN: Control total (crear usuarios, rutinas, configuración)
- ADULT: Control de dispositivos y rutinas personales
- TEEN: Control limitado (luces, música, NO seguridad)
- CHILD: Solo consultas y dispositivos específicos
- GUEST: Solo consultas, sin control
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """Niveles de permiso jerárquicos"""
    GUEST = 0      # Solo consultas
    CHILD = 1      # Consultas + dispositivos limitados
    TEEN = 2       # + Más dispositivos, sin seguridad
    ADULT = 3      # + Todo excepto admin
    ADMIN = 4      # Control total


# Permisos por nivel para diferentes dominios/acciones
PERMISSION_MATRIX = {
    # Dominio -> nivel mínimo requerido
    "domains": {
        "light": PermissionLevel.CHILD,
        "switch": PermissionLevel.CHILD,
        "media_player": PermissionLevel.CHILD,
        "fan": PermissionLevel.CHILD,
        "cover": PermissionLevel.TEEN,
        "climate": PermissionLevel.TEEN,
        "vacuum": PermissionLevel.TEEN,
        "lock": PermissionLevel.ADULT,
        "alarm_control_panel": PermissionLevel.ADULT,
        "camera": PermissionLevel.ADULT,
    },
    # Acciones especiales -> nivel mínimo
    "actions": {
        "create_routine": PermissionLevel.TEEN,
        "delete_routine": PermissionLevel.ADULT,
        "sync_commands": PermissionLevel.ADULT,
        "add_user": PermissionLevel.ADMIN,
        "remove_user": PermissionLevel.ADMIN,
        "change_permissions": PermissionLevel.ADMIN,
        "system_config": PermissionLevel.ADMIN,
    },
    # Entidades específicas que requieren permisos especiales
    "restricted_entities": {
        # entity_id -> nivel mínimo
        # Ejemplo: "lock.front_door": PermissionLevel.ADMIN
    }
}


@dataclass
class User:
    """Representa un usuario del sistema"""
    user_id: str
    name: str
    permission_level: PermissionLevel
    voice_embedding: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_active: bool = True

    # Configuración personal
    preferences: dict = field(default_factory=dict)
    allowed_entities: list[str] = field(default_factory=list)  # Lista blanca adicional
    blocked_entities: list[str] = field(default_factory=list)  # Lista negra

    def to_dict(self) -> dict:
        """Serializar a diccionario"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "permission_level": self.permission_level.value,
            "voice_embedding": self.voice_embedding.tolist() if self.voice_embedding is not None else None,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "is_active": self.is_active,
            "preferences": self.preferences,
            "allowed_entities": self.allowed_entities,
            "blocked_entities": self.blocked_entities
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Deserializar desde diccionario"""
        embedding = data.get("voice_embedding")
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float32)

        return cls(
            user_id=data["user_id"],
            name=data["name"],
            permission_level=PermissionLevel(data["permission_level"]),
            voice_embedding=embedding,
            created_at=data.get("created_at", time.time()),
            last_seen=data.get("last_seen", time.time()),
            is_active=data.get("is_active", True),
            preferences=data.get("preferences", {}),
            allowed_entities=data.get("allowed_entities", []),
            blocked_entities=data.get("blocked_entities", [])
        )


@dataclass
class PermissionCheckResult:
    """Resultado de verificación de permisos"""
    allowed: bool
    reason: str
    required_level: PermissionLevel
    user_level: PermissionLevel


class UserManager:
    """
    Gestiona usuarios y sus permisos.

    Funcionalidades:
    - CRUD de usuarios
    - Verificación de permisos
    - Almacenamiento persistente
    - Embeddings de voz para identificación
    """

    def __init__(self, data_path: str = "./data/users.json"):
        self.data_path = Path(data_path)
        self._users: dict[str, User] = {}
        self._version: int = 0  # Incrementa cuando usuarios/embeddings cambian (para cache)
        self._load()

    def _load(self):
        """Cargar usuarios desde archivo"""
        if self.data_path.exists():
            try:
                with open(self.data_path) as f:
                    data = json.load(f)

                for user_data in data.get("users", []):
                    user = User.from_dict(user_data)
                    self._users[user.user_id] = user

                logger.info(f"Cargados {len(self._users)} usuarios")

            except Exception as e:
                logger.error(f"Error cargando usuarios: {e}")

    def _save(self):
        """Guardar usuarios a archivo"""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        # Incrementar versión para invalidar caches de embeddings
        self._version += 1

        data = {
            "users": [user.to_dict() for user in self._users.values()],
            "last_updated": time.time()
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    # ==================== CRUD ====================

    def add_user(
        self,
        name: str,
        permission_level: PermissionLevel,
        voice_embedding: np.ndarray | None = None,
        requesting_user: User | None = None
    ) -> tuple[User | None, str]:
        """
        Agregar nuevo usuario.

        Args:
            name: Nombre del usuario
            permission_level: Nivel de permisos
            voice_embedding: Embedding de voz (puede agregarse después)
            requesting_user: Usuario que solicita la acción (para verificar permisos)

        Returns:
            (User creado o None, mensaje de resultado)
        """
        # Verificar permisos del solicitante
        if requesting_user is not None:
            check = self.check_action_permission(requesting_user, "add_user")
            if not check.allowed:
                return None, f"Permiso denegado: {check.reason}"

        # Generar ID único
        user_id = f"user_{name.lower().replace(' ', '_')}_{int(time.time())}"

        # Verificar que no exista usuario con mismo nombre
        existing = self.get_user_by_name(name)
        if existing:
            return None, f"Ya existe un usuario llamado {name}"

        user = User(
            user_id=user_id,
            name=name,
            permission_level=permission_level,
            voice_embedding=voice_embedding
        )

        self._users[user_id] = user
        self._save()

        logger.info(f"Usuario creado: {name} (nivel: {permission_level.name})")
        return user, f"Usuario {name} creado exitosamente"

    def remove_user(
        self,
        user_id: str,
        requesting_user: User | None = None
    ) -> tuple[bool, str]:
        """Eliminar usuario"""
        if requesting_user is not None:
            check = self.check_action_permission(requesting_user, "remove_user")
            if not check.allowed:
                return False, f"Permiso denegado: {check.reason}"

        if user_id not in self._users:
            return False, "Usuario no encontrado"

        user = self._users[user_id]

        # No permitir eliminar al último admin
        if user.permission_level == PermissionLevel.ADMIN:
            admin_count = sum(
                1 for u in self._users.values()
                if u.permission_level == PermissionLevel.ADMIN and u.is_active
            )
            if admin_count <= 1:
                return False, "No puedes eliminar al único administrador"

        del self._users[user_id]
        self._save()

        logger.info(f"Usuario eliminado: {user.name}")
        return True, f"Usuario {user.name} eliminado"

    def get_user(self, user_id: str) -> User | None:
        """Obtener usuario por ID"""
        return self._users.get(user_id)

    def get_user_by_name(self, name: str) -> User | None:
        """Obtener usuario por nombre"""
        name_lower = name.lower()
        for user in self._users.values():
            if user.name.lower() == name_lower:
                return user
        return None

    def get_all_users(self, active_only: bool = True) -> list[User]:
        """Obtener todos los usuarios"""
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.is_active]
        return users

    def update_user(self, user_id: str, **updates) -> tuple[bool, str]:
        """Actualizar datos de usuario"""
        if user_id not in self._users:
            return False, "Usuario no encontrado"

        user = self._users[user_id]

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        self._save()
        return True, "Usuario actualizado"

    # ==================== Voice Embeddings ====================

    def set_voice_embedding(
        self,
        user_id: str,
        embedding: np.ndarray
    ) -> bool:
        """Configurar embedding de voz para un usuario"""
        if user_id not in self._users:
            return False

        self._users[user_id].voice_embedding = embedding
        self._save()

        logger.info(f"Voice embedding actualizado para: {self._users[user_id].name}")
        return True

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Obtener todos los embeddings de voz registrados"""
        return {
            user_id: user.voice_embedding
            for user_id, user in self._users.items()
            if user.voice_embedding is not None and user.is_active
        }

    def update_last_seen(self, user_id: str):
        """Actualizar última vez visto"""
        if user_id in self._users:
            self._users[user_id].last_seen = time.time()
            # No guardar cada vez para no sobrecargar I/O

    # ==================== Permisos ====================

    def check_entity_permission(
        self,
        user: User,
        entity_id: str,
        service: str
    ) -> PermissionCheckResult:
        """
        Verificar si un usuario puede controlar una entidad.

        Args:
            user: Usuario que intenta la acción
            entity_id: ID de la entidad (e.g., light.living_room)
            service: Servicio a ejecutar (e.g., turn_on)

        Returns:
            PermissionCheckResult
        """
        domain = entity_id.split(".")[0]

        # 1. Verificar lista negra del usuario
        if entity_id in user.blocked_entities:
            return PermissionCheckResult(
                allowed=False,
                reason=f"Entidad bloqueada para {user.name}",
                required_level=PermissionLevel.ADMIN,
                user_level=user.permission_level
            )

        # 2. Verificar lista blanca del usuario (bypass de permisos)
        if entity_id in user.allowed_entities:
            return PermissionCheckResult(
                allowed=True,
                reason="Entidad permitida específicamente",
                required_level=PermissionLevel.GUEST,
                user_level=user.permission_level
            )

        # 3. Verificar entidades restringidas globalmente
        if entity_id in PERMISSION_MATRIX["restricted_entities"]:
            required = PERMISSION_MATRIX["restricted_entities"][entity_id]
            allowed = user.permission_level >= required

            return PermissionCheckResult(
                allowed=allowed,
                reason=f"Entidad restringida (requiere {required.name})" if not allowed else "OK",
                required_level=required,
                user_level=user.permission_level
            )

        # 4. Verificar por dominio
        required = PERMISSION_MATRIX["domains"].get(domain, PermissionLevel.ADULT)
        allowed = user.permission_level >= required

        return PermissionCheckResult(
            allowed=allowed,
            reason=f"Dominio {domain} requiere nivel {required.name}" if not allowed else "OK",
            required_level=required,
            user_level=user.permission_level
        )

    def check_action_permission(
        self,
        user: User,
        action: str
    ) -> PermissionCheckResult:
        """
        Verificar si un usuario puede ejecutar una acción especial.

        Args:
            user: Usuario
            action: Acción (e.g., "create_routine", "add_user")

        Returns:
            PermissionCheckResult
        """
        required = PERMISSION_MATRIX["actions"].get(action, PermissionLevel.ADULT)
        allowed = user.permission_level >= required

        return PermissionCheckResult(
            allowed=allowed,
            reason=f"Acción {action} requiere nivel {required.name}" if not allowed else "OK",
            required_level=required,
            user_level=user.permission_level
        )

    def can_control(self, user: User, entity_id: str, service: str = "turn_on") -> bool:
        """Atajo para verificar permiso de control"""
        return self.check_entity_permission(user, entity_id, service).allowed

    def can_perform_action(self, user: User, action: str) -> bool:
        """Atajo para verificar permiso de acción"""
        return self.check_action_permission(user, action).allowed

    # ==================== Utilidades ====================

    def format_permission_denied_message(
        self,
        user: User,
        check: PermissionCheckResult
    ) -> str:
        """Generar mensaje amigable de permiso denegado"""
        level_names_es = {
            PermissionLevel.GUEST: "invitado",
            PermissionLevel.CHILD: "niño",
            PermissionLevel.TEEN: "adolescente",
            PermissionLevel.ADULT: "adulto",
            PermissionLevel.ADMIN: "administrador"
        }

        required_es = level_names_es.get(check.required_level, "desconocido")

        return f"Lo siento {user.name}, esta acción requiere permisos de {required_es}."

    def get_stats(self) -> dict:
        """Obtener estadísticas de usuarios"""
        active_users = [u for u in self._users.values() if u.is_active]
        with_voice = [u for u in active_users if u.voice_embedding is not None]

        by_level = {}
        for level in PermissionLevel:
            count = sum(1 for u in active_users if u.permission_level == level)
            if count > 0:
                by_level[level.name] = count

        return {
            "total_users": len(self._users),
            "active_users": len(active_users),
            "users_with_voice": len(with_voice),
            "by_permission_level": by_level
        }

    def create_default_admin(self, name: str = "Admin") -> User:
        """Crear usuario admin por defecto si no existe ninguno"""
        admins = [u for u in self._users.values() if u.permission_level == PermissionLevel.ADMIN]

        if admins:
            return admins[0]

        user, _ = self.add_user(
            name=name,
            permission_level=PermissionLevel.ADMIN
        )

        logger.info(f"Admin por defecto creado: {name}")
        return user
