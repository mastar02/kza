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

from collections.abc import Callable
from copy import deepcopy
import json
import logging
import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from enum import IntEnum
from io import BufferedWriter, FileIO, TextIOWrapper, text_encoding
from pathlib import Path
from threading import RLock
from typing import TypeVar
from uuid import uuid4

import numpy as np

from src.core.account_metadata import AccountMetadata, ActiveAccounts, RemovalCheck, RemovalStatus
from src.memory import output_registration_v2 as output_registration


_CatalogResult = TypeVar("_CatalogResult")

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
            "preferences": deepcopy(self.preferences),
            "allowed_entities": deepcopy(self.allowed_entities),
            "blocked_entities": deepcopy(self.blocked_entities)
        }

    @classmethod
    def from_dict(cls, data: dict, *, include_voice: bool = True) -> "User":
        """Deserializar desde diccionario"""
        embedding = data.get("voice_embedding") if include_voice else None
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
            preferences=deepcopy(data.get("preferences", {})),
            allowed_entities=deepcopy(data.get("allowed_entities", [])),
            blocked_entities=deepcopy(data.get("blocked_entities", []))
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

    def __init__(self, data_path: str = "./data/users.json", *, voice_registry=None,
                 retirement_proof: Callable[[frozenset[str]], bool] | None = None):
        self._lock = RLock()
        self.legacy_path = Path(data_path)
        self.voice_registry = voice_registry
        self.voice_review_managed = voice_registry is not None
        self._retirement_proof = retirement_proof
        self.data_path = (self.legacy_path.with_name(self.legacy_path.stem+'_accounts_v1.json')
            if self.voice_review_managed and not self.legacy_path.stem.endswith('_accounts_v1') else self.legacy_path)
        self._users: dict[str, User] = {}
        self._load_error = False
        self._version: int = 0
        self._account_snapshot = ActiveAccounts(False, frozenset())
        self._load()

    def _load(self):
        """Publish only a completely validated catalog and its canonical membership."""
        with self._lock:
            source = self.data_path if self.data_path.exists() else self.legacy_path
            if not source.exists():
                self._users = {}
                self._load_error = False
                self._account_snapshot = ActiveAccounts(True, frozenset(), self._version)
                return
            candidate = {}
            try:
                with open(source) as stream:
                    data = json.load(stream)
                if self.voice_review_managed and source == self.data_path and data.get("schema") != "kza-users-accounts-v1":
                    raise ValueError("unverified_account_catalog")
                for row in data.get("users", []):
                    user = User.from_dict(row, include_voice=not self.voice_review_managed)
                    if user.user_id in candidate:
                        raise ValueError("duplicate_account_identity")
                    candidate[user.user_id] = user
                snapshot = self._account_state(candidate, self._version)
            except Exception as error:
                self._users = {}
                self._load_error = True
                self._account_snapshot = ActiveAccounts(False, frozenset(), self._version)
                logger.error("Account loading unavailable: %s", type(error).__name__)
                return
            self._users, self._account_snapshot = candidate, snapshot
            self._load_error = False
            logger.info("Cargados %s usuarios", len(candidate))

    @staticmethod
    def _account_state(users: dict[str, User], revision: int) -> ActiveAccounts:
        if any(type(user.user_id) is not str or identity != user.user_id
               or not 1 <= len(identity) <= 128 or any(ord(char) < 32 or ord(char) == 127 for char in identity)
               or type(user.name) is not str or type(user.permission_level) is not PermissionLevel
               or type(user.is_active) is not bool for identity, user in users.items()):
            raise ValueError("invalid_account_metadata")
        return ActiveAccounts(True, frozenset(identity for identity, user in users.items() if user.is_active is True), revision)

    def active_accounts(self) -> ActiveAccounts:
        """Return immutable canonical membership without locks, IO or voice material."""
        return self._account_snapshot

    def project_accounts(self, *, before_read: Callable[[], None] | None = None) -> tuple[AccountMetadata, ...]:
        """Project only administrative scalars under original read authority."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = tuple(AccountMetadata(user.user_id, user.name, user.permission_level.name.lower(), user.is_active)
                           for user in self._users.values())
            self._check_authority(before_read)
            return result

    def _require_writable(self):
        if self._load_error:
            raise RuntimeError('unverified_account_catalog')

    @staticmethod
    def _check_authority(before_save: Callable[[], None] | None) -> None:
        """Support raising validators and reject an explicitly false guard."""
        if before_save is not None and before_save() is False:
            raise PermissionError("user_catalog_authority_retired")

    def _mutate(self, edit: Callable[[dict[str, User]], tuple[bool, _CatalogResult]], *,
                before_save: Callable[[], None] | None = None) -> _CatalogResult:
        """Serialize copy, authority, persistence and detached result publication."""
        with self._lock:
            self._check_authority(before_save)
            self._require_writable()
            candidate = deepcopy(self._users)
            changed, result = edit(candidate)
            detached = deepcopy(result)
            if changed:
                self._save(candidate, before_save=before_save)
            return detached

    def _save(self, users: dict[str, User] | None = None, *,
              before_save: Callable[[], None] | None = None) -> None:
        """Require durable credential retirement before destructive catalog publication."""
        with self._lock:
            self._check_authority(before_save)
            self._require_writable()
            candidate = deepcopy(self._users if users is None else users)
            snapshot = self._account_state(candidate, self._version + 1)
            retired = frozenset(identity for identity, user in self._users.items()
                if identity not in candidate or user.is_active is True and candidate[identity].is_active is False)
            data = {"users": [user.to_dict() for user in candidate.values()],
                    "last_updated": time.time()}
            if self.voice_review_managed:
                data["schema"] = "kza-users-accounts-v1"
                for row in data["users"]:
                    row["voice_embedding"] = None
            # A managed publisher is admitted (and its extent reserved) before
            # any directory or temporary allocation; unmanaged use returns None.
            extent = output_registration.before_account_save(self, before_save)
            temporary = own = parent = final = pin = None
            published = False
            try:
                self.data_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                parent = self._directory_identity(self.data_path.parent)
                descriptor, temporary = tempfile.mkstemp(prefix=".accounts-", dir=self.data_path.parent)
                try:
                    pin = self._pin_temporary(descriptor)
                except BaseException:
                    # Unpinned: own stays None, so the name is never unlinked.
                    self._close_raw_descriptor(descriptor)
                    raise
                try:
                    raw = FileIO(descriptor, "w", closefd=True)
                except BaseException:
                    # Not transferred: this publisher still owns the raw descriptor.
                    own = self._close_raw_descriptor(descriptor)
                    raise
                # From here the FileIO object owns the descriptor; never os.close it.
                try:
                    # Same text semantics as the original os.fdopen(descriptor, "w").
                    stream = TextIOWrapper(BufferedWriter(raw), encoding=text_encoding(None))
                except BaseException:
                    own = self._close_raw_file(raw)
                    raise
                finally:
                    raw = None
                with stream:
                    # From here only the original file object closes the descriptor.
                    own = self._owned_identity(stream)
                    json.dump(data, stream, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                    final = self._owned_final(stream, own)
                self._check_authority(before_save)
                if retired and (self._retirement_proof is None or self._retirement_proof(retired) is not True):
                    raise PermissionError("account_retirement_required")
                if retired and self.voice_registry is not None:
                    try:
                        cleared = self.voice_registry.ledger.people.user_links_cleared(retired)
                    except Exception:
                        cleared = False
                    if cleared is not True:
                        raise PermissionError("account_voice_links_retirement_required")
                if extent is not None:
                    extent.before_replace(temporary, own, parent, final)
                self._check_authority(before_save)
                os.replace(temporary, self.data_path)
                # Approved replacement must publish without another fallible check.
                self._users, self._account_snapshot = candidate, snapshot
                self._version = snapshot.revision
                published = True
                if extent is not None:
                    extent.replaced(own, parent)
            finally:
                proven = False
                try:
                    proven = temporary is None or self._remove_own_temporary(temporary, own, parent, pin)
                finally:
                    try:
                        if pin is not None:
                            os.close(pin)
                    finally:
                        if extent is not None:
                            extent.finish(published=published, cleanup_proven=proven)

    @staticmethod
    def _directory_identity(path: Path) -> tuple[int, int] | None:
        info = os.lstat(path)
        return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None

    @staticmethod
    def _close_raw_descriptor(descriptor: int) -> tuple[int, int] | None:
        """Observe, then close exactly once, a raw descriptor still owned by _save."""
        own = None
        try:
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                own = (info.st_dev, info.st_ino)
        except OSError:
            own = None
        finally:
            # A close failure leaves the name unproven; never retry a reusable fd.
            os.close(descriptor)
        return own

    @staticmethod
    def _close_raw_file(raw: FileIO) -> tuple[int, int] | None:
        """Observe, then close through its owning FileIO, a not yet wrapped file."""
        own = None
        try:
            info = os.fstat(raw.fileno())
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                own = (info.st_dev, info.st_ino)
        except OSError:
            own = None
        finally:
            raw.close()
        return own

    @staticmethod
    def _owned_identity(stream) -> tuple[int, int]:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("account_temporary_changed")
        return info.st_dev, info.st_ino

    @staticmethod
    def _owned_final(stream, own: tuple[int, int]) -> tuple[int, int]:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != own):
            raise OSError("account_temporary_changed")
        return info.st_size, info.st_mtime_ns

    @staticmethod
    def _pin_temporary(descriptor: int) -> int:
        """Hold the own inode open: its number cannot be reused while the pin lives.

        ext4 hands a freed inode number to the next allocation at once, so a
        (st_dev, st_ino) pair names the own temporary only while it is pinned.
        """
        return os.dup(descriptor)

    @staticmethod
    def _is_pinned(info: os.stat_result, pin: int | None) -> bool:
        """The name's lstat is the pinned own inode, unchanged since it was written."""
        if pin is None:
            return False
        held = os.fstat(pin)
        return ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                == (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns, held.st_ctime_ns))

    @classmethod
    def _remove_own_temporary(cls, temporary: str, own: tuple[int, int] | None,
                              parent: tuple[int, int] | None, pin: int | None) -> bool:
        """Unlink only the exact own pinned allocation; a foreign or unproven name stays."""
        try:
            info = os.lstat(temporary)
        except FileNotFoundError:
            return True
        if (own is None or parent is None or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not cls._is_pinned(info, pin)
                or cls._directory_identity(Path(temporary).parent) != parent):
            logger.warning("Account temporary cleanup unproven; name retained")
            return False
        os.unlink(temporary)
        return True

    # ==================== CRUD ====================

    def add_user(
        self, name: str, permission_level: PermissionLevel,
        voice_embedding: np.ndarray | None = None, requesting_user: User | None = None, *,
        before_save: Callable[[], None] | None = None,
    ) -> tuple[User | None, str]:
        """Create an account only after its copied catalog snapshot commits."""
        def edit(users):
            if self.voice_review_managed and voice_embedding is not None:
                return False, (None, "La voz se revisa en el panel; no crea cuentas ni permisos.")
            if requesting_user is not None:
                requester = users.get(requesting_user.user_id)
                if requester is None or not requester.is_active:
                    return False, (None, "Permiso denegado: solicitante no disponible")
                check = self.check_action_permission(requester, "add_user")
                if not check.allowed:
                    return False, (None, f"Permiso denegado: {check.reason}")
            if any(user.name.lower() == name.lower() for user in users.values()):
                return False, (None, f"Ya existe un usuario llamado {name}")
            user_id = f"user_{uuid4().hex}"
            user = User(user_id=user_id, name=name, permission_level=PermissionLevel(permission_level),
                        voice_embedding=deepcopy(voice_embedding))
            users[user_id] = user
            return True, (user, f"Usuario {name} creado exitosamente")
        return self._mutate(edit, before_save=before_save)

    def _removal_check(self, users: dict[str, User], user_id: str,
                       requesting_user: User | None = None) -> RemovalCheck:
        """Apply the same canonical checks before retirement and final mutation."""
        if requesting_user is not None:
            requester = users.get(requesting_user.user_id)
            if requester is None or not requester.is_active:
                return RemovalCheck(RemovalStatus.PROTECTED, "Permiso denegado: solicitante no disponible")
            check = self.check_action_permission(requester, "remove_user")
            if not check.allowed:
                return RemovalCheck(RemovalStatus.PROTECTED, f"Permiso denegado: {check.reason}")
        user = users.get(user_id)
        if user is None:
            return RemovalCheck(RemovalStatus.MISSING, "Usuario no encontrado")
        if user.permission_level == PermissionLevel.ADMIN and user.is_active:
            admins = sum(u.permission_level == PermissionLevel.ADMIN and u.is_active for u in users.values())
            if admins <= 1:
                return RemovalCheck(RemovalStatus.PROTECTED, "No puedes eliminar al único administrador")
        return RemovalCheck(RemovalStatus.ALLOWED, f"Usuario {user.name} eliminado")

    def check_removal(self, user_id: str, requesting_user: User | None = None, *,
                      before_read: Callable[[], None] | None = None) -> RemovalCheck:
        """Check deletion without projecting voice vectors or account preferences."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = self._removal_check(self._users, user_id, requesting_user)
            self._check_authority(before_read)
            return result

    def remove_user(self, user_id: str, requesting_user: User | None = None, *,
                    before_save: Callable[[], None] | None = None) -> tuple[bool, str]:
        """Delete a catalog row only after durable credential retirement."""
        output_registration.before_account_remove(self, user_id, requesting_user, before_save)
        def edit(users):
            result = self._removal_check(users, user_id, requesting_user)
            if result.status is not RemovalStatus.ALLOWED:
                return False, (False, result.message)
            del users[user_id]
            return True, (True, result.message)
        return self._mutate(edit, before_save=before_save)

    def get_user(self, user_id: str, *, before_read: Callable[[], None] | None = None) -> User | None:
        """Return a detached account after checking authority inside the catalog lock."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = deepcopy(self._users.get(user_id))
            self._check_authority(before_read)
            return result

    def get_user_by_name(self, name: str, *, before_read: Callable[[], None] | None = None) -> User | None:
        """Resolve a name without exposing a live catalog object."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = next((deepcopy(user) for user in self._users.values()
                           if user.name.lower() == name.lower()), None)
            self._check_authority(before_read)
            return result

    def get_all_users(self, active_only: bool = True, *,
                      before_read: Callable[[], None] | None = None) -> list[User]:
        """Copy eligible accounts under the original projection authority."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = [deepcopy(user) for user in self._users.values() if not active_only or user.is_active]
            self._check_authority(before_read)
            return result

    def update_user(self, user_id: str, *, before_save: Callable[[], None] | None = None,
                    **updates) -> tuple[bool, str]:
        """Persist copied fields while keeping the canonical account ID stable."""
        def edit(users):
            if self.voice_review_managed and "voice_embedding" in updates:
                raise ValueError("voice_profile_review_required")
            user = users.get(user_id)
            if user is None:
                return False, (False, "Usuario no encontrado")
            allowed = User.__dataclass_fields__.keys() - {"user_id"}
            for key, value in updates.items():
                if key in allowed:
                    setattr(user, key, PermissionLevel(value) if key == "permission_level" else deepcopy(value))
            return True, (True, "Usuario actualizado")
        return self._mutate(edit, before_save=before_save)

    # ==================== Voice Embeddings ====================

    def set_voice_embedding(self, user_id: str, embedding: np.ndarray, *,
                            before_save: Callable[[], None] | None = None) -> bool:
        """Re-resolve a legacy account and persist its copied vector under authority."""
        def edit(users):
            if self.voice_review_managed or user_id not in users:
                return False, False
            users[user_id].voice_embedding = deepcopy(embedding)
            return True, True
        return self._mutate(edit, before_save=before_save)

    def get_all_embeddings(self, *, before_read: Callable[[], None] | None = None) -> dict[str, np.ndarray]:
        """Copy legacy vectors; managed voice remains available only through its ledger."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            result = {} if self.voice_review_managed else {
                identity: user.voice_embedding.copy() for identity, user in self._users.items()
                if user.voice_embedding is not None and user.is_active}
            self._check_authority(before_read)
            return result

    def update_last_seen(self, user_id: str, *, before_save: Callable[[], None] | None = None) -> None:
        """Publish a copied transient observation under the same catalog exclusion."""
        with self._lock:
            self._check_authority(before_save)
            if user_id not in self._users:
                return
            candidate = dict(self._users)
            user = deepcopy(candidate[user_id])
            user.last_seen = time.time()
            candidate[user_id] = user
            self._check_authority(before_save)
            self._users = candidate
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

    def get_stats(self, *, before_read: Callable[[], None] | None = None) -> dict:
        """Project counts from one guarded catalog snapshot."""
        with self._lock:
            self._check_authority(before_read)
            self._require_writable()
            active = [user for user in self._users.values() if user.is_active]
            by_level = {level.name: sum(user.permission_level == level for user in active)
                        for level in PermissionLevel}
            result = {"total_users": len(self._users), "active_users": len(active),
                      "users_with_voice": sum(user.voice_embedding is not None for user in active),
                      "by_permission_level": {key: count for key, count in by_level.items() if count}}
            self._check_authority(before_read)
            return result

    def create_default_admin(self, name: str = "Admin", *,
                             before_save: Callable[[], None] | None = None) -> User:
        """Return or create a detached initial account under the same writer lock."""
        with self._lock:
            self._check_authority(before_save)
            self._require_writable()
            admins = [user for user in self._users.values() if user.permission_level == PermissionLevel.ADMIN]
            if admins:
                return deepcopy(admins[0])
            user, _ = self.add_user(name=name, permission_level=PermissionLevel.ADMIN, before_save=before_save)
            if user is None:
                raise RuntimeError("default_admin_unavailable")
            return user
