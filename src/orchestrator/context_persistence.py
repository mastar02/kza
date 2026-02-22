"""
Context Persistence Module
Persiste el contexto de conversación por usuario a disco.

Permite que KZA recuerde conversaciones previas después de reiniciar,
manteniendo un historial mínimo por persona detectada.
"""

import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import asdict

from .context_manager import ContextManager, UserContext, ConversationTurn

logger = logging.getLogger(__name__)


class PersistentContextManager(ContextManager):
    """
    ContextManager con persistencia a disco.

    Guarda el contexto de cada usuario en archivos JSON individuales,
    permitiendo recuperar conversaciones después de reiniciar el sistema.

    Estructura:
        data/contexts/
        ├── user_abc123.json    # Contexto de usuario abc123
        ├── user_def456.json    # Contexto de usuario def456
        └── _metadata.json      # Estadísticas globales

    Ejemplo:
        manager = PersistentContextManager(
            storage_path="./data/contexts",
            max_history=10,               # Turnos en memoria
            persist_history=50,           # Turnos persistidos por usuario
            auto_save_interval=60         # Guardar cada 60 segundos
        )

        # Al crear/obtener contexto, se carga de disco si existe
        ctx = manager.get_or_create("user123", "Juan", "living_room")

        # El historial previo está disponible
        print(ctx.conversation_history)  # Incluye conversaciones anteriores
    """

    def __init__(
        self,
        storage_path: str = "./data/contexts",
        max_history: int = 10,          # Turnos en memoria activa
        persist_history: int = 50,      # Turnos a guardar por usuario
        inactive_timeout: float = 300,
        auto_save_interval: float = 60,
        system_prompt: str = None
    ):
        """
        Args:
            storage_path: Directorio para guardar contextos
            max_history: Máximo turnos en memoria activa
            persist_history: Máximo turnos a persistir por usuario
            inactive_timeout: Segundos sin actividad antes de guardar y limpiar RAM
            auto_save_interval: Intervalo de auto-guardado en segundos
            system_prompt: Prompt de sistema predeterminado
        """
        super().__init__(
            max_history=max_history,
            inactive_timeout=inactive_timeout,
            system_prompt=system_prompt
        )

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.persist_history = persist_history
        self.auto_save_interval = auto_save_interval

        # Thread de auto-guardado
        self._save_running = False
        self._save_thread: threading.Thread | None = None

        # Cargar metadata
        self._load_metadata()

        logger.info(f"PersistentContextManager inicializado: {storage_path}")

    def _get_user_file(self, user_id: str) -> Path:
        """Obtener path del archivo de un usuario"""
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"user_{safe_id}.json"

    def _metadata_file(self) -> Path:
        """Path del archivo de metadata"""
        return self.storage_path / "_metadata.json"

    def _load_metadata(self):
        """Cargar metadata global"""
        meta_file = self._metadata_file()
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self._total_contexts_created = meta.get("total_contexts_created", 0)
                    self._total_contexts_cleaned = meta.get("total_contexts_cleaned", 0)
            except Exception as e:
                logger.warning(f"Error cargando metadata: {e}")

    def _save_metadata(self):
        """Guardar metadata global"""
        meta = {
            "total_contexts_created": self._total_contexts_created,
            "total_contexts_cleaned": self._total_contexts_cleaned,
            "last_saved": time.time()
        }
        try:
            with open(self._metadata_file(), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")

    def start_auto_save(self):
        """Iniciar thread de auto-guardado"""
        if self._save_running:
            return

        self._save_running = True
        self._save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True,
            name="ContextAutoSave"
        )
        self._save_thread.start()
        logger.info(f"Auto-save iniciado (cada {self.auto_save_interval}s)")

    def stop_auto_save(self):
        """Detener thread de auto-guardado"""
        self._save_running = False
        if self._save_thread:
            self._save_thread.join(timeout=5)

    def _auto_save_loop(self):
        """Loop de auto-guardado"""
        while self._save_running:
            time.sleep(self.auto_save_interval)
            try:
                self.save_all()
            except Exception as e:
                logger.error(f"Error en auto-save: {e}")

    def get_or_create(
        self,
        user_id: str,
        user_name: str = None,
        zone_id: str = None,
        preferences: dict = None,
        permission_level: int = 0
    ) -> UserContext:
        """
        Obtener contexto existente (de memoria o disco) o crear uno nuevo.
        """
        with self._lock:
            # Primero verificar si está en memoria
            if user_id in self._contexts:
                ctx = self._contexts[user_id]
                if zone_id:
                    ctx.zone_id = zone_id
                ctx.last_active = time.time()
                return ctx

            # Intentar cargar de disco
            ctx = self._load_user_context(user_id)

            if ctx:
                # Actualizar datos si se proporcionaron
                if user_name:
                    ctx.user_name = user_name
                if zone_id:
                    ctx.zone_id = zone_id
                if preferences:
                    ctx.preferences.update(preferences)
                ctx.permission_level = permission_level
                ctx.last_active = time.time()

                # Poner en memoria
                self._contexts[user_id] = ctx
                logger.info(f"Contexto cargado de disco: {user_id} ({ctx.turns_count} turnos)")
                return ctx

            # Crear nuevo
            ctx = UserContext(
                user_id=user_id,
                user_name=user_name or f"Usuario_{user_id[:8]}",
                zone_id=zone_id,
                max_history=self.max_history,
                preferences=preferences or {},
                permission_level=permission_level
            )

            self._contexts[user_id] = ctx
            self._total_contexts_created += 1

            logger.debug(f"Contexto nuevo creado: {user_id}")
            return ctx

    def _load_user_context(self, user_id: str) -> UserContext | None:
        """Cargar contexto de usuario desde disco"""
        user_file = self._get_user_file(user_id)

        if not user_file.exists():
            return None

        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruir UserContext
            ctx = UserContext(
                user_id=data["user_id"],
                user_name=data.get("user_name", f"Usuario_{user_id[:8]}"),
                zone_id=data.get("zone_id"),
                max_history=self.max_history,
                preferences=data.get("preferences", {}),
                permission_level=data.get("permission_level", 0)
            )

            # Cargar historial (últimos N según max_history)
            history_data = data.get("conversation_history", [])
            for turn_data in history_data[-self.max_history:]:
                turn = ConversationTurn(
                    role=turn_data["role"],
                    content=turn_data["content"],
                    timestamp=turn_data.get("timestamp", time.time()),
                    intent=turn_data.get("intent"),
                    entities=turn_data.get("entities", [])
                )
                ctx.conversation_history.append(turn)

            # Restaurar estadísticas
            ctx.turns_count = data.get("total_turns", len(history_data))
            ctx.session_start = time.time()  # Nueva sesión

            return ctx

        except Exception as e:
            logger.error(f"Error cargando contexto {user_id}: {e}")
            return None

    def save_user_context(self, user_id: str) -> bool:
        """Guardar contexto de un usuario a disco"""
        with self._lock:
            ctx = self._contexts.get(user_id)
            if not ctx:
                return False

            return self._save_context_to_disk(ctx)

    def _save_context_to_disk(self, ctx: UserContext) -> bool:
        """Guardar un contexto específico a disco"""
        user_file = self._get_user_file(ctx.user_id)

        try:
            # Cargar historial existente para no perder turnos antiguos
            existing_history = []
            if user_file.exists():
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                        existing_history = existing.get("conversation_history", [])
                except Exception as e:
                    logger.warning(f"Failed to read existing context for {ctx.user_id}: {e}")

            # Combinar historial existente con nuevo (evitar duplicados)
            new_history = [t.to_dict() for t in ctx.conversation_history]

            # Usar timestamps para detectar turnos nuevos
            existing_timestamps = {h.get("timestamp", 0) for h in existing_history}

            for turn in new_history:
                if turn.get("timestamp", 0) not in existing_timestamps:
                    existing_history.append(turn)

            # Limitar a persist_history turnos
            combined_history = existing_history[-self.persist_history:]

            data = {
                "user_id": ctx.user_id,
                "user_name": ctx.user_name,
                "zone_id": ctx.zone_id,
                "preferences": ctx.preferences,
                "permission_level": ctx.permission_level,
                "conversation_history": combined_history,
                "total_turns": ctx.turns_count + len(existing_history) - len(ctx.conversation_history),
                "last_saved": time.time()
            }

            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error guardando contexto {ctx.user_id}: {e}")
            return False

    def save_all(self):
        """Guardar todos los contextos activos"""
        saved = 0
        with self._lock:
            for user_id, ctx in self._contexts.items():
                if self._save_context_to_disk(ctx):
                    saved += 1

        self._save_metadata()

        if saved > 0:
            logger.debug(f"Guardados {saved} contextos")

    def cleanup_inactive(self) -> int:
        """
        Limpiar contextos inactivos de memoria (pero mantener en disco).
        """
        cleaned = 0
        now = time.time()

        with self._lock:
            inactive_ids = [
                uid for uid, ctx in self._contexts.items()
                if (now - ctx.last_active) > self.inactive_timeout
            ]

            for uid in inactive_ids:
                # Guardar antes de eliminar de memoria
                ctx = self._contexts[uid]
                self._save_context_to_disk(ctx)

                del self._contexts[uid]
                cleaned += 1
                self._total_contexts_cleaned += 1

        if cleaned > 0:
            logger.info(f"Contextos movidos a disco: {cleaned}")

        return cleaned

    def get_all_known_users(self) -> list[dict]:
        """Obtener lista de todos los usuarios con contexto guardado"""
        users = []

        for user_file in self.storage_path.glob("user_*.json"):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    users.append({
                        "user_id": data["user_id"],
                        "user_name": data.get("user_name"),
                        "total_turns": data.get("total_turns", 0),
                        "last_saved": data.get("last_saved"),
                        "in_memory": data["user_id"] in self._contexts
                    })
            except Exception as e:
                logger.warning(f"Error leyendo {user_file}: {e}")

        return users

    def get_user_history_summary(self, user_id: str, last_n: int = 5) -> list[dict]:
        """Obtener resumen de las últimas N conversaciones de un usuario"""
        # Primero intentar de memoria
        with self._lock:
            if user_id in self._contexts:
                ctx = self._contexts[user_id]
                return [t.to_dict() for t in ctx.conversation_history[-last_n:]]

        # Si no está en memoria, cargar de disco
        user_file = self._get_user_file(user_id)
        if user_file.exists():
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history = data.get("conversation_history", [])
                    return history[-last_n:]
            except Exception as e:
                logger.warning(f"Failed to read history for {user_id}: {e}")

        return []

    def delete_user_context(self, user_id: str) -> bool:
        """Eliminar contexto de un usuario (memoria y disco)"""
        with self._lock:
            # Eliminar de memoria
            if user_id in self._contexts:
                del self._contexts[user_id]

            # Eliminar de disco
            user_file = self._get_user_file(user_id)
            if user_file.exists():
                try:
                    user_file.unlink()
                    logger.info(f"Contexto eliminado: {user_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error eliminando {user_file}: {e}")
                    return False

        return True

    def shutdown(self):
        """Apagar limpiamente: guardar todo y detener threads"""
        logger.info("Guardando contextos antes de apagar...")
        self.stop_auto_save()
        self.stop_cleanup_thread()
        self.save_all()
        logger.info("Contextos guardados")

    def get_stats(self) -> dict:
        """Obtener estadísticas extendidas"""
        base_stats = super().get_stats()

        # Contar archivos en disco
        disk_users = len(list(self.storage_path.glob("user_*.json")))

        base_stats.update({
            "disk_users": disk_users,
            "persist_history": self.persist_history,
            "storage_path": str(self.storage_path),
            "auto_save_interval": self.auto_save_interval
        })

        return base_stats
