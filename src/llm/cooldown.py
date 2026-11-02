"""
CooldownManager: backoff exponencial 1m → 5m → 25m → 1h por endpoint.

Patrón en OpenClaw model-failover.md. Cooldown se persiste a disco para
sobrevivir reinicios del proceso (si el endpoint estaba caído al apagar,
queremos seguir respetando su cooldown al reiniciar).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import json
import logging
import time
import os
import tempfile
from pathlib import Path
from threading import Lock

from src.llm.types import CooldownState, ErrorKind

logger = logging.getLogger(__name__)


# Backoff: 1min, 5min, 25min, 1h (cap)
BACKOFF_SCHEDULE_S = (60, 300, 1500, 3600)

# Para BILLING saltamos directo al máximo (no es transitorio)
BILLING_COOLDOWN_S = 3600


class CooldownManager:
    """Maneja cooldowns por endpoint con backoff exponencial."""

    def __init__(self, persistence_path: Path | str):
        self.persistence_path = Path(persistence_path)
        self._states: dict[str, CooldownState] = {}
        self._lock = Lock()
        self._load_from_disk()

    def is_available(self, endpoint_id: str, *, before_read: Callable[[], None] | None = None) -> bool:
        """Read availability only after the original guard acquires state exclusion."""
        with self._lock:
            if before_read is not None and before_read() is False:
                raise PermissionError("cooldown_authority_retired")
            state = self._states.get(endpoint_id)
            return state is None or time.time() >= state.cooldown_until

    def next_attempt_at(self, endpoint_id: str, *, before_read: Callable[[], None] | None = None) -> float:
        """Read the retry deadline under the original projection authority."""
        with self._lock:
            if before_read is not None and before_read() is False:
                raise PermissionError("cooldown_authority_retired")
            state = self._states.get(endpoint_id)
            now = time.time()
            return state.cooldown_until if state is not None and state.cooldown_until > now else 0.0

    def get_state(self, endpoint_id: str, *, before_read: Callable[[], None] | None = None) -> CooldownState:
        """Return a detached state after any original projection guard acquires the lock."""
        with self._lock:
            if before_read is not None and before_read() is False:
                raise PermissionError("cooldown_authority_retired")
            result = deepcopy(self._states.get(endpoint_id, CooldownState(endpoint_id=endpoint_id)))
            if before_read is not None and before_read() is False:
                raise PermissionError("cooldown_authority_retired")
            return result

    def record_failure(self, endpoint_id: str, kind: ErrorKind) -> None:
        """Registrar fallo y aplicar backoff según kind + error_count previo."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                state = CooldownState(endpoint_id=endpoint_id)
                self._states[endpoint_id] = state

            now = time.time()
            state.last_error_kind = kind

            if kind == ErrorKind.BILLING:
                cooldown_s = BILLING_COOLDOWN_S
            else:
                idx = min(state.error_count, len(BACKOFF_SCHEDULE_S) - 1)
                cooldown_s = BACKOFF_SCHEDULE_S[idx]

            state.error_count += 1
            state.cooldown_until = now + cooldown_s

            logger.warning(
                f"[Cooldown] {endpoint_id}: kind={kind.value} "
                f"count={state.error_count} cooldown={cooldown_s}s"
            )

            self._save_to_disk()

    def record_success(self, endpoint_id: str, *, before_save: Callable[[], None] | None = None) -> None:
        """Reset a copied state and publish it only after an authorized replacement."""
        with self._lock:
            if before_save is not None and before_save() is False:
                raise PermissionError("cooldown_authority_retired")
            candidate = deepcopy(self._states)
            state = candidate.setdefault(endpoint_id, CooldownState(endpoint_id=endpoint_id))
            state.error_count = 0
            state.cooldown_until = 0.0
            state.last_used = time.time()
            state.last_error_kind = None
            self._save_to_disk(candidate, before_save=before_save, strict=before_save is not None)
            self._states = candidate

    # ---- Persistence (file-based, simple) ----

    def _load_from_disk(self) -> None:
        """Cargar estado de disco sin crashear ante fichero corrupto."""
        if not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            for entry in data.get("states", []):
                state = CooldownState.from_dict(entry)
                self._states[state.endpoint_id] = state
            logger.info(f"[Cooldown] Loaded {len(self._states)} states from disk")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[Cooldown] Failed to load {self.persistence_path}: {e}")

    def _save_to_disk(self, states: dict[str, CooldownState] | None = None, *,
                      before_save: Callable[[], None] | None = None, strict: bool = False) -> None:
        """Replace a private snapshot; administrative denial must never be swallowed."""
        temporary = None
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            payload = {"states": [state.to_dict() for state in
                                 (self._states if states is None else states).values()]}
            descriptor, temporary = tempfile.mkstemp(prefix=".cooldown-", dir=self.persistence_path.parent)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if before_save is not None and before_save() is False:
                raise PermissionError("cooldown_authority_retired")
            os.replace(temporary, self.persistence_path)
        except PermissionError:
            if strict or before_save is not None:
                raise
            logger.error("[Cooldown] Persistence unavailable")
        except OSError:
            if strict:
                raise
            logger.error("[Cooldown] Persistence unavailable")
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
