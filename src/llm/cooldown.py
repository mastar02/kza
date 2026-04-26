"""
CooldownManager: backoff exponencial 1m → 5m → 25m → 1h por endpoint.

Patrón en OpenClaw model-failover.md. Cooldown se persiste a disco para
sobrevivir reinicios del proceso (si el endpoint estaba caído al apagar,
queremos seguir respetando su cooldown al reiniciar).
"""

from __future__ import annotations

import json
import logging
import time
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

    def is_available(self, endpoint_id: str) -> bool:
        """¿Se puede intentar este endpoint ahora?"""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                return True
            return time.time() >= state.cooldown_until

    def next_attempt_at(self, endpoint_id: str) -> float:
        """Epoch seconds del próximo intento permitido. 0 si está disponible."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                return 0.0
            now = time.time()
            return state.cooldown_until if state.cooldown_until > now else 0.0

    def get_state(self, endpoint_id: str) -> CooldownState:
        """Estado actual (crea uno fresh si no existía)."""
        with self._lock:
            return self._states.get(endpoint_id, CooldownState(endpoint_id=endpoint_id))

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

    def record_success(self, endpoint_id: str) -> None:
        """Resetear contador y limpiar cooldown."""
        with self._lock:
            state = self._states.get(endpoint_id)
            if state is None:
                state = CooldownState(endpoint_id=endpoint_id)
                self._states[endpoint_id] = state

            state.error_count = 0
            state.cooldown_until = 0.0
            state.last_used = time.time()
            state.last_error_kind = None

            self._save_to_disk()

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

    def _save_to_disk(self) -> None:
        """Atomic write (write temp + rename)."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persistence_path.with_suffix(".tmp")
            payload = {"states": [s.to_dict() for s in self._states.values()]}
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.persistence_path)
        except OSError as e:
            logger.error(f"[Cooldown] Failed to save {self.persistence_path}: {e}")
