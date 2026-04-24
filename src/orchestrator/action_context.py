"""
LastActionTracker — memoria de corto plazo por entity para manejo de comandos ambiguos.

Caso de uso: usuario dice "el escritorio" sin verbo. Si hace <60s apagamos la luz,
ahora el implícito es prenderla (toggle). Si no hay acción reciente, preguntar.

Ver Q6 del plan de ejecución (decisión C+B fallback).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class LastAction:
    entity_id: str
    service: str        # turn_on | turn_off
    timestamp: float
    data: dict | None = None


class LastActionTracker:
    """Cache por entity_id con TTL. Thread-safe (usado desde async pipeline)."""

    def __init__(self, ttl_seconds: float = 60.0, max_entries: int = 256):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._lock = threading.Lock()
        self._by_entity: dict[str, LastAction] = {}

    def record(self, entity_id: str, service: str, data: dict | None = None) -> None:
        with self._lock:
            self._by_entity[entity_id] = LastAction(entity_id, service, time.time(), data)
            if len(self._by_entity) > self._max:
                # Evict oldest
                oldest_key = min(self._by_entity, key=lambda k: self._by_entity[k].timestamp)
                self._by_entity.pop(oldest_key, None)

    def get_recent(self, entity_id: str) -> LastAction | None:
        """Retorna LastAction si existe y está dentro del TTL, None si no."""
        with self._lock:
            action = self._by_entity.get(entity_id)
            if action is None:
                return None
            if time.time() - action.timestamp > self._ttl:
                self._by_entity.pop(entity_id, None)
                return None
            return action

    def toggle_service(self, service: str) -> str:
        """turn_on ↔ turn_off. Identity para otros services (conservador)."""
        if service == "turn_on":
            return "turn_off"
        if service == "turn_off":
            return "turn_on"
        return service

    def clear(self) -> None:
        with self._lock:
            self._by_entity.clear()
