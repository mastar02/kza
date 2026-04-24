"""
Metrics emitter — escribe eventos JSONL a archivo para Logstash → ES.

Diseño:
- JSON Lines por archivo (fácil para Logstash file input + codec json).
- Escritura síncrona con `open(..., "a")` por línea: simple, durable, suficiente
  para el volumen de KZA (~decenas de eventos/min).
- Rotación por tamaño: cuando el archivo supera `rotate_bytes`, se renombra a
  `<file>.1` (descartando el .1 previo) y se arranca uno nuevo. Rotación
  externa más sofisticada (logrotate, Filebeat) la puede reemplazar.
- Tipos de evento: `request` (comando procesado), `wake` (detección o rechazo
  de wake word), `error` (condición anómala del pipeline).

Los campos son los descritos en la propuesta de schema — los consumers
(Logstash + index template ES) esperan esta forma.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROTATE_BYTES_DEFAULT = 100 * 1024 * 1024  # 100 MB


class MetricsEmitter:
    """Emite eventos estructurados a un archivo JSONL.

    Todos los métodos son idempotentes y fail-safe: si el write falla (disco
    lleno, permisos), log a WARNING y seguimos. Nunca propaga excepciones al
    pipeline de voz.
    """

    def __init__(
        self,
        path: str = "/home/kza/logs/kza-metrics.jsonl",
        rotate_bytes: int = _ROTATE_BYTES_DEFAULT,
        service_name: str = "kza-voice",
        logstash_host: str | None = None,
        logstash_port: int = 5515,
    ):
        self.path = Path(path)
        self.rotate_bytes = rotate_bytes
        self.service_name = service_name
        self.logstash_host = logstash_host
        self.logstash_port = logstash_port
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._sock_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _iso_now(self) -> str:
        # ES acepta ISO 8601 con offset "+00:00"; usamos UTC con Z-sufijo que
        # también parsea y es más corto.
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + (
            f".{int(time.time() * 1000) % 1000:03d}Z"
        )

    def _maybe_rotate(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size >= self.rotate_bytes:
                bak = self.path.with_suffix(self.path.suffix + ".1")
                if bak.exists():
                    bak.unlink()
                self.path.rename(bak)
        except OSError as e:
            logger.warning(f"MetricsEmitter rotate failed: {e}")

    def _get_sock(self) -> socket.socket | None:
        """Lazy TCP socket al Logstash. Reconecta ante fallos."""
        if not self.logstash_host:
            return None
        if self._sock is not None:
            return self._sock
        try:
            s = socket.create_connection(
                (self.logstash_host, self.logstash_port), timeout=1.5,
            )
            s.settimeout(1.0)
            self._sock = s
            logger.info(
                f"MetricsEmitter Logstash TCP conectado "
                f"→ {self.logstash_host}:{self.logstash_port}"
            )
            return s
        except OSError as e:
            logger.warning(f"MetricsEmitter Logstash conexion falló: {e}")
            return None

    def _send_tcp(self, line: str) -> None:
        """Envía una línea al Logstash. Best-effort; reconecta si falla."""
        with self._sock_lock:
            s = self._get_sock()
            if s is None:
                return
            try:
                s.sendall((line + "\n").encode("utf-8"))
            except OSError as e:
                logger.warning(f"MetricsEmitter TCP send failed: {e} — reconnect")
                try:
                    s.close()
                except OSError:
                    pass
                self._sock = None

    def _write(self, doc: dict[str, Any]) -> None:
        doc.setdefault("@timestamp", self._iso_now())
        doc.setdefault("service", self.service_name)
        line = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        # 1. Disk (durable).
        with self._lock:
            try:
                self._maybe_rotate()
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.warning(f"MetricsEmitter write failed: {e}")
        # 2. Logstash TCP (best-effort, no bloquea).
        if self.logstash_host:
            self._send_tcp(line)

    def emit_request(
        self,
        *,
        user_id: str,
        zone_id: str | None,
        text: str,
        intent: str | None,
        path: str | None,
        success: bool,
        timings: dict[str, float],
        audio_duration_ms: float | None = None,
        entity_id: str | None = None,
        service: str | None = None,
        used_wake_text: bool = False,
        early_dispatch: bool = False,
    ) -> None:
        """Emite un evento por cada comando despachado (fast o slow path)."""
        doc: dict[str, Any] = {
            "event_type": "request",
            "user_id": user_id,
            "zone_id": zone_id,
            "text": text,
            "intent": intent,
            "path": path,
            "success": success,
            "used_wake_text": used_wake_text,
            "early_dispatch": early_dispatch,
        }
        if audio_duration_ms is not None:
            doc["audio_duration_ms"] = float(audio_duration_ms)
        if entity_id:
            doc["entity_id"] = entity_id
        if service:
            doc["service"] = service
        # Timings: aplanados en el top-level con sufijo _ms para queries fáciles.
        for k, v in (timings or {}).items():
            try:
                doc[f"{k}_ms" if not k.endswith("_ms") else k] = float(v)
            except (TypeError, ValueError):
                continue
        self._write(doc)

    def emit_wake(
        self,
        *,
        room_id: str,
        matched: bool,
        wake_word: str | None,
        matched_via: str,  # "exact" | "fuzzy" | "alias" | "rejected"
        text: str | None,
        audio_duration_ms: float,
        wake_stt_ms: float,
        fuzzy_ratio: float | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        """Emite un evento por cada transcripción de wake (matcheado o rechazado)."""
        doc: dict[str, Any] = {
            "event_type": "wake",
            "room_id": room_id,
            "matched": matched,
            "matched_via": matched_via,
            "text": text,
            "audio_duration_ms": float(audio_duration_ms),
            "wake_stt_ms": float(wake_stt_ms),
        }
        if wake_word:
            doc["wake_word"] = wake_word
        if fuzzy_ratio is not None:
            doc["fuzzy_ratio"] = float(fuzzy_ratio)
        if rejection_reason:
            doc["rejection_reason"] = rejection_reason
        self._write(doc)

    def emit_error(
        self,
        *,
        module: str,
        error_msg: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emite un evento de error (exception, timeout, OOM, etc.)."""
        doc: dict[str, Any] = {
            "event_type": "error",
            "module": module,
            "error_msg": error_msg[:500],
        }
        if context:
            doc["context"] = {k: v for k, v in context.items() if v is not None}
        self._write(doc)
