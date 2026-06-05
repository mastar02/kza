"""AmbientGuard — compuerta acústica integral por habitación (spec 2026-06-05).

Con TV de fondo, el wake openwakeword dispara constantemente (0.4-0.9) y el
ambiente satura Whisper + LLM router ("no me escucha" = comandos reales
compitiendo contra la cola; 1 acción fantasma el 2026-06-04). Este guard
unifica TV-mode + circuit breaker en una escalera de 3 estados POR ROOM:

    NORMAL ──(rechazos de captura ≥ N en ventana)──► STRICT
    STRICT ──(rechazos persisten)──► COOLDOWN ──(expira)──► STRICT
    STRICT ──(quiet sostenido, histéresis)──► NORMAL

- NORMAL: comportamiento actual (threshold base del detector).
- STRICT: exige score de wake ≥ strict_wake_score (encima del threshold base
  del detector, que NO se muta) + RMS/SPENERGY mínimos si están calibrados.
  El follow_up queda deshabilitado (la cascada del 06-04 era follow_up
  siempre abierto). El bonus wake_acoustically_confirmed del grammar
  fast-path también se apaga (ver request_router).
- COOLDOWN: descarta toda captura por cooldown_duration_s. Garantiza que la
  cola del router NUNCA se satura, sea cual sea el estado de las señales
  acústicas (la señal de escalada es de software: capturas rechazadas).

Semántica de escalada (decisión de diseño, ver spec):
- Solo los RESULTADOS DE CAPTURA rechazados (noise/empty/timeout — ya
  gastaron Whisper/router) escalan. Los rechazos del propio guard en STRICT
  son gratis: no cuentan para COOLDOWN, pero refrescan el quiet timer
  (ambiente persiste → STRICT sigue vivo).
- accepted / other_fail (voz real con fallo downstream) no escalan.

Thread-safety: on_wake corre en el thread C de sounddevice;
on_capture_result en el event loop → lock interno. Reloj inyectable
(time_fn) para tests sin sleeps. enabled=False (default) = guard pasivo.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Outcomes de captura que evidencian ambiente hostil (gastaron pipeline y
# fueron basura). "accepted" y "other_fail" no escalan.
REJECT_OUTCOMES = ("noise", "empty", "timeout")


class GuardState(Enum):
    NORMAL = "normal"
    STRICT = "strict"
    COOLDOWN = "cooldown"


@dataclass
class AmbientGuardConfig:
    """Config del guard (settings.yaml → rooms.ambient_guard).

    Los umbrales acústicos (strict_wake_score / strict_min_rms /
    strict_min_spenergy) se calibran con tools/acoustic_calibration.py —
    NO adivinar (lección de las sesiones 05-31..06-04).
    """
    enabled: bool = False
    # Escalada NORMAL → STRICT: N capturas rechazadas dentro de la ventana.
    strict_entry_rejects: int = 4
    strict_entry_window_s: float = 60.0
    # Salida STRICT → NORMAL: histéresis de quiet (sin NINGÚN rechazo).
    strict_exit_quiet_s: float = 120.0
    # Compuertas en STRICT. 0.0 = no aplicar (señal no calibrada/muerta).
    strict_wake_score: float = 0.65
    strict_min_rms: float = 0.0
    strict_min_spenergy: float = 0.0
    # Escalada STRICT → COOLDOWN: N capturas rechazadas MÁS en la ventana.
    cooldown_entry_rejects: int = 6
    cooldown_entry_window_s: float = 60.0
    cooldown_duration_s: float = 30.0


@dataclass
class GuardDecision:
    accept: bool
    reason: str  # "ok" | "disabled" | "strict_score" | "strict_rms" | "strict_spenergy" | "cooldown"
    state: GuardState


@dataclass
class _RoomState:
    state: GuardState = GuardState.NORMAL
    capture_rejects: deque = field(default_factory=deque)  # timestamps
    last_reject_at: float = 0.0
    cooldown_until: float = 0.0


def classify_outcome(result: dict) -> str:
    """Mapea el dict resultado del RequestRouter a un outcome del guard.

    Returns:
        "accepted" | "empty" | "noise" | "timeout" | "other_fail"
    """
    if result.get("success"):
        return "accepted"
    text = (result.get("text") or "").strip()
    if not text:
        return "empty"
    intent = str(result.get("intent") or "")
    # "unavailable" la produce el LLMRouter en timeout/error local → puede
    # venir como "llm_rejected:unavailable": chequear ANTES que llm_rejected.
    if "timeout" in intent or "unavailable" in intent:
        return "timeout"
    if (
        intent == "gate_rejected"
        or intent.startswith("llm_rejected")
        or intent.startswith("low_confidence")
    ):
        return "noise"
    return "other_fail"


class AmbientGuard:
    """Escalera NORMAL/STRICT/COOLDOWN por room. Ver docstring del módulo."""

    def __init__(
        self,
        config: AmbientGuardConfig | None = None,
        time_fn=time.time,
    ):
        self.config = config or AmbientGuardConfig()
        self._time = time_fn
        self._rooms: dict[str, _RoomState] = {}
        self._lock = threading.Lock()

    # ---- API pública ----

    def on_wake(
        self,
        room_id: str,
        score: float,
        rms: float,
        spenergy_peak: float | None = None,
    ) -> GuardDecision:
        """Decisión sobre un wake detectado (llamado desde el audio thread).

        spenergy_peak=None = sin lectura del chip (fail-open del controller)
        → nunca se bloquea voz por un fallo USB.
        """
        if not self.config.enabled:
            return GuardDecision(True, "disabled", GuardState.NORMAL)
        now = self._time()
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, now)
            if rs.state is GuardState.COOLDOWN:
                return GuardDecision(False, "cooldown", rs.state)
            if rs.state is GuardState.STRICT:
                if score < self.config.strict_wake_score:
                    rs.last_reject_at = now  # ambiente persiste → quiet timer se refresca
                    return GuardDecision(False, "strict_score", rs.state)
                if self.config.strict_min_rms > 0.0 and rms < self.config.strict_min_rms:
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_rms", rs.state)
                if (
                    self.config.strict_min_spenergy > 0.0
                    and spenergy_peak is not None
                    and spenergy_peak < self.config.strict_min_spenergy
                ):
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_spenergy", rs.state)
            return GuardDecision(True, "ok", rs.state)

    def on_capture_result(self, room_id: str, outcome: str) -> None:
        """Reporta el resultado de una captura ya procesada (event loop)."""
        if not self.config.enabled:
            return
        now = self._time()
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, now)
            if outcome not in REJECT_OUTCOMES:
                return
            rs.last_reject_at = now
            rs.capture_rejects.append(now)
            window = (
                self.config.cooldown_entry_window_s
                if rs.state is GuardState.STRICT
                else self.config.strict_entry_window_s
            )
            cutoff = now - window
            while rs.capture_rejects and rs.capture_rejects[0] < cutoff:
                rs.capture_rejects.popleft()
            if (
                rs.state is GuardState.NORMAL
                and len(rs.capture_rejects) >= self.config.strict_entry_rejects
            ):
                rs.state = GuardState.STRICT
                rs.capture_rejects.clear()
                logger.warning(
                    f"[AmbientGuard] {room_id}: NORMAL → STRICT "
                    f"({self.config.strict_entry_rejects} capturas rechazadas en "
                    f"{self.config.strict_entry_window_s:.0f}s — ambiente hostil; "
                    f"wake ahora exige score ≥ {self.config.strict_wake_score})"
                )
            elif (
                rs.state is GuardState.STRICT
                and len(rs.capture_rejects) >= self.config.cooldown_entry_rejects
            ):
                rs.state = GuardState.COOLDOWN
                rs.cooldown_until = now + self.config.cooldown_duration_s
                rs.capture_rejects.clear()
                logger.warning(
                    f"[AmbientGuard] {room_id}: STRICT → COOLDOWN "
                    f"{self.config.cooldown_duration_s:.0f}s (rechazos persisten — "
                    f"breaker para no saturar el router)"
                )

    def state_for(self, room_id: str) -> GuardState:
        """Retorna el estado actual de la habitación (con lazy refresh)."""
        if not self.config.enabled:
            return GuardState.NORMAL
        with self._lock:
            rs = self._room(room_id)
            self._refresh(rs, self._time())
            return rs.state

    def follow_up_allowed(self, room_id: str) -> bool:
        """follow_up solo en NORMAL — en STRICT la ventana abierta era parte
        de la cascada de saturación del 06-04."""
        if not self.config.enabled:
            return True
        return self.state_for(room_id) is GuardState.NORMAL

    # ---- internos ----

    def _room(self, room_id: str) -> _RoomState:
        rs = self._rooms.get(room_id)
        if rs is None:
            rs = _RoomState()
            self._rooms[room_id] = rs
        return rs

    def _refresh(self, rs: _RoomState, now: float) -> None:
        """Transiciones por tiempo (lazy — no hay timers)."""
        if rs.state is GuardState.COOLDOWN and now >= rs.cooldown_until:
            rs.state = GuardState.STRICT
            rs.capture_rejects.clear()
            logger.info("[AmbientGuard] COOLDOWN expirado → STRICT")
        if (
            rs.state is GuardState.STRICT
            and rs.last_reject_at > 0.0
            and (now - rs.last_reject_at) >= self.config.strict_exit_quiet_s
        ):
            rs.state = GuardState.NORMAL
            rs.capture_rejects.clear()
            logger.info(
                f"[AmbientGuard] STRICT → NORMAL "
                f"(quiet ≥ {self.config.strict_exit_quiet_s:.0f}s)"
            )
