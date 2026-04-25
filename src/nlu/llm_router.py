"""LLM-based command router — reemplazo del par regex+ChromaDB.

Recibe el texto post-Whisper y delega a vLLM 7B Qwen para:
1. Validar si es un comando real del usuario (vs TV/eco/replay/noise).
2. Extraer intent (turn_on, turn_off, set_brightness, etc.).
3. Extraer entity_hint (frase natural, ej "luz del escritorio") + slots.

El motivo: el regex+Chroma actual no distingue una frase "Nexa bajá la luz"
dicha por el user de la misma frase alucinada por Whisper desde audio TV
ruidoso. Un LLM con contexto histórico (últimos N comandos del user) puede
detectar replays, frases TV-style, y validar coherencia.

Diseño:
- Una sola llamada al LLM por wake fire (no por cada chunk de audio).
- Input: texto + ventana de últimos N comandos válidos del user (con timestamps).
- Output: JSON estricto con clasificación.
- Fallback: si el LLM falla/timeout, el caller puede caer al path regex+Chroma.

Latencia esperada: 150-300ms con vLLM 7B AWQ en :8100.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Lista cerrada de intents que el sistema sabe ejecutar. Si el LLM responde
# algo fuera de esta lista, lo tratamos como is_command=False.
KNOWN_INTENTS: tuple[str, ...] = (
    "turn_on",
    "turn_off",
    "toggle",
    "set_brightness",
    "set_color_temp",
    "set_color",
    "set_volume",
    "set_temperature",
    "open",
    "close",
    "scene_activate",
    "media_play",
    "media_pause",
    "media_stop",
    "media_next",
    "media_previous",
)


@dataclass
class CommandClassification:
    """Resultado de clasificar un texto post-wake."""
    is_command: bool
    confidence: float = 0.0
    rejection_reason: Optional[str] = None  # tv_replay | tv_phrase | incomplete | noise | unknown_intent
    intent: Optional[str] = None
    entity_hint: Optional[str] = None
    slots: dict = field(default_factory=dict)
    raw_response: str = ""  # texto crudo del LLM (para debugging)
    elapsed_ms: float = 0.0


@dataclass
class _HistoryEntry:
    text: str
    intent: str
    timestamp: float


# Prompt SYSTEM optimizado para Qwen 7B en español. Pide JSON estricto.
_SYSTEM_PROMPT = """Sos KZA, un router de comandos para un asistente de domótica en español rioplatense.

Recibís un texto transcripto por Whisper de lo que oyó el micrófono y debés decidir:
1. ¿Es un comando real del usuario, o ruido (TV, eco, alucinación de Whisper, replay)?
2. Si es comando: ¿qué intent y qué entidad referencia?

DEVOLVÉ SOLO UN JSON con esta estructura exacta. Sin explicación, sin texto adicional:
{
  "is_command": true|false,
  "confidence": 0.0-1.0,
  "rejection_reason": null | "tv_replay" | "tv_phrase" | "incomplete" | "noise" | "unknown_intent",
  "intent": null | "turn_on" | "turn_off" | "toggle" | "set_brightness" | "set_color_temp" | "set_color" | "set_volume" | "set_temperature" | "open" | "close" | "scene_activate" | "media_play" | "media_pause" | "media_stop" | "media_next" | "media_previous",
  "entity_hint": null | "<frase natural del dispositivo, ej luz del escritorio>",
  "slots": {
    "brightness_pct": int|null,
    "color_temp_kelvin": int|null,
    "rgb_color": [r,g,b]|null,
    "value": any|null
  }
}

REGLAS DE RECHAZO (poné is_command=false):
- Texto idéntico o casi idéntico a un comando del HISTORIAL hace <30 segundos → "tv_replay" (Whisper alucinó del audio TV)
- Frases con tono narrativo/dramático/conversacional típico de TV ("¿qué pasa?", "no puedo creer", "¡Salud!", nombres propios random) → "tv_phrase"
- Texto incompleto ("Nexa..." solo, sin verbo o sin entidad) → "incomplete"
- Texto sin sentido o palabras random → "noise"
- Verbo de acción no soportado por la lista de intents → "unknown_intent"

REGLAS DE ACEPTACIÓN (poné is_command=true):
- Verbo claro de acción + entidad/objeto identificable
- Aunque sea casual ("apagame eso", "che bajá la luz") si el intent es claro
- Si pasaron >60s desde el último comando, los replays son menos sospechosos
"""


class LLMCommandRouter:
    """Clasificador de comandos via vLLM 7B con contexto histórico."""

    def __init__(
        self,
        fast_router,
        max_history: int = 5,
        history_ttl_s: float = 120.0,
        timeout_s: float = 1.5,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ):
        """
        Args:
            fast_router: instancia FastRouter (HTTP a vLLM compartido).
            max_history: cuántos comandos del user mantener en ventana.
            history_ttl_s: comandos más viejos que esto se descartan del prompt.
            timeout_s: cap de latencia del LLM call. Si excede, devolvemos
                un resultado de "no decidió" → caller decide fallback.
            max_tokens: cap del response del LLM. JSON suele caber en ~150.
            temperature: 0.0 para máxima determinismo en la clasificación.
        """
        self.router = fast_router
        self.max_history = max_history
        self.history_ttl_s = history_ttl_s
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._history: deque[_HistoryEntry] = deque(maxlen=max_history)

    def record_command(self, text: str, intent: str) -> None:
        """Registrar un comando ejecutado (para contexto de futuros calls)."""
        if not text or not intent:
            return
        self._history.append(_HistoryEntry(
            text=text.strip(),
            intent=intent,
            timestamp=time.time(),
        ))

    def _build_history_block(self) -> str:
        """Renderear los últimos comandos válidos como contexto en el prompt."""
        now = time.time()
        valid = [
            h for h in self._history
            if (now - h.timestamp) <= self.history_ttl_s
        ]
        if not valid:
            return "HISTORIAL DE COMANDOS RECIENTES: (vacío)"
        lines = ["HISTORIAL DE COMANDOS RECIENTES (más reciente primero):"]
        for h in reversed(valid):
            ago = int(now - h.timestamp)
            lines.append(f"  - hace {ago}s, intent={h.intent}: {h.text!r}")
        return "\n".join(lines)

    def _build_prompt(self, text: str, room_id: Optional[str] = None) -> str:
        """Construir el prompt completo: system + contexto + texto a clasificar."""
        room_block = f"HABITACIÓN ACTIVA: {room_id}" if room_id else ""
        history_block = self._build_history_block()
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{room_block}\n"
            f"{history_block}\n\n"
            f"TEXTO TRANSCRIPTO: {text!r}\n\n"
            f"JSON:"
        )

    async def classify(
        self,
        text: str,
        room_id: Optional[str] = None,
    ) -> CommandClassification:
        """Clasificar un texto. No bloquea — corre el LLM en thread pool."""
        t0 = time.perf_counter()
        prompt = self._build_prompt(text, room_id)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.router.generate,
                    [prompt],
                    self.max_tokens,
                    self.temperature,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"LLMCommandRouter timeout ({elapsed:.0f}ms > {self.timeout_s*1000:.0f}ms)"
            )
            return CommandClassification(
                is_command=False,
                rejection_reason="noise",
                raw_response="<timeout>",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(f"LLMCommandRouter error: {e}")
            return CommandClassification(
                is_command=False,
                rejection_reason="noise",
                raw_response=f"<error: {type(e).__name__}>",
                elapsed_ms=elapsed,
            )
        elapsed = (time.perf_counter() - t0) * 1000
        raw = (results[0] if results else "").strip()
        return self._parse_response(raw, elapsed)

    def _parse_response(self, raw: str, elapsed_ms: float) -> CommandClassification:
        """Extraer el JSON del response y validarlo. Permisivo con ruido alrededor."""
        if not raw:
            return CommandClassification(
                is_command=False,
                rejection_reason="noise",
                raw_response=raw,
                elapsed_ms=elapsed_ms,
            )

        # Buscar el primer bloque JSON {...} en el response. Qwen a veces
        # mete texto preámbulo aunque le pidamos solo JSON.
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            logger.warning(f"LLMCommandRouter: sin JSON en response: {raw[:200]!r}")
            return CommandClassification(
                is_command=False,
                rejection_reason="noise",
                raw_response=raw,
                elapsed_ms=elapsed_ms,
            )

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"LLMCommandRouter JSON inválido ({e}): {raw[:200]!r}")
            return CommandClassification(
                is_command=False,
                rejection_reason="noise",
                raw_response=raw,
                elapsed_ms=elapsed_ms,
            )

        is_cmd = bool(data.get("is_command", False))
        intent = data.get("intent")
        # Validar que el intent sea uno conocido (si is_command=True).
        if is_cmd and intent not in KNOWN_INTENTS:
            return CommandClassification(
                is_command=False,
                confidence=float(data.get("confidence", 0.0)),
                rejection_reason="unknown_intent",
                raw_response=raw,
                elapsed_ms=elapsed_ms,
            )

        slots = data.get("slots") or {}
        if not isinstance(slots, dict):
            slots = {}
        # Limpiar slots None — solo dejar los que tengan valor.
        slots = {k: v for k, v in slots.items() if v is not None}

        return CommandClassification(
            is_command=is_cmd,
            confidence=float(data.get("confidence", 0.0)),
            rejection_reason=data.get("rejection_reason"),
            intent=intent if is_cmd else None,
            entity_hint=data.get("entity_hint") if is_cmd else None,
            slots=slots if is_cmd else {},
            raw_response=raw,
            elapsed_ms=elapsed_ms,
        )
