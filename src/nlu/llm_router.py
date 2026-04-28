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
class CommandSegment:
    """Sub-comando dentro de un wake fire.

    El clasificador puede partir un texto en N segments cuando detecta
    conectores ("y", "también", "y luego") con verbos+entidades distintos.
    Cada segment se despacha de forma independiente en paralelo.
    `needs_reasoning=True` indica que el dispatcher fast no resuelve el
    segment y debe ir al reasoner 72B (timer/alarma/condicional/creativo).
    """
    text: str
    needs_reasoning: bool = False


@dataclass
class CommandClassification:
    """Resultado de clasificar un texto post-wake."""
    is_command: bool
    confidence: float = 0.0
    rejection_reason: Optional[str] = None  # tv_replay | tv_phrase | incomplete | noise | unknown_intent
    intent: Optional[str] = None
    entity_hint: Optional[str] = None
    slots: dict = field(default_factory=dict)
    # Lista de sub-comandos. Para casos single-intent contiene 1 entry con
    # text == texto original. Para multi-intent ("apagá X y prendé Y") tiene N.
    # Consumidores que no la usen siguen funcionando con intent/entity_hint/slots.
    segments: list[CommandSegment] = field(default_factory=list)
    raw_response: str = ""  # texto crudo del LLM (para debugging)
    elapsed_ms: float = 0.0


@dataclass
class _HistoryEntry:
    text: str
    intent: str
    timestamp: float


# Prompt SYSTEM ultra-compacto. El shape JSON y los enums de intent /
# rejection_reason los enforza vLLM vía response_format (ver _GUIDED_JSON_SCHEMA),
# por eso acá NO repetimos campos ni listas — solo la semántica de rechazo /
# aceptación que el modelo no puede inferir del schema solo.
#
# El cuerpo se mantiene byte-idéntico entre llamadas para maximizar prefix
# caching de vLLM (cachea bloques de 16 tokens del prefijo estable).
_SYSTEM_PROMPT = """Sos KZA, router de comandos para un asistente de hogar en español rioplatense. Devolvé JSON.

INTENTS soportados:
turn_on (prendé/encendé), turn_off (apagá/cortá), toggle (cambiá), set_brightness (subí/bajá la luz, al N%), set_color_temp + set_color (cálida/fría/blanca/roja), set_volume, set_temperature, open/close (persianas/garage), scene_activate, media_play/pause/stop/next/previous.

is_command=TRUE si el texto pide AL MENOS UNA acción ejecutable (domótica directa O algo que requiera razonamiento como timer/alarma/pregunta). Llená intent/entity_hint/slots con el PRIMER comando del texto.

is_command=FALSE solo si:
- tv_replay: copia casi exacta del historial <30s
- tv_phrase: tono narrativo de TV ("¡Qué increíble!", "no puedo creer")
- incomplete: solo wake word, sin verbo ni entidad
- noise: palabras random sin sentido
- unknown_intent: el verbo no aplica a domótica ni a reasoning (ej. insultos, idioma extraño)

SEGMENTS: lista de sub-comandos, cada uno {text, needs_reasoning}.
- Quitá la wake word ("nexa"/"kaza"/"alexa") del text de cada segment.
- needs_reasoning=true para timers, alarmas, condicionales, preguntas, creatividad. false para domótica directa.
- Splitteá cuando hay verbos+entidades distintos unidos por "y"/"también"/"y luego".
- NO splitteás cuando "y" une atributos del MISMO comando ("cálida y suave" = atributos de la misma luz).

EJEMPLOS:
"apagá la luz" → is_cmd=true, intent=turn_off, segments=[{text:"apagá la luz", needs_reasoning:false}]
"prendé la luz cálida y suave" → is_cmd=true, intent=turn_on, segments=[{text:"prendé la luz cálida y suave", needs_reasoning:false}]
"apagá la luz y prendé el aire" → is_cmd=true, intent=turn_off, segments=[{text:"apagá la luz", needs_reasoning:false}, {text:"prendé el aire", needs_reasoning:false}]
"apagá la luz y poné la alarma a las 6 am" → is_cmd=true, intent=turn_off, segments=[{text:"apagá la luz", needs_reasoning:false}, {text:"poné la alarma a las 6 am", needs_reasoning:true}]
"decime qué hora es" → is_cmd=true, intent=null, segments=[{text:"decime qué hora es", needs_reasoning:true}]
"apagá la luz también pausá la música" → is_cmd=true, intent=turn_off, segments=[{text:"apagá la luz", needs_reasoning:false}, {text:"pausá la música", needs_reasoning:false}]
"qué tal" → is_cmd=false, rejection_reason=incomplete, segments=[]"""


# JSON Schema para vLLM guided decoding. Garantiza output válido y corta
# generación al cerrar el objeto → ahorra ~40% de tokens vs free-form.
# Diseño:
# - is_command + confidence required (sin ellos no podemos decidir).
# - intent / entity_hint / rejection_reason / slots nullable (válidos en rechazos).
# - intent enum cerrado: el modelo no puede inventar verbos fuera de KNOWN_INTENTS.
# - slots: additionalProperties=true para que el modelo agregue claves específicas
#   por intent (brightness_pct, color_temp_kelvin, etc.) sin listarlas todas.
_GUIDED_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "is_command": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rejection_reason": {
            "type": ["string", "null"],
            "enum": [None, "tv_replay", "tv_phrase", "incomplete", "noise", "unknown_intent"],
        },
        "intent": {
            "type": ["string", "null"],
            "enum": [None, *KNOWN_INTENTS],
        },
        "entity_hint": {"type": ["string", "null"]},
        "slots": {"type": "object", "additionalProperties": True},
        # Multi-intent split. Para single-intent cuenta con 1 entry; el dispatcher
        # itera y despacha cada segment en paralelo. needs_reasoning routea al
        # reasoner 72B en lugar del fast path.
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "needs_reasoning": {"type": "boolean"},
                },
                "required": ["text", "needs_reasoning"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["is_command", "confidence"],
    "additionalProperties": False,
}


# Body que se reenvía como extra_body al endpoint /v1/completions de vLLM.
# vLLM compila un FSM por schema único (cache), por eso definimos esto a nivel
# módulo y no por llamada — primer call ~1s (compile), siguientes ~400-500ms.
_RESPONSE_FORMAT_BODY: dict = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "kza_command_classification",
            "schema": _GUIDED_JSON_SCHEMA,
        },
    }
}


class LLMCommandRouter:
    """Clasificador de comandos via vLLM 7B con contexto histórico."""

    def __init__(
        self,
        fast_router,
        max_history: int = 5,
        history_ttl_s: float = 120.0,
        timeout_s: float = 1.5,
        max_tokens: int = 150,
        temperature: float = 0.0,
    ):
        """
        Args:
            fast_router: instancia FastRouter (HTTP a vLLM compartido).
            max_history: cuántos comandos del user mantener en ventana.
            history_ttl_s: comandos más viejos que esto se descartan del prompt.
            timeout_s: cap de latencia del LLM call. Si excede, devolvemos
                un resultado de "no decidió" → caller decide fallback.
            max_tokens: cap del response del LLM. Con guided_json el JSON
                single-intent cabe en ~50-70 tokens; multi-intent (con segments)
                puede ir a 100-130. 150 da margen sin desperdicio significativo.
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
        """Construir el prompt completo: system (estable) + contexto + texto.

        Orden importa para vLLM prefix cache: el _SYSTEM_PROMPT va PRIMERO y es
        byte-idéntico entre llamadas (cache hit del prefill). Lo variable
        (history, room, text) va al final.

        El trailer `JSON:` se omite — response_format ya enforza salida JSON
        sin necesidad de hint textual y ahorra tokens de prefill.
        """
        room_block = f"HABITACIÓN: {room_id}" if room_id else ""
        history_block = self._build_history_block()
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{history_block}\n"
            f"{room_block}\n"
            f"TEXTO: {text!r}"
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
                    _RESPONSE_FORMAT_BODY,  # vLLM ≥0.10: response_format json_schema
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
        # Si is_command=True con un intent inválido (no en KNOWN_INTENTS y
        # tampoco null), tratamos como unknown_intent. Permitimos intent=null
        # cuando el comando va exclusivamente por reasoning (timer/pregunta).
        if is_cmd and intent is not None and intent not in KNOWN_INTENTS:
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

        # Parsear segments. Si el modelo no los emitió (caso legacy / rechazo),
        # construimos un único segment con el texto original — así downstream
        # no necesita branchear single vs multi.
        segments = self._parse_segments(data.get("segments"), is_cmd)

        return CommandClassification(
            is_command=is_cmd,
            confidence=float(data.get("confidence", 0.0)),
            rejection_reason=data.get("rejection_reason"),
            intent=intent if is_cmd else None,
            entity_hint=data.get("entity_hint") if is_cmd else None,
            slots=slots if is_cmd else {},
            segments=segments,
            raw_response=raw,
            elapsed_ms=elapsed_ms,
        )

    def _parse_segments(self, raw_segments, is_cmd: bool) -> list[CommandSegment]:
        """Convertir el array `segments` del JSON en CommandSegment objects.

        Si is_cmd=False o el array está vacío/inválido, devolvemos []. Para
        is_cmd=True garantizamos al menos 1 segment (fallback al texto crudo lo
        maneja el caller, que tiene acceso al texto original).
        """
        if not is_cmd or not isinstance(raw_segments, list):
            return []
        out: list[CommandSegment] = []
        for entry in raw_segments:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            out.append(CommandSegment(
                text=text.strip(),
                needs_reasoning=bool(entry.get("needs_reasoning", False)),
            ))
        return out
