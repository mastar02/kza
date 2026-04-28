"""LLM gate binario — validador thin para el fast path del request_router.

Cuando el RegexExtractor matchea limpio un comando domótico, el LLM gate
hace la última verificación: ¿este texto pide REALMENTE ejecutar la acción
ahora, o es un eco de TV / pregunta / pasado / replay?

Diseño:
- Output ULTRA corto (~5-10 tokens): un JSON `{"valid": bool}`.
- Schema enforced via vLLM response_format json_schema (FSM determinista).
- Mismo FastRouter que LLMCommandRouter — sin VRAM extra.
- Latencia objetivo: 50-100ms (output corto + prefix cache estable).

Si el gate falla (timeout, error de red, JSON inválido), el caller debe
caer al LLMCommandRouter completo. NUNCA bypasear sin validación.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Schema mínimo: solo {valid: bool}. vLLM compila el FSM una sola vez.
_GATE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
    },
    "required": ["valid"],
    "additionalProperties": False,
}

_GATE_RESPONSE_FORMAT: dict = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "kza_gate",
            "schema": _GATE_JSON_SCHEMA,
        },
    }
}


# Prompt SYSTEM compacto y estable (cache hit del prefill).
# Se mantiene byte-idéntico entre llamadas — la parte variable
# (texto + intent extraído) va al final.
_GATE_SYSTEM = """Sos un validador binario para un asistente domótico en español rioplatense.
Recibís un texto transcripto por Whisper y un intent ya extraído por regex.
Decidí si el texto está PIDIENDO EJECUTAR la acción AHORA.

valid=false si:
- El verbo está en pasado ("ya apagué", "prendiste").
- Es una pregunta ("apagaste?", "qué hora es").
- Es un reporte de habla ("le dije que apague").
- Es subjuntivo / condicional ("si prendés", "ojalá").
- Es 3ra persona narrativa ("ella apaga", "la casa apaga").
- Es eco de TV / replay (frase narrativa, conversacional).
- El texto principal NO coincide con la intención detectada.

valid=true si:
- Imperativo afirmativo claro, en presente, dirigido a KZA.
- El texto pide ejecutar la acción detectada por el regex.

Devolvé SOLO {"valid": true|false}."""


@dataclass(frozen=True)
class GateResult:
    """Resultado del gate. raw_response sirve para telemetría/debug."""
    valid: bool
    elapsed_ms: float
    raw_response: str = ""


class LLMGate:
    """Validador binario thin sobre vLLM compartido (7B AWQ).

    Uso típico:
        gate = LLMGate(fast_router)
        result = await gate.validate(
            text="Nexa apagá la luz del escritorio",
            intent="turn_off",
            entity_hint="escritorio",
        )
        if result.valid:
            # bypass LLM router completo, dispatch directo
            ...
    """

    def __init__(
        self,
        fast_router,
        timeout_s: float = 0.5,
        max_tokens: int = 20,
        temperature: float = 0.0,
    ):
        """
        Args:
            fast_router: instancia FastRouter (HTTP a vLLM compartido).
            timeout_s: cap de latencia. Si excede, devolvemos invalid.
            max_tokens: cap del response. {"valid":true} caben en ~5-10 tokens.
            temperature: 0.0 para determinismo.
        """
        self.router = fast_router
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, text: str, intent: str, entity_hint: str | None) -> str:
        """Construye el prompt — SYSTEM estable primero (prefix cache)."""
        ent_part = f" sobre '{entity_hint}'" if entity_hint else ""
        return (
            f"{_GATE_SYSTEM}\n\n"
            f"INTENT DETECTADO: {intent}{ent_part}\n"
            f"TEXTO: {text!r}"
        )

    async def validate(
        self,
        text: str,
        intent: str,
        entity_hint: str | None = None,
    ) -> GateResult:
        """Valida si el texto realmente pide ejecutar el intent.

        Devuelve GateResult con valid=True/False. En caso de error o timeout,
        valid=False (política conservadora: no ejecutar si no estamos seguros).
        """
        t0 = time.perf_counter()
        prompt = self._build_prompt(text, intent, entity_hint)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.router.generate,
                    [prompt],
                    self.max_tokens,
                    self.temperature,
                    _GATE_RESPONSE_FORMAT,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"LLMGate timeout ({elapsed:.0f}ms > {self.timeout_s*1000:.0f}ms) "
                f"— defaulting to invalid"
            )
            return GateResult(valid=False, elapsed_ms=elapsed, raw_response="<timeout>")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(f"LLMGate error: {e} — defaulting to invalid")
            return GateResult(
                valid=False,
                elapsed_ms=elapsed,
                raw_response=f"<error: {type(e).__name__}>",
            )

        elapsed = (time.perf_counter() - t0) * 1000
        raw = (results[0] if results else "").strip()
        return self._parse(raw, elapsed)

    def _parse(self, raw: str, elapsed_ms: float) -> GateResult:
        """Extrae el JSON del response. Permisivo con ruido alrededor."""
        if not raw:
            return GateResult(valid=False, elapsed_ms=elapsed_ms, raw_response=raw)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return GateResult(valid=False, elapsed_ms=elapsed_ms, raw_response=raw)
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return GateResult(valid=False, elapsed_ms=elapsed_ms, raw_response=raw)
        return GateResult(
            valid=bool(data.get("valid", False)),
            elapsed_ms=elapsed_ms,
            raw_response=raw,
        )
