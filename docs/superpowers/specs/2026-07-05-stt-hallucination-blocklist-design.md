# BoH-es: blocklist de alucinaciones + guard anti-eco del prompt — Diseño

**Fecha:** 2026-07-05
**Estado:** Aprobado (pendiente plan)
**Rama:** `feat/stt-hallucination-blocklist`
**Contexto:** incidente temperature_fallback cerrado (fix `216af3f` validado). Las
alucinaciones residuales de T=0 que PASAN el gate ('Aplausos', eco del
initial_prompt) queman ~700ms de LLMRouter cada una y mantienen riesgo residual
de acción fantasma. Respaldo externo: paper ICASSP 2025 (arXiv 2501.11378) —
blocklist BoH + delooping eliminó el 100% de alucinaciones detectadas; ⚠️ el
paper excluyó "thank you" del BoH por false-positives → acá las frases cortas
tipo "gracias" siguen siendo dominio del filler gate (whole-utterance), NO de
esta lista.

## Decisiones (usuario, 2026-07-05)

1. **Lista curada + script de cosecha que PROPONE** (nunca auto-agrega).
2. **Guard anti-eco del initial_prompt por similitud** (no frase fija).

## Cambios

### 1. Extender `_NOISE_PHRASES` (`src/nlu/command_gate.py`)

Agregar las confirmadas en logs de producción que hoy pasan el gate:
`"aplausos"` (25 ocurrencias en 48h, 0 comandos reales que la contengan).
El resto de candidatas sale del primer run del script de cosecha (paso de
implementación, revisadas a mano antes de entrar).

### 2. Hard rule nueva `prompt_echo` (`src/nlu/command_gate.py`)

- `CommandAcceptanceGate` recibe `initial_prompt: str | None = None` por
  constructor (DI desde `main.py`, valor de `stt.initial_prompt` del settings).
- Al construir: separa el prompt en oraciones (split por `.`), normaliza cada
  una con `_normalize()` y descarta las de <4 palabras (evita matches triviales).
- Regla: si `SequenceMatcher(None, _normalize(text), oracion).ratio() >= 0.8`
  para alguna oración del prompt → reject `prompt_echo`.
- Orden: corre junto a las demás hard rules (después de noise/filler).
- `earcon_gate._NOISE_PREFIXES` suma `"prompt_echo"` (el eco JAMÁS earcon).

### 3. Script de cosecha `tools/harvest_hallucinations.py`

- Input: export de journal por stdin (`journalctl --user -u kza-voice
  --since ... --output=cat | python tools/harvest_hallucinations.py`) o
  `--file`. Standalone stdlib (corre en el server sin PYTHONPATH), salvo el
  import opcional de la lista actual para excluir ya-bloqueadas (si
  `src.nlu.command_gate` está importable, la usa; si no, sigue sin excluir).
- Detecta el patrón: `[CommandProcessor] Text='X'` seguido de
  `[CommandGate] accept=True` y luego `is_command=False` del router
  (aceptado por el gate pero descartado por el router = candidata).
- Agrupa por texto normalizado, cuenta ocurrencias, y lista candidatas con
  `count >= --min-count` (default 3): tabla `count | texto | primera/última vez`.
- **Solo propone** — un humano decide qué entra a `_NOISE_PHRASES`.

## Errores y edge cases

- `initial_prompt=None` o vacío → guard inactivo (gate sigue igual).
- Comandos reales nunca matchean 0.8 contra oraciones del prompt (verificado
  en tests con el prompt real del settings).
- Substring over-broad: cada frase nueva a `_NOISE_PHRASES` necesita el mismo
  criterio del fix 2026-06-02 ("activa la"): no puede ser substring plausible
  de un comando válido.

## Testing

- Gate: 'Aplausos'/'¡Aplausos!' → reject `noise_phrase`; 'Esto es un asistente
  de voz.' → reject `prompt_echo` (con el prompt real); comandos válidos
  ('Nexa, prendé la luz', 'activá la escena lectura') → accept.
- Earcon: reason `prompt_echo` → nunca earcon.
- Harvest: parseo del patrón sobre fixture de journal sintético, conteo,
  exclusión de ya-bloqueadas, umbral min-count.

## Fuera de alcance

- Delooping adicional (el compression gate ya cubre loops).
- BoH gateada por n-gram LM log-prob (el conteo + curación manual cumple el
  mismo rol con menos piezas).
- Cambios al filler gate existente.
