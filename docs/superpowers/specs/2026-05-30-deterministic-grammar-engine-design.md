# Diseño — Motor de gramática determinístico para domótica simple

**Fecha:** 2026-05-30
**Estado:** aprobado (brainstorming) → pendiente plan de implementación
**Contexto previo:** análisis de latencia sesión XVF3800 (`docs/SESSION_2026-05-30_XVF3800_WAKE_NLU_FIXES.md`), pendiente #4 (LLMRouter Q4_K_M débil) y #7 (latencia <300ms).

## Problema

El routing de comandos en `request_router.py` tiene 3 mecanismos: (1) `regex_extractor+llm_gate` **apagado** (DI pasa `None`); (2) **grammar fast-path** activo (`command_grammar.py`), determinístico, que da los ~567ms / HA 55ms cuando matchea; (3) **LLMCommandRouter** (Qwen-7B Q4_K_M en `:8101`), lento (1–2.5s) e inconsistente.

La gramática determinística **solo cubre `turn_on`/`turn_off`**. Todo lo demás (ajustar brillo/color sin verbo on/off, persianas, ventilador, media) cae al LLMRouter. Ese path es el riesgo: en timeout (config `timeout_s: 2.5`) devuelve `rejection_reason="noise"` y **descarta comandos válidos** (caso real 2026-05-30 21:29: *"Nexa, prende la luz del escritorio"* perdido — el modelo respondió 100ms tarde y se etiquetó como ruido).

## Objetivo

Hacer la gramática determinística lo bastante **completa y autoritativa** como para que la domótica simple **nunca toque el LLM**, reescribiéndola como un **motor de reglas declarativo** (decisión: enfoque B). Arreglar el modo de falla del LLMRouter para que un timeout no descarte comandos válidos.

**No-objetivos:** optimización agresiva de latencia (se mantiene vector search siempre, se acepta ~500ms en resolución de entidad); `toggle` ciego; reescribir el subsistema de música (se reusa).

## Decisiones de alcance (del brainstorming)

| Decisión | Resultado |
|---|---|
| Cobertura del path determinístico | `on/off` + `set` luz (brillo/color/temp), persianas (`open`/`close`), ventilador (`on/off`), media (`play`/`pause`/`next`/`volume`) |
| `toggle` ciego | **Fuera de alcance** |
| Fallback cuando grammar no matchea | Dominio simple **siempre** por grammar; LLMRouter **solo** para no-domótica; arreglar su modo de falla |
| Resolución de entidad concreta | **Vector search siempre** (robustez sobre latencia) |
| Comandos de media | **Rutear al `music_dispatcher` existente** (Spotify, zonas, MA1260) |

## Arquitectura

```
text (post-wake)
  └─> CommandGrammar.parse(text) ──> ParsedCommand
                                       {intent, domain, room, slots,
                                        target, confidence, quality}
        request_router decide por target/quality:
          quality=full & conf≥θ:
             target=domotics → CommandClassification (bypass LLM) → dispatcher → vector resuelve entidad → HA
             target=music    → dispatcher path de música (_fast_music_path)
          else → LLMCommandRouter (SOLO no-domótica)  [timeout→unavailable, NO noise]
```

### Componentes y límites

- **`src/nlu/command_grammar.py`** — motor + tabla de reglas + `ParsedCommand`. Puro, sin I/O, depende solo de `slot_extractor`. Totalmente unit-testeable.
- **`src/nlu/slot_extractor.py`** — léxicos/extractores puros: intents, brillo, color, temperatura, **+ volumen** (nuevo).
- **`src/pipeline/request_router.py`** — orquesta: llama al motor, decide `target`, maneja el fallback LLM.
- **`src/nlu/llm_router.py`** — solo el cambio de modo de falla (timeout/error → `unavailable`, no `noise`).

## Motor declarativo

`classify_intent` (hoy 2 regex sueltos) se reemplaza por una tabla de reglas declarativas:

```python
@dataclass(frozen=True)
class IntentRule:
    intent: str                       # turn_on|turn_off|set|open|close|media_play|media_pause|media_next|volume_set
    verb_patterns: tuple[str, ...]    # lexemes/regex rioplatense
    domains: frozenset[str]           # dominios HA aplicables (matriz de compatibilidad)
    target: str = "domotics"          # "domotics" | "music"
    requires_slot: str | None = None  # ej. volume_set requiere slot de volumen

INTENT_RULES = (
  IntentRule("turn_on",     ("prend*","encend*","ilumin*","activ*"), frozenset({"light","fan","climate","switch"})),
  IntentRule("turn_off",    ("apag*","cort*","desactiv*"),           frozenset({"light","fan","climate","switch"})),
  IntentRule("set",         (),  frozenset({"light"}), requires_slot="any"),   # brillo/color/temp SIN verbo on/off
  IntentRule("open",        ("sub*","abr*","levant*"),               frozenset({"cover"})),
  IntentRule("close",       ("baj*","cerr*"),                        frozenset({"cover"})),
  IntentRule("media_play",  ("pon*","reproduc*","dale","segui*"),    frozenset({"media_player"}), target="music"),
  IntentRule("media_pause", ("paus*","par*","fren*","callate"),      frozenset({"media_player"}), target="music"),
  IntentRule("media_next",  ("siguiente","proxim*","cambia*"),       frozenset({"media_player"}), target="music"),
  IntentRule("volume_set",  ("volumen","fuerte","bajito"),           frozenset({"media_player"}), target="music", requires_slot="volume"),
)
```

### Matriz de compatibilidad intent↔dominio

El motor solo acepta un match si el dominio detectado está en `rule.domains`. Resuelve verbos compartidos según el dominio:
- *"subí la persiana"* → `open` (cover) ✓
- *"subí el volumen"* → `volume_set` (media_player) ✓
- *"subí el brillo"* → `set` (light, slot brightness) ✓
- *"abrí la luz"* → `open`+`light` incompatible → **no match** → fallback (no se dispatcha disparate).

### Resolución de intent ambiguo

Cuando varias reglas matchean el verbo, gana la que sea **compatible con el dominio detectado**. Si el dominio es ambiguo o ausente, el match es `partial` (no `full`) y no bypassa el LLM salvo que la confianza alcance el umbral por otras señales.

## Modelo de datos

```python
@dataclass
class ParsedCommand:
    intent: str | None = None
    domain: str | None = None        # light/cover/fan/climate/media_player
    room: str | None = None
    slots: dict = field(default_factory=dict)
    target: str = "domotics"         # o "music"
    confidence: float = 0.0
    quality: str = "none"            # "full" (intent+domain compatibles) | "partial" | "none"

    def ready_to_dispatch(self) -> bool:
        return self.quality == "full"
```

- **`set`** se traduce a `light.turn_on` + service_data (HA prende al setear brillo/color). El merge de slots usa `merge_service_data` (ya existe).
- **Confidence**: heurística actual extendida (base 0.7 + bonos wake/room/slots), con chequeo de `requires_slot`.
- **`parse_partial_command`** queda como wrapper fino sobre `parse()` para no romper el early_dispatch streaming (consumidor en `multi_room_audio_loop.py`).

## Routing en `request_router.py`

Reemplaza `_grammar_fastpath_classification` por una llamada al motor:
- `quality=="full"` & `confidence ≥ confidence_threshold`:
  - `target=="domotics"` → construir `CommandClassification(is_command=True, ...)` → bypass LLM (igual que hoy).
  - `target=="music"` → señalar al dispatcher (vía intent/flag en el resultado) que use el path de música existente.
- En otro caso → `LLMCommandRouter.classify` (ahora conceptualmente solo no-domótica).

## Manejo de errores

- **Intent/dominio incompatibles** → `quality != "full"` → no se dispatcha; cae al fallback.
- **Entidad ambigua** (múltiples candidatas del dominio) → la resuelve el vector search (se mantiene).
- **LLM unavailable** (timeout/error) → `rejection_reason="unavailable"`, `is_command=None`. El caller: si la grammar tenía señal parcial (intent *o* domain) → no descartar (pedir confirmación o dispatch best-effort); sin señal → drop seguro.
- **Acciones sensibles / baja confianza** → se mantiene `_check_confidence_gate` / `_build_confirmation_question`.

### Decisión abierta para el plan
Ante "LLM unavailable + grammar parcial": **dispatch best-effort** vs **pedir confirmación por voz**. Default propuesto: pedir confirmación (más seguro), salvo que el intent+domain estén ambos presentes (entonces dispatch).

## Testing

- **Table-driven por intent**: frases rioplatenses → `ParsedCommand` esperado. Incluir garbled reales de logs (*"apagalelu el victorio"*, *"prendela luces"*, *"Nexa, prende la luz del escritorio"*).
- **Matriz de compatibilidad**: *"abrí la luz"* (no match), *"subí la persiana"* (open), *"subí el volumen"* (volume_set), *"subí el brillo"* (set).
- **Regresión**: tests actuales de turn_on/turn_off siguen pasando.
- **LLM router**: timeout → `unavailable` (no `noise`); partial-grammar + LLM-unavailable → no se descarta.
- **Integración media**: frase de media → ruteada al music path (mock `music_dispatcher`), no toca el dispatcher de domótica.

## Riesgos / consideraciones

- **Léxico rioplatense incompleto**: los `verb_patterns` deben cubrir variantes reales; mitigar con tests basados en logs y fácil extensión de la tabla.
- **`parse_partial_command` en streaming**: el early_dispatch llama por chunk; el motor debe seguir siendo idempotente y barato (sin I/O).
- **Compat de `CommandClassification`**: mantener la forma que el resto del pipeline espera (history recording, hooks de intent).
- **Música vía request_router**: confirmar en el plan el punto exacto donde el dispatcher decide music vs domótica (`dispatcher.py:457`) para no duplicar la detección.

## Archivos afectados (estimado)

- `src/nlu/command_grammar.py` — reescritura a motor + tabla + `ParsedCommand`.
- `src/nlu/slot_extractor.py` — +extractor de volumen; intents pasan a la tabla.
- `src/pipeline/request_router.py` — routing por `target`/`quality`; manejo de `unavailable`.
- `src/nlu/llm_router.py` — timeout/error → `unavailable`.
- `config/settings.yaml` — posible ajuste de `nlu.llm_router.timeout_s` y limpieza del `fast_path` muerto (regex+gate).
- `tests/unit/nlu/` — suite table-driven nueva + ajustes.
