# Wake + TV Filter + Pipeline Regression Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolver 4 regresiones abiertas introducidas por el feature "omitir datos de tele" (commit `8820c84`): wake↔router text mismatch, TV false positives, HA state prefetch sin hit de cache, y warnings non-fatal al startup (FeatureManager + TTS response cache).

**Architecture:** Fixes localizados en los archivos existentes — sin refactor de arquitectura. Orden por impacto: P0 (rompe control por voz) → P1/P2 (core del feature) → P3 (latencia) → P4 (ruido en logs). Cada task TDD con test que reproduce la regresión antes del fix. Deploy a server via `scp` + `systemctl restart kza-voice.service` después de cada P completo.

**Tech Stack:** Python 3.13, pytest, Whisper (`faster_whisper`), `CommandEvent`/`RequestRouter` pipeline, Home Assistant WebSocket state cache, Kokoro TTS + DualTTS wrapper.

**Precondiciones:**
- Working directory: `/Users/yo/Documents/kza`, branch `main`
- Server: `kza@192.168.1.2`, services activos: `kza-voice.service`, `kza-72b.service` (nota: **Qwen 72B Q8_0 está borrado del disco** — el `:8200` devolverá error hasta que se decida re-bajarlo; ver sección de decisión al final)
- Commit base: `8820c84` (feature WIP ya está commiteado)

**Archivos que se tocarán (mapa global):**
- `src/wakeword/whisper_wake.py` — P1 (STT coalescing), P2 (TV stop phrases)
- `src/pipeline/request_router.py` — P0 (prioridad de texto)
- `src/pipeline/command_event.py` — P0 (asegurar propagación `wake_text`)
- `src/home_assistant/ha_client.py` — P3 (cache hit path)
- `src/orchestrator/request_dispatcher.py` o call site de `ha_client.call_service` — P3 (usar cache)
- `src/intercom/intercom_system.py` — P4 (guard contra atributos faltantes)
- `src/audio/zone_manager.py` — P4 (agregar campos opcionales a `Zone`) **O** `intercom_system.py` (usar `getattr`) — decisión en P4
- `src/tts/response_cache.py` — P4 (diagnóstico + fix sintetización)
- `tests/unit/wakeword/test_whisper_wake.py` — tests P1/P2
- `tests/unit/pipeline/test_request_router.py` — tests P0
- `tests/unit/home_assistant/test_ha_client_cache.py` — tests P3
- `tests/unit/tts/test_response_cache.py` — tests P4

**Decisiones de usuario requeridas antes de arrancar:**
- 🔷 **DECISIÓN 1 (P1 — STT coalescing):** cómo manejar "nexa aprende" (ver Task 2). Trade-off: regex laxa (FP risk) vs re-segmentación post-proceso (complejidad) vs initial_prompt (incierto) vs LoRA nocturno (futuro).
- 🔷 **DECISIÓN 2 (P2 — TV stop phrases):** qué frases concretas agregar. Necesita observación del usuario de sus últimos FPs reales.
- 🔷 **DECISIÓN 3 (Qwen 72B):** re-bajar los ~37GB para restaurar slow path, o deferrar hasta necesidad real (ver última sección).

---

## Task 0: Baseline — reproducir cada issue con un test que falla

**Objetivo:** antes de tocar código, capturar las 4 regresiones en tests unitarios para tener red de seguridad contra regresión futura. TDD estricto.

**Files:**
- Modify: `tests/unit/pipeline/test_request_router.py` (agregar test)
- Modify: `tests/unit/wakeword/test_whisper_wake.py` (agregar tests)
- Create: `tests/unit/tts/test_response_cache.py` (si no existe — chequear primero)
- Create: `tests/unit/home_assistant/test_ha_client_cache.py` (si no existe)

- [ ] **Step 0.1: Verificar estructura de tests existente**

Run:
```bash
ls tests/unit/pipeline/ tests/unit/wakeword/ tests/unit/tts/ tests/unit/home_assistant/ 2>/dev/null
```
Expected: ver qué archivos ya existen. Si falta alguno, crearlo vacío con `conftest.py` fixtures importadas.

- [ ] **Step 0.2: Correr la suite completa para tener baseline verde**

Run: `pytest tests/ -x -q 2>&1 | tail -20`
Expected: todos los tests existentes pasan. Si alguno falla **antes** de tocar nada, anotarlo — no es nuestro problema pero conviene no romperlo más.

- [ ] **Step 0.3: Commit checkpoint (sin cambios funcionales aún)**

```bash
git status  # debe estar clean
```
No commit todavía — los tests fallidos por regresión se escriben en cada Task.

---

## P0 · Task 1: Wake text ≠ routed text (CRÍTICO — rompe control por voz)

**Contexto del bug:** en `request_router.py:208-214`, si el `CommandEvent` trae `partial_command` (early dispatch), se usa `partial_command.raw_text` con prioridad sobre `wake_text`. Esto hace que el texto final sea la re-transcripción del streaming worker (a veces "Para encender la luz…" en vez de "Nexa encender la luz…"), y además puede perder alineación con el audio recortado.

**Hipótesis del fix:** cuando `wake_text` está presente Y es un superset léxico válido del `partial_command.raw_text` (o ≥ 80% match), preferir `wake_text` para el intent classifier. Además: loggear ambos para poder diagnosticar en producción.

**Alternativa considerada:** siempre preferir `wake_text` si está presente. Descartada porque rompería la optimización de early dispatch cuando el wake se dispara mid-utterance (el `partial_command` tiene audio más completo).

**Files:**
- Modify: `src/pipeline/request_router.py:196-240` (lógica de selección de texto)
- Test: `tests/unit/pipeline/test_request_router.py`

- [ ] **Step 1.1: Escribir test que reproduce el mismatch**

```python
# tests/unit/pipeline/test_request_router.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_event import PartialCommand  # ajustar si nombre difiere
import numpy as np

@pytest.mark.asyncio
async def test_wake_text_preferred_over_hallucinated_partial(
    request_router_factory,  # fixture existente o crear inline
):
    """
    Regresión 2026-04-24: cuando el wake detector capturó texto limpio
    ('Nexa encender la luz del escritorio') pero el early-dispatch partial
    lo alucinó como 'Para encender la luz del escritorio', el router debe
    preferir el wake_text.
    """
    router = request_router_factory()  # con orchestrator_enabled=False para simplicidad
    event = CommandEvent(
        audio=np.zeros(16000, dtype=np.float32),
        room_id="escritorio",
        wake_text="Nexa encender la luz del escritorio.",
        partial_command=PartialCommand(
            raw_text="Para encender la luz del escritorio.",
            ready_to_dispatch=True,
        ),
        early_dispatch=True,
    )
    # Mockear command_processor para capturar qué texto recibió
    router.command_processor.process_command = AsyncMock(
        return_value=MagicMock(text="encender la luz del escritorio", timings={}, user=None, emotion=None)
    )
    await router.process_command(event)
    call_kwargs = router.command_processor.process_command.await_args.kwargs
    # El pretranscribed debe venir del wake_text (tiene "Nexa"), no del partial
    assert "nexa" in call_kwargs["pretranscribed_text"].lower(), \
        f"Esperaba wake_text, recibí: {call_kwargs['pretranscribed_text']!r}"
```

- [ ] **Step 1.2: Correr el test para verificar que falla**

Run: `pytest tests/unit/pipeline/test_request_router.py::test_wake_text_preferred_over_hallucinated_partial -v`
Expected: FAIL — asserción cae porque el código actual toma `partial_command.raw_text`.

- [ ] **Step 1.3: Implementar fix — priorizar wake_text con log de disagreement**

Editar `src/pipeline/request_router.py:198-219` reemplazando el bloque de selección:

```python
pretranscribed_text: str | None = None
used_wake_text = False
early_dispatch = False
if isinstance(audio_or_event, CommandEvent):
    audio = audio_or_event.audio
    room_id = audio_or_event.room_id
    early_dispatch = audio_or_event.early_dispatch
    wake_text = audio_or_event.wake_text
    partial_text = (
        audio_or_event.partial_command.raw_text
        if audio_or_event.partial_command is not None else None
    )
    # Preferencia: wake_text > partial_command.raw_text.
    # Motivo: el wake detector transcribe el audio completo con prompt sesgado
    # a la keyword ("nexa"); el partial del streaming worker puede alucinar
    # la primera palabra ("Nexa" → "Para"). Si ambos difieren mucho, logear
    # para diagnóstico; igual elegimos wake_text.
    if wake_text:
        pretranscribed_text = wake_text
        used_wake_text = True
        if partial_text and _texts_diverge(wake_text, partial_text):
            logger.warning(
                f"Wake/partial text mismatch — using wake: "
                f"wake={wake_text!r} partial={partial_text!r}"
            )
        else:
            logger.info(
                f"Using wake-detector text as pretranscribed: {wake_text!r}"
            )
    elif partial_text is not None:
        pretranscribed_text = partial_text
else:
    audio = audio_or_event
    room_id = None
```

Y agregar al mismo archivo, cerca del top:

```python
def _texts_diverge(a: str, b: str, min_ratio: float = 0.6) -> bool:
    """True si dos transcripciones son lo bastante distintas para sospechar alucinación."""
    from difflib import SequenceMatcher
    import unicodedata
    def norm(t: str) -> str:
        t = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in t if unicodedata.category(c) != "Mn").strip()
    return SequenceMatcher(None, norm(a), norm(b)).ratio() < min_ratio
```

- [ ] **Step 1.4: Correr el test — debe pasar**

Run: `pytest tests/unit/pipeline/test_request_router.py::test_wake_text_preferred_over_hallucinated_partial -v`
Expected: PASS.

- [ ] **Step 1.5: Correr la suite completa para verificar no-regresión**

Run: `pytest tests/ -x -q`
Expected: todos verdes. Si algún test pre-existente de `request_router` asumía el orden viejo, actualizarlo en este mismo commit.

- [ ] **Step 1.6: Commit**

```bash
git add src/pipeline/request_router.py tests/unit/pipeline/test_request_router.py
git commit -m "fix(router): prefer wake_text over partial_command to avoid hallucinated re-transcriptions

Regression 2026-04-24: user said 'Nexa encender la luz' but partial
re-transcription became 'Para encender la luz', bypassing HA action.
Wake detector uses initial_prompt biased to 'nexa' so its text is more
reliable. Log divergence when texts disagree."
```

- [ ] **Step 1.7: Deploy + smoke test en server**

```bash
scp src/pipeline/request_router.py kza@192.168.1.2:/home/kza/app/src/pipeline/request_router.py
ssh kza@192.168.1.2 'systemctl --user restart kza-voice.service'
sleep 3
ssh kza@192.168.1.2 'systemctl --user is-active kza-voice.service'
```
Expected: `active`. Luego pedirle al usuario que diga "Nexa, encendé la luz del escritorio" y validar en journalctl que `Text=Nexa encender...` y que aparece la llamada a HA.

---

## P1 · Task 2: STT token coalescing ("nexa aprende") 🔷 DECISIÓN USUARIO

**Contexto del bug:** Whisper a veces pega la 'a' final de "nexa" al verbo siguiente → "nexa aprende la luz". El regex `_COMMAND_VERB_RE` no matchea porque la palabra después del wake empieza con 'a'. Agregar `aprend\w*` crearía FP con "aprendé algo nuevo" (verbo legítimo).

### 🔷 DECISIÓN USUARIO REQUERIDA — Elegí una estrategia antes de implementar

| Opción | Implementación | FP risk | Complejidad | Efecto latencia |
|--------|---------------|---------|-------------|-----------------|
| **A** Pre-proceso: si `norm` matchea `^nexa a(\w+)` → re-segmentar como `nexa <verb>` y revalidar | ~15 líneas en `whisper_wake.py` antes del verb check | Bajo — solo afecta casos post-wake | Baja | Nula |
| **B** Regex con prefijo opcional: `\ba?prend\w*\b`, `\ba?pag\w*\b` | Editar `_COMMAND_VERB_RE` | Alto — "aprend*" matchea verbos no-domótica | Muy baja | Nula |
| **C** Mejorar `initial_prompt` de WhisperWake: sumar ejemplos "nexa prendé", "nexa apagá" | Cambio en `config/settings.yaml` + init del detector | Incierto — depende de Whisper | Baja | Nula |
| **D** LoRA Whisper nocturno con voseo + wake | Plan aparte, ya hay roadmap (`wake_word_roadmap.md`) | Bajo | Alta | Nula en runtime |
| **E** Todas A+C: combinar pre-proceso con prompt mejorado | Suma de A + C | Bajo | Baja-Media | Nula |

**Recomendación:** **E** (pre-proceso + prompt). El pre-proceso cubre el caso conocido sin FPs, y el prompt mejorado reduce la probabilidad del coalescing desde la raíz. La opción D sigue siendo válida para la fase de roadmap.

**Responder antes de continuar. El resto del Task 2 asume opción E.**

**Files (asumiendo E):**
- Modify: `src/wakeword/whisper_wake.py:62-68` (helper `_normalize` — agregar función nueva `_decoalesce_post_wake`)
- Modify: `src/wakeword/whisper_wake.py:462-496` (usar decoalesce antes de `_COMMAND_VERB_RE.search`)
- Modify: `config/settings.yaml` (sección `wake_word.whisper.initial_prompt`)
- Test: `tests/unit/wakeword/test_whisper_wake.py`

- [ ] **Step 2.1: Escribir test para decoalesce**

```python
# tests/unit/wakeword/test_whisper_wake.py
from src.wakeword.whisper_wake import _decoalesce_post_wake, _normalize

def test_decoalesce_nexa_aprende_to_nexa_prende():
    """Whisper pega 'a' final de nexa al verbo. Re-segmentar."""
    text = _normalize("nexa aprende la luz del escritorio")
    fixed = _decoalesce_post_wake(text, wake_norm="nexa")
    assert fixed == "nexa prende la luz del escritorio"

def test_decoalesce_nexa_apaga_to_nexa_apaga():
    """'nexa apagá' — no debería tocar, ya es un verbo válido."""
    text = _normalize("nexa apaga la luz")
    fixed = _decoalesce_post_wake(text, wake_norm="nexa")
    assert fixed == "nexa apaga la luz"

def test_decoalesce_leaves_real_aprende():
    """'nexa aprendé español' — 'aprendé' es verbo real, pero no de domótica.
    Decoalesce cambia a 'prendé', regex matchea, el router de arriba decide.
    Esto es aceptable: preferimos FP en wake a FN (mejor preguntar de más)."""
    # Documentación del trade-off, no asserción. Este caso es raro en domótica.
    pass

def test_decoalesce_no_wake_match_passthrough():
    """Si texto no empieza con 'nexa', no tocar."""
    text = _normalize("che prendé la luz")
    fixed = _decoalesce_post_wake(text, wake_norm="nexa")
    assert fixed == text
```

- [ ] **Step 2.2: Correr tests para verificar que fallan**

Run: `pytest tests/unit/wakeword/test_whisper_wake.py -v -k decoalesce`
Expected: FAIL — `_decoalesce_post_wake` no existe aún.

- [ ] **Step 2.3: Implementar `_decoalesce_post_wake`**

En `src/wakeword/whisper_wake.py`, agregar después de `_normalize`:

```python
# Prefijos de verbos de comando que pueden quedar coalescidos con la 'a' final
# de "nexa". Mapeo: forma_coalescida → forma_real.
_COALESCED_VERB_PREFIXES = {
    "aprend": "prend",   # nexa aprendé → nexa prendé
    "apag": "apag",      # ya correcto (nexa apagá), no modificar
    "aencend": "encend", # nexa aencendé → nexa encendé (raro pero posible)
    "abaj": "baj",       # nexa abajá → nexa bajá
    "asub": "sub",       # nexa asubí → nexa subí
}


def _decoalesce_post_wake(norm_text: str, wake_norm: str) -> str:
    """Corregir pegado del wake con el verbo siguiente.

    Whisper ocasionalmente produce 'nexa aprende' cuando el usuario dijo
    'nexa, prendé' — la 'a' final del wake se pega al inicio del verbo.
    Detectamos y re-segmentamos las combinaciones conocidas.

    Args:
        norm_text: texto ya pasado por `_normalize`.
        wake_norm: wake word en forma normalizada (ej: 'nexa').

    Returns:
        Texto corregido, o el original si no había coalescing detectable.
    """
    words = norm_text.split()
    if len(words) < 2 or words[0] != wake_norm:
        return norm_text
    next_word = words[1]
    for coalesced, real in _COALESCED_VERB_PREFIXES.items():
        if next_word.startswith(coalesced) and coalesced != real:
            # Re-ensamblar: reemplazar prefijo
            suffix = next_word[len(coalesced):]
            words[1] = real + suffix
            return " ".join(words)
    return norm_text
```

Luego modificar `_transcribe_and_match` en `whisper_wake.py:462`, después de `norm = _normalize(text)`:

```python
norm = _normalize(text)
# Fix coalescing Whisper: 'nexa aprende' → 'nexa prende'
for wake_norm in self.wake_words_norm:
    norm_fixed = _decoalesce_post_wake(norm, wake_norm)
    if norm_fixed != norm:
        logger.info(f"Decoalesced post-wake: {norm!r} → {norm_fixed!r}")
        norm = norm_fixed
        break
logger.info(f"WhisperWake [{dur_ms:.0f}ms→{stt_ms:.0f}ms]: {norm!r}")
```

- [ ] **Step 2.4: Correr tests — deben pasar**

Run: `pytest tests/unit/wakeword/test_whisper_wake.py -v -k decoalesce`
Expected: PASS (3 tests — el 4º es documentación).

- [ ] **Step 2.5: Agregar test de integración del path completo**

```python
def test_transcribe_and_match_accepts_nexa_aprende():
    """Path end-to-end: 'nexa aprende la luz' debe triggerar wake tras decoalesce."""
    # Requiere fixture whisper_mock. Ver conftest.py existente.
    from src.wakeword.whisper_wake import WhisperWakeDetector
    # ... usar fixture existente si la hay, o armar mock inline con
    # whisper.transcribe → returns segments con text='nexa aprende la luz'
    # assert: detector.detect(...) devuelve 'nexa' (match), pending_text set
    pass  # completar con fixture existente — ver cómo se testean otros matches
```

**Nota:** si no hay fixture de whisper mock en `tests/unit/wakeword/conftest.py`, copiar el patrón de otro test del mismo archivo antes de escribir este. Si no existe ninguno, omitir este step y dejar solo los unit tests de `_decoalesce_post_wake`.

- [ ] **Step 2.6: Actualizar initial_prompt**

Editar `config/settings.yaml`, sección del wake word Whisper. Buscar la línea con `initial_prompt`:

```bash
grep -n "initial_prompt" config/settings.yaml
```

Actualizar (mantener ejemplos cortos, ~200 chars):

```yaml
  initial_prompt: "Nexa, prendé la luz. Nexa, apagá la tele. Nexa, subí el volumen. Nexa, bajá la persiana. Nexa, encendé el ventilador. Nexa, abrí la cortina."
```

- [ ] **Step 2.7: Commit P1**

```bash
git add src/wakeword/whisper_wake.py tests/unit/wakeword/test_whisper_wake.py config/settings.yaml
git commit -m "fix(wake): decoalesce 'nexa a<verb>' post-Whisper + biased prompt

Whisper occasionally coalesces 'nexa' final 'a' with the next verb,
producing 'nexa aprende' instead of 'nexa prende' and failing the
command verb regex. Add targeted re-segmentation for known coalesced
prefixes and bias initial_prompt with voseo command examples."
```

- [ ] **Step 2.8: Deploy + smoke**

```bash
scp src/wakeword/whisper_wake.py kza@192.168.1.2:/home/kza/app/src/wakeword/whisper_wake.py
scp config/settings.yaml kza@192.168.1.2:/home/kza/app/config/settings.yaml
ssh kza@192.168.1.2 'systemctl --user restart kza-voice.service'
```

Validar con `journalctl`:
```bash
ssh kza@192.168.1.2 'journalctl --user -u kza-voice.service -f | grep -E "Decoalesced|Wake word"'
```

---

## P2 · Task 3: TV false positives — expandir stop phrases 🔷 DECISIÓN USUARIO

**Contexto:** User reportó "2 ejecuciones sin éxito" — hay FPs nuevos no cubiertos por la lista actual. Antes de adivinar, pedirle al usuario los textos exactos que vio en los logs.

### 🔷 DECISIÓN USUARIO REQUERIDA — Frases a agregar

**Pedirle al usuario:**
1. Correr el siguiente comando en el server para extraer rejections y FPs recientes:
   ```bash
   ssh kza@192.168.1.2 'journalctl --user -u kza-voice.service --since "2 days ago" | \
     grep -E "Wake rejected|Wake word .* detectado" | tail -100'
   ```
2. De ahí, compartir las transcripciones de TV que **no** fueron rechazadas pero deberían haberlo sido (FPs reales).
3. Y los outputs del TV que sí fueron rechazados — validar que la regla correcta hizo el trabajo.

**Output esperado del usuario:** lista de ~5-15 frases/subcadenas nuevas, ej:
```
"descargá la app"
"seguinos en instagram"
"no te lo pierdas"
"lo que viene a continuación"
...
```

**Files:**
- Modify: `src/wakeword/whisper_wake.py:50-59` (lista `_TV_STOP_PHRASES`)
- Test: `tests/unit/wakeword/test_whisper_wake.py`

- [ ] **Step 3.1: Escribir tests parametrizados de rejection**

```python
# tests/unit/wakeword/test_whisper_wake.py
import pytest
from src.wakeword.whisper_wake import _TV_STOP_PHRASES, _normalize

# ACTUALIZAR con frases del usuario (DECISIÓN P2)
NEW_TV_FALSE_POSITIVES = [
    # "nexa descargá la app gratis",         # ← ejemplo, reemplazar con reales
    # "suscribite nexa al canal",
    # "nexa seguinos en instagram",
]

@pytest.mark.parametrize("utterance", NEW_TV_FALSE_POSITIVES)
def test_new_tv_phrases_have_stop_match(utterance):
    """Cada FP reportado por el usuario debe matchear algún stop phrase."""
    norm = _normalize(utterance)
    assert any(phrase in norm for phrase in _TV_STOP_PHRASES), \
        f"No stop phrase cubre: {utterance!r}"

VALID_COMMANDS_NOT_REJECTED = [
    "nexa prendé la luz del escritorio",
    "nexa apagá la tele del living",
    "nexa subí el volumen al cincuenta por ciento",
    "nexa bajá las persianas del cuarto",
]

@pytest.mark.parametrize("utterance", VALID_COMMANDS_NOT_REJECTED)
def test_valid_commands_do_not_match_tv_stop(utterance):
    """Comandos legítimos NO deben matchear stop phrases (no FN)."""
    norm = _normalize(utterance)
    assert not any(phrase in norm for phrase in _TV_STOP_PHRASES), \
        f"Comando legítimo matchea stop phrase: {utterance!r}"
```

- [ ] **Step 3.2: Correr tests — los de FP fallan, los de FN pasan**

Run: `pytest tests/unit/wakeword/test_whisper_wake.py -v -k "tv"`
Expected: tests de `NEW_TV_FALSE_POSITIVES` FAIL, `VALID_COMMANDS_NOT_REJECTED` PASS.

- [ ] **Step 3.3: Expandir `_TV_STOP_PHRASES` con las frases del usuario**

Editar `src/wakeword/whisper_wake.py:52-59`:

```python
_TV_STOP_PHRASES = (
    # YouTube/streamers
    "suscribe", "suscrib", "campanita", "gracias por ver",
    "dale like", "dale lie", "dale mega like",
    "canal de youtube", "activa la",
    # NUEVAS — FPs reportados 2026-04-24 (completar desde DECISIÓN P2)
    # "descarg la app",
    # "seguinos en",
    # "no te lo pierdas",
    # "a continuacion",
)
```

**IMPORTANTE:** usar forma normalizada (sin acentos, lowercase) porque el match es substring sobre `norm`. Ej: `"descargá"` guardar como `"descarg"`.

- [ ] **Step 3.4: Correr tests de nuevo**

Run: `pytest tests/unit/wakeword/test_whisper_wake.py -v -k "tv"`
Expected: todos PASS.

- [ ] **Step 3.5: Commit P2**

```bash
git add src/wakeword/whisper_wake.py tests/unit/wakeword/test_whisper_wake.py
git commit -m "fix(wake): expand TV stop phrases from observed FPs

Captured from production journalctl 2026-04-24: commercials and
YouTube intros reaching the wake detector. Add parametrized tests
over both FP list and legitimate-command guard to prevent overreach."
```

- [ ] **Step 3.6: Deploy + observación 24h**

```bash
scp src/wakeword/whisper_wake.py kza@192.168.1.2:/home/kza/app/src/wakeword/whisper_wake.py
ssh kza@192.168.1.2 'systemctl --user restart kza-voice.service'
```

Dejar correr 24h. Al día siguiente, re-correr el grep de journalctl y contar ratio `Wake rejected tv_stop_phrase / Wake rejected total`.

---

## P3 · Task 4: HA state prefetch cache no hit (latencia 288ms evitable)

**Contexto del bug:** En "apagá la luz del escritorio", el timing muestra `home_assistant=288ms` — una call REST completa. Si el prefetch WebSocket funcionara, debería ser <5ms (lookup en `_state_cache`). Hipótesis: el call path del dispatcher no consulta `_state_cache` antes de llamar REST, o lo consulta pero el entity no está en cache al momento del command.

**Files (exploración primero, implementación después):**
- Read: `src/home_assistant/ha_client.py` (métodos `call_service`, `get_state`, `_state_cache` usage)
- Read: `src/orchestrator/request_dispatcher.py` (call sites a HA)
- Modify: TBD según hallazgo
- Test: `tests/unit/home_assistant/test_ha_client_cache.py`

- [ ] **Step 4.1: Auditar call sites del cache**

Run:
```bash
grep -rn "_state_cache\|get_state\|call_service" src/home_assistant/ src/orchestrator/ | grep -v test | head -40
```
Anotar: (1) dónde se lee el cache, (2) qué métodos lo usan, (3) si `call_service` previo hace `get_state` y si ese `get_state` usa cache.

- [ ] **Step 4.2: Instrumentar con log temporal para confirmar hipótesis**

Agregar log en `ha_client.py` en el método `get_state` (o similar):

```python
async def get_state(self, entity_id: str) -> dict | None:
    cached = self._state_cache.get(entity_id)
    if cached is not None:
        logger.debug(f"HA cache HIT: {entity_id}")
        return cached
    logger.info(f"HA cache MISS: {entity_id} — hitting REST")
    # ... código existente
```

Deploy temporal:
```bash
scp src/home_assistant/ha_client.py kza@192.168.1.2:/home/kza/app/src/home_assistant/ha_client.py
ssh kza@192.168.1.2 'systemctl --user restart kza-voice.service'
```

Pedirle al usuario 3 comandos de domótica seguidos. Revisar:
```bash
ssh kza@192.168.1.2 'journalctl --user -u kza-voice.service --since "5 min ago" | grep "HA cache"'
```

- [ ] **Step 4.3: Clasificar el bug según evidencia**

**Caso A** — "HA cache MISS" aparece siempre: prefetch no está poblando el cache. Verificar que `_state_sync_running` está True al startup y que el primer snapshot REST (`_refresh_full_state_snapshot`) se ejecuta.

**Caso B** — "HA cache HIT" aparece pero timing sigue en 288ms: el call path no usa el resultado del cache, sino que igual hace REST después (bug en call_service).

**Caso C** — El path fast-domótica directamente no llama `get_state`, va directo a `call_service` (que necesariamente es REST) sin consultar cache para decidir si el state ya es el target (idempotency short-circuit).

El fix concreto depende del caso. Detenerse y reportar al usuario cuál caso aplica antes de implementar.

- [ ] **Step 4.4: Fix según caso (placeholder — completar tras Step 4.3)**

**Si Caso A:** verificar `start_state_sync` se llama en `main.py` DI — si no, agregarlo. Test:
```python
async def test_state_cache_populated_after_start(ha_client_mock):
    await ha_client_mock.start_state_sync()
    # mock WS devuelve snapshot con light.escritorio
    assert ha_client_mock._state_cache.get("light.escritorio") is not None
```

**Si Caso B:** refactorizar `call_service` para que no re-hidrate state post-call; confiar en el event WS subsiguiente.

**Si Caso C:** agregar short-circuit en dispatcher:
```python
# Antes de HA call: si cached state ya es target, skip
cached = ha_client._state_cache.get(entity_id)
if cached and cached["state"] == target_state:
    logger.info(f"Skip HA call — {entity_id} ya está {target_state}")
    return {"success": True, "from_cache": True}
```
Con su test correspondiente.

- [ ] **Step 4.5: Medir latencia antes/después**

Run (en server):
```bash
python tools/benchmark_latency.py --iterations 20 --path fast
```
Comparar mediana `home_assistant=` antes y después del fix. Objetivo: <50ms para el caso idempotente.

- [ ] **Step 4.6: Commit P3**

```bash
git add src/home_assistant/ha_client.py src/orchestrator/request_dispatcher.py tests/unit/home_assistant/
git commit -m "fix(ha): use prefetch cache on fast-path to avoid REST roundtrip

State prefetch via WS was populating _state_cache but the fast-path
dispatcher still hit REST for every command (~288ms). [Detailed fix
per diagnosed case: A/B/C]. Adds cache-hit/miss observability logs."
```

---

## P4 · Task 5: Warnings non-fatal al startup

### 5A: `FeatureManager start failed: 'Zone' object has no attribute 'media_player_entity'`

**Contexto:** `intercom_system.py:165-167` lee `zone.media_player_entity`, `zone.speaker_entity`, `zone.tts_target` — ninguno existe en el `@dataclass Zone` (`zone_manager.py:25`).

**Dos opciones:**
- **5A.1** Agregar los campos al dataclass `Zone` con defaults vacíos (compatible hacia atrás).
- **5A.2** Usar `getattr(zone, "media_player_entity", None)` en `intercom_system.py`.

**Preferir 5A.1** — si el intercom los necesita, deberían existir como contrato tipado, aunque sean `None` cuando la zona no está configurada para intercom.

**Files:**
- Modify: `src/audio/zone_manager.py:25-52`
- Test: `tests/unit/audio/test_zone_manager.py` (o crear)
- Test: `tests/unit/intercom/test_intercom_system.py` (smoke del load)

- [ ] **Step 5A.1: Test que reproduce el crash**

```python
# tests/unit/intercom/test_intercom_system.py
import pytest
from src.intercom.intercom_system import IntercomSystem
from src.audio.zone_manager import Zone, ZoneManager

@pytest.mark.asyncio
async def test_intercom_loads_zones_without_media_player_fields():
    """Regresión: Zone no tenía media_player_entity → FeatureManager fallaba."""
    zone = Zone(id="zone_1", name="Test", mic_device_index=0, ma1260_zone=1)
    zm = ZoneManager()
    zm._zones = {"zone_1": zone}
    intercom = IntercomSystem(zone_manager=zm, ha_client=None)
    # No debe crashear; debe cargar con media_player=None o similar
    await intercom._load_zones()
    assert "zone_1" in intercom._zones
```

- [ ] **Step 5A.2: Correr — falla con AttributeError**

Run: `pytest tests/unit/intercom/test_intercom_system.py::test_intercom_loads_zones_without_media_player_fields -v`
Expected: FAIL — `AttributeError: 'Zone' object has no attribute 'media_player_entity'`.

- [ ] **Step 5A.3: Extender dataclass `Zone`**

Editar `src/audio/zone_manager.py:25-42`:

```python
@dataclass
class Zone:
    """Configuración de una zona de audio"""
    id: str
    name: str
    mic_device_index: int
    ma1260_zone: int

    state: ZoneState = ZoneState.IDLE
    last_activity: float = 0.0
    volume: int = 50

    noise_floor: float = 0.01
    detection_threshold: float = 0.05

    default_users: list[str] = field(default_factory=list)

    # Integración con intercom/media — opcionales por zona.
    media_player_entity: str | None = None
    speaker_entity: str | None = None
    tts_target: str | None = None
```

- [ ] **Step 5A.4: Correr test — pasa**

Run: `pytest tests/unit/intercom/test_intercom_system.py -v`
Expected: PASS.

- [ ] **Step 5A.5: Correr suite completa**

Run: `pytest tests/ -x -q`
Expected: todos verdes. Chequear que `Zone.to_dict()` no tire KeyError con los nuevos campos (si los incluye, agregarlos).

### 5B: TTS response cache — todas las frases "audio vacío"

**Contexto:** `response_cache.py:133` loggea "audio vacío" para cada frase → 0 cacheadas post-S2. Significa que `_synthesize` retorna `None` o ndarray vacío para todo.

**Files:**
- Modify: `src/tts/response_cache.py:154-182` (diagnóstico de `_synthesize`)
- Test: `tests/unit/tts/test_response_cache.py`

- [ ] **Step 5B.1: Diagnosticar con log detallado**

Agregar en `_synthesize` después de `result = fn(text)` (línea ~170):

```python
logger.debug(
    f"TTS cache._synthesize: method={method_name}, "
    f"result_type={type(result).__name__}, "
    f"is_tuple={isinstance(result, (tuple, list))}, "
    f"len={len(result) if hasattr(result, '__len__') else 'n/a'}"
)
```

Deploy temporal, restart, ver logs startup:
```bash
ssh kza@192.168.1.2 'journalctl --user -u kza-voice.service --since "1 min ago" | grep "TTS cache"'
```

- [ ] **Step 5B.2: Clasificar**

**Caso D1:** `result_type=NoneType` → TTS.synthesize devuelve None. Ver el TTS real (DualTTS o Kokoro) y qué devuelve.

**Caso D2:** `result_type=tuple` pero el ndarray está vacío — ejecutar `np.asarray(item).shape` y ver.

**Caso D3:** `result_type=Coroutine` no await: el chequeo `iscoroutine` está, pero quizás el retorno de la coroutine vuelve a ser coroutine (doble await needed).

- [ ] **Step 5B.3: Test unitario con mock de cada forma de retorno**

```python
# tests/unit/tts/test_response_cache.py
import numpy as np
import pytest
from src.tts.response_cache import ResponseCache

class MockTTSTuple:
    sample_rate = 24000
    def synthesize(self, text):
        return (np.ones(24000, dtype=np.float32), 100.0)  # (audio, elapsed_ms)

class MockTTSTupleEngine:
    sample_rate = 24000
    async def synthesize(self, text):
        return (np.ones(24000, dtype=np.float32), 100.0, "kokoro")

class MockTTSNone:
    sample_rate = 24000
    def synthesize(self, text):
        return None

@pytest.mark.asyncio
async def test_cache_builds_from_tuple_return():
    cache = ResponseCache(MockTTSTuple())
    await cache.build()
    assert cache.get("Listo") is not None

@pytest.mark.asyncio
async def test_cache_builds_from_async_tuple_engine():
    cache = ResponseCache(MockTTSTupleEngine())
    await cache.build()
    assert cache.get("Listo") is not None

@pytest.mark.asyncio
async def test_cache_skips_silently_on_none():
    cache = ResponseCache(MockTTSNone())
    await cache.build()
    assert cache.get("Listo") is None  # Sin crash
```

Run: `pytest tests/unit/tts/test_response_cache.py -v`
Expected: según diagnóstico. Si falla el de tupla, hay bug en `_extract_ndarray`; si pasa, el bug está en el TTS real.

- [ ] **Step 5B.4: Fix según diagnóstico**

**Si D1 (TTS devuelve None al startup):** el issue es que el modelo TTS no está warmed up al momento de `build()`. Solución: en `response_cache.build()`, hacer una synth "dummy" antes del loop y verificar que funciona; si no, esperar.

**Si D2/D3:** ajustar `_extract_ndarray` o `_synthesize` para manejar el shape real.

- [ ] **Step 5B.5: Commit P4**

```bash
git add src/audio/zone_manager.py src/tts/response_cache.py tests/unit/audio tests/unit/intercom tests/unit/tts
git commit -m "fix(pipeline): non-fatal startup warnings — Zone intercom fields + TTS cache

- Zone dataclass: add optional media_player_entity, speaker_entity,
  tts_target for intercom integration (was AttributeError).
- TTS response cache: fix [diagnosed case] so startup synth of canonical
  phrases actually populates cache (was logging 'audio vacío' for all)."
```

- [ ] **Step 5B.6: Deploy + verificación startup**

```bash
scp src/audio/zone_manager.py kza@192.168.1.2:/home/kza/app/src/audio/zone_manager.py
scp src/tts/response_cache.py kza@192.168.1.2:/home/kza/app/src/tts/response_cache.py
ssh kza@192.168.1.2 'systemctl --user restart kza-voice.service'
sleep 15  # startup time con warmup
ssh kza@192.168.1.2 'journalctl --user -u kza-voice.service --since "1 min ago" | grep -E "FeatureManager|response cache listo|audio vacío"'
```

Expected:
- `FeatureManager started` (sin el warning).
- `TTS response cache listo: N frases cacheadas` con N>0.
- Ninguna línea `audio vacío`.

---

## 🔷 DECISIÓN 3: Re-bajar Qwen 72B Q8_0?

**Estado actual:** el archivo no está en `/home/kza/kza/models/Qwen2.5-72B-Instruct-Q8_0/`. `kza-72b.service` va a fallar o arrancar vacío. Esto **no bloquea** los Tasks 1-5 — todo el pipeline fast-path de domótica no usa el 72B.

**Opciones:**

| | Tiempo | Disco | Cuándo elegir |
|--|--|--|--|
| **Re-bajar** | ~2-3h descarga ~37GB + verify | +37GB en `/home/kza/kza/models/` | Si vas a usar el slow path (razonamiento complejo, conversaciones largas) esta semana |
| **Deferrar** | 0 | 0 | Si el uso es 100% fast-path (domótica + música) por ahora. El service quedará en `failed`; limpiar del journalctl con `systemctl --user mask kza-72b.service` temporal |
| **Probar con 7B como fallback** | 10 min config | 0 (7B ya está en :8100) | Si querés slow path pero con calidad menor; editar `reasoner.py` para usar vLLM :8100 como endpoint LLM completo |

**Comando para re-bajar (si elegís opción 1):**
```bash
ssh kza@192.168.1.2 '
  cd /home/kza/kza/models &&
  huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF \
    Qwen2.5-72B-Instruct-Q8_0.gguf \
    --local-dir Qwen2.5-72B-Instruct-Q8_0
'
```

**Comando para enmascarar el service (si elegís opción 2):**
```bash
ssh kza@192.168.1.2 'systemctl --user mask kza-72b.service'
```

Actualizar el memo `project_wake_tv_filter_pipeline_regression.md` con la decisión final.

---

## Resumen ejecutivo

| Prioridad | Task | Archivos principales | Esfuerzo | Bloquea |
|-----------|------|---------------------|----------|---------|
| P0 | 1 — Wake text priority | `request_router.py` | ~30min | Control por voz |
| P1 | 2 — STT decoalesce 🔷 | `whisper_wake.py`, `settings.yaml` | ~1h | Wake reliability |
| P2 | 3 — TV stop phrases 🔷 | `whisper_wake.py` | ~20min + 24h obs | Feature objective |
| P3 | 4 — HA cache hit | `ha_client.py`, `request_dispatcher.py` | ~1-2h (depende del caso) | Latencia <300ms |
| P4 | 5A — Zone fields | `zone_manager.py` | ~15min | Warning ruido |
| P4 | 5B — TTS cache 🔎 | `response_cache.py` | ~30-60min (depende del caso) | Latencia respuestas |
| — | Decisión Qwen 72B 🔷 | — | — | Slow path |

🔷 = decisión de usuario requerida · 🔎 = diagnóstico previo requerido

**Orden sugerido de ejecución:** P0 → P4.5A (quick win, warnings fuera) → pedir decisiones P1/P2 al user → P1 → P2 → P3 → P4.5B.

**Total estimado:** 4-6 horas hands-on, más 24h observación P2.
