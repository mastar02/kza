# Motor de gramática determinístico — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reescribir `command_grammar` como un motor de reglas declarativo que cubre on/off + set de luz, persianas, ventilador y media (ruteada al music path existente), de modo que la domótica simple nunca toque el LLMRouter débil; y arreglar el modo de falla del LLMRouter (timeout/error → no descartar comandos válidos como `noise`).

**Architecture:** Tabla declarativa `INTENT_RULES` (datos) + matriz de compatibilidad intent↔dominio + `CommandGrammar.parse()` que produce un `ParsedCommand{intent, domain, room, slots, target, confidence, quality}`. `request_router` decide por `target`/`quality`: domótica `full` bypassa el LLM; media `full` pasa el gate como comando válido y el dispatcher la rutea a su music path; lo demás cae al LLMRouter (que ya no marca timeout como `noise`). Vector search se mantiene para resolver la entidad concreta.

**Tech Stack:** Python 3.13, pytest, dataclasses, regex (`re`), unicodedata. Sin dependencias nuevas.

**Entorno de ejecución:** El código vive en el server (`kza@192.168.1.2:/home/kza/app`). Ejecutar tests con `.venv/bin/python -m pytest`. Commits en `main` del server; push desde la laptop (`git fetch kza:/home/kza/app main` + push del SHA — el server no tiene auth a GitHub). Ver `docs/SESSION_2026-05-30_XVF3800_WAKE_NLU_FIXES.md`.

**Spec:** `docs/superpowers/specs/2026-05-30-deterministic-grammar-engine-design.md`.

---

## File Structure

- `src/nlu/command_grammar.py` — **reescritura**: motor `CommandGrammar.parse()`, `IntentRule`, `INTENT_RULES`, matriz de compatibilidad, `ParsedCommand`. Conserva `extract_entity`/`extract_room`/`has_wake_word`. `parse_partial_command` queda como wrapper fino. Puro, sin I/O.
- `src/nlu/slot_extractor.py` — **modificar**: agregar `extract_volume()` y constante `SLOT_VOLUME`. `classify_intent` se mantiene (lo usa el motor para el verbo), pero la decisión de intent final pasa por la tabla.
- `src/pipeline/request_router.py` — **modificar**: `_grammar_fastpath_classification` usa el motor y respeta `target`; manejo de `rejection_reason="unavailable"`; eliminar el bloque muerto `regex_extractor+llm_gate`.
- `src/nlu/llm_router.py` — **modificar**: timeout/error → `rejection_reason="unavailable"`, `is_command=None`. Agregar `"unavailable"` al enum.
- `config/settings.yaml` — **modificar**: quitar el bloque `nlu.fast_path` muerto (regex+gate).
- `tests/unit/nlu/test_command_grammar.py` — **modificar/extender**: tests table-driven de los intents nuevos + matriz de compatibilidad. Conservar los existentes.
- `tests/unit/nlu/test_slot_extractor_volume.py` — **crear**: tests de `extract_volume`.
- `tests/unit/nlu/test_llm_router.py` — **modificar**: timeout/error → `unavailable`.
- `tests/unit/pipeline/test_request_router_grammar.py` — **crear**: routing por target/quality + fallback `unavailable`.

---

## Task 1: Slot extractor de volumen

**Files:**
- Modify: `src/nlu/slot_extractor.py`
- Test: `tests/unit/nlu/test_slot_extractor_volume.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nlu/test_slot_extractor_volume.py
import pytest
from src.nlu.slot_extractor import extract_volume, SLOT_VOLUME, extract_slots


@pytest.mark.parametrize("text,expected", [
    ("subí el volumen al 40", 40),
    ("ponelo al 40%", 40),
    ("volumen 80", 80),
    ("más fuerte", 90),
    ("bajito", 20),
    ("ponelo bien fuerte", 90),
    ("la luz al 50", None),          # 'al N' sin contexto de volumen → no es volumen
])
def test_extract_volume(text, expected):
    assert extract_volume(text) == expected


def test_volume_in_extract_slots():
    slots = extract_slots("subí el volumen al 30")
    assert slots.get(SLOT_VOLUME) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_slot_extractor_volume.py -v`
Expected: FAIL con `ImportError: cannot import name 'extract_volume'`.

- [ ] **Step 3: Implementar `extract_volume` y `SLOT_VOLUME`**

En `src/nlu/slot_extractor.py`, agregar la constante junto a las otras (`SLOT_BRIGHTNESS`, etc.):

```python
SLOT_VOLUME = "volume_pct"
```

Y la función (después de `extract_color_temp`):

```python
# ============================================================
# Volume (media_player)
# ============================================================
_RE_VOLUME_NUM = re.compile(
    r"\bvolumen\s+(?:al\s+|en\s+)?(\d{1,3})\b|"
    r"\b(?:ponelo|pon[eé]|subilo|bajalo)\s+(?:al\s+|en\s+)?(\d{1,3})\s*(?:%|por\s*ciento)?\b",
    re.IGNORECASE,
)
VOLUME_WORDS = {
    "fuerte": 90, "alto": 90, "alta": 90, "fortísimo": 100, "fortisimo": 100,
    "bajito": 20, "bajo": 20, "baja": 20, "suave": 30,
    "medio": 50, "media": 50,
}


def extract_volume(text: str) -> int | None:
    """Extrae volume_pct (0-100) SOLO si hay contexto de volumen. None si no."""
    t = text.lower()
    # Solo interpretamos números como volumen si la palabra 'volumen' o un verbo
    # de volumen está presente — evita robar 'al 50' de un comando de brillo.
    has_volume_context = ("volumen" in t or
                          any(re.search(rf"\b{v}\b", t)
                              for v in ("subilo", "bajalo", "ponelo")))
    if has_volume_context:
        m = _RE_VOLUME_NUM.search(t)
        if m:
            num = next((g for g in m.groups() if g), None)
            if num is not None:
                v = int(num)
                if 0 <= v <= 100:
                    return v
    for word, val in VOLUME_WORDS.items():
        # Las palabras de volumen requieren contexto de audio/volumen para no
        # pisar brightness ('luz fuerte'); exigimos 'volumen' o verbo de media.
        if has_volume_context and re.search(rf"\b{re.escape(word)}\b", t):
            return val
    return None
```

Y en `extract_slots`, agregar al final (antes del `return slots`):

```python
    vol = extract_volume(text)
    if vol is not None:
        slots[SLOT_VOLUME] = vol
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_slot_extractor_volume.py -v`
Expected: PASS (7 + 1 casos).

- [ ] **Step 5: Verificar que no rompe slots existentes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/ -k "slot or brightness or color" -v`
Expected: PASS (sin regresiones).

- [ ] **Step 6: Commit**

```bash
git add src/nlu/slot_extractor.py tests/unit/nlu/test_slot_extractor_volume.py
git commit -m "feat(nlu): extract_volume para comandos de media (volume_pct con contexto)"
```

---

## Task 2: `IntentRule` + tabla `INTENT_RULES`

**Files:**
- Modify: `src/nlu/command_grammar.py`
- Test: `tests/unit/nlu/test_command_grammar.py`

- [ ] **Step 1: Write the failing test**

Agregar a `tests/unit/nlu/test_command_grammar.py`:

```python
from src.nlu.command_grammar import IntentRule, INTENT_RULES, match_intent_rules


def test_intent_rules_cover_expected_intents():
    intents = {r.intent for r in INTENT_RULES}
    assert intents == {
        "turn_on", "turn_off", "set", "open", "close",
        "media_play", "media_pause", "media_next", "volume_set",
    }


@pytest.mark.parametrize("text,domain,expected_intent", [
    ("prendé la luz", "light", "turn_on"),
    ("apagá la luz", "light", "turn_off"),
    ("subí la persiana", "cover", "open"),
    ("bajá la persiana", "cover", "close"),
    ("subí el volumen", "media_player", "volume_set"),
    ("pausá la música", "media_player", "media_pause"),
    ("poné música", "media_player", "media_play"),
    ("abrí la luz", "light", None),       # open no aplica a light → sin match
    ("subí la luz", "light", "turn_on"),  # 'subí' con light → turn_on gana por compat
])
def test_match_intent_rules_respects_domain(text, domain, expected_intent):
    rule = match_intent_rules(text, domain)
    assert (rule.intent if rule else None) == expected_intent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "intent_rules" -v`
Expected: FAIL con `ImportError: cannot import name 'IntentRule'`.

- [ ] **Step 3: Implementar `IntentRule`, `INTENT_RULES`, `match_intent_rules`**

En `src/nlu/command_grammar.py`, después de los imports y `_norm`, agregar:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentRule:
    """Regla declarativa de intent. Datos, no código."""
    intent: str
    verb_patterns: tuple[str, ...]   # regex (con \b alrededor al compilar)
    domains: frozenset[str]          # dominios HA donde aplica
    target: str = "domotics"         # "domotics" | "music"
    requires_slot: str | None = None # "any" | "volume" | None


# Orden = prioridad cuando varias reglas matchean el verbo. turn_on/off antes
# que open/close para que 'subí la luz' (light) caiga en turn_on, no open.
INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule("turn_on",  (r"prend\w*", r"encend\w*", r"ilumin\w*", r"activ\w*", r"enciend\w*"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("turn_off", (r"apag\w*", r"cort\w*", r"desactiv\w*", r"apaguen"),
               frozenset({"light", "fan", "climate", "switch"})),
    IntentRule("set",      (),  frozenset({"light"}), requires_slot="any"),
    IntentRule("open",     (r"sub\w*", r"abr\w*", r"levant\w*"),
               frozenset({"cover"})),
    IntentRule("close",    (r"baj\w*", r"cerr\w*"),
               frozenset({"cover"})),
    IntentRule("media_play",  (r"pon\w*", r"reproduc\w*", r"dale", r"segu\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("media_pause", (r"paus\w*", r"par\w*", r"fren\w*", r"callate", r"silenci\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("media_next",  (r"siguiente", r"proxim\w*", r"cambi\w*", r"salt\w*"),
               frozenset({"media_player"}), target="music"),
    IntentRule("volume_set",  (r"volumen", r"fuerte", r"bajito"),
               frozenset({"media_player"}), target="music", requires_slot="volume"),
)


def _rule_verb_matches(rule: IntentRule, norm_text: str) -> bool:
    for pat in rule.verb_patterns:
        if re.search(rf"\b{pat}\b", norm_text):
            return True
    return False


def match_intent_rules(text: str, domain: str | None) -> IntentRule | None:
    """Devuelve la primera IntentRule cuyo verbo matchea Y es compatible con
    el dominio. Si domain es None, no se puede validar compat → None salvo que
    la regla aplique a cualquier dominio (no hay tales reglas hoy)."""
    if domain is None:
        return None
    t = _norm(text)
    for rule in INTENT_RULES:
        if domain not in rule.domains:
            continue
        if rule.intent == "set":
            # 'set' no tiene verbo; lo decide el motor por presencia de slots.
            continue
        if _rule_verb_matches(rule, t):
            return rule
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "intent_rules" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlu/command_grammar.py tests/unit/nlu/test_command_grammar.py
git commit -m "feat(nlu): tabla declarativa INTENT_RULES + match por compat intent-dominio"
```

---

## Task 3: `ParsedCommand` + `CommandGrammar.parse()`

**Files:**
- Modify: `src/nlu/command_grammar.py`
- Test: `tests/unit/nlu/test_command_grammar.py`

- [ ] **Step 1: Write the failing test**

```python
from src.nlu.command_grammar import ParsedCommand, parse_command


@pytest.mark.parametrize("text,intent,domain,target,quality", [
    ("nexa prendé la luz del escritorio", "turn_on", "light", "domotics", "full"),
    ("apagá la luz", "turn_off", "light", "domotics", "full"),
    ("subí la persiana del cuarto", "open", "cover", "domotics", "full"),
    ("subí el volumen", "volume_set", "media_player", "music", "full"),
    ("pausá la música", "media_pause", "media_player", "music", "full"),
    ("poné la luz al 70%", "set", "light", "domotics", "full"),       # set por slot, sin on/off
    ("ponela cálida", "set", "light", "domotics", "full"),            # set por color_temp
    ("abrí la luz", None, "light", "domotics", "partial"),            # incompat → no full
    ("hola qué tal", None, None, "domotics", "none"),                 # no domótica
])
def test_parse_command(text, intent, domain, target, quality):
    pc = parse_command(text)
    assert pc.intent == intent
    assert pc.domain == domain
    assert pc.target == target
    assert pc.quality == quality


def test_parse_command_set_includes_slot():
    pc = parse_command("poné la luz al 70%")
    assert pc.intent == "set"
    assert pc.slots.get("brightness_pct") == 70


def test_parse_command_ready_to_dispatch():
    assert parse_command("prendé la luz").ready_to_dispatch() is True
    assert parse_command("abrí la luz").ready_to_dispatch() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "parse_command" -v`
Expected: FAIL con `ImportError: cannot import name 'ParsedCommand'`.

- [ ] **Step 3: Implementar `ParsedCommand` y `parse_command`**

Reemplazar el bloque `PartialCommand` + `_compute_confidence` + `parse_partial_command` existente por:

```python
@dataclass
class ParsedCommand:
    intent: str | None = None
    domain: str | None = None
    room: str | None = None
    slots: dict = field(default_factory=dict)
    target: str = "domotics"
    confidence: float = 0.0
    quality: str = "none"        # "full" | "partial" | "none"
    raw_text: str = ""
    has_wake: bool = False

    def ready_to_dispatch(self) -> bool:
        return self.quality == "full"

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        return self.confidence >= threshold


def _compute_confidence(pc: ParsedCommand) -> float:
    if pc.quality != "full":
        return 0.0
    score = 0.7
    if pc.has_wake:
        score += 0.15
    if pc.room is not None:
        score += 0.10
    if pc.slots:
        score += 0.05
    return min(score, 1.0)


def parse_command(text: str) -> ParsedCommand:
    """Parser autoritativo para domótica simple. Determinístico, idempotente,
    sin I/O. Produce intent+domain+target+quality."""
    if not text:
        return ParsedCommand()
    pc = ParsedCommand(raw_text=text)
    pc.has_wake = has_wake_word(text)
    pc.domain = extract_entity(text)
    pc.room = extract_room(text)
    pc.slots = extract_slots(text)

    rule = match_intent_rules(text, pc.domain)
    if rule is not None:
        pc.intent = rule.intent
        pc.target = rule.target
        # requires_slot: volume_set necesita slot de volumen
        if rule.requires_slot == "volume" and "volume_pct" not in pc.slots:
            pc.intent = None
    elif pc.domain == "light" and pc.slots and not _has_any_onoff_verb(text):
        # 'set' implícito: slots de luz (brillo/color/temp) sin verbo on/off.
        pc.intent = "set"
        pc.target = "domotics"

    # quality
    if pc.intent is not None and pc.domain is not None:
        pc.quality = "full"
    elif pc.intent is not None or pc.domain is not None:
        pc.quality = "partial"
    else:
        pc.quality = "none"

    pc.confidence = _compute_confidence(pc)
    return pc


def _has_any_onoff_verb(text: str) -> bool:
    t = _norm(text)
    for rule in INTENT_RULES:
        if rule.intent in ("turn_on", "turn_off") and _rule_verb_matches(rule, t):
            return True
    return False
```

Nota: `set` mapea a `light.turn_on` + service_data aguas abajo; eso lo maneja el dispatcher/vector con `merge_service_data` (sin cambios en esta tarea).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "parse_command" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nlu/command_grammar.py tests/unit/nlu/test_command_grammar.py
git commit -m "feat(nlu): ParsedCommand + parse_command (motor con quality/target/set implícito)"
```

---

## Task 4: Wrapper de compat `parse_partial_command` (regresión streaming)

**Files:**
- Modify: `src/nlu/command_grammar.py`
- Test: `tests/unit/nlu/test_command_grammar.py`

El early_dispatch (`multi_room_audio_loop.py`) llama `parse_partial_command(text)` y usa `.ready_to_dispatch()`, `.intent`, `.entity`, `.confidence`, `.slots`. Mantener esa interfaz.

- [ ] **Step 1: Write the failing test**

```python
from src.nlu.command_grammar import parse_partial_command


def test_partial_command_compat_shape():
    pc = parse_partial_command("nexa prendé la luz del escritorio")
    assert pc.intent == "turn_on"
    assert pc.entity == "light"          # alias de domain para compat
    assert pc.room == "escritorio"
    assert pc.ready_to_dispatch() is True
    assert pc.confidence >= 0.75


def test_partial_command_not_ready():
    pc = parse_partial_command("nexa")
    assert pc.ready_to_dispatch() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "partial_command_compat or partial_command_not_ready" -v`
Expected: FAIL (`parse_partial_command` no existe tras la reescritura, o no expone `.entity`).

- [ ] **Step 3: Implementar el wrapper con alias `entity`**

Agregar al final de `command_grammar.py`:

```python
class _PartialCompat(ParsedCommand):
    """Shim de compatibilidad: expone .entity como alias de .domain para los
    consumidores del early_dispatch que aún usan la API vieja."""
    @property
    def entity(self) -> str | None:
        return self.domain


def parse_partial_command(text: str):
    """Compat wrapper sobre parse_command para el early_dispatch streaming.
    Devuelve un objeto con .entity (alias de .domain)."""
    pc = parse_command(text)
    shim = _PartialCompat(
        intent=pc.intent, domain=pc.domain, room=pc.room, slots=pc.slots,
        target=pc.target, confidence=pc.confidence, quality=pc.quality,
        raw_text=pc.raw_text, has_wake=pc.has_wake,
    )
    return shim
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -v`
Expected: PASS (incluye los tests viejos del archivo; si alguno usaba `PartialCommand` directamente, ajustarlo a `parse_partial_command`/`parse_command`).

- [ ] **Step 5: Verificar el caller del early_dispatch**

Run: `ssh kza 'cd ~/app && grep -n "parse_partial_command\|\.entity\b\|ready_to_dispatch" src/audio/multi_room_audio_loop.py'`
Confirmar que solo usa `.intent`, `.entity`, `.room`, `.slots`, `.ready_to_dispatch()`. Si usa algo más, agregarlo al shim.

- [ ] **Step 6: Commit**

```bash
git add src/nlu/command_grammar.py tests/unit/nlu/test_command_grammar.py
git commit -m "refactor(nlu): parse_partial_command como wrapper de compat (alias entity)"
```

---

## Task 5: LLMRouter — timeout/error → `unavailable` (no `noise`)

**Files:**
- Modify: `src/nlu/llm_router.py`
- Test: `tests/unit/nlu/test_llm_router.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import pytest
from src.nlu.llm_router import LLMCommandRouter, CommandClassification


class _TimeoutRouter:
    def generate(self, *a, **k):
        import time; time.sleep(5)
        return ["{}"]


@pytest.mark.asyncio
async def test_timeout_marks_unavailable_not_noise():
    r = LLMCommandRouter(router=_TimeoutRouter(), timeout_s=0.1)
    out = await r.classify("prendé la luz")
    assert out.rejection_reason == "unavailable"
    assert out.is_command is None


class _ErrorRouter:
    def generate(self, *a, **k):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_error_marks_unavailable_not_noise():
    r = LLMCommandRouter(router=_ErrorRouter(), timeout_s=2.0)
    out = await r.classify("prendé la luz")
    assert out.rejection_reason == "unavailable"
    assert out.is_command is None
```

(Ajustar la construcción de `LLMCommandRouter` a su `__init__` real — ver `src/nlu/llm_router.py` línea ~237. Si requiere args extra, pasarlos con defaults mínimos.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_llm_router.py -k "unavailable" -v`
Expected: FAIL — hoy devuelve `rejection_reason="noise"`, `is_command=False`.

- [ ] **Step 3: Cambiar el manejo de timeout/error y el enum**

En `src/nlu/llm_router.py`:

1. Permitir `is_command: Optional[bool]` en `CommandClassification` (ya es `bool`; cambiar a `Optional[bool] = None` si hace falta). Actualizar el comentario del enum de `rejection_reason` para incluir `unavailable`.

2. En el `except asyncio.TimeoutError`:
```python
            return CommandClassification(
                is_command=None,
                rejection_reason="unavailable",
                raw_response="<timeout>",
                elapsed_ms=elapsed,
            )
```

3. En el `except Exception as e`:
```python
            return CommandClassification(
                is_command=None,
                rejection_reason="unavailable",
                raw_response=f"<error: {type(e).__name__}>",
                elapsed_ms=elapsed,
            )
```

4. Si hay un `_GUIDED_JSON_SCHEMA`/enum de `rejection_reason` con valores permitidos, agregar `"unavailable"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_llm_router.py -k "unavailable" -v`
Expected: PASS.

- [ ] **Step 5: Subir el timeout (config) como margen**

En `config/settings.yaml`, `nlu.llm_router.timeout_s: 2.5` → `3.5` (con comentario: "subido 2026-05-30; la domótica ya no toca el LLM, pero damos margen a no-domótica").

- [ ] **Step 6: Commit**

```bash
git add src/nlu/llm_router.py tests/unit/nlu/test_llm_router.py config/settings.yaml
git commit -m "fix(nlu): LLMRouter timeout/error → unavailable (no descartar como noise)"
```

---

## Task 6: request_router — routing por target/quality + fallback `unavailable`

**Files:**
- Modify: `src/pipeline/request_router.py`
- Test: `tests/unit/pipeline/test_request_router_grammar.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/pipeline/test_request_router_grammar.py
import pytest
from src.pipeline.request_router import _grammar_fastpath_classification


@pytest.mark.parametrize("text,intent,is_cmd", [
    ("nexa prendé la luz del escritorio", "turn_on", True),
    ("subí la persiana del cuarto", "open", True),
    ("subí el volumen", "volume_set", True),     # media también es comando válido
    ("abrí la luz", None, None),                 # incompat → None (cae a fallback)
    ("hola qué tal", None, None),
])
def test_grammar_fastpath_classification(text, intent, is_cmd):
    cls = _grammar_fastpath_classification(text, 0.75)
    if is_cmd is None:
        assert cls is None
    else:
        assert cls.is_command is True
        assert cls.intent == intent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_request_router_grammar.py -v`
Expected: FAIL — hoy `_grammar_fastpath_classification` usa `parse_partial_command`/`ready_to_dispatch` (intent+entity) y NO reconoce open/volume_set como dispatchables vía la tabla nueva (o no devuelve para media).

- [ ] **Step 3: Reescribir `_grammar_fastpath_classification`**

Reemplazar la función en `request_router.py` (líneas ~78-118) por:

```python
def _grammar_fastpath_classification(text: str, confidence_threshold: float = 0.75):
    """CommandClassification si la gramática determinística produce un comando
    dispatchable de alta confianza (domótica o media); None si no.

    Para target='music' igual se devuelve is_command=True: el comando es válido
    y debe pasar el gate; el dispatcher lo rutea a su music path por
    _classify_request. El intent/target quedan en la clasificación para audit.
    """
    from src.nlu.command_grammar import parse_command
    from src.nlu.llm_router import CommandClassification

    pc = parse_command(text)
    if pc.quality != "full" or pc.confidence < confidence_threshold:
        return None

    return CommandClassification(
        is_command=True,
        confidence=pc.confidence,
        intent=pc.intent,
        entity_hint=pc.domain,
        slots=dict(pc.slots),
        raw_response=f"<grammar:{pc.target}>",
        elapsed_ms=0.0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/test_request_router_grammar.py -v`
Expected: PASS.

- [ ] **Step 5: Manejo de `unavailable` en el bloque del LLMRouter**

En `_process_command_orchestrated`, donde se evalúa `classification` del LLMRouter (después del `[LLMRouter ...ms]` log, ~línea 461), agregar — antes del early-return que descarta no-comandos — el manejo de `unavailable`:

```python
            # Fallback: el LLM no pudo decidir (timeout/error). NO descartar si
            # la gramática tenía señal parcial — pedir confirmación.
            if classification.rejection_reason == "unavailable":
                from src.nlu.command_grammar import parse_command
                pc = parse_command(text)
                if pc.intent is not None and pc.domain is not None:
                    # señal completa → dispatch best-effort
                    classification.is_command = True
                    classification.intent = pc.intent
                    classification.entity_hint = pc.domain
                    classification.slots = dict(pc.slots)
                elif pc.intent is not None or pc.domain is not None:
                    # señal parcial → pedir confirmación por voz
                    result["needs_confirmation"] = True
                    result["confirmation_question"] = self._build_confirmation_question(pc)
                    result["response"] = result["confirmation_question"]
                    return result
                # sin señal → cae al manejo normal de no-comando abajo
```

(Verificar la forma exacta de `_build_confirmation_question` — hoy toma un `PartialCommand`; `parse_command` devuelve un `ParsedCommand` con los mismos campos `intent`/`room`. Ajustar la firma si hace falta para aceptar `ParsedCommand`.)

- [ ] **Step 6: Run regression**

Run: `.venv/bin/python -m pytest tests/unit/pipeline/ -v`
Expected: PASS (sin regresiones en el router).

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/request_router.py tests/unit/pipeline/test_request_router_grammar.py
git commit -m "feat(pipeline): routing por target/quality del motor + fallback unavailable→confirmación"
```

---

## Task 7: Media llega al music path (integración)

**Files:**
- Modify: `src/orchestrator/dispatcher.py` (solo si los verbos de media no están en `MUSIC_CONTEXT_KEYWORDS`/`_classify_request`)
- Test: `tests/unit/orchestrator/test_dispatcher_music_routing.py` (create)

El dispatcher decide música vía `_classify_request(text_lower)` con `MUSIC_CONTEXT_KEYWORDS`. Hay que asegurar que un comando de media que pasó el gate (is_command=True por la gramática) sea ruteado a `FAST_MUSIC`/`SLOW_MUSIC` y NO a `FAST_DOMOTICS`.

- [ ] **Step 1: Inspeccionar las keywords actuales**

Run: `ssh kza 'cd ~/app && grep -n "MUSIC_CONTEXT_KEYWORDS\|MUSIC_KEYWORD\|def _classify_request" src/orchestrator/dispatcher.py && sed -n "/def _classify_request/,/FAST_DOMOTICS/p" src/orchestrator/dispatcher.py'`
Anotar qué keywords disparan música.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/orchestrator/test_dispatcher_music_routing.py
import pytest
from src.orchestrator.dispatcher import RequestDispatcher, PathType


def _make_dispatcher_with_music():
    # Construir un dispatcher con music_dispatcher mock — ajustar a la firma real.
    class _Music: pass
    d = RequestDispatcher.__new__(RequestDispatcher)
    d.music = _Music()
    d.MUSIC_CONTEXT_KEYWORDS = RequestDispatcher.MUSIC_CONTEXT_KEYWORDS
    return d


@pytest.mark.parametrize("text", [
    "pausá la música",
    "subí el volumen",
    "poné música",
    "siguiente canción",
])
def test_media_routes_to_music(text):
    d = _make_dispatcher_with_music()
    path, _ = d._classify_request(text.lower())
    assert path in (PathType.FAST_MUSIC, PathType.SLOW_MUSIC)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_music_routing.py -v`
Expected: probablemente FAIL para "subí el volumen" / "pausá" si esas palabras no están en las keywords de música.

- [ ] **Step 4: Extender `MUSIC_CONTEXT_KEYWORDS` / `_classify_request`**

Agregar los lexemes de control de media que falten (ej: `"volumen"`, `"pausá"`, `"pausa"`, `"siguiente"`, `"canción"`, `"música"`) a la lista que usa `_classify_request`. Mantener el patrón existente del archivo (no reestructurar).

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_music_routing.py -v`
Expected: PASS.

- [ ] **Step 6: Run regression**

Run: `.venv/bin/python -m pytest tests/unit/orchestrator/ -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator/dispatcher.py tests/unit/orchestrator/test_dispatcher_music_routing.py
git commit -m "feat(orchestrator): rutear controles de media (volumen/pausa/siguiente) al music path"
```

---

## Task 8: Limpiar el `fast_path` muerto (regex+gate)

**Files:**
- Modify: `src/pipeline/request_router.py` (eliminar bloque `regex_extractor`/`llm_gate`)
- Modify: `config/settings.yaml` (quitar `nlu.fast_path`)
- Modify: `src/main.py` (si pasa `regex_extractor=`/`llm_gate=` al construir RequestRouter)

- [ ] **Step 1: Localizar el wiring muerto**

Run: `ssh kza 'cd ~/app && grep -rn "regex_extractor\|llm_gate\|fast_path" src/main.py src/pipeline/request_router.py config/settings.yaml'`

- [ ] **Step 2: Quitar el bloque del fast path en request_router**

Eliminar el bloque `if self.regex_extractor is not None and self.llm_gate is not None:` (líneas ~441-487) y los params `regex_extractor=None`, `llm_gate=None` del `__init__` + sus `self.regex_extractor`/`self.llm_gate`. Quitar imports muertos asociados.

- [ ] **Step 3: Quitar `nlu.fast_path` de settings.yaml**

Eliminar el bloque `fast_path:` bajo `nlu:` (enabled/gate_timeout_s/gate_max_tokens).

- [ ] **Step 4: Quitar el paso de args en main.py (si existe)**

Si `main.py` construye `RequestRouter(..., regex_extractor=..., llm_gate=...)`, quitar esos kwargs.

- [ ] **Step 5: Run full nlu+pipeline regression**

Run: `.venv/bin/python -m pytest tests/unit/nlu/ tests/unit/pipeline/ -v`
Expected: PASS. Nota: `tests/unit/nlu/test_regex_extractor.py` y `test_llm_gate.py` quedan testeando módulos que ya no se cablean — dejarlos (los módulos siguen existiendo) o marcarlos skip si fallan por el desmantelado. Decidir en ejecución; no borrar módulos en esta tarea.

- [ ] **Step 6: Verificar arranque (import + py_compile)**

Run: `ssh kza 'cd ~/app && .venv/bin/python -m py_compile src/main.py src/pipeline/request_router.py && echo OK'`
Expected: OK.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/request_router.py config/settings.yaml src/main.py
git commit -m "chore(pipeline): eliminar fast_path muerto (regex+llm_gate, nunca cableado)"
```

---

## Task 9: Poblar el léxico desde logs reales

**Files:**
- Modify: `src/nlu/command_grammar.py` (verb_patterns/ENTITY_TERMS), `src/nlu/slot_extractor.py`
- Test: `tests/unit/nlu/test_command_grammar.py` (casos reales)

- [ ] **Step 1: Recolectar utterances reales del journal**

Run:
```bash
ssh kza 'journalctl --user -u kza-voice.service --since "7 days ago" --no-pager | \
  grep -oE "detectado en: .*|Text=.*" | sort -u | head -200'
```
Anotar comandos reales (incluidos garbled: "apagalelu el victorio", "prendela luces, aire acondicionado", etc.).

- [ ] **Step 2: Write failing tests con los casos reales**

Agregar a `test_command_grammar.py` un bloque parametrizado con las frases reales recolectadas y su `ParsedCommand` esperado. Ejemplo (completar con lo recolectado):

```python
@pytest.mark.parametrize("text,intent,domain", [
    ("nexa apagalelu el victorio", "turn_off", "light"),     # 'apagá la luz...'
    ("nexa prendela luces aire acondicionado", "turn_on", "light"),
    ("nexa prende la luz del escritorio", "turn_on", "light"),
])
def test_real_world_utterances(text, intent, domain):
    pc = parse_command(text)
    assert pc.intent == intent
    assert pc.domain == domain
```

- [ ] **Step 3: Run, ver cuáles fallan**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -k "real_world" -v`
Expected: algunos FAIL (verbos/términos no cubiertos).

- [ ] **Step 4: Extender los patrones mínimos para cubrirlos**

Ajustar `verb_patterns` / `ENTITY_TERMS` lo justo para los casos reales. No sobre-generalizar.

- [ ] **Step 5: Run hasta PASS**

Run: `.venv/bin/python -m pytest tests/unit/nlu/test_command_grammar.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/nlu/command_grammar.py src/nlu/slot_extractor.py tests/unit/nlu/test_command_grammar.py
git commit -m "feat(nlu): poblar léxico de verbos/entidades desde utterances reales"
```

---

## Task 10: Validación end-to-end en el server

**Files:** ninguno (verificación).

- [ ] **Step 1: Suite completa**

Run: `ssh kza 'cd ~/app && .venv/bin/python -m pytest tests/unit/nlu/ tests/unit/pipeline/ tests/unit/orchestrator/ -q'`
Expected: PASS (anotar pre-existentes rotos ajenos — ver pendiente #5 del doc de sesión).

- [ ] **Step 2: Reiniciar el servicio y verificar arranque**

Run:
```bash
ssh kza 'systemctl --user restart kza-voice.service && sleep 12 && systemctl --user is-active kza-voice.service'
```
Expected: `active`. Revisar que no haya tracebacks de import en `journalctl --user -u kza-voice -n 50`.

- [ ] **Step 3: Comando real por voz (manual)**

Decir, con el mic del escritorio:
- "Nexa, prendé la luz del escritorio" → esperar `[GRAMMAR_FASTPATH] intent=turn_on` + `[HA-CALL] ... success=True`.
- "Nexa, poné la luz al 30%" → `set` → HA con brightness.
- "Nexa, subí la persiana" → `open` (si hay cover; si no, confirmar que no rompe).
- "Nexa, pausá la música" → ruteo a music path.

Run (observar): `ssh kza 'journalctl --user -u kza-voice.service -f | grep -E "GRAMMAR_FASTPATH|HA-CALL|music|LLMRouter|unavailable"'`

- [ ] **Step 4: Push a origin (desde la laptop)**

```bash
cd /Users/yo/Documents/kza
git fetch kza:/home/kza/app main:refs/remotes/kzaserver/main
git push origin refs/remotes/kzaserver/main:refs/heads/main
git update-ref -d refs/remotes/kzaserver/main
```
Expected: fast-forward limpio.

---

## Notas de ejecución

- **TDD estricto**: cada test rojo antes de implementar. No saltear el "run para ver fallar".
- **Determinismo**: el motor no debe hacer I/O ni llamar modelos. Si un test necesita red/HA, está mal ubicado.
- **`set` → turn_on**: la traducción a `light.turn_on`+service_data ocurre aguas abajo (dispatcher/vector + `merge_service_data`); el motor solo marca `intent="set"` con slots. Verificar en Task 10 step 3 que el brillo efectivamente se aplica.
- **Pre-existentes rotos** (`test_llm_router.py` 7 casos, `test_endpointing.py`): ajenos a este trabajo (pendiente #5 doc sesión); no confundir con regresiones.
