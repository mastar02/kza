# Ruteo clima/AC por inferencia — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el guard de substrings que desambigua "clima" (tiempo vs aire acondicionado) por un clasificador binario contra el 7B local, disparado solo cuando el texto contiene vocabulario contestado.

**Architecture:** `_classify_request()` queda intacta (sync, pura, determinista) y produce la opinion por defecto. Cuando devuelve `FAST_DOMOTICS` o `FAST_WEATHER` **y** el texto contiene un sustantivo de `_CLIMATE_DOMAIN_NOUNS`, `dispatch()` corre un clasificador binario contra `:8101` en paralelo con el vector search ya existente. El clasificador solo puede mover la decision entre esos dos paths; ante cualquier fallo, timeout o salida no reconocida, vale la regla.

**Tech Stack:** Python 3.13, asyncio, pytest, llama-server (Qwen2.5-7B-Instruct Q4_K_M) en `:8101` via OpenAI-compat.

**Spec:** `docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md`

## Global Constraints

- Python 3.13. Correr SIEMPRE con `.venv/bin/python3` — el `python3` del sistema es 3.9 y rompe en `dataclass(slots=True)`.
- Comentarios y logs en **ingles**. Docs y strings al usuario en **español**. Sin emojis en ningun archivo.
- Imports absolutos (`from src.modulo import Clase`), nunca relativos.
- DI por constructor. Composicion, nunca herencia profunda.
- `logging.getLogger(__name__)` — nunca `print()`.
- Toda operacion de I/O es `async/await`. Prohibido bloquear el event loop.
- Config nueva va en `config/settings.yaml`. No crear archivos de configuracion nuevos.
- **No tocar `src/analytics/`** — fuera de alcance.
- Baseline de tests: **2832 passed, 1 xfailed**. Comando:
  `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/ tests/integration/ tests/safety/ -q`
- El server es **produccion real** (lo usa el hogar a diario). Diagnostico read-only sin pedir permiso; cualquier restart/deploy se consulta primero.

## File Structure

| Archivo | Responsabilidad | Task |
|---|---|---|
| `benchmarks/router/climate_set.yaml` | Set B held-out. Congelado antes de escribir el clasificador | 1 |
| `src/llm/reasoner.py:837-875` | `FastRouter.complete()` gana soporte de `stop` | 2 |
| `src/nlu/climate_intent.py` | Prompt, llamada y parseo estricto. Un solo proposito | 3 |
| `tests/unit/nlu/test_climate_intent.py` | Unit del clasificador con router mockeado | 3 |
| `benchmarks/router/climate_eval.py` | Runner del eval contra `:8101` vivo. Fuera de pytest | 4 |
| `src/orchestrator/dispatcher.py` | Borrado del guard + rama `contested` en `dispatch()` | 5 |
| `tests/unit/orchestrator/test_dispatcher_world_routing.py` | 1 caso se muda de capa | 5 |

**Orden no negociable:** Task 1 congela el set de evaluacion **antes** de que exista una linea del clasificador. Task 4 es un **gate**: si el set B no llega al umbral, Task 5 no se ejecuta y la conclusion es no adoptar.

---

### Task 1: Congelar el set B (held-out)

El numero que se reporta sale de este archivo. Si se escribe despues de ver el
prompt, mide tuning en vez de generalizacion.

**IMPORTANTE para quien ejecute esta task:** no leas
`docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md` (contiene el
prompt) ni `src/nlu/climate_intent.py` (todavia no existe). Trabajá solo con lo
que esta escrito en esta task.

**Por que los casos sinteticos de los Steps 2 y 3 no vienen escritos acá:** no es
una omision del plan. Quien escribio este plan ya vio el prompt del clasificador,
asi que cualquier frase que redactara arrastraria ese conocimiento y el set
dejaria de ser held-out. Los Steps 2 y 3 dan la especificacion completa (cuantos
casos, que formas cubrir, que formato) para que los escriba alguien que no vio el
prompt. Esa separacion **es** el mecanismo, no un detalle de proceso.

**Files:**
- Create: `benchmarks/router/climate_set.yaml`

**Interfaces:**
- Consumes: nada.
- Produces: `benchmarks/router/climate_set.yaml` con la clave raiz `cases`, lista de dicts con las claves exactas `id: str`, `utterance: str`, `expected: "ACCION" | "CONSULTA"`, `source: "corpus" | "sintetico"`, `note: str`. Task 4 lo lee con `yaml.safe_load(...)["cases"]`.

**El criterio de etiquetado, y nada mas que esto:**

- `ACCION` = el usuario ordena encender, apagar o ajustar el aire acondicionado,
  termostato o calefaccion. Es una orden sobre un aparato.
- `CONSULTA` = el usuario pregunta o comenta como esta el tiempo afuera
  (pronostico, temperatura exterior, si llueve). No pide que pase nada.

Regla de desempate: preguntate que espera el usuario que ocurra despues de
hablar. Si espera que algo se prenda o cambie, es `ACCION`. Si espera escuchar
una respuesta, es `CONSULTA`.

- [ ] **Step 1: Crear el archivo con las 20 frases reales del corpus**

Son transcripciones literales de Whisper de `events.db` (production, 2026). El
ruido del STT es parte del caso de test: **no las limpies ni las corrijas**.
Todas son ordenes sobre el aire acondicionado, por eso todas van `ACCION`.

```yaml
# Set B — held-out para el clasificador binario clima/AC.
#
# Escrito ANTES de que exista el clasificador y sin ver su prompt. Este es el
# unico archivo del que sale el numero que se reporta. Si lo editas despues de
# haber corrido el eval, deja de ser held-out y el numero pierde validez.
#
# Etiquetas:
#   ACCION   = ordena encender/apagar/ajustar el aire, termostato o calefaccion
#   CONSULTA = pregunta o comenta como esta el tiempo afuera
#
# Las frases con source=corpus son transcripciones LITERALES de Whisper sacadas
# de events.db en produccion. El ruido es parte del caso: no limpiarlas.

cases:
  - id: corpus-001
    utterance: "Nexa bajá la temperatura del aire."
    expected: ACCION
    source: corpus
    note: "la frase mas frecuente del corpus (113 ocurrencias)"
  - id: corpus-002
    utterance: "Nexa bajá la temperatura."
    expected: ACCION
    source: corpus
    note: "sin objeto explicito"
  - id: corpus-003
    utterance: "Nexa bajá la temperatura del escritorio."
    expected: ACCION
    source: corpus
    note: "objeto = habitacion, no aparato"
  - id: corpus-004
    utterance: "Nexa bajá la temperatura del aire acondicionado,"
    expected: ACCION
    source: corpus
    note: "aparato nombrado completo, corte de STT al final"
  - id: corpus-005
    utterance: "Nexa apagá la temperatura del aire."
    expected: ACCION
    source: corpus
    note: "verbo semanticamente raro sobre temperatura"
  - id: corpus-006
    utterance: "Nexa apagá la luz del aire."
    expected: ACCION
    source: corpus
    note: "STT mezcla dominios luz/aire"
  - id: corpus-007
    utterance: "Nexa bajá la luz del aire."
    expected: ACCION
    source: corpus
    note: "STT mezcla dominios luz/aire"
  - id: corpus-008
    utterance: "Nexa poné la luz del aire."
    expected: ACCION
    source: corpus
    note: "STT mezcla dominios luz/aire"
  - id: corpus-009
    utterance: "Nexa apagá la luz del aire acondicionado."
    expected: ACCION
    source: corpus
    note: "STT mezcla dominios, aparato completo"
  - id: corpus-010
    utterance: "Nexa, prendela luces, aire acondicionado."
    expected: ACCION
    source: corpus
    note: "palabras pegadas por el STT"
  - id: corpus-011
    utterance: "Nexa bajá la temperatura de la luz del escritorio,"
    expected: ACCION
    source: corpus
    note: "tres dominios mezclados por el STT"
  - id: corpus-012
    utterance: "Nexa prendé la luz del escritorio, Nexa bajá la temperatura del escritorio,"
    expected: ACCION
    source: corpus
    note: "dos comandos concatenados en una captura"
  - id: corpus-013
    utterance: "Nexa bajá la temperatura del aire, Nexa bajá la temperatura del aire,"
    expected: ACCION
    source: corpus
    note: "repeticion del wake dentro de la captura"
  - id: corpus-014
    utterance: "Nexa bajá la temperatura, Nexa bajá la temperatura, Nexa bajá la temperatura,"
    expected: ACCION
    source: corpus
    note: "triple repeticion"
  - id: corpus-015
    utterance: "Comandos típicos, Nexa bajá la temperatura del aire, Nexa bajá la temperatura del aire,"
    expected: ACCION
    source: corpus
    note: "preambulo de TV o charla + comando real"
  - id: corpus-016
    utterance: "Nexa de luz del espíritu, Nexa bajá la temperatura del aire."
    expected: ACCION
    source: corpus
    note: "alucinacion del STT antes del comando real"
  - id: corpus-017
    utterance: "Nexa bajá la luz del aire, desde la base del cláneo."
    expected: ACCION
    source: corpus
    note: "cola alucinada por el STT"
  - id: corpus-018
    utterance: "Nexa bajá la luz del aire.  Vámonos."
    expected: ACCION
    source: corpus
    note: "habla ambiente pegada al comando"
  - id: corpus-019
    utterance: "Nexa bajá la luz del aire al amor de la comida.  O sea, ponte redonda."
    expected: ACCION
    source: corpus
    note: "comando + cola larga alucinada"
  - id: corpus-020
    utterance: "Nexa bajá la temperatura del aire,"
    expected: ACCION
    source: corpus
    note: "corte de STT al final"
```

- [ ] **Step 2: Agregar 20 casos sinteticos de CONSULTA**

El corpus no tiene ni una pregunta de clima (el path se agrego el 2026-08-04),
asi que esta mitad hay que escribirla. Escribilas como las diria alguien en
Buenos Aires, en voz alta, y asumi que Whisper se come los signos de
interrogacion la mitad de las veces — **al menos 8 de las 20 van sin `¿` ni `?`**.

Cubri estas formas, una por caso como minimo: pregunta directa por el tiempo;
pregunta por temperatura exterior; pregunta por lluvia; pregunta por pronostico
de mañana; comentario sin pregunta ("esta lindo el dia"); comentario con
negacion explicita de accion ("no hace falta prender nada"); pregunta que
menciona el aparato pero pide informacion; pregunta con el verbo de domotica en
otra clausula.

Formato identico al del Step 1 con `source: sintetico`. Ejemplo de la forma:

```yaml
  - id: sint-001
    utterance: "qué tiempo hace afuera"
    expected: CONSULTA
    source: sintetico
    note: "pregunta directa, sin signos (Whisper se los come)"
```

- [ ] **Step 3: Agregar 10 casos borde de ACCION, sin ver el prompt**

Comandos reales que mencionan el tiempo como justificacion. Esta es la clase que
rompio tres veces, asi que tiene que estar representada. Al menos 4 con relleno
entre el verbo y el objeto ("prendé YA el clima", "apagá AHORA el termostato") y
al menos 4 con clausula de justificacion climatica ("..., hace calor").

Formato identico, `source: sintetico`, `expected: ACCION`.

- [ ] **Step 4: Validar que el YAML carga y las etiquetas son legales**

Run:
```bash
cd ~/Documents/kza && .venv/bin/python3 -c "
import yaml, collections
cases = yaml.safe_load(open('benchmarks/router/climate_set.yaml'))['cases']
assert len(cases) >= 50, f'esperaba >=50 casos, hay {len(cases)}'
ids = [c['id'] for c in cases]
assert len(ids) == len(set(ids)), 'ids duplicados'
for c in cases:
    assert set(c) == {'id','utterance','expected','source','note'}, f\"claves mal en {c['id']}\"
    assert c['expected'] in ('ACCION','CONSULTA'), f\"etiqueta ilegal en {c['id']}\"
    assert c['utterance'].strip(), f\"utterance vacia en {c['id']}\"
print(collections.Counter(c['expected'] for c in cases))
print(collections.Counter(c['source'] for c in cases))
print(f'OK {len(cases)} casos')
"
```
Expected: imprime el conteo, `ACCION` y `CONSULTA` ambos >= 20, y `OK N casos` con N >= 50.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/kza
git add benchmarks/router/climate_set.yaml
git commit -m "test(bench): set B held-out para el clasificador clima/AC

Escrito antes del clasificador y sin ver su prompt. 20 frases reales del
corpus de produccion con su ruido de STT + sinteticos de consulta y de
comando con justificacion climatica."
```

---

### Task 2: `stop` en `FastRouter.complete()`

Hoy `complete()` es async pero no soporta `stop`, y su firma termina en
`**_ignored`: si se le pasa `stop=[...]` lo **descarta en silencio**. El
benchmark de mayo (`benchmarks/router/REPORT.md`) midio que los stop tokens
valen +18 puntos de accuracy en clasificacion.

**Files:**
- Modify: `src/llm/reasoner.py:837-875`
- Test: `tests/unit/llm/test_fast_router_stop.py` (create)

**Interfaces:**
- Consumes: nada.
- Produces: `FastRouter.complete(prompt: str, max_tokens: int = 256, temperature: float = 0.3, stop: list[str] | None = None, **_ignored) -> str`. Cuando `stop` es `None` o vacio, la llamada al cliente OpenAI **no** incluye la key `stop` (comportamiento identico al de hoy).

- [ ] **Step 1: Escribir el test que falla**

```python
"""FastRouter.complete() must forward stop sequences to the endpoint.

Regression guard: the signature ends in **_ignored, so a `stop` kwarg used to
be swallowed silently -- the caller believed it had stop tokens and did not.
"""

from unittest.mock import MagicMock

import pytest

from src.llm.reasoner import FastRouter


@pytest.fixture
def router():
    r = FastRouter(base_url="http://127.0.0.1:8101/v1", model="test-model")
    r._client = MagicMock()
    r._client.completions.create.return_value = MagicMock(
        choices=[MagicMock(text="ACCION_AIRE")], usage=None
    )
    return r


@pytest.mark.asyncio
async def test_complete_forwards_stop_sequences(router):
    await router.complete("prompt", max_tokens=10, stop=["\n", "Texto:"])

    kwargs = router._client.completions.create.call_args.kwargs
    assert kwargs["stop"] == ["\n", "Texto:"]


@pytest.mark.asyncio
async def test_complete_omits_stop_when_not_given(router):
    await router.complete("prompt", max_tokens=10)

    kwargs = router._client.completions.create.call_args.kwargs
    assert "stop" not in kwargs


@pytest.mark.asyncio
async def test_complete_omits_stop_when_empty(router):
    await router.complete("prompt", max_tokens=10, stop=[])

    kwargs = router._client.completions.create.call_args.kwargs
    assert "stop" not in kwargs
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/llm/test_fast_router_stop.py -v`
Expected: FAIL en `test_complete_forwards_stop_sequences` con `KeyError: 'stop'` (el kwarg se lo comio `**_ignored`). Los otros dos pasan ya.

- [ ] **Step 3: Implementar**

En `src/llm/reasoner.py`, reemplazar la firma y la llamada de `complete()` (linea 837 en adelante):

```python
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        stop: list[str] | None = None,
        **_ignored,
    ) -> str:
```

Y dentro de `_call()`, reemplazar la construccion de la request:

```python
        def _call():
            if self._client is None:
                self.load()
            t0 = time.perf_counter()
            # `stop` is opt-in: omitting the key keeps the request byte-identical
            # to the pre-2026-08-04 behaviour for every existing caller.
            extra = {"stop": stop} if stop else {}
            resp = self._client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
```

El resto del cuerpo (`elapsed_ms`, `usage`, `_last_metrics`, `return`) no se toca.

- [ ] **Step 4: Correr los tests**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/llm/ -q`
Expected: PASS, 3 tests nuevos verdes y ninguno de los existentes en rojo.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/kza
git add src/llm/reasoner.py tests/unit/llm/test_fast_router_stop.py
git commit -m "feat(llm): FastRouter.complete acepta stop sequences

Antes las tragaba **_ignored en silencio. El bench de mayo midio +18
puntos de accuracy en classify con stop tokens."
```

---

### Task 3: El clasificador

**Files:**
- Create: `src/nlu/climate_intent.py`
- Create: `tests/unit/nlu/test_climate_intent.py`

**Interfaces:**
- Consumes: `FastRouter.complete(..., stop=[...])` de Task 2.
- Produces:
  - `ClimateIntent` — `Enum` con miembros `ACTION` (valor `"ACCION"`) y `QUERY` (valor `"CONSULTA"`).
  - `CLIMATE_PROMPT: str` — plantilla con un unico placeholder `{text}`.
  - `PROMPT_FINGERPRINT: str` — sha256 hex de `CLIMATE_PROMPT.encode()`, primeros 16 chars.
  - `has_contested_vocabulary(text: str) -> bool`
  - `ClimateIntentClassifier(router, timeout_s: float = 0.15)` con `async def classify(self, text: str) -> ClimateIntent | None`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""Binary climate/AC intent classifier -- unit level, no network.

Everything here is deterministic: the router is mocked, so these are plain
assertEqual tests. The probabilistic part lives in benchmarks/router/, scored
against a threshold, never asserted for equality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nlu.climate_intent import (
    CLIMATE_PROMPT,
    PROMPT_FINGERPRINT,
    ClimateIntent,
    ClimateIntentClassifier,
    has_contested_vocabulary,
)


def _router(reply: str):
    r = MagicMock()
    r.complete = AsyncMock(return_value=reply)
    return r


# --- gate -----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "prendé el clima",
    "poné la temperatura en 22",
    "apagá el termostato",
    "prendé la calefacción",
    "apagá el aire",
    "subí los grados",
    "QUE TEMPERATURA HACE",          # case-insensitive
    "bajá la calefaccion",           # sin acento
])
def test_gate_fires_on_contested_vocabulary(text):
    assert has_contested_vocabulary(text) is True


@pytest.mark.parametrize("text", [
    "prendé la luz del living",
    "poné música de Spinetta",
    "agregá leche a la lista",
    "recordame sacar la basura",
    "por qué el cielo es azul",
    "",
])
def test_gate_stays_quiet_without_contested_vocabulary(text):
    assert has_contested_vocabulary(text) is False


def test_gate_does_not_match_substrings():
    # "aire" inside "airear", "grados" inside "posgrados": the gate must use
    # word boundaries or it becomes the very substring bug it replaces.
    assert has_contested_vocabulary("hay que airear la pieza") is False
    assert has_contested_vocabulary("terminé los posgrados") is False


# --- parsing --------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_action_for_action_label():
    c = ClimateIntentClassifier(_router("ACCION_AIRE"))
    assert await c.classify("prendé el clima") == ClimateIntent.ACTION


@pytest.mark.asyncio
async def test_returns_query_for_query_label():
    c = ClimateIntentClassifier(_router("PREGUNTA_TIEMPO"))
    assert await c.classify("está el clima lindo") == ClimateIntent.QUERY


@pytest.mark.asyncio
async def test_tolerates_whitespace_and_case():
    c = ClimateIntentClassifier(_router("  accion_aire\n"))
    assert await c.classify("prendé el clima") == ClimateIntent.ACTION


@pytest.mark.parametrize("reply", [
    "",                                  # empty
    "   ",                               # blank
    "OTRO",                              # off-label
    "clima",                             # the ambiguous word itself
    "clima\nRespuesta: Lo siento, no",   # contaminated output seen in probing
    "no estoy seguro",                   # hallucination
    "ACCION",                            # partial label, not the contract
])
@pytest.mark.asyncio
async def test_unrecognised_output_returns_none(reply):
    c = ClimateIntentClassifier(_router(reply))
    assert await c.classify("prendé el clima") is None


# --- failure modes --------------------------------------------------------

@pytest.mark.asyncio
async def test_router_exception_returns_none():
    r = MagicMock()
    r.complete = AsyncMock(side_effect=ConnectionError("8101 down"))
    assert await ClimateIntentClassifier(r).classify("prendé el clima") is None


@pytest.mark.asyncio
async def test_timeout_returns_none():
    async def _slow(*a, **k):
        await asyncio.sleep(1.0)
        return "ACCION_AIRE"

    r = MagicMock()
    r.complete = _slow
    c = ClimateIntentClassifier(r, timeout_s=0.05)
    assert await c.classify("prendé el clima") is None


@pytest.mark.asyncio
async def test_timeout_is_enforced_quickly():
    async def _slow(*a, **k):
        await asyncio.sleep(1.0)

    r = MagicMock()
    r.complete = _slow
    c = ClimateIntentClassifier(r, timeout_s=0.05)
    t0 = asyncio.get_running_loop().time()
    await c.classify("prendé el clima")
    assert asyncio.get_running_loop().time() - t0 < 0.5


@pytest.mark.asyncio
async def test_no_router_returns_none():
    assert await ClimateIntentClassifier(None).classify("prendé el clima") is None


# --- call contract --------------------------------------------------------

@pytest.mark.asyncio
async def test_calls_router_with_deterministic_parameters():
    r = _router("ACCION_AIRE")
    await ClimateIntentClassifier(r).classify("prendé el clima")

    kwargs = r.complete.call_args.kwargs
    assert kwargs["temperature"] == 0.0, "temp must be 0: determinism is the testing contract"
    assert kwargs["max_tokens"] == 10
    assert "\n" in kwargs["stop"]
    assert "prendé el clima" in r.complete.call_args.args[0]


# --- prompt tripwire ------------------------------------------------------

def test_prompt_fingerprint_is_pinned():
    """If this fails you edited CLIMATE_PROMPT.

    The prompt is a contract, not decoration: the few-shot examples carry
    measured accuracy. Re-run the eval before updating this constant:

        .venv/bin/python3 benchmarks/router/climate_eval.py

    Then paste the new fingerprint printed by the runner.
    """
    import hashlib
    actual = hashlib.sha256(CLIMATE_PROMPT.encode()).hexdigest()[:16]
    assert actual == PROMPT_FINGERPRINT, (
        f"prompt changed (fingerprint {actual}); re-run the eval and update "
        f"PROMPT_FINGERPRINT"
    )
```

- [ ] **Step 2: Correr para verificar que falla**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/nlu/test_climate_intent.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.nlu.climate_intent'`.

- [ ] **Step 3: Implementar el modulo**

```python
"""Binary intent classifier for the climate/AC ambiguity.

In Rioplatense Spanish "clima" means both the weather and the air conditioner:

    "prendé el clima"      -> turn the AC on   (command)
    "está el clima lindo"  -> what's it like   (question)

Three rounds of substring rules failed to separate these (see
docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md): every rule
was a proxy for "is this an order or an observation", and that proxy leaks in
spoken Spanish without reliable punctuation.

This module asks the local 7B instead, and only when the text actually contains
contested vocabulary. It can only ever answer ACTION or QUERY; anything else --
timeout, endpoint down, unparseable output -- returns None, which means "let the
rules decide". The caller keeps its rule-based answer as the default.
"""

import asyncio
import hashlib
import logging
import re
import unicodedata
from enum import Enum

logger = logging.getLogger(__name__)


class ClimateIntent(Enum):
    """What the speaker wants to happen."""

    ACTION = "ACCION"   # operate the AC / thermostat / heating
    QUERY = "CONSULTA"  # asking or remarking about the weather outside


# Vocabulary that is genuinely contested between the two readings. Mirrors
# _CLIMATE_DOMAIN_NOUNS in src/orchestrator/dispatcher.py -- keep in sync. No
# new vocabulary is introduced here on purpose: the gate is a presence test,
# not an intent guess.
_CONTESTED_NOUNS: tuple[str, ...] = (
    "clima", "temperatura", "termostato", "calefaccion", "aire", "grados",
)
_CONTESTED_RE = re.compile(
    r"\b(?:" + "|".join(_CONTESTED_NOUNS) + r")\b"
)

# Labels must NOT be named after the ambiguous word. Measured 2026-08-04: with
# "clima" as an option label the 7B answered "clima" for "prendé el clima" --
# it inherits the exact ambiguity we are trying to resolve. Unambiguous label
# names took the same model from failing both hard cases to 21/22.
_LABEL_ACTION = "ACCION_AIRE"
_LABEL_QUERY = "PREGUNTA_TIEMPO"

# The four few-shot examples are part of the contract, not decoration: they
# carry measured accuracy. Editing this string trips test_prompt_fingerprint_is_pinned,
# which is deliberate -- re-run benchmarks/router/climate_eval.py before changing it.
CLIMATE_PROMPT = f"""Sos el router de un asistente de hogar. Decidí qué hace el usuario.

{_LABEL_ACTION}     = ordena encender, apagar o ajustar el aire / termostato / calefacción.
{_LABEL_QUERY} = pregunta o comenta cómo está el tiempo afuera.

En rioplatense "clima" significa las dos cosas: el aparato y el tiempo.
Decidí por lo que el usuario QUIERE que pase, no por la palabra.

Texto: apagá el aire
Etiqueta: {_LABEL_ACTION}
Texto: ¿va a llover mañana?
Etiqueta: {_LABEL_QUERY}
Texto: está lindo el día, no prendas nada
Etiqueta: {_LABEL_QUERY}
Texto: poné el aire en 22 que hace calor
Etiqueta: {_LABEL_ACTION}
Texto: {{text}}
Etiqueta:"""

PROMPT_FINGERPRINT = "PLACEHOLDER_SET_IN_STEP_4"

_STOP = ["\n", "Texto:", "Etiqueta:"]


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def has_contested_vocabulary(text: str) -> bool:
    """True if the text mentions a noun that means both the device and the weather.

    Deliberately trivial: this asks whether contested WORDS are present, never
    what the speaker intends. That is what makes it safe -- when it is wrong it
    costs a model call (latency), it cannot misroute anything.

    Args:
        text: User text, post-STT. Case and accents are normalised here.

    Returns:
        True if any contested noun appears as a whole word.
    """
    if not text:
        return False
    return _CONTESTED_RE.search(_strip_accents(text.lower())) is not None


class ClimateIntentClassifier:
    """Resolves climate/AC ambiguity via the local 7B on :8101.

    Args:
        router: Object exposing `async complete(prompt, max_tokens, temperature,
            stop) -> str`. Typically FastRouter. None disables the classifier.
        timeout_s: Hard budget for the call. Measured p95 is 118ms; the default
            sits just above it so the tail is cut without touching the typical
            case. Moving this means re-checking the eval's p95 threshold.
    """

    def __init__(self, router, timeout_s: float = 0.15):
        self.router = router
        self.timeout_s = timeout_s

    async def classify(self, text: str) -> ClimateIntent | None:
        """Classify text as an AC command or a weather question.

        Returns:
            ClimateIntent.ACTION, ClimateIntent.QUERY, or None. None means the
            classifier abstained -- no router, timeout, transport error, or
            output that is not exactly one of the two labels. The caller must
            fall back to its rule-based decision on None.
        """
        if self.router is None:
            return None

        try:
            raw = await asyncio.wait_for(
                self.router.complete(
                    CLIMATE_PROMPT.format(text=text),
                    max_tokens=10,
                    temperature=0.0,
                    stop=_STOP,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "climate_intent: timeout after %.0fms, falling back to rules",
                self.timeout_s * 1000,
            )
            return None
        except Exception as exc:
            logger.warning("climate_intent: classifier unavailable (%s), falling back to rules", exc)
            return None

        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> ClimateIntent | None:
        """Strict label parsing. Anything unexpected abstains.

        No fuzzy matching on purpose: a substring match here would reintroduce
        the failure mode this module exists to remove.
        """
        if not raw:
            return None
        head = raw.strip().upper()
        if head.startswith(_LABEL_ACTION):
            return ClimateIntent.ACTION
        if head.startswith(_LABEL_QUERY):
            return ClimateIntent.QUERY
        logger.debug("climate_intent: unrecognised label %r, abstaining", raw[:40])
        return None
```

Crear tambien `tests/unit/nlu/__init__.py` vacio si el directorio no existe.

- [ ] **Step 4: Fijar el fingerprint del prompt**

Run:
```bash
cd ~/Documents/kza && .venv/bin/python3 -c "
import hashlib
from src.nlu.climate_intent import CLIMATE_PROMPT
print(hashlib.sha256(CLIMATE_PROMPT.encode()).hexdigest()[:16])
"
```
Copiar el hash impreso y reemplazar `PROMPT_FINGERPRINT = "PLACEHOLDER_SET_IN_STEP_4"` por ese valor.

- [ ] **Step 5: Correr los tests**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/nlu/test_climate_intent.py -v`
Expected: PASS, los ~28 tests verdes (8 gate positivos + 6 negativos + 1 substring + 3 parseo + 7 output no reconocido + 4 fallos + 1 contrato + 1 tripwire).

- [ ] **Step 6: Commit**

```bash
cd ~/Documents/kza
git add src/nlu/climate_intent.py tests/unit/nlu/
git commit -m "feat(nlu): clasificador binario de intencion clima/AC

Reemplaza el guard de substrings por una consulta al 7B local, disparada
solo ante vocabulario contestado. Abstiene (None) ante timeout, endpoint
caido o cualquier salida fuera de las dos etiquetas."
```

---

### Task 4: Eval contra el set B — GATE de go/no-go

**Esta task decide si el proyecto sigue.** Si el set B no llega al umbral, Task 5
no se ejecuta y se escribe la conclusion negativa con los numeros.

**Files:**
- Create: `benchmarks/router/climate_eval.py`

**Interfaces:**
- Consumes: `benchmarks/router/climate_set.yaml` (Task 1), `src.nlu.climate_intent` (Task 3).
- Produces: script ejecutable que imprime accuracy global, los dos tipos de error por separado, percentiles de latencia y el veredicto GO/NO-GO.

- [ ] **Step 1: Escribir el runner**

```python
#!/usr/bin/env python3
"""Scored eval of the climate/AC classifier against the live :8101.

This is NOT a pytest test. A probabilistic classifier is measured against a
threshold, not asserted for equality -- the assertEqual half lives in
tests/unit/nlu/test_climate_intent.py with a mocked router.

Run:
    .venv/bin/python3 benchmarks/router/climate_eval.py
    .venv/bin/python3 benchmarks/router/climate_eval.py --set benchmarks/router/climate_set.yaml

Go/no-go, all three must hold:
    - overall accuracy >= 90%          (rules baseline is 77.3%)
    - zero QUERY misread as ACTION     (unrequested physical action)
    - p95 <= 150ms                     (must stay under the classifier timeout)
"""

import argparse
import asyncio
import hashlib
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.reasoner import FastRouter  # noqa: E402
from src.nlu.climate_intent import (  # noqa: E402
    CLIMATE_PROMPT,
    ClimateIntent,
    ClimateIntentClassifier,
)

ACCURACY_FLOOR = 0.90
P95_CEILING_MS = 150.0

EXPECTED = {"ACCION": ClimateIntent.ACTION, "CONSULTA": ClimateIntent.QUERY}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="benchmarks/router/climate_set.yaml")
    ap.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--timeout-s", type=float, default=5.0,
                    help="generous here on purpose: we want to MEASURE the tail, "
                         "not have it swallowed by the production timeout")
    args = ap.parse_args()

    cases = yaml.safe_load(Path(args.set).read_text())["cases"]
    router = FastRouter(base_url=args.base_url, model=args.model)
    clf = ClimateIntentClassifier(router, timeout_s=args.timeout_s)

    # Warm the prefix cache so the first case does not skew the tail.
    await clf.classify("calentar")

    latencies: list[float] = []
    correct = 0
    query_as_action: list[str] = []   # expensive error: unrequested action
    action_as_query: list[str] = []   # cheap error: says forecast, does nothing
    abstained: list[str] = []

    for case in cases:
        want = EXPECTED[case["expected"]]
        t0 = time.perf_counter()
        got = await clf.classify(case["utterance"])
        latencies.append((time.perf_counter() - t0) * 1000)

        if got == want:
            correct += 1
        elif got is None:
            abstained.append(case["id"])
        elif want is ClimateIntent.QUERY:
            query_as_action.append(case["id"])
        else:
            action_as_query.append(case["id"])

        mark = "OK  " if got == want else "MISS"
        label = got.value if got else "ABSTAIN"
        print(f"{mark} {case['id']:12} want={case['expected']:8} got={label:8} "
              f"{latencies[-1]:6.1f}ms  {case['utterance'][:52]}")

    n = len(cases)
    accuracy = correct / n
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(int(n * 0.95), n - 1)]

    print("\n" + "=" * 72)
    print(f"casos                 {n}")
    print(f"accuracy              {correct}/{n} = {accuracy * 100:.1f}%   (piso {ACCURACY_FLOOR * 100:.0f}%)")
    print(f"consulta -> accion    {len(query_as_action)}  {query_as_action}   (error caro, piso 0)")
    print(f"accion -> consulta    {len(action_as_query)}  {action_as_query}   (error barato)")
    print(f"abstenciones          {len(abstained)}  {abstained}")
    print(f"latencia              p50 {p50:.0f}ms   p95 {p95:.0f}ms   (techo {P95_CEILING_MS:.0f}ms)")
    print(f"prompt fingerprint    {hashlib.sha256(CLIMATE_PROMPT.encode()).hexdigest()[:16]}")

    checks = {
        "accuracy": accuracy >= ACCURACY_FLOOR,
        "cero consulta->accion": not query_as_action,
        "p95": p95 <= P95_CEILING_MS,
    }
    for name, passed in checks.items():
        print(f"  [{'OK' if passed else 'NO'}] {name}")

    verdict = all(checks.values())
    print(f"\nVEREDICTO: {'GO' if verdict else 'NO-GO'}")
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 2: Correr el eval contra `:8101`**

El endpoint `:8101` corre en el server, no en la laptop. Desde la laptop hace
falta un tunel SSH — es read-only, no toca produccion:

```bash
ssh -f -N -L 8101:127.0.0.1:8101 kza
```

`:8101` **exige bearer auth** (verificado 2026-08-04: sin header devuelve
`HTTP 401`). El `llama-server` arranca con `--api-key-file
/home/kza/secrets/llama-api-key`, y el env file expone la misma key como
`LLAMA_API_KEY`. Traerla al entorno local sin escribirla en ningun archivo del
repo:

```bash
export LLAMA_API_KEY="$(ssh kza 'cat /home/kza/secrets/llama-api-key')"
cd ~/Documents/kza && .venv/bin/python3 benchmarks/router/climate_eval.py
```

`FastRouter` toma la key del env var que reciba en `api_key_env`. Si el runner da
401, pasarselo explicito construyendo el router con
`FastRouter(base_url=..., model=..., api_key_env="LLAMA_API_KEY")` en
`climate_eval.py`.

Expected: tabla por caso + el bloque resumen + `VEREDICTO: GO` o `NO-GO`.

- [ ] **Step 3: Correrlo 3 veces y confirmar que es determinista**

```bash
cd ~/Documents/kza && for i in 1 2 3; do
  .venv/bin/python3 benchmarks/router/climate_eval.py | grep -E "^accuracy|^consulta|^VEREDICTO"
done
```
Expected: las 3 corridas dan accuracy identica. `temperature=0.0` lo garantiza;
si varian, hay algo mal (temperatura, o el server reconfigurado) y hay que
resolverlo antes de seguir.

- [ ] **Step 4: El gate**

- **GO** (los 3 checks en OK) -> commitear el runner y seguir a Task 5.
- **NO-GO** -> **parar**. No implementar Task 5. Escribir los numeros en
  `docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md` bajo una
  seccion nueva "Resultado del eval: no adoptar", con la tabla por caso y que
  check fallo. Es un resultado valido del proyecto, no un fracaso.

- [ ] **Step 5: Commit**

```bash
cd ~/Documents/kza
git add benchmarks/router/climate_eval.py
git commit -m "test(bench): runner scoreado del clasificador clima/AC

Umbral, no igualdad: accuracy >=90%, cero consulta->accion, p95 <=150ms.
Fuera de pytest porque necesita :8101 vivo."
```

---

### Task 5: Integrar en el dispatcher

**Solo si Task 4 dio GO.**

**Files:**
- Modify: `src/orchestrator/dispatcher.py` (borrar guard: lineas ~422-446 y ~700-724; agregar rama en `dispatch()`)
- Modify: `tests/unit/orchestrator/test_dispatcher_world_routing.py:85-108`
- Test: `tests/unit/orchestrator/test_dispatcher_climate_classifier.py` (create)

**Interfaces:**
- Consumes: `ClimateIntentClassifier`, `ClimateIntent`, `has_contested_vocabulary` de Task 3.
- Produces: `RequestDispatcher.__init__(..., climate_classifier=None)`; atributo `self.climate_classifier`.

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""The classifier overrides the rules only on the climate/weather axis."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nlu.climate_intent import ClimateIntent
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import PathType, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue


def _dispatcher(intent):
    clf = MagicMock()
    clf.classify = AsyncMock(return_value=intent)
    d = RequestDispatcher(
        chroma_sync=MagicMock(), ha_client=MagicMock(), routine_manager=MagicMock(),
        router=None, llm=None, context_manager=ContextManager(),
        priority_queue=PriorityRequestQueue(), climate_classifier=clf,
    )
    d.music = MagicMock()
    return d, clf


@pytest.mark.asyncio
async def test_classifier_flips_weather_to_domotics():
    d, clf = _dispatcher(ClimateIntent.ACTION)
    path = await d._resolve_climate_path("prendé ya el clima, hace calor", PathType.FAST_WEATHER)
    assert path == PathType.FAST_DOMOTICS
    clf.classify.assert_awaited_once()


@pytest.mark.asyncio
async def test_classifier_flips_domotics_to_weather():
    d, _ = _dispatcher(ClimateIntent.QUERY)
    path = await d._resolve_climate_path("¿tengo que prender el clima o hace calor afuera?",
                                         PathType.FAST_DOMOTICS)
    assert path == PathType.FAST_WEATHER


@pytest.mark.asyncio
async def test_abstention_keeps_the_rule_decision():
    d, _ = _dispatcher(None)
    assert await d._resolve_climate_path("prendé el clima", PathType.FAST_WEATHER) == PathType.FAST_WEATHER
    assert await d._resolve_climate_path("prendé el clima", PathType.FAST_DOMOTICS) == PathType.FAST_DOMOTICS


@pytest.mark.asyncio
async def test_no_classifier_configured_keeps_the_rule_decision():
    d, _ = _dispatcher(ClimateIntent.QUERY)
    d.climate_classifier = None
    assert await d._resolve_climate_path("prendé el clima", PathType.FAST_DOMOTICS) == PathType.FAST_DOMOTICS


@pytest.mark.asyncio
async def test_gate_does_not_fire_without_contested_vocabulary():
    d, clf = _dispatcher(ClimateIntent.QUERY)
    path = await d._resolve_climate_path("prendé la luz del living", PathType.FAST_DOMOTICS)
    assert path == PathType.FAST_DOMOTICS
    clf.classify.assert_not_awaited()


@pytest.mark.parametrize("other_path", [
    PathType.FAST_MUSIC, PathType.FAST_LIST, PathType.FAST_REMINDER, PathType.SLOW_LLM,
])
@pytest.mark.asyncio
async def test_classifier_never_touches_other_paths(other_path):
    d, clf = _dispatcher(ClimateIntent.ACTION)
    assert await d._resolve_climate_path("subí la temperatura de la música", other_path) == other_path
    clf.classify.assert_not_awaited()
```

- [ ] **Step 2: Correr para verificar que falla**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/orchestrator/test_dispatcher_climate_classifier.py -v`
Expected: FAIL con `TypeError: __init__() got an unexpected keyword argument 'climate_classifier'`.

- [ ] **Step 3: Borrar el guard**

En `src/orchestrator/dispatcher.py`:

1. Borrar el bloque `_DOMOTICS_VERBS_STRIPPED` / `_DOMOTICS_CLIMATE_ADJACENCY_RE` y su comentario (aprox. lineas 422-446).
2. Borrar de `WEATHER_KEYWORDS` las tres entradas de la ronda 2 y su comentario: `"hace calor"`, `"hace frío"`, `"hace frio"`.
3. En `_classify_request`, reemplazar el bloque del guard (aprox. lineas 705-724) por el loop simple:

```python
        # Weather -> local source in HA. Runs BEFORE domotics because
        # "qué temperatura hace" shares its noun with the thermostat; the
        # complement ("hace"/"afuera") is what disambiguates. Runs AFTER
        # music/lists/reminders, which are more specific.
        #
        # The verb-adjacency guard that used to live here was removed on
        # 2026-08-04: three rounds of it each closed one class of error and
        # opened the neighbouring one. Genuine ambiguity is now resolved by
        # ClimateIntentClassifier in dispatch(), which sees the whole
        # utterance instead of scanning for substrings. What survives here is
        # the plain keyword scan, whose answer is the DEFAULT the classifier
        # may override -- never the final word.
        for keyword in self.WEATHER_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_WEATHER, Priority.HIGH
```

- [ ] **Step 4: Agregar el parametro y el metodo**

En `__init__`, agregar el parametro `climate_classifier=None` (despues de `music_dispatcher`) y documentarlo en el docstring:

```python
        self.climate_classifier = climate_classifier
```

Agregar el metodo junto a `_classify_request`:

```python
    async def _resolve_climate_path(self, text: str, path: PathType) -> PathType:
        """Let the classifier arbitrate the climate/weather ambiguity.

        Only ever moves the decision between FAST_DOMOTICS and FAST_WEATHER, and
        only when the text contains contested vocabulary. Everything else is
        returned untouched. On abstention the rule-based `path` stands, so the
        worst case of this method is exactly the pre-classifier behaviour.

        Args:
            text: User text, post-STT (original case, not lowered).
            path: The path the keyword rules already chose.

        Returns:
            The final path.
        """
        if self.climate_classifier is None:
            return path
        if path not in (PathType.FAST_DOMOTICS, PathType.FAST_WEATHER):
            return path
        if not has_contested_vocabulary(text):
            return path

        intent = await self.climate_classifier.classify(text)
        if intent is None:
            self._climate_fallbacks += 1
            logger.info(
                "climate_intent: abstained, keeping rule decision %s (%d fallbacks so far)",
                path.value, self._climate_fallbacks,
            )
            return path

        resolved = (
            PathType.FAST_DOMOTICS if intent is ClimateIntent.ACTION else PathType.FAST_WEATHER
        )
        if resolved != path:
            logger.info("climate_intent: %s -> %s for %r", path.value, resolved.value, text[:60])
        return resolved
```

Inicializar el contador en `__init__`: `self._climate_fallbacks = 0`.

Agregar el import arriba: `from src.nlu.climate_intent import ClimateIntent, has_contested_vocabulary`.

- [ ] **Step 5: Correr los tests nuevos**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/orchestrator/test_dispatcher_climate_classifier.py -v`
Expected: PASS, los 9 verdes.

- [ ] **Step 6: Cablear en `dispatch()`**

En `src/orchestrator/dispatcher.py:596`, justo despues de la clasificacion por
reglas y antes del `if/elif` que rutea (linea 599):

```python
        # 2. Detectar intent y prioridad
        path, priority = self._classify_request(text_lower, service_filter=service_filter)
        # The rules' answer is the default; the classifier may override it on the
        # climate/weather axis only. No-op for every other path.
        path = await self._resolve_climate_path(text, path)
```

**Es secuencial a proposito, y eso es un cambio respecto del spec.** El spec
proponia solaparlo con el vector search via `asyncio.gather` para un costo neto
de ~23ms. No se hace en esta task: el vector search vive *dentro* de
`_fast_path()` (linea 611), o sea **despues** de la bifurcacion que este arbitraje
tiene que decidir. Solaparlos exige reestructurar `_fast_path`, que es un cambio
invasivo y sin relacion con la correccion del ruteo.

Costo real de la version secuencial, con los numeros medidos: +71ms p50 sobre el
13.5% del trafico = **~9.6ms de promedio**. El subconjunto contestado pasa de
150-280ms a 221-351ms, o sea que puede rozar el techo de 300ms en su cola.

Queda como follow-up explicito con su propia medicion, no como deuda difusa:
medir la latencia real del path contestado en produccion y, si el p95 rompe los
300ms, recien ahi reestructurar `_fast_path` para solapar. Anotarlo en el spec
bajo "Resultado del eval" al cerrar el trabajo.

- [ ] **Step 7: Mudar de capa el unico caso de test que cambia**

En `tests/unit/orchestrator/test_dispatcher_world_routing.py`, en
`test_domotics_climate_adjacency_guard_finding_3`, **sacar** este caso de la lista
parametrizada:

```python
    ("¿tengo que prender el clima o hace calor afuera?", PathType.FAST_WEATHER),
```

Verificado por ejecucion el 2026-08-04: los otros 8 casos siguen ruteando igual sin
el guard. Este es el unico que dependia de el.

Renombrar el test a `test_climate_collision_cases_still_route_by_rules` y reemplazar
su docstring/comentario por:

```python
    # The verb-adjacency guard these cases used to exercise was removed on
    # 2026-08-04 (see docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md).
    # Eight of the nine original cases route identically without it -- the
    # two-word WEATHER_KEYWORDS entries already covered them. The ninth,
    # "¿tengo que prender el clima o hace calor afuera?", now depends on the
    # classifier and is asserted in test_dispatcher_climate_classifier.py
    # instead: it moved to the layer where the behaviour actually lives.
```

Agregar el caso mudado a `test_dispatcher_climate_classifier.py`:

```python
@pytest.mark.asyncio
async def test_hybrid_question_moved_from_the_rules_layer():
    """Was case 9 of test_domotics_climate_adjacency_guard_finding_3.

    The rules alone now answer FAST_DOMOTICS for this ("prende" is a substring
    of "prender"); the classifier is what makes it FAST_WEATHER again.
    """
    d, _ = _dispatcher(ClimateIntent.QUERY)
    text = "¿tengo que prender el clima o hace calor afuera?"
    assert d._classify_request(text.lower())[0] == PathType.FAST_DOMOTICS
    assert await d._resolve_climate_path(text, PathType.FAST_DOMOTICS) == PathType.FAST_WEATHER
```

- [ ] **Step 8: Correr la suite completa**

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/ tests/integration/ tests/safety/ -q`

Cuenta esperada, sumando caso parametrizado por caso parametrizado:

| origen | delta |
|---|---:|
| baseline | 2832 |
| Task 2 — `stop` en `complete()` | +3 |
| Task 3 — gate (8 positivos + 6 negativos + 1 substring) | +15 |
| Task 3 — parseo (3) + salida no reconocida (7) | +10 |
| Task 3 — fallos (4) + contrato de llamada (1) + tripwire (1) | +6 |
| Task 5 — clasificador (1+1+1+1+1+4) | +9 |
| Task 5 — caso mudado de capa | +1 |
| Task 5 — caso sacado de la parametrizacion del guard | -1 |
| **total** | **2875** |

Expected: `2875 passed, 1 xfailed`.

Si el numero no coincide exacto, **no seguir**: identificar cada test que se
movio y por que antes de commitear. El baseline es 2832 passed / 1 xfailed.

- [ ] **Step 9: Commit**

```bash
cd ~/Documents/kza
git add src/orchestrator/dispatcher.py tests/unit/orchestrator/
git commit -m "feat(orchestrator): el clasificador arbitra la ambiguedad clima/AC

Borra el guard de adyacencia verbo-sustantivo y las keywords de la ronda 2.
El clasificador ocupa ese lugar en vez de apilarse encima. Solo puede mover
la decision entre FAST_DOMOTICS y FAST_WEATHER; ante abstencion vale la
regla, asi que el peor caso es el comportamiento previo.

Verificado: 8 de los 9 casos del guard rutean igual sin el. El noveno se
muda a la capa donde ahora vive la decision."
```

---

### Task 6: Cablear en `main.py` y documentar

**Files:**
- Modify: `src/main.py` (donde se construye `RequestDispatcher`)
- Modify: `config/settings.yaml` (bloque `nlu:`)

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: el clasificador vivo en produccion, apagable por config.

- [ ] **Step 1: Agregar la config**

En `config/settings.yaml`, dentro del bloque `nlu:` existente:

```yaml
  # Desambiguacion clima/AC por inferencia (2026-08-04). En rioplatense "clima"
  # es el tiempo Y el aire acondicionado; tres rondas de reglas por substrings
  # no lograron separarlos (ver docs/superpowers/specs/2026-08-04-ruteo-clima-por-modelo-design.md).
  # Solo se consulta al 7B cuando el texto trae vocabulario contestado: ~13.5%
  # del trafico medido. Ante timeout o endpoint caido vale la regla, asi que
  # apagarlo devuelve exactamente el comportamiento anterior.
  climate_intent:
    enabled: true
    timeout_s: 0.15   # p95 medido 118ms; moverlo obliga a revisar el umbral del eval
```

- [ ] **Step 2: Cablear la DI**

En `src/main.py`, donde se construye el `RequestDispatcher`, crear el clasificador
reusando el `FastRouter` que ya existe (no instanciar uno nuevo: comparte el
prefix cache de `:8101`):

```python
    climate_cfg = config.get("nlu", {}).get("climate_intent", {})
    climate_classifier = (
        ClimateIntentClassifier(router, timeout_s=climate_cfg.get("timeout_s", 0.15))
        if climate_cfg.get("enabled", False) and router is not None
        else None
    )
```

y pasarlo: `climate_classifier=climate_classifier`.

Import arriba: `from src.nlu.climate_intent import ClimateIntentClassifier`.

- [ ] **Step 3: Verificar que el sistema arranca**

Run: `cd ~/Documents/kza && .venv/bin/python3 -c "import src.main"`
Expected: sin excepciones (verifica la cadena de imports, no el arranque completo).

Run: `cd ~/Documents/kza && .venv/bin/python3 -m pytest tests/unit/ tests/integration/ tests/safety/ -q`
Expected: mismo total que en Task 5 Step 8.

- [ ] **Step 4: Commit**

```bash
cd ~/Documents/kza
git add src/main.py config/settings.yaml
git commit -m "feat(main): cablear el clasificador clima/AC, apagable por config"
```

- [ ] **Step 5: Anotar los hallazgos laterales**

Los cuatro hallazgos de la seccion "Hallazgos laterales" del spec no se arreglan
en este plan. Los dos que son comentarios mentirosos en config **si** se corrigen
acá, porque cuestan una linea y desinforman a quien lea:

En `config/settings.yaml:435-437`, reemplazar el comentario que dice que el
fallback es GLM-Air local:

```yaml
  # Fallback de startup: el dispatcher arranca contra el fast endpoint :8101.
  # Si :8101 esta caido al boot, el LLMRouter rota al reasoner_cloud (MiniMax
  # via gateway, timeout 60s) -- NO a un modelo local: GLM-Air en :8200 se
  # retiro el 2026-05-29 y quedo solo como rollback comentado mas arriba.
  # Si ambos caen, KZA degrada a regex+ChromaDB.
```

En `config/settings.yaml:299`, corregir la latencia:

```yaml
  # Latencia warm medida 2026-08-04: 71ms p50 / 118ms p95 para clasificacion
  # binaria (~150 tok de prompt). La cifra anterior de 245-272ms era del
  # ExecStart con -ngl 20; el drop-in de systemd ya corre -ngl 99.
```

```bash
cd ~/Documents/kza
git add config/settings.yaml
git commit -m "docs(config): corregir dos comentarios desactualizados

El fallback de :8101 no es GLM-Air local desde 2026-05-29 (es MiniMax
cloud). La latencia warm del router es 71-118ms, no 245-272ms: el
drop-in ya corre -ngl 99."
```

---

## Verificacion final (no automatizable)

Lo que este plan **no** prueba, y arrastra desde la Task 3 original: la
verificacion end-to-end por voz. Nada de esto demuestra que `"prendé el clima"`
prenda el aire de verdad contra el HA real.

Antes de dar el trabajo por terminado, alguien tiene que hablarle al dispositivo
y confirmar, como minimo:

1. `"prendé el clima"` -> prende el aire (no da el pronostico)
2. `"prendé ya el clima, hace calor"` -> prende el aire (el bug vivo de hoy)
3. `"qué tiempo hace"` -> da el clima hablado (no prende nada)
4. `"¿hace calor afuera?"` -> da el clima hablado
5. Latencia percibida sin regresion respecto de hoy

Deploy al server: seguir `docs/superpowers/specs/` runbook de deploy
(`reference_runbook_deploy_main_2026-08-03` en memoria). Recordar reiniciar
tambien `kza-code-index`. **Consultar antes de reiniciar `kza-voice`**: es
produccion real y la usa el hogar a diario.
