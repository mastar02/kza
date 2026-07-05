# BoH-es (blocklist + prompt_echo + cosecha) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cortar en el CommandGate las alucinaciones residuales de Whisper que hoy pasan al router ('Aplausos', eco del initial_prompt) y agregar un script que cosecha candidatas nuevas desde el journal para curación manual.

**Architecture:** Todo sobre infra existente: una entrada nueva en `_NOISE_PHRASES`, una hard rule nueva `prompt_echo` en `CommandAcceptanceGate` (similitud por bloque contiguo contra oraciones del initial_prompt, inyectado por DI desde main), el prefijo en `earcon_gate`, y un script standalone stdlib que parsea el journal y propone candidatas (nunca auto-agrega).

**Tech Stack:** Python 3.13, stdlib (`difflib`, `re`, `argparse`), pytest (asyncio no necesario — todo síncrono).

**Spec:** `docs/superpowers/specs/2026-07-05-stt-hallucination-blocklist-design.md`

## Global Constraints

- Python del venv SIEMPRE: `/Users/yo/Documents/kza/.venv/bin/python -m pytest ...`
- Imports absolutos `from src.nlu...`; type hints; docstrings Google-style; `logging.getLogger(__name__)`.
- El gate es **fail-open** (`evaluate` ya envuelve en try/except) — la regla nueva vive adentro de `_hard_reason`, no cambiar esa semántica.
- Reason de la regla nueva: exactamente `"prompt_echo"` (el earcon gate matchea por `startswith`).
- `prompt_echo` solo aplica si la transcripción normalizada tiene ≥4 palabras Y el prompt tiene oraciones válidas (≥4 palabras); gate sin prompt = comportamiento idéntico al actual (backward compat: los tests existentes de `test_command_gate.py` deben seguir pasando SIN modificaciones).
- Umbral: bloque contiguo común más largo / len(transcripción normalizada) ≥ 0.8 (constante módulo `_PROMPT_ECHO_RATIO = 0.8`). *(superseded en review: 0.9 — ver spec, mediciones 0.879/0.963)*
- El script de cosecha es standalone stdlib (corre en el server sin venv); import de `src.nlu.command_gate` OPCIONAL (si falla, sigue sin excluir ya-bloqueadas). SOLO propone — nunca escribe código/config.
- Commits `feat(gate): ...` / `feat(tools): ...`; working dir `/Users/yo/Documents/kza`, rama `feat/stt-hallucination-blocklist`.

---

### Task 1: Blocklist "aplausos" + hard rule `prompt_echo` + prefijo earcon

**Files:**
- Modify: `src/nlu/command_gate.py` (tupla `_NOISE_PHRASES` ~línea 23; constructor ~línea 56; `_hard_reason` ~línea 101)
- Modify: `src/pipeline/earcon_gate.py:10` (`_NOISE_PREFIXES`)
- Modify: `docs/superpowers/specs/2026-07-05-stt-hallucination-blocklist-design.md` (1 línea: mecanismo de similitud)
- Test: `tests/unit/nlu/test_command_gate.py` (agregar al final)
- Test: `tests/unit/pipeline/test_earcon_gate.py` (agregar al final)

**Interfaces:**
- Consumes: `_normalize(text) -> str` y `_NOISE_PHRASES`/`_FILLER_WORDS` existentes en `command_gate.py`; `should_play_earcon(reason, wake_score, rms, cfg)` existente en `earcon_gate.py`.
- Produces: `CommandAcceptanceGate(..., initial_prompt: str | None = None)` (param NUEVO, keyword, default None — Task 2 lo cablea); helper módulo `_prompt_sentences(prompt: str | None) -> tuple[str, ...]`; reason `"prompt_echo"` emitido por `_hard_reason`.

- [ ] **Step 1: Write the failing tests**

Agregar al final de `tests/unit/nlu/test_command_gate.py`:

```python
# --- BoH-es: blocklist + prompt_echo (spec 2026-07-05) ---

REAL_PROMPT = (
    "Esto es un asistente de voz llamado Nexa que controla luces, aire "
    "acondicionado, persianas y música en el escritorio, el living, la cocina, "
    "el baño y el hall. Habla rioplatense con voseo: prendé, apagá, subí, "
    "bajá, poné."
)


def test_rejects_aplausos_as_noise():
    d = _gate().evaluate("¡Aplausos!")
    assert d.accept is False
    assert d.reason.startswith("noise_phrase")


def test_prompt_echo_rejects_prompt_fragment():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("Esto es un asistente de voz.")
    assert d.accept is False
    assert d.reason == "prompt_echo"


def test_prompt_echo_rejects_slightly_garbled_fragment():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("Esto es un asistente de vos")
    assert d.accept is False
    assert d.reason == "prompt_echo"


def test_prompt_echo_does_not_reject_real_commands():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    for cmd in (
        "nexa prendé la luz del escritorio",
        "nexa subí el volumen en el living",
        "activá la escena lectura",
        "apagá el aire acondicionado del living",
    ):
        d = g.evaluate(cmd)
        assert d.accept is True, f"false-reject: {cmd!r} → {d.reason}"


def test_prompt_echo_skips_short_texts():
    # <4 palabras jamás dispara prompt_echo aunque estén en el prompt
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("en el living")
    assert d.reason != "prompt_echo"


def test_prompt_echo_inactive_without_prompt():
    d = CommandAcceptanceGate().evaluate("Esto es un asistente de voz.")
    assert d.accept is True
```

Agregar al final de `tests/unit/pipeline/test_earcon_gate.py`:

```python
def test_silent_on_prompt_echo():
    cfg = {"enabled": True, "min_wake_score": 0.55, "min_rms": 0.02, "reasons": ("empty",)}
    assert should_play_earcon("prompt_echo", 0.9, 0.5, cfg) is False
```

(Si el archivo importa `should_play_earcon` con otro nombre/estilo, seguir el patrón de los tests existentes del archivo.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py tests/unit/pipeline/test_earcon_gate.py -v`
Expected: FAIL — `TypeError: CommandAcceptanceGate.__init__() got an unexpected keyword argument 'initial_prompt'` en los tests nuevos (y `test_rejects_aplausos_as_noise` falla con accept=True). Los tests preexistentes PASAN.

- [ ] **Step 3: Implement**

En `src/nlu/command_gate.py`:

a) Import: agregar `import difflib` junto a los imports stdlib existentes.

b) En `_NOISE_PHRASES`, agregar la entrada con su comentario:

```python
    # BoH-es (spec 2026-07-05): alucinación de Whisper sobre audio
    # ininteligible; 25 ocurrencias en 48h de prod, 0 comandos reales.
    "aplausos",
```

c) Constante módulo (debajo de `_FILLER_WORDS`):

```python
# prompt_echo: fracción mínima de la transcripción que debe aparecer como
# bloque contiguo dentro de una oración del initial_prompt para considerarla
# eco (Whisper regurgitando el prompt). Ratio sobre len(transcripción), NO
# ratio simétrico de SequenceMatcher: el eco es un FRAGMENTO corto de una
# oración larga y el ratio simétrico quedaría ~0.3.
_PROMPT_ECHO_RATIO = 0.8
_PROMPT_ECHO_MIN_WORDS = 4
```

d) Helper módulo (encima de la clase):

```python
def _prompt_sentences(prompt: str | None) -> tuple[str, ...]:
    """Oraciones normalizadas del initial_prompt con ≥4 palabras.

    Se usan como referencia para detectar eco del prompt en la transcripción
    (Whisper a veces devuelve texto del propio initial_prompt).
    """
    if not prompt:
        return ()
    sentences = []
    for raw in prompt.split("."):
        norm = _normalize(raw)
        if len(norm.split()) >= _PROMPT_ECHO_MIN_WORDS:
            sentences.append(norm)
    return tuple(sentences)
```

e) Constructor: agregar el parámetro keyword al final de la firma y su
docstring Args, y en el cuerpo:

```python
        initial_prompt: str | None = None,
```
```python
            initial_prompt: Texto del stt.initial_prompt configurado. Si se
                provee, transcripciones que son eco del prompt (Whisper lo
                regurgita sobre audio ininteligible) se rechazan con reason
                'prompt_echo'. None = regla inactiva.
```
```python
        self._prompt_sentences = _prompt_sentences(initial_prompt)
```

f) En `_hard_reason`, después del bloque `word_repetition` y ANTES del bloque
`missing_wake`:

```python
        if self._prompt_sentences and len(words) >= _PROMPT_ECHO_MIN_WORDS:
            for sentence in self._prompt_sentences:
                m = difflib.SequenceMatcher(None, norm, sentence).find_longest_match(
                    0, len(norm), 0, len(sentence)
                )
                if m.size / len(norm) >= _PROMPT_ECHO_RATIO:
                    return "prompt_echo"
```

(Nota: `words` ya existe en el método — lo computa la regla de repetition.
Si el orden actual computa `words` después, mover la línea `words = norm.split()`
arriba de ambas reglas, sin cambiar el orden de las reglas previas.)

En `src/pipeline/earcon_gate.py:10`:

```python
_NOISE_PREFIXES = ("noise_phrase", "filler_word", "word_repetition", "missing_wake", "prompt_echo")
```

En la spec (`docs/superpowers/specs/2026-07-05-stt-hallucination-blocklist-design.md`),
reemplazar la línea del mecanismo:

```
- Regla: si `SequenceMatcher(None, _normalize(text), oracion).ratio() >= 0.8`
  para alguna oración del prompt → reject `prompt_echo`.
```
por:

```
- Regla: bloque contiguo común más largo (SequenceMatcher.find_longest_match)
  entre la transcripción normalizada y cada oración del prompt; si
  `bloque / len(transcripción) >= 0.8` → reject `prompt_echo`. (Corrección
  2026-07-05: el ratio simétrico daba ~0.3 para ecos fragmento-de-oración.)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/nlu/test_command_gate.py tests/unit/pipeline/test_earcon_gate.py -v`
Expected: TODOS passed (nuevos + preexistentes), 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add src/nlu/command_gate.py src/pipeline/earcon_gate.py tests/unit/nlu/test_command_gate.py tests/unit/pipeline/test_earcon_gate.py docs/superpowers/specs/2026-07-05-stt-hallucination-blocklist-design.md
git commit -m "feat(gate): blocklist 'aplausos' + hard rule prompt_echo por bloque contiguo"
```

---

### Task 2: Cablear `initial_prompt` en la DI de main.py

**Files:**
- Modify: `src/main.py:1221-1228` (construcción de `CommandAcceptanceGate`)

**Interfaces:**
- Consumes: `CommandAcceptanceGate(initial_prompt=...)` (Task 1); variable `stt_cfg` ya definida en `main()` (se usa en `main.py:704` — verificar con `grep -n "stt_cfg =" src/main.py` que está asignada antes de la línea ~1221; si el nombre difiere en ese scope, usar la variable que contiene `config["stt"]`).
- Produces: gate de producción con la regla prompt_echo activa usando el prompt real del settings.

- [ ] **Step 1: Modificar la construcción**

En `src/main.py`, agregar el kwarg al final de la llamada existente:

```python
    command_gate = CommandAcceptanceGate(
        wake_words=_gate_wake_words,
        enforce_confidence=_gate_cfg.get("enforce_confidence", False),
        max_no_speech_prob=_gate_cfg.get("max_no_speech_prob", 0.60),
        min_avg_logprob=_gate_cfg.get("min_avg_logprob", -1.20),
        enforce_compression_ratio=_gate_cfg.get("enforce_compression_ratio", False),
        max_compression_ratio=_gate_cfg.get("max_compression_ratio", 2.2),
        # prompt_echo (spec 2026-07-05): Whisper regurgita el initial_prompt
        # sobre audio ininteligible — el gate lo corta antes del router.
        initial_prompt=stt_cfg.get("initial_prompt"),
    )
```

NO tocar el fallback `command_gate or CommandAcceptanceGate(...)` de
`request_router.py:309` — main siempre inyecta el gate; el fallback queda sin
prompt (regla inactiva ahí, comportamiento seguro).

- [ ] **Step 2: Verificar sintaxis y scope**

Run:
```bash
grep -n "stt_cfg = \|stt_cfg=" src/main.py | head -3
/Users/yo/Documents/kza/.venv/bin/python -c "import ast; ast.parse(open('src/main.py').read()); print('main.py parse OK')"
/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/nlu/ -q
```
Expected: `stt_cfg` asignada en una línea ANTERIOR a la construcción del gate; `main.py parse OK`; tests nlu passed.

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat(gate): inyectar stt.initial_prompt al CommandAcceptanceGate (prompt_echo activo)"
```

---

### Task 3: Script de cosecha `tools/harvest_hallucinations.py`

**Files:**
- Create: `tools/harvest_hallucinations.py`
- Create: `tests/unit/tools/__init__.py` (vacío; crear el dir si no existe)
- Test: `tests/unit/tools/test_harvest_hallucinations.py`

**Interfaces:**
- Consumes: formato real del journal de kza-voice (`--output=cat`, con códigos ANSI): líneas `[CommandGate] accept=True ... text='X'` y `[LLMRouter 718ms] is_command=False intent=None reason=None text='X'`. Import opcional de `src.nlu.command_gate` (`_NOISE_PHRASES`, `_FILLER_WORDS`, `_normalize`).
- Produces: CLI `journalctl ... --output=cat | python tools/harvest_hallucinations.py [--min-count N] [--file PATH]`. Funciones testeables: `strip_ansi(line) -> str`, `normalize(text) -> str`, `parse_candidates(lines) -> dict[str, Candidate]` (`Candidate(text, count, first_seen, last_seen)`), `filter_candidates(cands, min_count, blocked_phrases, filler_words) -> list[Candidate]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/tools/test_harvest_hallucinations.py
"""Tests del script de cosecha de alucinaciones (carga por importlib: tools/ no es package)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "harvest_hallucinations",
    Path(__file__).resolve().parents[3] / "tools" / "harvest_hallucinations.py",
)
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)

ANSI = "\x1b[32m"
RESET = "\x1b[0m"

JOURNAL = f"""\
09:44:08.341 INFO src.nlu.command_gate [CommandGate] accept=True reason=ok no_speech=0.0 avg_logprob=-0.50 compression=0.5 would_reject=None text='Aplausos'
09:44:09.059 INFO {ANSI}...ipeline.request_router{RESET} [LLMRouter 718ms] is_command=False intent=None reason=None text='Aplausos'
09:52:28.541 INFO src.nlu.command_gate [CommandGate] accept=True reason=ok text='Nexa, prendé la luz.'
09:52:29.100 INFO ...ipeline.request_router [LLMRouter 500ms] is_command=True intent=turn_on reason=None text='Nexa, prendé la luz.'
10:01:00.000 INFO ...ipeline.request_router [LLMRouter 600ms] is_command=False intent=None reason=None text='Aplausos'
10:02:00.000 INFO ...ipeline.request_router [LLMRouter 600ms] is_command=False intent=None reason=None text='¡Aplausos!'
10:03:00.000 INFO ...ipeline.request_router [LLMRouter 610ms] is_command=False intent=None reason=None text='Gracias por ver el video.'
10:04:00.000 INFO ...ipeline.request_router [LLMRouter 620ms] is_command=False intent=None reason=None text='Una sola vez'
"""


def test_strip_ansi():
    assert harvest.strip_ansi(f"{ANSI}hola{RESET}") == "hola"


def test_parse_counts_router_rejects_normalized():
    cands = harvest.parse_candidates(JOURNAL.splitlines())
    # 'Aplausos' x2 + '¡Aplausos!' x1 agrupan por normalización
    assert cands["aplausos"].count == 3
    assert cands["aplausos"].first_seen == "09:44:09.059"
    assert cands["aplausos"].last_seen == "10:02:00.000"
    # comandos reales (is_command=True) NO son candidatas
    assert "nexa prende la luz" not in cands


def test_filter_excludes_blocked_and_low_count():
    cands = harvest.parse_candidates(JOURNAL.splitlines())
    out = harvest.filter_candidates(
        cands,
        min_count=2,
        blocked_phrases=("gracias por ver",),
        filler_words=frozenset(),
    )
    texts = [c.text for c in out]
    assert texts == ["aplausos"]          # única con count>=2 no bloqueada
    # 'gracias por ver el video' excluida por substring de blocked_phrases
    # 'una sola vez' excluida por min_count


def test_filter_excludes_fillers():
    cands = harvest.parse_candidates(
        ["10:00:00.000 X [LLMRouter 1ms] is_command=False intent=None reason=None text='Dale'"]
    )
    out = harvest.filter_candidates(
        cands, min_count=1, blocked_phrases=(), filler_words=frozenset({"dale"})
    )
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/tools/test_harvest_hallucinations.py -v`
Expected: FAIL — `FileNotFoundError` (tools/harvest_hallucinations.py no existe)

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
# tools/harvest_hallucinations.py
"""Cosecha candidatas a la blocklist de alucinaciones desde el journal.

Detecta textos que el CommandGate ACEPTÓ pero el LLMRouter DESCARTÓ
(is_command=False) — la firma de una alucinación de Whisper que pasó el gate.
SOLO PROPONE: un humano decide qué entra a _NOISE_PHRASES (criterio del fix
2026-06-02: jamás substrings plausibles de comandos válidos).

Uso (en el server):
    journalctl --user -u kza-voice --since '7 days ago' --output=cat \\
        | python3 tools/harvest_hallucinations.py --min-count 3
    python3 tools/harvest_hallucinations.py --file journal_export.txt

Standalone stdlib a propósito; el import de src.nlu.command_gate es opcional
(excluye ya-bloqueadas si está disponible).
"""

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ROUTER_REJECT_RE = re.compile(
    r"\[LLMRouter [^\]]*\] is_command=False .*?text='(?P<text>[^']*)'"
)
_TS_RE = re.compile(r"^(?P<ts>\d{2}:\d{2}:\d{2}\.\d+)")


def strip_ansi(line: str) -> str:
    """Sacar códigos de color ANSI del journal."""
    return _ANSI_RE.sub("", line)


def normalize(text: str) -> str:
    """Lowercase sin acentos ni puntuación (duplicado de command_gate a
    propósito: el script corre standalone en el server)."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


@dataclass
class Candidate:
    """Texto candidato a blocklist con conteo y rango temporal."""

    text: str        # normalizado (clave de agrupación)
    count: int
    first_seen: str  # timestamp de la primera línea (HH:MM:SS.mmm)
    last_seen: str


def parse_candidates(lines) -> dict:
    """Agrupar por texto normalizado los rejects del router.

    Un reject del router implica que el gate aceptó antes (precondición del
    pipeline), así que la línea is_command=False alcanza como firma.
    """
    candidates: dict[str, Candidate] = {}
    for raw in lines:
        line = strip_ansi(raw)
        m = _ROUTER_REJECT_RE.search(line)
        if not m:
            continue
        norm = normalize(m.group("text"))
        if not norm:
            continue
        ts_match = _TS_RE.match(line.strip())
        ts = ts_match.group("ts") if ts_match else ""
        if norm in candidates:
            c = candidates[norm]
            c.count += 1
            c.last_seen = ts or c.last_seen
        else:
            candidates[norm] = Candidate(norm, 1, ts, ts)
    return candidates


def filter_candidates(candidates, min_count, blocked_phrases, filler_words):
    """Filtrar por umbral y exclusiones; ordenar por count desc."""
    out = []
    for c in candidates.values():
        if c.count < min_count:
            continue
        if any(p in c.text for p in blocked_phrases):
            continue
        if c.text in filler_words:
            continue
        out.append(c)
    return sorted(out, key=lambda c: -c.count)


def _load_existing_blocklist():
    """Import opcional de la lista vigente (para excluir ya-bloqueadas)."""
    try:
        from src.nlu.command_gate import _FILLER_WORDS, _NOISE_PHRASES

        return tuple(_NOISE_PHRASES), frozenset(_FILLER_WORDS)
    except Exception:
        print(
            "(aviso: src.nlu.command_gate no importable — no se excluyen "
            "frases ya bloqueadas)",
            file=sys.stderr,
        )
        return (), frozenset()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--file", help="export del journal (default: stdin)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    blocked, fillers = _load_existing_blocklist()
    candidates = filter_candidates(
        parse_candidates(lines), args.min_count, blocked, fillers
    )

    if not candidates:
        print("Sin candidatas nuevas (todo bloqueado o bajo el umbral).")
        return 0

    print(f"{'count':>5}  {'primera':<12} {'última':<12} texto")
    for c in candidates:
        print(f"{c.count:>5}  {c.first_seen:<12} {c.last_seen:<12} {c.text!r}")
    print(
        "\nRevisar a mano antes de agregar a _NOISE_PHRASES "
        "(command_gate.py): nada que sea substring de un comando válido."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`chmod +x tools/harvest_hallucinations.py` y crear `tests/unit/tools/__init__.py` vacío.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/tools/test_harvest_hallucinations.py tests/unit/nlu/ tests/unit/pipeline/test_earcon_gate.py -v`
Expected: todos passed, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add tools/harvest_hallucinations.py tests/unit/tools/
git commit -m "feat(tools): harvest_hallucinations — cosecha de candidatas BoH desde el journal"
```

---

## Post-plan (manual)

1. Merge a main + push + deploy (git pull en server — el hook del code-index reindexa solo) + restart kza-voice coordinado.
2. Primera cosecha real: `journalctl --user -u kza-voice --since '14 days ago' --output=cat | python3 tools/harvest_hallucinations.py` en el server → curar candidatas → PR chico con las aprobadas.
   - Primera cosecha: usar `--since` desde la fecha del deploy (el journal pre-deploy contiene ecos del prompt que prompt_echo ya corta — no proponerlos).
3. Después: audit ciego del shadow Parakeet (siguiente pieza del roadmap STT).
