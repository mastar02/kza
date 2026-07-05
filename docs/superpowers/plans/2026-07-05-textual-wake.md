# Wake textual "nexa" (Etapa A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canal de disparo textual: si una utterance del ambient (no-TV) contiene "nexa", emitir un CommandEvent pretranscripto al request_router — red de seguridad del wake acústico débil.

**Architecture:** Componente puro `TextualWakeDetector` (matcher + dedup) en `src/ambient/textual_wake.py`; hook de 1 llamada en `AmbientTranscriber._handle_segment` (fail-open); wiring por `attach_command_dispatch()` desde main.py (patrón attach existente); config en `ambient.textual_wake`. El CommandEvent lleva `wake_text` (el router ya saltea el STT con eso) + el audio del segmento (speaker-ID/emotion siguen funcionando).

**Tech Stack:** Python 3.13 stdlib (difflib para edit distance NO — es token corto: implementar Levenshtein ≤1 inline trivial), pytest-asyncio auto.

**Spec:** `docs/superpowers/specs/2026-07-05-textual-wake-design.md`

## Global Constraints

- Python del venv SIEMPRE: `/Users/yo/Documents/kza/.venv/bin/python -m pytest ...`
- Imports absolutos; type hints; docstrings Google-style; `logging.getLogger(__name__)`; async/await para I/O.
- Log prefix del canal: `[TextualWake]` — todo disparo/skip decisorio se loguea INFO con texto, room, source, speaker y decisión.
- El hook en `_handle_segment` es FAIL-OPEN: una excepción del detector jamás impide persistir la utterance ni mata el worker (envolver en try/except).
- El detector NUNCA dispara con `source == "tv"`. Sin "nexa" (matcher) → nada. Config `ambient.textual_wake.{enabled, dedup_window_s, variants, max_edit_distance}`.
- "alexa" NO matchea (edit distance 2 — verificarlo con test).
- Match a nivel TOKEN del texto normalizado (lowercase sin acentos/puntuación); "next up" es bigrama especial (dos tokens consecutivos).
- Rama `feat/textual-wake`; commits `feat(ambient): ...`.

---

### Task 1: TextualWakeDetector (matcher + dedup, lógica pura)

**Files:**
- Create: `src/ambient/textual_wake.py`
- Test: `tests/unit/ambient/test_textual_wake.py` (crear dir con `__init__.py` si no existe)

**Interfaces:**
- Consumes: nada del proyecto.
- Produces:
  - `normalize_text(text: str) -> str` (lowercase, sin acentos, sin puntuación, espacios colapsados — mismo algoritmo que command_gate._normalize, duplicado documentado: ambient no debe importar nlu).
  - `matches_wake(text: str, variants: tuple[str, ...] = ("nexa", "next up"), max_edit_distance: int = 1) -> bool` — True si algún token (o bigrama para variantes con espacio) matchea exacto una variante, o algún token está a edit-distance ≤ max del token base "nexa". "alexa" NO matchea.
  - `class TextualWakeDetector` con `__init__(dispatch_fn, last_acoustic_command_ts_fn, enabled=True, dedup_window_s=8.0, variants=("nexa","next up"), max_edit_distance=1)`:
    - `dispatch_fn: Callable[[CommandEvent], Awaitable[dict]]` (el process_command del router, inyectado)
    - `last_acoustic_command_ts_fn: Callable[[str], float]` (timestamp monotónico del último dispatch acústico por room; 0.0 = nunca)
    - `async maybe_dispatch(room_id: str, text: str, source: str, speaker: str | None, audio) -> bool` — True si despachó. Reglas en orden: disabled→False; source=="tv"→False (log INFO skip); no match→False (sin log, es el caso común); dedup (now - last_acoustic < window)→False (log INFO skip); él mismo despachó en esa room hace <window→False; si pasa todo → construir `CommandEvent(audio=audio, room_id=room_id, wake_text=text, wake_score=1.0)` → `await dispatch_fn(event)` → log INFO `[TextualWake] DISPARO room=... text=...` → registrar su propio ts por room → True. Excepciones del dispatch: log ERROR, return False (no re-raise).
    - Reloj inyectable `now_fn: Callable[[], float]` (default time.monotonic) para tests.

- [ ] **Step 1: Write the failing tests** — casos mínimos obligatorios (estilo pytest del repo, sin clases o con clases como test_ambient_guard, mirar vecinos):

```python
# tests/unit/ambient/test_textual_wake.py — casos clave (el implementador puede sumar):
# matcher:
#   matches_wake("Nexa, apagá la luz") is True
#   matches_wake("Next up, apagá la luz") is True            (bigrama)
#   matches_wake("neza apaga") is True                        (edit dist 1)
#   matches_wake("lexa apaga") is True                        (edit dist 1)
#   matches_wake("alexa apaga la luz") is False               (dist 2, excluida)
#   matches_wake("la anexa el documento") is False            ("anexa" dist 1 de "nexa"... ¡VERIFICAR!: a-n-e-x-a vs n-e-x-a = 1 inserción → matchearía. DECISIÓN: aceptable v1 (falso positivo raro en habla; documentar en docstring) — el test pinnea el comportamiento REAL elegido, no lo oculta)
#   matches_wake("") is False; matches_wake("apagá la luz") is False
# detector (con FakeClock y dispatch mock):
#   dispara con source="human_direct" y texto con nexa → dispatch llamado, CommandEvent con wake_text=texto y room correcto
#   NO dispara: source="tv" / disabled / sin match
#   dedup acústico: last_acoustic hace 3s (window 8) → no dispara; hace 20s → dispara
#   dedup propio: dos utterances con nexa en 3s → solo la primera despacha
#   dispatch_fn que raise → return False, sin excepción propagada
```

Escribir los tests COMPLETOS (no comentarios) siguiendo ese contrato, correr → RED (ModuleNotFoundError).

- [ ] **Step 2: Implement** `src/ambient/textual_wake.py` completo según Interfaces. Levenshtein ≤1 inline (early-exit por diferencia de largo >1). Docstring del módulo documenta: variantes calibradas con ambient.db 2026-07-05 (nexa=19, next up=7, lexa=1) y el falso positivo aceptado "anexa".

- [ ] **Step 3: GREEN** — `pytest tests/unit/ambient/ -v` todos, 0 warnings nuevos.

- [ ] **Step 4: Commit** `feat(ambient): TextualWakeDetector — matcher nexa + dedup dual (acústico/propio)`

---

### Task 2: Integración — hook en transcriber, timestamp acústico, wiring main, config

**Files:**
- Modify: `src/ambient/transcriber.py` (constructor + `_handle_segment` + `attach_textual_wake()`)
- Modify: `src/pipeline/multi_room_audio_loop.py` (registrar ts de dispatch por room + getter)
- Modify: `src/main.py` (wiring post-request_router)
- Modify: `config/settings.yaml` (bloque `ambient.textual_wake`)
- Test: `tests/unit/ambient/test_textual_wake_integration.py`

**Interfaces:**
- Consumes: `TextualWakeDetector` (Task 1); `AmbientTranscriber._handle_segment` (existente); `MultiRoomAudioLoop._dispatch_command` (existente).
- Produces:
  - `AmbientTranscriber.attach_textual_wake(detector) -> None` (attr `self._textual_wake = None` en __init__).
  - En `_handle_segment`, DESPUÉS de `await self._store.add(utt)` (la utterance SIEMPRE se persiste primero) y solo si `self._textual_wake is not None`: `try: await self._textual_wake.maybe_dispatch(room_id, text, source, speaker, audio=<audio asr del seg — la misma vista usada para STT>) except Exception: logger.exception(...)`.
  - `MultiRoomAudioLoop`: en `_dispatch_command`, tras obtener result: `self._last_command_dispatch_ts[event.room_id] = time.monotonic()` (dict nuevo init {}) + método público `last_command_dispatch_ts(room_id) -> float` (0.0 default).
  - main.py: tras construir `request_router`, si `ambient_path` existe y `config ambient.textual_wake.enabled` (default False en código, true se decide en settings): construir detector con `dispatch_fn=request_router.process_command`, `last_acoustic_command_ts_fn=multi_room_loop.last_command_dispatch_ts`, kwargs de config; `ambient_path.transcriber.attach_textual_wake(detector)`; log INFO `[TextualWake] activo (rooms=...)`. Best-effort try/except como el bloque ambient existente.
  - settings.yaml al final del bloque `ambient:` (respetar indentación del bloque):

```yaml
  # Wake textual (spec 2026-07-05): si una utterance no-TV contiene "nexa",
  # dispara el command path con el texto ya transcripto (wake_text). Red de
  # seguridad del wake acústico débil post-mudanza del mic. Kill-switch acá.
  textual_wake:
    enabled: true
    dedup_window_s: 8.0
    variants: ["nexa", "next up"]
    max_edit_distance: 1
```

- [ ] **Step 1: tests de integración** (transcriber con detector mock: utterance persiste ANTES del dispatch; detector que raise no impide persistir; source tv no llega a dispatch pero SÍ persiste; loop: ts se registra en dispatch y getter devuelve 0.0 para room desconocida). RED primero.
- [ ] **Step 2: implementar** los 4 archivos. En main: verificar orden real (request_router se construye DESPUÉS del bloque ambient → el attach va después de request_router; validar línea exacta leyendo main.py).
- [ ] **Step 3: GREEN** — `pytest tests/unit/ambient/ tests/unit/pipeline/ -q` sin fallas nuevas (baseline: 1 falla pre-existente test_request_router_gate conocida). Validar yaml: `python -c "import yaml; c=yaml.safe_load(open('config/settings.yaml'))['ambient']['textual_wake']; assert c['enabled'] is True and c['dedup_window_s']==8.0; print('yaml OK')"`.
- [ ] **Step 4: Commit** `feat(ambient): wire TextualWakeDetector — hook post-persist, ts acústico, config`

---

### Task 3: Verificación entera + docs mínimas

**Files:**
- Modify: `CLAUDE.md` (fila en Mapa de Archivos: `src/ambient/textual_wake.py`)
- Test: suite completa de vecindad

- [ ] **Step 1:** fila en CLAUDE.md tras la de `src/rooms/room_context.py`: `| src/ambient/textual_wake.py | Wake textual "nexa" sobre stream ambient (:red de seguridad) | Cambios en disparo textual |` (ajustar formato a la tabla real).
- [ ] **Step 2:** `pytest tests/unit/ambient/ tests/unit/pipeline/ tests/unit/nlu/ -q` → sin fallas nuevas; `python -c "import src.ambient.textual_wake, src.ambient.transcriber; print('imports OK')"`.
- [ ] **Step 3: Commit** `docs(ambient): textual wake en mapa de archivos + verificación de suite`

## Post-plan (manual)

1. Merge a main + push + deploy (pull + restart coordinado).
2. Validación de voz: decir "nexa apagá la luz" en voz BAJA (para que el acústico NO dispare) → verificar `[TextualWake] DISPARO` en journal y ejecución ~1.5-3s después. Decir un comando normal (acústico dispara) → verificar dedup (textual se abstiene).
3. Auditar `[TextualWake]` los primeros días (falsos positivos tipo "anexa").
