# Umbral de wake adaptativo por vad en STRICT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que el `AmbientGuard`, en estado STRICT, module el umbral del wake según el `vad_prob` de Silero del audio que disparó el trigger (0.72 con TV/lejos → ~0.50 con voz clara), en vez de un umbral fijo que mata los comandos far-field del usuario durante conversación.

**Architecture:** Cambio acotado a 3 archivos. `AmbientGuard.on_wake` gana un parámetro opcional `wake_vad` y, en STRICT, interpola el umbral entre `strict_wake_score` (duro) y `strict_wake_score_min` (blando) según `wake_vad` contra los breakpoints `strict_vad_lo`/`strict_vad_hi`. Arranca en **shadow** (`strict_vad_adaptive=false`): computa y loguea el umbral adaptativo pero decide con el fijo. `multi_room_audio_loop` corre Silero sobre el `audio_chunk` del wake y pasa `wake_vad`. Fail-safe: `wake_vad=None` → umbral fijo.

**Tech Stack:** Python 3.13, pytest, Silero VAD (`src.ambient.segmenter.make_silero_predictor`), faster-whisper venv `/Users/yo/Documents/kza/.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-09-wake-vad-adaptive-threshold-design.md`

---

## File Structure

- `src/pipeline/ambient_guard.py` — MODIFY: 4 campos nuevos en `AmbientGuardConfig` + `__post_init__` clamp; `_effective_strict_threshold(wake_vad)`; `on_wake` acepta `wake_vad` y aplica el umbral (shadow/enforce).
- `src/pipeline/multi_room_audio_loop.py` — MODIFY: computar `wake_vad` con Silero sobre el audio del wake y pasarlo a `on_wake` vía `_should_accept_wakeword`.
- `config/settings.yaml` — MODIFY: bloque `rooms.ambient_guard` (4 campos nuevos).
- `tests/unit/pipeline/test_ambient_guard.py` — MODIFY: casos de escalado + shadow + fail-safe.

---

## Task 1: Config — campos nuevos + invariantes

**Files:**
- Modify: `src/pipeline/ambient_guard.py` (clase `AmbientGuardConfig`, ~líneas 56-99)
- Test: `tests/unit/pipeline/test_ambient_guard.py`

- [ ] **Step 1: Write the failing test**

Agregar al final de `tests/unit/pipeline/test_ambient_guard.py`:

```python
from src.pipeline.ambient_guard import AmbientGuardConfig


class TestVadAdaptiveConfig:
    def test_new_fields_have_shadow_defaults(self):
        cfg = AmbientGuardConfig()
        assert cfg.strict_vad_adaptive is False  # shadow primero
        assert cfg.strict_wake_score_min == 0.50
        assert cfg.strict_vad_lo == 0.30
        assert cfg.strict_vad_hi == 0.70

    def test_clamps_lo_below_hi(self):
        # lo >= hi rompería la interpolación (división por <=0): clamp, no crash
        cfg = AmbientGuardConfig(strict_vad_lo=0.8, strict_vad_hi=0.4)
        assert cfg.strict_vad_lo < cfg.strict_vad_hi

    def test_clamps_min_not_above_hard(self):
        # el umbral blando nunca debe superar al duro
        cfg = AmbientGuardConfig(strict_wake_score=0.65, strict_wake_score_min=0.90)
        assert cfg.strict_wake_score_min <= cfg.strict_wake_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestVadAdaptiveConfig -v`
Expected: FAIL con `AttributeError: 'AmbientGuardConfig' object has no attribute 'strict_vad_adaptive'`

- [ ] **Step 3: Write minimal implementation**

En `src/pipeline/ambient_guard.py`, agregar los campos a `AmbientGuardConfig` (después de `strict_follow_up_grace_s`, antes de `__post_init__`):

```python
    # Umbral de wake adaptativo por vad en STRICT (spec 2026-06-09). El umbral
    # fijo strict_wake_score (TV/lejos) se interpola hacia strict_wake_score_min
    # (voz clara) según el vad de Silero del audio del trigger. Arranca en
    # shadow: computa y loguea el umbral adaptativo pero decide con el fijo.
    strict_vad_adaptive: bool = False
    strict_wake_score_min: float = 0.50   # umbral con wake_vad >= strict_vad_hi
    strict_vad_lo: float = 0.30           # wake_vad <= esto → umbral duro
    strict_vad_hi: float = 0.70           # wake_vad >= esto → umbral blando
```

En `__post_init__` (después del clamp de cooldown existente) agregar:

```python
        if self.strict_vad_lo >= self.strict_vad_hi:
            logger.warning(
                f"[AmbientGuardConfig] strict_vad_lo={self.strict_vad_lo} >= "
                f"strict_vad_hi={self.strict_vad_hi}; clamping lo a hi-0.1"
            )
            self.strict_vad_lo = self.strict_vad_hi - 0.1
        if self.strict_wake_score_min > self.strict_wake_score:
            logger.warning(
                f"[AmbientGuardConfig] strict_wake_score_min={self.strict_wake_score_min} > "
                f"strict_wake_score={self.strict_wake_score}; clamping a hard"
            )
            self.strict_wake_score_min = self.strict_wake_score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestVadAdaptiveConfig -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ambient_guard.py tests/unit/pipeline/test_ambient_guard.py
git commit -m "feat(guard): config de umbral de wake adaptativo por vad (shadow default)"
```

---

## Task 2: Interpolación del umbral — `_effective_strict_threshold`

**Files:**
- Modify: `src/pipeline/ambient_guard.py` (clase `AmbientGuard`, método nuevo)
- Test: `tests/unit/pipeline/test_ambient_guard.py`

- [ ] **Step 1: Write the failing test**

```python
class TestEffectiveThreshold:
    def test_vad_none_returns_hard(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(None) == 0.72

    def test_vad_at_or_below_lo_returns_hard(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(0.30) == 0.72
        assert guard._effective_strict_threshold(0.10) == 0.72

    def test_vad_at_or_above_hi_returns_soft(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(0.70) == 0.50
        assert guard._effective_strict_threshold(0.95) == 0.50

    def test_vad_midpoint_interpolates_linear(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        # midpoint vad=0.50 → frac 0.5 → 0.72 - 0.5*(0.22) = 0.61
        assert guard._effective_strict_threshold(0.50) == pytest.approx(0.61)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestEffectiveThreshold -v`
Expected: FAIL con `AttributeError: 'AmbientGuard' object has no attribute '_effective_strict_threshold'`

- [ ] **Step 3: Write minimal implementation**

En `src/pipeline/ambient_guard.py`, dentro de `class AmbientGuard`, en la sección `# ---- internos ----` (antes de `_room`):

```python
    def _effective_strict_threshold(self, wake_vad: float | None) -> float:
        """Umbral de wake en STRICT, interpolado por vad del trigger.

        wake_vad None (sin señal) → umbral duro (fail-safe: nunca más
        permisivo por falta de dato). <= lo → duro; >= hi → blando; lineal
        entre medio.
        """
        hard = self.config.strict_wake_score
        if wake_vad is None:
            return hard
        soft = self.config.strict_wake_score_min
        lo, hi = self.config.strict_vad_lo, self.config.strict_vad_hi
        if wake_vad <= lo:
            return hard
        if wake_vad >= hi:
            return soft
        frac = (wake_vad - lo) / (hi - lo)
        return hard - frac * (hard - soft)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestEffectiveThreshold -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ambient_guard.py tests/unit/pipeline/test_ambient_guard.py
git commit -m "feat(guard): interpolación lineal del umbral de wake por vad"
```

---

## Task 3: `on_wake` aplica el umbral (shadow + enforce)

**Files:**
- Modify: `src/pipeline/ambient_guard.py` (método `on_wake`, ~líneas 159-193)
- Test: `tests/unit/pipeline/test_ambient_guard.py`

- [ ] **Step 1: Write the failing test**

```python
def make_strict_guard(clock=None, **overrides):
    """Guard ya escalado a STRICT, para probar on_wake con wake_vad."""
    guard = make_guard(clock=clock, **overrides)
    for _ in range(3):  # strict_entry_rejects=3
        guard.on_capture_result("escritorio", "noise")
    assert guard.state_for("escritorio") is GuardState.STRICT
    return guard


class TestVadAdaptiveOnWake:
    def test_enforce_high_vad_accepts_mid_score(self):
        # voz clara: umbral baja a 0.50, score 0.60 pasa
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_hi=0.70)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.80)
        assert d.accept is True

    def test_enforce_low_vad_rejects_mid_score(self):
        # TV/lejos: umbral duro 0.72, score 0.60 NO pasa
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_lo=0.30)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.10)
        assert d.accept is False
        assert d.reason == "strict_score"

    def test_enforce_vad_none_uses_hard(self):
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=None)
        assert d.accept is False  # umbral duro 0.72

    def test_shadow_decides_with_hard_despite_high_vad(self):
        # shadow: aunque vad alto bajaría el umbral, la decisión usa el fijo
        guard = make_strict_guard(strict_vad_adaptive=False, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_hi=0.70)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.80)
        assert d.accept is False  # decide con 0.72, no con 0.50
        assert d.reason == "strict_score"

    def test_wake_vad_ignored_in_normal(self):
        guard = make_guard(strict_vad_adaptive=True)
        d = guard.on_wake("escritorio", score=0.41, rms=0.05, wake_vad=0.10)
        assert d.accept is True  # NORMAL no aplica umbral estricto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestVadAdaptiveOnWake -v`
Expected: FAIL con `TypeError: on_wake() got an unexpected keyword argument 'wake_vad'`

- [ ] **Step 3: Write minimal implementation**

En `src/pipeline/ambient_guard.py`, cambiar la firma de `on_wake` (agregar `wake_vad`):

```python
    def on_wake(
        self,
        room_id: str,
        score: float,
        rms: float,
        spenergy_peak: float | None = None,
        wake_vad: float | None = None,
    ) -> GuardDecision:
```

Reemplazar el bloque STRICT del score (el `if score < self.config.strict_wake_score:` dentro de `if rs.state is GuardState.STRICT:`) por:

```python
            if rs.state is GuardState.STRICT:
                adaptive = self._effective_strict_threshold(wake_vad)
                threshold = (
                    adaptive if self.config.strict_vad_adaptive
                    else self.config.strict_wake_score
                )
                if self.config.strict_vad_adaptive is False and wake_vad is not None:
                    would = "sí" if (score >= adaptive) != (score >= self.config.strict_wake_score) else "no"
                    logger.info(
                        f"[AmbientGuard-vadshadow] room={room_id} wake_vad={wake_vad:.2f} "
                        f"score={score:.2f} umbral_fijo={self.config.strict_wake_score:.2f} "
                        f"umbral_adaptativo={adaptive:.2f} cambiaria={would}"
                    )
                if score < threshold:
                    rs.last_reject_at = now  # ambiente persiste → quiet timer se refresca
                    return GuardDecision(False, "strict_score", rs.state)
                if self.config.strict_min_rms > 0.0 and rms < self.config.strict_min_rms:
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_rms", rs.state)
                if (
                    self.config.strict_min_spenergy > 0.0
                    and spenergy_peak is not None
                    and spenergy_peak < self.config.strict_min_spenergy
                ):
                    rs.last_reject_at = now
                    return GuardDecision(False, "strict_spenergy", rs.state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py -v`
Expected: PASS (toda la clase nueva + las existentes sin regresión)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ambient_guard.py tests/unit/pipeline/test_ambient_guard.py
git commit -m "feat(guard): on_wake aplica umbral adaptativo por vad (shadow loguea)"
```

---

## Task 4: Integración — Silero sobre el audio del wake en el loop

**Files:**
- Modify: `src/pipeline/multi_room_audio_loop.py` (`__init__`, `_should_accept_wakeword:244`, call site `:604`)
- Test: `tests/unit/pipeline/test_ambient_guard.py` (test de wiring del helper)

- [ ] **Step 1: Write the failing test**

El cómputo de `wake_vad` es una función pura testeable. Agregar al final de `tests/unit/pipeline/test_ambient_guard.py`:

```python
import numpy as np
from src.pipeline.multi_room_audio_loop import compute_wake_vad


class TestComputeWakeVad:
    def test_uses_predictor_max_over_chunk(self):
        calls = []
        def fake_predict(mono):
            calls.append(mono)
            return 0.83
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, fake_predict) == 0.83
        assert len(calls) == 1

    def test_none_predictor_returns_none(self):
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, None) is None

    def test_predictor_error_returns_none(self):
        def boom(mono):
            raise RuntimeError("silero down")
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, boom) is None  # fail-safe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestComputeWakeVad -v`
Expected: FAIL con `ImportError: cannot import name 'compute_wake_vad'`

- [ ] **Step 3: Write minimal implementation**

En `src/pipeline/multi_room_audio_loop.py`, agregar una función módulo-nivel (cerca de los imports / antes de la clase):

```python
def compute_wake_vad(audio_chunk, vad_predict) -> float | None:
    """Prob de Silero sobre el audio del wake-trigger (fail-safe → None).

    vad_predict es el predictor de make_silero_predictor (devuelve prob máx
    sobre las ventanas de 512 del chunk). Cualquier fallo → None: el guard
    usa entonces el umbral fijo (nunca más permisivo por falta de señal).
    """
    if vad_predict is None:
        return None
    try:
        import numpy as np
        mono = audio_chunk
        if getattr(mono, "ndim", 1) > 1:
            mono = mono[:, 0]
        return float(vad_predict(np.ascontiguousarray(mono)))
    except Exception:
        return None
```

En `__init__` de la clase del loop, agregar un atributo lazy para el predictor (buscar dónde se guardan deps del guard/ambient; agregar):

```python
        self._wake_vad_predict = None  # lazy: predictor Silero para wake_vad
```

En `_should_accept_wakeword`, cambiar la firma para aceptar `wake_vad`:

```python
    def _should_accept_wakeword(
        self, room_id: str, rms: float, timestamp: float,
        wake_score: float = 1.0, wake_vad: float | None = None,
    ) -> bool:
```

y pasar `wake_vad` al guard:

```python
            decision = self._guard.on_wake(room_id, wake_score, rms, wake_vad=wake_vad)
```

En el call site (`:604`), computar `wake_vad` y pasarlo:

```python
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
                    if self._wake_vad_predict is None:
                        from src.ambient.segmenter import make_silero_predictor
                        try:
                            self._wake_vad_predict = make_silero_predictor()
                        except Exception:
                            self._wake_vad_predict = False  # no reintentar
                    predict = self._wake_vad_predict or None
                    wake_vad = compute_wake_vad(audio_chunk, predict)
                    if self._should_accept_wakeword(
                        rs.room_id, rms, time.time(),
                        wake_score=detection[1], wake_vad=wake_vad,
                    ):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py::TestComputeWakeVad -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full guard + ambient suite (no regressions)**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/pipeline/test_ambient_guard.py tests/unit/ambient/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/multi_room_audio_loop.py tests/unit/pipeline/test_ambient_guard.py
git commit -m "feat(guard): computar wake_vad con Silero y pasarlo a on_wake"
```

---

## Task 5: Config en settings.yaml

**Files:**
- Modify: `config/settings.yaml` (bloque `rooms.ambient_guard`, ~líneas 1205-1220)

- [ ] **Step 1: Agregar los campos (shadow primero)**

En `config/settings.yaml`, dentro de `rooms.ambient_guard`, después de `strict_follow_up_grace_s`:

```yaml
    # Umbral de wake adaptativo por vad en STRICT (spec 2026-06-09). El audio
    # del trigger pasa por Silero; voz clara (vad alto) baja el umbral hacia
    # strict_wake_score_min, TV/lejos (vad bajo) lo mantiene en strict_wake_score.
    # SHADOW primero: loguea [AmbientGuard-vadshadow] qué haría, sin decidir con
    # él. Calibrar lo/hi/min con esos logs (incluir comandos reales) y flipear.
    strict_vad_adaptive: false
    strict_wake_score_min: 0.50
    strict_vad_lo: 0.30
    strict_vad_hi: 0.70
```

- [ ] **Step 2: Verify YAML parsea y el guard lo lee**

Run:
```bash
/Users/yo/Documents/kza/.venv/bin/python -c "
import yaml
c = yaml.safe_load(open('config/settings.yaml'))
ag = c['rooms']['ambient_guard']
assert ag['strict_vad_adaptive'] is False
assert ag['strict_wake_score_min'] == 0.50
print('OK', ag['strict_vad_lo'], ag['strict_vad_hi'])
"
```
Expected: `OK 0.3 0.7`

- [ ] **Step 3: Verificar que el wiring del guard en main.py pasa los campos nuevos**

Buscar dónde se construye `AmbientGuardConfig` en `src/main.py` (grep `AmbientGuardConfig`). Si pasa kwargs explícitos campo por campo, agregar los 4 nuevos leyendo de `ag_cfg`. Si hace `AmbientGuardConfig(**ag_cfg)` o equivalente, no hay cambio. Mostrar el bloque y, si falta, agregar:

```python
                strict_vad_adaptive=ag_cfg.get("strict_vad_adaptive", False),
                strict_wake_score_min=ag_cfg.get("strict_wake_score_min", 0.50),
                strict_vad_lo=ag_cfg.get("strict_vad_lo", 0.30),
                strict_vad_hi=ag_cfg.get("strict_vad_hi", 0.70),
```

- [ ] **Step 4: Commit**

```bash
git add config/settings.yaml src/main.py
git commit -m "feat(guard): wire umbral adaptativo por vad en settings (shadow)"
```

---

## Task 6: Suite completa + deploy

**Files:** ninguno (verificación + deploy)

- [ ] **Step 1: Suite completa local (sin regresiones)**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/ -q --ignore=tests/unit/test_preflight_vram.py --ignore=tests/unit/vectordb`
Expected: mismas 13 fallas pre-existentes del baseline (NO nuevas). Verificar que las nuevas de `test_ambient_guard.py` pasan.

- [ ] **Step 2: Push y deploy al server**

```bash
git push origin feat/nexa-command-detection-fixes
ssh kza 'cd /home/kza/app && git pull --ff-only && systemctl --user restart kza-voice && sleep 30 && systemctl --user is-active kza-voice'
```
Expected: `active`

- [ ] **Step 3: Verificar arranque + que el shadow loguea**

```bash
ssh kza 'journalctl --user -u kza-voice --since "2 minutes ago" --no-pager | grep -iE "ambient path construido|AmbientGuard ACTIVO|error|traceback" | head'
```
Expected: "AmbientGuard ACTIVO", "Ambient path construido", sin tracebacks. El log `[AmbientGuard-vadshadow]` aparecerá cuando haya un wake en STRICT (puede tardar; verificar más tarde con conversación/TV).

- [ ] **Step 4: (más tarde) revisar logs shadow para calibrar**

```bash
ssh kza 'journalctl --user -u kza-voice --since "today" --no-pager | grep "AmbientGuard-vadshadow"'
```
Cruzar `wake_vad` de los comandos reales del usuario (los que deberían pasar) vs los de TV → ajustar `strict_vad_lo`/`strict_vad_hi`/`strict_wake_score_min` y flipear `strict_vad_adaptive: true`.

---

## Self-Review (completado)

- **Cobertura del spec**: señal Silero-en-vivo (Task 4), interpolación parcial 0.72→0.50 (Task 2), shadow→enforce (Task 3 + config Task 5), fail-safe vad=None (Tasks 2,3,4), NORMAL/COOLDOWN sin cambios (Task 3 test), limitación TV-diálogo mitigada por escalado parcial (config). ✓
- **Placeholders**: ninguno — todo el código está escrito. Task 5 Step 3 es condicional (depende de cómo main.py construye el config) pero da el código exacto a agregar y cómo decidir. ✓
- **Consistencia de tipos**: `wake_vad: float | None` en compute_wake_vad/on_wake/_effective_strict_threshold; `strict_vad_adaptive/strict_wake_score_min/strict_vad_lo/strict_vad_hi` idénticos en config, tests, settings, main. `_should_accept_wakeword` firma extendida con default. ✓
