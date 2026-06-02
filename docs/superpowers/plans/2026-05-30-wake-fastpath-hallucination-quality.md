# Wake / Fast-Path Hallucination & Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠️ Entorno de ejecución:** este trabajo se desarrolla **en vivo en el server** (`kza@192.168.1.2`, `~/app`), NO en la laptop. El repo de la laptop está drifteado. Antes de cada cambio comparar `git hash-object` por archivo (ver `feedback_drift_compare_hashes`). VRAM en cuda:1 apretadísima (~850-930MB libres con kza-voice arriba) — correr preflight de VRAM antes de cada reinicio del servicio (ver `project_stt_double_load_oom`).

**Goal:** Eliminar las activaciones espurias de TV-mode (causadas por alucinaciones de Whisper sobre silencio), recuperar comandos reales perdidos, e instrumentar el wake detector para calibrar con datos en vez de a ojo — habilitando además el uso del array/beamforming/DOA del XVF3800 reubicado en el escritorio.

**Architecture:** Defensa en capas sobre `src/wakeword/whisper_wake.py`. (1) Las alucinaciones dejan de alimentar el contador de TV-mode. (2) Se capturan señales acústicas (`no_speech_prob`/`avg_logprob`/`compression_ratio`) que hoy se descartan, para clasificar alucinación-vs-voz con datos. (3) Denylist normalizada como red final barata. (4) Recuperación del verbo comido. (5) Spike de beamforming/DOA del XVF3800. Cada fase es un deploy independiente y testeable.

**Tech Stack:** Python 3.13, faster-whisper (Silero VAD interno disponible), pytest + fixtures, systemd `--user` (`kza-voice.service`), métricas JSONL (`src/obs/metrics_emitter.py` → `~/logs/kza-metrics.jsonl`).

---

## File Structure

| Archivo | Responsabilidad | Fases que lo tocan |
|---------|-----------------|--------------------|
| `config/settings.yaml` | Binding USB del mic + thresholds wake configurables | 0, 1, 2 |
| `src/rooms/room_context.py` | Resolución `mic_usb_port` → device index | 0 (solo verificación) |
| `src/wakeword/whisper_wake.py` | Detección wake, TV-mode, denylist, captura de confianza, verbo | 1, 2, 3, 4 |
| `src/obs/metrics_emitter.py` | `emit_wake` — agregar campos de confianza | 1 |
| `src/orchestrator/dispatcher.py` | Guard domain-conflict (vector path) | 6 (verificación) |
| `tests/unit/wakeword/test_whisper_wake.py` | Tests del wake detector | 1, 2, 4 |

---

## Fase 0 — Re-binding del mic tras el swap físico (BLOQUEANTE, AHORA)

> Al mover el XVF3800 del living al escritorio cambia el **puerto USB físico**. El binding mic→habitación es por `mic_usb_port` (config/settings.yaml). Si no se reconfigura, el escritorio queda mapeado al device equivocado o sin mic. Valores actuales: living `1-2.2`, escritorio `3-1.1` (memoria `project_living_mic_usb_binding`).

### Task 0.1: Detectar el nuevo puerto USB del XVF3800 en el escritorio

**Files:**
- Read-only: ejecución de tool

- [ ] **Step 1: Enumerar dispositivos USB de audio tras enchufar el XVF3800 en el escritorio**

Run (en el server):
```bash
ssh kza 'cd ~/app && python -m src.rooms.room_context --detect'
```
Expected: lista de mics con su `usb_port` y `device_index`. Identificar la línea del **XVF3800** (ReSpeaker, multi-canal) y anotar su `usb_port` (ej. `1-1.3`) y `device_index`.

- [ ] **Step 2: Confirmar el número de canales del XVF3800 (insumo para Fase 5)**

Run:
```bash
ssh kza 'python3 -c "import sounddevice as sd; [print(i, d[\"name\"], d[\"max_input_channels\"]) for i,d in enumerate(sd.query_devices()) if d[\"max_input_channels\"]>0]"'
```
Expected: el XVF3800 reporta >1 canal de entrada (array). Anotar el count (típico: 1 procesado + N raw, según firmware).

### Task 0.2: Actualizar el binding en settings.yaml

**Files:**
- Modify: `config/settings.yaml:1157` (bloque `escritorio`, `mic_usb_port`)

- [ ] **Step 1: Cambiar `mic_usb_port` del escritorio al puerto detectado**

En `config/settings.yaml`, en el bloque de la habitación `escritorio` (línea ~1157), reemplazar el valor por el `usb_port` detectado en Task 0.1:
```yaml
    mic_usb_port: "<PUERTO_DETECTADO>"   # XVF3800 reubicado 2026-05-30 (era 3-1.1, UAC1.0)
    mic_device_index: null   # lo puebla room_context desde mic_usb_port al iniciar
```

- [ ] **Step 2: Si el mic del living queda vacío o se reasigna, actualizar/comentar su binding**

Si el living queda sin mic temporalmente, comentar su `mic_usb_port` (línea ~1099) o ponerlo en `null` para que no intente resolver un device inexistente y loguee warning en loop.

### Task 0.3: Reiniciar y verificar (con preflight de VRAM)

- [ ] **Step 1: Preflight de VRAM antes de reiniciar**

Run:
```bash
ssh kza 'nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 1'
```
Expected: si <1500MB libres en cuda:1, el preflight del propio servicio avisará. Proceder solo si hay margen (ver `project_stt_double_load_oom`).

- [ ] **Step 2: Reiniciar el servicio**

Run:
```bash
ssh kza 'systemctl --user restart kza-voice.service && sleep 8 && systemctl --user is-active kza-voice.service'
```
Expected: `active`.

- [ ] **Step 3: Verificar que el escritorio resolvió el device correcto y el stream abre**

Run:
```bash
ssh kza 'journalctl --user -u kza-voice.service --since "1 min ago" --no-pager | grep -iE "escritorio|Mic USB port|stream|device index"'
```
Expected: log `Mic USB port <puerto> → device index N` para `escritorio`, y stream abierto sin error. **OJO no cruzar** living/escritorio.

- [ ] **Step 4: Smoke test de wake en el escritorio**

Decir en el escritorio: "Nexa, prendé la luz del escritorio" y verificar en `journalctl -f` que el wake se acepta y llega a `fast_domotics`.

---

## Fase 1 — Frenar TV-mode espurio + instrumentar confianza (1 deploy, riesgo bajo)

> Evidencia (870h de métricas): TV-mode se activó ~716 veces, alimentado por `_record_reject` que cuenta **todos** los rejects sin filtrar — incluyendo 7.255 `tv_stop_phrase` (alucinaciones YouTube) y 42.730 `below_fuzzy_threshold` (dominado por "¡Gracias!" ×12.440). El umbral es 4 rejects/300s (`TV_MODE_ENTRY_REJECTS`). Objetivo: que las alucinaciones conocidas NO cuenten, y capturar las señales de confianza que hoy se tiran para calibrar el resto.

### Task 1.1: Excluir reasons de alucinación del contador de TV-mode

**Files:**
- Modify: `src/wakeword/whisper_wake.py` (`_record_reject`, ~994; constante nueva cerca de `_TV_STOP_PHRASES` ~74)
- Test: `tests/unit/wakeword/test_whisper_wake.py`

- [ ] **Step 1: Escribir el test que falla**

```python
def test_record_reject_ignores_hallucination_reasons(wake_detector):
    """Las reasons de alucinación NO deben acumular hacia TV-mode."""
    det = wake_detector
    # 10 rejects de alucinación conocida no deben activar TV-mode
    for _ in range(10):
        det._record_reject("tv_stop_phrase")
    assert not det._is_tv_mode_active()
    for _ in range(10):
        det._record_reject("multi_wake_hallucination")
    assert not det._is_tv_mode_active()

def test_record_reject_counts_real_audio_reasons(wake_detector):
    """Reasons de audio real (no_command_verb) sí cuentan hacia TV-mode."""
    det = wake_detector
    for _ in range(4):  # == TV_MODE_ENTRY_REJECTS
        det._record_reject("no_command_verb")
    assert det._is_tv_mode_active()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/test_whisper_wake.py -k record_reject -v'`
Expected: FAIL (hoy `tv_stop_phrase` sí activa TV-mode).

- [ ] **Step 3: Implementar el filtro de reasons**

Cerca de `_TV_STOP_PHRASES` (~línea 90) agregar la constante:
```python
# Reasons de reject que NO son evidencia de un entorno "TV-like" real, sino
# alucinaciones de Whisper sobre silencio/ruido. NO deben alimentar TV-mode
# (evidencia métricas 2026-05-30: estas reasons disparaban TV-mode espurio).
_NON_TV_MODE_REJECT_REASONS = frozenset({
    "tv_stop_phrase",            # denylist de alucinaciones YouTube conocidas
    "multi_wake_hallucination",  # loop alucinatorio sobre wake word
    "pathological_repeat",       # loop repetitivo
    "implausible_speech_rate",   # >N palabras/seg físicamente imposible
    "no_speech_hallucination",   # (Fase 3) low-confidence acústica
})
```

En `_record_reject` (~994), guardar al inicio:
```python
    def _record_reject(self, reason: str) -> None:
        """Registra un reject y activa/extiende TV-mode si supera el threshold.

        Las reasons en _NON_TV_MODE_REJECT_REASONS se ignoran: son
        alucinaciones de Whisper, no evidencia de un entorno ruidoso real.
        """
        # Las alucinaciones conocidas no cuentan como "ambiente TV".
        if reason in _NON_TV_MODE_REJECT_REASONS:
            return
        now = time.time()
        self._reject_timestamps.append(now)
        # ... (resto sin cambios)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/test_whisper_wake.py -k record_reject -v'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ssh kza 'cd ~/app && git add src/wakeword/whisper_wake.py tests/unit/wakeword/test_whisper_wake.py && git commit -m "fix(wake): alucinaciones no alimentan TV-mode (era activación espuria)"'
```

### Task 1.2: Capturar señales de confianza acústica en el wake detector

**Files:**
- Modify: `src/wakeword/whisper_wake.py` (`_transcribe_and_match`, ~745; todas las llamadas a `_emit_wake`)
- Modify: `src/obs/metrics_emitter.py` (`emit_wake`, 171)
- Test: `tests/unit/wakeword/test_whisper_wake.py`

- [ ] **Step 1: Test que falla — la confianza se extrae de los segments**

```python
def test_transcribe_captures_confidence(wake_detector, monkeypatch):
    """no_speech_prob/avg_logprob se extraen y propagan al emit_wake."""
    class FakeSeg:
        def __init__(self, text, nsp, lp):
            self.text, self.no_speech_prob, self.avg_logprob = text, nsp, lp
    segs = [FakeSeg("gracias", 0.92, -0.3)]
    monkeypatch.setattr(wake_detector.whisper, "transcribe",
                        lambda *a, **k: (iter(segs), None))
    captured = {}
    monkeypatch.setattr(wake_detector, "_emit_wake",
                        lambda *a, **k: captured.update(k))
    wake_detector._transcribe_and_match(_silence_audio(), 800.0)
    assert captured.get("no_speech_prob") == 0.92
    assert captured.get("avg_logprob") == -0.3
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/test_whisper_wake.py -k captures_confidence -v'`
Expected: FAIL (`_emit_wake` no recibe `no_speech_prob`).

- [ ] **Step 3: Extraer confianza en `_transcribe_and_match`**

En `_transcribe_and_match` (~755), reemplazar la materialización de segments:
```python
            seg_list = list(segments)
            text = " ".join(s.text for s in seg_list).strip()
            # Señales acústicas para clasificar alucinación-vs-voz (Fase 1
            # instrumentación; Fase 3 las usa para gatear). faster-whisper
            # expone no_speech_prob/avg_logprob por segment.
            no_speech_prob = (max((s.no_speech_prob for s in seg_list), default=None)
                              if seg_list else None)
            avg_logprob = (min((s.avg_logprob for s in seg_list), default=None)
                           if seg_list else None)
```
Loguear junto a la línea existente `WhisperWake [...]`:
```python
        logger.info(
            f"WhisperWake [{dur_ms:.0f}ms→{stt_ms:.0f}ms]: {norm!r} "
            f"(no_speech={no_speech_prob if no_speech_prob is None else round(no_speech_prob,2)}, "
            f"avg_logprob={avg_logprob if avg_logprob is None else round(avg_logprob,2)})"
        )
```
Propagar `no_speech_prob`/`avg_logprob` a **todas** las llamadas `self._emit_wake(...)` de este método (agregar como kwargs).

- [ ] **Step 4: Extender `_emit_wake` y `emit_wake` para aceptar/escribir los campos**

En `whisper_wake.py` `_emit_wake` (~1035), agregar params:
```python
        no_speech_prob: Optional[float] = None,
        avg_logprob: Optional[float] = None,
```
y pasarlos al `self.metrics_emitter.emit_wake(...)`.

En `src/obs/metrics_emitter.py` `emit_wake` (171), agregar params keyword-only y al doc:
```python
        no_speech_prob: float | None = None,
        avg_logprob: float | None = None,
```
```python
        if no_speech_prob is not None:
            doc["no_speech_prob"] = float(no_speech_prob)
        if avg_logprob is not None:
            doc["avg_logprob"] = float(avg_logprob)
```
Mantener el `except TypeError` de backward-compat ya existente para emitters viejos.

- [ ] **Step 5: Correr y verificar que pasa + no rompe el resto**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/ -v && pytest tests/ -k metrics -v'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ssh kza 'cd ~/app && git add src/wakeword/whisper_wake.py src/obs/metrics_emitter.py tests/unit/wakeword/test_whisper_wake.py && git commit -m "feat(wake): captura no_speech_prob/avg_logprob para calibrar anti-alucinación"'
```

### Task 1.3: Deploy + verificación en vivo

- [ ] **Step 1: Preflight VRAM + restart** (idéntico a Task 0.3 Steps 1-2).

- [ ] **Step 2: Confirmar que las nuevas señales aparecen en métricas**

Run (tras ~2 min de operación):
```bash
ssh kza 'tail -50 ~/logs/kza-metrics.jsonl | grep -o "no_speech_prob[^,]*" | head'
```
Expected: líneas wake con `no_speech_prob`.

- [ ] **Step 3: Confirmar caída de activaciones de TV-mode**

Run:
```bash
ssh kza 'journalctl --user -u kza-voice.service --since "10 min ago" | grep "TV-mode ACTIVATED" | wc -l'
```
Expected: cerca de 0 con ambiente normal (antes: múltiples por hora).

---

## Fase 2 — Denylist normalizada como red final (barata, data-backed)

> Las alucinaciones se concentran en poquísimas frases (3 frases = 7.000 `tv_stop_phrase`; "¡Gracias!" sola = 12.440 `below_fuzzy`). Una denylist normalizada de "utterance completa = alucinación" cubre >80% del volumen. **No** es la cura (frágil por idioma/versión) pero es red barata. Ref: dataset comunitario `sachaarbonel/whisper-hallucinations` (HuggingFace, subset `lang=es`).

### Task 2.1: Denylist de alucinaciones de utterance-completa (no substring)

**Files:**
- Modify: `src/wakeword/whisper_wake.py` (constante nueva + chequeo en `_transcribe_and_match` antes de las heurísticas)
- Test: `tests/unit/wakeword/test_whisper_wake.py`

- [ ] **Step 1: Test que falla**

```python
import pytest

@pytest.mark.parametrize("text", [
    "¡Gracias!", "Gracias.", "Amén.", "Adiós.", "¿Verdad?",
])
def test_full_utterance_hallucinations_rejected_as_such(wake_detector, text):
    """Alucinaciones de saludo/cierre deben rechazarse SIN contar TV-mode."""
    reason = wake_detector._full_utterance_hallucination_reason(_normalize(text))
    assert reason == "no_speech_hallucination"

def test_real_command_with_gracias_not_rejected(wake_detector):
    """'gracias' DENTRO de un comando real no debe matchear (no es utterance completa)."""
    assert wake_detector._full_utterance_hallucination_reason(
        _normalize("nexa prendé la luz gracias")) is None
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/test_whisper_wake.py -k hallucination -v'`
Expected: FAIL (`_full_utterance_hallucination_reason` no existe).

- [ ] **Step 3: Implementar la denylist de utterance-completa**

Cerca de `_TV_STOP_PHRASES` agregar (normalizado, sin acentos/puntuación, lowercase — el match es contra `norm`):
```python
# Alucinaciones de Whisper sobre silencio que son frases CORTAS y COMPLETAS
# (no substrings): saludos/cierres que el modelo inventa. Se matchean solo
# cuando son la utterance entera, para no false-rejectar "...gracias" real.
# Evidencia métricas 2026-05-30 (top below_fuzzy_threshold).
_FULL_UTTERANCE_HALLUCINATIONS = frozenset({
    "gracias", "muchas gracias", "amen", "adios", "verdad", "vamos",
    "gracias por ver el video", "suscribete al canal", "suscribete",
})
```
Método nuevo:
```python
    def _full_utterance_hallucination_reason(self, norm: str) -> Optional[str]:
        """Si la utterance COMPLETA es una alucinación conocida, devuelve la
        reason; None si no. Match exacto contra norm (no substring)."""
        if norm in _FULL_UTTERANCE_HALLUCINATIONS:
            return "no_speech_hallucination"
        return None
```
En `_transcribe_and_match`, tras calcular `norm` y ANTES de las heurísticas de TV/verbo, agregar:
```python
        hallu = self._full_utterance_hallucination_reason(norm)
        if hallu:
            logger.info(f"Wake rejected — alucinación de utterance completa: {text!r}")
            self._emit_wake(False, None, "rejected", text, dur_ms, stt_ms,
                            rejection_reason=hallu,
                            no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)
            return (None, text)
```
> Nota: `no_speech_hallucination` ya está en `_NON_TV_MODE_REJECT_REASONS` (Task 1.1), así que NO alimenta TV-mode. ✅

- [ ] **Step 4: Correr y verificar que pasa**

Run: `ssh kza 'cd ~/app && pytest tests/unit/wakeword/test_whisper_wake.py -k hallucination -v'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ssh kza 'cd ~/app && git add src/wakeword/whisper_wake.py tests/unit/wakeword/test_whisper_wake.py && git commit -m "feat(wake): denylist de alucinaciones de utterance-completa (no cuenta TV-mode)"'
```

---

## Fase 3 — Gate por confianza acústica (DECISION GATE, requiere ≥1 día de datos de Fase 1)

> **No implementar sin datos.** La investigación advierte que el AND interno de faster-whisper (`no_speech_prob>0.6 AND avg_logprob<-1.0`) NO atrapa alucinaciones fluidas (avg_logprob alto). Hay que medir si un umbral **disyuntivo** separa alucinación de voz real en TU mic/entorno.

### Task 3.1: Análisis de separación (decision gate)

- [ ] **Step 1: Correr la query de separación sobre los datos nuevos**

```bash
ssh kza 'python3 - << "PY"
import json, collections
H = {"gracias","muchas gracias","amen","adios","verdad","gracias por ver el video","suscribete al canal","suscribete"}
buckets = {"hallu": [], "accepted": []}
with open("/home/kza/logs/kza-metrics.jsonl") as f:
    for line in f:
        try: d = json.loads(line)
        except: continue
        if d.get("event_type") != "wake" or d.get("no_speech_prob") is None: continue
        nsp, lp = d["no_speech_prob"], d.get("avg_logprob")
        if d.get("matched"): buckets["accepted"].append((nsp, lp))
        elif (d.get("text","").strip().lower().strip("¡!¿?.,")) in H: buckets["hallu"].append((nsp, lp))
for k, v in buckets.items():
    if not v: print(k, "sin datos aún"); continue
    nsps = sorted(x[0] for x in v)
    print(f"{k}: n={len(v)} no_speech_prob p10={nsps[len(nsps)//10]:.2f} median={nsps[len(nsps)//2]:.2f} p90={nsps[len(nsps)*9//10]:.2f}")
PY'
```
Expected: distribuciones de `no_speech_prob` para alucinaciones vs wakes aceptados.

- [ ] **Step 2: DECISIÓN**

- **Si hay separación clara** (ej. alucinaciones con `no_speech_prob` mayoritariamente >X y aceptados <X, con solapamiento <5%): implementar el gate disyuntivo (Task 3.2) con `no_speech_threshold = X` (config nuevo `wake_word.no_speech_reject_threshold`, default = valor medido).
- **Si NO hay separación** (alucinaciones fluidas con no_speech_prob bajo): saltar a Fase 5 (VAD/DOA del XVF3800) como vía robusta; el gate de confianza no alcanza.

### Task 3.2: (Condicional) Implementar gate disyuntivo

> Solo si Task 3.1 dio separación. Mismo patrón TDD que Task 2.1: test con un seg de `no_speech_prob` alto → reject `no_speech_hallucination`; test con voz real (no_speech_prob bajo) → no reject. El reject usa la reason `no_speech_hallucination` (ya excluida de TV-mode). Umbral desde config, default = valor de Task 3.1. Implementación, test, commit.

---

## Fase 4 — Recuperar el verbo comido (`no_command_verb` false-rejects)

> Evidencia: comandos reales rechazados como `no_command_verb` cuando Whisper se come el verbo y loopea el sustantivo: `'Nexa, la luz del escritorio, la luz del escritorio...'`. El usuario dijo "Nexa **prendé** la luz del escritorio". Hay wake + entity/room conocida pero sin verbo → rechazado → sin acción.

### Task 4.1: Spike — caracterizar el fallo

- [ ] **Step 1: Extraer todos los `no_command_verb` con wake presente + entity/room conocida**

```bash
ssh kza 'python3 - << "PY"
import json
with open("/home/kza/logs/kza-metrics.jsonl") as f:
    for line in f:
        try: d=json.loads(line)
        except: continue
        if d.get("rejection_reason")=="no_command_verb" and "escritorio" in (d.get("text","") or "").lower():
            print(d.get("@timestamp","")[:19], repr(d.get("text","")))
PY'
```
Expected: lista de comandos reales perdidos. Cuantificar el volumen (¿vale la pena?).

- [ ] **Step 2: DECISIÓN de diseño** (presentar al usuario; hay ambigüedad real)

Opciones:
- (a) **De-loop**: detectar el patrón "X, X, X" (sustantivo repetido sin verbo) y, si hay wake + entity/room conocida, armar follow-up window agresiva.
- (b) **Verbo por defecto contextual**: si hay wake + entity de tipo `light` sin verbo, asumir `toggle`. ⚠️ riesgo de acción no deseada — discutir.
- (c) Subir `beam_size`/initial_prompt solo para recuperar el verbo (costo latencia/VRAM).

No implementar sin elegir. La opción (a) es la más conservadora.

---

## Fase 5 — Spike: beamforming / DOA del XVF3800 (hardware, exploratorio)

> El XVF3800 reubicado en el escritorio tiene array de mics + beamforming + DOA on-device. Dos oportunidades: (1) audio beamformed = mejor SNR → menos alucinaciones en origen; (2) DOA (dirección de arribo) → rechazar audio que viene de la dirección de la TV (filtrado espacial). Esto es research, no código pre-especificable.

### Task 5.1: Inventariar canales y DOA disponibles

- [ ] **Step 1: Documentar los canales del XVF3800** (del count de Task 0.1 Step 2). Identificar cuál es el canal procesado (beamformed) y si hay canales raw por mic.

- [ ] **Step 2: Investigar exposición de DOA**

El XVF3800 expone DOA vía control interface (USB HID/I2C según firmware/dev-board). Investigar:
```bash
ssh kza 'lsusb -v 2>/dev/null | grep -iA5 "XMOS\|XVF\|respeaker" | head -40'
```
y revisar si hay un driver/tool de control (`usb_4_mic_array`, `xvf_host`, o similar) ya presente. Documentar hallazgos. Si DOA no es accesible sin hardware/firmware extra, marcarlo y cerrar el spike en (1).

### Task 5.2: DECISIÓN — qué señal del array integrar

- [ ] Presentar al usuario: (a) usar solo el canal beamformed como input al wake (cambio mínimo en captura), (b) además leer DOA y agregar un gate espacial "rechazar wake si DOA ≈ dirección TV" (requiere calibrar la dirección de la TV), (c) usar VAD multi-canal. Elegir antes de escribir código. Cada opción sería su propio plan/fase.

---

## Fase 6 — Verificación: guard de domain-conflict cubre los misroutes

> Métricas mostraron `'Nexa bajá la temperatura del aire.'` → `light.escritorio` success=True (2026-05-29 17:51). El guard `_conflicting_domain` (`dispatcher.py:120`) ya cubre temperatura→climate y se deployó en `73a9f9b` ese mismo día. Probablemente esos eventos son **previos** al fix.

### Task 6.1: Confirmar que el guard atrapa el caso

- [ ] **Step 1: Test unitario del guard con el texto real**

```bash
ssh kza 'cd ~/app && python3 -c "
from src.orchestrator.dispatcher import _conflicting_domain
print(_conflicting_domain(\"baja la temperatura del aire\", \"light\"))  # esperado: climate
print(_conflicting_domain(\"baja la luz del escritorio\", \"light\"))     # esperado: None
"'
```
Expected: `climate` y `None` respectivamente.

- [ ] **Step 2:** Si el primero devuelve `None` (el guard NO cubre "temperatura del aire"), agregar el sustantivo faltante a `_NON_LIGHT_DOMAIN_NOUNS` con test TDD. Si devuelve `climate`, cerrar como verificado (los misroutes eran pre-fix).

---

## Self-Review

- **Spec coverage:** Las 7 recomendaciones de la sesión mapean a fases: TV-mode decouple→F1.1, instrumentar→F1.2, denylist→F2, gate confianza→F3, verbo comido→F4, beamforming/DOA→F5, domain-conflict→F6, mic swap→F0. ✅
- **Dependencias:** F0 bloqueante (mic físico). F1 habilita F3 (datos). F3/F4/F5 son decision gates — no se implementan a ciegas. ✅
- **Consistencia de tipos:** `_NON_TV_MODE_REJECT_REASONS` (F1.1) incluye `no_speech_hallucination` que usan F2 y F3.2 → coherente. `no_speech_prob`/`avg_logprob` mismos nombres en `_transcribe_and_match`, `_emit_wake`, `emit_wake`. ✅
- **Decision gates explícitos** (F3.1, F4.2, F5.2, F6.2) NO son placeholders: tienen query/comando concreto + regla de decisión. Lo que depende de datos/hardware no se inventa.
