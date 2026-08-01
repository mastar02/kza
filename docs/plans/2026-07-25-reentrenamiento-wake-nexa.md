# Re-entrenamiento del wake acústico `nexa.onnx`

**Fecha:** 2026-07-25
**Estado:** plan — ⚠️ **PREMISA INVALIDADA el mismo día, ver §0**

## 0. ⚠️ El wake NO está sordo — medición del 2026-07-25 18:19

Este plan se escribió sobre la premisa "el wake acústico está prácticamente
sordo, 0.132 máx contra umbral 0.4". **Los datos de producción la
contradicen.** Disparos acústicos reales en 24h:

| habitación | disparos | score medio | max |
|---|---|---|---|
| escritorio | 36 | **0.54** | 0.87 |
| cocina | 50 | **0.60** | 0.94 |

86 disparos con media muy por encima del umbral 0.4, y el mic **nuevo**
(cocina) rinde *mejor* que el viejo. En una secuencia de comandos reales
medida a las 18:19:43-47 el wake scoreó **0.871 / 0.930 / 0.916 / 0.821**.

Los scores de 0.001-0.204 que parecían probar la sordera eran de
**conversación ambiente e idle**, no de intentos de comando. El
`[oww-dbg]` loguea las dos cosas sin distinguirlas.

**Conclusión: re-entrenar el wake NO es la prioridad.** El costo real de
latencia y de comandos perdidos viene de otro lado (§0.2). Lo que sigue
(§1-§5) queda como referencia válida sobre el estado del dataset, para
cuando el re-entrenamiento sí tenga sentido — y el hallazgo de §1 (el
dataset no contiene falsos negativos) sigue siendo cierto e importante.

### 0.2 Las causas reales, medidas

**Comandos perdidos: la regla `[STT-veto]`** (`command_processor.py:257`):

```python
if text.strip() and not r.text.strip():   # parakeet vacío
    return ""                              # descarta la captura, en silencio
```

A las 18:19:43 el wake disparó a 0.87, Whisper transcribió `'De la luz.'`
(comando real, parcial) y Parakeet devolvió vacío → **vetado → `Text=''` →
silencio total**. El intento siguiente pasó sólo porque Parakeet devolvió
*gibberish en francés* (`'Ça, la belle à douce?'`) en vez de vacío: **el
veto chequea vacuidad, no acuerdo.** 62 vetos en 24h; la mayoría son
alucinaciones legítimas (`'¡Gracias!'`), pero atrapa comandos reales.

**La red de seguridad se baja justo cuando hace falta.** El wake textual
había reconocido `'Next up, prende la luz.'` pero lo descartó con
`decision=dedup_acoustic` (`textual_wake.py:297`, ventana 8s) porque el
acústico había disparado 0.3s antes. `last_command_dispatch_ts` se setea
al **disparar el wake**, no al producir un comando → una captura vetada o
vacía **suprime el canal textual 8 segundos**. Ahí está el "al tercer
intento recién agarró".

**Latencia: el LLMRouter, no el wake textual.** El comando que sí entró
midió `[SLOW] Total: 1340ms` con **`llm_router=716ms`** (24h: n=2,
media 1070ms, max 1424ms). Cae al SLOW path porque el STT destroza "Nexa"
(`'Next up'`, `'¡Pensá'`) y el fast path por gramática/vectorial no lo
reclama. Arreglar el wake acústico no toca nada de esto.

---

## Contexto original (mantenido como referencia del dataset)

**Síntoma reportado:** el wake acústico scorearía 0.132 máx contra umbral
0.4 mientras `wake_vad` marca 0.94-1.00. Marker:
`[oww-dbg] ... max=nexa:0.008 thr=0.4` (`src/wakeword/detector.py:233`).
**Ver §0: no se sostiene.**

---

## 1. Hallazgo que cambia el plan: el dataset NO puede arreglar la sordera

La premisa de entrada era "el dataset del WakeClipWriter ya está juntado".
El dataset existe y es grande, pero **no contiene el modo de falla**.

Distribución real de scores en `/home/kza/app/data/wakeword_training/captured`:

| bucket | aceptados (raíz) | rechazados (`rejected/`) |
|---|---|---|
| 0.4 | 272 | 1838 |
| 0.5 | 753 | 822 |
| 0.6 | 470 | 535 |
| 0.7 | 291 | 373 |
| 0.8 | 162 | 313 |
| 0.9 | 52 | — |
| **total** | **2000** | **3881** |

**El bucket mínimo es 0.4 en las dos carpetas. No hay un solo clip por
debajo del umbral.** Causa: `WakeClipWriter.submit()` sólo se invoca desde
el camino del wake acústico (`multi_room_audio_loop.py:869,927`), o sea
únicamente cuando el wake **ya disparó**. Los "nexa" reales que scorearon
0.132 nunca se escribieron a disco.

Entrenar con esto afila la frontera en la región ≥0.4 y deja sin modelar
la región 0.1-0.4, que es justo donde está cayendo la voz real. **No
arregla la sordera.**

### 1.2 Los negativos están contaminados

`rejected/` no es un set limpio de negativos. El docstring de
`WakeClipWriter` lo dice explícitamente: *"dataset de hard-negatives +
positivos far-field que STRICT mató"*. Usar los 3881 como negativos le
enseña al modelo a rechazar habla far-field real — **profundiza la
sordera** en vez de curarla.

### 1.3 Sesgo de habitación

`5832 escritorio` vs **`49 cocina`**. El mic de cocina se dio de alta el
2026-07-24, así que no tiene datos. El dataset es monoambiente; el pedido
es que aplique a los dos micros.

### 1.4 Ventana rotativa, no archivo histórico

`max_files=2000` / `max_rejected_files=4000` y las dos carpetas están **en
el tope** (2000 / 3881) → los clips viejos se están purgando. Además la
cadena acústica cambió tres veces dentro de la ventana:

| fecha | cambio | efecto |
|---|---|---|
| 2026-07-02 | mic fuera del cielo raso | acústica distinta |
| 2026-07-05 | AGC 8 → 16 | ganancia distinta |
| 2026-07-16 | XVF `ATTNS_MODE 1` + `MIN_NN 0.4` | post-proceso distinto |

Sólo los clips **desde 07-16** representan la cadena actual: ≈1424 clips
(07-17 a 07-25). El resto es de otro micrófono, en otra posición, con otra
ganancia.

---

## 2. Etapa A — instrumentar la captura de los fallos (bloqueante)

Sin ejemplos etiquetados de los "misses" no hay nada que entrenar. Hay un
oráculo gratis ya corriendo: **el wake textual**.

`src/ambient/textual_wake.py` ya recibe `last_acoustic_command_ts_fn` y
decide `decision=dedup_acoustic` (:296-300) cuando el camino acústico
disparó hace poco. Entonces:

> **wake textual dispara** ∧ **no hubo dedup acústico** ⇒ el wake acústico
> falló ese "nexa". Falso negativo, etiquetado, con el audio en la mano
> (`CommandEvent.audio`).

**Cambios:**

1. `WakeClipWriter`: tercer bucket `missed/` con su propio tope de
   rotación (`max_missed_files`, sugerido 2000), simétrico a `rejected/`.
2. `textual_wake.py`: en el disparo no-deduplicado, `submit(room_id,
   score=<score acústico real de la ventana>, audio, bucket="missed")`.
   Hoy construye `CommandEvent(..., wake_score=1.0)` hardcodeado (:312) —
   hay que pasarle el score acústico verdadero, si no el nombre del
   archivo miente.
3. Correr **≥7 días** en las dos habitaciones para juntar misses reales.
   Con la tasa de uso actual esto debería dar unos cuantos cientos.

**Sin la Etapa A, el re-entrenamiento es a ciegas.** Es la diferencia
entre "afinar el modelo que ya tenemos" y "arreglar el problema".

---

## 3. Etapa B — armado del dataset

| Clase | Fuente | Notas |
|---|---|---|
| **Positivos reales** | `captured/` raíz, **sólo ≥ 2026-07-16** | ≈700-900 clips de la cadena actual |
| **Positivos duros** | `missed/` (Etapa A) | **los más valiosos** — el modo de falla |
| **Positivos curados** | `nexa/positive` (de 130 wav) | grabados de cerca, pre-relocalización |
| **Positivos sintéticos** | piper-sample-generator, voces ES | ya soportado por `scripts/train_custom_wake.py` |
| **Negativos duros** | `rejected/` **triado a mano** | ⚠️ hay que separar TV/ruido de far-field real |
| **Negativos generales** | MUSAN (speech/music/noise) | ya en el pipeline |

**El triaje de `rejected/` es trabajo manual y no es opcional** (§1.2).
Atajo razonable: escuchar sólo una muestra estratificada por bucket de
score (p.ej. 60 por bucket ≈ 300 clips) para estimar qué fracción es
far-field real; si es baja, se puede usar el resto como negativos con
peso reducido. Si es alta, hay que triar todo.

**Balance de habitaciones:** con 49 clips de cocina no se puede entrenar
per-room. El modelo tiene que ser único y robusto a las dos acústicas →
augmentación con reverb/EQ que cubra ambas, y **holdout separado por
habitación** para medir cada una (§4).

---

## 4. Etapa C — entrenamiento y validación con holdout

Pipeline existente: `scripts/train_custom_wake.py`
(`generate` → `features` → `train` → `export`). Requiere cuda:0 libre:

```bash
systemctl --user stop kza-voice.service
python -m scripts.train_custom_wake all
systemctl --user start kza-voice.service
```

⚠️ Parar kza-voice deja el hogar sin voz mientras entrena — coordinar
horario.

### Diseño del holdout

**Split por sesión temporal, no aleatorio.** Un split aleatorio pone
clips de la misma pronunciación en train y test (están a milisegundos uno
del otro) e infla el recall. Reservar los **últimos 3 días completos**
como test, entrenar con lo anterior.

Estratificar el holdout en cuatro celdas y reportar cada una por separado:

| celda | qué mide |
|---|---|
| positivos escritorio | recall en la habitación con datos |
| positivos cocina | recall en la habitación sin datos (el riesgo real) |
| `missed/` retenidos | **si la sordera se curó** — la métrica que importa |
| negativos (TV/ruido) | que no se compró recall a cambio de falsos positivos |

### Criterios de aceptación

| métrica | umbral | por qué |
|---|---|---|
| Recall en `missed/` holdout | **≥ 80% @ thr 0.4** | es el bug que estamos arreglando |
| Recall positivos, ambas rooms | ≥ 90% | no regresionar lo que ya funciona |
| Falsos positivos sobre TV/ruido | **≤ el actual** | la compuerta acústica de 06-12 se ganó con esfuerzo; no rifarla |
| Score mediano de positivos | ≥ 0.7 | margen sobre 0.4, no apenas pasando |

Si sube el recall pero también los falsos positivos, **no se deploya**:
ese trade se paga con toggles fantasma, que ya fueron un incidente
(`project_escritorio_light_phantom_toggles_2026-05-29`).

### Validación en vivo antes del flip

1. Deploy del `.onnx` nuevo **con el wake textual todavía activo** (red de
   seguridad).
2. 48h de shadow: loguear el score acústico en cada disparo textual. Si el
   modelo nuevo hubiera disparado, el score queda ≥0.4 en el log.
3. Recién con esa evidencia, considerar bajar la dependencia del textual.

**No tocar `wake_word.threshold` para compensar un modelo malo.** Bajar el
umbral a 0.13 haría "funcionar" el wake actual y traería una avalancha de
falsos positivos. El umbral se queda en 0.4; el modelo tiene que subir.

---

## 5. Orden de ejecución

1. **Etapa A** (instrumentar `missed/`) — cambio de código chico, va con TDD.
2. Esperar ≥7 días de captura en las dos habitaciones.
3. Triaje muestreado de `rejected/` (se puede hacer en paralelo a 2).
4. **Etapa B + C** — armar dataset, entrenar, evaluar contra los 4 criterios.
5. Deploy en shadow 48h → flip.

Lo que **no** hay que hacer ahora: entrenar con el dataset tal como está.
Produciría un modelo que scorea mejor los clips que ya disparaban y sigue
sordo a los que no.

---

## Referencias

- `src/wakeword/detector.py:233` — marker `[oww-dbg]`
- `src/wakeword/wake_clip_writer.py:32` — writer, buckets y rotación
- `src/pipeline/multi_room_audio_loop.py:869,927` — únicos callers de `submit()`
- `src/ambient/textual_wake.py:296-312` — dedup acústico y `CommandEvent`
- `scripts/train_custom_wake.py` — pipeline de entrenamiento existente
- Memoria: `project_parakeet_fastpath_shadow_2026-07-14`,
  `project_compuerta_acustica_integral_2026-06-12`,
  `project_mic_relocation_2026-07-02`
