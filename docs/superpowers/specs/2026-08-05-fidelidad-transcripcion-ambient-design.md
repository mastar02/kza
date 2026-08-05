# Fidelidad de la transcripción ambient: medición y fusión multi-micrófono

**Fecha:** 2026-08-05
**Estado:** diseño aprobado, pendiente de plan de implementación
**Origen:** investigación de los toggles fantasma de `light.grupo_escritorio` (2026-08-04),
que derivó en el análisis de calidad del ambient path.

---

## 1. Motivación

El usuario observó que las transcripciones del ambient "traducen 1 de 5 palabras bien" y
planteó la visión de tener el 100% de lo conversado en casa transcripto por micrófono, con
micrófonos que se complementen cuando escuchan lo mismo.

La medición sobre `ambient.db` (ventana de 48,8 h, 5.939 utterances, 2 habitaciones) muestra
que la impresión es correcta **para la mayoría del volumen** e incorrecta para la cola alta.
El driver no es el modelo: es la relación señal/ruido.

| `vad_prob` | n | español | inglés (garble) | largo medio |
|---|---|---|---|---|
| 0.00–0.20 | 221 | 5,9% | 86,9% | 11 |
| 0.20–0.35 | 1.238 | 14,5% | 73,3% | 15 |
| 0.35–0.50 | 1.462 | 29,8% | 50,8% | 22 |
| 0.50–0.65 | 1.551 | 52,5% | 23,5% | 35 |
| 0.65–0.80 | 1.368 | 81,1% | 7,3% | 79 |
| 0.80–1.00 | 283 | **94,0%** | 4,6% | 161 |

Mismo modelo (Parakeet-TDT), misma habitación. A `vad>0.85` produce texto fiel y coherente;
debajo de 0.35 produce fragmentos cortos en inglés. El 55% del volumen vive debajo de 0.5.

Por habitación: escritorio 49,6% español, cocina 24,2% — la cocina está más lejos de donde
se habla.

**Sobre la complementariedad entre micrófonos:** existen 500 pares de utterances solapadas
entre escritorio y cocina. En el 82,4% de esos pares el micrófono con mejor `vad_prob` cae en
otro bucket de calidad, y en el 42,4% el mejor da español mientras el peor no. La idea es
válida y está medida — pero esos 500 pares son el **8,4%** de las utterances. El 91,6%
restante lo escucha un solo micrófono.

### Por qué no se puede evaluar hoy

Nadie sabe qué se dijo realmente en esas utterances. El ambient path **transcribe y descarta
el audio**: a `ambient.db` solo va texto. Sin audio no hay referencia posible, no se puede
calcular WER, y no se puede re-transcribir cuando mejore el modelo.

Cualquier afirmación sobre "95% de fidelidad" hoy sería inventada.

---

## 2. Objetivo

Construir el instrumento que permita **medir** la fidelidad real de la transcripción ambient,
y recién entonces evaluar la fusión multi-micrófono contra ese instrumento.

### No-objetivos (explícitos)

- **No** se construye el archivo permanente de lo conversado. Ese es un spec aparte, y su
  diseño depende de si el enrolamiento far-field funciona (sesión del 2026-08-05 21:47).
- **No** se cambia el motor de STT del ambient ni el del command path.
- **No** se toca el command path, el wake, ni el dispatcher.
- **No** se persigue "95% global". Con la distribución actual eso no es alcanzable por
  software: debajo de `vad 0.35` la información no llegó al micrófono. La meta correcta es
  medir por bucket y subir la proporción de audio que cae en los buckets altos.

---

## 3. Pieza A — El instrumento de medición

### A1 · Persistencia de audio del ambient

**Qué:** guardar el WAV/FLAC del segmento junto a cada utterance, con el mismo TTL que el
texto (48 h).

**Decisión de diseño crítica — el texto vacío NO puede cortar la persistencia.**
Hoy `AmbientTranscriber._handle_segment` hace `if not text: return` antes de persistir. Si el
audio se guardara después de ese chequeo, el dataset solo contendría los casos donde el modelo
dijo *algo* — y sería imposible medir el modo de falla más importante: los segmentos de habla
real que el modelo transcribió como vacío. Esa es exactamente la clase de proxy mentiroso que
ya nos costó incidentes (`arecord exit 0`, `last_frame_ts`). Con `keep_audio` activo, todo
segmento que el segmenter emitió llega a `store.add()`, con o sin texto.

**Orden exacto de las operaciones** (importa, porque el nombre del archivo depende del id):
1. `store.add(utt)` → devuelve `utterance_id` (con `text_empty=1` si el texto vino vacío).
2. Se escribe `data/ambient_audio/<room_id>/<utterance_id>.flac` en un hilo aparte.
3. `store.set_audio_path(utterance_id, path)` en un UPDATE.

Si el paso 2 o 3 falla, la fila queda con `audio_path = NULL` y el pipeline sigue. Nunca al
revés: no se escribe audio que no tenga fila, para que la purga por TTL no pueda dejar
huérfanos por diseño.

**Formato:** FLAC mono 16 kHz (lossless, vía `soundfile`, ya en `requirements.txt`).
Se elige lossless porque el TTL es corto y el propósito es medir y re-transcribir; una pérdida
de compresión contaminaría la medición. Costo: ~190 MB/día para 2 habitaciones (13,7% de habla
sobre 24 h), ~380 MB en la ventana de 48 h. El server tiene 314 GB libres.

**Layout:** `data/ambient_audio/<room_id>/<utterance_id>.flac`.
Para segmentos sin texto (que no generan fila en `utterances`) hace falta un id igual — ver
esquema abajo.

**Config nueva** (`config/settings.yaml`, bloque `ambient`):
```yaml
  keep_audio:
    enabled: false          # OFF por default; se prende para la campaña de medición
    dir: "./data/ambient_audio"
    format: "flac"
```

**Esquema:** el audio se referencia desde la fila. Dos cambios en `utterances`:
- `audio_path TEXT` — ruta relativa al FLAC, o NULL si `keep_audio` estaba apagado.
- Las utterances con texto vacío hoy no se insertan. Se agrega `text_empty INTEGER DEFAULT 0`
  y se insertan igual **solo cuando `keep_audio.enabled`** — para no cambiar el volumen ni el
  comportamiento de la DB en operación normal. El distiller ya filtra por `lang_ok` y
  `vad_prob`, y se le agrega `text_empty=0` para que nunca las consuma.

**Purga:** `AmbientStore.purge_expired` hoy borra filas. Se extiende para borrar también el
archivo de audio referenciado. Un archivo huérfano (fila borrada, archivo presente) se limpia
en el mismo barrido escaneando el directorio contra las filas vivas.

### A2 · Set de ground truth (transcripción humana, ciega)

**Qué:** 40 utterances estratificadas por bucket de `vad_prob` (buckets de la tabla de §1),
que el usuario escucha y transcribe a mano.

**Ciego por diseño.** La herramienta muestra el audio y un campo de texto vacío. **No** muestra
lo que transcribió el modelo, ni antes ni durante. Mostrarlo contaminaría la referencia por
anclaje — es la lección directa del eval de clima/AC, donde reportamos 95,5% midiendo sobre el
mismo set con el que se construyó la solución, y el held-out limpio reveló el fallo que canceló
el proyecto.

**Estratificación: asignación IGUAL por bucket, no proporcional.** Seis buckets × 7 utterances
= 42. Una asignación proporcional le daría al bucket `0.80–1.00` apenas 2 muestras (283 de
5.939), que es justamente el bucket donde queremos saber si ya estamos en ~95%. El muestreo es
aleatorio dentro de cada bucket, con semilla fija registrada en el archivo de salida.

Consecuencia a tener presente al leer el reporte: el WER **agregado** de este set no es el WER
del sistema, porque el set sobre-representa los buckets altos. El agregado real se obtiene
re-ponderando cada bucket por su volumen en la DB, y el runner lo reporta de las dos formas
(por bucket, y agregado re-ponderado).

**Entregable:** `tools/ambient_groundtruth.py`
- `--export N` → arma el set: copia los FLAC a un directorio de trabajo y genera un
  `groundtruth.json` con `{utt_id, room, vad_prob, audio_path, reference: null}`.
- Genera además un `index.html` autocontenido (audio + textarea + guardar) para transcribir
  cómodo. Sin red, sin dependencias.
- `--import` → recibe el JSON completado y lo valida (todas las referencias presentes).

**Casos especiales que la referencia debe poder expresar:**
- `""` (vacío) — el segmento no contenía habla inteligible. Necesario para medir inserciones.
- `[ininteligible]` — hay habla pero el humano tampoco la entiende. Se excluye del WER y se
  reporta aparte: es el techo real del audio, no un error del modelo.
- `[tv]` / `[media]` — la fuente es un parlante. Se mide aparte de la voz de la sala.

### A3 · Runner de WER

**Entregable:** `tools/ambient_wer.py`

**Sin dependencias nuevas.** WER por distancia de edición a nivel palabra (Levenshtein con
matriz, stdlib). El proyecto no incorpora `jiwer` por una métrica de 40 líneas.

**Normalización antes de comparar** (documentada y testeada, porque define el número):
minúsculas, colapso de espacios, quitar puntuación de borde. **Se conservan los acentos y la ñ**
— son señal real de calidad en español y quitarlos inflaría el resultado.

**Métricas reportadas, siempre por bucket de `vad_prob` y en agregado:**
- WER, y su descomposición en sustituciones / inserciones / **deleciones**.
- **Tasa de deleción total**: fracción de segmentos con habla real (referencia no vacía) cuyo
  texto producido fue vacío. Esta es la métrica que A1 habilita y que hoy es invisible.
- **Tasa de alucinación**: fracción de segmentos con referencia vacía cuyo texto producido no
  lo fue.
- Correlación entre `vad_prob` y WER, para validar o refutar que el bucket es buen predictor.

**Salida:** tabla en consola + `data/wer_report_<fecha>.json` para comparar corridas.

---

## 4. Pieza B — Fusión multi-micrófono

Se implementa **después** de A, y su mérito se juzga con A3.

### Decisión de arquitectura: fusión en lectura, no en escritura

La fusión **no** modifica lo que se persiste. `ambient.db` sigue siendo el registro crudo: una
fila por micrófono por segmento. La fusión es una **vista** sobre el store.

Razones:
- Colapsar en escritura destruye información de forma irreversible; si el criterio de selección
  resulta malo, no hay vuelta atrás.
- No toca el hot path del ambient (nada de latencia añadida, nada que pueda romper la captura).
- Es trivialmente testeable: función pura sobre filas.

### Componentes

**`src/ambient/fusion.py`** (módulo nuevo, ~120 líneas)

- `find_overlaps(utterances) -> list[OverlapGroup]`
  Agrupa utterances de **distintas** habitaciones cuyos intervalos `[t0, t1]` se solapan.
  Función pura sobre una lista ordenada por `t0`. Barrido lineal con ventana, no O(n²).

- `@dataclass FusedEvent`
  `t0`, `t1`, `text` (del micrófono ganador), `winner_room`, `sources: list[RoomHypothesis]`
  con la hipótesis y el `vad_prob` de cada sala. **La procedencia se conserva entera**: el
  evento fusionado nunca borra lo que dijo el otro micrófono.

- `select_best(group) -> FusedEvent`
  v1: gana el `vad_prob` más alto. Criterio simple y ya medido (82,4% de los pares cambian de
  bucket). Se deja explícitamente afuera la fusión ROVER a nivel palabra: con dos hipótesis, una
  de las cuales suele ser basura en otro idioma, combinar es peor que elegir. Si A3 muestra que
  elegir no alcanza, se reevalúa con datos.

**`AmbientStore.fused_between(t0, t1)`** — reader que aplica `find_overlaps` + `select_best`
sobre `utterances_between` de todas las salas.

### Umbral de decisión

B se mergea solo si A3 muestra **mejora de WER en los eventos solapados**, medida contra el
mismo ground truth. Si el WER del ganador no es mejor que el de la fila que hoy se guardaría,
B no aporta y no entra. El criterio se fija **antes** de correr la medición.

---

## 5. Manejo de errores

Todo lo nuevo es **best-effort y fail-open**, igual que el resto del ambient path:

- **Escritura de audio (A1):** un fallo de disco (lleno, permisos, I/O) se loguea como WARNING
  y la utterance se persiste igual, con `audio_path = NULL`. **Nunca** puede tumbar el worker
  de una habitación ni bloquear la transcripción. La escritura va por `asyncio.to_thread` para
  no bloquear el event loop.
- **Disco lleno:** chequeo de espacio libre antes de escribir; debajo de un piso configurable
  (1 GB) se desactiva la captura de audio en caliente y se loguea una sola vez por hora.
- **Purga:** un archivo que no se puede borrar se loguea y no aborta el resto del barrido.
- **A2/A3** son herramientas offline (`tools/`), fuera del servicio. Fallan ruidosamente, que es
  lo correcto para un instrumento de medición: un runner que se traga un error produce números
  falsos, y números falsos son peores que ningún número.
- **Fusión (B):** función pura; una entrada degenerada (intervalos invertidos, `vad_prob` NULL)
  devuelve el grupo sin fusionar en vez de lanzar.

---

## 6. Testing

Siguiendo el patrón del proyecto: pytest, fixtures en `conftest.py`, mocks en `tests/mocks/`.

**A1 — `tests/unit/ambient/test_audio_persistence.py`**
- Se escribe el FLAC y `audio_path` queda seteado.
- **Se persiste audio también cuando el texto es vacío** (el caso que habilita medir deleciones).
- `keep_audio.enabled=false` → no se escribe nada y el comportamiento actual no cambia (test de
  regresión sobre el volumen de filas).
- Fallo de escritura → utterance persistida con `audio_path=NULL`, sin excepción propagada.
- `purge_expired` borra fila **y** archivo; huérfanos limpiados.

**A3 — `tests/unit/tools/test_ambient_wer.py`**
- WER de casos conocidos a mano (identidad = 0; una sustitución en 4 palabras = 0.25).
- La normalización conserva acentos y ñ (`"apagá" != "apaga"` cuenta como error).
- Deleción e inserción se contabilizan en la métrica correcta.
- `[ininteligible]` se excluye del WER y aparece en el reporte aparte.

**B — `tests/unit/ambient/test_fusion.py`**
- Solapamiento detectado solo entre salas distintas (dos utterances de la misma sala que se
  solapan no se agrupan).
- Gana el `vad_prob` más alto; la procedencia del perdedor se conserva.
- Sin solapamiento → cada utterance queda como evento propio.
- `vad_prob` NULL en un lado → no se fusiona, no lanza.
- Barrido lineal: test de performance sobre 10k filas sintéticas.

---

## 7. Criterios de éxito

1. Existe un número de WER por bucket de `vad_prob`, reproducible, con referencia humana ciega.
2. Se conoce la **tasa de deleción** del ambient — hoy invisible.
3. Queda decidido con datos si la fusión multi-micrófono mejora la fidelidad, o no.
4. Nada de lo anterior cambia el comportamiento del command path, el wake ni el dispatcher.

## 8. Riesgos y dependencias

- **La campaña de medición requiere prender `keep_audio` en producción** durante ≥48 h. Es un
  cambio de config en una máquina que usa el hogar a diario: se coordina con el usuario, se
  verifica espacio, y se apaga al terminar.
- **40 utterances es una muestra chica.** Da una estimación con intervalo ancho, no una medición
  fina. Es suficiente para distinguir "1 de 5 palabras" de "9 de 10" por bucket, que es la
  pregunta planteada. Si algún bucket queda ambiguo, se amplía ese bucket.
- **El habla leída de la sesión de enrolamiento sobreestima la calidad** respecto del habla
  espontánea. Las dos fuentes son complementarias y se reportan por separado, nunca mezcladas
  en un promedio.
- **Privacidad:** `keep_audio` guarda audio crudo de todo lo que se hable en la casa durante la
  campaña, incluidas terceras personas. TTL de 48 h y borrado verificado al terminar.
