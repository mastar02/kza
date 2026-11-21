# T062 · Evaluación pre-campo de P6

Protocolo `p6-v1`, congelado en desarrollo sobre
`c7c7cd84e098af79d2b71a7855d6aada2e370b85`. Registra el método de aceptación
de C12–C18. T071 debe congelar el candidato acústico y T072 debe producir
evidencia con la reserva. Los estados de campo siguen pendientes.

## Sellos y traspaso T071 → T072

- El custodio asigna inventarios completos antes de ajustar el candidato.
  Mantiene el inventario de reserva y su evidencia privada fuera del alcance
  de quienes calibran. Publica sólo el SHA-256 del inventario sellado y
  cantidades previstas por estrato. La adjudicación humana se hace sin ver la
  hipótesis, clasificación, asignación ni variante del candidato.
- T071 usa sesiones consentidas de calibración y fija revisión Git, contrato
  de corpus, binario/runtime, locks, configuración efectiva sin secretos,
  pesos y descriptor de cada modelo (incluido el encoder H3), DSP observado,
  canal/lane, umbrales, exclusiones, familias, acciones/entidades esperadas y
  sus plazos, orden de captura, ventanas, matriz de estratos y formato de
  evidencia. El manifiesto privado referencia el SHA-256 de este documento
  y el SHA-256 de cada artefacto ejecutado. El custodio concilia audio, hashes,
  sesiones y consentimiento antes de certificar la independencia.
- T072 verifica los sellos antes de abrir la reserva, luego ejecuta en las
  sesiones de otro día y custodia el resultado original. Cualquier cambio de
  identidad del candidato, fórmula, umbral, exclusión o instrumento que afecte
  la comparabilidad requiere otro sello, con justificación y otra reserva
  independiente para los casos afectados. Conservar la primera evaluación y
  sus fallos. Ausencia de datos, sello roto o estrato insuficiente impiden
  aceptar el criterio afectado.
- Los SHA-256 de archivos se calculan sobre sus bytes tal como quedaron en
  Git (`shasum -a 256 ruta`). Los manifiestos JSON usan UTF-8 sin BOM, claves
  ordenadas, separadores `,` y `:` sin espacios, Unicode literal, sin
  NaN/infinito, y preservan orden de arrays. El test ancla el SHA-256 de este
  documento y comprueba los archivos y símbolos especificados abajo. La
  proyección del roadmap incluye metas, mínimos y acuerdos de C12–C18, la
  estratificación, los vínculos T062/T071/T072 y los tres hallazgos con sus
  tareas. El seguimiento de estado y evidencia puede avanzar por separado.

## Admisión y aislamiento

- Registrar por clip ID opaco único, SHA-256 de PCM canónico del clip y
  SHA-256 de PCM canónico de su origen íntegro, sesión, inicio/fin absolutos y
  sujetos audibles. Calcular ambos hashes antes de resamplear, cambiar
  ganancias o aumentar. Audio canónico significa PCM intercalado de la
  fuente, entero con signo little-endian, precedido por tasa, número/orden de
  canales y ancho de muestra declarados, con cabecera ASCII
  `<Hz>:<canales>:<bits>\n` y orden físico de canales del manifiesto. Conservar
  el hash de archivo y genealogía de cada copia, recorte o aumento. La
  genealogía siempre llega al mismo original. Una procedencia incierta
  impide la admisión.
- El inventario distingue `training`, `calibration` y `reserved`. Inventariar
  fuentes de ajuste del candidato que podrían solapar con la captura de campo.
  Si no hubo entrenamiento en ese ámbito, registrar inventario de
  entrenamiento vacío y procedencia de los pesos preexistentes. T071
  calibra con el corpus consentido. Una finalidad de evaluación no habilita
  reentrenamiento, LoRA, clonación ni incorporación al corpus ordinario.
- `validate_inventory` rechaza IDs/hashes/orígenes/sesiones compartidos entre
  particiones. Una ventana semiabierta `[inicio, fin)`, normalizada a UTC,
  no puede solaparse entre desarrollo (entrenamiento/calibración) y reserva,
  aun con IDs distintos. El panel consentido y declarado de antemano puede
  repetirse entre calibración y reserva en sesiones y ventanas independientes.
  Es necesario para evaluar las mismas voces en H3 y permite informar
  resultados domésticos condicionados a ese panel. Las demás voces se
  mantienen disjuntas y una voz de entrenamiento jamás pasa a la reserva.
- `CorpusManifest` v2 sigue rigiendo sesión, origen, derivados, procedencia y
  plazos. `calibration` y `reserved` de este inventario corresponden a
  `CorpusSplit` del mismo valor. Para STT, cada `CorpusComparison` vincula audio,
  referencia humana e hipótesis al mismo intervalo de frames y canal,
  incluidos previews. Verificar el descriptor de DSP observado y separar
  lanes/revisiones. Una hipótesis de otro motor sirve para contraste.
- El custodio verifica una afirmación explícita previa a la captura, con
  finalidad de calibración/evaluación, sesión, alcance y personas audibles.
  Rechazar desconocidos sin permiso, negativas, falta de comprobante,
  revocación y caducidad. Un clip de silencio sin voces requiere igualmente
  una sesión autorizada. Al abrir cada artefacto, revalidar consentimiento
  durable, estado de borrado, identidad del archivo y el menor plazo entre
  vencimiento original, captura + 30 días, cierre y revocación. Una copia o
  restauración hereda el plazo original. El validador puro comprueba el
  inventario y recibos aportados por el custodio. La apertura usa la
  autoridad de evaluación vigente.
- Etiqueta de voz de referencia, agrupación acústica sugerida, persona,
  cuenta, roles y consentimiento tienen referencias separadas. La
  adjudicación acústica usa IDs opacos de voces; un perfil nombrado, un
  coseno alto, una cuenta activa o una asociación manual no crean una
  identidad verificada, un permiso ni consentimiento. No exportar WAV,
  transcripciones, embeddings, nombres o relaciones privadas a Git.

## Orden, estratos y fallos

- Congelar el orden original por secuencia de captura. Cada permutación H3
  ordena por `(SHA256(JSON(["p6-v1","h3",semilla,clip_id])), clip_id)`, con
  semillas 0–9. Cada corrida empieza con estado de agrupación vacío. Se usa
  el mismo inventario y los mismos parámetros en las once corridas y se
  informa cada una. Para A/B se usa el dominio `listening` y semilla 0.
- En C12–C17, fijar plan de muestreo con Living, Hall, Escritorio, Cocina,
  1 m y 3 m, tres participantes, claro y adversos admitidos (ruido habitual,
  TV y demás condiciones preregistradas). Mínimo de diez casos elegibles por
  habitación, distancia y participante para los recorridos aplicables, más
  mínimos globales y por condición. Reportar n y métricas por estrato.
  Aplicar las metas de su condición a cada estrato; el promedio global no
  rescata un estrato fallido o insuficiente.
- Antes de ver salidas, marcar exclusiones por ausencia de consentimiento,
  captura fuera de guion, habla ininteligible adjudicada sin hipótesis o
  fuente sin procedencia comparable. Reportar cantidades y causas. Silencio
  y media sin una petición se tabulan aparte. Una petición elegible con STT
  vacío, sin respuesta, sin efecto o con fallo de instrumentación permanece
  en su denominador. No se aceptan denominadores sin evidencia suficiente.
  Publicar numerador, denominador, cantidad y faltantes por resultado.
  Comparar los valores sin redondeo al umbral; indicar resolución temporal
  y efectos de ausencia/ambigüedad en las cotas.

## C12 · Reconocimiento y AUD-06

Con referencia humana alineada, mínimo **200** peticiones claras, **100**
adversas elegibles y **20** por condición crítica. Calcular por condición y
estrato sobre el inventario completo:

- `N` es el número de peticiones elegibles. `n` cuenta pares alineados y
  verificados. Cobertura `n/N ≥ 0,95`. Una hipótesis confirmada como vacía
  entra con todas sus deleciones; una hipótesis ausente o no verificable
  reduce cobertura.
- Usar `normalize_words` y `score` de `src.ambient.wer`: minúsculas, espacio
  como separador, puntuación de bordes fuera, acentos y ñ conservados.
  Levenshtein unitario con retroceso que prioriza coincidencia, sustitución,
  inserción y deleción. `WER_micro = Σ(S+I+D) / Σ(palabras de referencia)`
  con máximos **0,10** claro y **0,20** adverso. Publicar `WER_macro =
  media_i((S_i+I_i+D_i)/palabras_i)`, S/I/D totales y vacíos sobre habla.
  `CER_micro = Σ(distancia de edición de caracteres) / Σ(caracteres de
  referencia)`, con puntos de código de los tokens normalizados concatenados
  sin espacios. Para silencio revisado, informar hipótesis no vacías sobre
  ventanas de silencio verificadas y sus inserciones, separadas del WER.
- El macro y el reponderado por buckets VAD de `tools/ambient_wer.py` conservan
  sus rótulos y su denominador de origen. Scores STT, VAD y coseno son
  diagnósticos del motor/versión. Si T071 calibra una regla basada en scores,
  preregistra señal, objetivo binario, umbral, cohortes y conteo por intervalo
  de score exclusivamente con calibración. T072 informa por intervalo n,
  proporción de aciertos y ausencias en reserva; si el motor no emite la
  señal, se registra ausente. Los scores crudos no son probabilidades
  calibradas. El efecto de tarea se mide en C13.

## C13 · Éxito real de tarea

Evaluar todas las peticiones elegibles de C12, con **10** casos como mínimo
por familia. Registrar intención, objetivo/entidad y valor esperados,
resultado de tarea, evidencia independiente del efecto esperado y ámbito
(software, HA o físico). Si la tarea es de consulta, el resultado verificable
es el dato correcto y vigente; para cambios de estado o tareas compuestas,
verificar todos los efectos requeridos en el plazo preregistrado. Contar
`completadas verificadas / elegibles` con mínimos **0,98** claro y **0,95**
adverso. Los resultados atendidos, despachos aceptados y confirmaciones
habladas sin efecto verificable quedan en el denominador como fallos.
Informar errores de intención, entidad y efecto por familia. Usar
`TaskOutcome.task_completed` como evidencia del dominio junto con la
adjudicación independiente; medir entrega por separado.

## C14 · Wake

Preregistrar **100** o más intentos intencionales con ventana de correlación
fija por intento. Inventariar tanto esos intentos como todas las emisiones
admitidas al flujo de wake del candidato en **48 horas útiles de reloj de
casa** o más. Una emisión se admite al abrir un turno de escucha o iniciar
feedback/acción de wake. Un intento verdadero necesita una emisión
admitida, atribuida al mismo evento físico dentro de su ventana. Una
emisión admitida sin wake intencional correspondiente es falsa si la
revisión lo confirma. Conservar aparte detecciones crudas y descartes;
un descarte posterior a la admisión no borra la emisión. Medir la unión de
segundos con las **cuatro** entradas útiles simultáneamente, fuera del
denominador los huecos documentados y sin multiplicar por micrófonos. Registrar
por separado todas las emisiones por micrófono y los resultados por
habitación con sus correspondientes horas útiles. Sólo deduplicar emisiones
con evidencia sincronizada de un mismo evento físico; proximidad temporal
sola no permite deduplicar. Una emisión incierta sin corroboración conserva
su propio conteo.

- Revisión de emisiones `revisadas / emitidas ≥ 0,95`. Cota de rechazo:
  `(intentos rechazados + intentos inciertos) / intentos intencionales ≤ 0,05`.
  Conservar intentos sin wake, asociados a rechazo cuando la revisión lo
  confirma. Cota de falsos wakes: `(emisiones falsas confirmadas + emisiones
  inciertas) / horas útiles de casa ≤ 0,1`. Publicar además tasas observadas
  sin inciertos y ambos conteos de incertidumbre. Un denominador ausente o
  inventario incompleto impide dictamen.
- `WakeWordDetector.detect` y `WhisperWakeDetector.detect_with_provenance`
  aportan la decisión del detector según el backend.
  `WakeVerifier.evaluate` aporta evidencia por entrada del filtro posterior
  a STT. La verdad de campo se etiqueta de forma independiente con el guion
  de intentos y el registro físico de todas las emisiones.

## C15 · Demora doméstica

En **100** recorridos simples como mínimo, con **10** o más por familia y
desglose frío/caliente definido antes de la ejecución, registrar en reloj
monotónico común o con sincronización y error medidos:

- `t_inicio`: final de la última muestra del habla del usuario, adjudicado
  en captura. `t_voz`: comienzo audible de la primera palabra pertinente.
  Un earcon, relleno y primer PCM entregado al dispositivo no puntúan.
  `t_voz − t_inicio` con **p50 ≤ 800 ms**, **p95 ≤ 1500 ms**.
- `t_efecto`: primera evidencia verificable del resultado software o
  estado HA de una acción rápida. `t_efecto − t_inicio` con
  **p95 ≤ 2000 ms** sobre las acciones rápidas elegibles. Un movimiento
  largo lleva plazo propio en el manifiesto; registrar observación HA y
  resultado físico como ámbitos separados.
- Usar rango más próximo `ordenados[ceil(p × n) − 1]` (p = 0,50 o 0,95),
  representando un endpoint fallido, ausente o sin reloj comparable por
  +∞ y publicando esos conteos. Si la incertidumbre de reloj alcanza el
  umbral, evaluar la cota superior. Publicar las latencias de componentes
  como diagnóstico separado. El benchmark de componentes calcula su
  resumen sobre éxitos.

## C16 · Entrega e interrupción

En **100** entregas debidas o más, contar `entregas completas / entregas
debidas ≥ 0,99`. Verificar `ResponseResult.success`, inicio PCM y final de
todo el contenido audible mediante observación independiente. Registrar
entrada en cola, primer PCM y fin audible/último sonido, generación, entrega
y timeout. Un resultado parcial o sin evidencia de fin queda en el
denominador como fallo. Medir el estado declarado frente al hecho observado:
`FAILED`, `BLOCKED`, `TIMED_OUT` según causa, `CANCELLED` si hubo
cancelación confirmada. Registrar si fue parcial con evidencia de audio
iniciado y contenido restante; jamás clasificarlo como completo. Exigir
estado correcto en **todos** los fallos, parciales y cancelaciones.
Las cancelaciones voluntarias confirmadas se informan por separado, con su
estado, sin contar como entrega debida a término. Un fallo sin evidencia de
cancelación voluntaria permanece como entrega debida.

Medir **20** interrupciones físicas o más con barge-in, captura real
concurrente y cobertura de respuesta larga no cacheada. Desde inicio audible
de la solicitud de interrupción hasta el último sonido atribuible a la
respuesta interrumpida (0 ms si ya no es audible al inicio). Aplicar mismo
rango próximo y cota de reloj de C15, con **p95 ≤ 250 ms**. Una parada sin
confirmación observable se conserva como endpoint ausente.

## C17 · Claridad y AUD-04

Al menos **30** pares de respuestas cortas/largas con el mismo contenido,
escuchados por **tres** participantes. Sellar un lote de tamaño par,
ordenado por el dominio A/B anterior. Ordenar los participantes por su ID
opaco. En posición `i` (base cero), el participante `j` oye primero la
variante interna 0 cuando `(i+j) % 2 == 0`, y la 1 en los demás casos.
Sólo el custodio conoce el mapa global de variantes; revelar después de
fijar las notas. Usar capturas nuevas y consentidas antes/después del cambio,
con volumen, nivel, frase, distancia y condición de ruido comparables,
canal conocido, misma configuración salvo la variable ensayada y
referencias alineadas.

Rúbrica por respuesta: **1** ininteligible, **2** exige reconstruir contenido,
**3** se entiende con esfuerzo o repetición, **4** clara a primera escucha con
defecto menor como máximo, **5** clara a primera escucha sin defecto
perceptible. Marcar metalización o recorte grave si distorsiona o enmascara
contenido, o dificulta entenderlo a primera escucha. Preservar cualquier
marca grave de un participante y publicar conteos por variante y tipo.
En la voz actual, exigir **mediana ≥ 4/5 en cada participante** y **cero
respuestas con marca grave** en todo el lote. Publicar resultados por
variante y participante, con estratos aplicables. Alcance de la rúbrica:
claridad interna, sin MOS validada. Contrastar WER por separado. RMS,
plateaus, saltos y lectura de ganancia son diagnósticos; por sí solos no
atribuyen metalización, recorte ni causa física.

## C18 · Agrupación H3 y AUD-07

Mínimo **tres voces** consentidas, **10** segmentos claros por voz en
calibración y **30** por voz en reserva. Cubrir **dos habitaciones y dos
distancias** y **10** casos por condición adversa (cortos <2 s, TV, eco,
solapamiento). Congelar en T071, junto al encoder, el coseno, margen,
duración y límite de miembros. La implementación fijada de
`VoiceGroups.add` usa enlace completo, 0,88, margen 0,08, duración mínima
2 s y tope 32. Estos son parámetros del candidato en código; su validez
acústica requiere campo. No incorporar calibración, grupos manuales o
perfiles existentes como miembros de la evaluación.

Puntuar las asignaciones de cada corrida en la reserva completa. Asignar
un grupo predicho sólo cuando el evaluador sugiere grupo; una etiqueta
`unknown` significa abstención aun si en almacenamiento tiene ID. Un
grupo unitario sugerido cuenta para cobertura y aporta cero pares.

- Entre todos los pares no ordenados de clips claros elegibles, VP son
  misma voz y mismo grupo, FP distintas voces y mismo grupo, FN misma
  voz y grupos distintos o alguna abstención. Exigir
  `VP/(VP+FP) ≥ 0,99`, `VP/(VP+FN) ≥ 0,80`, y
  `claros asignados / claros elegibles ≥ 0,70`. Un denominador nulo
  impide aceptación. Publicar abstenciones, separaciones y latencia de
  proceso, por condición y corrida.
- Exigir cero pares agrupados entre personas distintas en toda condición
  con identidad adjudicable, y abstención en **todos** los cortos y
  solapamientos. La identidad múltiple de un solapamiento no se proyecta
  a una sola voz. Estos casos se prueban contra las salidas reales aun
  cuando el agrupador no tenga señal explícita de solapamiento. Evaluar
  orden original y diez órdenes con semillas 0–9; cada corrida debe
  satisfacer todas las metas.

AUD-04, AUD-06 y AUD-07 quedan cubiertos por protocolo en T062. T072
aporta evidencia de campo para verificar los resultados y cualquier
afirmación de mejora o aceptación acústica.

## Manifiesto verificable

```json
{
  "protocol": "p6-v1",
  "roadmap_contract_sha256": "326d238d74dc42991587d3e1ece5f56eb74645ee70e75dbd3f4eafc738b42595",
  "limits": {
    "C12": {"wer_clear_max_pct": 10, "wer_adverse_max_pct": 20, "coverage_min_pct": 95, "clear_min": 200, "adverse_min": 100, "critical_condition_min": 20},
    "C13": {"success_clear_min_pct": 98, "success_adverse_min_pct": 95, "family_min": 10},
    "C14": {"rejection_max_pct": 5, "false_per_house_hour_max": 0.1, "intentional_min": 100, "useful_house_hours_min": 48, "simultaneous_inputs": 4, "review_min_pct": 95},
    "C15": {"voice_p50_ms_max": 800, "voice_p95_ms_max": 1500, "rapid_effect_p95_ms_max": 2000, "simple_runs_min": 100, "family_min": 10},
    "C16": {"fully_delivered_min_pct": 99, "acoustic_interrupt_p95_ms_max": 250, "deliveries_min": 100, "physical_interrupts_min": 20},
    "C17": {"clarity_median_min": 4, "clarity_scale": 5, "severe_max": 0, "paired_responses_min": 30, "listeners_min": 3},
    "C18": {"mixed_pairs_max": 0, "pair_precision_min": 0.99, "pair_recall_min": 0.8, "clear_coverage_min": 0.7, "voices_min": 3, "calibration_clear_per_voice_min": 10, "reserve_clear_per_voice_min": 30, "rooms_and_distances_each_min": 2, "per_adverse_condition_min": 10, "first_seed": 0, "last_seed": 9}
  },
  "sources": {
    "src/ambient/wer.py": {"sha256": "e53b52c627012ce9095a7c874de16947b8fa87b0bea4d78b7d8e90ac9162203e", "symbols": ["normalize_words", "is_excluded", "score", "_edit_ops"]},
    "src/core/task_outcome.py": {"sha256": "c214a0adf96820e2cf796950e16b0ce920b1394631fbbf027878f769e3f6d8a4", "symbols": ["TaskOutcome.task_completed", "project_turn_outcome"]},
    "src/core/voice_delivery.py": {"sha256": "19427ffca636619249b01de13112ec054115dc12e826f3162226f1b5dabab894", "symbols": ["ResponseResult.success", "DeliveryState"]},
    "src/evaluation/corpus.py": {"sha256": "49480cd5c015f70b6e04836fd2956abbc84b399bd1c5114408e9ee53bb3f9b79", "symbols": ["CorpusManifest.require_comparison", "CorpusManifest.deadline"]},
    "src/evaluation/freeze_manifest.py": {"sha256": "80bb5d0f71bd379f883c78fb5d526afb58978627fd9ef05382bbc15f10945f06", "symbols": ["validate_inventory", "canonical_sha256", "seeded_order", "recognition", "complete_task", "WakeCounts.rejection_upper", "WakeCounts.false_per_hour_upper", "WakeCounts.review_coverage", "nearest_rank", "complete_delivery", "clarity_median", "voice_pair_counts", "PairCounts.precision", "PairCounts.recall", "PairCounts.coverage"]},
    "src/evaluation/provenance.py": {"sha256": "958cea046aaac2f978156732f74134e89fa97a443b6c16f3fd3e142c91f574d1", "symbols": ["CaptureProvenance.comparable", "FrameInterval.millisecond_bounds"]},
    "src/nlu/household_intents.py": {"sha256": "0085c9abe5579d67d361b10bccd0499dd573c0459d7b6fc2910aee85c2c54f6a", "symbols": ["HouseholdIntentClassifier.classify"]},
    "src/pipeline/post_wake_endpoint.py": {"sha256": "82722926c2c0e13e1a84fd072c65183b489aacafe52cffd9ccf2e5c62c2c7ae6", "symbols": ["EndpointState.analyze"]},
    "src/pipeline/response_handler.py": {"sha256": "11d229715fe54820de5b52590bd1c8e6b0c9c1a1e8976c386f1cdc90f5197004", "symbols": ["ResponseHandler.aplay_earcon", "ResponseHandler._result", "ResponseHandler._render", "ResponseHandler.cancel"]},
    "src/pipeline/wake_verification.py": {"sha256": "389719aec759a76ebcac291d2bb9d6a17252614b09efdc1eb1609a4c2a8a33e0", "symbols": ["WakeVerifier.evaluate"]},
    "src/stt/parakeet_stt.py": {"sha256": "4f8f4e86df5825d6c817cca8c747505c76971e18c9c37521d28adbdb66abd1c2", "symbols": ["ParakeetSTT.transcribe_with_confidence"]},
    "src/stt/whisper_fast.py": {"sha256": "6344dfbbff7b950e4d77647ee7c7387faf465264e5c8712440c8a17289c6a78b", "symbols": ["FastWhisperSTT.transcribe_with_confidence"]},
    "src/users/speaker_identifier.py": {"sha256": "af96a453fe7fb6b1d7819bf288215cfb4cc27b7c471b6b84e9f3ca37d0fc1297", "symbols": ["SpeakerIdentifier.get_embedding"]},
    "src/users/voice_groups.py": {"sha256": "a8f37c1d91925bcfb356d60bd569e8b9e8a844bcdca4115473f6b51aeffa9ab8", "symbols": ["VoiceGroups.add", "VoiceGroups.metadata"]},
    "src/wakeword/detector.py": {"sha256": "2bf6523140c5267d12366a4a1edaf31e3535e862ae55da7494396186caf17d29", "symbols": ["WakeWordDetector.detect"]},
    "src/wakeword/whisper_wake.py": {"sha256": "dda37cd47905a1e1325891bf7ac12a88e450d2c1aedf04458cb7a5c4af90a354", "symbols": ["WhisperWakeDetector.detect_with_provenance"]},
    "tools/acoustic_calibration.py": {"sha256": "7779a270b5b60dc3173547f18b9932b42af643fb0ebe50b3a0ebe886021847c4", "symbols": ["summarize", "signal_gap"]},
    "tools/ambient_wer.py": {"sha256": "27d39e18691814d0dae9e8465e0cf5e84497bb05eade3030ed68b1fb8b24f747", "symbols": ["build_report"]},
    "tools/analyze_audio_quality.py": {"sha256": "72bbbfb055770cfbd4248c78a8eff782934f47c7fa9754240295b9b8b6c37b83", "symbols": ["measure_channel"]},
    "tools/benchmark_latency.py": {"sha256": "6ab0a53e35078273f2f51185482f4c95f9d04df32f543abf745388194d97ee0a", "symbols": ["LatencyBenchmark._summarize"]}
  }
}
```
