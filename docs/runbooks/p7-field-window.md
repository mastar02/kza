# P7 · Ventana física y decisiones Q16/Q17

Preparar este expediente durante T061, según Q8. Antes de abrir P7, el
coordinador comprueba T060–T064, Tier 2, el ajuste del artefacto al perfil y
las decisiones Q16/Q17 registradas. El operador autorizado conduce la adopción
Q4 y el titular autoriza las intervenciones. El custodio controla el corpus y
la reserva. Una revisión posterior coteja los registros originales.

- [Orden de ejecución y gates](../plans/2026-11-21-precedence-roadmap-and-test-strategy.md)
- [Release P6, adopción W0–W10 y L4](p6-release.md)
- [Protocolo sellado de evaluación p6-v1](../plans/p6-evaluation.md)
- [Aceptación física de Cocina y guion F1–F4](2026-11-16-cocina-acceptance.md)

## Solicitud privada

Usar CPython 3.13 desde la raíz del checkout. Definir `PY` con el intérprete
del venv y `PRIVATE` con un directorio privado fuera del checkout y del árbol
de producción. Preparar `proposal.json` con esta forma. Los identificadores,
horarios y hashes de abajo son ilustrativos.

```json
{
  "candidate": {
    "git_commit": "<SHA Git completo>",
    "archive_sha256": "<SHA-256 del tar compilado de P6>",
    "schema_contract_epoch": 2
  },
  "q16": {
    "inventory_sha256": "<SHA-256 del inventario privado revisado>",
    "differences": "<selección vigente, diferencias y límites>",
    "selection": {
      "I01": {"energy": ["sensor.energia_ejemplo"], "power": ["sensor.potencia_ejemplo"]},
      "I02": "todo.compartida_ejemplo",
      "I03": {"kind": "ics", "source": "hogar-ejemplo.ics"},
      "I04": {"entity": "media_player.ejemplo", "controls": ["play", "pause", "volume"]},
      "I05": {"entities": ["binary_sensor.salud_ejemplo"], "video": false}
    }
  },
  "q17": {
    "operator": "<referencia del operador>",
    "window": {"start": "2026-11-23T00:00:00Z", "end": "2026-11-30T00:00:00Z"},
    "interventions": ["adopt", "promote", "HA", "WS", "F1", "F2", "F3", "F4"]
  }
}
```

En I03 se admite también `{"kind":"ha","source":"calendar.ejemplo"}`. El
nombre `.ics` identifica el archivo privado bajo `state/data/`. Los dos
adaptadores son de lectura. En I01 asociar cada sensor a su tipo, unidad y
vigencia. En I02 elegir una sola entidad compartida y confirmar
reconciliación, deduplicación y privacidad de listas locales. En I04
comprobar el dispositivo y acotar las operaciones a sus capacidades. En I05
usar sólo disponibilidad y eventos técnicos ya expuestos y autorizados.

```bash
umask 077
"$PY" -B -m tools.p7_field_protocol render-q16-q17 "$PRIVATE/proposal.json" \
  --release-review "$PRIVATE/release-review.json" > "$PRIVATE/q16-q17.json"
"$PY" -B -m tools.p7_field_protocol render-q16-q17 "$PRIVATE/proposal.json" \
  --release-review "$PRIVATE/release-review.json" --format markdown > "$PRIVATE/q16-q17.md"
```

El paquete registra el SHA de los bytes de `release-review.json`, el SHA
Git y el del archivo candidato. Solicita una sola adopción desde
`release_20261029T185008966364Z_e9669ec` en `SCHEMA_CONTRACT_EPOCH = 2`.
El origen lleva once columnas de recordatorios. La migración preserva
filas y agrega `scope='private'` y `delivery_room=NULL`. L4 recupera el
runtime no gestionado y su estado congelado. Su plazo vence 14 × 24 h tras
el primer cierre limpio W9. Para ejecutar L4 se necesita la decisión
específica del titular con la frase literal **revivir datos borrados**.

Guardar por separado las decisiones Q16 y Q17 recibidas del titular. El
recibo JSON registra `question`, `decision: "approved"`, `package_sha256`
y `approved_at` con zona horaria previa al inicio. La evidencia referencia ese
mismo objeto con `receipt_sha256` añadido, calculado sobre el archivo del
recibo. El custodio confirma la autoría y conserva el original autorizado.
El Markdown y el JSON renderizados llevan los mismos datos privados y el
mismo SHA-256 canónico. Revisarlos y compartirlos por el canal privado acordado.

## Operación y puntos de parada

1. Verificar las precondiciones de W0, el artefacto transferido, el contenedor
   legado completo y el expediente Q17. Ejecutar la adopción con el guion de
   release P6. Registrar SHA del slot activo y READY. Si cambia el candidato
   o alguna intervención, volver a obtener la autorización correspondiente.
2. Ejecutar T070 en las cuatro entradas y con las selecciones reales Q16.
   En Cocina contrastar físicamente el XVF3800 del puerto `7-5.3`, el canal
   estimulado, el canal observado, el puntero de captura y el heartbeat.
   Bloquear si una fuente no opera o si falla el contrato de una integración.
3. El custodio separa días UTC de calibración y reserva, verifica los recibos
   antes de abrir artefactos y conserva los sellos y resultados originales.
   T071 congela el candidato y T072 abre la reserva con los mismos sellos.
   Ante revocación, expiración, discrepancia o estrato fallido, detener la
   evaluación afectada y preservar el resultado no sensible. Una
   reevaluación usa un nuevo sello y una reserva independiente adecuada.
4. Ejecutar T073 en el iPhone físico con el navegador elegido y en Chrome
   de Terra. Registrar cada intento, las notas manuales de accesibilidad y
   las observaciones de los tres participantes. Tramitar la verificación
   física de navegador por el especialista KZA de UI.
5. Terminar el bloque fijo T074 dentro de siete días. Situar las inyecciones
   fuera del bloque y seguir F1–F4 en ese orden. F3 exige una ventana propia
   con el titular presente. El operador registra huecos, pausas autorizadas,
   fallos espontáneos y la recuperación de cada dependencia.
6. Decidir T075 con la evidencia completa. Registrar el primer cierre limpio
   W9 en UTC y su vencimiento, conciliando censo V4, restore compatible y
   ensayo de L4. Si se vuelve al legado, conservar la decisión y su
   resultado operativo. La elegibilidad para T082 exige promoción del
   candidato y evidencia de campo del mismo SHA.

## Forma de la evidencia

Cada JSON incluye `task` (por ejemplo `T070`), `kind: "field"`, `candidate`,
`request` (el objeto `package` renderizado), `q17_approval`, `observed_at`
en ISO 8601 con zona horaria y `evidence_sha256` del registro de observación.
Los hashes de archivos se calculan sobre bytes. Los sellos canónicos JSON
usan UTF-8, Unicode literal, claves ordenadas, separadores sin espacios y
orden de arrays preservado, según p6-v1. Usar referencias opacas de sujetos,
operadores, sesiones y dispositivos. Conservar audios, contenido de
calendarios y listas, grabaciones, credenciales y todos los datos del hogar
en el ámbito privado y sujeto al vencimiento aplicable.

| Tarea | Contenido |
| --- | --- |
| T070 | `q16_approval`, objetos `microphones` por habitación con puerto, canal, estímulo, prueba de captura y `heartbeat` sano durante 300 s. `integrations` por I01–I05 con `selection` igual a Q16, `source_kind: "real"`, `simulated: false`, operación observada, pruebas del contrato, rechazo, caducidad, permisos y `observation_sha256`. Registrar `idempotency_passed: true` para I02 e I04, reconciliación de I02, lectura de I03, controles de I04 y sólo señales técnicas en I05. |
| T071 | `clips` de calibración, `receipts`, `frozen_at`, `reserve_opened_at`, `evaluation_close`, `seals` y su digest `acoustic_sha256`. Mínimo tres voces con diez clips claros cada una. La vigencia termina con el primero de captura + 30 días, cierre, caducidad original y revocación. |
| T072 | Mismos sellos de T071, `protocol: "p6-v1"`, `protocol_sha256` del documento de P6, inventario sellado completo en `clips` y recibos vigentes. `plan` fija panel, condiciones, familias e inventario de muestreo. Cada intento conserva el ID y estrato del plan. Los campos de observaciones se describen abajo. |
| T073 | `accessibility` con alcance preregistrado, informes automático/manual, teclado, foco y lector. `plan` sellado por `plan_sha256`, tres `participants` con diez intentos cada uno y `clients.iphone`/`clients.terra_chrome` con versión, reflow y tiempos. |
| T074 | `soak` con inicio/fin preregistrados, márgenes del muestreador, huecos, pausas, fallos espontáneos y `useful_seconds` para KZA y cada habitación. `fault_plan` fija IDs e incluye SHA del guion. `faults` conserva HA, WS y F1–F4 con instantes, métodos, trazas y disponibilidad por entrada sana. |
| T075 | `outcome: "promoted"`, `adoption` con contrato Q4, único cambio, persistencia y recibo, `census` V4 con 32 productores, sin huecos y con `global_inventory_complete: true`, `restore` compatible con tombstones y sin resurrección/reenvío. `rollback` conserva el contrato L4, sello del cierre limpio, `day_zero`, `deadline` y ensayo. Adjuntar `promotion_decision_sha256` y `integration_recheck` con `selection` igual a Q16, `source_kind: "real"` y `simulated: false` para cada una de las cinco integraciones. |

### Inventarios y recuentos de T071/T072

Usar `clip_id`, `session_id`, `split`, `start`, `end`, `speaker_refs`,
`consent_ref`, `audio_sha256` y `source_sha256`. Cada recibo usa `id`,
`session_id`, `speaker_refs`, `purpose: "consented_calibration_and_evaluation"`,
`granted`, `granted_at`, `expires_at`, `revoked_at` (nulo hasta revocación) y
`receipt_sha256`. Marcar clips de calibración con `clear` y
`deletion_checked`. T072 incluye calibraje y reserva previamente asignados
por el custodio, y fuentes de entrenamiento cuando correspondan. Separa
días UTC, genealogía y sesiones según p6-v1. En `seals` registrar
`runtime`, `models`, `dsp`, `configuration`, `corpus_contract`,
`exclusions`, `instrumentation`, `reserve_inventory` (digest canónico de
`clips` en T072) y `strata_plan` (digest canónico de `plan`).

En `plan.scheduled`, indexar cada sección por ID de intento, con `room`,
`distance_m`, `participant`, `condition` (`clear`, `noise`, `tv` y
adversos admitidos). Incluir `registered_at`, `panel`, `conditions`,
`critical_conditions`, `task_families`, `simple_families` y
`h3_inventory_sha256` (digest canónico de `h3.clips`). Cada observación
repite la asignación sellada y agrega los resultados. El mínimo de diez
se aplica a cada marginal cubierto por el recorrido. Preregistrar más
intentos cuando haga falta para satisfacer los mínimos de habitación,
distancia, participante o condición sin exigir su factorial completo.

- `requests` añade `family`, un `recognition` alineado con palabras,
  caracteres y conteos de sustituciones, inserciones, deleciones y edits
  por carácter, o nulo cuando falta hipótesis verificable. Registrar
  `effect_verified` y `effect_sha256` para cada efecto confirmado. El ID
  de petición enlaza el clip reservado. Publicar los resultados macro,
  CER y desgloses junto con el WER micro de cada estrato.
- `wakes` registra `outcome` (`accepted`, `rejected`, `uncertain`).
  `wake` registra `clock: "house_clock"`, segundos declarados, intervalos
  útiles con las cuatro entradas simultáneas, conteos originales por
  habitación y emisiones con ID, habitaciones y adjudicación. Una emisión
  verdadera enlaza el intento con `correlation_sha256` y la habitación
  observada. Deduplicar sólo con corroboración física.
  Los inciertos cuentan en las cotas del FRR y falsos wakes.
- `simple` añade `family`, `cache` (`cold`/`hot`), `rapid_ha` y latencias
  `voice_ms` y `ha_ms` donde apliquen. `deliveries` registra estado y
  observación de cola, PCM y final audible. Informar
  `voluntary_cancellations` por separado. `interruptions` registra
  `stop_ms` y barge-in físico con captura concurrente, incluida respuesta
  larga no cacheada. Mantener el endpoint ausente como nulo en la serie.
- `listening_pairs` añade `length` (`short`/`long`) y `ratings` por
  participante con `first_variant`, `reference`, `candidate` y
  `severe`. Sellar el lote antes de revelar el mapa A/B y aplicar el orden
  definido en p6-v1. La mediana del candidato se comprueba por oyente.
- `h3.clips` declara ID, habitación, distancia y condición. `h3.runs`
  incluye `original` y cada semilla `"0"`–`"9"`, con `started_empty` y
  `assignments` en orden de ejecución. Cada asignación registra ID de
  clip y `group` (nulo para abstención). Conservar métricas y abstenciones
  por corrida en el registro revisable.

### Medición web y recuperación

En T073 sellar `plan.user_tasks` por participante y `plan.timing` por
cliente y estado `cold`/`hot`. Declarar `iphone_browser` en el plan y
`widths` con claves 320, 390, 768, 1280 y 1440 por cliente. Registrar
pintura útil y audio reproducible con gesto nativo, además de saltos de
objetivo. En iPhone, adjuntar grabación física de 120 o 240 fps, dos
observadores y dos recuentos de frames para cada medida de contenido,
interacción, feedback y preview. Usar la cota superior. CLS se mide cuando
existe API fiable. Terra registra LCP, INP, CLS, feedback y preview. Los
intentos fallidos quedan en el lote y requieren resolución. Cada cliente
lleva al menos 40 intentos por estado frío y caliente.

En T074, computar la disponibilidad desde el reloj fijo y conservar los
segundos del muestreador sin cobertura en el denominador de cada serie. En
cada falla registrar `start`, `end`, `dependency_healthy_at`,
`recovered_at`, `sampler_start`, `sampler_end`, `trace_sha256` y
`healthy_inputs` con cobertura, edad máxima y staleness máxima. La
recuperación se mide desde dependencia sana (incluida reenumeración de mic)
hasta el primer efecto o captura verificable.

```bash
for TASK in t070 t071 t072 t073 t074 t075; do
  "$PY" -B -m tools.p7_field_protocol "validate-$TASK" "$PRIVATE/$TASK.json" || break
done
```

Cada ejecución devuelve 0 con `evidence_valid` o 1 con el control fallido
en stderr. El dictamen mecánico se entrega con los registros originales al
revisor de P8. El revisor confirma procedencia, atribución física, identidad
del artefacto, cobertura y autenticidad de las decisiones.
