# Modo release: transiciones gestionadas después de R1 (SC-5b)

Runbook del operador para cambiar el código que sirve producción después de
R1: una release nueva, un arreglo (hotfix), un rollback o un roll-forward.
Diseño: `sc-release-design-v4.json` (aceptado el 2026-11-15, enmiendas A1 y
A2), `operator_procedure`, `crash_table` y la unidad SC-5b. Decisiones del
usuario: SC-D2, SC-D4, SC-D5, SC-D6, SC-D7 y SC-D8 = A
(`u3b_sc_release_mode_questions`), SC4B-Q1 y SC4B-Q2
(`sc4b_questions_20261116`) y SC4C-Q1, SC4C-Q2b y SC4C-Q6, todas en
`data/autonomous-roadmap-20261102/t040-t041/user-decisions-20261110.json`
(local, fuera de Git). Los comandos y códigos de rechazo son los de
`scripts/stage_runtime.py`, `scripts/compaction.py` y
`scripts/managed_launcher.py`; `tests/unit/tools/test_sc5b_docs.py` los
compara con el código.

Estado: **desarrollado, sin release y nunca ejecutado.** Producción sigue en
la release operativa; R1 (época 4) todavía no se instaló y su ventana sigue
sin programarse. Este runbook no autoriza ni programa ninguna transición: la
programa el usuario, como operador de `kza`.

Las piezas de la mecánica están en el runbook de compactación
([2026-11-14-owner-compaction.md](2026-11-14-owner-compaction.md), «Pasos» y
«Salidas del modo release después de V») y en DEPLOYMENT
([«Procedimiento del modo release (SC-5b)»](../architecture/DEPLOYMENT.md)).
Este runbook es el orden completo, de punta a punta, con cada salida.

## Compuerta: cuándo se puede usar

Todas estas condiciones, antes de la primera transición:

- R1 instalado y aceptado (W9, DÍA 0 de P7) según
  [el runbook de R1](2026-11-13-r1-adoption-window.md).
- El modo release revisado de forma independiente (SC-R) y ensayado
  compilado y con forma de producción desde `controller/current` (SC-6b). La
  primera release de época 5 lleva el par `(época, revisión)` y su digest en
  `RELEASED_CONTROLLERS` (`tests/unit/tools/test_sc1a_controller_install.py`).
- **Primera transición dentro de los 14 días de L4 (SC4B-Q1).** La primera
  transición gestionada después de R1 termina, con su retiro, antes de DÍA 0
  + 14 × 24 h y antes de que exista la intención del retiro Q5. Si cae
  después, no se hace: primero se implementa la salida de la fila PA (testigo
  de cierre por registro de recuperación y `--accept-regressed-source`), que
  hoy no existe en el código. Sin ella, si la release nueva ya aceptada muere
  en cada arranque sin recibo (fila PA), la única salida es L4, y L4 existe
  sólo hasta el retiro Q5.
- **Paso 0 con `resurrected` en 0 (SC4B-Q2)**, abajo. Si no da 0, no hay
  transición: se implementan antes la línea base salada de K y el
  reencolado (R5V-2, R5V-3).
- Un solo contrato de esquema (SC-D4 A): NUEVO declara el mismo
  `schema_contract` que el slot de H y que el controlador instalado. Un
  cambio de esquema se diseña aparte, antes de esa release; `plan` rechaza
  `runtime_admission_compaction_schema_contract`.

## Términos

- **H**: el head cerrado que sirve hoy. **K**: la copia nueva de H que hace
  `compact start`, con el código NUEVO. **K'**: K después de la corrida V.
- **NUEVO**: el archivo de la release de destino (época 5 o mayor).
- **V**: la corrida de verificación aislada de K (sin micrófono, sin panel,
  sin efectos en la casa), que `serve` detiene sola a los N segundos de
  `READY`.
- **W**: la comparación de K' con K al volver a H (W0: H intacto; W1: clase
  `equal`, `loss_only` o `restrictive_or_unclear`).
- **Triple del controlador**: (`controller_epoch`, `controller_revision`,
  `controller_sources_sha256`) del `release.json`.
- **served(root)**: el archivo de cada log del modo release terminado
  (`complete`) con `verified`, más la release del head actual.
- **Filas** (T0, T0S, T0F, P, S, A0, V1 a V4, AT, AB, SS, HD, CI, RS, RF, RX,
  C, PA, DR, DL, DO): las de `crash_table` del diseño.
- **Aceptación**: el retiro (`compact retire` terminado). Recién desde ahí
  el tráfico de la casa llega a K'.

## Reglas que valen en todo el procedimiento

- **Sólo `controller/current`.** Todo `$ST` es el de `controller/current`,
  salvo el T0, que corre desde el NUEVO extraído. Nunca el `stage_runtime` de
  un slot ni un `controller/<commit>` viejo, y nunca el `$ST` instalado de
  época 4 para el T0: el instalador equivocado no corre ninguna comprobación
  de la primera instalación (DEPLOYMENT, «Instalador equivocado»).
- **Venv.** Desde que se detiene la unidad hasta que el log termina
  (`complete` o `aborted`), no cambiar `/home/kza/kza/.venv` (nada de `pip
  install` ni `pip upgrade`): cada binding guarda la versión de chromadb de
  ese venv. La única excepción es la salida de `sdk_changed`: reinstalar la
  versión ligada que nombra el rechazo y repetir.
- **Repetir no daña.** Todo rechazo de W0 (y todo rechazo con código del
  abort con W, del retiro y del plan) no escribe nada; el retiro sólo agrega
  el `verified` de siempre. Repetir el mismo comando es seguro.
- **Nunca `adopt` con un log del modo release pendiente.** Desde `compact
  start` hasta que el log termina, `adopt` y `adopt --dry-run` rechazan
  `runtime_admission_compaction_pending`. Un K' abierto (fila V2, o SS
  después de V2) se maneja sólo con `compact status` y `compact abort --to-h`.
- **No reiniciar el servidor** durante una transición.
- **Presupuesto de bajada (SC-D2 A).** Como mucho 10 minutos sin servicio
  por transición, desde que se detiene la unidad hasta el `READY` del head
  nuevo (o de H después de un abort sin pérdida), con el T0 y su arranque de
  H incluidos. Con los valores medidos hoy (T_copy ≈ 146 s, arranque hasta
  `READY` B ≈ 146 s) la primera transición, con T0, usa N = 10. Anotar las
  horas de la parada y del `READY`; si pasan los 10 minutos, se anota y el
  diseño se revisa antes de la próxima transición. Las paradas SS, HD, CI,
  T0F y PA quedan fuera del presupuesto: van por la regla de 48 horas.
- **Regla de 48 horas.** Ver «Decisión escrita cada 48 horas» al final. Vale
  para todo período con la unidad deshabilitada.

## Construir NUEVO (bajo `yo`)

1. Construir con `scripts/build_runtime.py` desde el commit de la release.
   Su `release.json` declara el triple del controlador. La clausura del
   controlador incluye casi todo `src` (unas 300 fuentes, revisión SC-R
   SCR-C-3): **toda release que cambia `src` cambia el triple y necesita un
   T0 con `CONTROLLER_REVISION` subido.** El mismo `(época, revisión)` con
   otra clausura no se instala (`runtime_admission_controller_revision_conflict`).
2. En el mismo commit, agregar el par `(época, revisión): digest` a
   `RELEASED_CONTROLLERS`; desde ahí un cambio de la clausura sin subir la
   revisión falla en los tests antes que en `install-controller`.
3. **Un arreglo después de un rollback se construye sobre la línea del
   controlador instalado** (R5-3): su triple tiene que ser igual o mayor que
   el instalado. Nunca subir sólo la revisión de un arreglo hecho desde la
   línea vieja: `plan` lo rechaza
   (`runtime_admission_compaction_controller_behind`) y, si se forzara con un
   T0, instalaría código de controlador viejo sobre el nuevo.
4. Anotar el sha256 del archivo y transferirlo a `/home/kza/releases/`.
   **Conservar cada archivo de served(root), con su sha256, mientras se
   quiera poder volver a esa release** (R4-OPS-10c): el retiro destruye el
   runtime del slot anterior, así que un rollback sólo puede usar el archivo.

## Variables (cuenta `kza`)

```bash
PY=/home/kza/kza/.venv/bin/python
ST="$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline"
NEW_ARCHIVE=/home/kza/releases/<archivo de la release NUEVA>.tar.gz
NEW_SHA256=<sha256 anotado al construir NUEVO>
K=<nombre nuevo de slot, nunca usado antes>
```

Verificar y extraer NUEVO (una sola vez por sesión; si la sesión se pierde,
repetir las variables y este bloque):

```bash
if [ "$(sha256sum "$NEW_ARCHIVE" | cut -d' ' -f1)" = "$NEW_SHA256" ]; then echo "NUEVO fijado: OK"; else echo "NUEVO distinto: DETENER"; false; fi
unset NEW
NEW=$(mktemp -d) && tar -xzf "$NEW_ARCHIVE" -C "$NEW" && echo "NUEVO extraído: OK"
```

Sin `NUEVO fijado: OK` y `NUEVO extraído: OK`, detener: no hay transición.
`$NEW` sólo tiene código; borrarlo (`rm -rf "$NEW"`) al terminar.

## Pasos de una transición

### Paso 0: residuo de resurrección (sólo lectura, el servicio sigue en H)

```bash
$ST completion-status
```

Cada valor de `resurrected` tiene que ser 0 (respuesta SC4B-Q2 del usuario,
2026-11-16). El retiro exige K' sin ningún id resucitado (escaneo absoluto),
y K' es una copia de H: con un valor distinto de 0 la transición pararía en
el retiro, después de toda la bajada, y volvería a H sin pérdida. Si no da
0, no seguir: el servicio queda en H y se pasa a desarrollo.

Justo antes del paso 1: `live_capture_owners` en 0 (R5V-1; runbook de
compactación, «Antes de empezar»).

### Paso 1: detener y deshabilitar

```bash
systemctl --user stop kza-voice.service
systemctl --user disable kza-voice.service
```

Salida 0 obligatoria en cada uno. Desde acá corre el presupuesto de 10
minutos y vale la regla del venv.

### Paso 2: ¿hace falta T0? (sólo lectura)

```bash
"$PY" -I - /home/kza/releases/controller/current/release.json "$NEW/release.json" <<'PYEOF'
import json, sys
def triple(path):
    with open(path) as handle:
        manifest = json.load(handle)
    return (manifest.get("controller_epoch", 0), manifest.get("controller_revision", 0)), manifest.get("controller_sources_sha256")
(installed, a), (new, b) = triple(sys.argv[1]), triple(sys.argv[2])
print("instalado", installed, a)
print("NUEVO", new, b)
print("T0" if new > installed else "sin T0" if new < installed or a == b else "conflicto: no seguir")
PYEOF
```

- `T0`: el triple del NUEVO es estrictamente mayor (por época y revisión).
  Siempre en la primera transición después de R1 (R1 es época 4). Seguir con
  el paso 3.
- `sin T0`: el mismo triple, o uno menor (un rollback a una release de
  served(root)). Un rollback no necesita T0: `install-controller` lo
  rechazaría como bajada (`runtime_admission_controller_downgrade`). Saltar
  al paso 4.
- `conflicto: no seguir`: la misma época y revisión con otra clausura. No
  hay transición con ese archivo (`plan` rechazaría
  `runtime_admission_compaction_controller_revision_conflict`): habilitar y
  arrancar la unidad (sirve H) y reconstruir NUEVO con la revisión subida.

`plan` repite esta comprobación con el triple que lee él mismo: un T0
faltante rechaza `runtime_admission_compaction_t0_required` con los dos
triples.

### Paso 3: T0, instalar el controlador del NUEVO (sólo con `T0`)

```bash
"$PY" -I -B "$NEW"/scripts/stage_runtime.pyc --root /home/kza/releases --offline install-controller "$NEW_ARCHIVE"
```

Desde el NUEVO extraído, nunca desde `$ST`: cada módulo de la clausura que
corre la instalación tiene que ser el del archivo
(`runtime_admission_controller_installer_mismatch`). Un `.pyc` no es
ejecutable solo: siempre `"$PY" -I -B`.

Rechazos antes de mover `current` (fila T0; no escriben nada):

| Código | Causa | Salida |
| --- | --- | --- |
| `runtime_admission_compaction_pending` | un log de compactación pendiente | terminarlo (con su runbook) y repetir |
| `runtime_admission_compaction_log_invalid` | un log ilegible | «Compactación trabada» del runbook de compactación |
| `runtime_admission_not_closed` | head abierto o desconocido | cerrarlo con R-b (arrancar la unidad una vez) o `adopt` (sin log pendiente lo admite; después de `adopt --dry-run`, la comprobación de su copia en claro de la fila T0F, punto 2), y repetir |
| `runtime_admission_delegation_frozen` | una entrada en `admission/delegations/` | filas DL/DO, abajo |
| `runtime_admission_owners_present` | un head con dueños | fila DO, abajo |
| `runtime_admission_controller_binding_changed` | SR-1: el código nuevo calcula otro binding para el slot de H | no instalar ese NUEVO; habilitar y arrancar (sirve H) y pasar a desarrollo |
| `runtime_admission_compaction_sdk_changed` | SR-1: sólo difiere `sdk_versions` | reinstalar la versión ligada que nombra y repetir |
| `runtime_admission_controller_schema_contract` | otro contrato de esquema | no hay transición con ese NUEVO (SC-D4 A) |
| `runtime_admission_controller_downgrade` | triple menor | no es un T0: es un rollback (paso 4 sin T0) |
| `runtime_admission_controller_revision_conflict` | mismo par, otro digest | reconstruir NUEVO con la revisión subida |
| `runtime_admission_controller_tampered` | `controller/current/release.json` ilegible | trabajo de desarrollo, servicio caído, regla de 48 horas |

En cualquiera de esos rechazos el controlador anterior sigue en `current`:
corregir la causa y repetir, o habilitar y arrancar la unidad (H sigue
sirviendo con el controlador anterior).

**Corte entre el cambio de `current` y `controller.json` (fila T0S).** Todo
lo que no sea `install-controller`, `serve` incluido, rechaza
`runtime_admission_controller_install_incomplete`. Repetir el mismo comando
de instalación con el mismo archivo: vuelve a correr las mismas
comprobaciones y termina.

**Comprobar que H sirve con el controlador nuevo, una vez:**

```bash
systemctl --user enable kza-voice.service
systemctl --user start kza-voice.service
systemctl --user show -p ActiveState kza-voice.service   # ActiveState=active: H llegó a READY
systemctl --user stop kza-voice.service
systemctl --user disable kza-voice.service
```

Después de servir H una vez, **detener y deshabilitar otra vez antes del
plan** (`plan` rechaza `runtime_admission_not_closed` mientras H corre).
Mientras la unidad esté `activating`, no correr `enable` ni `disable` (recargan
el gestor; runbook de R1, W8).

Esta comprobación se saltea **sólo** en dos casos, y en los dos SR-1 es la
única compuerta de la instalación:

- cuando H es un head de recuperación de una release que se sabe que falla
  (fila PA, más abajo): ese H no llega a `READY` con ningún controlador, y su
  falla parecería un T0F;
- en la fila CI (un controlador corregido con un log del modo release
  pendiente, más abajo): ahí H no es el head y arrancar la unidad correría la
  recuperación de `serve` sobre K'.

**Sin vuelta atrás (fila T0F).** Una vez que `current` cambió, no hay regreso
al controlador de época 4: una copia de época 4 no toma el lease de una raíz
de época 5 (`runtime_admission_controller_downgrade`) e `install-controller`
rechaza un triple menor. Si H no llega a `READY` con el controlador nuevo:

1. Unidad detenida y deshabilitada; el servicio queda caído (regla de 48
   horas, una copia, borrados por plazo en pausa).
2. Si la corrida que falló dejó el head de H abierto (sin recibo, y R-b no
   lo cerró), cerrarlo primero con el controlador instalado: R-b acotado (un
   arranque más de la unidad) o `$ST adopt --dry-run` y `$ST adopt --confirm
   <head_sha256>`. `install-controller` rechaza un head desconocido
   (`runtime_admission_not_closed`). Sin log pendiente, `adopt` y `adopt
   --dry-run` corren, también acá, en la primera transición, y el dry-run
   copia todo el estado en claro a un directorio temporal del sistema
   («Después de la aceptación»). Después del dry-run, cortado o no,
   comprobar que no quedó ninguna copia:

   ```bash
   ls -d "${TMPDIR:-/tmp}"/kza-adopt-dry-run-*   # sin salida: no quedó ninguna
   ```

   Si aparece alguna, borrarla a mano (`rm -rf` sobre esa ruta exacta) antes
   de seguir: está fuera de todo retiro y de todo borrado.
3. Salida: un controlador corregido con una revisión mayor (otro T0 sobre el
   head cerrado, con SR-1), hecho por desarrollo; o L4 mientras exista (sin
   log pendiente su compuerta lo admite), con la frase y la decisión que pide
   el runbook de R1.

### Paso 4: plan (sólo lectura; se puede repetir)

```bash
$ST compact plan "$K" --archive "$NEW_ARCHIVE"
```

En el modo release `--retire-prepare-leftovers` queda forzado. Revisar, sin
datos personales en la salida:

- `members`, `untouched` y `coverage`: como en el runbook de compactación
  (paso 2). Con `unexplained > 0` en `coverage`, `start` exige
  `--accept-owner-only-loss`: esos datos se pierden.
- `capacity.need_bytes` (y `temp_need_bytes`): el espacio libre del pico
  medido.
- `chroma` (SC-4c, revisión del plan del modo release): por almacén de
  Chroma de H, `rebuilt`, `copied`, `dropped`, `dropped_collections` y
  `other_files`. Lo esperado en producción:
  - `data/chroma_db`: `dropped_collections` `["home_assistant_routines"]`
    (el índice de rutinas dictadas; el arranque de K lo recrea vacío, así
    que **en cada transición** «listar mis rutinas» por voz queda vacío
    hasta que se dicte una rutina nueva; las rutinas programadas de
    `routines.db` no cambian), `copied` según el índice de comandos;
  - `data/memory_db`: `rebuilt: 1`, `dropped: 1` y `dropped_collections`
    `["user_memories"]` (las 876 memorias heredadas en cuarentena: K no las
    recibe y se borran para siempre cuando el retiro destruye H);
  - los dos con `other_files: 0`. Si `other_files` no es 0, mirar qué se
    dejó a mano dentro del almacén antes de seguir (SC4C-Q5): K no lo recibe.
  Cualquier otra colección hace rechazar el plan
  (`runtime_admission_compaction_unlisted_collection`, con el almacén y el
  nombre).
- `plan_sha256` y `untouched_sha256`, para el paso 5.

Rechazos del plan (fila P: no escribe nada; arreglar la causa y repetir, o
habilitar y arrancar la unidad, que sirve H):

| Código | Qué hacer |
| --- | --- |
| `runtime_admission_compaction_t0_required` | falta el T0 del NUEVO: paso 3 |
| `runtime_admission_compaction_controller_behind` | triple menor y fuera de served(root): un arreglo de una línea vieja; rehacerlo sobre la línea instalada |
| `runtime_admission_compaction_controller_revision_conflict` | mismo par, otra clausura: reconstruir con la revisión subida |
| `runtime_admission_compaction_release_epoch` | NUEVO de época menor que 5 (R1 nunca es destino, SC-D7 A) |
| `runtime_admission_compaction_schema_contract` | otro contrato de esquema (SC-D4 A) |
| `runtime_admission_launcher_contract_missing`, `runtime_admission_rollback_format_closed` | NUEVO no declara el contrato de lanzador o el formato que hacen falta |
| `runtime_admission_compaction_release_unproven_copy` | una entrada sin prueba bajo `slots/`: un `slots/.prepare-*` vacío se quita con `rmdir` y se repite; cualquier otra no se borra (servicio en H, averiguar, regla de 48 horas) |
| `runtime_admission_compaction_unlisted_collection` | una colección de Chroma fuera de la lista cerrada |
| `runtime_admission_compaction_capacity` | falta espacio para el pico |
| `runtime_admission_delegation_frozen`, `runtime_admission_compaction_owners_present` | filas DL/DO |
| `runtime_admission_not_closed` | la unidad sigue corriendo o el head no está cerrado |

### Paso 5: start

```bash
$ST compact start "$K" --archive "$NEW_ARCHIVE" --plan-sha256 <plan_sha256> --acknowledge-untouched <untouched_sha256> --destroy-rollback-states
```

Agregar `--accept-owner-only-loss` sólo si el plan lo pidió, y las mismas
`--unit` que en el plan. La salida trae `compaction_id` y `next: compact
verify <id>`. `runtime_admission_compaction_plan_changed`: repetir el plan.

Anotar el binding de H para `--confirm-h` (toda vuelta a H lo pide, quizá
días después). Ni el plan ni `compact status` lo muestran; está en el
registro de intención del log:

```bash
"$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_binding_id"])' /home/kza/releases/admission/compaction/v1/<compaction_id>/intent.json
```

**Start cortado o rechazado por el trabajador (fila S)**, por ejemplo una
colección fuera de la lista cerrada: `$ST compact resume <compaction_id>
--archive "$NEW_ARCHIVE"` sigue desde el último paso, o `$ST compact abort
<compaction_id>` (el abort simple, sin `--to-h`) vuelve a H sin pérdida.
Después de un abort simple, habilitar y arrancar la unidad.

### Paso 6: corrida de verificación V

```bash
$ST compact verify <compaction_id> --smoke-seconds <N>
systemctl --user start kza-voice.service   # la unidad sigue deshabilitada
systemctl --user show -p ActiveState kza-voice.service   # esperar ActiveState=inactive: serve la detuvo sola
```

N va de 10 a 60 y es obligatorio
(`runtime_admission_compaction_release_smoke_seconds`): N = min(60, 600 −
T_copy − arranques × B − paradas); la primera transición, con T0, usa 10. No
habilitar la unidad, no detenerla a mano y no usar el panel: `serve` arranca
NUEVO sin panel, micrófono, rutinas ni recordatorios, mide N segundos desde
el `READY` reconocido y lo detiene.

### Paso 7: estado

```bash
$ST compact status <compaction_id>
```

Muestra `phase`, `armed_run` (`not armed`, `not consumed` o `consumed`),
`verified` y `verify_branch_reached` (`yes`, `no`, `not run` o
`unreadable`). No calcula W ni escribe nada.

- `verified: true`: paso 8.
- `armed_run: not consumed` (la corrida armada no corrió: `serve` murió antes
  de empezarla, fila A0): arrancar la unidad otra vez para V (paso 6, sin
  repetir `verify`), o el abort simple `$ST compact abort <compaction_id>`,
  sin pérdida. `--to-h` acá rechaza
  `runtime_admission_compaction_abort_to_h_before_run`. Después de un abort
  simple, habilitar y arrancar la unidad.
- `armed_run: consumed`, `verify_branch_reached: yes` y `verified: false`:
  V pudo salir bien con el registro de `verified` diferido (`serve` lo anota
  después del sello; si esa escritura falla sólo deja `compaction verified
  deferred` en su log). Correr primero el retiro, paso 8, no el abort: `$ST
  compact retire <compaction_id>` registra él mismo el `verified` si K'
  cerró limpio, y si no rechaza `runtime_admission_compaction_not_verified`
  (fila V3) sin escribir nada; recién ahí volver a H con W. Volver a H
  primero tiraría una V buena y la transición repetiría su bajada.
- Cualquier otro resultado (no verificada: `verify_branch_reached: no`,
  intervalo corto, V muerta con o sin recibo): «Volver a H con W», abajo.
  No arrancar la unidad otra vez para «probar»: `serve` rechaza
  `runtime_admission_compaction_verify_consumed`.

### Paso 8: retiro (la aceptación)

```bash
$ST compact retire <compaction_id>
```

Rutear un rechazo por su código. Ningún rechazo escribe nada más que el
`verified` de siempre, salvo un `runtime_admission_compaction_io_error`
después de `retire_started`: ése es un retiro cortado (fila RS).

| Código | Fila | Salida |
| --- | --- | --- |
| `runtime_admission_compaction_identity` | RF | revisar; sólo si es seguro: `$ST compact retire <compaction_id> --member <n> --accept-identity-change` |
| `runtime_admission_compaction_member_missing` | RF | devolverlo a su ruta, o `$ST compact retire <compaction_id> --member <n> --accept-missing` |
| `runtime_admission_compaction_owner_only_data` | RF | leer las cuentas (`new_unexplained`); aceptar que esos datos se destruyen con H: `$ST compact retire <compaction_id> --accept-owner-only-loss <coverage_sha256>`; si no, volver a H con W |
| `runtime_admission_compaction_plan_changed` | RF | una entrada nueva en la raíz (`untouched_added`): quitarla o decidir y repetir |
| `runtime_admission_compaction_retire_protected_changed` | RX | volver a H con W (el informe W viene en la salida) |
| `runtime_admission_compaction_retire_resurrected` | RX | volver a H con W |
| `runtime_admission_compaction_k_binding_changed` | RX | (`privacy_authority_stale`) volver a H con W |
| `runtime_admission_compaction_io_error` | RS o AT | según la fase: mirar `phases` en `$ST compact status <compaction_id>`. Con `retire_started`, el retiro ya empezó (fila RS) y el abort rechaza `runtime_admission_compaction_phase`: quitar la causa del `errno` y repetir `$ST compact retire <compaction_id>` o `$ST compact resume <compaction_id>`, nunca volver a H. Sin `retire_started`: quitar la causa del `errno` y repetir una vez el mismo retiro, como en la fila AT; si rechaza igual, volver a H con W |
| `runtime_admission_compaction_not_verified` | V3 | volver a H con W |
| `runtime_admission_compaction_sdk_changed` | AT | reinstalar la versión ligada que nombra y repetir |
| `runtime_admission_compaction_h_transient`, `runtime_admission_compaction_k_transient`, `runtime_admission_writer_active`, `runtime_admission_busy`, `runtime_admission_sdk_version_unavailable` | AT | «Rechazos transitorios» |
| `runtime_admission_compaction_h_changed` | HD | parada terminal HD |
| `runtime_admission_compaction_head_moved`, `runtime_admission_compaction_k_closure_unreadable` | como HD | parada como HD |
| cualquier otro código (por ejemplo `runtime_admission_compaction_phase`, `runtime_admission_head_changed`, `runtime_admission_compaction_k_delegated`, `runtime_admission_compaction_k_foreign_target`, `runtime_admission_compaction_database_unreadable`, `runtime_admission_compaction_retiring_residue` o un código del lease) | ninguna | no improvisar: la unidad sigue deshabilitada; guardar la salida y la de `$ST compact status <compaction_id>`; entregarlo a desarrollo, con la decisión escrita cada 48 horas |

Un enlace, punto de montaje o archivo especial dentro de un miembro
(`runtime_admission_compaction_symlink`, `..._hard_link`, `..._mount`,
`..._special_file`, `..._cross_device`): quitar la causa a mano (de un
enlace, el enlace, nunca su destino) y repetir el mismo `$ST compact retire
<compaction_id>`, no `compact resume`: antes de `retire_started`, `resume`
no hace nada (devuelve `"noop": true` con la fase `activated`). El mismo
retiro también termina uno ya empezado. **Retiro cortado (fila RS)**: `$ST
compact resume <compaction_id>` o el mismo `$ST compact retire
<compaction_id>` lo termina.

### Paso 9: servir K'

Con el retiro terminado (fase `complete`, fila C):

```bash
systemctl --user enable kza-voice.service
systemctl --user start kza-voice.service
systemctl --user show -p ActiveState kza-voice.service   # ActiveState=active: K' llegó a READY
```

Anotar la hora del `READY` (fin del presupuesto) y borrar `$NEW`. Desde acá
la release nueva está aceptada; un problema después se resuelve con otra
transición (hotfix o rollback), nunca volviendo a H, que ya no existe.

## Salidas después de V

### Volver a H con W (filas V1, V2, V3, V4 no retirable y RX)

```bash
$ST compact abort <compaction_id> --to-h --confirm-h <primeros 12+ caracteres del binding de H>
```

Funciona también con K' abierto (V2): recupera en el lugar un journal o un
WAL de K' (nunca crea bases) y calcula W. Con `equal` o `loss_only` vuelve a
H. Después, habilitar y arrancar la unidad:

```bash
systemctl --user enable kza-voice.service
systemctl --user start kza-voice.service
```

El abort simple después de V rechaza
`runtime_admission_compaction_abort_after_run`; `--discard-k-writes` es sólo
del modo simple (`runtime_admission_compaction_plain_mode_only`).

Un rechazo de arranque del NUEVO antes de `READY` en V (fila V2) da `equal`
y la vuelta es sin pérdida. Si H, ya de vuelta, rechaza igual al arrancar,
es una condición del estado (fuera del modo release, regla de 48 horas), no
HD; si H arranca, el defecto es del NUEVO y se arregla con otra transición.

### Rechazos transitorios (fila AT)

`runtime_admission_recovery_child_alive` (el hijo de V sigue vivo),
`runtime_admission_writer_active` (un proceso tiene un lock en H o en K':
buscarlo y detenerlo), `runtime_admission_busy`,
`runtime_admission_sdk_version_unavailable`,
`runtime_admission_compaction_sdk_changed` (cambió el chromadb del venv
compartido: reinstalar sólo esa versión ligada, por ejemplo `"$PY" -m pip
install --no-deps 'chromadb==<versión ligada>'`, sin tocar nada bajo la
raíz), `runtime_admission_compaction_h_transient` (un errno o un código
SQLite pasajero al leer H, nombrado en la salida),
`runtime_admission_compaction_k_transient` (un lock SQLite o un errno de
recursos en K', por ejemplo una sesión `sqlite3` abierta sobre K') y
`runtime_admission_compaction_abort_confirmation` (prefijo `--confirm-h`
equivocado o corto). **Un error al leer H de estos no es una deriva de H**:
buscar y quitar la causa primero, y repetir el mismo comando, que no escribió
nada. Si sigue igual con la causa corregida (por ejemplo EIO repetido en H:
daño del disco), tratarlo como HD.

### Parada segura SS (`runtime_admission_compaction_abort_protected_changed`)

K' tiene una diferencia protegida que no es sólo pérdida (W1
`restrictive_or_unclear`): una lápida, una revocación, una base de
memoria/privacidad borrada o ilegible, un valor sin clasificar o
`k_writers: unverifiable`. El rechazo imprime el informe W sin contenido y su
digest `w_sha256`. No se destruye ni se revive nada: quedan H, el snapshot
y K' (tres copias), la unidad detenida y deshabilitada y los borrados por
plazo en pausa en las tres. **Dejar la unidad deshabilitada y guardar el
informe** (el estado por base dice si una base está ilegible).

Procedimiento SC-D8 A:

1. Una persona revisa el informe, también la línea de encabezado
   `verify_branch_reached` (rama de verificación alcanzada). `yes` sólo
   prueba que el NUEVO llegó a su rama de verificación: **no prueba que nadie
   usó el sistema**. `no` quiere decir que ni eso quedó probado. Con
   cualquiera de los dos, confirmar sólo por conocimiento directo de que
   nadie usó el sistema durante V; si no, el servicio sigue apagado. Con
   `k_writers: unverifiable`, comprobar además que ningún proceso tiene K'
   abierto (`fuser -v` o `lsof +D` sobre `slots/<K>/state`).
2. Si confirma, **lo deja por escrito** junto al digest, antes del comando:

   ```text
   Confirmación SC-D8 A — <fecha y hora>
   compaction_id: <id>
   w_sha256 del último rechazo: <64 hex>
   verify_branch_reached del informe: <yes|no>
   Confirmo por conocimiento directo que nadie usó el sistema durante V
   (de <hora de arranque de V> a <hora de parada>): <nombre>
   ```

   y corre, con el digest del **último** rechazo:

   ```bash
   $ST compact abort <compaction_id> --to-h --confirm-h <prefijo> --discard-k-changes <12+ hex del w_sha256 del último rechazo>
   ```

   Recalcula W, exige el mismo digest, registra digest y cuentas en
   `abort_started`, destruye K' y vuelve a H. Después habilitar y arrancar
   la unidad. Un digest distinto rechaza
   `runtime_admission_compaction_discard_k_changes_mismatch` sin escribir.
   Mirar la clase del informe W de ese rechazo: con `restrictive_or_unclear`,
   tomar el digest del último rechazo (si no es el que la persona revisó,
   revisar el informe nuevo y confirmar por escrito otra vez); con `equal` o
   `loss_only` no queda nada que descartar y `--discard-k-changes` rechaza
   siempre: sacarlo y correr la vuelta a H sin él, `$ST compact abort
   <compaction_id> --to-h --confirm-h <prefijo>`. Si el borrado de K rechaza
   `runtime_admission_compaction_symlink` (un enlace de lock en K'), quitar
   ese enlace a mano (el enlace, nunca su destino) y `$ST compact resume
   <compaction_id>`.
3. **Si no puede confirmarlo, el servicio sigue apagado** y pasa a
   desarrollo, con la decisión escrita cada 48 horas.

No hay otra salida de SS: ni el abort simple, ni el retiro, ni `adopt`.

### Parada terminal HD (`runtime_admission_compaction_h_changed`)

W0 encontró un cambio en los archivos de H (el binding, el marcador, el
snapshot o la prueba de cierre: un escritor no autorizado, una edición a mano
o daño del disco). El informe trae `"w0": "drift"` y el código de abajo. **No
hay salida por herramienta, tampoco L4**: `abort --to-h` (con o sin
`--discard-k-changes`), el abort simple, el retiro, `adopt` y el
`uninstall-launcher` de L4 rechazan mientras el log esté pendiente
(`runtime_admission_compaction_pending` en L4), y ninguno escribe nada, así
que repetirlos no daña. Qué hacer:

1. Dejar la unidad deshabilitada.
2. No cambiar nada bajo `/home/kza/releases` ni en `/home/kza/kza/.venv`.
3. Guardar el código, el informe y la salida de `$ST compact status
   <compaction_id>`.
4. Buscar una copia vieja de `stage_runtime` que haya corrido o una edición a
   mano, sin borrar nada.
5. Entregarlo a desarrollo, con la decisión escrita cada 48 horas (tres
   copias, borrados por plazo en pausa).

**Paradas como HD**: `runtime_admission_compaction_head_moved` (el head
cerrado no es el sello de V: una corrida de K detrás de la compuerta, por
ejemplo desde un controlador viejo), `runtime_admission_compaction_k_closure_unreadable`
(la prueba de cierre de K' falta o no se lee) y
`runtime_admission_run_record_schema` (el registro de la corrida de un K'
abierto no se lee; revisión SC-R SCR-C-4). Sólo aparecen con daño o una
edición bajo `admission/`: misma entrega que HD.

### Abort cortado (fila AB)

Repetir el mismo `$ST compact abort <compaction_id> --to-h --confirm-h
<prefijo>` o `$ST compact resume <compaction_id>`; después habilitar y
arrancar la unidad. **Nunca sacar nada de `compaction/<id>/retiring/`.**

Si rechaza `runtime_admission_compaction_abort_recheck_required` (el head se
movió después de `abort_started`, por ejemplo se arrancó la unidad y R-a
selló K'), correr otra vez `$ST compact abort <compaction_id> --to-h
--confirm-h <prefijo>`: rehace W0 y W1. Si el abort original usó
`--discard-k-changes`, agregarlo con el digest del **propio rechazo de esta
re-comprobación**; si ese digest no es el que la persona revisó, la persona
revisa el informe nuevo y confirma por escrito antes (R5-OPS-1).

### Controlador corregido con K' abierto (fila CI)

Un defecto fuera de la proyección compartida (por ejemplo en el clasificador
W1 o en un rechazo) con un log del modo release pendiente: quedan tres copias,
la unidad deshabilitada y los borrados en pausa. Desarrollo construye el
controlador corregido con una revisión mayor; se instala desde su archivo
extraído (`"$PY" -I -B <tmp>/scripts/stage_runtime.pyc --root
/home/kza/releases --offline install-controller <archivo corregido>`), que
exige que el binding de H recalculado sea igual al del log
(`runtime_admission_controller_binding_changed` si no, antes de mover
`current`, y el log queda como estaba).

**No es un T0 completo: no hay comprobación de que H sirve.** La unidad
sigue detenida y deshabilitada y no se arranca después de la instalación.
Con el log pendiente H no es el head: arrancarla correría la recuperación de
`serve` sobre K' antes de que la compuerta de compactación rechace, movería
el head (el abort siguiente rechazaría
`runtime_admission_compaction_abort_recheck_required`) y K' recibiría
escrituras que nadie revisó. Después se termina el log con el mismo `$ST`
(ahora el controlador corregido): `$ST compact status <compaction_id>` y la
salida que corresponda. Mientras se espera, la decisión escrita cada 48 horas.

## Rollback (sin T0)

Los mismos pasos 0, 1, 4 a 9, con un archivo anterior de served(root) como
`NEW_ARCHIVE` (verificado contra su sha256 anotado). **Un rollback no lleva
T0**: su triple es menor, `install-controller` lo rechazaría
(`runtime_admission_controller_downgrade`) y `plan` lo admite por estar en
served(root). R1 no es destino (SC-D7 A,
`runtime_admission_compaction_release_epoch`). Los datos van hacia adelante
y el código hacia atrás: nada se reconstruye. Una V fallida de un rollback
vuelve sin pérdida al head aceptado, o para en SS.

Después de un rollback, el controlador instalado sigue siendo el más nuevo:
un arreglo se construye sobre esa línea («Construir NUEVO», punto 3).

## Después de la aceptación

- **La release nueva muere en cada arranque sin recibo (fila PA).** El head
  queda desconocido, sin log pendiente. Cerrarlo con R-b acotado (el próximo
  arranque) o con `adopt`:

  ```bash
  $ST adopt --dry-run
  $ST adopt --confirm <head_sha256 del dry-run> --confirm-regressions <familias exactamente como las lista el dry-run>
  ```

  `--confirm-regressions` sólo si el dry-run informa regresiones.
  `install-controller` rechaza un head desconocido
  (`runtime_admission_not_closed`) hasta que se cierre. **Hoy un head
  cerrado por recuperación no es fuente de ninguna transición** (`plan`
  rechaza: el testigo de cierre por registro de recuperación y
  `--accept-regressed-source` no están implementados). Así que hoy la salida
  de PA es **L4 mientras exista** (SC-D7 A, SC4B-Q1), con el servicio caído
  mientras tanto y la decisión escrita cada 48 horas. Cuando esa parte
  exista: un head de recuperación limpio empieza el hotfix o el rollback
  directamente; con regresiones confirmadas, el plan las liga con
  `--accept-regressed-source`; y un hotfix con triple mayor lleva su T0 con
  la comprobación de H salteada (paso 3).
- **`adopt --dry-run` hace una copia completa en claro.** Corre siempre que
  no haya un log del modo release pendiente: después de la aceptación (fila
  PA) y también durante la primera transición, antes del plan, en la salida
  T0F (punto 2) y en el rechazo `runtime_admission_not_closed` del T0; con
  un log pendiente rechaza. Hoy la copia va a un
  directorio temporal del sistema (`kza-adopt-dry-run-*` bajo `$TMPDIR` o
  `/tmp`) y se borra al terminar; si el dry-run se corta, queda ahí, fuera de
  todo retiro y de todo borrado: comprobarlo con `ls -d
  "${TMPDIR:-/tmp}"/kza-adopt-dry-run-*` y borrarlo a mano. La sombra bajo
  `<raíz>/admission/dry-run/` que el diseño pide (fila DR) es trabajo
  pendiente.
- **`adopt --confirm`** pide el `head_sha256` que imprime `adopt --dry-run`
  (`runtime_admission_adopt_confirmation` si no coincide).
- **Defectos de las reparaciones que V saltea** (SC-D6 A): aparecen recién
  con el tráfico real, después de aceptar; arreglarlos (otra transición) no
  devuelve lo que borraron.

## Registro de delegación después del T0 (filas DL y DO)

`plan` y `serve` rechazan `runtime_admission_delegation_frozen`: corrió una
copia vieja de `stage_runtime`. Investigar a mano cuál fue. Antes de tocar
nada, copiar `admission/delegations/` fuera de la raíz como evidencia.

- **Sólo el registro (fila DL)**: el head es el mismo que sirvió después del
  T0 (el `binding_id` de `admission/head.json` no cambió) y su política no
  lista destinos delegados. Quitar las entradas de `admission/delegations/` a
  mano devuelve `serve`.
- **Dueños en el head (fila DO)**: la copia vieja además activó un head
  delegado (el `binding_id` cambió). **No quitar el registro**: aun sin él,
  `serve` rechaza `runtime_admission_owners_present` y `plan`
  `runtime_admission_compaction_owners_present`. La salida es L4 mientras
  exista y no haya log pendiente, o trabajo de desarrollo, con la decisión
  escrita cada 48 horas. Ante la duda, tratarlo como DO.

`uninstall-launcher` (L4) no queda bloqueado por la guarda.

## Hueco posterior a R1 (SC-D5 A)

R1 sale apenas esté listo y el modo release se empieza enseguida. Hasta que
el modo release esté revisado (SC-R) y ensayado (SC-6b), ningún arreglo se
puede instalar, y desde el día 14 tampoco queda L4. **Si en ese hueco aparece
un problema grave de privacidad, detener y deshabilitar la unidad y dejar el
servicio caído** hasta que el arreglo pueda salir como la primera transición
gestionada; vale la regla de 48 horas (una copia, borrados por plazo en
pausa).

## Decisión escrita cada 48 horas

Para **todo** período con la unidad deshabilitada, sea cual sea la causa: SS,
HD, CI, T0F, una release aceptada rota que espera un hotfix (PA), un paso que
no terminó, o una parada del hueco posterior a R1. Si el servicio no vuelve a
servir dentro de las 48 horas desde que se detuvo la unidad (el plazo más
corto de retención, el audio ambiental), el operador anota una decisión, y
otra cada 48 horas:

```text
Decisión escrita (48 h) — <fecha y hora>
Unidad detenida desde: <fecha y hora>   Causa: SS | HD | CI | T0F | PA | hueco | paso sin terminar
compaction_id: <id o «ninguno»>   Último código: <código>
Copias que se conservan: <tres: H, snapshot y K' (SS, HD, CI) | una>
Borrados por plazo (audio ambiental, vencimiento de hechos, sesiones vencidas, poda de dueños retirados): en pausa en cada copia
Decisión: <la salida que se toma> | <seguir sin servicio hasta ...>
Próxima revisión: <fecha + 48 h>   Responsable: <operador>
```

Se anota en el handoff, en la sección de esa transición.

## Qué no cubre

- Transiciones de esquema (SC-D4 A): se diseñan aparte.
- Raíces con dueños: existen sólo en tests y ensayos de época 4 o menor, se
  recrean y siguen en modo simple; producción no puede tener dueños.
- Una copia fuera de `/home/kza/releases` (respaldos externos, snapshots del
  disco): ningún mecanismo la alcanza. El borrado de la compactación es sólo
  `unlink`.
