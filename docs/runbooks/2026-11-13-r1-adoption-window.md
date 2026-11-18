# Ventana R1: adopción del estado de producción (U3b-0e3)

Procedimiento del operador para la ventana W0–W10 del diseño
`t040-t041/u3b-0e-adoption-design-v3.json` (secciones 6 y 7), con las
decisiones P1–P8 de `user-decisions-20261110.json`. Los comandos están en
`scripts/adoption_commands.py` y se llaman desde `stage_runtime`. U3b-0e5
agrega `retire-legacy-state` (Q5), la comprobación de enlaces del destino de
rollback, el paso del día 14 y el texto de DEPLOYMENT. Este runbook todavía no
autoriza la ventana.

## Compuerta de programación (P7, Q5)

- La ventana **no se programa** hasta que U3b-0e5 tenga implementados,
  revisados y ensayados (T-A17):
  - la comprobación de enlaces del destino de rollback (`_verify_legacy_slot`
    rechaza un enlace del contenedor roto o colgante);
  - el retiro Q5 del estado b0866e8 (registro inmutable que saca el
    contenedor de e9669ec y b0866e8 de L4; excluye `/home/kza/kza/models`, `/home/kza/kza/.venv` y el
    slot R1).
- Estado (2026-11-15): esta compuerta está cumplida. Los dos están
  implementados en U3b-0e5 (`retire-legacy-state` y `_verify_legacy_slot`),
  revisados y ensayados:
  - revisión independiente: la ronda 1 dio «aprobado con cambios»
    (2026-11-13, `t040-t041/u3b-0e5-independent-review.json`); los arreglos
    (85f5b3e, 3648caf, 090e8de, ba98c73 y 268c852, en
    `u3b-0e5-evidence.json`, `review_round`) y las respuestas Q1–Q3 del
    usuario se aprobaron en la ronda 2 (`round_2`, el mismo día), con un
    endurecimiento bajo (R2-1) que no bloquea la ventana;
  - T-A17 compilado en Linux (U3b-0e4): `tests/integration/test_linux_adoption_window.py`
    corre el día 14 con el CLI compilado del archivo bajo systemd de usuario
    real. Pasó en el server bajo `yo` en SC-6a (7 de 7, 261 s, sobre 81ed568,
    0a25ca0 y 9ec7d95; `/home/yo/projects/runs/sc6a` y `sc6a-review-fix`) y
    en la suite batch de R1-B sobre a8daf68, el mismo árbol que c6462e5
    (7 de 7; `/home/yo/projects/runs/full-r1b`).
- Esta compuerta no es lo único pendiente antes de la ventana. Siguen
  abiertas dos revisiones del lado de desarrollo: la re-revisión de la ronda
  de arreglos de SC-6a (cbb509c y 9ec7d95; `t040-t041/sc6a-evidence.json`,
  `review_round_1`) y la revisión independiente de la ronda 1 de arreglos de
  R1-B1/R1-B2 (`t040-t041/r1b1-evidence.json`, `review_round_1` y
  `r1b2.review_round_1`), cuyo código ya va en el R1 fijado. El visto bueno
  visual del panel de R1-B2 sigue con el usuario. Si alguna de esas
  revisiones cambia código, R1 se vuelve a construir y a fijar.
- Después quedan los pasos del operador. Este runbook no autoriza ni programa
  la ventana: la programa el usuario, como operador de `kza`. Antes, el
  usuario registra la autorización de la ventana y los ids P1=D y L4 («Antes
  de la ventana»); el operador transfiere a `kza` el archivo fijado
  (`r1-c6462e5.tar.gz`) y lo verifica con «Archivo R1 fijado». En la ventana, en W7 el usuario aplica el drop-in
  sudo y lo verifica en lectura, y en W8 se mide el arranque real hasta READY
  (la aplicación con sus modelos, que los ensayos no cargan).
- **Paso del día 14**: el borrado Q5 corre desde DÍA 0 + 14 × 24 h (DÍA 0 = el
  sello de cierre de W9, el `ended_at` de esa corrida; Q1 = A). Borra el estado
  de b0866e8 y sus diez copias SQLite del 2026-10-29 en `backups/$C/databases`
  (Q2 = A; alcance y decisiones pendientes en «Día 14»). Responsable: el
  usuario, como operador de `kza`; la autorización ya está registrada (P7 = B,
  `u3b_0e5_retirement`). En W10 se anota en el handoff la fecha exacta y el
  responsable. Hasta ese momento sólo vale L4 hacia el contenedor de e9669ec
  (decisión registrada y frase); después no hay rollback no gestionado y `/home/kza/kza`
  no es un destino (Q3 = A).

R1 se construye desde un árbol que ya contiene U3b-0c (decisión
`u3b_0c_review` D1 = A) y SC-2 (delegación retirada, decisión
`u3b_f8_self_contained_releases`): su `release.json` declara
`controller_epoch` 4, que es lo que `adoption-commit` exige, y así ningún
controlador de época 2 o 3 queda en la raíz gestionada. Un R1 construido antes
de SC-2 (época 3) no sirve: sus propios comandos lo aceptarían e instalarían
un controlador que todavía concede delegaciones (PS-13). Además, toda copia de
`stage_runtime` de 69a79e6 a b0c322d~1 (toda la época 2 y la época 3 anterior
a b0c322d) no comprueba la época y puede conceder y activar otro slot en una
raíz de época 4 (AUTHZ3-1). SC-6a (2026-11-15) reconstruyó R1 desde 81ed568
(época 4, contrato de esquema 1), dos veces bajo `yo` en el server con
resultado idéntico byte a byte (una tercera construcción en la Mac también
coincide), y fijó su sha256 y su commit abajo, en «Archivo R1 fijado», con el
paso obligatorio `sha256sum "$R1_ARCHIVE"` antes de W1. Borró todo R1
anterior y todo archivo, árbol extraído, checkout y copia suelta de
`stage_runtime` construidos desde ese rango, bajo `yo` y en cada ubicación de
transferencia (cada ubicación está en el handoff; del lado de `kza` no había
ninguno, y no hizo falta renombrar nada a `*.superseded-epoch3`). El primer
inventario buscaba sólo por nombre (`stage_runtime.py(c)`, `*.tar.gz`, `.git`)
y no vio cuatro copias sueltas renombradas en la Mac (`sr_head.py` en
`/private/tmp` y en el tmp del roadmap, y dos `stage_runtime*.backup.py`); la
revisión las encontró, se borraron y el inventario se repitió por contenido
(cada archivo, cada miembro de cada archivo comprimido) en la Mac, en el
server bajo `yo` y, sólo leyendo, del lado de `kza`: ninguna copia de ese
rango ni de época 3. El ensayo de adopción corre ahora el `delegate` y el
`activate` de una copia de ese rango sobre una raíz de época 4: todavía
concede y activa, y el test lo fija tal como es. La barrera es correr sólo
`controller/current` y que no quede ninguna copia de ese rango que correr.
R1-B1/R1-B2 (integradas en c6462e5) cambiaron `src/` y el frontend después:
R1 se reconstruyó de la misma forma desde c6462e5 (época 4, contrato de
esquema 1; dos veces en el server, idéntico byte a byte, y una en la Mac,
igual) y el pin pasó a ese archivo. El de 81ed568 quedó al lado, renombrado
`r1-81ed568.tar.gz.superseded-r1b`: no se transfiere ni se usa.

Después de R1 no hay transiciones gestionadas (ningún arreglo de código) hasta
que exista el modo release; el único rollback es L4 durante sus 14 días. Ver
«Sin transiciones gestionadas después de R1 hasta el modo release» en
`docs/architecture/DEPLOYMENT.md`. El usuario respondió SC-D5 = A el
2026-11-15: se acepta ese hueco, R1 sale apenas esté listo, el modo release se
empieza enseguida y, ante un problema grave de privacidad en el hueco, se
apaga el servicio hasta tenerlo. El retiro Q5 del día 14 no se posterga.

Todo corre como `kza` y nunca como root. Cada comando rechaza con salida 78
y `{"refused": <código>}` en estos casos:

- otro usuario, o root;
- una raíz que no sea `/home/kza/releases` real (sin enlaces, con `slots/`,
  sin escritura de grupo u otros);
- una segunda ejecución.

```bash
PY=/home/kza/kza/.venv/bin/python
C=release_20261029T185008966364Z_e9669ec
```

`$C` es el contenedor que sirve `current` desde el 2026-10-29 15:50 (e9669ec,
la publicación de carga de voces). Así lo decidió el usuario el 2026-11-18
(`batch_g_questions_20261118`, «R1 base»), en reemplazo de la premisa de P5
(63ea8d3). El contenedor anterior,
`release_20261029T174830709997Z_63ea8d3`, sigue en `slots/` con los
mismos cinco enlaces. No es destino de L4: `uninstall-launcher` lo rechaza con
`runtime_admission_uninstall_legacy_commit`. Si `$C` no es la release de
e9669ec, la corrida de prueba y W2 rechazan
`runtime_admission_adoption_container_commit` antes de escribir nada.

`$R1` (el árbol extraído, con U3b-0c y SC-2, época 4) y `$ST` se definen
recién en «Extraer R1 verificado», en un directorio nuevo, justo después de
`R1 fijado: OK`. Los comandos de abajo pasan siempre `"$R1_ARCHIVE"`, el mismo
archivo verificado, nunca una ruta relativa ni otra copia.

### Archivo R1 fijado (paso obligatorio antes de W1)

R1 se construyó bajo `yo` con `scripts/build_runtime.py` desde c6462e5, dos
veces y con resultado idéntico byte a byte, y aquí se fija su sha256 y su
commit (SC-6a lo había fijado desde 81ed568; se re-fijó después de
R1-B1/R1-B2).
El valor vive en un commit de documentación posterior al commit construido: un
archivo no puede contener su propio hash, y el archivo no incluye `docs/`.
El archivo queda bajo `yo` en el server, en
`/home/yo/projects/releases-r1/r1-c6462e5.tar.gz` (modo 0600, directorio
0700, con su `SHA256SUMS`). Llevarlo a `kza` es parte de la release y lo hace
el operador; esa copia transferida es la que se verifica aquí.

Antes de extraer `$R1`, antes de la corrida de prueba y otra vez justo antes
de W1, correr exactamente:

```bash
R1_ARCHIVE=<ruta del r1.tar.gz transferido>
R1_SHA256=733b6fc35dc929accc00b78b231092531a7ca131663ccc8b663f9d9e48851b86
R1_COMMIT=c6462e5f2794e21e797f6f8b397f1f3bf911dd29
if [ "$(sha256sum "$R1_ARCHIVE" | cut -d' ' -f1)" = "$R1_SHA256" ]; then echo "R1 fijado: OK"; else echo "R1 distinto del fijado: DETENER"; false; fi
```

Si no imprime `R1 fijado: OK` (sale con 1), **el procedimiento se detiene**:
no extraer, no correr ningún `$ST` ni W1. Comparar el commit de
`release.json` no alcanza: un R1 de época 3 construido antes de SC-2 se acepta
a sí mismo (sus propios comandos lo instalarían) y una reconstrucción distinta
del mismo commit tiene otro sha256. Sólo cuenta el sha256 del archivo
(`R1_COMMIT` sólo sirve después, para comprobar que el árbol extraído viene de
este archivo). El ensayo corre este bloque tal como está escrito
(`tests/mocks/r1_runbook.py`).

### Extraer R1 verificado (justo después de `R1 fijado: OK`)

En la misma sesión de shell, inmediatamente después de `R1 fijado: OK`,
extraer ese mismo `$R1_ARCHIVE` en un directorio nuevo y vacío, y definir
`$ST` sólo desde ahí:

```bash
PY=/home/kza/kza/.venv/bin/python
unset R1 ST
R1=$(mktemp -d)   # directorio nuevo, vacío y privado (0700): nunca una extracción anterior
tar -xzf "$R1_ARCHIVE" -C "$R1"
if [ "$("$PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["git_commit"])' "$R1/release.json")" = "$R1_COMMIT" ]; then ST="$PY -I -B $R1/scripts/stage_runtime.pyc --root /home/kza/releases --offline"; echo "R1 extraído: OK"; else echo "R1 extraído distinto del fijado: DETENER"; false; fi
```

Sin `R1 extraído: OK`, detener: `$ST` queda sin definir. Antes de W3 no
existe `controller/current`, así que el sha256 fijado es la única garantía del
código que corre `$ST`: por eso `$R1` sale siempre de un `mktemp -d` nuevo,
extraído del archivo recién verificado, y nunca de un directorio que ya
existía. El bloque define `$PY` él mismo: sin eso, en una sesión nueva
`$PY` queda vacío y el bloque se detiene con «R1 extraído distinto del
fijado» aunque el archivo sea el correcto. Si la sesión se pierde, repetir el
bloque `PY=`/`C=` de arriba (los comandos de la ventana usan `$C`), «Archivo R1
fijado» y este bloque (otro `mktemp -d`). El ensayo corre los dos bloques tal
como están escritos, sin agregarles nada (sólo cambia el valor de `PY=` por el
intérprete del ensayo).

## Antes de la ventana

- Corrida de prueba consultiva con el servicio en marcha. Es de solo lectura:
  no toma lock ni escribe en la raíz. Usa un scratch en tmpfs
  `/run/user/<uid>` que se borra al terminar, ante un rechazo y ante SIGTERM.
  ```bash
  $ST adoption-dry-run --container-slot $C --archive "$R1_ARCHIVE" --slot <slot R1> \
      --window-timer <timer> ... [--code-index-unit <unidad>]
  ```
  Entrega el censo completo aunque haya rechazos blandos (review F2):
  - `warnings` no bloquea:
    - `advisory: source moving`: el servicio escribió durante la copia
      (`adoption.source_moving` cuenta los archivos); el censo usa los bytes
      copiados. Prepare y commit sí rechazan ese cambio (ahí el servicio está
      parado).
    - `advisory: source moving, retry`: una escritura en vivo dejó una copia
      SQLite inconsistente (`adoption.source_moving.torn`); esa base quedó
      fuera del censo. Repetir la corrida de prueba hasta que `torn` sea 0.
      Parar kza-voice para la corrida solo con autorización registrada del
      usuario. Prepare y commit rechazan la misma situación
      (`quick_check` o `sqlite_corrupt`).
    - `advisory: legacy_training assumed retired (P1=D pending, W1b)`: hay
      archivos LoRA; la corrida los da por retirados y sigue.
      `adoption.legacy_training.would_retire` cuenta lo que W1b borrará.
  - `problems` lista **todos** los rechazos, también los del pipeline
    (`adoption.refusals`): contratos, targets ajenos, relocación, hijos,
    residuo, S3 y conjunto de targets.

  Revisar en el JSON (sin contenido):
  - `swap`: si `active` es true, las páginas del tmpfs pueden ir a swap.
  - `sizes` y `free`.
  - `units.reverse_dependencies`: debe estar vacío. Incluye `TriggeredBy`,
    `OnFailureOf`, `OnSuccessOf` y `PartOf`.
  - `units.units`: cada timer de `--window-timer` con `LoadState=loaded`
    (un nombre mal escrito da `unit_not_loaded:<timer>`).
  - `timer_dispositions` (abajo).
  - `code_index`.
  - `legacy_path_census`.
  - `journal_files`.
  - `adoption.residue_census`: `voice_profile`, `voice_consent` y
    `conversation_unmarkable` deben ser 0 (si no, P3=A aborta en W2).
  - `problems` y `warnings`.
- **Precondición de W1b**: una corrida de prueba con `problems` vacío
  (`warnings` permitidos, salvo `retry`) sobre el estado actual. W1b es irreversible; si W2
  rechazara después, la ventana fallaría con los adaptadores ya borrados.
- `legacy_path_census.legacy_checkout_targets > 0` aparece como
  `runtime_admission_adoption_foreign_target:<participante>:legacy_checkout`.
  Es un locator de antes del 10-28 (`/home/kza/kza`). Hace falta una regla
  revisada en código antes de la ventana (0e2 F2). No hay `--confirm-foreign`.
- `runtime_admission_recovery_erasure_journal_ambiguous` (`journal_files` > 1)
  indica una copia suelta de un `.db` con tablas del diario de borrado. Quitar
  ese archivo del estado legado, bajo legado y con decisión registrada, y
  repetir la corrida de prueba. R-b rechaza lo mismo (0d4a F4).
  - Hacerlo **antes de W2**: prepare fija el manifiesto del estado legado, que
    también es la imagen de L4 y la línea base de W-abort-2. Quitarlo después
    hace fallar el commit y el abort.
  - Dejar una nota sin contenido (ruta relativa por clase, tamaño, sha256 y
    id de decisión), nunca el contenido.
- `runtime_admission_adoption_residue:voice_profile` (o `:voice_consent`,
  `:conversation`) aborta por P3=A. Necesita su propia unidad revisada.
- `runtime_admission_adoption_retirement_incomplete`: un W1b anterior se cortó
  entre el borrado y su registro. Repetir W1b con el mismo id P1 lo completa.
- El usuario deja registrados de antemano:
  - la autorización de la ventana (id para `--decision` de prepare);
  - el id P1=D;
  - el id L4 para `uninstall-launcher`, preparado aunque no se use.
- **Justo antes de W1** (y de cualquier otra parada gestionada): dueños de
  captura vivos en 0 (R5V-1). En el panel, Privacidad → Captura global, o en
  `GET /api/capture/control` con la sesión de administrador,
  `live_capture_owners` debe ser 0. Si no, cerrar las pestañas del panel que
  graban y las herramientas nativas (entrenamiento o diagnóstico de wake word,
  calibración, zonas, comando manual), esperar a que baje a 0 y volver a
  mirar. Un dueño vivo que no retira deja la parada colgada hasta el SIGKILL
  de `TimeoutStopSec`, sin recibo de cierre (head `UNKNOWN`).
  `carried_capture_owners` > 0 no frena la parada. Una release anterior a
  R5V-1 no informa esos campos: ahí `pending_browser_recorders` y
  `pending_native_clients` deben ser 0.

## Ventana

- **W0** (lectura). Registrar:
  - las propiedades de kza-voice;
  - las unidades y timers kza habilitados;
  - `readlink current`, VRAM y swap.

  Disposición por timer según `timer_dispositions`:
  - `restart unchanged` solo sin referencias;
  - `review`: el timer referencia `current`, un slot, `/home/kza/kza/src`,
    el contenedor o el estado legado. Se reapunta con un cambio revisado o
    queda deshabilitado con motivo.

  Si alguna unidad que no sea `default.target` arrastra kza-voice, no seguir.
  Las unidades de servicio que W1b va a leer, como mínimo:
  - `kza-voice.service`;
  - `kza-llm-fast.service` (llama-server :8101, fast-path NLU);
  - cualquier otra con `llama-server`, `--lora`, `lora_adapters`, la raíz de
    modelos o `/home/kza/kza` en su comando, entorno o `EnvironmentFile`, o con
    un volumen de contenedor sin resolver o que cubra los modelos.

  Timers de la ventana: todos los `kza-*.timer` habilitados de W0, por nombre.
- **W1-pre** (obligatorio): repetir la verificación de «Archivo R1 fijado» con
  el mismo `$R1_ARCHIVE`. Sin `R1 fijado: OK`, detener: no hay W1. `$ST` sigue
  siendo el de «Extraer R1 verificado» en esta misma sesión; si no, repetir
  el bloque `PY=`/`C=`, «Archivo R1 fijado» y ese bloque (otro `mktemp -d`).
- **W1**. Ejecutar:
  ```bash
  systemctl --user stop kza-voice.service      # salida 0 obligatoria
  systemctl --user disable kza-voice.service   # salida 0 obligatoria
  systemctl --user is-enabled kza-voice.service   # debe imprimir 'disabled'
  ```
  Solo la salida 0 cuenta como éxito (systemctl sale 1 ante un fallo).
  Después, parar los timers de la ventana y comprobar con `fuser` que nadie
  tiene abierto el estado legado.
- **Riesgo de reinicio entre W1 y W8**: la única protección es el `disable`
  de W1. No habilitar ni arrancar kza-voice entre W1 y W8 salvo por los pasos
  listados aquí (W8, W-abort-1, W-abort-2). Desde W4, `current` apunta a R1:
  un arranque legado a mano ejecutaría R1 sin ticket y el controlador lo
  rechaza (G1).
- **W1b**: `$ST retire-legacy-training --decision <id P1> --container-slot $C --unit kza-llm-fast.service --unit <unidad>...`.
  - Va solo si la corrida de prueba encontró archivos bajo
    `models/lora_adapters`, y solo después de una corrida de prueba sin
    `problems`.
  - El comando lista él mismo las unidades de usuario (`list-unit-files` y
    `list-units`) y lee siempre kza-voice y `kza-llm-fast.service`. Rechaza
    `runtime_admission_retire_unit_not_listed:<unidad>` si una unidad que
    alcanza los modelos falta en los `--unit`, y `unit_not_loaded:<unidad>` si
    una unidad nombrada no está cargada.
  - Resuelve `${VAR}` y `$VAR` de `ExecStart` con `Environment=` y cada
    `EnvironmentFile=` (leído sin seguir enlaces; el registro guarda solo los
    nombres de las claves). Una variable o un archivo sin resolver rechaza.
    Única excepción: un `$VAR` sin resolver dentro de un argumento (un
    script `sh -c`, que systemd deja al programa) en una unidad no nombrada
    que no menciona `llama-server`, `--lora`, `lora_adapters` ni una ruta de
    kza (las unidades de Ubuntu de IBus y unity). En kza-voice,
    kza-llm-fast y cada `--unit`, y en el escaneo de Q5, rechaza igual.
  - Rechaza sin borrar nada en estos casos:
    - enlace, FIFO o socket dentro;
    - cruce de montaje;
    - ruta de modelo configurada bajo la raíz;
    - argumento de unidad bajo la raíz, `llama-server --lora` incluido;
    - volumen de podman/docker que cubre una raíz o no se puede resolver.
  - Antes de borrar escribe `admission/adoption/retire-intent-<ts>.json`
    (decisión, raíces, censo previo, digest del manifiesto). Si se corta
    después del borrado, repetir el mismo comando con el mismo id P1 lo
    completa; prepare rechaza mientras tanto.
  - Es irreversible por decisión (P1=D). Deja
    `admission/adoption/retire-legacy-training-<ts>.json`.
- **W2**: `$ST adoption-prepare --container-slot $C --archive "$R1_ARCHIVE" --slot <slot R1> --decision <id ventana> --window-timer <timer>...`.
  - Requiere kza-voice inactivo y deshabilitado, timers inactivos y scratch en
    tmpfs.
  - Construye `slots/<slot R1>` con la marca de copia de adopción. No hace CAS
    y no cambia `current`.
  - Imprime `plan_sha256`. Si rechaza, ir a W-abort-1. Si W-abort-1 responde
    `runtime_admission_adoption_nothing_prepared` (prepare ya limpió lo suyo),
    volver al legado a mano:
    ```bash
    systemctl --user enable kza-voice.service      # salida 0
    systemctl --user is-enabled kza-voice.service  # 'enabled'
    systemctl --user start kza-voice.service
    ```
    y arrancar los timers de la ventana.
- **W3**: `$ST install-controller "$R1_ARCHIVE"`. El archivo debe declarar `controller_epoch` 4, exactamente la época que comprueba `adoption-commit`.
- **W4**: `$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline adoption-commit --confirm <plan_sha256>`.
  - Verifica todo el plan de nuevo y rechaza con
    `runtime_admission_adoption_plan_mismatch:<ítem>` antes de cualquier
    registro o CAS. Los ítems son: head, controller, units (timers y sus
    disposiciones incluidos), current, layout, retirement (también: ningún
    archivo nuevo bajo las raíces LoRA cuando no hubo retiro), archive,
    tables, source, code_index y prepared.
  - Después escribe `admission/adoption/commit-intent-<plan>.json`, quita la
    marca, escribe el sello, el registro y el head `adoption` (generación 1)
    y publica `current`.
  - **Si se corta** (corte de luz, SIGKILL, error de disco) en cualquier punto:
    - repetir el mismo `adoption-commit --confirm <plan_sha256>`: termina el
      mismo commit (antes del CAS lo reverifica todo y repite las escrituras
      inmutables; después del CAS solo publica `current`); o
    - si no se quiere seguir y el head no cambió: W-abort-1 con
      `--prepared-slot <slot R1>` borra el slot (aunque ya no tenga la marca)
      y ese plan queda rechazado (`plan_aborted`); si el head ya cambió:
      W-abort-2.
    - Mientras tanto, prepare y la corrida de prueba rechazan
      `prepared_pending`: el slot nunca queda fuera de todo linaje.
- **W5**:
  ```bash
  mkdir -m 0700 /home/kza/releases/logs
  ```
  Después, agregar la estrofa de `console-logrotate`.
- **W6**: `install-launcher`. La comprobación previa tiene que salir limpia.
  - `--timeout-start`: dejar el valor por defecto, 240 s. Cubre un arranque
    completo con R-b sin depender de `EXTEND_TIMEOUT_USEC` (DEPLOYMENT,
    «EXTEND no alcanza solo»). No pasar un valor menor.
- **W7**: el usuario aplica el drop-in sudo (N1=B) y lo verifica en lectura.
  - Texto y comandos exactos (no escribe nada):
    ```bash
    $ST stop-timeout-dropin --unit-timeout-stop <TimeoutStopSec de kza-voice> [--chain-extra <s>]
    ```
    Archivo `/etc/systemd/system/user@<uid>.service.d/zz-kza-stop-timeout.conf`,
    modo `0644 root:root`, contenido:
    ```
    [Service]
    TimeoutStopSec=<B>s
    ```
    con B = TimeoutStopSec de kza-voice + cadena posterior + 15 s. Aplicar con
    `sudo install` y `sudo systemctl daemon-reload`. **No reiniciar**
    `user@<uid>.service` (pararía todas las unidades kza).
  - Verificación de solo lectura (es lo que lee el `RebootBudget` de serve):
    ```bash
    systemctl show user@<uid>.service -p TimeoutStopUSec,DropInPaths   # B, y zz- último
    systemctl cat user@<uid>.service
    ls /etc/systemd/system.control /run/systemd/system.control
    ```
  - Si el usuario no puede aplicarlo ahora: todavía no hay ticket, así que
    W-abort-2 es válido. Por defecto abortar y reprogramar. Seguir en modo de
    respaldo (presupuesto de 5 s) solo con una decisión registrada del usuario.
- **W8**: habilitar y arrancar kza-voice.
  - Desde aquí, mientras kza-voice esté `activating`
    (`systemctl --user show -p ActiveState kza-voice.service`), no correr en
    el manager de `kza` `systemctl --user daemon-reload` ni los comandos que
    recargan implícitamente: `enable`, `disable`, `reenable`, `mask`,
    `unmask`, `link`, `preset`, `revert`, `edit`. Un reload descarta la
    extensión del arranque y lo corta si ya pasó `TimeoutStartSec`. Esperar
    a `active`, `failed` o `inactive`. El `enable` de W8 va antes del
    `start`; un reinicio automático en curso también cuenta como
    `activating`. No actualizar el paquete systemd durante la ventana (re-
    ejecuta los managers de usuario).
  - Límite de arranques: cinco en 300 s, contando los reinicios
    automáticos. Si un arranque falla con «Start request repeated too
    quickly», resolver la causa y correr
    `systemctl --user reset-failed kza-voice.service` antes de volver a
    arrancar. Pasa, por ejemplo, tras un crash y cuatro rechazos 69 seguidos.
  - Reiniciar solo los timers `restart unchanged`.
  - Hacer las comprobaciones de aceptación.
  - Medir el arranque real hasta READY (la aplicación con sus modelos, que
    los ensayos no cargan):
    ```bash
    systemctl --user show kza-voice.service -p ExecMainStartTimestampMonotonic,ActiveEnterTimestampMonotonic,TimeoutStartUSec
    ```
    La diferencia de los dos timestamps, en microsegundos, es el arranque
    hasta READY. Anotarla en W10. Con `TimeoutStartSec=240`, si pasa de
    120 s (la mitad del plazo), detenerse: no seguir dependiendo de ese
    valor. Volver a correr `install-launcher` con un `--timeout-start` de al
    menos el doble de lo medido (máximo 3600), con kza-voice detenido, y
    repetir la medición. Si un arranque llega a cortarse por tiempo
    (`Result=timeout`), el head queda seguro (ver DEPLOYMENT, «EXTEND no
    alcanza solo») y se aplica lo mismo.
  - Volver a aplicar `PP_MIN_NN` de Hall.
  - Confirmar que no existe `data/audit.db` en el estado gestionado.
- **W9**: un stop y un start limpios, con recibo y sello de cierre. Ese sello
  es el DÍA 0 de P7.
  - Repetir la medición de W8 sobre este segundo arranque y anotarla en W10.
  - Después de W9, cuando la aplicación tuvo tiempo de procesar sus colas
    (repetir cada pocos minutos durante la primera hora si hace falta),
    comprobar que el trabajo de privacidad reencolado o reubicado por la
    adopción terminó. Es de solo lectura y no muestra contenido:
    ```bash
    $PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline completion-status
    ```
    Todo debe dar 0 y `complete: true`: `erasure_steps_not_complete` vacío,
    `legacy_training_steps_not_complete`,
    `policy_index_deletions_not_complete`, `index_delete_jobs`, cada valor de
    `conversation_exempt` y de `resurrected`. `honest_blocker_steps` queda
    aparte: son los pasos de los destinos literales sin limpiador
    (`unresolved_*_inventory`, U3b-0d4a), que nunca terminan por diseño y no
    cuentan; anotar sus conteos en W10. Un paso pendiente exime sus
    copias del control de resurrección, así que sin esta comprobación
    quedaría sin ver. Si algo sigue abierto pasada la primera hora, anotar
    los conteos en W10 y avisar al usuario antes de cerrar la ventana.
- **W10**: evidencia. Anotar en el handoff y en DEPLOYMENT («Retiro del estado
  congelado de b0866e8») el DÍA 0 (sello de W9), la fecha de DÍA 0 + 14 × 24 h y el
  responsable (el usuario, como operador de `kza`). Anotar también los dos
  arranques hasta READY de W8/W9 y la salida de `completion-status`.

## Día 14: retiro Q5 del estado de b0866e8 y sus copias del 2026-10-29

Desde DÍA 0 + 14 × 24 h exactas («día 14», decisión Q1 = A), nunca antes: el
comando lo rechaza (`runtime_admission_legacy_retire_too_early`, con
`eligible_from`). Borra el estado del slot de b0866e8 y las diez bases de
`backups/$C/databases/` (Q2 = A). Después no hay rollback no gestionado y
`/home/kza/kza` no es un destino (Q3 = A).

Alcance con `$C` = e9669ec (R1E, 2026-11-18): las diez bases que borra son las
de `backups/release_20261029T185008966364Z_e9669ec/databases`, tomadas a las
15:50. Las diez de la publicación de las 14:48 son copias del mismo estado y
están en `backups/release_20261029T174830709997Z_63ea8d3/databases`. Quedan
como exclusión comprobada y este paso no las borra. Falta que el usuario
confirme si Q2 = A cubre también ese segundo conjunto.

Antes de este paso hacen falta además dos decisiones del usuario. Sin ellas
la corrida de prueba (paso 2) rechaza sin borrar nada, así que no hay que
borrar nada a mano:

- el slot de 63ea8d3 conserva sus enlaces al estado de b0866e8, y el retiro
  rechaza
  `runtime_admission_legacy_retire_referenced:link:slots/release_20261029T174830709997Z_63ea8d3`;
- el slot de b0866e8 tiene `activation-report.json`, `live-report.json` y
  `prepare-report.json` junto a `runtime/` y `state/`, y el retiro rechaza
  `runtime_admission_legacy_retire_slot_unexpected`.

El día 14 se corre en una sesión nueva de `kza`, no en la de la ventana: el
`$ST` de «Extraer R1 verificado» ya no existe y su `mktemp -d` no se
conserva. Desde W3 sólo se corre `controller/current`, así que, como en W4 y
en `completion-status`, los comandos de este paso se definen desde el
controlador instalado, al empezar:

```bash
PY=/home/kza/kza/.venv/bin/python
C=release_20261029T185008966364Z_e9669ec
ST="$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline"
```

Desde SC-2 tampoco hay rollback gestionado hasta que exista el modo release:
después de Q5 no queda ningún rollback. Antes de empezar, revisar la
respuesta a SC-D5 en `docs/architecture/DEPLOYMENT.md` («Sin transiciones
gestionadas después de R1 hasta el modo release»). La respuesta registrada es
A (2026-11-15): este paso corre el día 14. Sólo con la opción B se
postergaría; si esa respuesta cambiara, no ejecutarlo.

0. Si L4 se usó desde la adopción (existe algún
   `/home/kza/releases/admission/launcher/uninstall-*.json`), parar aquí: el
   retiro rechaza `runtime_admission_legacy_retire_l4_used`, y los datos que
   L4 sirvió en ese período necesitan una decisión nueva del usuario.
1. El mismo día, antes de la corrida de prueba, quitar la estrofa de la
   consola vieja de `console-logrotate.conf`. Mientras exista, el retiro
   rechaza `referenced:logrotate`. Su texto exacto se escribió en el server
   el 2026-10-28 y no está en Git: es el bloque que empieza en la línea con
   la ruta de la consola bajo `/home/kza/releases/slots/` (la de
   `state/logs/console.log`) y termina en su `}`. Según el runbook del
   2026-10-28 tiene esta forma (horario, 24 rotaciones, `maxage 1`,
   `copytruncate`):
   ```
   /home/kza/releases/slots/<slot>/state/logs/console.log {
       hourly
       rotate 24
       maxage 1
       copytruncate
       ...
   }
   ```
   Mostrarlo (sólo lee) y borrar exactamente esas líneas, sin tocar la
   estrofa de `/home/kza/releases/logs/console.log`:
   ```bash
   CONF=/home/kza/releases/console-logrotate.conf
   awk '/\/home\/kza\/releases\/slots\//{p=1} p{print NR": "$0} p&&/}/{p=0}' "$CONF"
   ```
   Comprobar que ya no está (sólo lee):
   ```bash
   grep -n '/slots/' "$CONF"                                # sin salida (salida 1)
   grep -n '^/home/kza/releases/logs/console.log' "$CONF"   # la estrofa gestionada sigue
   /usr/sbin/logrotate -d "$CONF"                           # modo de prueba: no cambia nada
   ```
2. Corrida de prueba (sólo lee; puede ir con el servicio en marcha):
   ```bash
   $ST retire-legacy-state --decision <id P7> --container-slot $C --dry-run
   ```
   Tiene que decir `"eligible": true` y listar diez nombres en
   `backups.databases`. Anotar `plan_sha256`.
   - `referenced:<qué>`: `problems` lista todo lo que todavía apunta a los
     slots de b0866e8 o e9669ec o a esas bases. Corregirlo antes de seguir. No
     borrar a mano.
   - `backups_set`: el respaldo no es el conjunto revisado (diez SQLite de ese
     estado). Parar y avisar.
   - `scan`: un directorio del estado vivo desapareció durante el recorrido.
     Repetir, o correrla con el servicio detenido.
3. Retiro:
   ```bash
   systemctl --user stop kza-voice.service
   $ST retire-legacy-state --decision <id P7> --container-slot $C --confirm <plan_sha256>
   systemctl --user start kza-voice.service
   ```
   - Termina con `record` = `legacy-retirement/retired-<plan>.json`.
   - `confirm_mismatch` quiere decir que algo cambió desde la corrida de
     prueba: repetir el paso 2.
   - Comprobar sólo leyendo. No probar L4 con `uninstall-launcher`: si el retiro
     no hubiera ocurrido, ese comando cambiaría la unidad a L4.
     ```bash
     ls /home/kza/releases/admission/legacy-retirement/      # intent-<plan>.json y retired-<plan>.json
     ls -d /home/kza/releases/slots/release_20261028T145442Z_b0866e8/state \
           /home/kza/releases/backups/$C/databases           # ambos: No such file or directory
     ls /home/kza/releases/backups/$C                        # quedan release.conf, config y .env
     ```
     Con ese registro, `uninstall-launcher` rechaza los dos slots con
     `runtime_admission_uninstall_legacy_retired` (ensayado en T-A17).
4. Si se corta (corte de luz, SIGKILL), repetir el mismo comando con el mismo
   id y el mismo `--confirm`: termina desde
   `admission/legacy-retirement/intent-<plan>.json`. Desde que existe esa
   intención, L4 ya rechaza.
5. Comprobar que kza-voice volvió a READY.

## Abortar

- **W-abort-1 (antes de W4)**:
  ```bash
  $ST adoption-abort --prepared-slot <slot R1>
  ```
  - Borra el slot marcado y vuelve a habilitar kza-voice.
  - El retiro de W1b queda hecho.
  - Comprueba el `enable` (salida 0 y `is-enabled` = `enabled`); si falla
    sale 78 con `runtime_admission_adoption_enable_failed` y hay que habilitar
    a mano (W2, arriba).
  - Arrancar a mano el servicio legado y los timers. `current` nunca cambió.
  - Sirve también para un commit cortado antes del CAS (slot sin marca, con
    `commit-intent`, igual al manifiesto del plan y sin head que lo ligue).
- **W-abort-2 (después de W4, sin ningún ticket)**:
  ```bash
  $ST adoption-abort --unit-dir ~/.config/systemd/user/kza-voice.service.d
  ```
  Exige todo esto:
  - unidad inactiva;
  - head igual al head de adopción;
  - ningún registro de corrida, recibo, recuperación ni cierre;
  - estado legado byte a byte igual al manifiesto de prepare.

  Efectos:
  - renderiza `legacy.conf` si `managed.conf` existe;
  - restaura `current`;
  - escribe `admission/adoption/abort-<sha256 del head>.json`;
  - borra el slot y vuelve a habilitar kza-voice.

  Después:
  ```bash
  systemctl --user is-enabled kza-voice.service   # 'enabled'; si no, enable a mano (salida 0)
  systemctl --user start kza-voice.service
  ```
  y reiniciar los timers de la ventana. Si el comando sale 78 con
  `enable_failed`, el abort ya está registrado: habilitar a mano.

  Ese head queda terminal: `begin_run` y `activate` lo rechazan. Solo se puede
  volver a adoptar en un head nuevo con un prepare nuevo. Repetir
  `adoption-abort` solo completa el mismo abort (por ejemplo, si el `enable`
  falló). Rechaza `already_aborted` cuando el abort ya habilitó kza-voice
  (`admission/adoption/enabled-after-abort-<sha256 del head>.json`) o cuando
  una ventana posterior ya escribió algo (intento o registro de retiro, plan,
  intento de commit o W-abort-1): nunca vuelve a habilitar kza-voice en medio
  de otra ventana.
- **Si existe un ticket** (W8 escribió uno, aunque el arranque fallara antes de
  READY): solo vale L4, con `uninstall-launcher`, el id de decisión y la frase
  `revivir datos borrados`. Vale hasta el retiro Q5 (desde DÍA 0 + 14 × 24 h);
  desde que existe una intención Q5 rechaza `legacy_retired`. Usarlo bloquea
  el retiro (`l4_used`) hasta una decisión nueva del usuario.
- Los dos aborts y L4 comprueban primero los enlaces del contenedor de
  e9669ec: si alguno cuelga, rechazan `runtime_admission_uninstall_legacy_state`
  sin cambiar nada.
- **Después de que W8 empezó**: la activación de W8 deja un head `selection`
  aunque todavía no haya ticket, y `adoption-abort` rechaza
  `runtime_admission_adoption_abort_head`. Cualquier `abort_head` a partir de
  W8 significa L4.
- **Plazo**: abortar si W8 no llegó a READY 60 minutos después de W1.
