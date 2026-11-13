# Ventana R1: adopción del estado de producción (U3b-0e3)

Procedimiento del operador para la ventana W0–W10 del diseño
`t040-t041/u3b-0e-adoption-design-v3.json` (secciones 6 y 7), con las
decisiones P1–P8 de `user-decisions-20261110.json`. Los comandos están en
`scripts/adoption_commands.py` y se llaman desde `stage_runtime`. U3b-0e5
agrega el texto de DEPLOYMENT y la fecha de P7. Este runbook todavía no
autoriza la ventana.

## Compuerta de programación (P7, Q5)

- La ventana **no se programa** hasta que U3b-0e5 tenga implementados,
  revisados y ensayados (T-A17):
  - la comprobación de enlaces del destino de rollback (`_verify_legacy_slot`
    rechaza un enlace del contenedor roto o colgante);
  - el retiro Q5 del estado b0866e8 (registro inmutable que saca 63ea8d3 y
    b0866e8 de L4; excluye `/home/kza/kza/models`, `/home/kza/kza/.venv` y el
    slot R1).
- **Paso del día 15**: el borrado Q5 corre el DÍA 0 + 15 (DÍA 0 = el sello de
  cierre de W9). Responsable: el usuario, como operador de `kza`. En W10 se
  anota en el handoff la fecha exacta y el responsable. Hasta el día 14 sólo
  vale L4 hacia 63ea8d3 (decisión registrada y frase); desde el día 15 no hay
  rollback no gestionado.

Todo corre como `kza` y nunca como root. Cada comando rechaza con salida 78
y `{"refused": <código>}` en estos casos:

- otro usuario, o root;
- una raíz que no sea `/home/kza/releases` real (sin enlaces, con `slots/`,
  sin escritura de grupo u otros);
- una segunda ejecución.

```bash
R1=<tmp privado>/r1            # R1 extraído (build_runtime.py, bajo yo)
PY=/home/kza/kza/.venv/bin/python
ST="$PY -I -B $R1/scripts/stage_runtime.pyc --root /home/kza/releases --offline"
C=release_20261029T174830709997Z_63ea8d3
```

## Antes de la ventana

- Corrida de prueba consultiva con el servicio en marcha. Es de solo lectura:
  no toma lock ni escribe en la raíz. Usa un scratch en tmpfs
  `/run/user/<uid>` que se borra al terminar, ante un rechazo y ante SIGTERM.
  ```bash
  $ST adoption-dry-run --container-slot $C --archive r1.tar.gz --slot <slot R1> \
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
- **W2**: `$ST adoption-prepare --container-slot $C --archive r1.tar.gz --slot <slot R1> --decision <id ventana> --window-timer <timer>...`.
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
- **W3**: `$ST install-controller r1.tar.gz`. Exige época de controlador 2.
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
  - Reiniciar solo los timers `restart unchanged`.
  - Hacer las comprobaciones de aceptación.
  - Volver a aplicar `PP_MIN_NN` de Hall.
  - Confirmar que no existe `data/audit.db` en el estado gestionado.
- **W9**: un stop y un start limpios, con recibo y sello de cierre. Ese sello
  es el DÍA 0 de P7.
- **W10**: evidencia y DEPLOYMENT (U3b-0e5).

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
  `revivir datos borrados`.
- **Después de que W8 empezó**: la activación de W8 deja un head `selection`
  aunque todavía no haya ticket, y `adoption-abort` rechaza
  `runtime_admission_adoption_abort_head`. Cualquier `abort_head` a partir de
  W8 significa L4.
- **Plazo**: abortar si W8 no llegó a READY 60 minutos después de W1.
