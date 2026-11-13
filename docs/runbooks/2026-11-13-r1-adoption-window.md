# Ventana R1: adopción del estado de producción (U3b-0e3)

Procedimiento del operador para la ventana W0–W10 del diseño
`t040-t041/u3b-0e-adoption-design-v3.json` (secciones 6 y 7), con las
decisiones P1–P8 de `user-decisions-20261110.json`. Los comandos están en
`scripts/adoption_commands.py` y se llaman desde `stage_runtime`. U3b-0e5
agrega el texto de DEPLOYMENT y la fecha de P7. Este runbook todavía no
autoriza la ventana.

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
  Revisar en el JSON (sin contenido):
  - `swap`: si `active` es true, las páginas del tmpfs pueden ir a swap.
  - `sizes` y `free`.
  - `units.reverse_dependencies`: debe estar vacío.
  - `timer_dispositions` (abajo).
  - `code_index`.
  - `legacy_path_census`.
  - `journal_files`.
  - `adoption.residue`.
  - `problems`.
- `legacy_path_census.legacy_checkout_targets > 0` aparece como
  `runtime_admission_adoption_foreign_target:<participante>:legacy_checkout`.
  Es un locator de antes del 10-28 (`/home/kza/kza`). Hace falta una regla
  revisada en código antes de la ventana (0e2 F2). No hay `--confirm-foreign`.
- `runtime_admission_recovery_erasure_journal_ambiguous` (`journal_files` > 1)
  indica una copia suelta de un `.db` con tablas del diario de borrado. Quitar
  ese archivo del estado legado, bajo legado y con decisión registrada, y
  repetir la corrida de prueba. R-b rechaza lo mismo (0d4a F4).
- `runtime_admission_adoption_residue:voice_profile` aborta por P3=A.
  Necesita su propia unidad revisada.
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
- **W1**. Ejecutar:
  ```bash
  systemctl --user stop kza-voice.service
  systemctl --user disable kza-voice.service
  ```
  Después, parar los timers de la ventana y comprobar con `fuser` que nadie
  tiene abierto el estado legado.
- **W1b**: `$ST retire-legacy-training --decision <id P1> --container-slot $C --unit <unidad>...`.
  - Va solo si la corrida de prueba encontró archivos bajo
    `models/lora_adapters`.
  - Pasar todas las unidades kza de W0.
  - Rechaza sin borrar nada en estos casos:
    - enlace, FIFO o socket dentro;
    - cruce de montaje;
    - ruta de modelo configurada bajo la raíz;
    - argumento de unidad bajo la raíz, `llama-server --lora` incluido.
  - Es irreversible por decisión (P1=D). Deja
    `admission/adoption/retire-legacy-training-<ts>.json`.
- **W2**: `$ST adoption-prepare --container-slot $C --archive r1.tar.gz --slot <slot R1> --decision <id ventana> --window-timer <timer>...`.
  - Requiere kza-voice inactivo y deshabilitado, timers inactivos y scratch en
    tmpfs.
  - Construye `slots/<slot R1>` con la marca de copia de adopción. No hace CAS
    y no cambia `current`.
  - Imprime `plan_sha256`. Si rechaza, ir a W-abort-1.
- **W3**: `$ST install-controller r1.tar.gz`. Exige época de controlador 2.
- **W4**: `$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline adoption-commit --confirm <plan_sha256>`.
  - Verifica todo el plan de nuevo y rechaza con
    `runtime_admission_adoption_plan_mismatch:<ítem>` antes de cualquier
    registro o CAS. Los ítems son: head, controller, units, current, layout,
    retirement, archive, tables, source y prepared.
  - Después quita la marca y escribe el sello, el registro y el head
    `adoption` (generación 1). Publica `current`.
- **W5**:
  ```bash
  mkdir -m 0700 /home/kza/releases/logs
  ```
  Después, agregar la estrofa de `console-logrotate`.
- **W6**: `install-launcher`. La comprobación previa tiene que salir limpia.
- **W7**: el usuario aplica el drop-in sudo (N1=B) y lo verifica en lectura.
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
  - Arrancar a mano el servicio legado y los timers. `current` nunca cambió.
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

  Ese head queda terminal: `begin_run` y `activate` lo rechazan. Solo se puede
  volver a adoptar en un head nuevo con un prepare nuevo.
- **Si existe un ticket** (W8 escribió uno, aunque el arranque fallara antes de
  READY): solo vale L4, con `uninstall-launcher`, el id de decisión y la frase
  `revivir datos borrados`.
- **Plazo**: abortar si W8 no llegó a READY 60 minutos después de W1.
