# Compactación de estados dueños (U3b-0c)

Runbook del operador para `stage_runtime compact`. Diseño:
`owner-compaction-design-v2.json` a `v5` (con las correcciones E1 a E3 de la
revisión de v5) y las decisiones del usuario `u3b_0c_compaction` (P1 a P7) y
`u3b_0c_compaction_v2` (Q1 a Q7, VACUUM sólo en el estado nuevo), más las
correcciones de la revisión independiente (F1 a F12) con las decisiones
`u3b_0c_review` (D1 a D4). Evidencia:
`data/autonomous-roadmap-20261102/t040-t041/u3b-0c-evidence.json` (local,
fuera de Git).

Estado: **desarrollada y probada en raíces de prueba; nunca se ejecutó sobre
`/home/kza/releases`.** Esta unidad es la última planificada antes de R1.

## Qué hace

Toma el head cerrado H y crea un estado nuevo K que es una copia idéntica de H,
salvo que todas las rutas apuntan a K: los destinos de borrado que H tenía
delegados de dueños anteriores pasan a ser de K. Después de **una** corrida
gestionada limpia de K (la corrida de verificación, que deja K'), destruye con
`unlink` todas las demás copias que el controlador puede **probar** que hizo:

- slots gestionados con binding y marker válido, slots preparados y nunca
  activados (registro de `prepare`), estados que fueron destino de una
  delegación. Un slot es miembro sólo si contiene exactamente `runtime/` y
  `state/`: cualquier otra entrada hecha a mano (por ejemplo
  `slots/a/respaldo-2026-10-29/`) lo deja **entero** en la lista intacta, con
  la clase `unexpected_entries` y los nombres de esas entradas (nunca su
  contenido). Lo mismo vale dentro de `runtime/`: sólo los archivos que lista
  su `release.json` y los enlaces `config`, `data`, `.env`; un archivo más
  (`runtime/notas.txt`) deja el slot entero con el nombre
  `runtime/notas.txt`. Un slot probado sólo por un registro de staging (un
  `prepare` cortado después del rename) no prueba qué hay en su `state/`: queda
  intacto (`prepare_leftover_kept`) salvo con `--retire-prepare-leftovers`.
  Los miembros se destruyen enteros;
- restos `slots/.prepare-*` con registro de staging (sólo con
  `--retire-prepare-leftovers`) y restos `.activate-*` vacíos o con un único
  enlace `current`. Sus enlaces (`runtime/config`, `runtime/data`,
  `runtime/.env`, `current`) se borran como enlaces, nunca se siguen;
- copias del catálogo dentro de la raíz, **incluidas las que estén bajo
  `backups/`** (decisión D3 = A): un snapshot que el controlador catalogó
  es una copia que probó haber hecho, esté donde esté dentro de la raíz. Sólo
  si cada entrada de primer nivel de la copia está en sus extents o en el
  inventario de su preparación (por ejemplo `state/` y `manifest.json`):
  algo puesto a mano adentro (`backups/snapshot-x/respaldo-a-mano/`) deja la
  copia **entera** en la lista intacta (`unexpected_entries`, con los nombres).
  El resto de `backups/`, incluidos los respaldos del 2026-10-29, queda intacto;
- el trabajo propio de compactaciones anteriores.

Así, lo que una persona pidió borrar (una conversación, un recuerdo, su voz)
deja de existir también en los slots anteriores (P2).

## Qué NO hace (límites, P5 = C)

- **Sólo `unlink`.** No sobrescribe, no trunca, no corre `fstrim`. Los bloques
  liberados siguen en el disco hasta que el sistema de archivos los reutilice;
  quien tenga acceso al dispositivo puede recuperarlos con herramientas
  forenses. La compactación es un borrado lógico.
- **VACUUM sólo en K** (decisión P5_vacuum_new_state). Las bases SQLite de K
  pasan por `secure_delete=ON` y `VACUUM`, así que las filas que H ya había
  borrado no quedan en páginas libres de K. `chroma.sqlite3` no se vacía (es
  opaco al SDK): sus páginas libres pueden conservar residuos; es un límite
  documentado.
- **Nada fuera de la raíz gestionada** (Q4): `/home/kza/kza`, los respaldos
  externos, snapshots del disco y cualquier copia fuera de
  `/home/kza/releases` quedan intactos.
- **Nada no probado dentro de la raíz**: `backups/` (salvo las copias
  catalogadas, ver arriba), `logs/`, `deployments/`,
  `dashboard-proxy/`, `controller/`, archivos de release, `console-logrotate.conf`,
  slots sin prueba (por ejemplo, el contenedor de enlaces de 63ea8d3 o el slot
  de b0866e8), restos `.prepare-*` sin registro. Se listan por nombre y tipo
  (sin abrirlos) en la «lista intacta», y el operador la reconoce por su
  digest. El retiro de b0866e8 es el de Q5, no éste.

## Antes de empezar

Requisitos (el comando los verifica y rechaza con un código si falta alguno):

- R1 construido desde un árbol que ya contiene U3b-0c (decisión D1 = A): así
  ningún controlador de época 2 existe en la raíz gestionada. La adopción exige
  `controller_epoch == 3`;
- controlador época 3 o posterior instalado en `controller/current`
  (`install-controller`), y el lanzador instalado con contrato 3 o posterior
  (`install-launcher`; P13). La release de H (y por lo tanto de K) declara
  `launcher_contract >= 3`;
- head cerrado (no `running` ni `unknown`), sin retiro Q5 pendiente;
- servicio detenido y deshabilitado (P5: ningún escritor abierto en H, en los
  dueños ni en los miembros);
- ninguna unidad systemd del usuario ni configuración de logrotate nombra un
  miembro (P14). El escaneo falla cerrado si no hay bus de usuario. Las
  referencias `$VAR` / `${VAR}` de las líneas `Exec*` se resuelven con el
  entorno de la unidad y con `systemctl --user show-environment` (las que
  systemd pone y nunca son rutas, como `$MAINPID`, pasan). Una que no se
  resuelve sigue la regla de U3b-0e4: se tolera sólo en una unidad que nadie
  nombró (ni `kza-voice`, ni `kza-llm-fast`, ni `kza-code-index`, ni las que
  el operador pase con `--unit`) y cuyo texto y entorno propio no mencionan
  una ruta de kza, un miembro ni la raíz (así pasan las unidades de IBus,
  podman y unity de la cuenta). En una unidad nombrada o que menciona kza,
  rechaza aunque no tenga `/`. Una unidad es «de kza» si su nombre empieza
  con `kza`, o si menciona una ruta de kza: la raíz, un miembro,
  `/home/kza/kza`, su forma relativa al home con prefijo (`~/kza`, `~kza/kza`,
  `%h/releases`, `$HOME/kza`, `"$HOME"/kza`), o un `kza` relativo como palabra
  entera en una unidad que corre desde el home (sin `WorkingDirectory=`, o
  con `%h`, `~`, `$HOME`, también entre comillas). También cuentan un enlace
  directo en el home que apunta a una ruta de kza (`/home/kza/app`, que
  apunta a `/home/kza/kza`), las formas con `//` o `/./` y `${HOME:-...}`.
  Un `kza` dentro de otra ruta no cuenta, porque con
  `HOME=/home/kza` está en toda ruta de la cuenta (un drop-in de podman bajo
  `/home/kza/.config` no hace de kza a la unidad). La adopción de U3b-0e4 usa
  la misma regla. Además:
  - las unidades nombradas se escanean siempre, vivan donde vivan; una que el
    operador pasó con `--unit` y systemd no conoce rechaza
    (`unit_unparsed:<unidad>:LoadState`);
  - una unidad enlazada con `systemctl --user link` (estado `linked`) o con su
    nombre en `~/.config/systemd/user` se escanea aunque su archivo esté
    afuera;
  - una plantilla (`nombre@.service`) en un directorio de unidades del usuario,
    o con un drop-in ahí, se lee como texto aunque no haya instancias. Se unen
    las líneas de continuación (`\`), se sustituyen `$VAR`/`${VAR}` con el
    entorno del gestor, su `Environment=` y sus `EnvironmentFile=` (como en
    systemd, el valor del archivo gana sobre `Environment=` esté donde esté la
    línea; la ruta del archivo se compara antes de leerlo), se expanden `%h`,
    `%t` y los demás especificadores fijos de la cuenta, y `$HOME`, `${HOME}`,
    `${HOME:-...}`, `$$HOME`, `~` y `~kza` valen el home. Después, regla de
    subcadena sobre **todo** el texto (comentarios y claves que el escaneo no
    conoce incluidos) y sobre cada valor de su entorno, también los que sólo
    lee el programa: rechaza (`unit_path`) si aparece la raíz o un miembro, o
    si un trozo de ruta (cortado en espacios, `;|&()<>`, comillas, `=`, `,` y
    `:`, así que vale dentro de un `sh -c '...'` o de una lista
    `PYTHONPATH=a:b`) llega a un miembro o tiene una parte fija, antes de
    `%i`/`%I` o de un `$VAR` sin valor, que es la raíz, está debajo, o está
    encima y lo que sigue puede empezar el nombre de la raíz
    (`/home/kza/releases/%I`, `%h/%i`, `%h/rel%i` rechazan; `%h/backup-%i` y
    `%t/%i` pasan). Un comodín de shell (`*`, `?`, `[...]`) en un trozo
    absoluto se juzga igual, componente por componente como lo expande el
    shell: rechaza si puede nombrar algo debajo de la raíz o la raíz misma
    (`%h/rel*/slots/%i`, `%h/r?leases/slots/*/state`, `%h/*/slots/a`,
    `%h/rel*`; un `..` después del comodín, un `[...]` o una parte
    desconocida dejan abierto el resto), también a través de un enlace del
    home a la raíz; `%h/backups/*.tgz` y `%h/.cache/*` pasan. Los escapes C
    (`\x65`, `\145`, `\u0065`, `\s`) se decodifican como los decodifica
    systemd, dentro y fuera de comillas y antes de los especificadores
    (`%h/rel\x65ases` es `/home/kza/releases`); un escape desconocido queda
    como está, y se juzgan las dos lecturas, la de systemd y la del shell.
    Un trozo cuya parte desconocida viene antes de toda `/`
    (`%i`, `${SRC}`, `-C${SRC}`, `backup-%i`) no se puede juzgar, ni
    cualquier variable dentro del guion de un `sh -c` (o `bash -ec`, etc.):
    rechaza (`unit_unparsed:<plantilla>:<clave>`) salvo que la plantilla
    tenga la tolerancia de U3b-0e4 de arriba, y entonces aparece en
    `units.tolerated_unresolved`. Unas comillas sin cerrar o un
    `EnvironmentFile=` ilegible también rechazan. La salida lista las
    plantillas en `units.templates_scanned`;
  - en cualquier unidad, un valor se corta también en `,`, en cada `=` y en
    los caracteres del shell, así que `podman run --mount
    type=bind,src=<miembro>` (y la unidad que genera un Quadlet `Mount=`)
    rechaza como `-v <miembro>:/data`. Un comodín de shell en un trozo
    absoluto (`sh -c 'tar czf /srv/b.tgz <raíz>/slots/*/state'`,
    `<raíz>/slots/a*`, `/home/kza/rel*/slots/*/state`) se juzga por su parte
    fija como en las plantillas, salvo que en una unidad normal nombrar la
    raíz misma (`/home/kza/rel*`) sigue pasando, como nombrarla por su
    ruta. Un comodín relativo (`find -path '*/cache/*'`,
    `rsync --include='*/'`) es casi siempre un patrón que lee el programa y
    no se juzga;
  - un valor de `systemctl --user show-environment` que nombra un miembro
    rechaza (`unit_path:show-environment`) como si fuera un `Environment=` de
    la unidad: cualquier programa del gestor lo hereda. Los valores que
    systemd imprime como `$'...'` (con `*`, `;`, espacios, comillas) se
    decodifican para esta comparación, y un comodín (`<raíz>/slots/*/state`)
    se juzga por su parte fija como en las plantillas; uno que no se puede
    decodificar rechaza (`unit_unparsed:show-environment:<nombre>`). La regla
    de subcadena de las plantillas vale también aquí: un valor que nombra la
    raíz misma, o un miembro pegado a cualquier carácter (`x;<miembro>`,
    `a|<miembro>`, `(<miembro>)`), rechaza;
- ninguna copia del catálogo en preparación (P8,
  `compaction_copy_in_progress`): `reserved` o `writing` con su proceso dueño
  vivo, o un estado de preparación desconocido. «Vivo» es el mismo proceso:
  existe un proceso con ese pid **y** arrancó antes de la reserva (el primer
  archivo de revisión del registro), así un pid reutilizado no cuenta. Si el
  proceso dueño ya no existe, la copia quedó abandonada: el plan la lista en
  `abandoned_copies` y
  se trata como cualquier copia del catálogo (miembro sólo con contenido
  probado);
- ninguna copia del catálogo fuera de la raíz (P8,
  `compaction_copy_outside_root`). Una copia así sólo rechaza mientras
  existe: si es un snapshot viejo que ya no se quiere, **el usuario** decide
  borrarla a mano (no lo hace el controlador). Si se quiere conservar, no se
  puede compactar hasta moverla dentro de la raíz;
- espacio libre para un snapshot más K, y binding de K de 12 MiB o menos (P9).

**No reinicies el servidor durante una compactación.** Si hace falta, primero
`compact abort` (antes de la verificación) o terminá el retiro.

**Nunca corras `stage_runtime` desde un slot (`slots/*/runtime/scripts/`) ni
desde un `controller/<commit>` viejo**: sólo desde `controller/current`. Una
copia de época 2 no conoce la compuerta (no lee `admission/controller.json`).
Desde la época 3, la entrada al controlador y `prepare` rechazan
(`runtime_admission_controller_downgrade`) cuando la época instalada es mayor
que la propia, lo que protege las épocas 4 en adelante.

Variables de este runbook (cuenta `kza`):

```bash
PY=/home/kza/kza/.venv/bin/python
ST="$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc --root /home/kza/releases --offline"
H_ARCHIVE=/home/kza/releases/<archivo de la release de H>.tar.gz   # el mismo artefacto que corre H
K=<nombre nuevo de slot, nunca usado antes>
```

## Pasos

1. Detener y deshabilitar el servicio (salida 0 obligatoria en cada uno):

   ```bash
   systemctl --user stop kza-voice.service
   systemctl --user disable kza-voice.service
   ```

2. Plan (sólo lectura; se puede repetir):

   ```bash
   $ST compact plan "$K" --archive "$H_ARCHIVE" [--retire-prepare-leftovers] [--unit <unidad de kza> ...]
   ```

   Revisar, sin datos personales en la salida:
   - `members`: cada miembro con su clase, prueba y conteos (archivos, bytes);
   - `untouched`: lo que queda intacto. Si algo que debería borrarse aparece
     acá, **no sigas**: no hay prueba de que el controlador lo haya hecho;
   - `coverage`: filas y archivos que existen sólo en miembros, por clase
     (`expired`, `erased`, `superseded`, `scrubbed`, `unexplained`;
     `removed_by_k_run` vale 0 en el plan). Con `unexplained > 0`, `start`
     exige `--accept-owner-only-loss`: esos datos se pierden con la
     compactación. Ese reconocimiento cubre **sólo** los conteos del plan;
   - `advisory`: si la compactación está recomendada y por qué (6 dueños o
     borrados pedidos por el usuario que siguen en estados retenidos);
   - `plan_sha256` y `untouched_sha256`.

3. Iniciar (escribe la intención, crea K, lo materializa, lo autoriza y lo
   activa; el head pasa a K):

   ```bash
   $ST compact start "$K" --archive "$H_ARCHIVE" --plan-sha256 <plan_sha256> \
       --acknowledge-untouched <untouched_sha256> --destroy-rollback-states \
       [--accept-owner-only-loss] [--retire-prepare-leftovers] [--unit <unidad de kza> ...]
   ```

   Las mismas `--unit` que en el plan (forman parte de su digest); el retiro
   vuelve a escanear con esa lista.

   Si el plan cambió desde el paso 2 (otro slot, otra entrada en la raíz),
   rechaza con `runtime_admission_compaction_plan_changed`: repetir el plan.
   La salida trae `compaction_id`.

4. Verificación (una sola corrida de K):

   ```bash
   $ST compact verify <compaction_id> --smoke-seconds 120
   systemctl --user enable kza-voice.service
   systemctl --user start kza-voice.service
   # esperar READY y unos 2 minutos de servicio; no usar el dashboard
   systemctl --user stop kza-voice.service
   systemctl --user disable kza-voice.service
   $ST compact status <compaction_id>    # debe incluir 'verified'
   ```

   Cualquier otro arranque de `kza-voice` sale con 78 mientras la
   compactación esté pendiente (y `RestartPreventExitStatus=78` evita el
   bucle).

5. Retiro (destruye los miembros; irreversible):

   ```bash
   $ST compact retire <compaction_id>
   $ST compact status <compaction_id>    # fase 'complete'
   systemctl --user enable kza-voice.service
   systemctl --user start kza-voice.service
   ```

   Antes de destruir nada, el retiro vuelve a comprobar:
   - identidades, enlaces simbólicos y duros, y que cada slot miembro siga
     teniendo sólo `runtime/` y `state/`, con el **mismo** `state/` del plan
     (si alguien lo reemplazó, rechaza `compaction_identity`);
   - el conjunto de miembros y la lista intacta, recalculados: una entrada
     nueva en la raíz (por ejemplo un slot copiado después del plan) rechaza
     `compaction_plan_changed` con `untouched_added`. Quitarla (o decidir qué
     hacer con ella) y repetir `retire`;
   - que cada miembro siga en su ruta: uno movido entero (por ejemplo a
     `backups/` o `logs/`) o borrado después del plan rechaza
     `compaction_member_missing` con su número. Devolverlo a su ruta y repetir
     `retire`, o, si se decide no destruirlo, `compact retire <id> --member <n>
     --accept-missing`: queda registrado (`missing_accepted`), sin lápida ni
     registro `copies-retired`, y la copia sigue donde esté (el retiro no la
     destruyó y no dice que lo hizo). Un miembro que esta compactación ya
     terminó de recorrer (evento `walked`, que se registra con la cuarentena
     vacía y antes de quitarla) no se puede aceptar así. Uno con `retiring`
     pero sin `renamed` (un corte antes del `rename` y después un movimiento a
     mano), o con `renamed` pero sin `walked` (su cuarentena se sacó a mano a
     mitad del borrado), sí: no hay prueba de que esta compactación lo haya
     borrado entero. Lo aceptado tiene que quedar fuera: si vuelve a su ruta o
     a la cuarentena, `retire` y `abort` rechazan
     `compaction_accepted_member_back` con el número y la ruta. Moverlo otra
     vez fuera de la raíz (o borrarlo a mano) y repetir el comando; la
     aceptación registrada no se deshace;
   - el escaneo de unidades (P14);
   - la cobertura, ahora **contra H** (decisión D2 = A): una fila o archivo que
     sigue en K' está cubierto; lo que estaba en H y la corrida de K borró o
     cambió se cuenta como `removed_by_k_run` y no bloquea (por ejemplo el
     mantenimiento de `live_connections` o un borrado pedido durante la
     verificación). Si aparecen datos sólo en miembros que el plan no tenía,
     rechaza `compaction_owner_only_data` con `new_unexplained` y
     `coverage_sha256`; para aceptarlos: `compact retire <id>
     --accept-owner-only-loss <coverage_sha256>`. El flag de `start` no los
     cubre.

   Por cada miembro escribe `retiring`, lo renombra dentro de la raíz
   (`renamed`), lo borra de abajo hacia arriba sin seguir enlaces (`walked`,
   con la cuarentena ya vacía), quita la cuarentena, escribe la lápida
   (`admission/compaction/v1/retired/<digest>.json`) y después `destroyed`.
   Las copias del catálogo reciben un registro
   `admission/copies-retired/<copy_id>.json`; la cadena del catálogo no se
   modifica.

## Si algo falla

- **Se cortó `start`** (corte de luz, SIGKILL, error): `compact resume <id>
  --archive "$H_ARCHIVE"` sigue desde el último evento registrado, o
  `compact abort <id>` vuelve a H y borra sólo lo que creó esta compactación
  (K, el snapshot, el placeholder, un resto `.prepare-*` con su registro de
  staging). Un corte justo después de renombrar K y antes de su registro de
  `prepare` se reconoce por el registro de staging (mismo inode).
  Una materialización a medias no se rehace: `resume` rechaza con
  `compaction_resume_abort_required`; abortar y empezar de nuevo con **otro**
  nombre de K (el nombre usado queda retirado).
- **La corrida de verificación falló antes de arrancar** (por ejemplo disco
  lleno al escribir el ticket; el head sigue cerrado en K): la corrida no se
  consumió. Otro `systemctl --user start kza-voice.service` la reintenta, o
  `compact abort <id>` (sin `--to-h`) vuelve a H. `allow-run` rechaza
  `compaction_not_verified`.
- **La corrida de verificación falló** (serve sale con 69, head `unknown`): no
  se destruyó nada. Con 69 la unidad **no** se reinicia sola (69 está en
  `RestartPreventExitStatus`). Si hubo recibo, R-a lo cierra en el próximo
  arranque manual (`systemctl --user start kza-voice.service`): el resultado
  esperado es salida 78 con `verified` registrado en `compact status`; después
  seguir con el retiro. Si no hubo recibo, `compact abort <id> --to-h
  --confirm-h <primeros 12+ caracteres del binding de H>`. La recuperación
  automática R-b está apagada mientras haya una compactación pendiente.
- **Volver a H después de verificar** (Q6): `compact abort <id> --to-h
  --confirm-h <prefijo>`. Se rechaza (`compaction_abort_revives`) si durante la
  corrida de K se borró una conversación, un recuerdo o un perfil de voz, o
  si cambió cualquier otra tabla protegida (`compaction_abort_protected_changed`;
  las sesiones del dashboard y los intentos de credencial no cuentan, por eso:
  **no uses el dashboard durante la verificación**). Si K' escribió filas
  nuevas que H no tiene, exige `--discard-k-writes`: esas filas se pierden.
  No está disponible una vez que arrancó una corrida bajo `allow-run`
  (`allow_run_started`); un `allow-run` concedido y nunca usado no lo impide.
- **Hay que servir antes de retirar** (Q7):
  `compact allow-run <id> --reason "<motivo>" --hours <1..72>`. Deja arrancar
  el servicio hasta el plazo; no permite releases ni prepare. Exige K'
  verificado (`compaction_not_verified` si no). Si el servicio se cae durante
  allow-run: `$ST adopt --dry-run` y después `$ST adopt --confirm <sha256 del
  head> [--confirm-regressions <familias exactamente como las lista
  --dry-run>]` (operador); el reinicio automático está apagado mientras haya
  una compactación pendiente. Al vencer el plazo, serve sale con 78 al
  arrancar.
- **Un miembro cambió de identidad** (otro inode con el mismo nombre):
  `retire` rechaza con `compaction_identity`. Revisar qué es. Sólo si es
  seguro borrarlo: `compact retire <id> --member <n> --accept-identity-change`
  (queda registrado; sólo un miembro por vez).
- **Un miembro tiene un enlace simbólico, un enlace duro hacia K, un punto de
  montaje o un archivo especial**: el borrado rechaza sin seguirlo
  (`compaction_symlink`, `compaction_hard_link`, `compaction_mount`,
  `compaction_special_file`, `compaction_cross_device`). Revisar y quitar la
  causa a mano, después `compact resume <id>`.
- **Se cortó `retire`**: `compact resume <id>` termina el retiro; cada paso es
  idempotente y deja el mismo resultado. Mientras el retiro no termine, serve
  sale con 78 (`compaction_retire_in_progress`), aun con un `allow-run`
  vigente.
- **Se cortó `abort`** (en cualquier punto, también entre el cambio del head a
  H y el evento `returned`): repetir el mismo `compact abort <id> ...` o
  `compact resume <id>`. Un head H cerrado con el registro de retorno de esta
  compactación (`admission/compaction/v1/returns/`) cuenta como retorno
  hecho, y `current` se vuelve a publicar hacia H antes de borrar K.
  Desde `abort_started`, serve sale con 78 (`compaction_abort_pending`):
  ni la corrida armada ni un `allow-run` vigente arrancan K. Un abort
  reanudado encuentra cada miembro por el número y la identidad que quedaron
  registrados (también K y un snapshot a medias que ya estaban en
  `compaction/<id>/retiring/`) y termina de borrarlo; `aborted` (y `complete`
  en el retiro) no se registra mientras quede algo en esa cuarentena
  (`compaction_retiring_residue`, con los nombres): revisar qué es. **Nunca
  sacar nada de `compaction/<id>/retiring/`**: lo que está ahí se está
  borrando, y si se saca a medias el controlador no lo cuenta como destruido
  (rechaza `compaction_member_missing`) pero lo que quedó sigue en disco.
- **El abort rechaza `compaction_member_missing`** (por ejemplo, se borró a
  mano `compaction/<id>/snapshot-N` o el `placeholder` para liberar espacio,
  o se movió `slots/<K>` después de construirlo): la salida trae el número
  del miembro (K es `<cantidad de miembros del plan> + 100000`). Si no se
  puede devolver a su ruta, `compact abort <id> --member <n> --accept-missing`
  lo deja fuera, registrado (`missing_accepted`), sin lápida ni registro
  `copies-retired`, y sigue con el abort. Repetir con el próximo número si
  falta más de uno. Rechaza (`compaction_member`) si el miembro sigue en su
  ruta o en la cuarentena, o si esta compactación ya terminó de borrarlo
  (`walked`). Lo aceptado no se borra: **K movido es una copia completa de
  H** (conversaciones, recuerdos, perfiles de voz); si ya no se quiere,
  borrarlo a mano donde quedó. Si algo aceptado vuelve a su ruta, el abort
  rechaza `compaction_accepted_member_back` (ver el paso 5).
- **El head cambió después de empezar el abort** (una corrida de K que la
  compuerta debió impedir, por ejemplo desde un controlador viejo): `resume` y
  `abort` rechazan `compaction_abort_recheck_required`. Correr `compact abort
  <id> --to-h --confirm-h <prefijo> [--discard-k-writes]`: vuelve a aplicar
  Q6 y el descarte a lo que corrió (rechaza `compaction_abort_revives` si esa
  corrida borró una conversación, un recuerdo o un perfil de voz) y registra
  `abort_rechecked`.
- **La re-comprobación rechaza `compaction_abort_revives`**: volver a H
  resucitaría lo que la persona borró durante esa corrida, así que el
  controlador no lo hace, y ningún comando de esta versión sale de ese
  estado (`resume` y `abort` piden la re-comprobación, `retire` y `allow-run`
  rechazan después de `abort_started`). El servicio queda detenido (serve
  sale con 78) y los datos intactos: H y K' siguen en disco, y K' es el que
  conserva el borrado. Qué hacer:
  1. No editar el log, no borrar K ni H, no usar `--discard-k-writes` como
     salida (no cambia el rechazo) y no correr un `stage_runtime` viejo.
  2. Guardar evidencia: copiar `admission/compaction/v1/<id>/` fuera de la
     raíz y anotar `compact status <id>`.
  3. Averiguar quién corrió K detrás del abort (un controlador viejo o una
     copia de época 2) y retirarlo.
  4. Salir con un controlador corregido (decisión D4 = A, «Compactación
     trabada», paso 3) que siga **desde K'**: anular el abort que nunca volvió
     a H y retirar desde K', que es la dirección que conserva el borrado.
     Volver a H no es una salida.

## Compactación trabada

Mientras haya una compactación pendiente, sólo pasan los comandos `compact`,
la corrida de verificación armada, las corridas de un `allow-run` vigente e
`install-controller`. `install-launcher`, `prepare`, `activate`, `run`,
`delegate` y la adopción rechazan con `runtime_admission_compaction_pending`.

1. Ver el estado: `$ST compact status`. Cada compactación muestra su fase; un
   log ilegible aparece como `{"compaction_id": ..., "invalid": true, "code":
   "runtime_admission_compaction_log_invalid"}` en vez de un traceback.
2. Si el log es válido, seguir lo que diga la fase: `compact resume <id>`
   (con `--archive` si todavía no existe K), `compact abort <id>` antes de la
   corrida, `compact abort <id> --to-h --confirm-h <prefijo>` después, o
   `compact retire <id>`.
3. Si un comando falla siempre con el mismo código y no es un problema del
   entorno (espacio, permisos), puede ser un defecto del propio controlador.
   La salida es un controlador corregido (decisión D4 = A):
   `$ST install-controller <archivo nuevo>` funciona aunque haya una
   compactación pendiente o un log inválido, siempre que el archivo sea de
   época 3 o posterior y no menor que la instalada. Después, repetir el paso 2
   con el `$ST` del controlador nuevo.
4. Log inválido (por ejemplo un evento truncado por un corte de luz en un
   sistema de archivos sin orden de escritura): todos los comandos que toman
   el lock rechazan y serve sale con 78 (falla cerrado). Antes de tocar nada,
   copiar `admission/compaction/v1/<id>/` completo fuera de la raíz como
   evidencia. Sólo si el **último** archivo de `events/` está vacío o
   truncado, moverlo fuera de la raíz y volver a `compact status`: el paso
   que registraba se rehace o se reconoce de forma idempotente. Cualquier otro
   daño (un hueco en la numeración, una cadena rota en el medio, un evento
   ajeno): no editar a mano; instalar un controlador corregido (paso 3) o
   pedir ayuda con la evidencia guardada.

## Códigos de rechazo frecuentes

| Código | Qué significa |
|---|---|
| `runtime_admission_compaction_pending` | Hay una compactación sin terminar: sólo los comandos `compact` y la corrida de verificación pasan. |
| `runtime_admission_compaction_plan_changed` | El plan o la lista intacta cambió desde `plan`. |
| `runtime_admission_compaction_launcher_missing` / `_launcher_contract` | Falta el controlador o el lanzador, o su contrato es menor que 3. |
| `runtime_admission_compaction_unit_path:<unidad>` | Una unidad systemd o logrotate nombra un miembro. |
| `runtime_admission_compaction_absolute_locator` | `config/` o `.env` de H nombran una ruta absoluta de H, de un dueño o de un miembro. |
| `runtime_admission_compaction_capacity` | El binding o el archivo histórico de K pasaría de 12 MiB, o no hay espacio. |
| `runtime_admission_compaction_owner_only_data` | Hay datos sólo en miembros; en `start` requiere `--accept-owner-only-loss`, en `retire` (datos nuevos desde el plan) `--accept-owner-only-loss <coverage_sha256>`. |
| `runtime_admission_compaction_identity` | Un miembro o su `state/` ya no es el del plan, o un slot miembro tiene entradas nuevas. |
| `runtime_admission_compaction_log_invalid` | Un log de compactación ilegible; ver «Compactación trabada». |
| `runtime_admission_compaction_database_unreadable` | Un archivo `*.db` de un miembro no es SQLite legible (la salida no incluye la ruta). |
| `runtime_admission_compaction_io_error` | Error de E/S, con su `errno` (sin ruta). |
| `runtime_admission_controller_downgrade` | Este `stage_runtime` es de una época menor que la instalada: usar `controller/current`. |
| `runtime_admission_compaction_not_verified` | `retire` antes de que K' cerrara limpio. |
| `runtime_admission_retired_locator` | Un nombre o ruta ya retirado por una compactación; usar otro nombre. |
| `runtime_admission_compaction_abort_pending` / `_retire_in_progress` | serve: hay un abort o un retiro empezado; terminarlo con `compact`. |
| `runtime_admission_compaction_abort_recheck_required` | El head cambió después de `abort_started`: `compact abort <id> --to-h --confirm-h ...`. |
| `runtime_admission_compaction_member_missing` | Un miembro (o K en el abort) ya no está en su ruta y esta compactación no terminó de borrarlo (sin `walked`). En el retiro, ver el paso 5; en el abort, «El abort rechaza `compaction_member_missing`». En ambos: `--member <n> --accept-missing`. |
| `runtime_admission_compaction_accepted_member_back` | Un miembro aceptado como faltante volvió a su ruta o a la cuarentena (la salida trae número y ruta): moverlo otra vez fuera de la raíz o borrarlo a mano, y repetir. |
| `runtime_admission_compaction_retiring_residue` | Quedó algo en `compaction/<id>/retiring/`; no se registra el final. |
| `runtime_admission_compaction_copy_in_progress` | Una copia del catálogo se está preparando (o su estado es desconocido). |
| `runtime_admission_compaction_unit_unparsed:<unidad>:LoadState` | Una unidad pasada con `--unit` no existe para systemd. |

## Decisiones tomadas en la implementación (la opción más estricta)

- La fase de datos (snapshot y clon) corre en un proceso aparte con la
  capacidad `KZA_COMPACTION_WORKER=1` y el bootstrap exacto de
  `src.core.entry_guard`, porque el controlador no importa `src` bajo una raíz
  gestionada (G1).
- `utterances.audio_path` se reubica; cualquier otra ruta absoluta de H o de un
  dueño dentro de los datos rechaza.
- D5 compara también byte a byte los archivos opacos al SDK.
- Volver a H usa un head manual nuevo, `compaction_return`.
- La época del controlador sube a 3. Un controlador de época 2 no puede
  reemplazar a uno de época 3 con `install-controller`, pero una copia de
  época 2 (un slot viejo o un `controller/<commit>` viejo) **sí** puede
  correrse directamente y no conoce la compuerta: por eso R1 se construye ya
  con U3b-0c (D1 = A) y el runbook prohíbe correr `stage_runtime` desde otro
  lugar que `controller/current`. Desde la época 3 hay comprobación hacia
  adelante.
- Retiro, cobertura contra H: las filas de arrendamiento efímero
  (`live_connections`, `live_resets`) **no** tienen una clase propia; cuando la
  corrida de K las borra cuentan como `removed_by_k_run`, y si existen sólo en
  un miembro siguen siendo `unexplained` (más estricto que la clase
  «ephemeral» que proponía la revisión).
- `--accept-owner-only-loss` de `start` no se arrastra al retiro: datos nuevos
  desde el plan exigen reconocer el digest de la cobertura del retiro.
- El retiro recalcula el plan y rechaza cualquier diferencia (más estricto que
  pedir un reconocimiento nuevo).
- Una referencia `$VAR` sin resolver en una línea `Exec*` rechaza en las
  unidades nombradas o que mencionan kza, un miembro o la raíz (regla de
  U3b-0e4, mismo criterio `mentions_markers`). Un valor del entorno del
  gestor con escapes (`$'...'`) no se **sustituye** en las unidades (una
  referencia a él cuenta como sin resolver), pero sí se decodifica para
  compararlo con los miembros (ver «Antes de empezar»). La salida del plan
  lista por nombre las unidades toleradas (`units.tolerated_unresolved`).
- Un miembro que pasa a contener un enlace simbólico después de `start` no se
  reclasifica: el retiro se detiene sin recorrerlo hasta que el operador quite
  el enlace.
- Segunda revisión (N1-N10): el abort reanudado tras un cambio de head exige
  `--to-h` y la confirmación de H aunque el abort original fuera «antes de la
  corrida»; una plantilla se juzga por la parte fija de cada ruta (más
  estricto que esperar a que exista una instancia); un `kza` dentro de otra
  ruta ya no marca una unidad (el nombre `kza*` volvió a marcarla en la
  tercera revisión);
  una copia catalogada sin extents ni inventario no prueba ninguna entrada y
  queda intacta.
- Tercera revisión (R3-1 a R3-7): una unidad con nombre `kza*` vuelve a ser
  «de kza» para la tolerancia (lo estricto es el lado seguro), igual en la
  adopción de U3b-0e4; una palabra de plantilla que empieza con `%i` o con un
  `$VAR` sin valor rechaza en una plantilla de kza y sólo se tolera, listada,
  en una plantilla ajena con la regla de U3b-0e4; el dueño de una copia en
  preparación se reconoce por pid y hora de arranque; el abort acepta un
  miembro faltante con el mismo reconocimiento que el retiro; `retiring` sin
  `renamed` nunca prueba un borrado.
- Cuarta revisión (R4-1 a R4-7): en las plantillas y en el entorno del
  gestor la regla es de subcadena (la raíz o un miembro en cualquier lugar del
  texto, también en un comentario, rechaza) en vez de sumar una forma de
  palabra más; cualquier variable dentro del guion de un `sh -c` de una
  plantilla de kza rechaza, también las del propio shell (`$$f`); los valores
  de entorno con `$` cuentan como desconocidos. Sólo prueba un borrado
  `walked` (no `renamed`). Un miembro aceptado que vuelve rechaza con un
  mensaje propio en vez de aceptarse solo: el operador decide de nuevo. K
  movido a mano en un abort se acepta con el mismo comando y no se borra. Un
  enlace del home es marcador de kza sólo con su ruta o un prefijo de home,
  nunca su nombre suelto (`app` es una palabra común).
- Quinta revisión (R5-1, R5-2): un comodín de shell en un trozo de ruta
  absoluto se juzga con la regla de la parte fija en las unidades normales,
  en las plantillas y en el entorno del gestor, comparando cada componente
  como lo expande el shell (antes, en el entorno del gestor, todo lo que
  seguía directamente al home rechazaba; ahora `/home/kza/*.log` pasa
  porque no puede ser `releases`). Un `[...]`, un `..` o una parte
  desconocida después del comodín se tratan como «cualquier cosa». Los
  enlaces del home a la raíz cuentan como la raíz también para `%i`.
  Los comodines relativos no se juzgan (riesgo residual aceptado: un
  `cd <raíz> && tar c slots/*` en una unidad normal pasa, igual que
  `cd <raíz> && rm -rf slots/a` antes de esta revisión). En las plantillas
  los escapes C se decodifican como systemd (comprobado en systemd 255 de
  kza-yo-lan1); el texto decodificado también decide si la plantilla es de
  kza para la tolerancia de U3b-0e4, sin tocar la regla compartida con la
  adopción. Quedan fuera la expansión de llaves de bash
  (`rel{e,}ases`) y los extglob/zsh (`@(releases)`): ofuscación
  deliberada, no una unidad doméstica.
- Antes y después de la verificación se usa `disable` (no `mask`), como en la
  ventana R1: la unidad vive en `~/.config/systemd/user` y `mask` no puede
  reemplazarla.
