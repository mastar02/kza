# Compactación de estados dueños (U3b-0c)

Runbook del operador para `stage_runtime compact`. Diseño:
`owner-compaction-design-v2.json` a `v5` (con las correcciones E1 a E3 de la
revisión de v5) y las decisiones del usuario `u3b_0c_compaction` (P1 a P7) y
`u3b_0c_compaction_v2` (Q1 a Q7, VACUUM sólo en el estado nuevo). Evidencia:
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
  delegación;
- restos `slots/.prepare-*` con registro de staging (sólo con
  `--retire-prepare-leftovers`) y restos `.activate-*` vacíos o con un único
  enlace `current`;
- copias del catálogo dentro de la raíz, y el trabajo propio de compactaciones
  anteriores.

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
- **Nada no probado dentro de la raíz**: `backups/`, `logs/`, `deployments/`,
  `dashboard-proxy/`, `controller/`, archivos de release, `console-logrotate.conf`,
  slots sin prueba (por ejemplo, el contenedor de enlaces de 63ea8d3 o el slot
  de b0866e8), restos `.prepare-*` sin registro. Se listan por nombre y tipo
  (sin abrirlos) en la «lista intacta», y el operador la reconoce por su
  digest. El retiro de b0866e8 es el de Q5, no éste.

## Antes de empezar

Requisitos (el comando los verifica y rechaza con un código si falta alguno):

- controlador época 3 o posterior instalado en `controller/current`
  (`install-controller`), y el lanzador instalado con contrato 3 o posterior
  (`install-launcher`; P13). La release de H (y por lo tanto de K) declara
  `launcher_contract >= 3`;
- head cerrado (no `running` ni `unknown`), sin retiro Q5 pendiente;
- servicio detenido y deshabilitado (P5: ningún escritor abierto en H, en los
  dueños ni en los miembros);
- ninguna unidad systemd del usuario ni configuración de logrotate nombra un
  miembro (P14). El escaneo falla cerrado si no hay bus de usuario;
- espacio libre para un snapshot más K, y binding de K de 12 MiB o menos (P9).

**No reinicies el servidor durante una compactación.** Si hace falta, primero
`compact abort` (antes de la verificación) o terminá el retiro.

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
   $ST compact plan "$K" --archive "$H_ARCHIVE" [--retire-prepare-leftovers]
   ```

   Revisar, sin datos personales en la salida:
   - `members`: cada miembro con su clase, prueba y conteos (archivos, bytes);
   - `untouched`: lo que queda intacto. Si algo que debería borrarse aparece
     acá, **no sigas**: no hay prueba de que el controlador lo haya hecho;
   - `coverage`: filas y archivos que existen sólo en miembros, por clase
     (`expired`, `erased`, `superseded`, `scrubbed`, `unexplained`). Con
     `unexplained > 0`, `start` exige `--accept-owner-only-loss`: esos datos se
     pierden con la compactación;
   - `advisory`: si la compactación está recomendada y por qué (6 dueños o
     borrados pedidos por el usuario que siguen en estados retenidos);
   - `plan_sha256` y `untouched_sha256`.

3. Iniciar (escribe la intención, crea K, lo materializa, lo autoriza y lo
   activa; el head pasa a K):

   ```bash
   $ST compact start "$K" --archive "$H_ARCHIVE" --plan-sha256 <plan_sha256> \
       --acknowledge-untouched <untouched_sha256> --destroy-rollback-states \
       [--accept-owner-only-loss] [--retire-prepare-leftovers]
   ```

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

   El retiro vuelve a comprobar identidades, enlaces duros, el escaneo de
   unidades (P14) y la cobertura antes de empezar. Por cada miembro escribe
   `retiring`, lo renombra dentro de la raíz, lo borra de abajo hacia arriba
   sin seguir enlaces, escribe la lápida (`admission/compaction/v1/tombstones/`)
   y después `destroyed`. Las copias del catálogo reciben un registro
   `admission/copies-retired/<copy_id>.json`; la cadena del catálogo no se
   modifica.

## Si algo falla

- **Se cortó `start`** (corte de luz, SIGKILL, error): `compact resume <id>`
  sigue desde el último evento registrado, o `compact abort <id>` vuelve a H y
  borra sólo lo que creó esta compactación (K, el snapshot, el placeholder).
  Una materialización a medias no se rehace: `resume` rechaza con
  `compaction_resume_abort_required`; abortar y empezar de nuevo con **otro**
  nombre de K (el nombre usado queda retirado).
- **La corrida de verificación falló** (serve sale con 69, head `unknown`): no
  se destruyó nada. Si hubo recibo, la recuperación R-a lo cierra sola al
  próximo arranque. Si no, `compact abort <id> --to-h --confirm-h <primeros 12+
  caracteres del binding de H>`. La recuperación automática R-b está apagada
  mientras haya una compactación pendiente.
- **Volver a H después de verificar** (Q6): `compact abort <id> --to-h
  --confirm-h <prefijo>`. Se rechaza (`compaction_abort_revives`) si durante la
  corrida de K se borró una conversación, un recuerdo o un perfil de voz, o
  si cambió cualquier otra tabla protegida (`compaction_abort_protected_changed`;
  las sesiones del dashboard y los intentos de credencial no cuentan, por eso:
  **no uses el dashboard durante la verificación**). Si K' escribió filas
  nuevas que H no tiene, exige `--discard-k-writes`: esas filas se pierden.
  No está disponible después de un `allow-run`.
- **Hay que servir antes de retirar** (Q7):
  `compact allow-run <id> --reason "<motivo>" --hours <1..72>`. Deja arrancar
  el servicio hasta el plazo; no permite releases ni prepare. Si el servicio
  se cae durante allow-run: `$ST recover --adopt` (operador); el reinicio
  automático está apagado mientras haya una compactación pendiente. Al vencer
  el plazo, serve sale con 78 al arrancar.
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
  idempotente y deja el mismo resultado.

## Códigos de rechazo frecuentes

| Código | Qué significa |
|---|---|
| `runtime_admission_compaction_pending` | Hay una compactación sin terminar: sólo los comandos `compact` y la corrida de verificación pasan. |
| `runtime_admission_compaction_plan_changed` | El plan o la lista intacta cambió desde `plan`. |
| `runtime_admission_compaction_launcher_missing` / `_launcher_contract` | Falta el controlador o el lanzador, o su contrato es menor que 3. |
| `runtime_admission_compaction_unit_path:<unidad>` | Una unidad systemd o logrotate nombra un miembro. |
| `runtime_admission_compaction_absolute_locator` | `config/` o `.env` de H nombran una ruta absoluta de H, de un dueño o de un miembro. |
| `runtime_admission_compaction_capacity` | El binding o el archivo histórico de K pasaría de 12 MiB, o no hay espacio. |
| `runtime_admission_compaction_owner_only_data` | Hay datos sólo en miembros; requiere `--accept-owner-only-loss`. |
| `runtime_admission_compaction_not_verified` | `retire` antes de que K' cerrara limpio. |
| `runtime_admission_retired_locator` | Un nombre o ruta ya retirado por una compactación; usar otro nombre. |

## Decisiones tomadas en la implementación (la opción más estricta)

- La fase de datos (snapshot y clon) corre en un proceso aparte con la
  capacidad `KZA_COMPACTION_WORKER=1` y el bootstrap exacto de
  `src.core.entry_guard`, porque el controlador no importa `src` bajo una raíz
  gestionada (G1).
- `utterances.audio_path` se reubica; cualquier otra ruta absoluta de H o de un
  dueño dentro de los datos rechaza.
- D5 compara también byte a byte los archivos opacos al SDK.
- Volver a H usa un head manual nuevo, `compaction_return`.
- La época del controlador sube a 3; un controlador de época 2 no conoce la
  compuerta y no puede reemplazar a uno de época 3.
- Un miembro que pasa a contener un enlace simbólico después de `start` no se
  reclasifica: el retiro se detiene sin recorrerlo hasta que el operador quite
  el enlace.
- Antes y después de la verificación se usa `disable` (no `mask`), como en la
  ventana R1: la unidad vive en `~/.config/systemd/user` y `mask` no puede
  reemplazarla.
