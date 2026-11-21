# P6 · Manifiesto revisable y guion de release de KZA

T064 prepara la revisión del candidato y el guion de P7. Q4 fija una única
adopción directa en `SCHEMA_CONTRACT_EPOCH = 2` desde el contenedor
`release_20261029T185008966364Z_e9669ec`
(`e9669ec1edb101ca6b190ef8df09314e1ac63562`), cuyo estado legado corresponde
al slot `release_20261028T145442Z_b0866e8`. Por Q8, durante las 72 h del soak
T061 se preparan los guiones de P7 y P8. La intervención en producción
requiere el GO registrado de Q17 para el candidato exacto, con fecha,
operador, acciones, duración y plan de reversión.
[Precedencia y decisiones](../plans/2026-11-21-precedence-roadmap-and-test-strategy.md)
· [Contrato de esquema y despliegue](../architecture/DEPLOYMENT.md#contrato-de-esquema-época-2-schema-v2)

## Sellado y revisión bajo `yo`

En el checkout de desarrollo en Linux, definir `PY` con el intérprete del
venv CPython 3.13, `COMMIT` con el SHA completo integrado de P6 y `REVIEW`
con un directorio privado de evidencia fuera de Git. `LOCK` apunta al lock
resuelto de runtime para la plataforma objetivo. Guardar el lock y sus
artefactos de instalación fuera de Git y reproducir el entorno con sus
versiones transitivas y hashes de distribución. `requirements.txt` declara
rangos de versiones y `requirements-test.txt` describe el entorno de pruebas.

```bash
"$PY" scripts/build_runtime.py "$REVIEW/candidate.tar.gz" --revision "$COMMIT"
"$PY" scripts/build_runtime.py "$REVIEW/rebuilt.tar.gz" --revision "$COMMIT"
cmp "$REVIEW/candidate.tar.gz" "$REVIEW/rebuilt.tar.gz"
sha256sum "$REVIEW/candidate.tar.gz" "$REVIEW/rebuilt.tar.gz"
```

Registrar la plataforma, la versión completa del intérprete y los dos
resultados de T063. Detener el sellado ante una diferencia. Copiar el
SHA-256 del registro de build revisado a `ARCHIVE_SHA256`.

```bash
"$PY" -B -m tools.release_manifest_review "$REVIEW/candidate.tar.gz" \
  --commit "$COMMIT" --archive-sha256 "$ARCHIVE_SHA256" --runtime-lock "$LOCK" \
  > "$REVIEW/release-review.json"
"$PY" -B -m tools.release_manifest_review "$REVIEW/candidate.tar.gz" \
  --commit "$COMMIT" --archive-sha256 "$ARCHIVE_SHA256" --runtime-lock "$LOCK" \
  --verify "$REVIEW/release-review.json"
```

Exigir salida 0 en ambos comandos. El JSON registra el SHA completo, hash
del archivo, ABI `cp313`, magic de CPython, inventario sin fuentes, digest
canónico Epoch 2 y hashes de las declaraciones Git y del lock externo. El
revisor ejecuta bytecode del artefacto sellado en un directorio temporal
privado y elimina el scratch al salir. El ensayo valida `current` y los
cinco enlaces del contenedor sintético con las comprobaciones de adopción
y L4, y ejecuta la fase *schema-only* sobre una copia SQLite.
Conserva las filas de recordatorios y de una acción HA ya enviada, crea las
tablas de Epoch 2 vacías y compara byte a byte la base congelada.

El hash del lock acredita su identidad. Adjuntar la evidencia de instalación
reproducible y validación de la configuración efectiva de arranque de la
plataforma objetivo, incluida la comprobación de que el entrenamiento
nocturno sigue deshabilitado si su utilidad y recursos carecen de
aceptación. Conservar la configuración privada y los datos reales en su
entorno de ejecución. El expediente lleva identificadores, hashes,
resultados y conteos.

### Evidencia que acompaña al candidato

| Criterio | Evidencia que se adjunta |
| --- | --- |
| C02 | Matriz T060 con al menos 200 negativos/fallos y controles de acciones inciertas, reenvío, confirmaciones y resurrección. En el ensayo T063, restaurar una copia sintética y comprobar que no reaparecen datos borrados ni se reenvían acciones. |
| C05 | Identidad y reproducibilidad T063, arranque y restore fuera del árbol de fuentes, lock verificado, configuración de arranque, ensayo de runtime y datos para L4. Tier 2 y `compiled-solo` del SHA sellado, sin fallos ni xfail en requisitos incluidos. Documentar warnings nuevos de aplicación y su resolución o justificación. |
| C19 | T061 y T074 con el mismo candidato sellado. Soak local de 72 h y bloque físico fijo de al menos 72 h dentro de siete días. Para KZA y cada entrada requerida, reportar segundos útiles / segundos de ventana (≥99,9 % por serie), huecos, pausas autenticadas según política y disponibilidad total. Prerregistrar fallos inducidos antes/después del bloque y distinguirlos de fallos espontáneos. |

Guardar junto al manifiesto los resúmenes, los hashes y el resultado de la
revisión. T064 aporta el preflight de esquema y el guion. T063, Tier 2, T061
y P7 aportan las pruebas de restore completo, systemd y operación física.
Un cambio de identidad requiere volver a sellar y verificar la evidencia
ligada al candidato. [Criterios y tareas](../plans/2026-11-02-autonomous-roadmap.json)

## Adopción directa en P7

1. El operador autorizado verifica el `current` legado, commit completo,
   cinco enlaces, estado congelado, espacio, swap, estado de las unidades
   y timers, cero dueños de captura y expediente Q17 completo. Usar la
   secuencia W0–W10 del [runbook de adopción](2026-11-13-r1-adoption-window.md)
   con el archivo sellado Epoch 2 de P6 en W0–W8. El archivo histórico R1
   de contrato 1, su pin y su valor histórico de `controller_epoch` quedan
   fuera de esta ventana. Usar la época y revisión del controlador que
   declara el candidato.
2. Verificar el SHA-256 transferido contra el manifiesto antes de extraer,
   antes del dry-run y antes de W1. Extraer el archivo verificado en un
   directorio privado nuevo y vacío. Corroborar `git_commit` y
   `schema_contract: 2` de su `release.json`. Ejecutar el controlador
   compilado de esa extracción en el dry-run, W2 y W3. Las funciones
   siguientes representan el comando del operador, con `PY`, `ROOT` y
   `EXTRACTED` fijados en el handoff de P7.

```bash
C=release_20261029T185008966364Z_e9669ec
staged() {
  "$PY" -I -B "$EXTRACTED/scripts/stage_runtime.pyc" --root "$ROOT" --offline "$@"
}
installed() {
  "$PY" -I -B "$ROOT/controller/current/scripts/stage_runtime.pyc" --root "$ROOT" --offline "$@"
}
staged adoption-dry-run --container-slot "$C" --archive "$ARCHIVE" \
  --slot "$CANDIDATE_SLOT" "${WINDOW_TIMER_ARGS[@]}"
```

3. Antes de W1, exigir `problems: []`, `source_moving.torn = 0` cuando
   aparezca, scratch liberado, sin dependencias inversas ni targets ajenos
   y cero residuos de voz/consentimiento/conversación sin destino. Revisar
   warnings, temporizadores, tamaños, disponibilidad de scratch y censo del
   diario de borrado. Parar ante una precondición incumplida. W1 detiene y
   deshabilita la unidad, para los timers acordados y comprueba que no hay
   escritores. W1b sólo se usa con el hallazgo y autorización de retiro de
   entrenamiento que exige el runbook de adopción.
4. Aplicar W2–W4 en orden. `WINDOW_TIMER_ARGS` contiene un par
   `--window-timer nombre.timer` por timer acordado en W0. En W4,
   `PREPARE_PLAN_SHA256` es el de W2 y `WINDOW_DECISION` corresponde a Q17.

```bash
staged adoption-prepare --container-slot "$C" --archive "$ARCHIVE" \
  --slot "$CANDIDATE_SLOT" --decision "$WINDOW_DECISION" "${WINDOW_TIMER_ARGS[@]}"
staged install-controller "$ARCHIVE"
installed adoption-commit --confirm "$PREPARE_PLAN_SHA256"
```

5. Completar W5–W8 con el logrotate, lanzador gestionado y presupuesto
   systemd revisados. Medir READY. Ante un fallo antes de iniciar W8, usar
   W-abort-1 o W-abort-2 según exista commit de adopción, siempre que el
   controlador confirme sus precondiciones. Desde el head `selection` de
   W8 o si existe un ticket, aplicar el guion L4 si el usuario lo autoriza.
   Ante un fallo de un comando, conservar salida, estado y registros para
   diagnosticar antes de seguir. El límite de W1 a READY es de 60 minutos.
6. W9 exige un cierre limpio con recibo y sello. Registrar su `ended_at`
   en UTC como DÍA 0 y registrar DÍA 0 + 1 209 600 s como fecha de Q5.
   Reiniciar, comprobar READY y leer `installed completion-status` hasta
   que terminen las colas de privacidad. Registrar conteos pendientes y
   bloqueantes según W9. Ejecutar T074 dentro de la ventana acordada y
   registrar la decisión T075 de promover o volver a la imagen congelada.

## L4 · Vuelta a e9669ec

Desde W8 puede requerirse L4 aunque todavía no exista el primer cierre
limpio. El plazo de 14 × 24 h se cuenta desde ese primer cierre y el guion
limita L4 hasta su vencimiento, previo a Q5. La intención de retiro
invalida los destinos legados en el controlador. Volver a la imagen
congelada puede volver a servir datos borrados y dejar fuera cambios hechos
en el estado gestionado. El usuario autoriza esa consecuencia con una
decisión L4 registrada y la frase literal `revivir datos borrados`.

1. Verificar el destino exacto y su integridad, que resuelvan todos sus
   enlaces y que no haya transición pendiente del modo release ni intención
   Q5. Parar la unidad, comprobar que sigue inactiva y que no quedan
   escritores. Desde el controlador instalado ejecutar sólo L4. `UNIT_DIR`
   corresponde al directorio de drop-ins de `kza-voice.service`.

```bash
systemctl --user stop kza-voice.service
installed uninstall-launcher --unit-dir "$UNIT_DIR" \
  --user-decision "$L4_DECISION" --phrase 'revivir datos borrados' --legacy-slot "$C"
```

2. Antes de arrancar, exigir salida 0 y `mismatches: []`, y verificar por
   `systemctl --user show` el `Type=simple`, el `WorkingDirectory` y el
   `PYTHONPATH` explícitos del contenedor e9669ec. Así se sirve el runtime
   legado con sus datos congelados. Registrar `uninstall-*.json`, arranque
   y health autenticado, y sólo reactivar los timers acordados. Si falla
   alguno de los controles, mantener la unidad detenida y escalar al
   operador. [Contrato de L4](../architecture/DEPLOYMENT.md#lanzador-gestionado-u3b-0d--desarrollado-no-instalado)

## Q5 · Retiro al día 14

El operador de P7 calcula la fecha exacta desde el sello W9. Sin sello,
antes del vencimiento o con un `uninstall-*.json` de L4, el retiro queda
bloqueado. Si se usó L4, los datos de ese período requieren otra decisión
del usuario.

El día acordado, revisar las referencias de unidades, enlaces,
configuración y registros, y retirar la estrofa de logrotate de la consola
legada según el [procedimiento Q5](2026-11-13-r1-adoption-window.md#día-14-retiro-q5-del-estado-de-b0866e8-y-sus-copias-del-2026-10-29).
Ejecutar un dry-run con la decisión P7 registrada y el contenedor e9669ec.
Exigir `eligible: true` y revisar el censo del estado congelado, informes,
dos conjuntos de copias SQLite y slots superados, con sus exclusiones.

```bash
installed retire-legacy-state --decision "$P7_DECISION" --container-slot "$C" --dry-run
# Después de revisar el plan y su SHA, detener y verificar el cierre gestionado.
systemctl --user stop kza-voice.service
installed retire-legacy-state --decision "$P7_DECISION" --container-slot "$C" \
  --confirm "$Q5_PLAN_SHA256"
```

El retiro es irreversible. Si se corta con una intención ya escrita,
reasumir con la misma decisión y el mismo `--confirm`. Verificar el
registro `retired-<plan>.json` y la prueba de ausencia antes de arrancar.
Guardar en el handoff el censo, la evidencia de health y que la reversión
L4 quedó cerrada. Los cambios posteriores avanzan por el flujo gestionado.
