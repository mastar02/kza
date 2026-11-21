# P6 · Bytecode reproducible y restauración compatible

El bloque Bash corresponde al host Linux `kza-yo`. Ejecutarlo antes de P7
como `yo` desde el checkout de desarrollo `/home/yo/projects/kza`, con las
dependencias del candidato en `.venv` y CPython 3.13 (ABI `cp313`). En Mac,
adaptar las comprobaciones de usuario y ruta a la cuenta y al checkout de
desarrollo autorizados. Seleccionar el commit completo del candidato y
confirmar que el árbol versionado está limpio.

```bash
set -eu
test "$(id -un)" = yo
test "$(git rev-parse --show-toplevel)" = /home/yo/projects/kza
test -z "$(git status --porcelain --untracked-files=no)"
COMMIT="$(git rev-parse --verify HEAD^{commit})"
PY="$PWD/.venv/bin/python"
EVIDENCE="$(mktemp -d /tmp/kza-p6-t063.XXXXXXXX)"
KZA_P6_REVISION="$COMMIT" "$PY" scripts/run_test_suite.py \
  --tier 0 --workers 1 --no-node --out-dir "$EVIDENCE/logs" \
  --basetemp "$EVIDENCE/work" \
  --targets tests/integration/test_p6_bytecode_restore_reproducibility.py

TASK="$(dirname "$(find "$EVIDENCE/work/tier0" -name build-proof.json -type f -print)")"
for BUILD in a b; do
  test -z "$(find "$TASK/$BUILD/runtime" -name '*.py' -print)"
done
cmp "$TASK/a/runtime/release.json" "$TASK/b/runtime/release.json"
cmp "$TASK/a/runtime.tar.gz" "$TASK/b/runtime.tar.gz"
sha256sum "$TASK"/{a,b}/runtime.tar.gz "$TASK"/{a,b}/runtime/release.json
```

El ensayo invoca `scripts/build_runtime.py --revision` en dos procesos y
directorios temporales independientes. `build-proof.json` conserva los
SHA-256 del artefacto, del manifiesto y de cada `.pyc` extraído. El ensayo
verifica cada archivo contra su manifiesto y ejecuta los puntos de entrada
compilados desde un proceso `-I -S -B` aislado. `boot.log` registra la salida
del CLI de preparación.

La prueba usa datos sintéticos bajo `--basetemp`. El controlador compilado
cierra el estado original O, retira una conversación y cierra el estado N.
Con ambos snapshots cerrados, `materialize_derived_state` prepara la copia
en un slot aparte bajo la autoridad de N. El ensayo verifica que las filas y
blobs retirados están ausentes, que el contenido de los hechos y preferencias
borrados es nulo (`content IS NULL`) en `privacy.db` y que los identificadores
retirados están ausentes de la tabla `embeddings` y de la lectura con el SDK de
Chroma. Los datos testigo vigentes se conservan y la base de N que contiene
`reminders`, `timers` y `alert_preferences` coincide byte por byte con la de
la copia. La copia permanece fuera de servicio y el head sigue en N.

En el almacén Chroma copiado, las entradas anteriores de `embeddings_queue`
y los segmentos pueden conservar físicamente texto y vectores hasta la
reconstrucción del almacén en la compactación SC-4c. El último registro de
cada identificador retirado en la cola es `DELETE` (`op=3`), por lo que esos
residuos no vuelven a formar parte del índice lógico.

Para la revisión, conservar `tier0.log`, `tier0.xml`, `summary.json`,
`build-proof.json`, `boot.log` y `restore-proof.json`. El informe de
restauración vincula las dos ejecuciones V4, los hashes de ambos snapshots y
el plan materializado con el SHA del candidato de Epoch 2. Esta evidencia
local cubre el alcance de T063. La aceptación de C04 y C05 sigue el roadmap.
