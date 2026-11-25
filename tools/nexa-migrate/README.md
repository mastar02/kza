# Migración offline de KZA a Nexa v2

Ejecutar con Python 3.13 desde un checkout que contenga los esquemas Rust.
El migrador usa la biblioteca estándar de Python y carga `sqlite-vec` 0.1.7
para crear y validar las tablas vectoriales exigidas por `nexa-memoryd`. Indicar
la ubicación de la extensión cargable de esa versión mediante `NEXA_SQLITE_VEC`.

```sh
export NEXA_SQLITE_VEC=/ruta/privada/vec0.so
python3.13 tools/nexa-migrate/nexa_migrate.py --source /ruta/kza-data --dest /ruta/nexa-data --dry-run
python3.13 tools/nexa-migrate/nexa_migrate.py --source /ruta/kza-data --dest /ruta/nexa-data
```

Detener los escritores de KZA antes de evaluar la migración o hacer el cutover.
El padre del destino debe existir. En macOS, la fuente y el staging deben estar
en el mismo volumen APFS para clonar el árbol sin cambiar sus timestamps. En
Linux, la lectura usa `O_NOATIME` y requiere que la cuenta pueda abrir el origen
con ese flag. Las bases se leen con `mode=ro` sobre una copia privada y
reconstruyen el estado comprometido del WAL sin tocar el SHM original.

Una ejecución exitosa promociona juntos el destino y `migration-report.json`.
`--report` permite escribir además una copia en una ruta explícita. Con
`--dry-run`, el informe se emite por stdout o en la ruta externa de `--report`.
Los checksums se enumeran por orden de ruta relativa, sin incluir nombres de
archivo. Los conteos de origen y destino se muestran por separado para que las
filas generadas, agrupadas o descartadas queden identificadas por sus totales.

El ledger Python se verifica y conserva completo en
`memory/legacy-deletion-ledger.json`, con permisos `0600`. Se usa el mismo
salt al construir la cadena Nexa. Los tipos que Nexa admite se convierten al
formato nativo y se agregan los fences de todas las conversaciones borradas y
personas retiradas identificables. Los eventos sin equivalente quedan en el
archivo privado. Una cadena que ya tiene formato Nexa conserva sus filas
exactas. El informe contiene conteos y checksums de archivos.

Las sesiones Python se descartan. Nexa aplica sus permisos de rol a los hashes
de credenciales importados. El valor configurado de
`memory.voice_proposals_enabled` se aplica al abrir el daemon y debe
corresponder al estado global autorizado para el cutover.

Nexa conserva los embeddings compatibles presentes en las bases migradas.
Los hechos legados cuyo vector sólo está en Chroma quedan en `facts` y aún
requieren indexación con BGE-M3 para aparecer en la búsqueda vectorial. Los
archivos de audio no se copian y sus rutas pendientes de borrado sí se
preservan. Los recordatorios recurrentes conservan su próxima ocurrencia. Los
horarios de rutinas legadas y sus condiciones carecen de un campo en el
esquema Nexa. Las rutinas condicionadas se importan deshabilitadas para
evitar acciones sin su condición. Los runs Python que estaban ejecutándose
quedan interrumpidos y no repiten efectos.

Las pruebas usan un `nexa-memoryd` existente en
`$CARGO_TARGET_DIR/debug/nexa-memoryd`. Si `NEXA_SQLITE_VEC` no está definido,
la suite compila únicamente la extensión en un directorio temporal a partir
del fuente 0.1.7 de la caché de Cargo.
