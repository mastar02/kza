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

Para un consentimiento activo de Python, la migración mantiene la comprobación
por persona y revisión contra los retiros registrados. Una fila nativa con
`granted=1` carece de revisión de consentimiento legada. Si el ledger legado
contiene un retiro, se importan revocados los consentimientos nativos activos
`(granted=0)` y se limpian sus referencias al perfil y al vínculo. Los hashes
de persona y revisión no permiten descartar que el retiro corresponda a la
fila, y la fecha de actualización de un respaldo no prueba una nueva
autorización. Las conversaciones borradas excluyen también recordatorios y
listas compartidas y sus registros dependientes.

Las sesiones Python se descartan. Nexa aplica sus permisos de rol a los hashes
de credenciales importados. El valor configurado de
`memory.voice_proposals_enabled` se aplica al abrir el daemon y debe
corresponder al estado global autorizado para el cutover.

Nexa conserva los embeddings compatibles presentes en las bases migradas.
Los hechos legados cuyo vector sólo está en Chroma quedan en `facts` y aún
requieren indexación con BGE-M3 para aparecer en la búsqueda vectorial. Los
archivos de audio no se copian y sus rutas pendientes de borrado sí se
preservan. Los recordatorios recurrentes conservan su próxima ocurrencia y
los recordatorios nativos conservan su revisión. La base de hogar se crea con
el esquema v3 de Nexa. Los horarios diarios y de una sola ejecución se
normalizan al formato de Nexa. El próximo vencimiento se calcula a partir de
la hora de migración para rutinas activas, excluyendo ocurrencias registradas.
Las rutinas con condiciones legadas u horarios no representables se importan
deshabilitadas. El informe cuenta los horarios legados y las
deshabilitaciones por condiciones u horarios no representables. Los runs de
Python que estaban ejecutándose quedan interrumpidos.

Las pruebas usan un `nexa-memoryd` existente en
`$CARGO_TARGET_DIR/debug/nexa-memoryd`. Si `NEXA_SQLITE_VEC` no está definido,
la suite compila únicamente la extensión en un directorio temporal a partir
del fuente 0.1.7 de la caché de Cargo.
