# P6 · Soak local de T061

T061 mide 72 horas continuas de disponibilidad del candidato sellado y ejecuta una campaña local de fallos fuera de ese bloque. Se prueba el grafo de `src.main.build_application` del artefacto CPython 3.13 con Home Assistant REST/WS y cuatro micrófonos sintéticos. Para programar el bloque físico T074 y adjuntar ambas evidencias, seguir el [guion de release P6](p6-release.md).

## Preparar el candidato bajo `yo` en `kza-yo`

Usar el venv CPython 3.13 del checkout de desarrollo y el `runtime.tar.gz` que se va a evaluar. Generar y verificar `release-review.json` con `tools.release_manifest_review` y el lock de runtime según el guion de release. Reservar el bloque continuo en coordinación con ThermoUY. El preflight consulta los timers y servicios de Tier 2 nocturno y prune con `systemctl --user show`. Un servicio en `active`, `activating` o `reloading` marca interferencia. Revisar el estado y la próxima activación UTC de cada timer antes de iniciar.

```bash
PY="$PWD/.venv/bin/python"
REVIEW="/ruta/privada/al/expediente"
COMMIT="<SHA completo del candidato>"
STATE="$HOME/.local/state/kza-soak/$COMMIT"

"$PY" -B -m tools.p6_local_soak preflight \
  --archive "$REVIEW/runtime.tar.gz" --review "$REVIEW/release-review.json"
```

El preflight compara SHA-256, inventario y ABI con el manifiesto revisado. Extrae el runtime con directorios `0555` y archivos `0444`, crea un área de trabajo privada y registra el plan de fallos y los hashes. Rechaza UID 1000, root, la cuenta `kza` y rutas de producción bajo `/home/kza` y `/opt/homeassistant`, incluidos enlaces que resuelven allí. Los puertos de HA y dashboard se asignan al hacer bind en `127.0.0.1:0`. El diario registra el puerto efectivo del dashboard de cada fase. El worker usa el entorno sintético del arnés y vuelve a verificar el runtime antes de importar código.

## Ejecutar la unidad de usuario

La unidad de usuario de `yo` configura afinidad a CPU 24–47, nice 5, `MemoryHigh=24G`, `NoNewPrivileges=yes`, `UMask=0077` y `CUDA_VISIBLE_DEVICES=`. Cada fase ejecutada por la unidad comprueba que la afinidad efectiva esté dentro de ese conjunto de CPU, que nice sea al menos 5 y que `NoNewPrivs` y `memory.high` estén aplicados en el proceso y su cgroup v2.

El worker verifica el selector CUDA vacío y el hook de auditoría de Python antes de importar el candidato. El hook limita las conexiones auditadas a `127.0.0.1` y `::1` y rechaza operaciones de archivos auditadas sobre rutas de producción, incluidas las que resuelven por enlace. El arnés usa modelos sintéticos en CPU y registra las interfaces de red y la visibilidad de `/dev/dri` como observaciones del host. Esta política opera en espacio de usuario y su cobertura corresponde a las operaciones auditadas por Python. En Ubuntu 24.04 funciona con la configuración de AppArmor existente.

Ejecutar primero una unidad transitoria de humo con las mismas propiedades. Debe terminar con código 0 y mostrar el límite `memory_high_bytes=25769803776` en el resultado del chequeo.

```bash
systemd-run --user --wait --pipe --collect --working-directory="$PWD" \
  --property=CPUAffinity=24-47 --property=Nice=5 --property=MemoryHigh=24G \
  --property=NoNewPrivileges=yes --property=UMask=0077 \
  --property=Environment=CUDA_VISIBLE_DEVICES= \
  "$PY" -B -c 'import os, sys; from tools.p6_local_soak import linux_isolation, offline_io; assert os.umask(0o077) == 0o077; sys.addaudithook(offline_io); print(linux_isolation())'
```

Con el humo aprobado, generar e iniciar `kza-soak-ID.service`.

```bash
UNIT="kza-soak-${COMMIT:0:12}.service"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
"$PY" -B -m tools.p6_local_soak render-systemd-unit --state "$STATE" > "$UNIT_DIR/$UNIT"
systemd-analyze --user verify "$UNIT_DIR/$UNIT"
systemctl --user daemon-reload
systemctl --user start --no-block "$UNIT"
systemctl --user status "$UNIT"
journalctl --user -u "$UNIT"
```

La unidad ejecuta `run-soak` seguido de `inject-faults`. Es posible invertir el orden al generar la unidad con `--faults-first`. Las dos fases registran sus comprobaciones efectivas en `faults.jsonl` y en `isolation_at_start` del resumen. `ExecStopPost` llama a `summarize` al completar o interrumpir la unidad. El resultado del comando acredita la generación del resumen. El resultado de T061 está en los booleanos del JSON. Cada fase puede abrirse una sola vez por expediente. Para repetir, conservar el expediente anterior y preparar otro con `preflight --state-root`.

## Interpretar y archivar la evidencia

- `samples.jsonl` conserva el estado del dashboard, REST y WS, la edad del último PCM no nulo y el indicador de heartbeat para cada habitación. La ventana UTC es fija y cada segundo admite una muestra. Un segundo sin muestra válida cuenta como hueco y permanece en el denominador. Se identifica la pausa autenticada de captura cuando el control verifica mute persistido y sellado. Esos segundos cuentan como no útiles para las habitaciones. La disponibilidad de KZA se evalúa por separado.
- KZA requiere HTTP 200 autenticado, REST y WS vigentes y subsistemas operativos sanos. El dashboard conserva su estado agregado y el detalle de cada subsistema. El audio se contabiliza por habitación y el monitor de latencia puede figurar como degradado por falta de comandos en la carga sintética. Un heartbeat de habitación es útil si existe evidencia de PCM no nulo con una antigüedad de hasta un segundo.
- `faults.jsonl` conserva el plan, los límites UTC, los fallos inyectados y sus observaciones. HA devuelve 503 en REST y cierra/rechaza WS. Durante diez lecturas REST con demora de 200 ms el arnés mide el heartbeat de 5 ms en el mismo loop de KZA, con p95 ≤25 ms y máximo ≤75 ms. La recuperación requiere nueva evidencia REST y WS en ≤60 s desde que el simulador vuelve a estar sano.
- Para el micrófono, el watchdog observa la pérdida de PCM en cocina y retira su stream. El dispositivo reaparece con catálogo tardío. La recuperación exige PCM no nulo en el camino de captura dentro de 180 s desde que el dispositivo queda enumerado. Se registran aperturas y edades máximas de los otros tres heartbeats.
- El ensayo de mute usa una sesión de operador del dashboard, verifica persistencia al reconstruir el grafo y prueba el rechazo por la entrada de voz y sin credenciales. Tras reactivar por sesión de operador, comprueba que la autoridad sigue denegando el origen de un evento borrado y que la copia anterior de SQLite termina con cero filas de ese evento.
- `soak-summary.json` presenta el bloque fijo, la campaña de fallos y el total UTC continuo entre ambos. Las pausas, fallos planificados, fallos espontáneos y huecos quedan contabilizados. La disponibilidad mínima es 99,9 % para KZA y para cada una de `living`, `hall`, `escritorio` y `cocina`. Archivar el resumen, ambos JSONL y el manifiesto de preflight junto con la revisión del candidato.

```bash
"$PY" -B -m tools.p6_local_soak summarize --state "$STATE"
jq '{c19_72h_passed, c20_faults_passed, c03_h2_passed, c04_mute_passed, overall_t061_local_passed}' \
  "$STATE/soak-summary.json"
```

El veredicto `overall_t061_local_passed` exige las cuatro comprobaciones locales y una ventana terminada de al menos 259 200 segundos. Los resultados de H2 y mute corresponden a los escenarios de T061. El expediente de release conserva además las pruebas del contrato completo de C03/C04, restauración de T063 y el bloque físico T074.

## Verificar el arnés en una corrida corta

Para un ensayo de desarrollo, preparar otro directorio privado con el mismo preflight y usar su ruta de estado. Las fases directas verifican el hook y el selector CUDA en el worker. La prueba de tres segundos ejercita bytecode y transportes locales. C19 y el veredicto global quedan en falso.

```bash
"$PY" -B -m tools.p6_local_soak run-soak --state "$STATE" --duration-s 3
"$PY" -B -m tools.p6_local_soak inject-faults --state "$STATE"
"$PY" -B -m tools.p6_local_soak summarize --state "$STATE"
```
