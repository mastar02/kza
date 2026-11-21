# T090 — Aceleración Exponencial de la Suite de Tests KZA

- **Fecha:** 2026-11-20
- **Rama inicial:** `feat/t090-test-acceleration` (sobre `recovery/production-2026-10-25`)
- **Alcance:** Acelerar la ejecución diaria y de validación completa (Linux `kza-yo` y Mac M4 Pro) mediante paralelización real (`pytest-xdist`), caché content-addressed de artefactos compilados (`build_runtime`), tmpfs `/dev/shm` y separación de carriles sin debilitar un solo oráculo, timeout ni garantía de aislamiento.

## Baseline medido (`full-batch-q` en `kza-yo`)

Medición directa sobre los artefactos JUnit XML y metadatos de `/home/yo/projects/runs/full-batch-q/` en `kza-yo` (AMD Ryzen Threadripper PRO 7965WX 24-core / 48 threads, 125 GiB RAM, aislamiento en `taskset -c 24-47`):

- **Shards core XML (12 shards en 2 subshells serie):** `17,536` casos, `8,340.2 s` de tiempo acumulado de tests.
- **Fases secuenciales adicionales:** `compiled-solo` (`8` tests = `1,096.36 s`) + `sc6b1` + `sc6b2` (~`1,510 s` en runner v7).
- **Duración total:** ~`100–115 min` por corrida completa.

### Cuellos de botella identificados

1. **Concurrencia capada a 2 procesos en 24 hilos dedicados:** `run-server.sh` ejecutaba 2 colas en serie y sin `pytest-xdist`, dejando más de 90% de los cores libres mientras `test_legacy_state_retirement.py` (`761.9 s`), `test_adoption_commands.py` (`616.3 s`) y `tests/unit/tools/` (`3,397.1 s`) corrían en 1 hilo. Además, `compiled-solo` y `sc6b1/sc6b2` esperaban al final de toda la suite.
2. **Recompilación repetida de `build_runtime()` (~25–30 builds por corrida):** múltiples fixtures reconstruyen los mismos tarballs `.tar.gz` (`X5`, `X4`, commits históricos fijados `b0866e8`, `26de262`, `e9669ec`, `40aa05b`, `357dcb5`, `b15ce86`, `238b463`) desde cero en cada módulo.
3. **I/O en disco físico en vez de RAM (`/dev/shm`):** miles de extracciones de tarballs, `sqlite3` e inventarios `0555` se escribían en disco ext4/LVM con `63 GiB` libres en `/dev/shm`.
4. **Sobrecarga evitable en fixtures (`T090-U3`):** recreación repetida de `TestClient` + `DashboardAPI` (~400 parametrizaciones en `tests/unit/dashboard/test_private_access.py`, `125.1 s`) y clonado/poblado repetido de mundos base.

## Arquitectura de 6 Unidades (`T090-U1`..`T090-U6`)

| Unidad | Descripción | Estado |
|---|---|---|
| `T090-U1` | Runner portátil `scripts/run_test_suite.py` con `pytest-xdist` (`--dist=worksteal`), Tiers `0/1/2`, `/dev/shm` acotado a Tier 0, `basetemp` único autolimpiado en disco (`out_dir/basetemp-*`) y caché persistente en disco (`~/.cache/kza-runtime-artifacts`) en Tier 1/2 con límite predeterminado de `8` workers y carril `systemd` paralelo desde `t=0` en Tier 2 | `verified_local` |
| `T090-U2` | Caché unificado content-addressed `tests/mocks/runtime_artifact_cache.py` (`<commit>-<python_magic>`) con `fcntl.flock` y hardlinks atómicos cableado en los fixtures/helpers que invocan `build_runtime` como insumo | `verified_local` |
| `T090-U3` | Optimización de fixtures lentos sin tocar oráculos (`test_private_access.py`, árboles base inmutables en `test_legacy_state_retirement.py` / `test_adoption_commands.py` / `compaction_world.py`) | En curso (asiento paralelo) |
| `T090-U4` | Aceleración de `systemd` (`sc6b` + `test_linux_*`) y `compiled-solo` preservando `systemd-journald NRestarts` y presupuestos reales: memoización en proceso de `schema_compatible_revision` y `_table_columns`, precalentamiento de `schema_compatible_revision(repo, "HEAD")` y `epoch4` en `~/.cache/kza-runtime-artifacts`, y `--solo-workers` (`--dist=load`, default `2`) en `lane-c-compiled-solo` tras cerrar Carriles A y B | `verified_local` |
| `T090-U5` | Integración de Tiers en el runbook de Mac, el workflow multi-asiento y `.github/workflows/ci.yml`, con tests de estructura del workflow | `verified_local` |
| `T090-U6` | Selección de pasos en Tier 2, trazabilidad por SHA y preflight de interferencia de ThermoUY | `verified_local` |

## Niveles de ejecución (Tiers)

Ejecutar desde el checkout del SHA bajo prueba con el venv KZA de Python 3.13.

- **Tier 0**, feedback de asiento con objetivo de 15–45 s. Selecciona los
  tests afectados con `--targets` y usa `/dev/shm` en Linux si hay espacio
  suficiente. Omitir `--no-node` si el cambio afecta al frontend.
- **Tier 1**, validación sin systemd en Mac M4 Pro o en `kza-yo` con objetivo
  de ~3–5 min. Usa hasta 8 workers por defecto, preconstruye artefactos en
  `~/.cache/kza-runtime-artifacts` (incluidos `X5/X4/Y5/Y4`, `PRE_LAUNCHER_COMMIT`,
  `HEAD`/`epoch4` y `schema_compatible_revision(repo, "HEAD")`) y limpia el
  `out_dir/basetemp-*` único al salir. Ejecuta Node y la suite Python salvo los
  cinco módulos de systemd real y los ocho `COMPILED_SOLO_TESTS`.

```bash
.venv/bin/python scripts/run_test_suite.py --tier 0 --no-node --targets tests/unit/tools/test_run_test_suite.py
.venv/bin/python scripts/run_test_suite.py --tier 1
```

**Tier 2**, gate completo (objetivo ~8–15 min), como `yo` en
`/home/yo/projects/kza` de `kza-yo`. Ejecutar con seis workers y los límites
indicados dentro de `MemoryHigh=32 GiB` y `MemorySwapMax=0` de `yo`. Mantener
el `NI=5` nativo de `yo` con afinidad a las CPU 24–47. Agregar `nice -n 10`
lleva a `NI=15` y puede provocar inanición de asyncio bajo carga.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 \
  taskset -c 24-47 .venv/bin/python scripts/run_test_suite.py --tier 2 --workers 6
```

Node, el carril A (suite sin systemd con xdist) y el carril B (cinco módulos
seriales con `use_xdist=False` y `KZA_REQUIRE_REAL_SYSTEMD=1`, incluido SC-6b en
dos fases para preservar `daemon-reload` y `before_NRestarts == after_NRestarts`
de `systemd-journald`) se ejecutan en paralelo. Una vez terminados ambos
carriles A y B, el carril C (`lane-c-compiled-solo`) ejecuta los ocho nodeids
`COMPILED_SOLO_TESTS` con `--solo-workers` (por defecto `2` cuando `pytest-xdist`
está disponible y `--workers >= 2`, si no `1`; distribución `--dist=load`) con
CPU y disco completamente libres para respetar el presupuesto real de `55 s` de
`advance-N` sin alterar timeouts de tests. Se puede forzar ejecución estrictamente
serial con `--solo-workers 1`. El runner toma ambas lecturas de
`systemd-journald` bajo la misma exclusión por `SOLO_WINDOW` y `flock`.
El gate requiere `exit_code == 0`, `nrestarts_ok == true` y dos lecturas
válidas con `before_NRestarts == after_NRestarts` en `summary.json`.

### Corridas parciales y preflight de Tier 2 (`T090-U6`)

Usar `--lanes` para repetir pasos de Tier 2 sobre el mismo SHA. Acepta
`all` (por defecto), `node`, `a`, `b`, `b-linux`, `b-sc6b1`, `b-sc6b2` y `c`,
separados por comas o espacios. Por ejemplo, agregar `--lanes b-sc6b2,c`
repite la segunda fase de SC-6b y luego los ensayos compilados. `b` selecciona
los tres pasos seriales del carril, en orden `b-linux`, `b-sc6b1`, `b-sc6b2`.
`--no-node` omite `frontend-node` aunque estuviera seleccionado.

Cada corrida parcial conserva `SOLO_WINDOW`, el bloqueo exclusivo de
`feature-pytest.lock` y la verificación de estabilidad de `NRestarts`. El
carril C espera a que terminen los pasos seleccionados de Node, A y B.
`summary.json` registra el `git_commit` del checkout (`null` si Git no
puede resolverlo) y `selected_lanes` con los nombres de los pasos en orden
canónico. Para validar resultados de corridas separadas, comprobar que todos
los pasos requeridos estén cubiertos por resúmenes exitosos del mismo SHA.

Al seleccionar cualquier paso de B o C en Linux, el runner consulta en modo
de solo lectura las unidades `.service` y `.timer` de
`thermouy-tier2-nightly` y `thermouy-ci-prune`. Si un servicio está activo,
activándose o desactivándose, o si un timer vence en los próximos
`--quiet-window-minutes` minutos (45 por defecto, comparados en UTC), termina
con código 2 antes de pytest, libera `SOLO_WINDOW` y registra los motivos
en `host_interference` de `summary.json` y en stderr. La consulta tolera
unidades ausentes o un `systemctl` no disponible. `--ignore-host-timers`
omite explícitamente este preflight.

### Detalle de optimizaciones `T090-U3` y `T090-U4`

- **`T090-U3` (fixtures lentos):** optimización de fixtures sin modificar
  oráculos ni relajar aserciones (`test_private_access.py`, reutilización de
  árboles base inmutables en `test_legacy_state_retirement.py`,
  `test_adoption_commands.py` y `compaction_world.py`).
- **`T090-U4` (carriles `systemd` y `compiled-solo`):**
  1. Memoización en proceso de `schema_compatible_revision(repo, new_revision)` y
     `_table_columns(repo, revision)` en `tests/mocks/slot_reconciliation.py` por
     commit SHA resuelto (con copia defensiva y `clear_revision_caches()`),
     evitando re-escanear `git rev-list` + `git show` + `ast.parse` en cada test
     compilado incluso entre clones temporales `epoch4_checkout`.
  2. Extensión de `prebuild_standard_artifacts(repo, cache_dir)` en
     `tests/mocks/runtime_artifact_cache.py` para precalentar en
     `~/.cache/kza-runtime-artifacts` `schema_compatible_revision(repo, "HEAD")` y
     los commits `epoch4` (`x4`, `y4` y `epoch4(HEAD)` cuando difiere del overlay)
     antes de lanzar los workers de Tier 1/2.
  3. Flag `--solo-workers` en `scripts/run_test_suite.py` (default `2` cuando
     `has_xdist` y `--workers >= 2`, si no `1`, con `--dist=load`) para
     `lane-c-compiled-solo`, lanzado exclusivamente al concluir Carril A y
     Carril B, manteniendo Carril B estrictamente serial (`use_xdist=False`).

GitHub Actions ejecuta `unit-tests`, `safety-tests` e `integration-tests` con
`pytest -n auto --dist=worksteal`. El job de integración excluye los cinco
módulos de systemd real asignados al carril B y safety sigue siendo bloqueante.

## Reglas de integridad inviolables

1. Prohibido debilitar aserciones, agregar `gc.collect()` artificial, saltar tests reales o inflar timeouts en tests.
2. `scripts/build_runtime.py` y sus pruebas dedicadas (`tests/unit/tools/test_build_runtime*.py`) permanecen intactos sin usar caché.
3. Respeto estricto de `SOLO_WINDOW` en `kza-yo` y aislamiento absoluto respecto de `/home/kza/*` y ThermoUY (`tu-*` / `gsm-*`).
