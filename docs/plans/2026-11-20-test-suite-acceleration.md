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

## Arquitectura de 5 Unidades (`T090-U1`..`T090-U5`)

| Unidad | Descripción | Estado |
|---|---|---|
| `T090-U1` | Runner portátil `scripts/run_test_suite.py` con `pytest-xdist` (`--dist=worksteal`), Tiers `0/1/2`, `/dev/shm` acotado a Tier 0, `basetemp` único autolimpiado en disco (`out_dir/basetemp-*`) y caché persistente en disco (`~/.cache/kza-runtime-artifacts`) en Tier 1/2 con límite predeterminado de `8` workers y carril `systemd` paralelo desde `t=0` en Tier 2 | `verified_local` |
| `T090-U2` | Caché unificado content-addressed `tests/mocks/runtime_artifact_cache.py` (`<commit>-<python_magic>`) con `fcntl.flock` y hardlinks atómicos cableado en los fixtures/helpers que invocan `build_runtime` como insumo | `verified_local` |
| `T090-U3` | Optimización de fixtures lentos sin tocar oráculos (`test_private_access.py`, árboles base inmutables en `test_legacy_state_retirement.py` / `test_adoption_commands.py` / `compaction_world.py`) | Planificada |
| `T090-U4` | Aceleración del carril `systemd` (`sc6b` + `test_linux_*`) y `compiled-solo` preservando `systemd-journald NRestarts` y presupuestos reales | Planificada |
| `T090-U5` | Integración de Tiers en el runbook de Mac, el workflow multi-asiento y `.github/workflows/ci.yml`, con tests de estructura del workflow | `verified_local` |

## Niveles de ejecución (Tiers)

Ejecutar desde el checkout del SHA bajo prueba con el venv KZA de Python 3.13.

- **Tier 0**, feedback de asiento con objetivo de 15–45 s. Selecciona los
  tests afectados con `--targets` y usa `/dev/shm` en Linux si hay espacio
  suficiente. Omitir `--no-node` si el cambio afecta al frontend.
- **Tier 1**, validación sin systemd en Mac M4 Pro o en `kza-yo` con objetivo
  de ~3–5 min. Usa hasta 8 workers por defecto, preconstruye artefactos en
  `~/.cache/kza-runtime-artifacts` y limpia el `out_dir/basetemp-*` único al
  salir. Ejecuta Node y la suite Python salvo los cinco módulos de systemd
  real y los ocho `COMPILED_SOLO_TESTS`.

```bash
.venv/bin/python scripts/run_test_suite.py --tier 0 --no-node --targets tests/unit/tools/test_run_test_suite.py
.venv/bin/python scripts/run_test_suite.py --tier 1
```

**Tier 2**, gate completo (objetivo ~8–15 min), como `yo` en
`/home/yo/projects/kza` de `kza-yo`. Ejecutar con seis workers y los límites
indicados dentro de `MemoryHigh=32 GiB` y `MemorySwapMax=0` de `yo`.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 \
  taskset -c 24-47 nice -n 10 .venv/bin/python scripts/run_test_suite.py --tier 2 --workers 6
```

Node, el carril A (suite sin systemd con xdist) y el carril B (cinco módulos
seriales con `KZA_REQUIRE_REAL_SYSTEMD=1`, incluido SC-6b en dos fases) se
ejecutan en paralelo. Una vez terminados, el carril C ejecuta los ocho
nodeids `compiled-solo` en serie. El runner toma ambas lecturas de
`systemd-journald` bajo la misma exclusión por `SOLO_WINDOW` y `flock`.
El gate requiere `exit_code == 0`, `nrestarts_ok == true` y dos lecturas
válidas con `before_NRestarts == after_NRestarts` en `summary.json`.

GitHub Actions ejecuta `unit-tests`, `safety-tests` e `integration-tests` con
`pytest -n auto --dist=worksteal`. El job de integración excluye los cinco
módulos de systemd real asignados al carril B y safety sigue siendo bloqueante.

## Reglas de integridad inviolables

1. Prohibido debilitar aserciones, agregar `gc.collect()` artificial, saltar tests reales o inflar timeouts en tests.
2. `scripts/build_runtime.py` y sus pruebas dedicadas (`tests/unit/tools/test_build_runtime*.py`) permanecen intactos sin usar caché.
3. Respeto estricto de `SOLO_WINDOW` en `kza-yo` y aislamiento absoluto respecto de `/home/kza/*` y ThermoUY (`tu-*` / `gsm-*`).
