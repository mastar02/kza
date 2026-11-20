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
| `T090-U1` | Runner portátil `scripts/run_test_suite.py` con `pytest-xdist` (`--dist=worksteal`), Tiers `0/1/2`, `/dev/shm` automático en Linux con limpieza `0555`, pre-build de caché y carril `systemd` paralelo desde `t=0` en Tier 2 | En curso |
| `T090-U2` | Caché unificado content-addressed `tests/mocks/runtime_artifact_cache.py` (`<commit>-<python_magic>`) con `fcntl.flock` y hardlinks atómicos cableado en los fixtures/helpers que invocan `build_runtime` como insumo | En curso |
| `T090-U3` | Optimización de fixtures lentos sin tocar oráculos (`test_private_access.py`, árboles base inmutables en `test_legacy_state_retirement.py` / `test_adoption_commands.py` / `compaction_world.py`) | Planificada |
| `T090-U4` | Aceleración del carril `systemd` (`sc6b` + `test_linux_*`) y `compiled-solo` preservando `systemd-journald NRestarts` y presupuestos reales | Planificada |
| `T090-U5` | Integración formal de Tiers en runbooks de desarrollo, workflow multi-asiento y `.github/workflows/ci.yml` | Planificada |

## Niveles de ejecución (Tiers)

- **Tier 0 (`python scripts/run_test_suite.py --tier 0 --targets ...`):** Feedback inmediato (objetivo `15–45 s`): tests JS de frontend (`node --test tests/frontend/*.cjs`, opcional con `--no-node`) + pytest paralelo sobre módulos dirigidos.
- **Tier 1 (`python scripts/run_test_suite.py --tier 1`):** Validación rápida completa sin `systemd` (objetivo `~3–5 min` en Mac M4 Pro con 8 workers o en `kza-yo` con 12–16 workers): preconstruye artefactos en caché, corre `node --test` y toda la suite excepto los 5 módulos `test_linux_*` y los 7 tests `compiled-solo`.
- **Tier 2 (`python scripts/run_test_suite.py --tier 2` en `kza-yo`):** Gate completo paralelo (objetivo `~8–15 min`): ejecuta en paralelo desde `t=0` `node --test`, el Carril A (suite no-systemd con `xdist`) y el Carril B (`KZA_REQUIRE_REAL_SYSTEMD=1` secuencial para los 5 módulos `test_linux_*` / `sc6b1` / `sc6b2`), y al terminar tanto A como B lanza el Carril C (`compiled-solo`, con los 7 nodeids exactos sin contención CPU), verificando `after_NRestarts == before_NRestarts` con dos lecturas válidas de `systemd-journald` y exclusión por `SOLO_WINDOW`.

## Reglas de integridad inviolables

1. Prohibido debilitar aserciones, agregar `gc.collect()` artificial, saltar tests reales o inflar timeouts en tests.
2. `scripts/build_runtime.py` y sus pruebas dedicadas (`tests/unit/tools/test_build_runtime*.py`) permanecen intactos sin usar caché.
3. Respeto estricto de `SOLO_WINDOW` en `kza-yo` y aislamiento absoluto respecto de `/home/kza/*` y ThermoUY (`tu-*` / `gsm-*`).
