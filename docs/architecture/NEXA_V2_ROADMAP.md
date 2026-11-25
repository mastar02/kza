# Roadmap ejecutable de Nexa v2

Estado al 2026-11-25. Base de integración: `v2/main@dab0b945`. Esta es la referencia del repositorio para las olas de Nexa. Los planes P6/P7 del monolito corresponden a otra línea de desarrollo.

## Olas y evidencia de cierre

| Ola | Estado | Resultado o criterio para cerrarla |
| --- | --- | --- |
| 0 · Base | Completada | Workspace Rust, contratos gRPC, configuración tipada, cuenta dedicada, primeras fixtures y CI integrados en `7d9a8800`. |
| 1 · Componentes | Completada | DSP y captura, worker ML y motores de NLU, HA, hogar, LLM y memoria implementados. Revisar T2.1–T4.5 en el historial y los contratos en [crates](../../crates/). |
| 2 · Integración | Completada | Adaptadores de hardware T2.2, pipeline de voz T4.6, plano de control Axum T5.1 y SPA T5.2 integrados. Evidencia en `2d47215b`, `e99446b0`, `1d004471` y `390d798c`. |
| 3 · Entrega y paridad | Completada | L4 Playwright, migración offline T6.1 y paquete/rollback T6.2 con smoke L5. La CI enlaza las suites en `f9f4e3e5`. |
| 4 · Validación y cutover | En curso | Validar el candidato Linux en hardware, ensayar migración y rollback, reconciliar perfil y units, obtener GO y documentar estabilidad en campo. |
| 5 · Retiro y opcionales | Pendiente | Revisar T7.0a, ejecutar T7.0b en asiento separado tras aprobación y retirar el runtime antiguo únicamente bajo el gate de Fase B. Priorizar los opcionales por separado. |

Los bloques A y B de paridad y los fixes Linux se reportan completados en la línea de trabajo. En esta base se pueden inspeccionar el protocolo (`046c69c0`), el endurecimiento de NLU/LLM/ops (`732f2551`, `50d4b334`, `8a97c0b7`), memoria y hogar (`98dc3693`, `cf33fb38`), web, migrador y brain (`0355975a`, `5ede0220`, `dab0b945`). Antes de usar un paquete, verificar que incluya los fixes Linux del perfil operativo; hay diferencias concretas en las units de esta base. [Contrato y diferencias de despliegue](DEPLOYMENT.md).

### Secuencia de la Ola 4

1. **Ops** identifica SHA y manifiesto limpios. Contrasta daemon de audio compilado con hardware, ORT, sello escribible, PYTHONPATH, caché offline y autenticación Bearer del LLM con las units efectivas. Registra PCI, VRAM y puertos según [hardware](HARDWARE.md).
2. **QA y privacidad** revisan L3, L4 y L5 en el candidato, con QA independiente por familia real. Verifican cuatro entradas físicas, permisos UDS, denegación de red en ML/memoria, revocaciones, RBAC y ausencia del modo de pruebas en release. Registrar MA1260 en simulación como tal.
3. **Ops** ensaya migración offline en copia consistente y rollback cronometrado (objetivo < 60 s). Entrega hashes, conteos y resultado sin datos privados. Si falla una condición, conservar el candidato anterior y corregir en la rama de trabajo.
4. **Coordinación** presenta evidencia y solicita GO de cutover. Tras la promoción autorizada, registrar observación estable, incidencias y disponibilidad de rollback. La retirada del host requiere una decisión posterior.

## T7.0a · Manifiesto de purga propuesto

T7.0a documenta el alcance. La aprobación del manifiesto habilita preparar T7.0b en su propio asiento. El tag base inmutable `archive/legacy-monolith-2026-11` apunta a `d0defada018b90a45bb95714dde6dbd02b48eca9`, el mismo commit que `recovery/production-2026-10-25` al levantar el inventario. No mover ni reemplazar el tag.

### Fase A · Solo Git, destino de integración v2/main

| Lote | Candidatos del monolito | Qué preservar o comprobar |
| --- | --- | --- |
| Código y tests | `src/` y `tests/` | Primero trasladar `tests/unit/workflow/` a `tests-v2/workflow/`, ajustar sus referencias relativas, fixtures, CI y documentación y ejecutar la suite en su nueva ruta. Preservar pruebas v2 existentes. |
| Scripts | Inventario orientativo de ~47 scripts antiguos en `scripts/`: bytecode, adopción/compactación, launcher, sync Chroma, entrenamiento, setup y operación del runtime anterior | Conservar `kza_quota.py`, `kza_router.py`, `kza_seat.py`, `kza_safety_guard.py`, `kza_precompact.py` y sus consumidores. Excluir dependencias de servicios vigentes, incluidos instalador de hook y unit del índice de código. |
| Tools | Inventario orientativo de ~42 herramientas legacy en `tools/`: diagnóstico/bench del monolito, P6–P8, SC0/SC4, fixtures web y entrenamiento | Conservar `tools/nexa-migrate/` completo y herramientas con consumidores activos como `tools/code_search.py`. Resolver dependencias por ruta antes de aprobar el diff. |
| Infraestructura histórica | `docker/`, `containers/`, `systemd/`, `okf/`, `config/settings.yaml`, `.github/workflows/ci.yml` | Conservar `deploy/nexa/` con sus units y udev, `config/nexa/` y `.github/workflows/v2-ci.yml`. Comprobar dependientes de ficheros auxiliares de raíz antes de incluirlos. |
| Documentación | Resúmenes de Emotion, alertas y tests y plan de mejoras de raíz; material obsoleto en `docs/plans/`, `docs/research/`, `docs/superpowers/` y arquitectura histórica | Conservar los nueve documentos de T7.0a, los runbooks vigentes del [índice](../INDEX.md), decisiones e inventarios aún citados (incluido el manifiesto de rescate). Revisar los enlaces desde documentación viva y hooks. |

**Preparación del asiento T7.0b.** Fijar el SHA de partida, enumerar rutas con `git ls-files` y comparar el contenido de cada candidato contra el tag con `git diff --name-status archive/legacy-monolith-2026-11 HEAD -- RUTA`. Las cantidades anteriores orientan la revisión. El tag conserva el monolito de rescate, pero hay diferencias posteriores en dashboard, privacidad y pruebas de workflow. Registrar el SHA recuperable de cada delta antes de retirarlo. Conservar toda ruta dudosa hasta identificar su consumidor y dueño.

**Cierre de Fase A.** Revisar el listado de bajas explícitas y de movimientos con el padre. Tras aprobación, ejecutar la suite workflow reubicada y la CI v2, validar enlaces, imports, hooks, skills y construcción del paquete desde checkout limpio. Entregar diff, SHA y evidencia a QA independiente antes de integrar en `v2/main`. Esta fase solo cambia archivos versionados.

### Fase B · Servidor 192.168.1.2

Requiere cutover estable documentado **y GO explícito de retiro en el servidor**. El dueño de operación identifica cada ruta, unit, proceso, volumen y dependencia del runtime sustituido, verifica el respaldo privado y el rollback y presenta la lista exacta al aprobador. Conservar estado, ledgers y revocaciones durante toda la transición; aplicar las políticas de retención y borrado también a las copias.

Excluir expresamente `kza-code-index`, `kza-chroma`, `postpilot-embeddings-*`, `infra`, `ha-core`, cualquier servicio de otro dueño y todos los recursos nuevos de `nexa`. No inferir propiedad por prefijo, puerto, GPU o cuenta. Una vez autorizado el inventario, retirar solo los elementos aprobados, verificar salud de Nexa y servicios compartidos y guardar evidencia sin PII. Si la atribución es ambigua o el rollback deja de estar disponible, suspender el retiro.
